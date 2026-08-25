r"""Shared measurement primitives for every benchmark in this directory.

Why this module exists
======================
``Measurement``, ``measure``, ``_reset_cuda``, ``_is_oom`` and the peak-memory
predictors had been copy-pasted into four separate benchmark scripts. Four
copies means four chances for the timing methodology to drift apart, and a
thesis table assembled from scripts that measure *slightly differently* is not
comparable. Everything measurement-related now lives here.

Timing methodology, stated once
===============================
* **CUDA events, not wall clock.** Kernel launches are asynchronous, so
  ``time.perf_counter`` around a launch measures queueing, not execution.
  ``torch.cuda.Event`` records on the stream and times the actual device work.
* **Warm-up excluded.** The first call pays Triton JIT compilation and
  caching-allocator growth. Every measurement runs the operation once,
  discards it, resets peak statistics, and only then starts timing.
* **Minimum over repeats, not mean.** The minimum is the cleanest estimate of
  achievable device time; larger samples only accumulate scheduler noise.
* **Peak memory is the allocator's view.** ``max_memory_allocated`` reports
  what the PyTorch caching allocator handed out, which is the right scope --
  every tensor here goes through it. It is *not* total process VRAM.

Failure handling, and one non-negotiable rule
=============================================
Two distinct failure modes, which must not be conflated:

* **Predicted OOM** -- refused before launch by a cheap arithmetic check.
* **Actual OOM** -- the allocator raised; catchable and recoverable.

Anything *else* propagates. In particular an ``illegal memory access`` must
never be swallowed: it poisons the CUDA context for the whole process, so
every subsequent measurement in the sweep would report nonsense. Phase 4's
first Colab run died exactly that way. :func:`is_oom` deliberately matches only
genuine out-of-memory conditions.
"""

from __future__ import annotations

import gc
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch

__all__ = [
    "BYTES_PER_MIB",
    "BYTES_PER_GIB",
    "VRAM_SAFETY_FRACTION",
    "Measurement",
    "free_vram_bytes",
    "is_oom",
    "reset_cuda",
    "measure",
    "format_bytes",
    "markdown_table",
]

BYTES_PER_MIB = 1024.0**2
BYTES_PER_GIB = 1024.0**3

#: Fraction of *free* VRAM a run may require before it is refused unlaunched.
#: Below 1.0 deliberately: the caching allocator fragments, Triton and cuBLAS
#: keep workspaces, and the driver reserves a slice, so a run needing 99% of
#: nominally-free memory fails in practice.
VRAM_SAFETY_FRACTION = 0.90


@dataclass
class Measurement:
    """Timing and peak allocation for one backend at one problem size.

    Attributes:
        milliseconds: Best observed device time, or ``None`` if it never ran.
        peak_bytes: Observed peak allocation, or ``None``.
        failed_oom: Attempted, and the allocator refused it.
        predicted_oom: **Never attempted** -- a pre-flight estimate showed it
            could not fit. Distinct from ``failed_oom`` because the reason a
            row is empty is itself a result worth reporting.
        predicted_bytes: What a refused configuration would have needed.
        note: Free-form detail for the report (e.g. why it was refused).
    """

    milliseconds: Optional[float] = None
    peak_bytes: Optional[int] = None
    failed_oom: bool = False
    predicted_oom: bool = False
    predicted_bytes: Optional[int] = None
    note: str = ""

    @property
    def ok(self) -> bool:
        """Whether the measurement completed and carries a timing."""
        return self.milliseconds is not None

    @property
    def out_of_memory(self) -> bool:
        """Whether the configuration failed to run for either OOM reason."""
        return self.failed_oom or self.predicted_oom

    @property
    def peak_mib(self) -> Optional[float]:
        return None if self.peak_bytes is None else self.peak_bytes / BYTES_PER_MIB

    @property
    def peak_gib(self) -> Optional[float]:
        return None if self.peak_bytes is None else self.peak_bytes / BYTES_PER_GIB


def free_vram_bytes() -> int:
    """Return currently free device memory, from the driver's view.

    ``torch.cuda.mem_get_info`` sees memory held by other processes and by
    cached-but-unreleased blocks, which ``max_memory_allocated`` does not.

    Returns:
        Free bytes on the current device.
    """
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def is_oom(error: BaseException) -> bool:
    """Recognise a genuine out-of-memory failure, and nothing else.

    Deliberately narrow. An ``illegal memory access`` is *not* an OOM: it
    poisons the CUDA context, so treating it as a recoverable skip would make
    every later measurement in the sweep silently invalid.

    Args:
        error: The caught exception.

    Returns:
        ``True`` only for a real allocator OOM.
    """
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def reset_cuda() -> None:
    """Synchronise, release cached blocks, and clear peak statistics.

    Called between every backend and every problem size. At multi-GiB
    allocation sizes the caching allocator would otherwise hold the previous
    iteration's blocks, both fragmenting the heap and making the next peak
    reading meaningless. Synchronising first ensures no in-flight kernel still
    holds a reference when the cache is dropped.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()


def measure(
    operation: Callable[[], Optional[torch.Tensor]],
    *,
    repeats: int = 3,
    keep_output: bool = False,
) -> tuple[Measurement, Optional[torch.Tensor]]:
    """Time ``operation`` on the CUDA stream and record its peak allocation.

    Args:
        operation: Zero-argument callable performing the work under test. It
            must release its own large tensors; peak memory is the quantity of
            interest, so anything left alive pollutes the next reading.
        repeats: Timed iterations. The minimum is reported.
        keep_output: Retain the final returned tensor for a correctness
            cross-check. Only do this for small outputs (an EE profile), never
            a path matrix.

    Returns:
        ``(measurement, output_or_None)``. On out-of-memory the measurement is
        flagged and the output is ``None``.

    Raises:
        Exception: Anything that is not an OOM propagates unchanged.
    """
    try:
        warm = operation()  # absorbs Triton JIT and allocator growth
        torch.cuda.synchronize()
        del warm
        reset_cuda()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        best = math.inf
        output: Optional[torch.Tensor] = None
        for index in range(repeats):
            start.record()
            result = operation()
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end))
            if keep_output and index == repeats - 1 and result is not None:
                output = result.detach().clone()
            del result

        peak = torch.cuda.max_memory_allocated()
        return Measurement(milliseconds=best, peak_bytes=peak), output

    except Exception as error:  # noqa: BLE001 - OOM is an expected outcome
        if not is_oom(error):
            raise
        reset_cuda()
        return Measurement(failed_oom=True, note="allocator OOM"), None


def format_bytes(value: Optional[int], *, unit: str = "auto") -> str:
    """Render a byte count for a report table.

    Args:
        value: Bytes, or ``None``.
        unit: ``"MiB"``, ``"GiB"``, or ``"auto"`` to switch at 1 GiB.

    Returns:
        A formatted string, or ``"-"`` when ``value`` is ``None``.
    """
    if value is None:
        return "-"
    if unit == "auto":
        unit = "GiB" if value >= BYTES_PER_GIB else "MiB"
    if unit == "GiB":
        return f"{value / BYTES_PER_GIB:,.2f} GiB"
    return f"{value / BYTES_PER_MIB:,.1f} MiB"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Render a GitHub-flavoured Markdown table with aligned columns.

    Columns are padded to a consistent width so the raw Markdown stays readable
    in a diff or a plain-text thesis appendix, not only once rendered.

    Args:
        headers: Column headers.
        rows: Row cells, each row the same length as ``headers``.

    Returns:
        The table as a multi-line string.

    Raises:
        ValueError: If any row's length differs from ``headers``.
    """
    for index, row in enumerate(rows):
        if len(row) != len(headers):
            raise ValueError(
                f"row {index} has {len(row)} cells, expected {len(headers)}"
            )

    widths = [
        max(len(str(headers[column])), *(len(str(row[column])) for row in rows))
        if rows
        else len(str(headers[column]))
        for column in range(len(headers))
    ]

    def render(cells: Sequence[str]) -> str:
        return (
            "| "
            + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(cells))
            + " |"
        )

    lines = [render(headers)]
    lines.append("|" + "|".join("-" * (width + 2) for width in widths) + "|")
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)
