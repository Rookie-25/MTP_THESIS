r"""Phase 3 benchmark: fused Triton GBM vs pure PyTorch, time and peak VRAM.

What is being measured and why
==============================
The pure-PyTorch simulator materialises roughly five :math:`M \times N` tensors
in global memory per call (scale, add, scan, concatenate, exponentiate). The
fused kernel reads ``dW`` once and writes the path matrix once. The arithmetic
is identical; only the number of HBM round trips differs. So this benchmark
tracks two quantities:

**Wall time** via ``torch.cuda.Event``. CUDA launches are asynchronous, so
``time.perf_counter`` around a kernel launch measures queueing, not execution.
Events are recorded *on the stream* and therefore time the actual device work.
Every measurement is preceded by a warm-up iteration, because the first launch
pays Triton's JIT compilation and cuBLAS/allocator initialisation.

**Peak allocated memory** via ``torch.cuda.max_memory_allocated``, reset with
``reset_peak_memory_stats`` immediately before each measured region. Note this
reports memory *PyTorch* allocated, not total VRAM in use by the process --
which is the right number here, since the caching allocator is exactly what the
intermediate tensors go through.

An honest note on the memory claim
==================================
``dW`` itself is :math:`M \times N` and is supplied by the caller, so it is
counted in the baseline of *both* backends. The fused kernel therefore does not
reduce memory to :math:`O(M)`; it removes the *intermediates*, taking peak from
roughly :math:`6MN` down to about :math:`2MN` (input plus output). The remaining
:math:`2MN` floor is inherent to any design that hands the simulator a
pre-drawn Brownian sample.

Eliminating that floor requires folding a counter-based (Philox) RNG into the
kernel so increments are generated in registers and never touch HBM. That is
the natural follow-on optimisation and is flagged in
:mod:`src.csrc.triton_gbm`; it is not what this benchmark measures.

Reading the table
=================
``OOM`` in the PyTorch column is a *result*, not a failure of the benchmark: it
is the point at which the intermediate materialisation exceeds the device while
the fused path still completes. The script catches it and carries on.

Usage
=====
    python benchmarks/bench_phase3.py
    python benchmarks/bench_phase3.py --paths 10000 100000 1000000 --steps 252
    python benchmarks/bench_phase3.py --dtype float64 --repeats 10 --csv phase3.csv
    python benchmarks/bench_phase3.py --backward        # include the adjoint
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.csrc.triton_gbm import HAS_TRITON, is_available, triton_simulate_gbm  # noqa: E402
from src.models.gbm import simulate_gbm  # noqa: E402

S0 = 100.0
MU = 0.03
SIGMA = 0.20
MATURITY = 1.0

_BYTES_PER_MIB = 1024.0 * 1024.0


@dataclass
class Measurement:
    """Timing and memory for one backend at one problem size."""

    milliseconds: Optional[float]
    peak_mib: Optional[float]
    failed_oom: bool = False

    @property
    def ok(self) -> bool:
        return self.milliseconds is not None


@dataclass
class BenchmarkRow:
    """One problem size, both backends."""

    n_paths: int
    n_steps: int
    torch_result: Measurement
    triton_result: Measurement
    max_abs_diff: Optional[float]

    @property
    def speedup(self) -> Optional[float]:
        if not (self.torch_result.ok and self.triton_result.ok):
            return None
        if self.triton_result.milliseconds <= 0.0:
            return None
        return self.torch_result.milliseconds / self.triton_result.milliseconds

    @property
    def memory_ratio(self) -> Optional[float]:
        if not (self.torch_result.ok and self.triton_result.ok):
            return None
        if not self.triton_result.peak_mib:
            return None
        return self.torch_result.peak_mib / self.triton_result.peak_mib


def _is_oom(error: BaseException) -> bool:
    """Recognise an out-of-memory failure across PyTorch versions."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _free_cuda_memory() -> None:
    """Return cached blocks to the driver so the next size starts clean."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure(
    operation: Callable[[], torch.Tensor],
    *,
    repeats: int,
    keep_output: bool = False,
) -> tuple[Measurement, Optional[torch.Tensor]]:
    """Time ``operation`` on the CUDA stream and record its peak allocation.

    Args:
        operation: Zero-argument callable performing the work under test.
        repeats: Number of timed iterations; the minimum is reported, since the
            minimum is the cleanest estimate of achievable device time (larger
            samples only add scheduler noise).
        keep_output: Whether to return the final output tensor for a correctness
            comparison.

    Returns:
        ``(measurement, output_or_None)``. On out-of-memory the measurement is
        flagged and the output is ``None``.
    """
    try:
        # Warm-up: absorbs Triton JIT compilation and allocator growth so they
        # are not billed to the first timed iteration.
        warm = operation()
        torch.cuda.synchronize()
        del warm

        _free_cuda_memory()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        best_ms = float("inf")
        output: Optional[torch.Tensor] = None
        for index in range(repeats):
            start.record()
            result = operation()
            end.record()
            torch.cuda.synchronize()
            best_ms = min(best_ms, start.elapsed_time(end))
            if keep_output and index == repeats - 1:
                output = result.detach().clone()
            del result

        peak_mib = torch.cuda.max_memory_allocated() / _BYTES_PER_MIB
        return Measurement(milliseconds=best_ms, peak_mib=peak_mib), output

    except Exception as error:  # noqa: BLE001 - OOM is an expected outcome here
        if not _is_oom(error):
            raise
        _free_cuda_memory()
        return Measurement(milliseconds=None, peak_mib=None, failed_oom=True), None


def benchmark_one(
    n_paths: int,
    n_steps: int,
    *,
    dtype: torch.dtype,
    repeats: int,
    seed: int,
    include_backward: bool,
    verify: bool,
) -> BenchmarkRow:
    """Benchmark both backends at one problem size.

    Args:
        n_paths: Monte-Carlo paths :math:`M`.
        n_steps: Time steps :math:`N`.
        dtype: Working precision.
        repeats: Timed iterations per backend.
        seed: RNG seed for the Brownian sample.
        include_backward: Time forward + backward instead of forward alone.
        verify: Compare the two backends' outputs.

    Returns:
        A populated :class:`BenchmarkRow`.
    """
    dt = MATURITY / n_steps
    device = torch.device("cuda")

    _free_cuda_memory()

    # dW is an input to both backends, so it is deliberately allocated outside
    # the measured region and counted in neither peak.
    generator = torch.Generator(device=device).manual_seed(seed)
    dW = torch.randn(
        (n_paths, n_steps), device=device, dtype=dtype, generator=generator
    ) * (dt ** 0.5)

    def build(simulator: Callable[..., torch.Tensor]) -> Callable[[], torch.Tensor]:
        if not include_backward:
            def forward_only() -> torch.Tensor:
                with torch.no_grad():
                    return simulator(S0, MU, SIGMA, dW, dt)

            return forward_only

        def forward_and_backward() -> torch.Tensor:
            s0 = torch.tensor(S0, device=device, dtype=dtype, requires_grad=True)
            mu = torch.tensor(MU, device=device, dtype=dtype, requires_grad=True)
            sigma = torch.tensor(SIGMA, device=device, dtype=dtype, requires_grad=True)
            paths = simulator(s0, mu, sigma, dW, dt)
            paths.sum().backward()
            return paths

        return forward_and_backward

    torch_measurement, torch_output = measure(
        build(simulate_gbm), repeats=repeats, keep_output=verify
    )
    _free_cuda_memory()
    triton_measurement, triton_output = measure(
        build(triton_simulate_gbm), repeats=repeats, keep_output=verify
    )

    max_abs_diff: Optional[float] = None
    if torch_output is not None and triton_output is not None:
        max_abs_diff = float((torch_output - triton_output).abs().max())
    del torch_output, triton_output, dW
    _free_cuda_memory()

    return BenchmarkRow(
        n_paths=n_paths,
        n_steps=n_steps,
        torch_result=torch_measurement,
        triton_result=triton_measurement,
        max_abs_diff=max_abs_diff,
    )


# ==========================================================================
# Reporting
# ==========================================================================
def _format_measurement_time(measurement: Measurement) -> str:
    if measurement.failed_oom:
        return "OOM"
    if not measurement.ok:
        return "-"
    return f"{measurement.milliseconds:,.2f}"


def _format_measurement_memory(measurement: Measurement) -> str:
    if measurement.failed_oom:
        return "OOM"
    if measurement.peak_mib is None:
        return "-"
    return f"{measurement.peak_mib:,.1f}"


def render_table(rows: Sequence[BenchmarkRow]) -> str:
    """Render the sweep as an ASCII table.

    Args:
        rows: Completed benchmark rows in sweep order.

    Returns:
        The formatted table.
    """
    headers = (
        "M", "N", "torch (ms)", "triton (ms)", "speedup",
        "torch peak (MiB)", "triton peak (MiB)", "mem saved", "max |diff|",
    )

    body = []
    for row in rows:
        speedup = row.speedup
        ratio = row.memory_ratio
        body.append(
            (
                f"{row.n_paths:,}",
                f"{row.n_steps:,}",
                _format_measurement_time(row.torch_result),
                _format_measurement_time(row.triton_result),
                f"{speedup:,.2f}x" if speedup is not None else "-",
                _format_measurement_memory(row.torch_result),
                _format_measurement_memory(row.triton_result),
                f"{ratio:,.2f}x" if ratio is not None else "-",
                f"{row.max_abs_diff:.2e}" if row.max_abs_diff is not None else "-",
            )
        )

    widths = [
        max(len(headers[column]), *(len(line[column]) for line in body))
        for column in range(len(headers))
    ]
    rule = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(cells: Sequence[str]) -> str:
        return (
            "| "
            + " | ".join(cell.rjust(widths[index]) for index, cell in enumerate(cells))
            + " |"
        )

    lines = [rule, format_row(headers), rule]
    lines.extend(format_row(line) for line in body)
    lines.append(rule)
    return "\n".join(lines)


def render_analysis(rows: Sequence[BenchmarkRow], dtype: torch.dtype) -> str:
    """Summarise the sweep, including the theoretical memory expectation.

    Args:
        rows: Completed benchmark rows.
        dtype: Working precision, used for the analytic memory floor.

    Returns:
        A multi-line analysis block.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    completed = [row for row in rows if row.speedup is not None]

    lines = ["ANALYSIS", "-" * 78]

    if completed:
        speedups = [row.speedup for row in completed]
        ratios = [row.memory_ratio for row in completed if row.memory_ratio]
        lines.append(
            f"  Speedup            min {min(speedups):,.2f}x   "
            f"max {max(speedups):,.2f}x"
        )
        if ratios:
            lines.append(
                f"  Peak-memory saving min {min(ratios):,.2f}x   "
                f"max {max(ratios):,.2f}x"
            )
        worst = max(
            (row.max_abs_diff for row in rows if row.max_abs_diff is not None),
            default=None,
        )
        if worst is not None:
            lines.append(f"  Worst forward disagreement between backends: {worst:.2e}")
    else:
        lines.append("  No size completed on both backends.")

    oom_rows = [row for row in rows if row.torch_result.failed_oom]
    if oom_rows:
        lines.append("")
        lines.append("  PyTorch ran out of memory where the fused kernel survived:")
        for row in oom_rows:
            status = "completed" if row.triton_result.ok else "also OOM"
            lines.append(
                f"    M={row.n_paths:,} N={row.n_steps:,}  ->  triton {status}"
            )

    lines.extend(
        [
            "",
            "  Expected memory floor (both backends must pay this):",
            f"    dW     : M*N*{element_size} bytes   (caller-supplied, outside the measured peak)",
            f"    output : M*(N+1)*{element_size} bytes",
            "  PyTorch additionally materialises ~4 intermediate M*N tensors",
            "  (scale, add, scan, concat), which is what the fused kernel removes.",
            "",
            "  Note: peak is torch.cuda.max_memory_allocated, i.e. memory the",
            "  PyTorch caching allocator handed out - not total process VRAM.",
        ]
    )
    return "\n".join(lines)


def write_csv(rows: Sequence[BenchmarkRow], destination: Path) -> None:
    """Persist the sweep for plotting or inclusion in the write-up."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "n_paths", "n_steps", "torch_ms", "triton_ms", "speedup",
                "torch_peak_mib", "triton_peak_mib", "memory_ratio",
                "max_abs_diff", "torch_oom", "triton_oom",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.n_paths,
                    row.n_steps,
                    "" if row.torch_result.milliseconds is None else f"{row.torch_result.milliseconds:.6f}",
                    "" if row.triton_result.milliseconds is None else f"{row.triton_result.milliseconds:.6f}",
                    "" if row.speedup is None else f"{row.speedup:.6f}",
                    "" if row.torch_result.peak_mib is None else f"{row.torch_result.peak_mib:.3f}",
                    "" if row.triton_result.peak_mib is None else f"{row.triton_result.peak_mib:.3f}",
                    "" if row.memory_ratio is None else f"{row.memory_ratio:.6f}",
                    "" if row.max_abs_diff is None else f"{row.max_abs_diff:.6e}",
                    int(row.torch_result.failed_oom),
                    int(row.triton_result.failed_oom),
                ]
            )


# ==========================================================================
# Entry point
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI."""
    parser = argparse.ArgumentParser(
        prog="bench_phase3.py",
        description=(
            "Benchmark the fused Triton GBM kernel against pure PyTorch on "
            "execution time and peak VRAM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paths", type=int, nargs="+", default=[10_000, 100_000, 1_000_000, 5_000_000],
        help="Path counts M to sweep.",
    )
    parser.add_argument("--steps", type=int, default=252, help="Time steps N.")
    parser.add_argument(
        "--dtype", choices=["float32", "float64"], default="float32",
        help="Working precision. float32 is the realistic GPU choice; float64 "
        "halves throughput and doubles memory.",
    )
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="Timed iterations per measurement (minimum reported).",
    )
    parser.add_argument("--seed", type=int, default=20260818, help="RNG seed.")
    parser.add_argument(
        "--backward", action="store_true",
        help="Time forward + backward rather than forward only.",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip the output comparison (saves a full-size clone per size).",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    return parser


def main() -> int:
    """Run the sweep and print the report.

    Returns:
        ``0`` on success, ``1`` if the fused path is unavailable, ``2`` on bad
        arguments.
    """
    args = build_parser().parse_args()

    if any(count <= 0 for count in args.paths):
        print("--paths must all be positive", file=sys.stderr)
        return 2
    if args.steps <= 0 or args.repeats <= 0:
        print("--steps and --repeats must be positive", file=sys.stderr)
        return 2

    if not is_available():
        print(
            "\n  Cannot run: the fused Triton path is unavailable.\n"
            f"    triton installed : {HAS_TRITON}\n"
            f"    cuda available   : {torch.cuda.is_available()}\n\n"
            "  This benchmark is GPU-only. Run it on a CUDA machine with Triton\n"
            "  installed (e.g. Google Colab). The CPU-side correctness tests in\n"
            "  tests/test_phase3.py run anywhere and validate the adjoint maths.\n",
            file=sys.stderr,
        )
        return 1

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device_name = torch.cuda.get_device_name(0)
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)

    print()
    print("=" * 78)
    print("  FUSED TRITON GBM vs PURE PYTORCH  --  time and peak VRAM")
    print("=" * 78)
    print(f"  Device    : {device_name}  ({total_vram_gib:,.1f} GiB)")
    print(f"  torch     : {torch.__version__}   cuda {torch.version.cuda}")
    print(f"  dtype     : {args.dtype}   steps N = {args.steps:,}")
    print(
        f"  Timing    : torch.cuda.Event, {args.repeats} repeats "
        f"(minimum reported), warm-up excluded"
    )
    print(f"  Measuring : {'forward + backward' if args.backward else 'forward only'}")
    print()

    rows: List[BenchmarkRow] = []
    for n_paths in sorted(set(args.paths)):
        print(f"  running M = {n_paths:,} ...", end="", flush=True)
        row = benchmark_one(
            n_paths,
            args.steps,
            dtype=dtype,
            repeats=args.repeats,
            seed=args.seed,
            include_backward=args.backward,
            verify=not args.no_verify,
        )
        rows.append(row)
        speedup = row.speedup
        print(
            f" torch {_format_measurement_time(row.torch_result)} ms"
            f" | triton {_format_measurement_time(row.triton_result)} ms"
            f" | {f'{speedup:,.2f}x' if speedup is not None else 'n/a'}"
        )

    print()
    print(render_table(rows))
    print()
    print(render_analysis(rows, dtype))
    print()

    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"  CSV written to {args.csv}")
        print()

    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
