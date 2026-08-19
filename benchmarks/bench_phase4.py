r"""Phase 4 benchmark: pre-allocated ``dW`` vs in-kernel Philox RNG.

The claim under test
====================
Phase 3 fused the arithmetic but still required the caller to hand it an
:math:`M \times N` increment matrix. Phase 4 generates increments inside the
kernel, so that allocation disappears. The expected consequences:

* **Peak memory roughly halves** -- from ``dW + output`` down to ``output``.
* **Time improves modestly**, because :math:`M N` floats no longer make a round
  trip through HBM. The kernel does more integer work (Philox rounds) in
  exchange, so this is a bandwidth-for-arithmetic trade, not a free win. On a
  bandwidth-bound kernel it should still come out ahead.
* **The attainable path count roughly doubles** on a given device.

The last point is the honest version of "shattering the ceiling". Peak memory is
still :math:`O(MN)` -- the *output* path matrix remains -- so the ceiling moves,
it does not vanish:

============  ====================  ====================  ===================
paths (fp32)  Phase 3 (dW + out)    Phase 4 (out only)    fits in 16 GiB?
============  ====================  ====================  ===================
1M            1.88 GiB              0.94 GiB              both
5M            9.41 GiB              4.71 GiB              both
10M           18.81 GiB             9.42 GiB              Phase 4 only
20M           37.63 GiB             18.85 GiB             neither
============  ====================  ====================  ===================

So on a 16 GiB card the practical limit goes from ~7.9M paths to ~15.9M. Reaching
20M needs either an 80 GiB device or the Phase 5 step of fusing the payoff
reduction into the kernel so paths are consumed as produced and peak becomes
:math:`O(M)`.

``OOM`` in a column is therefore a **result**, not a benchmark failure: it marks
exactly where each design runs out of device. The script catches it and carries
on so the crossover is visible in the table.

Methodology notes
=================
* Timing uses ``torch.cuda.Event``, recorded on the stream. Wall-clock timers
  around an async launch would measure queueing, not execution.
* Every measurement is preceded by a warm-up, which absorbs Triton's JIT
  compilation and allocator growth.
* ``dW`` generation is timed **inside** the Phase 3 measurement. This is the
  fair comparison: Phase 4 must generate its increments too, it just does so in
  registers. Excluding the draw would flatter Phase 3 by pretending its input
  materialises for free.
* Peak memory is ``torch.cuda.max_memory_allocated`` -- what the PyTorch caching
  allocator handed out, which is the right scope since every tensor here goes
  through it.

Usage
=====
    python benchmarks/bench_phase4.py
    python benchmarks/bench_phase4.py --paths 1000000 5000000 10000000 20000000
    python benchmarks/bench_phase4.py --backward --csv phase4.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.csrc.triton_gbm import HAS_TRITON, is_available, triton_simulate_gbm  # noqa: E402
from src.csrc.triton_philox_gbm import philox_simulate_gbm  # noqa: E402

S0 = 100.0
MU = 0.03
SIGMA = 0.20
MATURITY = 1.0

_BYTES_PER_GIB = 1024.0**3
_BYTES_PER_MIB = 1024.0**2

#: Fraction of *free* VRAM a run is allowed to need before it is refused.
#: Deliberately below 1.0: the caching allocator fragments, cuBLAS/Triton keep
#: workspaces, and the driver reserves a slice, so a run needing 99% of free
#: memory will fail in practice.
VRAM_SAFETY_FRACTION = 0.90


def free_vram_bytes() -> int:
    """Return currently free device memory in bytes.

    Uses ``torch.cuda.mem_get_info``, which reports the driver's view rather
    than PyTorch's, so memory held by other processes or by cached-but-unfreed
    blocks is accounted for.

    Returns:
        Free bytes on the current device.
    """
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def predict_peak_bytes(
    n_paths: int,
    n_steps: int,
    element_size: int,
    *,
    include_dw: bool,
    include_backward: bool,
) -> int:
    """Predict the peak allocation a configuration will require.

    Counts only the :math:`O(MN)` tensors, which dominate everything else by
    orders of magnitude at these scales:

    * output paths -- ``M * (N + 1)``, both backends
    * ``dW`` -- ``M * N``, Phase 3 only
    * incoming adjoint -- ``M * (N + 1)``, backward only
    * ``grad_dW`` -- ``M * N``, Phase 3 backward only

    Args:
        n_paths: Monte-Carlo paths :math:`M`.
        n_steps: Time steps :math:`N`.
        element_size: Bytes per element.
        include_dw: Whether the backend materialises ``dW`` (Phase 3 does).
        include_backward: Whether the adjoint is included in the measurement.

    Returns:
        Predicted peak bytes.
    """
    output_bytes = n_paths * (n_steps + 1) * element_size
    increment_bytes = n_paths * n_steps * element_size

    total = output_bytes
    if include_dw:
        total += increment_bytes
    if include_backward:
        total += output_bytes  # the incoming adjoint is output-shaped
        if include_dw:
            total += increment_bytes  # grad_dW
    return total


@dataclass
class Measurement:
    """Timing and peak allocation for one backend at one problem size.

    Attributes:
        milliseconds: Best observed device time, or ``None`` if it never ran.
        peak_bytes: Observed peak allocation, or ``None``.
        failed_oom: The run was attempted and the allocator refused it.
        predicted_oom: The run was **never attempted** because a pre-flight
            estimate showed it could not fit. This distinction matters: an
            attempted run that overflows a 32-bit offset raises
            ``illegal memory access``, which poisons the CUDA context for the
            rest of the process and cannot be caught. Refusing up front keeps
            the remaining sweep alive.
        predicted_bytes: What the refused configuration would have needed.
    """

    milliseconds: Optional[float] = None
    peak_bytes: Optional[int] = None
    failed_oom: bool = False
    predicted_oom: bool = False
    predicted_bytes: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.milliseconds is not None

    @property
    def skipped(self) -> bool:
        """Whether this configuration never ran, for either OOM reason."""
        return self.failed_oom or self.predicted_oom

    @property
    def peak_gib(self) -> Optional[float]:
        if self.peak_bytes is None:
            return None
        return self.peak_bytes / _BYTES_PER_GIB

    @property
    def predicted_gib(self) -> Optional[float]:
        if self.predicted_bytes is None:
            return None
        return self.predicted_bytes / _BYTES_PER_GIB


@dataclass
class BenchmarkRow:
    """One problem size, both backends."""

    n_paths: int
    n_steps: int
    phase3: Measurement
    phase4: Measurement

    @property
    def speedup(self) -> Optional[float]:
        if not (self.phase3.ok and self.phase4.ok):
            return None
        if not self.phase4.milliseconds:
            return None
        return self.phase3.milliseconds / self.phase4.milliseconds

    @property
    def memory_ratio(self) -> Optional[float]:
        if not (self.phase3.ok and self.phase4.ok):
            return None
        if not self.phase4.peak_bytes:
            return None
        return self.phase3.peak_bytes / self.phase4.peak_bytes

    @property
    def throughput_phase4(self) -> Optional[float]:
        """Millions of path-steps per second for the Philox kernel."""
        if not self.phase4.ok:
            return None
        return (self.n_paths * self.n_steps) / (self.phase4.milliseconds * 1e3)


def _is_oom(error: BaseException) -> bool:
    """Recognise an out-of-memory failure across PyTorch versions."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def _reset_cuda() -> None:
    """Release cached blocks and clear peak stats so the next size starts clean.

    Called between every backend and every problem size. At these allocation
    sizes the caching allocator will otherwise hold multi-GiB blocks from the
    previous iteration, which both fragments the heap and makes the next
    measurement's peak meaningless. ``synchronize`` first, so no in-flight
    kernel is still holding a reference when the cache is dropped.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()


def measure(operation: Callable[[], None], *, repeats: int) -> Measurement:
    """Time ``operation`` on the CUDA stream and record its peak allocation.

    Args:
        operation: Zero-argument callable performing the work under test. It
            must release its own tensors, since peak memory is the quantity of
            interest.
        repeats: Timed iterations. The minimum is reported: it is the cleanest
            estimate of achievable device time, and larger samples only add
            scheduler noise.

    Returns:
        A :class:`Measurement`, flagged ``failed_oom`` if the device ran out.
    """
    try:
        operation()  # warm-up: absorbs Triton JIT and allocator growth
        torch.cuda.synchronize()
        _reset_cuda()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        best = math.inf
        for _ in range(repeats):
            start.record()
            operation()
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end))

        return Measurement(
            milliseconds=best, peak_bytes=torch.cuda.max_memory_allocated()
        )

    except Exception as error:  # noqa: BLE001 - OOM is an expected outcome
        if not _is_oom(error):
            raise
        _reset_cuda()
        return Measurement(failed_oom=True)


def benchmark_one(
    n_paths: int,
    n_steps: int,
    *,
    dtype: torch.dtype,
    repeats: int,
    seed: int,
    include_backward: bool,
) -> BenchmarkRow:
    """Benchmark both backends at one problem size.

    Args:
        n_paths: Monte-Carlo paths :math:`M`.
        n_steps: Time steps :math:`N`.
        dtype: Working precision.
        repeats: Timed iterations per backend.
        seed: RNG seed / Philox key.
        include_backward: Time forward + backward rather than forward alone.

    Returns:
        A populated :class:`BenchmarkRow`.
    """
    dt = MATURITY / n_steps
    device = torch.device("cuda")

    def phase3_operation() -> None:
        """Phase 3: draw dW into HBM, then run the fused kernel."""
        # The draw is deliberately inside the measured region -- Phase 4 pays
        # for its increments too, so excluding this would flatter Phase 3.
        generator = torch.Generator(device=device).manual_seed(seed)
        dW = torch.randn(
            (n_paths, n_steps), device=device, dtype=dtype, generator=generator
        ) * math.sqrt(dt)
        if include_backward:
            s0 = torch.tensor(S0, device=device, dtype=dtype, requires_grad=True)
            mu = torch.tensor(MU, device=device, dtype=dtype, requires_grad=True)
            sigma = torch.tensor(SIGMA, device=device, dtype=dtype, requires_grad=True)
            triton_simulate_gbm(s0, mu, sigma, dW, dt).sum().backward()
        else:
            with torch.no_grad():
                triton_simulate_gbm(S0, MU, SIGMA, dW, dt)
        del dW

    def phase4_operation() -> None:
        """Phase 4: increments generated in-kernel, nothing extra allocated."""
        if include_backward:
            s0 = torch.tensor(S0, device=device, dtype=dtype, requires_grad=True)
            mu = torch.tensor(MU, device=device, dtype=dtype, requires_grad=True)
            sigma = torch.tensor(SIGMA, device=device, dtype=dtype, requires_grad=True)
            philox_simulate_gbm(
                s0, mu, sigma, n_paths, n_steps, dt, seed=seed, dtype=dtype
            ).sum().backward()
        else:
            with torch.no_grad():
                philox_simulate_gbm(
                    S0, MU, SIGMA, n_paths, n_steps, dt, seed=seed, dtype=dtype
                )

    # ---- pre-flight VRAM check -------------------------------------
    # Refuse configurations that provably cannot fit rather than letting CUDA
    # fail asynchronously. This is not just tidier: a run that exceeds device
    # memory *or* overflows a 32-bit pointer offset can raise
    # "illegal memory access", which poisons the CUDA context and aborts the
    # whole sweep. A cheap arithmetic check up front keeps later sizes running.
    element_size = torch.tensor([], dtype=dtype).element_size()

    _reset_cuda()
    budget = int(VRAM_SAFETY_FRACTION * free_vram_bytes())

    phase3_need = predict_peak_bytes(
        n_paths, n_steps, element_size,
        include_dw=True, include_backward=include_backward,
    )
    phase4_need = predict_peak_bytes(
        n_paths, n_steps, element_size,
        include_dw=False, include_backward=include_backward,
    )

    if phase3_need > budget:
        phase3 = Measurement(predicted_oom=True, predicted_bytes=phase3_need)
    else:
        phase3 = measure(phase3_operation, repeats=repeats)

    _reset_cuda()
    # Re-read the budget: Phase 3 may have left the allocator in a different
    # state, and Phase 4 deserves an honest, current figure.
    budget = int(VRAM_SAFETY_FRACTION * free_vram_bytes())

    if phase4_need > budget:
        phase4 = Measurement(predicted_oom=True, predicted_bytes=phase4_need)
    else:
        phase4 = measure(phase4_operation, repeats=repeats)

    _reset_cuda()

    return BenchmarkRow(n_paths=n_paths, n_steps=n_steps, phase3=phase3, phase4=phase4)


# ==========================================================================
# Reporting
# ==========================================================================
def _time_cell(measurement: Measurement) -> str:
    if measurement.predicted_oom:
        return "OOM (pred)"
    if measurement.failed_oom:
        return "OOM"
    if not measurement.ok:
        return "-"
    return f"{measurement.milliseconds:,.1f}"


def _memory_cell(measurement: Measurement) -> str:
    if measurement.predicted_oom:
        # Report what it *would* have needed: that is the informative number.
        return f"~{measurement.predicted_gib:,.2f}" if measurement.predicted_gib else "OOM (pred)"
    if measurement.failed_oom:
        return "OOM"
    if measurement.peak_gib is None:
        return "-"
    return f"{measurement.peak_gib:,.2f}"


def render_table(rows: Sequence[BenchmarkRow]) -> str:
    """Render the sweep as an ASCII table."""
    headers = (
        "M", "N",
        "P3 dW (ms)", "P4 philox (ms)", "speedup",
        "P3 peak (GiB)", "P4 peak (GiB)", "mem saved",
        "P4 Mpath-steps/s",
    )

    body = []
    for row in rows:
        speedup = row.speedup
        ratio = row.memory_ratio
        throughput = row.throughput_phase4
        body.append(
            (
                f"{row.n_paths:,}",
                f"{row.n_steps:,}",
                _time_cell(row.phase3),
                _time_cell(row.phase4),
                f"{speedup:,.2f}x" if speedup is not None else "-",
                _memory_cell(row.phase3),
                _memory_cell(row.phase4),
                f"{ratio:,.2f}x" if ratio is not None else "-",
                f"{throughput:,.0f}" if throughput is not None else "-",
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


def render_analysis(
    rows: Sequence[BenchmarkRow], dtype: torch.dtype, total_vram_bytes: int
) -> str:
    """Summarise the sweep and state the remaining ceiling honestly."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    completed = [row for row in rows if row.speedup is not None]

    lines = ["ANALYSIS", "-" * 86]

    if completed:
        speedups = [row.speedup for row in completed]
        ratios = [row.memory_ratio for row in completed if row.memory_ratio]
        lines.append(
            f"  Speedup (P4 over P3)     min {min(speedups):,.2f}x   max {max(speedups):,.2f}x"
        )
        if ratios:
            lines.append(
                f"  Peak-memory saving       min {min(ratios):,.2f}x   max {max(ratios):,.2f}x"
                "   (theory: ~2.00x, dW eliminated)"
            )
        best = max(completed, key=lambda row: row.throughput_phase4 or 0.0)
        lines.append(
            f"  Best P4 throughput       {best.throughput_phase4:,.0f} Mpath-steps/s "
            f"at M={best.n_paths:,}"
        )
    else:
        lines.append("  No size completed on both backends.")

    # Where each design runs out of device.
    p3_ceiling = total_vram_bytes / (element_size * (2 * 252 + 1))
    p4_ceiling = total_vram_bytes / (element_size * 253)
    survived = [row for row in rows if row.phase3.skipped and row.phase4.ok]

    lines.append("")
    if survived:
        lines.append("  CEILING SHATTERED -- Phase 4 completed where Phase 3 ran out:")
        for row in survived:
            reason = "OOM (predicted)" if row.phase3.predicted_oom else "OOM"
            lines.append(
                f"    M={row.n_paths:,}  P3 {reason}  ->  P4 "
                f"{row.phase4.milliseconds:,.1f} ms at "
                f"{row.phase4.peak_gib:,.2f} GiB peak"
            )
    else:
        lines.append(
            "  No size in this sweep separated the two designs by OOM. Push --paths "
            "higher\n  to find the crossover on this device."
        )

    both_oom = [row for row in rows if row.phase3.skipped and row.phase4.skipped]
    if both_oom:
        lines.append("")
        lines.append("  Beyond BOTH designs (the output tensor itself no longer fits):")
        for row in both_oom:
            need = row.n_paths * (row.n_steps + 1) * element_size / _BYTES_PER_GIB
            lines.append(
                f"    M={row.n_paths:,}  output alone needs {need:,.2f} GiB"
            )

    lines.extend(
        [
            "",
            "  WHAT PHASE 4 DOES AND DOES NOT FIX",
            f"    Phase 3 peak ~ (dW + output) = M*(2N+1)*{element_size} bytes",
            f"    Phase 4 peak ~ (output)      = M*(N+1)*{element_size} bytes",
            "    So the attainable path count roughly DOUBLES, but peak is still",
            "    O(M*N): the output path matrix remains. Estimated ceilings on this",
            f"    device ({total_vram_bytes / _BYTES_PER_GIB:,.1f} GiB, N=252, "
            f"{str(dtype).replace('torch.', '')}):",
            f"      Phase 3: ~{p3_ceiling / 1e6:,.1f}M paths",
            f"      Phase 4: ~{p4_ceiling / 1e6:,.1f}M paths",
            "    Making memory O(M) requires fusing the payoff/exposure reduction",
            "    into the kernel so paths are consumed as produced -- Phase 5.",
            "",
            "  Note: dW generation is timed inside the Phase 3 measurement, since",
            "  Phase 4 must produce its increments too. Excluding it would flatter",
            "  Phase 3 by treating its input as free.",
            "",
            "  'OOM (pred)' means the run was refused by the pre-flight VRAM check",
            f"  (needs more than {VRAM_SAFETY_FRACTION:.0%} of free memory) and never",
            "  launched. Attempting it risks an uncatchable illegal-memory-access",
            "  abort that would poison the CUDA context and kill the whole sweep.",
        ]
    )
    return "\n".join(lines)


def write_csv(rows: Sequence[BenchmarkRow], destination: Path) -> None:
    """Persist the sweep for plotting or the write-up."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "n_paths", "n_steps",
                "phase3_ms", "phase4_ms", "speedup",
                "phase3_peak_bytes", "phase4_peak_bytes", "memory_ratio",
                "phase4_mpath_steps_per_s", "phase3_oom", "phase4_oom",
                "phase3_predicted_oom", "phase4_predicted_oom",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.n_paths,
                    row.n_steps,
                    "" if row.phase3.milliseconds is None else f"{row.phase3.milliseconds:.6f}",
                    "" if row.phase4.milliseconds is None else f"{row.phase4.milliseconds:.6f}",
                    "" if row.speedup is None else f"{row.speedup:.6f}",
                    "" if row.phase3.peak_bytes is None else row.phase3.peak_bytes,
                    "" if row.phase4.peak_bytes is None else row.phase4.peak_bytes,
                    "" if row.memory_ratio is None else f"{row.memory_ratio:.6f}",
                    "" if row.throughput_phase4 is None else f"{row.throughput_phase4:.3f}",
                    int(row.phase3.failed_oom),
                    int(row.phase4.failed_oom),
                    int(row.phase3.predicted_oom),
                    int(row.phase4.predicted_oom),
                ]
            )


# ==========================================================================
# Entry point
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI."""
    parser = argparse.ArgumentParser(
        prog="bench_phase4.py",
        description=(
            "Benchmark the in-kernel Philox GBM kernel (Phase 4) against the "
            "pre-allocated-dW kernel (Phase 3) on time and peak VRAM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paths", type=int, nargs="+",
        default=[1_000_000, 5_000_000, 10_000_000, 20_000_000],
        help="Path counts M to sweep.",
    )
    parser.add_argument("--steps", type=int, default=252, help="Time steps N.")
    parser.add_argument(
        "--dtype", choices=["float32", "float64"], default="float32",
        help="Working precision. float32 is the realistic choice at these scales; "
        "float64 doubles memory and halves the attainable path count.",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Timed iterations per measurement (minimum reported).",
    )
    parser.add_argument("--seed", type=int, default=20260819, help="RNG seed / Philox key.")
    parser.add_argument(
        "--backward", action="store_true",
        help="Time forward + backward. This is where Phase 4 gains most, since "
        "rematerialisation avoids both a stored Z and a grad_dW buffer.",
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
            "  installed (e.g. Google Colab). The CPU-side tests in\n"
            "  tests/test_phase4.py run anywhere and validate the adjoint maths\n"
            "  and the Philox offset-aliasing guard.\n",
            file=sys.stderr,
        )
        return 1

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    properties = torch.cuda.get_device_properties(0)
    total_vram = properties.total_memory

    print()
    print("=" * 86)
    print("  IN-KERNEL PHILOX (Phase 4) vs PRE-ALLOCATED dW (Phase 3)")
    print("=" * 86)
    print(f"  Device    : {properties.name}  ({total_vram / _BYTES_PER_GIB:,.1f} GiB)")
    print(f"  torch     : {torch.__version__}   cuda {torch.version.cuda}")
    print(f"  dtype     : {args.dtype}   steps N = {args.steps:,}")
    print(
        f"  Timing    : torch.cuda.Event, {args.repeats} repeats "
        f"(minimum), warm-up excluded"
    )
    print(f"  Measuring : {'forward + backward' if args.backward else 'forward only'}")
    print(
        f"  Guard     : pre-flight VRAM check at {VRAM_SAFETY_FRACTION:.0%} of free "
        f"({free_vram_bytes() / _BYTES_PER_GIB:,.1f} GiB free now)"
    )
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
        )
        rows.append(row)
        speedup = row.speedup
        print(
            f" P3 {_time_cell(row.phase3)} ms | P4 {_time_cell(row.phase4)} ms"
            f" | {f'{speedup:,.2f}x' if speedup is not None else 'n/a'}"
            f" | peak {_memory_cell(row.phase3)} -> {_memory_cell(row.phase4)} GiB"
        )

    print()
    print(render_table(rows))
    print()
    print(render_analysis(rows, dtype, total_vram))
    print()

    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"  CSV written to {args.csv}")
        print()

    print("=" * 86)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
