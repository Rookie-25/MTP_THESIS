r"""Phase 5 benchmark: materialised paths (Phase 4) vs fused reduction (Phase 5).

The claim under test
====================
Phase 4 removed the ``dW`` matrix but still wrote an :math:`M \times (N+1)` path
matrix, so peak memory stayed :math:`O(MN)`. Phase 5 consumes each path inside
the kernel and writes only a bounded partial-sum buffer, so peak becomes

.. math:: O(\texttt{max\_programs} \times N) \quad\text{-- no } M .

At :math:`N = 252`, fp32, that is about 4 MiB regardless of path count, against:

===========  ======================  ==================
paths        Phase 4 path matrix     Phase 5 partials
===========  ======================  ==================
1M           0.94 GiB                ~4 MiB
5M           4.71 GiB                ~4 MiB
10M          9.42 GiB                ~4 MiB
50M          47.13 GiB               ~4 MiB
===========  ======================  ==================

So 50M paths is unreachable for Phase 4 on any current single GPU, and routine
for Phase 5 on a 16 GiB card. That is the headline result.

Comparing like with like
========================
The two backends do **not** produce the same object, so the benchmark is careful
about what it times:

* Phase 4 produces a path matrix. To make it a fair comparison it is followed by
  the same MtM -> exposure -> mean reduction Phase 5 performs internally, so
  both columns measure *"parameters in, EE profile out"*. Timing only Phase 4's
  path generation would flatter it by omitting work Phase 5 does.
* Phase 4 and Phase 5 draw different sample paths (different Philox addressing),
  so their EE profiles agree only within Monte-Carlo error. This script reports
  the worst relative deviation as a sanity signal, not as a correctness test --
  ``tests/test_phase5.py`` does correctness properly, against sampling-error
  tolerances.

Pre-flight refusal, not crash-and-recover
=========================================
Every configuration is costed before launch and refused if it needs more than
90% of free VRAM (``OOM (pred)``). This is deliberate rather than defensive: an
allocation that overruns the device can surface as an asynchronous
``illegal memory access``, which **poisons the CUDA context for the whole
process** and would abort the rest of the sweep. Phase 4's first Colab run died
exactly that way. Costing the run up front keeps the interesting rows alive.

Usage
=====
    python benchmarks/bench_phase5.py
    python benchmarks/bench_phase5.py --paths 1000000 5000000 10000000 50000000
    python benchmarks/bench_phase5.py --backward --csv phase5.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks._harness import (  # noqa: E402
    BYTES_PER_GIB as _BYTES_PER_GIB,
    BYTES_PER_MIB as _BYTES_PER_MIB,
    Measurement,
    VRAM_SAFETY_FRACTION,
    free_vram_bytes,
    is_oom as _is_oom,
    measure,
    reset_cuda as _reset_cuda,
)

from src.csrc.triton_cva_fusion import (  # noqa: E402
    DEFAULT_MAX_PROGRAMS,
    build_affine_coefficients,
    fused_expected_exposure,
    is_available,
    select_fused_block_sizes,
)
from src.csrc.triton_gbm import HAS_TRITON  # noqa: E402
from src.csrc.triton_philox_gbm import philox_simulate_gbm  # noqa: E402
from src.pricer.options import SwapLeg  # noqa: E402

S0 = 100.0
MU = 0.03
RATE = 0.03
SIGMA = 0.20
MATURITY = 1.0

#: Refuse a run needing more than this fraction of *free* VRAM.
VRAM_SAFETY_FRACTION = 0.90


def portfolio() -> List[SwapLeg]:
    """The netting set benchmarked throughout: mixed signs, staggered maturity."""
    return [
        SwapLeg(notional=1.0, strike=100.0, maturity=MATURITY),
        SwapLeg(notional=-0.4, strike=110.0, maturity=MATURITY),
        SwapLeg(notional=0.7, strike=95.0, maturity=0.5),
    ]


@dataclass
class BenchmarkRow:
    """One problem size, both backends."""

    n_paths: int
    n_steps: int
    phase4: Measurement
    phase5: Measurement
    worst_relative_deviation: Optional[float] = None

    @property
    def speedup(self) -> Optional[float]:
        if not (self.phase4.ok and self.phase5.ok) or not self.phase5.milliseconds:
            return None
        return self.phase4.milliseconds / self.phase5.milliseconds

    @property
    def memory_ratio(self) -> Optional[float]:
        if not (self.phase4.ok and self.phase5.ok) or not self.phase5.peak_bytes:
            return None
        return self.phase4.peak_bytes / self.phase5.peak_bytes

    @property
    def throughput_phase5(self) -> Optional[float]:
        """Millions of path-steps per second."""
        if not self.phase5.ok:
            return None
        return (self.n_paths * self.n_steps) / (self.phase5.milliseconds * 1e3)


def predict_phase4_bytes(
    n_paths: int, n_steps: int, element_size: int, *, include_backward: bool
) -> int:
    """Predict Phase 4's peak: the path matrix, plus the adjoint if requested."""
    path_matrix = n_paths * (n_steps + 1) * element_size
    total = path_matrix
    # The exposure surface and its clamped copy are also O(M*N); PyTorch will
    # materialise at least one of them during the reduction.
    total += path_matrix
    if include_backward:
        total += path_matrix  # incoming adjoint
    return total


def predict_phase5_bytes(
    n_paths: int, n_steps: int, element_size: int, *, max_programs: int
) -> int:
    """Predict Phase 5's peak. Note the absence of ``n_paths`` in the formula."""
    block_m, _block_t = select_fused_block_sizes(n_steps, element_size)
    n_blocks = (n_paths + block_m - 1) // block_m
    n_programs = min(n_blocks, max_programs)
    # Partial buffer, plus a handful of length-(N+1) vectors (B, C, weight, EE).
    return n_programs * (n_steps + 1) * element_size + 8 * (n_steps + 1) * element_size


def benchmark_one(
    n_paths: int,
    n_steps: int,
    *,
    dtype: torch.dtype,
    repeats: int,
    seed: int,
    include_backward: bool,
    max_programs: int,
) -> BenchmarkRow:
    """Benchmark both backends at one problem size.

    Args:
        n_paths: Monte-Carlo paths :math:`M`.
        n_steps: Time steps :math:`N`.
        dtype: Working precision.
        repeats: Timed iterations per backend.
        seed: Philox key.
        include_backward: Time forward + backward rather than forward alone.
        max_programs: Phase 5 launch-grid cap.

    Returns:
        A populated :class:`BenchmarkRow`.
    """
    device = torch.device("cuda")
    dt = MATURITY / n_steps
    element_size = torch.tensor([], dtype=dtype).element_size()
    legs = portfolio()

    times = torch.linspace(0.0, MATURITY, n_steps + 1, device=device, dtype=torch.float64)
    coeff_b, coeff_c = build_affine_coefficients(legs, times, RATE)

    def phase4_operation() -> torch.Tensor:
        """Materialise paths, then reduce to EE -- the same output as Phase 5."""
        if include_backward:
            s0 = torch.tensor(S0, device=device, dtype=dtype, requires_grad=True)
            sigma = torch.tensor(SIGMA, device=device, dtype=dtype, requires_grad=True)
            paths = philox_simulate_gbm(
                s0, MU, sigma, n_paths, n_steps, dt, seed=seed, dtype=dtype
            )
            mtm = coeff_b.reshape(1, -1) * paths - coeff_c.reshape(1, -1)
            profile = torch.clamp(mtm, min=0.0).mean(dim=0)
            profile.sum().backward()
            return profile.detach()
        with torch.no_grad():
            paths = philox_simulate_gbm(
                S0, MU, SIGMA, n_paths, n_steps, dt, seed=seed, dtype=dtype
            )
            mtm = coeff_b.reshape(1, -1) * paths - coeff_c.reshape(1, -1)
            return torch.clamp(mtm, min=0.0).mean(dim=0)

    def phase5_operation() -> torch.Tensor:
        """Fused: no path matrix ever exists."""
        if include_backward:
            s0 = torch.tensor(S0, device=device, dtype=dtype, requires_grad=True)
            sigma = torch.tensor(SIGMA, device=device, dtype=dtype, requires_grad=True)
            profile = fused_expected_exposure(
                s0, MU, sigma, legs, times, RATE, n_paths,
                seed=seed, max_programs=max_programs,
            )
            profile.sum().backward()
            return profile.detach()
        with torch.no_grad():
            return fused_expected_exposure(
                S0, MU, SIGMA, legs, times, RATE, n_paths,
                seed=seed, max_programs=max_programs,
            )

    # ---- pre-flight costing ------------------------------------------
    _reset_cuda()
    budget = int(VRAM_SAFETY_FRACTION * free_vram_bytes())
    phase4_need = predict_phase4_bytes(
        n_paths, n_steps, element_size, include_backward=include_backward
    )
    phase5_need = predict_phase5_bytes(
        n_paths, n_steps, element_size, max_programs=max_programs
    )

    if phase4_need > budget:
        phase4 = Measurement(predicted_oom=True, predicted_bytes=phase4_need)
        phase4_profile = None
    else:
        phase4, phase4_profile = measure(
            phase4_operation, repeats=repeats, keep_output=True
        )

    _reset_cuda()
    budget = int(VRAM_SAFETY_FRACTION * free_vram_bytes())

    if phase5_need > budget:
        phase5 = Measurement(predicted_oom=True, predicted_bytes=phase5_need)
        phase5_profile = None
    else:
        phase5, phase5_profile = measure(
            phase5_operation, repeats=repeats, keep_output=True
        )

    worst_deviation: Optional[float] = None
    if phase4_profile is not None and phase5_profile is not None:
        scale = phase4_profile.abs().max().clamp(min=1e-12)
        worst_deviation = float(
            ((phase5_profile - phase4_profile).abs() / scale).max()
        )

    del phase4_profile, phase5_profile
    _reset_cuda()

    return BenchmarkRow(
        n_paths=n_paths,
        n_steps=n_steps,
        phase4=phase4,
        phase5=phase5,
        worst_relative_deviation=worst_deviation,
    )


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
        gib = measurement.predicted_gib
        return f"~{gib:,.2f}" if gib is not None else "OOM (pred)"
    if measurement.failed_oom:
        return "OOM"
    gib = measurement.peak_gib
    return "-" if gib is None else f"{gib:,.4f}"


def render_table(rows: Sequence[BenchmarkRow]) -> str:
    """Render the sweep as an ASCII table."""
    headers = (
        "M", "N",
        "P4 paths (ms)", "P5 fused (ms)", "speedup",
        "P4 peak (GiB)", "P5 peak (GiB)", "mem saved",
        "P5 Mpath-steps/s", "rel dev",
    )

    body = []
    for row in rows:
        speedup, ratio = row.speedup, row.memory_ratio
        throughput = row.throughput_phase5
        body.append(
            (
                f"{row.n_paths:,}",
                f"{row.n_steps:,}",
                _time_cell(row.phase4),
                _time_cell(row.phase5),
                f"{speedup:,.2f}x" if speedup is not None else "-",
                _memory_cell(row.phase4),
                _memory_cell(row.phase5),
                f"{ratio:,.0f}x" if ratio is not None else "-",
                f"{throughput:,.0f}" if throughput is not None else "-",
                f"{row.worst_relative_deviation:.1e}"
                if row.worst_relative_deviation is not None else "-",
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
    """Summarise the sweep and state what the memory result does and does not mean."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    completed = [row for row in rows if row.speedup is not None]
    phase5_ran = [row for row in rows if row.phase5.ok]

    lines = ["ANALYSIS", "-" * 100]

    if completed:
        speedups = [row.speedup for row in completed]
        ratios = [row.memory_ratio for row in completed if row.memory_ratio]
        lines.append(
            f"  Speedup (P5 over P4)      min {min(speedups):,.2f}x   "
            f"max {max(speedups):,.2f}x"
        )
        if ratios:
            lines.append(
                f"  Peak-memory reduction     min {min(ratios):,.0f}x   "
                f"max {max(ratios):,.0f}x"
            )
    else:
        lines.append("  No size completed on both backends.")

    if phase5_ran:
        peaks = [row.phase5.peak_bytes for row in phase5_ran]
        largest_m = max(row.n_paths for row in phase5_ran)
        smallest_m = min(row.n_paths for row in phase5_ran)
        lines.append("")
        lines.append(
            f"  THE O(N) CLAIM: Phase 5 peak across M={smallest_m:,} to "
            f"M={largest_m:,}"
        )
        lines.append(
            f"    min {min(peaks) / _BYTES_PER_MIB:,.2f} MiB   "
            f"max {max(peaks) / _BYTES_PER_MIB:,.2f} MiB   "
            f"ratio {max(peaks) / max(min(peaks), 1):,.2f}x"
        )
        lines.append(
            f"    (path count grew {largest_m / max(smallest_m, 1):,.0f}x; "
            "peak should be flat)"
        )
        best = max(phase5_ran, key=lambda row: row.throughput_phase5 or 0.0)
        lines.append(
            f"  Best P5 throughput        {best.throughput_phase5:,.0f} "
            f"Mpath-steps/s at M={best.n_paths:,}"
        )

    survived = [row for row in rows if row.phase4.skipped and row.phase5.ok]
    if survived:
        lines.append("")
        lines.append("  CEILING BROKEN -- Phase 5 completed where Phase 4 could not:")
        for row in survived:
            reason = "OOM (predicted)" if row.phase4.predicted_oom else "OOM"
            lines.append(
                f"    M={row.n_paths:,}  P4 {reason} (needed "
                f"~{row.phase4.predicted_gib or float('nan'):,.1f} GiB)  ->  P5 "
                f"{row.phase5.milliseconds:,.1f} ms at "
                f"{row.phase5.peak_bytes / _BYTES_PER_MIB:,.2f} MiB"
            )

    both_failed = [row for row in rows if row.phase4.skipped and row.phase5.skipped]
    if both_failed:
        lines.append("")
        lines.append("  Beyond BOTH designs:")
        for row in both_failed:
            lines.append(f"    M={row.n_paths:,}")

    block_m, block_t = select_fused_block_sizes(
        rows[0].n_steps if rows else 252, element_size
    )
    lines.extend(
        [
            "",
            "  MEMORY MODEL",
            f"    Phase 4 peak ~ 2-3 x M*(N+1)*{element_size}   (path matrix + exposure surface)",
            f"    Phase 5 peak ~ n_programs*(N+1)*{element_size} + O(N)   -- no M term at all",
            f"    Launch config: BLOCK_M={block_m}, BLOCK_T={block_t}, "
            f"grid capped at {DEFAULT_MAX_PROGRAMS}",
            "",
            "  WHAT THIS DOES NOT CLAIM",
            "    * Peak is bounded by a constant, not bitwise identical across M:",
            "      the grid is min(ceil(M/BLOCK_M), max_programs), so small M",
            "      launches fewer programs and uses slightly LESS.",
            "    * 'rel dev' compares two DIFFERENT random samples (P4 and P5 use",
            "      different Philox addressing), so a nonzero value is Monte-Carlo",
            "      error, not a bug. Correctness lives in tests/test_phase5.py,",
            "      which compares against sampling-error tolerances.",
            "    * Phase 4 here is timed WITH the MtM+exposure+mean reduction, so",
            "      both columns measure parameters-in / EE-profile-out. Timing only",
            "      its path generation would flatter it.",
            "",
            "    'OOM (pred)' means refused by the pre-flight VRAM check, never",
            "    launched. Attempting it risks an uncatchable illegal-memory-access",
            "    abort that would poison the CUDA context and kill the sweep.",
            f"    Device total: {total_vram_bytes / _BYTES_PER_GIB:,.1f} GiB.",
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
                "n_paths", "n_steps", "phase4_ms", "phase5_ms", "speedup",
                "phase4_peak_bytes", "phase5_peak_bytes", "memory_ratio",
                "phase5_mpath_steps_per_s", "worst_relative_deviation",
                "phase4_oom", "phase5_oom",
                "phase4_predicted_oom", "phase5_predicted_oom",
                "phase4_predicted_bytes", "phase5_predicted_bytes",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.n_paths, row.n_steps,
                    "" if row.phase4.milliseconds is None else f"{row.phase4.milliseconds:.6f}",
                    "" if row.phase5.milliseconds is None else f"{row.phase5.milliseconds:.6f}",
                    "" if row.speedup is None else f"{row.speedup:.6f}",
                    "" if row.phase4.peak_bytes is None else row.phase4.peak_bytes,
                    "" if row.phase5.peak_bytes is None else row.phase5.peak_bytes,
                    "" if row.memory_ratio is None else f"{row.memory_ratio:.6f}",
                    "" if row.throughput_phase5 is None else f"{row.throughput_phase5:.3f}",
                    "" if row.worst_relative_deviation is None
                    else f"{row.worst_relative_deviation:.6e}",
                    int(row.phase4.failed_oom), int(row.phase5.failed_oom),
                    int(row.phase4.predicted_oom), int(row.phase5.predicted_oom),
                    "" if row.phase4.predicted_bytes is None else row.phase4.predicted_bytes,
                    "" if row.phase5.predicted_bytes is None else row.phase5.predicted_bytes,
                ]
            )


# ==========================================================================
# Entry point
# ==========================================================================


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI."""
    parser = argparse.ArgumentParser(
        prog="bench_phase5.py",
        description=(
            "Benchmark the fused O(N)-memory exposure kernel (Phase 5) against "
            "the materialised-path kernel (Phase 4)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paths", type=int, nargs="+",
        default=[1_000_000, 5_000_000, 10_000_000, 50_000_000],
        help="Path counts M to sweep.",
    )
    parser.add_argument("--steps", type=int, default=252, help="Time steps N.")
    parser.add_argument(
        "--dtype", choices=["float32", "float64"], default="float32",
        help="Working precision. float32 is the realistic choice at these scales.",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Timed iterations per measurement (minimum reported).",
    )
    parser.add_argument("--seed", type=int, default=20260820, help="Philox key.")
    parser.add_argument(
        "--max-programs", type=int, default=DEFAULT_MAX_PROGRAMS,
        help="Phase 5 launch-grid cap. Peak memory is proportional to this.",
    )
    parser.add_argument(
        "--backward", action="store_true",
        help="Time forward + backward. Phase 5 gains most here: rematerialisation "
        "means the adjoint allocates nothing of size M either.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output.")
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
    if args.steps <= 0 or args.repeats <= 0 or args.max_programs <= 0:
        print("--steps, --repeats and --max-programs must be positive", file=sys.stderr)
        return 2

    if not is_available():
        print(
            "\n  Cannot run: the fused Triton path is unavailable.\n"
            f"    triton installed : {HAS_TRITON}\n"
            f"    cuda available   : {torch.cuda.is_available()}\n\n"
            "  This benchmark is GPU-only. Run it on a CUDA machine with Triton\n"
            "  installed (e.g. Google Colab). The CPU-side tests in\n"
            "  tests/test_phase5.py run anywhere and validate the affine collapse\n"
            "  and the fused adjoint algebra.\n",
            file=sys.stderr,
        )
        return 1

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    properties = torch.cuda.get_device_properties(0)

    print()
    print("=" * 100)
    print("  FUSED EXPOSURE REDUCTION (Phase 5) vs MATERIALISED PATHS (Phase 4)")
    print("=" * 100)
    print(
        f"  Device    : {properties.name}  "
        f"({properties.total_memory / _BYTES_PER_GIB:,.1f} GiB)"
    )
    print(f"  torch     : {torch.__version__}   cuda {torch.version.cuda}")
    print(f"  dtype     : {args.dtype}   steps N = {args.steps:,}")
    print(
        f"  Timing    : torch.cuda.Event, {args.repeats} repeats "
        f"(minimum), warm-up excluded"
    )
    print(f"  Measuring : {'forward + backward' if args.backward else 'forward only'}")
    print(f"  P5 grid   : capped at {args.max_programs:,} programs")
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
            max_programs=args.max_programs,
        )
        rows.append(row)
        speedup = row.speedup
        print(
            f" P4 {_time_cell(row.phase4)} ms | P5 {_time_cell(row.phase5)} ms"
            f" | {f'{speedup:,.2f}x' if speedup is not None else 'n/a'}"
            f" | peak {_memory_cell(row.phase4)} -> {_memory_cell(row.phase5)} GiB"
        )

    print()
    print(render_table(rows))
    print()
    print(render_analysis(rows, dtype, properties.total_memory))
    print()

    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"  CSV written to {args.csv}")
        print()

    print("=" * 100)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
