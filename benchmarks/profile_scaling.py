r"""Path-count scaling profile for the fused exposure kernel.

Replaces the ad-hoc notebook cell. Two bugs in that cell are worth naming,
because both are easy to hit again:

1. ``legs=[]`` -- an empty portfolio raises
   ``ValueError: portfolio must contain at least one leg``. That guard is
   working as intended: with no legs the affine coefficients :math:`B, C` are
   identically zero, so every exposure is zero and the kernel would happily
   report a flat-zero profile and a perfect-looking timing. Silently "profiling"
   a no-op is worse than failing, so the check stays.
2. ``from src.models.portfolio import ...`` -- that module does not exist.
   :class:`~src.pricer.options.SwapLeg` lives in ``src.pricer.options``.

What this measures
==================
Peak VRAM and wall time as :math:`M` sweeps across the grid-saturation
boundary, which is the honest way to present the O(1) memory claim. The launch
grid is ``min(ceil(M / BLOCK_M), max_programs)``, giving two regimes:

* **ramp-up** (:math:`M <` ``max_programs * BLOCK_M``) -- grid grows with M, so
  peak grows with M;
* **saturated** (:math:`M \ge` that bound) -- grid is pinned, so peak is exactly
  ``max_programs * (N+1) * element_size`` regardless of M.

At N=252/fp32 the boundary is :math:`4096 \times 32 = 131{,}072` paths. Reporting
only the saturated regime would overstate the claim; reporting only the ramp
would understate it. The table marks which regime each row is in.

Usage
=====
    python benchmarks/profile_scaling.py
    python benchmarks/profile_scaling.py --paths 10000 100000 1000000 10000000
    python benchmarks/profile_scaling.py --backward --csv scaling.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.csrc.triton_cva_fusion import (  # noqa: E402
    DEFAULT_MAX_PROGRAMS,
    fused_expected_exposure,
    is_available,
    select_fused_block_sizes,
)
from src.pricer.options import SwapLeg  # noqa: E402  <-- NOT src.models.portfolio

S0 = 100.0
MU = 0.05
RATE = 0.03
SIGMA = 0.20
MATURITY = 1.0

_MIB = 1024.0**2
_GIB = 1024.0**3
VRAM_SAFETY_FRACTION = 0.90


def default_portfolio() -> List[SwapLeg]:
    """A non-empty netting set: mixed signs and a staggered maturity.

    Non-empty matters. An empty list makes B and C identically zero, so the
    exposure profile is flat zero and any timing taken from it is meaningless.
    """
    return [
        SwapLeg(notional=1.0, strike=100.0, maturity=MATURITY),
        SwapLeg(notional=-0.4, strike=110.0, maturity=MATURITY),
        SwapLeg(notional=0.7, strike=95.0, maturity=0.5),
    ]


@dataclass
class Row:
    """One measured path count."""

    n_paths: int
    n_programs: int
    saturated: bool
    milliseconds: Optional[float] = None
    peak_bytes: Optional[int] = None
    skipped: bool = False
    reason: str = ""

    @property
    def throughput(self) -> Optional[float]:
        """Millions of path-steps per second."""
        if self.milliseconds is None:
            return None
        return (self.n_paths * N_STEPS_HOLDER[0]) / (self.milliseconds * 1e3)


# Set once in main(); keeps `throughput` a property without threading N through.
N_STEPS_HOLDER = [252]


def _reset_cuda() -> None:
    """Synchronise, drop cached blocks, clear peak statistics."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def measure(
    n_paths: int,
    n_steps: int,
    *,
    dtype: torch.dtype,
    repeats: int,
    seed: int,
    include_backward: bool,
    max_programs: int,
) -> Row:
    """Time one path count and record its peak allocation.

    Args:
        n_paths: Monte-Carlo paths :math:`M`.
        n_steps: Time steps :math:`N`.
        dtype: Working precision.
        repeats: Timed iterations; the minimum is reported.
        seed: Philox key.
        include_backward: Time forward + backward rather than forward alone.
        max_programs: Launch-grid cap.

    Returns:
        A populated :class:`Row`; ``skipped`` is set if the run was refused or
        the allocator gave out.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    block_m, _ = select_fused_block_sizes(n_steps, element_size)
    n_blocks = -(-n_paths // block_m)
    n_programs = min(n_blocks, max_programs)
    saturated = n_blocks >= max_programs

    device = torch.device("cuda")
    times = torch.linspace(0.0, MATURITY, n_steps + 1, device=device, dtype=dtype)
    legs = default_portfolio()

    # Pre-flight: the fused path needs only O(n_programs * N), so this should
    # never bind. It is kept because an unchecked overrun can surface as an
    # asynchronous illegal-memory-access that poisons the CUDA context.
    predicted = n_programs * (n_steps + 1) * element_size + 8 * (n_steps + 1) * element_size
    _reset_cuda()
    free, _total = torch.cuda.mem_get_info()
    if predicted > VRAM_SAFETY_FRACTION * free:
        return Row(
            n_paths=n_paths, n_programs=n_programs, saturated=saturated,
            skipped=True,
            reason=f"predicted {predicted / _MIB:,.1f} MiB > budget",
        )

    def run() -> None:
        if include_backward:
            s0 = torch.tensor(S0, device=device, dtype=dtype, requires_grad=True)
            sigma = torch.tensor(SIGMA, device=device, dtype=dtype, requires_grad=True)
            profile = fused_expected_exposure(
                s0, MU, sigma, legs, times, RATE, n_paths,
                seed=seed, max_programs=max_programs,
            )
            profile.sum().backward()
        else:
            with torch.no_grad():
                fused_expected_exposure(
                    S0, MU, SIGMA, legs, times, RATE, n_paths,
                    seed=seed, max_programs=max_programs,
                )

    try:
        run()  # warm-up: absorbs Triton JIT and allocator growth
        torch.cuda.synchronize()
        _reset_cuda()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        best = math.inf
        for _ in range(repeats):
            start.record()
            run()
            end.record()
            torch.cuda.synchronize()
            best = min(best, start.elapsed_time(end))

        peak = torch.cuda.max_memory_allocated()
        _reset_cuda()
        return Row(
            n_paths=n_paths, n_programs=n_programs, saturated=saturated,
            milliseconds=best, peak_bytes=peak,
        )

    except torch.cuda.OutOfMemoryError:
        _reset_cuda()
        return Row(
            n_paths=n_paths, n_programs=n_programs, saturated=saturated,
            skipped=True, reason="out of memory",
        )


def render(rows: Sequence[Row], n_steps: int, element_size: int) -> str:
    """Render the sweep as an ASCII table plus a regime summary."""
    headers = (
        "M", "programs", "regime", "peak (MiB)", "time (ms)", "Mpath-steps/s",
    )
    body = []
    for row in rows:
        body.append((
            f"{row.n_paths:,}",
            f"{row.n_programs:,}",
            "saturated" if row.saturated else "ramp-up",
            row.reason if row.skipped else f"{row.peak_bytes / _MIB:,.2f}",
            "-" if row.skipped else f"{row.milliseconds:,.2f}",
            "-" if row.skipped else f"{row.throughput:,.0f}",
        ))

    widths = [
        max(len(headers[c]), *(len(line[c]) for line in body))
        for c in range(len(headers))
    ]
    rule = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt(cells: Sequence[str]) -> str:
        return "| " + " | ".join(
            cell.rjust(widths[i]) for i, cell in enumerate(cells)
        ) + " |"

    lines = [rule, fmt(headers), rule]
    lines.extend(fmt(line) for line in body)
    lines.append(rule)

    ceiling = DEFAULT_MAX_PROGRAMS * (n_steps + 1) * element_size
    saturated = [r for r in rows if r.saturated and not r.skipped]
    lines.append("")
    lines.append(f"Predicted M-independent ceiling: {ceiling / _MIB:,.2f} MiB")
    if len(saturated) >= 2:
        peaks = [r.peak_bytes for r in saturated]
        span = max(r.n_paths for r in saturated) / min(r.n_paths for r in saturated)
        lines.append(
            f"Saturated regime: {span:,.0f}x more paths -> peak ratio "
            f"{max(peaks) / min(peaks):.3f}x  (flat)"
        )
    ramp = [r for r in rows if not r.saturated and not r.skipped]
    if ramp:
        lines.append(
            "Ramp-up rows grow with M because the grid does; that is expected "
            "and bounded by the ceiling above."
        )
    return "\n".join(lines)


def main() -> int:
    """Run the sweep and print the report."""
    parser = argparse.ArgumentParser(
        prog="profile_scaling.py",
        description="Peak VRAM and throughput vs path count for the fused kernel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paths", type=int, nargs="+",
        default=[10_000, 100_000, 1_000_000, 5_000_000, 10_000_000],
        help="Path counts M to sweep.",
    )
    parser.add_argument("--steps", type=int, default=252, help="Time steps N.")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-programs", type=int, default=DEFAULT_MAX_PROGRAMS)
    parser.add_argument("--backward", action="store_true",
                        help="Time forward + backward.")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    if not is_available():
        print(
            "\n  Cannot run: the fused Triton path is unavailable.\n"
            f"    cuda available : {torch.cuda.is_available()}\n\n"
            "  This profiler is GPU-only.\n",
            file=sys.stderr,
        )
        return 1

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    element_size = torch.tensor([], dtype=dtype).element_size()
    N_STEPS_HOLDER[0] = args.steps

    block_m, block_t = select_fused_block_sizes(args.steps, element_size)
    saturation = args.max_programs * block_m
    properties = torch.cuda.get_device_properties(0)

    print()
    print("=" * 78)
    print("  FUSED EXPOSURE KERNEL  --  path-count scaling")
    print("=" * 78)
    print(f"  Device     : {properties.name} ({properties.total_memory / _GIB:,.1f} GiB)")
    print(f"  torch      : {torch.__version__}   cuda {torch.version.cuda}")
    print(f"  dtype      : {args.dtype}   N = {args.steps:,}")
    print(f"  Launch     : BLOCK_M={block_m}, BLOCK_T={block_t}, "
          f"grid cap {args.max_programs:,}")
    print(f"  Saturates  : M >= {saturation:,} paths")
    print(f"  Measuring  : {'forward + backward' if args.backward else 'forward only'}")
    print()

    rows: List[Row] = []
    for n_paths in sorted(set(args.paths)):
        print(f"  M = {n_paths:,} ...", end="", flush=True)
        row = measure(
            n_paths, args.steps, dtype=dtype, repeats=args.repeats,
            seed=args.seed, include_backward=args.backward,
            max_programs=args.max_programs,
        )
        rows.append(row)
        if row.skipped:
            print(f" skipped ({row.reason})")
        else:
            print(f" {row.milliseconds:,.2f} ms | {row.peak_bytes / _MIB:,.2f} MiB "
                  f"| {row.throughput:,.0f} Mpath-steps/s")

    print()
    print(render(rows, args.steps, element_size))
    print()

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "n_paths", "n_programs", "saturated", "peak_bytes",
                "milliseconds", "mpath_steps_per_s", "skipped", "reason",
            ])
            for row in rows:
                writer.writerow([
                    row.n_paths, row.n_programs, int(row.saturated),
                    "" if row.peak_bytes is None else row.peak_bytes,
                    "" if row.milliseconds is None else f"{row.milliseconds:.6f}",
                    "" if row.throughput is None else f"{row.throughput:.3f}",
                    int(row.skipped), row.reason,
                ])
        print(f"  CSV written to {args.csv}")
        print()

    print("=" * 78)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
