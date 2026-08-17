r"""Scaling benchmark: AAD is O(1) in risk-factor count, bump-and-revalue is O(n).

The claim being measured
========================
Reverse-mode adjoint algorithmic differentiation computes **all** :math:`n`
sensitivities of a scalar valuation in a single backward sweep whose cost is a
small constant multiple of one forward pass:

.. math:: T_{\text{AAD}}(n) = O(C), \qquad\text{independent of } n.

Bump-and-revalue must revalue the whole Monte-Carlo engine twice per risk
factor (central differences), plus once for the base:

.. math:: T_{\text{FD}}(n) = (2n + 1)\,O(C) = O(nC).

So the speedup should grow **linearly** in :math:`n`. On a real XVA book
carrying hundreds of risk factors this is the difference between an overnight
batch and an intraday one, and it is the central performance argument of the
thesis.

Experimental design -- why the mock is built this way
=====================================================
Naively adding risk factors by adding *assets* would confound the measurement:
the Monte-Carlo cost would grow with :math:`n` too, and both methods would
slow down together, hiding the effect under test.

Instead the simulation cost is held **fixed** while the risk-factor count
varies. The :math:`n` mock factors are a variance decomposition of a single
asset's volatility across independent risk buckets:

.. math:: \sigma_{\text{eff}} = \sqrt{\textstyle\sum_{i=1}^{n} \sigma_i^2},

which is how a multi-factor volatility model actually aggregates. Every
:math:`\sigma_i` is a genuine risk factor with a distinct non-zero gradient, but
combining them costs :math:`O(n)` scalar operations against an
:math:`O(M \times N)` simulation -- negligible, and reported explicitly by the
``forward (ms)`` column so the reader can confirm it stays flat.

The valuation itself is the full Phase 2 pipeline: GBM paths, netted portfolio
MtM, expected exposure, unilateral CVA. Nothing is stubbed.

Usage
=====
    python benchmarks/bench_phase2.py
    python benchmarks/bench_phase2.py --factors 1 5 10 50 100 --repeats 3
    python benchmarks/bench_phase2.py --paths 50000 --steps 100 --csv results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Mapping, Sequence

import torch
from torch import Tensor

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.gbm import GBMSimulator, resolve_device, simulate_gbm  # noqa: E402
from src.pricer.greeks import aad_greeks, finite_difference_greeks  # noqa: E402
from src.pricer.options import SwapLeg, portfolio_swap_mtm  # noqa: E402
from src.xva.cva import compute_unilateral_cva  # noqa: E402
from src.xva.exposure import expected_exposure  # noqa: E402

DTYPE = torch.float64

# Market / credit setup held fixed across the whole sweep.
SPOT = 100.0
STRIKE = 100.0
RATE = 0.03
BASE_VOL = 0.20
MATURITY = 1.0
HAZARD_RATE = 0.02
RECOVERY_RATE = 0.4


# ==========================================================================
# The n-risk-factor CVA valuation
# ==========================================================================
def make_multifactor_cva_fn(
    simulator: GBMSimulator,
    dW: Tensor,
    legs: Sequence[SwapLeg],
    n_factors: int,
):
    r"""Build a CVA closure driven by ``n_factors`` independent volatility factors.

    The factors aggregate as a variance decomposition,
    :math:`\sigma_{\text{eff}} = \sqrt{\sum_i \sigma_i^2}`, so each carries a
    distinct non-zero sensitivity
    :math:`\partial\sigma_{\text{eff}}/\partial\sigma_i = \sigma_i/\sigma_{\text{eff}}`
    and none is a degenerate duplicate of another.

    Args:
        simulator: Configured simulator supplying the grid and step size.
        dW: Fixed Brownian increments, captured so that AAD and
            bump-and-revalue differentiate the same Monte-Carlo realisation.
        legs: Portfolio legs defining the netting set.
        n_factors: Number of mock risk factors :math:`n`.

    Returns:
        A callable ``price_fn(params) -> Tensor`` expecting keys ``"vol_0"``
        through ``f"vol_{n_factors - 1}"``.
    """
    legs = tuple(legs)
    times = simulator.time_grid()
    factor_names = [f"vol_{index}" for index in range(n_factors)]

    def cva_fn(params: Mapping[str, Tensor]) -> Tensor:
        # Aggregate the factor variances. torch.stack keeps this a single
        # graph node rather than an n-deep chain of additions.
        variances = torch.stack([params[name] ** 2 for name in factor_names])
        effective_vol = torch.sqrt(variances.sum())

        paths = simulate_gbm(SPOT, RATE, effective_vol, dW, simulator.dt)
        mtm = portfolio_swap_mtm(paths, times, legs, RATE)
        return compute_unilateral_cva(
            expected_exposure(mtm),
            times,
            HAZARD_RATE,
            RECOVERY_RATE,
            discount_rate=RATE,
        )

    return cva_fn


def make_factor_params(n_factors: int) -> Dict[str, float]:
    r"""Split the base volatility evenly across ``n_factors`` buckets.

    Each factor gets :math:`\sigma_i = \sigma_{\text{base}}/\sqrt{n}` so that
    :math:`\sigma_{\text{eff}} = \sigma_{\text{base}}` regardless of :math:`n`.
    Holding the *effective* volatility constant means the CVA being
    differentiated is identical at every rung of the sweep, so the timings
    compare like with like.

    Args:
        n_factors: Number of mock risk factors.

    Returns:
        Mapping from factor name to its initial value.
    """
    per_factor = BASE_VOL / math.sqrt(n_factors)
    return {f"vol_{index}": per_factor for index in range(n_factors)}


# ==========================================================================
# Timing
# ==========================================================================
@dataclass
class BenchmarkRow:
    """One rung of the sweep."""

    n_factors: int
    forward_ms: float
    aad_ms: float
    fd_ms: float
    aad_valuations: int
    fd_valuations: int
    max_abs_error: float

    @property
    def speedup(self) -> float:
        """How many times faster AAD is than bump-and-revalue."""
        return self.fd_ms / self.aad_ms if self.aad_ms > 0.0 else float("inf")

    @property
    def aad_cost_in_forwards(self) -> float:
        """AAD wall time expressed as a multiple of one forward valuation."""
        return self.aad_ms / self.forward_ms if self.forward_ms > 0.0 else float("nan")


def _time_call(function, repeats: int) -> float:
    """Return the median wall time of ``function`` in milliseconds.

    The median is used rather than the mean because a single OS scheduling
    hiccup would otherwise dominate a small sample.

    Args:
        function: Zero-argument callable to time.
        repeats: Number of timed repetitions.

    Returns:
        Median elapsed time in milliseconds.
    """
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        samples.append((perf_counter() - start) * 1e3)
    return statistics.median(samples)


def benchmark_one(
    n_factors: int,
    *,
    n_paths: int,
    n_steps: int,
    repeats: int,
    seed: int,
    device: torch.device,
) -> BenchmarkRow:
    """Measure AAD and bump-and-revalue timings at one risk-factor count.

    Args:
        n_factors: Number of mock risk factors :math:`n`.
        n_paths: Monte-Carlo paths, held fixed across the sweep.
        n_steps: Time steps, held fixed across the sweep.
        repeats: Timed repetitions; the median is reported.
        seed: RNG seed for the Brownian sample.
        device: Execution device.

    Returns:
        A populated :class:`BenchmarkRow`.
    """
    simulator = GBMSimulator(
        maturity=MATURITY, n_steps=n_steps, device=device, dtype=DTYPE
    )
    dW = simulator.draw_increments(n_paths, seed=seed)
    legs = [SwapLeg(notional=1.0, strike=STRIKE, maturity=MATURITY)]

    cva_fn = make_multifactor_cva_fn(simulator, dW, legs, n_factors)
    params = make_factor_params(n_factors)

    # A bare forward valuation, for the "how many forwards does AAD cost?"
    # column. Run under no_grad so no tape is built.
    def forward_only() -> None:
        with torch.no_grad():
            tensor_params = {
                name: torch.as_tensor(value, dtype=DTYPE, device=device)
                for name, value in params.items()
            }
            cva_fn(tensor_params)

    # Warm-up: the first call pays for lazy allocator and kernel setup, which
    # would otherwise be charged to whichever method happens to run first.
    forward_only()
    aad_greeks(cva_fn, params)

    forward_ms = _time_call(forward_only, repeats)
    aad_ms = _time_call(lambda: aad_greeks(cva_fn, params), repeats)
    fd_ms = _time_call(
        lambda: finite_difference_greeks(cva_fn, params, scheme="central"), repeats
    )

    # Correctness gate: a speed comparison between two methods that disagree
    # would be meaningless, so verify they produce the same Greeks.
    aad = aad_greeks(cva_fn, params)
    fd = finite_difference_greeks(cva_fn, params, scheme="central")
    max_abs_error = max(
        abs(aad.greeks[name] - fd.greeks[name]) for name in aad.greeks
    )

    return BenchmarkRow(
        n_factors=n_factors,
        forward_ms=forward_ms,
        aad_ms=aad_ms,
        fd_ms=fd_ms,
        aad_valuations=aad.n_valuations,
        fd_valuations=fd.n_valuations,
        max_abs_error=max_abs_error,
    )


# ==========================================================================
# Reporting
# ==========================================================================
def render_table(rows: Sequence[BenchmarkRow]) -> str:
    """Render the sweep as an ASCII table.

    Args:
        rows: Completed benchmark rows, in sweep order.

    Returns:
        The formatted table as a multi-line string.
    """
    headers = (
        "n", "forward (ms)", "AAD (ms)", "FD (ms)", "speedup",
        "AAD/fwd", "AAD vals", "FD vals", "max |err|",
    )
    body = [
        (
            f"{row.n_factors:,}",
            f"{row.forward_ms:,.2f}",
            f"{row.aad_ms:,.2f}",
            f"{row.fd_ms:,.2f}",
            f"{row.speedup:,.2f}x",
            f"{row.aad_cost_in_forwards:,.2f}x",
            f"{row.aad_valuations:,}",
            f"{row.fd_valuations:,}",
            f"{row.max_abs_error:.2e}",
        )
        for row in rows
    ]

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


def render_analysis(rows: Sequence[BenchmarkRow]) -> str:
    """Summarise the observed complexity against the theoretical prediction.

    Args:
        rows: Completed benchmark rows, in sweep order.

    Returns:
        A multi-line analysis block.
    """
    if len(rows) < 2:
        return "Need at least two rungs to infer scaling."

    first, last = rows[0], rows[-1]
    factor_growth = last.n_factors / first.n_factors
    aad_growth = last.aad_ms / first.aad_ms if first.aad_ms > 0 else float("nan")
    fd_growth = last.fd_ms / first.fd_ms if first.fd_ms > 0 else float("nan")

    lines = [
        "ANALYSIS",
        "-" * 74,
        f"  Risk factors grew           {factor_growth:>10,.1f}x   "
        f"(n = {first.n_factors} -> {last.n_factors})",
        f"  AAD wall time grew          {aad_growth:>10,.2f}x   "
        f"(theory: ~1x, independent of n)",
        f"  FD  wall time grew          {fd_growth:>10,.2f}x   "
        f"(theory: ~{factor_growth:,.1f}x, linear in n)",
        "",
        f"  Speedup at n={first.n_factors:<4}          {first.speedup:>10,.2f}x",
        f"  Speedup at n={last.n_factors:<4}          {last.speedup:>10,.2f}x",
        "",
        f"  Worst AAD-vs-FD disagreement across the sweep: "
        f"{max(row.max_abs_error for row in rows):.2e}",
        "",
        "  Reading this table:",
        "    * 'AAD/fwd' is the cost of a full gradient as a multiple of one",
        "      forward valuation. Reverse-mode theory bounds this by a small",
        "      constant (typically 2-5x) regardless of n -- if that column stays",
        "      flat while 'FD vals' grows as 2n+1, the O(1)-vs-O(n) claim holds.",
        "    * 'forward (ms)' should also stay flat: it confirms the mock adds",
        "      risk factors without inflating the simulation itself, which is",
        "      what makes the comparison fair.",
    ]
    return "\n".join(lines)


def write_csv(rows: Sequence[BenchmarkRow], destination: Path) -> None:
    """Persist the sweep for later plotting or inclusion in the write-up.

    Args:
        rows: Completed benchmark rows.
        destination: Output CSV path.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "n_factors", "forward_ms", "aad_ms", "fd_ms", "speedup",
                "aad_cost_in_forwards", "aad_valuations", "fd_valuations",
                "max_abs_error",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.n_factors, f"{row.forward_ms:.6f}", f"{row.aad_ms:.6f}",
                    f"{row.fd_ms:.6f}", f"{row.speedup:.6f}",
                    f"{row.aad_cost_in_forwards:.6f}", row.aad_valuations,
                    row.fd_valuations, f"{row.max_abs_error:.6e}",
                ]
            )


# ==========================================================================
# Entry point
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI."""
    parser = argparse.ArgumentParser(
        prog="bench_phase2.py",
        description=(
            "Measure AAD vs bump-and-revalue wall time as the number of risk "
            "factors grows, demonstrating O(1) vs O(n) scaling."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--factors", type=int, nargs="+", default=[1, 5, 10, 50, 100],
        help="Risk-factor counts to sweep.",
    )
    parser.add_argument("--paths", type=int, default=20_000, help="Monte-Carlo paths M.")
    parser.add_argument("--steps", type=int, default=64, help="Time steps N.")
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timed repetitions per measurement (median reported)."
    )
    parser.add_argument("--seed", type=int, default=20260817, help="RNG seed.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is visible.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    return parser


def main() -> int:
    """Run the sweep and print the report.

    Returns:
        Process exit status: ``0`` on success, ``2`` on invalid arguments.
    """
    args = build_parser().parse_args()

    if any(n <= 0 for n in args.factors):
        print("--factors must all be positive", file=sys.stderr)
        return 2
    if args.paths <= 0 or args.steps <= 0 or args.repeats <= 0:
        print("--paths, --steps and --repeats must all be positive", file=sys.stderr)
        return 2

    device = torch.device("cpu") if args.cpu else resolve_device()
    factors = sorted(set(args.factors))

    print()
    print("=" * 74)
    print("  AAD vs BUMP-AND-REVALUE  --  scaling in the number of risk factors")
    print("=" * 74)
    print(
        f"  Valuation : GBM -> portfolio MtM -> EE -> unilateral CVA\n"
        f"  Paths     : {args.paths:,}   Steps: {args.steps:,}   "
        f"dtype: {str(DTYPE).replace('torch.', '')}\n"
        f"  Device    : {device}\n"
        f"  Repeats   : {args.repeats} (median reported)   Seed: {args.seed:,}"
    )
    print()

    rows: List[BenchmarkRow] = []
    for n_factors in factors:
        print(f"  running n = {n_factors:,} ...", end="", flush=True)
        row = benchmark_one(
            n_factors,
            n_paths=args.paths,
            n_steps=args.steps,
            repeats=args.repeats,
            seed=args.seed,
            device=device,
        )
        rows.append(row)
        print(f" AAD {row.aad_ms:,.1f} ms | FD {row.fd_ms:,.1f} ms | {row.speedup:,.1f}x")

    print()
    print(render_table(rows))
    print()
    print(render_analysis(rows))
    print()

    if args.csv is not None:
        write_csv(rows, args.csv)
        print(f"  CSV written to {args.csv}")
        print()

    print("=" * 74)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
