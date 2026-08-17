r"""Interactive manual sandbox for the Phase 2 XVA engine.

PURPOSE
=======
This script is a *hands-on inspection harness*, not part of the library. It
exists so that a human can nudge one market or credit parameter at a time from
the terminal and immediately see three things:

    1. how the exposure profile (EE / PFE) reshapes,
    2. how the CVA number moves,
    3. how the AAD sensitivities move.

Everything the engine does in Phases 1-2 is exercised end-to-end in one file,
in the order the mathematics actually runs:

    market + credit parameters  (leaf tensors, requires_grad=True)
        |
        v   src.models.gbm.GBMSimulator.simulate
    GBM asset paths                     shape (n_paths, n_steps + 1)
        |
        v   src.pricer.options.portfolio_swap_mtm
    netted mark-to-market surface       shape (n_paths, n_steps + 1)
        |
        v   src.xva.exposure.compute_exposure_profile
    EE(t), ENE(t), PFE(t)               shape (n_steps + 1,)
        |
        v   src.xva.cva.compute_unilateral_cva
    CVA                                 scalar
        |
        v   cva.backward()
    dCVA/dS0, dCVA/dsigma, dCVA/dlambda, dCVA/dr   -- ONE reverse sweep

THE HEADLINE PROPERTY BEING DEMONSTRATED
========================================
That entire chain is a single unbroken autograd graph. One call to
``cva.backward()`` populates *every* sensitivity simultaneously, at a cost of
roughly one extra forward pass -- regardless of how many risk factors there
are. The bump-and-revalue alternative needs ``2n + 1`` full Monte-Carlo
revaluations for ``n`` risk factors. Run with ``--verify`` to see both methods
side by side on the same random draw.

USAGE
=====
    python manual_sandbox.py
    python manual_sandbox.py --vol 0.45 --hazard 0.08
    python manual_sandbox.py --help

Writes ``sandbox_exposures.png`` to the repository root on every run.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import torch

# Matplotlib must be told to use a non-interactive backend *before* pyplot is
# imported. Without this the script can block or fail outright when run over a
# terminal with no display attached (which is the normal case here).
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (deliberately after the backend call)

# Running `python manual_sandbox.py` from the repo root already puts that root
# on sys.path, so `from src...` resolves. This guard only matters if the script
# is invoked from some other working directory.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.gbm import GBMSimulator, resolve_device  # noqa: E402
from src.pricer.greeks import compare_greeks, format_comparison  # noqa: E402
from src.pricer.options import SwapLeg, portfolio_swap_mtm  # noqa: E402
from src.xva.cva import (  # noqa: E402
    compute_unilateral_cva,
    cva_aad_greeks,
    cva_bump_and_revalue_greeks,
    make_cva_valuation_fn,
)
from src.xva.exposure import compute_exposure_profile  # noqa: E402

# --------------------------------------------------------------------------
# Presentation constants.
#
# These hexes are a validated categorical pair: measured CVD separation
# dE 24.7 (protan) and normal-vision separation dE 33.6 against the light
# surface, both comfortably clear of the >=8 / >=15 thresholds. Do not swap
# them for arbitrary colours without re-validating -- the point of a fixed pair
# is that EE and PFE stay distinguishable for a colour-blind reader.
# --------------------------------------------------------------------------
COLOR_SURFACE = "#fcfcfb"
COLOR_INK_PRIMARY = "#0b0b0b"
COLOR_INK_SECONDARY = "#52514e"
COLOR_INK_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_EE = "#2a78d6"    # categorical slot 1 (blue)
COLOR_PFE = "#eb6834"   # categorical slot 2 (orange)

# float64 throughout. This is a correctness/inspection tool, and the optional
# --verify path does central finite differences, which lose roughly half their
# significant digits in float32 and would make the comparison meaningless.
DTYPE = torch.float64

DEFAULT_PLOT_PATH = _REPO_ROOT / "sandbox_exposures.png"


# ==========================================================================
# Argument parsing
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI.

    The five *market/credit* inputs called for by the Phase 2 review are
    ``--spot``, ``--vol``, ``--rate``, ``--hazard`` and ``--recovery``. The
    remaining flags control the simulation itself and are provided so the
    sandbox can be pushed around (path count, grid resolution, horizon) without
    editing the file.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="manual_sandbox.py",
        description=(
            "Interactive XVA sandbox: nudge market/credit parameters, inspect "
            "the resulting exposure profile, CVA and AAD Greeks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    market = parser.add_argument_group("market inputs")
    market.add_argument("--spot", type=float, default=100.0, help="Initial spot price S0.")
    market.add_argument("--vol", type=float, default=0.20, help="Volatility sigma (annualised).")
    market.add_argument("--rate", type=float, default=0.05, help="Risk-free rate r (continuous).")

    credit = parser.add_argument_group("credit inputs")
    credit.add_argument(
        "--hazard", type=float, default=0.02, help="Flat hazard rate lambda of the counterparty."
    )
    credit.add_argument(
        "--recovery", type=float, default=0.40, help="Recovery rate R in [0, 1]."
    )

    contract = parser.add_argument_group("contract")
    contract.add_argument(
        "--notional",
        type=float,
        default=1.0,
        help="Signed notional of the forward. Negative flips the direction of the trade.",
    )
    contract.add_argument(
        "--maturity", type=float, default=1.0, help="Contract maturity T in years."
    )

    simulation = parser.add_argument_group("simulation controls")
    simulation.add_argument("--paths", type=int, default=10_000, help="Monte-Carlo paths M.")
    simulation.add_argument("--steps", type=int, default=252, help="Time steps N.")
    simulation.add_argument(
        "--confidence", type=float, default=0.95, help="PFE quantile level in [0, 1]."
    )
    simulation.add_argument(
        "--seed",
        type=int,
        default=20260814,
        help="RNG seed. Hold this fixed when comparing scenarios so differences "
        "come from the parameter you nudged, not from resampling.",
    )
    simulation.add_argument(
        "--antithetic",
        action="store_true",
        help="Use antithetic variates (halves the independent draws, reduces variance). "
        "Requires an even --paths.",
    )
    simulation.add_argument(
        "--cpu", action="store_true", help="Force CPU even when a CUDA device is visible."
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--plot",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Destination PNG for the exposure chart.",
    )
    output.add_argument(
        "--no-plot", action="store_true", help="Skip chart generation (numbers only)."
    )
    output.add_argument(
        "--verify",
        action="store_true",
        help="Also run bump-and-revalue finite differences on the same random draw "
        "and print an AAD-vs-FD comparison table.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Reject parameter combinations the engine cannot represent.

    Args:
        args: Parsed namespace.

    Raises:
        SystemExit: Via :meth:`argparse.ArgumentParser.error`-style messaging,
            with a non-zero status, so a bad invocation fails fast and loudly
            rather than producing a misleading chart.
    """
    problems = []
    if args.spot <= 0.0:
        problems.append(f"--spot must be positive (got {args.spot})")
    if args.vol <= 0.0:
        problems.append(f"--vol must be positive (got {args.vol})")
    if not 0.0 <= args.recovery <= 1.0:
        problems.append(f"--recovery must lie in [0, 1] (got {args.recovery})")
    if not 0.0 <= args.confidence <= 1.0:
        problems.append(f"--confidence must lie in [0, 1] (got {args.confidence})")
    if args.maturity <= 0.0:
        problems.append(f"--maturity must be positive (got {args.maturity})")
    if args.paths <= 0:
        problems.append(f"--paths must be positive (got {args.paths})")
    if args.steps <= 0:
        problems.append(f"--steps must be positive (got {args.steps})")
    if args.antithetic and args.paths % 2 != 0:
        problems.append(f"--antithetic requires an even --paths (got {args.paths})")
    if args.hazard < 0.0:
        problems.append(f"--hazard must be non-negative (got {args.hazard})")

    if problems:
        print("Invalid arguments:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        raise SystemExit(2)


# ==========================================================================
# Results container
# ==========================================================================
@dataclass
class SandboxResult:
    """Everything one sandbox run produces, ready for printing and plotting.

    Profile arrays are stored as detached NumPy copies because they are only
    consumed by matplotlib and the console report at that point -- keeping them
    as live graph nodes would pin the whole autograd tape in memory for no
    reason.
    """

    # Inputs, echoed back so the chart and the console agree on what was run.
    spot: float
    vol: float
    rate: float
    hazard: float
    recovery: float
    notional: float
    strike: float
    maturity: float
    n_paths: int
    n_steps: int
    confidence: float
    seed: int
    antithetic: bool
    device: str

    # Profiles (detached, NumPy).
    times: np.ndarray
    ee: np.ndarray
    ene: np.ndarray
    pfe: np.ndarray

    # Scalars.
    cva: float
    epe: float
    max_pfe: float
    peak_pfe_time: float
    risk_free_value: float

    # Sensitivities from the single backward pass.
    greeks: Dict[str, float] = field(default_factory=dict)


# ==========================================================================
# The pipeline
# ==========================================================================
def run_pipeline(args: argparse.Namespace) -> SandboxResult:
    """Run simulation -> MtM -> exposure -> CVA -> backward, and collect results.

    Args:
        args: Parsed and validated CLI namespace.

    Returns:
        A populated :class:`SandboxResult`.
    """
    device = torch.device("cpu") if args.cpu else resolve_device()

    # ----------------------------------------------------------------------
    # 1. Parameters as autograd leaves.
    #
    # These four are the differentiation targets. Each is a 0-dim leaf tensor
    # with requires_grad=True, so `.grad` will be populated by the single
    # backward() call at the end.
    #
    # `rate` is included as a leaf on purpose: because the engine uses it as
    # BOTH the risk-neutral drift and the discount rate (see
    # `resolve_rate_and_drift` in src/pricer/options.py), differentiating
    # w.r.t. it yields the *total* Rho, and it costs nothing extra -- it rides
    # along in the same reverse sweep. That is the whole point of AAD.
    # ----------------------------------------------------------------------
    spot = torch.tensor(args.spot, dtype=DTYPE, device=device, requires_grad=True)
    vol = torch.tensor(args.vol, dtype=DTYPE, device=device, requires_grad=True)
    rate = torch.tensor(args.rate, dtype=DTYPE, device=device, requires_grad=True)
    hazard = torch.tensor(args.hazard, dtype=DTYPE, device=device, requires_grad=True)

    # `recovery` is deliberately NOT a leaf tensor. It is a contractual/credit
    # assumption rather than a market observable, no recovery sensitivity was
    # requested, and the engine validates it as a plain float in [0, 1].
    recovery = float(args.recovery)

    # ----------------------------------------------------------------------
    # 2. The contract.
    #
    # SUBTLE BUT IMPORTANT: the strike is a *float snapshot* of the spot, not
    # the `spot` tensor itself. A forward struck "at the initial spot" fixes K
    # contractually at inception; it does not move when we later differentiate
    # with respect to S0. Writing `strike=spot` would silently make the strike
    # track the bumped spot and would produce a Delta of essentially zero --
    # a classic and very hard-to-spot AAD wiring bug.
    # ----------------------------------------------------------------------
    strike = float(args.spot)
    legs = [SwapLeg(notional=args.notional, strike=strike, maturity=args.maturity)]

    # ----------------------------------------------------------------------
    # 3. Simulator.
    #
    # NOTE the keyword is `maturity`, not `T` (see GBMSimulator in
    # src/models/gbm.py). `n_steps` is the number of intervals, so the time
    # grid and every profile below have length n_steps + 1.
    # ----------------------------------------------------------------------
    simulator = GBMSimulator(
        maturity=args.maturity,
        n_steps=args.steps,
        device=device,
        dtype=DTYPE,
        antithetic=args.antithetic,
    )
    times = simulator.time_grid()

    # Draw the Brownian increments explicitly rather than letting `simulate`
    # do it internally. Two reasons:
    #   (a) the draw is then reproducible from --seed, so two runs differing
    #       only in --vol really do differ only in vol; and
    #   (b) --verify can reuse the identical sample, which is what makes
    #       bump-and-revalue a fair oracle (common random numbers).
    dW = simulator.draw_increments(args.paths, seed=args.seed)

    # `simulate` takes exactly one of `dW` or `n_paths` (passing both raises).
    # Drift is the risk-free rate: we are pricing under Q.
    paths = simulator.simulate(spot, rate, vol, dW=dW)

    # ----------------------------------------------------------------------
    # 4. Mark-to-market surface, shape (n_paths, n_steps + 1).
    # ----------------------------------------------------------------------
    mtm = portfolio_swap_mtm(paths, times, legs, rate)

    # ----------------------------------------------------------------------
    # 5. Exposure profiles. All still on the tape.
    # ----------------------------------------------------------------------
    profile = compute_exposure_profile(mtm, times, confidence_level=args.confidence)

    # ----------------------------------------------------------------------
    # 6. CVA. `discount_rate` is keyword-only and mutually exclusive with an
    #    explicit `curve`.
    # ----------------------------------------------------------------------
    cva = compute_unilateral_cva(
        profile.ee,
        times,
        hazard,
        recovery,
        discount_rate=rate,
    )

    # Grab everything needed for reporting BEFORE backward() frees the graph.
    # (The tensor *values* survive backward; only the graph is released. These
    # are detached copies either way, so the order is defensive rather than
    # strictly required.)
    epe = float(profile.epe.detach())
    max_pfe = float(profile.max_pfe.detach())
    times_np = times.detach().cpu().numpy()
    ee_np = profile.ee.detach().cpu().numpy()
    ene_np = profile.ene.detach().cpu().numpy()
    pfe_np = profile.pfe.detach().cpu().numpy()
    risk_free_value = float(mtm[0, 0].detach())  # t=0 MtM is identical on every path

    # ----------------------------------------------------------------------
    # 7. THE SINGLE BACKWARD PASS.
    #
    # One call. Four sensitivities. This is the entire thesis argument in one
    # line: the cost is ~one extra forward pass and is INDEPENDENT of how many
    # leaves are attached.
    # ----------------------------------------------------------------------
    cva.backward()

    greeks = {
        "delta_dCVA_dS0": _grad_of(spot, "spot"),
        "vega_dCVA_dsigma": _grad_of(vol, "vol"),
        "credit_dCVA_dlambda": _grad_of(hazard, "hazard"),
        "rho_dCVA_dr": _grad_of(rate, "rate"),
    }

    peak_index = int(np.argmax(pfe_np))

    return SandboxResult(
        spot=args.spot,
        vol=args.vol,
        rate=args.rate,
        hazard=args.hazard,
        recovery=recovery,
        notional=args.notional,
        strike=strike,
        maturity=args.maturity,
        n_paths=args.paths,
        n_steps=args.steps,
        confidence=args.confidence,
        seed=args.seed,
        antithetic=args.antithetic,
        device=str(device),
        times=times_np,
        ee=ee_np,
        ene=ene_np,
        pfe=pfe_np,
        cva=float(cva.detach()),
        epe=epe,
        max_pfe=max_pfe,
        peak_pfe_time=float(times_np[peak_index]),
        risk_free_value=risk_free_value,
        greeks=greeks,
    )


def _grad_of(leaf: torch.Tensor, name: str) -> float:
    """Read a populated ``.grad`` off a leaf, failing loudly if it is missing.

    A ``None`` gradient here means the tape was severed somewhere upstream --
    typically an accidental ``.detach()``, ``.item()`` or NumPy round-trip. It
    is far better to crash with a clear message than to silently report a
    sensitivity of zero.

    Args:
        leaf: The parameter tensor.
        name: Human-readable name for the error message.

    Returns:
        The gradient as a Python float.

    Raises:
        RuntimeError: If no gradient reached ``leaf``.
    """
    if leaf.grad is None:
        raise RuntimeError(
            f"no gradient reached '{name}': the autograd graph is broken upstream "
            "(look for .detach(), .item(), torch.no_grad() or a NumPy conversion)"
        )
    return float(leaf.grad)


# ==========================================================================
# Charting
# ==========================================================================
def plot_exposure_profile(result: SandboxResult, destination: Path) -> None:
    """Render the EE and PFE curves to a PNG.

    Design notes (why the chart looks the way it does):

    * **One y-axis.** EE and PFE are the same measure in the same units, so
      they share a scale. A second axis would let the eye infer crossings that
      do not exist.
    * **Two validated hues plus direct labels.** Identity is never carried by
      colour alone: each curve is named at its right-hand end and in the
      legend, so the chart still reads in greyscale or with colour-vision
      deficiency.
    * **Shaded area under EE.** CVA is the discounted, default-weighted
      integral of exactly this curve, so the fill is the quantity being
      integrated -- it is meaningful, not decoration. PFE is a limit-monitoring
      line and gets no fill, because nothing integrates it.

    Args:
        result: A completed :class:`SandboxResult`.
        destination: Output path for the PNG.
    """
    figure, axes = plt.subplots(figsize=(11.0, 6.2), dpi=150)
    figure.patch.set_facecolor(COLOR_SURFACE)
    axes.set_facecolor(COLOR_SURFACE)

    # --- marks ---------------------------------------------------------
    axes.fill_between(result.times, 0.0, result.ee, color=COLOR_EE, alpha=0.10, linewidth=0)
    axes.plot(result.times, result.ee, color=COLOR_EE, linewidth=2.0, label="Expected Exposure (EE)",
              solid_capstyle="round")
    axes.plot(result.times, result.pfe, color=COLOR_PFE, linewidth=2.0,
              label=f"Potential Future Exposure (PFE, {result.confidence:.0%})",
              solid_capstyle="round")

    # --- direct labels at the right-hand end ---------------------------
    # Text stays in ink colours; the adjacent line carries the identity.
    x_end = float(result.times[-1])
    x_pad = 0.018 * (x_end - float(result.times[0]))
    axes.annotate(
        "PFE", xy=(x_end, float(result.pfe[-1])), xytext=(x_end + x_pad, float(result.pfe[-1])),
        color=COLOR_INK_SECONDARY, fontsize=10, va="center", ha="left", annotation_clip=False,
    )
    axes.annotate(
        "EE", xy=(x_end, float(result.ee[-1])), xytext=(x_end + x_pad, float(result.ee[-1])),
        color=COLOR_INK_SECONDARY, fontsize=10, va="center", ha="left", annotation_clip=False,
    )

    # --- peak PFE marker: the number a credit officer actually looks at -
    axes.plot(
        [result.peak_pfe_time], [result.max_pfe],
        marker="o", markersize=8, color=COLOR_PFE,
        markeredgecolor=COLOR_SURFACE, markeredgewidth=2.0, zorder=5,
    )
    axes.annotate(
        f"peak PFE {result.max_pfe:,.3f} @ t={result.peak_pfe_time:.2f}y",
        xy=(result.peak_pfe_time, result.max_pfe),
        xytext=(0, 12), textcoords="offset points",
        color=COLOR_INK_SECONDARY, fontsize=9, ha="center",
    )

    # --- chrome: recessive grid and axes -------------------------------
    axes.grid(axis="y", color=COLOR_GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(COLOR_AXIS)
        axes.spines[side].set_linewidth(1.0)
    axes.tick_params(colors=COLOR_INK_MUTED, labelsize=9, length=0)

    axes.set_xlim(float(result.times[0]), x_end + 7.0 * x_pad)
    axes.set_ylim(bottom=0.0)
    axes.set_xlabel("Time (years)", color=COLOR_INK_SECONDARY, fontsize=10, labelpad=8)
    axes.set_ylabel("Exposure (currency units)", color=COLOR_INK_SECONDARY, fontsize=10, labelpad=8)

    # --- titles --------------------------------------------------------
    axes.set_title(
        "Counterparty Exposure Profile",
        color=COLOR_INK_PRIMARY, fontsize=15, fontweight="bold", loc="left", pad=26,
    )
    subtitle = (
        f"Forward, notional {result.notional:g}, struck {result.strike:g}, {result.maturity:g}y  |  "
        f"S0 {result.spot:g}   sigma {result.vol:.2%}   r {result.rate:.2%}   "
        f"lambda {result.hazard:.2%}   R {result.recovery:.0%}"
    )
    axes.annotate(
        subtitle, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 12),
        textcoords="offset points", color=COLOR_INK_SECONDARY, fontsize=9.5, ha="left", va="bottom",
    )

    legend = axes.legend(
        loc="upper left", frameon=False, fontsize=9.5, handlelength=1.8, borderpad=0.0,
    )
    for text in legend.get_texts():
        text.set_color(COLOR_INK_SECONDARY)

    # --- footnote: the headline numbers, so the PNG stands alone -------
    footnote = (
        f"CVA {result.cva:,.6f}    EPE {result.epe:,.4f}    "
        f"dCVA/dS0 {result.greeks['delta_dCVA_dS0']:+.6f}    "
        f"dCVA/dsigma {result.greeks['vega_dCVA_dsigma']:+.6f}    "
        f"dCVA/dlambda {result.greeks['credit_dCVA_dlambda']:+.6f}    "
        f"|  {result.n_paths:,} paths x {result.n_steps} steps, seed {result.seed}"
    )
    figure.text(0.012, 0.017, footnote, color=COLOR_INK_MUTED, fontsize=8.5, ha="left")

    figure.tight_layout(rect=(0.0, 0.045, 1.0, 1.0))
    figure.savefig(destination, facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)


# ==========================================================================
# Console report
# ==========================================================================
def _rule(width: int = 74, char: str = "-") -> str:
    return char * width


def print_report(result: SandboxResult, plot_path: Path | None) -> None:
    """Print a structured, human-readable summary of the run.

    Args:
        result: A completed :class:`SandboxResult`.
        plot_path: Where the chart was written, or ``None`` if plotting was
            skipped.
    """
    print()
    print(_rule(74, "="))
    print("  XVA MANUAL SANDBOX  --  Phase 2 (exposure profiling + unilateral CVA)")
    print(_rule(74, "="))

    print("\n  MARKET INPUTS")
    print(_rule())
    print(f"    Spot                S0        {result.spot:>16,.6f}")
    print(f"    Volatility          sigma     {result.vol:>16,.6f}   ({result.vol:.2%})")
    print(f"    Risk-free rate      r         {result.rate:>16,.6f}   ({result.rate:.2%})")

    print("\n  CREDIT INPUTS")
    print(_rule())
    print(f"    Hazard rate         lambda    {result.hazard:>16,.6f}   ({result.hazard:.2%})")
    print(f"    Recovery rate       R         {result.recovery:>16,.6f}   ({result.recovery:.0%})")
    print(f"    Loss given default  1-R       {1.0 - result.recovery:>16,.6f}")
    # The credit triangle: the CDS spread this hazard rate is consistent with.
    print(
        f"    Implied CDS spread  s~L(1-R)  "
        f"{result.hazard * (1.0 - result.recovery):>16,.6f}   "
        f"({result.hazard * (1.0 - result.recovery) * 1e4:.1f} bp)"
    )

    print("\n  CONTRACT")
    print(_rule())
    print(f"    Instrument                    {'equity forward (linear)':>16}")
    print(f"    Notional            N         {result.notional:>16,.6f}")
    print(f"    Strike              K         {result.strike:>16,.6f}   (struck at initial spot)")
    print(f"    Maturity            T         {result.maturity:>16,.6f}   years")
    print(f"    Risk-free value     V(0)      {result.risk_free_value:>16,.6f}")

    print("\n  SIMULATION")
    print(_rule())
    print(f"    Paths               M         {result.n_paths:>16,}")
    print(f"    Time steps          N         {result.n_steps:>16,}")
    print(f"    Antithetic                    {str(result.antithetic):>16}")
    print(f"    Seed                          {result.seed:>16,}")
    print(f"    Device                        {result.device:>16}")

    print("\n  EXPOSURE PROFILE")
    print(_rule())
    print(f"    Peak EE                       {float(np.max(result.ee)):>16,.6f}")
    print(f"    EPE (time-avg EE)             {result.epe:>16,.6f}")
    print(
        f"    Peak PFE @ {result.confidence:.0%}              "
        f"{result.max_pfe:>16,.6f}   at t = {result.peak_pfe_time:.4f}y"
    )
    print(f"    Peak ENE                      {float(np.max(result.ene)):>16,.6f}")

    print("\n  VALUATION ADJUSTMENT")
    print(_rule())
    print(f"    Unilateral CVA                {result.cva:>16,.8f}")
    if abs(result.risk_free_value) > 1e-12:
        share = result.cva / abs(result.risk_free_value)
        print(f"    CVA / |V(0)|                  {share:>16,.6f}   ({share:.3%})")
    print(f"    Credit-adjusted value         {result.risk_free_value - result.cva:>16,.8f}")

    print("\n  AAD SENSITIVITIES  (one backward pass, all four at once)")
    print(_rule())
    print(f"    Delta       dCVA/dS0          {result.greeks['delta_dCVA_dS0']:>16,.8f}")
    print(f"    Vega        dCVA/dsigma       {result.greeks['vega_dCVA_dsigma']:>16,.8f}")
    print(f"    Credit01    dCVA/dlambda      {result.greeks['credit_dCVA_dlambda']:>16,.8f}")
    print(f"    Rho         dCVA/dr           {result.greeks['rho_dCVA_dr']:>16,.8f}")
    print()
    # Scaled to the bumps a trader actually quotes, which is how these numbers
    # get sanity-checked on a desk.
    print(f"    per +1.00 spot                {result.greeks['delta_dCVA_dS0'] * 1.0:>16,.8f}")
    print(f"    per +1 vol point (0.01)       {result.greeks['vega_dCVA_dsigma'] * 0.01:>16,.8f}")
    print(f"    per +1bp hazard (0.0001)      {result.greeks['credit_dCVA_dlambda'] * 1e-4:>16,.8f}")
    print(f"    per +1bp rate (0.0001)        {result.greeks['rho_dCVA_dr'] * 1e-4:>16,.8f}")

    if plot_path is not None:
        print("\n  CHART")
        print(_rule())
        print(f"    Written to  {plot_path}")

    print()
    print(_rule(74, "="))
    print()


def run_verification(args: argparse.Namespace, result: SandboxResult) -> None:
    """Cross-check the AAD Greeks against bump-and-revalue on the same draw.

    This rebuilds the CVA graph through the library's own
    :func:`~src.xva.cva.make_cva_valuation_fn` closure and runs both
    differentiation methods over it. Because the closure captures the identical
    Brownian sample, the two methods differentiate the *same* Monte-Carlo
    realisation -- so any gap measures the differentiation scheme, not
    resampling noise.

    Args:
        args: Parsed CLI namespace (used to rebuild an identical simulator).
        result: The completed run, used only for the strike and notional.
    """
    device = torch.device("cpu") if args.cpu else resolve_device()
    simulator = GBMSimulator(
        maturity=args.maturity,
        n_steps=args.steps,
        device=device,
        dtype=DTYPE,
        antithetic=args.antithetic,
    )
    dW = simulator.draw_increments(args.paths, seed=args.seed)
    legs = [SwapLeg(notional=args.notional, strike=result.strike, maturity=args.maturity)]

    cva_fn = make_cva_valuation_fn(
        simulator, dW, legs, recovery_rate=result.recovery, rate=args.rate
    )
    params = {
        "s0": args.spot,
        "sigma": args.vol,
        "hazard_rate": args.hazard,
        "rate": args.rate,
    }

    aad = cva_aad_greeks(cva_fn, params)
    fd = cva_bump_and_revalue_greeks(cva_fn, params, scheme="central")

    print("  VERIFICATION  --  AAD vs bump-and-revalue (common random numbers)")
    print(_rule())
    print(format_comparison(compare_greeks(fd, aad)))
    print()
    print(
        "    Note: finite differences carry an O(h) kink bias here, because EE\n"
        "    is a mean of max(V, 0) and a bump flips the sign of V on O(M*h)\n"
        "    paths. AAD returns the unbiased pathwise derivative, so AAD is the\n"
        "    more accurate method as well as the faster one."
    )
    print()
    print(_rule(74, "="))
    print()


def main() -> int:
    """Entry point.

    Returns:
        Process exit status: ``0`` on success.
    """
    args = build_parser().parse_args()
    validate_args(args)

    result = run_pipeline(args)

    plot_path: Path | None = None
    if not args.no_plot:
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_exposure_profile(result, plot_path)

    print_report(result, plot_path)

    if args.verify:
        run_verification(args, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
