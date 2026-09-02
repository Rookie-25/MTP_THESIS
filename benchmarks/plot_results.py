r"""Publication-ready figures for the XVA engine benchmarks and exposure profiles.

Produces three figures:

1. **Execution time vs M** -- log-log, one line per backend.
2. **Peak VRAM vs M** -- log-log, with the device's capacity drawn as a
   reference line so the OOM cliff has a visible cause.
3. **Exposure profiles** -- EE and PFE, with and without a CSA.

Where the data comes from
=========================
Figures 1 and 2 are parsed from the Markdown that ``bench_all_phases.py``
writes (``--markdown results.md``); nothing is simulated for them, so if that
file does not exist the figures are skipped with the command that produces it.
**No placeholder or illustrative numbers are ever plotted** -- a benchmark
figure that silently shows invented data is worse than a missing figure.

Figure 3 is computed live on the CPU from ``src.xva``: it needs no GPU and no
benchmark file, because it is a property of the model rather than of the
hardware.

Design decisions
================
The palette is not a matter of taste. Slots come from the project's
data-visualisation reference and were checked with its validator:

* **Backends** use blue / orange / aqua / violet. The obvious fourth choice,
  yellow, fails the all-pairs normal-vision gate against orange
  (:math:`\Delta E` 13.7, below the floor of 15) -- readers with ordinary colour
  vision cannot reliably separate them. Blue/orange/aqua/violet passes both the
  adjacent and all-pairs gates (worst all-pairs normal :math:`\Delta E` 16.3,
  CVD 9.2).
* **Exposure metrics** use blue for EE and orange for PFE -- the widest
  separation available and both above 3:1 contrast on white.
* Aqua sits at 2.82:1 on white, below the 3:1 bar, so every series also carries
  a **direct end-of-line label**. That is the documented relief for a
  low-contrast slot, and it doubles as the redundant encoding that keeps the
  figures readable in greyscale print.
* Colour identifies the *entity*, so a backend keeps its hue across both
  figures. Line style and marker vary with it, so identity never rests on
  colour alone.
* Figure 3 encodes the **metric** in colour and the **CSA** in line style
  rather than giving four series four hues: the comparison the reader needs is
  within each metric, and composite encoding makes that pairing visible.

There is deliberately **no dual-axis figure**. Time and memory have unrelated
units, so they get one axis each; overlaying them on twin y-scales would let
the crossing point be set by axis choice rather than by data.

Usage
=====
    python benchmarks/plot_results.py
    python benchmarks/plot_results.py --results results.md --outdir figures
    python benchmarks/plot_results.py --format pdf --dpi 300
    python benchmarks/plot_results.py --only exposure
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless: these are files, not windows
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BYTES_PER_MIB = 1024.0**2
BYTES_PER_GIB = 1024.0**3

# ---- palette (validated; see the module docstring) ----------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
GRID = "#e4e3df"
SURFACE = "#ffffff"

#: Backend hue order. Colour follows the entity, so these are keyed by name and
#: stay fixed across every figure; a backend missing from one run must not
#: repaint the others.
BACKEND_STYLE: Dict[str, Tuple[str, str, str]] = {
    "PyTorch baseline": ("#2a78d6", "-", "o"),
    "Phase 3 Triton": ("#eb6834", "--", "s"),
    "Phase 4 Philox": ("#1baf7a", "-.", "^"),
    "Phase 5 fused": ("#4a3aa7", ":", "D"),
}
FALLBACK_STYLE = (INK_SECONDARY, "-", "v")

EE_COLOR = "#2a78d6"
PFE_COLOR = "#eb6834"

LINE_WIDTH = 2.0
MARKER_SIZE = 8.0
MARKER_RING = 1.5  # surface ring, so overlapping markers stay separable

#: Multiplicative right-hand headroom on the log x-axis, reserving room for the
#: end-of-line labels. Without it they are placed in offset points from the last
#: data point and clip at the spine.
X_HEADROOM = 3.5


def _end_label(axes, x, y, name: str, colour: str, oom: bool) -> None:
    """Label a series at its right-hand end, noting an OOM on a second line.

    One text object per series end: an OOM drawn as a separate offset
    annotation lands on top of the name, which is what the first render did.

    Args:
        axes: Target axes.
        x: Last x value.
        y: Last y value.
        name: Series name.
        colour: Series colour, used only for the OOM word.
        oom: Whether the series ran out of memory beyond this point.
    """
    axes.annotate(
        name, xy=(x, y), xytext=(8, 0), textcoords="offset points",
        va="center", ha="left", fontsize=8, color=INK_SECONDARY, zorder=4,
    )
    if oom:
        axes.annotate(
            "OOM beyond", xy=(x, y), xytext=(8, -9),
            textcoords="offset points", va="center", ha="left", fontsize=7,
            color=colour, fontweight="bold", zorder=4,
        )


def apply_style() -> None:
    """Set publication defaults: recessive frame, no chartjunk."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "figure.dpi": 120,
    })


# ==========================================================================
# Parsing the benchmark Markdown
# ==========================================================================
_OOM = re.compile(r"\*\*OOM\*\*", re.I)
_BYTES = re.compile(r"^([\d,.]+)\s*(MiB|GiB|KiB|B)$", re.I)
_GPU_VRAM = re.compile(r"\(([\d,.]+)\s*GiB\)")


def parse_number(cell: str) -> Optional[float]:
    """Parse a millisecond cell, returning ``None`` for OOM or a blank.

    Args:
        cell: Raw table cell.

    Returns:
        The value, or ``None`` where the backend did not complete. ``None`` is
        deliberately distinct from zero: a run that could not start is not a
        run that took no time, and plotting it as zero would invent a data
        point at the exact place the finding lives.
    """
    text = cell.strip()
    if not text or text in {"-", "not run"} or _OOM.search(text):
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def parse_bytes(cell: str) -> Optional[int]:
    """Parse a ``format_bytes`` cell back to a byte count.

    Args:
        cell: Raw table cell, e.g. ``"4.3 MiB"`` or ``"14.15 GiB"``.

    Returns:
        Bytes, or ``None`` for OOM, a blank, or an unrecognised unit.
    """
    text = cell.strip()
    if not text or text in {"-", "not run"} or _OOM.search(text):
        return None
    match = _BYTES.match(text)
    if match is None:
        return None
    value = float(match.group(1).replace(",", ""))
    scale = {
        "b": 1.0, "kib": 1024.0, "mib": BYTES_PER_MIB, "gib": BYTES_PER_GIB,
    }[match.group(2).lower()]
    return int(value * scale)


def parse_markdown_table(lines: Sequence[str]) -> Tuple[List[str], List[List[str]]]:
    """Parse the first Markdown table found in ``lines``.

    Args:
        lines: Lines to scan, starting at or before the table.

    Returns:
        ``(headers, rows)`` with cells stripped. Empty if no table is present.
    """
    table: List[List[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if table:
                break  # table ended
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue  # separator row
        table.append(cells)
    if not table:
        return [], []
    return table[0], table[1:]


def _section(lines: Sequence[str], heading: str) -> List[str]:
    """Return the lines following a Markdown heading, up to the next one."""
    lowered = heading.lower()
    for index, line in enumerate(lines):
        if line.strip().lower().startswith(lowered):
            rest = []
            for follow in lines[index + 1:]:
                if follow.strip().startswith("## "):
                    break
                rest.append(follow)
            return rest
    return []


@dataclass
class BenchmarkResults:
    """Timings and peak memory parsed from a benchmark report.

    Attributes:
        path: Source file.
        gpu: Device description from the Environment table, if present.
        total_vram_bytes: Device capacity, parsed out of the GPU string.
        n_steps: Time steps, for the caption.
        times_ms: ``{backend: {n_paths: milliseconds}}``, successes only.
        peak_bytes: ``{backend: {n_paths: bytes}}``, successes only.
        oom_paths: ``{backend: [n_paths that ran out of memory]}``.
    """

    path: Path
    gpu: Optional[str] = None
    total_vram_bytes: Optional[int] = None
    n_steps: Optional[int] = None
    times_ms: Dict[str, Dict[int, float]] = field(default_factory=dict)
    peak_bytes: Dict[str, Dict[int, int]] = field(default_factory=dict)
    oom_paths: Dict[str, List[int]] = field(default_factory=dict)

    @property
    def backends(self) -> List[str]:
        """Backend names, ordered by the palette where possible."""
        seen = list(self.times_ms) or list(self.peak_bytes)
        known = [name for name in BACKEND_STYLE if name in seen]
        return known + [name for name in seen if name not in BACKEND_STYLE]

    @property
    def has_timings(self) -> bool:
        return any(self.times_ms.values())

    @property
    def has_memory(self) -> bool:
        return any(self.peak_bytes.values())


def load_results(path: Path) -> BenchmarkResults:
    """Parse a ``bench_all_phases`` / ``bench_phase6`` Markdown report.

    Tolerant by design: an absent section yields no series rather than an
    error, so a partial run still plots what it measured.

    Args:
        path: Markdown file.

    Returns:
        A populated :class:`BenchmarkResults`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    results = BenchmarkResults(path=path)

    # ---- environment -------------------------------------------------
    for row in parse_markdown_table(_section(lines, "## environment"))[1]:
        if len(row) < 2:
            continue
        key, value = row[0].lower(), row[1]
        if key == "gpu":
            results.gpu = value
            match = _GPU_VRAM.search(value)
            if match:
                results.total_vram_bytes = int(
                    float(match.group(1).replace(",", "")) * BYTES_PER_GIB
                )
        elif key.startswith("time steps"):
            try:
                results.n_steps = int(value.replace(",", ""))
            except ValueError:
                pass

    def ingest(heading: str, convert, store: Dict[str, Dict], record_oom: bool):
        headers, rows = parse_markdown_table(_section(lines, heading))
        if not headers or headers[0].strip().upper() != "M":
            return
        backends = headers[1:]
        for name in backends:
            store.setdefault(name, {})
            if record_oom:
                results.oom_paths.setdefault(name, [])
        for row in rows:
            if len(row) < 2:
                continue
            try:
                n_paths = int(row[0].replace(",", ""))
            except ValueError:
                continue
            for name, cell in zip(backends, row[1:]):
                value = convert(cell)
                if value is None:
                    if record_oom and _OOM.search(cell):
                        results.oom_paths[name].append(n_paths)
                else:
                    store[name][n_paths] = value

    ingest("## execution time", parse_number, results.times_ms, True)
    ingest("## peak vram", parse_bytes, results.peak_bytes, False)
    # bench_phase6 splits the timings in two; fold the forward in if present.
    if not results.has_timings:
        ingest("## forward time", parse_number, results.times_ms, True)
    return results


# ==========================================================================
# Figure 1: execution time
# ==========================================================================
def plot_execution_time(
    results: BenchmarkResults, output: Path, *, dpi: int
) -> Optional[Path]:
    """Execution time against path count, log-log.

    Log-log because both axes span orders of magnitude and the quantity of
    interest is the *exponent*: linear scaling is a straight line of slope 1,
    so a departure from linearity is visible as a change in slope rather than
    something the reader has to infer from curvature.

    Args:
        results: Parsed benchmark data.
        output: Path stem (extension supplied by the caller).
        dpi: Raster resolution.

    Returns:
        The written path, or ``None`` if there was nothing to plot.
    """
    if not results.has_timings:
        return None

    figure, axes = plt.subplots(figsize=(7.0, 4.6))
    axes.set_axisbelow(True)
    axes.grid(True, which="major", axis="both")
    axes.grid(True, which="minor", axis="both", alpha=0.4)

    for name in results.backends:
        series = results.times_ms.get(name) or {}
        if not series:
            continue
        colour, style, marker = BACKEND_STYLE.get(name, FALLBACK_STYLE)
        paths = sorted(series)
        values = [series[m] for m in paths]
        axes.plot(
            paths, values, color=colour, linestyle=style, marker=marker,
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
            markeredgecolor=SURFACE, markeredgewidth=MARKER_RING,
            label=name, zorder=3,
        )
        # Direct label at the line end: the documented relief for the
        # low-contrast slot, and it survives greyscale printing.
        _end_label(
            axes, paths[-1], values[-1], name, colour,
            any(oom > paths[-1] for oom in results.oom_paths.get(name, [])),
        )

    axes.set_xscale("log")
    axes.set_yscale("log")
    _reserve_label_room(axes, results)
    axes.set_xlabel("Monte-Carlo paths $M$")
    axes.set_ylabel("Execution time (ms)")
    axes.set_title("Execution time scaling", color=INK_PRIMARY, loc="left")
    _legend_below(axes)
    _caption(figure, results, "A line stops where that backend ran out of memory.")

    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


# ==========================================================================
# Figure 2: peak VRAM
# ==========================================================================
def plot_peak_vram(
    results: BenchmarkResults, output: Path, *, dpi: int
) -> Optional[Path]:
    """Peak allocation against path count, with the device capacity shown.

    The capacity line is the point of the figure: without it a reader sees
    lines stopping for no reason, and with it the OOM boundary is visibly the
    intersection of an ``O(M*N)`` line with a fixed ceiling.

    Args:
        results: Parsed benchmark data.
        output: Output path.
        dpi: Raster resolution.

    Returns:
        The written path, or ``None`` if there was nothing to plot.
    """
    if not results.has_memory:
        return None

    figure, axes = plt.subplots(figsize=(7.0, 4.6))
    axes.set_axisbelow(True)
    axes.grid(True, which="major", axis="both")
    axes.grid(True, which="minor", axis="both", alpha=0.4)

    for name in results.backends:
        series = results.peak_bytes.get(name) or {}
        if not series:
            continue
        colour, style, marker = BACKEND_STYLE.get(name, FALLBACK_STYLE)
        paths = sorted(series)
        values = [series[m] / BYTES_PER_MIB for m in paths]
        axes.plot(
            paths, values, color=colour, linestyle=style, marker=marker,
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
            markeredgecolor=SURFACE, markeredgewidth=MARKER_RING,
            label=name, zorder=3,
        )
        _end_label(axes, paths[-1], values[-1], name, colour, False)

    axes.set_xscale("log")
    axes.set_yscale("log")
    _reserve_label_room(axes, results)

    if results.total_vram_bytes:
        # Right-aligned at the end of its own line, clear of both the series
        # labels and the legend -- at upper-left it sat underneath both.
        capacity = results.total_vram_bytes / BYTES_PER_MIB
        axes.axhline(
            capacity, color=INK_MUTED, linestyle=(0, (6, 3)), linewidth=1.2,
            zorder=2,
        )
        axes.annotate(
            f"device capacity {capacity / 1024.0:,.1f} GiB",
            xy=(0.99, capacity), xycoords=("axes fraction", "data"),
            xytext=(0, 6), textcoords="offset points", ha="right",
            fontsize=8, color=INK_SECONDARY, zorder=4,
        )

    axes.set_xlabel("Monte-Carlo paths $M$")
    axes.set_ylabel("Peak allocated memory (MiB)")
    axes.set_title("Peak device memory scaling", color=INK_PRIMARY, loc="left")
    _legend_below(axes)
    _caption(
        figure, results,
        "A flat line is memory independent of $M$; a sloped line is $O(M)$.",
    )

    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


def _reserve_label_room(axes, results: BenchmarkResults) -> None:
    """Extend the x limit so the end-of-line labels sit inside the figure.

    Args:
        axes: Target axes, already on a log x-scale.
        results: Used for the largest measured path count.
    """
    measured = [
        n for series in (*results.times_ms.values(), *results.peak_bytes.values())
        for n in series
    ]
    if measured:
        axes.set_xlim(right=max(measured) * X_HEADROOM)


def _legend_below(axes) -> None:
    """Place the legend under the axes.

    Inside the axes it collided with the capacity annotation on the memory
    figure and with the series labels on the timing one. Below the plot it
    cannot collide with anything, which is also the usual arrangement for a
    figure destined for a paper.

    Args:
        axes: Target axes.
    """
    axes.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=4,
        columnspacing=1.4, handlelength=2.4, borderaxespad=0.0,
    )


def _caption(figure, results: BenchmarkResults, note: str) -> None:
    """Put provenance under the figure, so a plot is never anonymous."""
    parts = [note]
    if results.gpu:
        parts.append(results.gpu)
    if results.n_steps:
        parts.append(f"N = {results.n_steps:,} steps")
    parts.append(f"source: {results.path.name}")
    figure.text(
        0.0, -0.20, "   ".join(parts), fontsize=7.5, color=INK_MUTED,
        ha="left", va="top", transform=figure.transFigure,
    )


# ==========================================================================
# Figure 3: exposure profiles
# ==========================================================================
@dataclass
class ExposureCurves:
    """EE and PFE with and without a CSA, plus the terms that produced them."""

    times: np.ndarray
    ee: np.ndarray
    pfe: np.ndarray
    ee_collateralized: np.ndarray
    pfe_collateralized: np.ndarray
    confidence_level: float
    n_paths: int
    threshold: float
    minimum_transfer_amount: float
    margin_period_of_risk: float


def compute_exposure_curves(
    *,
    n_paths: int = 20_000,
    n_steps: int = 96,
    maturity: float = 5.0,
    spot: float = 100.0,
    volatility: float = 0.25,
    rate: float = 0.03,
    strike: float = 100.0,
    threshold: float = 5.0,
    minimum_transfer_amount: float = 1.0,
    margin_period_of_risk: float = 10.0 / 252.0,
    confidence_level: float = 0.95,
    seed: int = 20260903,
) -> ExposureCurves:
    """Simulate one netting set and profile it with and without a CSA.

    Runs on the CPU in float64: this figure illustrates a model property, so
    reproducibility matters more than speed, and a fixed seed makes the two
    profiles differ only by the CSA rather than by sampling noise.

    Args:
        n_paths: Monte-Carlo paths.
        n_steps: Time steps.
        maturity: Horizon in years.
        spot: :math:`S_0`.
        volatility: Flat volatility.
        rate: Risk-free rate, used for drift and discounting.
        strike: Swap strike.
        threshold: CSA unsecured threshold.
        minimum_transfer_amount: CSA MTA.
        margin_period_of_risk: MPOR in years.
        confidence_level: PFE quantile.
        seed: RNG seed.

    Returns:
        A populated :class:`ExposureCurves`.
    """
    import torch

    from src.models.gbm import GBMSimulator
    from src.pricer.options import SwapLeg, portfolio_swap_mtm
    from src.xva.exposure import (
        CSATerms,
        compute_collateralized_exposure_profile,
        compute_exposure_profile,
    )

    torch.manual_seed(seed)
    simulator = GBMSimulator(maturity=maturity, n_steps=n_steps, dtype=torch.float64)
    times = simulator.time_grid()

    with torch.no_grad():
        paths = simulator.simulate(spot, rate, volatility, n_paths=n_paths)
        mtm = portfolio_swap_mtm(
            paths, times,
            [SwapLeg(notional=1.0, strike=strike, maturity=maturity)],
            rate,
        )
        uncollateralized = compute_exposure_profile(
            mtm, times, confidence_level=confidence_level
        )
        terms = CSATerms(
            threshold=threshold,
            minimum_transfer_amount=minimum_transfer_amount,
            margin_period_of_risk=margin_period_of_risk,
        )
        collateralized = compute_collateralized_exposure_profile(
            mtm, times, terms, confidence_level=confidence_level
        )

    return ExposureCurves(
        times=times.numpy(),
        ee=uncollateralized.ee.numpy(),
        pfe=uncollateralized.pfe.numpy(),
        ee_collateralized=collateralized.ee.numpy(),
        pfe_collateralized=collateralized.pfe.numpy(),
        confidence_level=confidence_level,
        n_paths=n_paths,
        threshold=threshold,
        minimum_transfer_amount=minimum_transfer_amount,
        margin_period_of_risk=margin_period_of_risk,
    )


def plot_exposure_profiles(
    curves: ExposureCurves, output: Path, *, dpi: int
) -> Path:
    """EE and PFE, with and without collateral, on one axis.

    Colour carries the *metric* and line style the *CSA*, rather than giving
    four series four hues. The comparison a reader needs is within each metric
    -- how much the CSA removes -- and pairing them by colour makes that the
    thing the eye does first. The shaded bands are the reduction itself, which
    is the quantity a collateral decision turns on.

    Args:
        curves: Profiles to draw.
        output: Output path.
        dpi: Raster resolution.

    Returns:
        The written path.
    """
    figure, axes = plt.subplots(figsize=(7.0, 4.6))
    axes.set_axisbelow(True)
    axes.grid(True, axis="y")

    quantile = f"{curves.confidence_level:.0%}"

    axes.fill_between(
        curves.times, curves.pfe_collateralized, curves.pfe,
        color=PFE_COLOR, alpha=0.10, linewidth=0, zorder=1,
    )
    axes.fill_between(
        curves.times, curves.ee_collateralized, curves.ee,
        color=EE_COLOR, alpha=0.14, linewidth=0, zorder=1,
    )

    horizon = float(curves.times[-1])
    for values, colour, style, label in (
        (curves.pfe, PFE_COLOR, "-", f"PFE {quantile}, uncollateralised"),
        (curves.pfe_collateralized, PFE_COLOR, "--", f"PFE {quantile}, with CSA"),
        (curves.ee, EE_COLOR, "-", "EE, uncollateralised"),
        (curves.ee_collateralized, EE_COLOR, "--", "EE, with CSA"),
    ):
        axes.plot(
            curves.times, values, color=colour, linestyle=style,
            linewidth=LINE_WIDTH, label=label, zorder=3,
        )
        # Terminal value as a direct label. This replaces a "PFE peak"
        # annotation: the profile rises monotonically to maturity, so the peak
        # is always the last point and naming it restated the axis.
        axes.annotate(
            f"{values[-1]:,.1f}", xy=(horizon, values[-1]), xytext=(7, 0),
            textcoords="offset points", va="center", ha="left", fontsize=8,
            color=colour, zorder=4,
        )

    reduction = 1.0 - curves.ee_collateralized.sum() / curves.ee.sum()
    peak_reduction = 1.0 - float(curves.pfe_collateralized.max()) / float(
        curves.pfe.max()
    )
    axes.annotate(
        f"CSA removes {reduction:.0%} of aggregate EE\n"
        f"and {peak_reduction:.0%} of peak PFE",
        xy=(0.02, 0.96), xycoords="axes fraction", ha="left", va="top",
        fontsize=8.5, color=INK_PRIMARY, linespacing=1.5,
    )

    # Headroom on the right for the terminal labels.
    axes.set_xlim(curves.times[0], horizon * 1.10)
    axes.set_ylim(bottom=0.0)
    axes.set_xlabel("Time (years)")
    axes.set_ylabel("Exposure")
    axes.set_title(
        "Counterparty exposure profile, with and without a CSA",
        color=INK_PRIMARY, loc="left",
    )
    axes.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2,
        columnspacing=1.6, handlelength=2.4, borderaxespad=0.0,
    )

    figure.text(
        0.0, -0.215,
        "Colour = metric, dashes = CSA applied; shaded bands are the reduction.   "
        f"{curves.n_paths:,} paths   "
        f"threshold {curves.threshold:g}, MTA {curves.minimum_transfer_amount:g}, "
        f"MPOR {curves.margin_period_of_risk * 252:.0f}bd",
        fontsize=7.5, color=INK_MUTED, ha="left", va="top",
        transform=figure.transFigure,
    )

    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output


# ==========================================================================
# Table view (accessibility fallback)
# ==========================================================================
def write_table_view(
    results: Optional[BenchmarkResults],
    curves: Optional[ExposureCurves],
    output: Path,
) -> Path:
    """Write the plotted numbers as CSV.

    A figure alone is not accessible: a reader using a screen reader, or
    checking a number, needs the values. Emitting them alongside costs nothing
    and removes the need to read data off a chart.

    Args:
        results: Benchmark data, if any.
        curves: Exposure curves, if any.
        output: CSV path.

    Returns:
        The written path.
    """
    rows: List[str] = ["figure,series,x,y,unit"]
    if results is not None:
        for name in results.backends:
            for paths, value in sorted((results.times_ms.get(name) or {}).items()):
                rows.append(f"execution_time,{name},{paths},{value:.6f},ms")
            for paths, value in sorted((results.peak_bytes.get(name) or {}).items()):
                rows.append(f"peak_vram,{name},{paths},{value},bytes")
            for paths in results.oom_paths.get(name, []):
                rows.append(f"execution_time,{name},{paths},,OOM")
    if curves is not None:
        for label, values in (
            ("ee_uncollateralized", curves.ee),
            ("ee_collateralized", curves.ee_collateralized),
            ("pfe_uncollateralized", curves.pfe),
            ("pfe_collateralized", curves.pfe_collateralized),
        ):
            for time, value in zip(curves.times, values):
                rows.append(f"exposure,{label},{time:.6f},{value:.8f},exposure")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return output


# ==========================================================================
# Entry point
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI."""
    parser = argparse.ArgumentParser(
        prog="plot_results.py",
        description=(
            "Generate publication-ready figures: execution time vs M, peak "
            "VRAM vs M, and exposure profiles with and without collateral."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results", type=Path, default=_REPO_ROOT / "results.md",
        help="Markdown report from bench_all_phases.py (--markdown).",
    )
    parser.add_argument(
        "--outdir", type=Path, default=_REPO_ROOT / "figures",
        help="Directory for the figures.",
    )
    parser.add_argument(
        "--format", choices=["png", "pdf", "both"], default="both",
        help="Raster, vector, or both. PDF is the one to submit.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Raster DPI.")
    parser.add_argument(
        "--only", choices=["time", "vram", "exposure", "all"], default="all",
        help="Restrict which figures are produced.",
    )
    parser.add_argument(
        "--paths", type=int, default=20_000,
        help="Monte-Carlo paths for the exposure figure.",
    )
    parser.add_argument(
        "--threshold", type=float, default=5.0, help="CSA threshold."
    )
    parser.add_argument(
        "--mta", type=float, default=1.0, help="CSA minimum transfer amount."
    )
    parser.add_argument(
        "--mpor-days", type=float, default=10.0,
        help="Margin period of risk in business days.",
    )
    return parser


def main() -> int:
    """Generate the figures.

    Returns:
        ``0`` on success, ``1`` if nothing could be produced.
    """
    args = build_parser().parse_args()
    apply_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    suffixes = {"png": [".png"], "pdf": [".pdf"], "both": [".png", ".pdf"]}[
        args.format
    ]

    wanted = (
        {"time", "vram", "exposure"} if args.only == "all" else {args.only}
    )
    written: List[Path] = []

    # ---- benchmark figures -------------------------------------------
    results: Optional[BenchmarkResults] = None
    if wanted & {"time", "vram"}:
        if args.results.exists():
            results = load_results(args.results)
            print(f"  Parsed {args.results}")
            if results.gpu:
                print(f"    device   : {results.gpu}")
            for name in results.backends:
                timings = len(results.times_ms.get(name) or {})
                memory = len(results.peak_bytes.get(name) or {})
                ooms = len(results.oom_paths.get(name, []))
                print(f"    {name:<20} {timings} timings, {memory} memory"
                      f"{f', {ooms} OOM' if ooms else ''}")
        else:
            print(
                f"\n  No benchmark report at {args.results}.\n"
                "  Figures 1 and 2 need measured data and nothing is invented "
                "to stand in for it.\n"
                "  Generate it on a GPU machine with:\n\n"
                "      python benchmarks/bench_all_phases.py --find-oom "
                f"--markdown {args.results.name}\n"
            )

    if results is not None:
        for kind, plotter in (
            ("time", plot_execution_time), ("vram", plot_peak_vram)
        ):
            if kind not in wanted:
                continue
            stem = {"time": "execution_time_vs_paths",
                    "vram": "peak_vram_vs_paths"}[kind]
            for suffix in suffixes:
                path = plotter(results, args.outdir / f"{stem}{suffix}",
                               dpi=args.dpi)
                if path is not None:
                    written.append(path)
            if not any(p.stem == stem for p in written):
                print(f"  Skipped {stem}: the report has no matching table.")

    # ---- exposure figure ---------------------------------------------
    curves: Optional[ExposureCurves] = None
    if "exposure" in wanted:
        print(f"  Simulating {args.paths:,} paths for the exposure profile ...")
        curves = compute_exposure_curves(
            n_paths=args.paths,
            threshold=args.threshold,
            minimum_transfer_amount=args.mta,
            margin_period_of_risk=args.mpor_days / 252.0,
        )
        for suffix in suffixes:
            written.append(
                plot_exposure_profiles(
                    curves, args.outdir / f"exposure_profiles{suffix}",
                    dpi=args.dpi,
                )
            )

    if not written:
        print("  Nothing produced.")
        return 1

    written.append(write_table_view(results, curves, args.outdir / "figure_data.csv"))
    print()
    for path in written:
        print(f"  wrote {path.relative_to(_REPO_ROOT) if _REPO_ROOT in path.parents else path}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
