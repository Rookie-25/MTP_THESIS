r"""Cross-phase benchmark: PyTorch baseline vs the Phase 3/4/5 Triton kernels.

What is compared, and why it is a fair comparison
=================================================
All four backends compute **the same thing**: parameters in, expected-exposure
profile of shape ``(N+1,)`` out. That equality is what makes the table
meaningful, and it is not automatic -- three of the backends produce a path
matrix as an intermediate and must then be reduced, while Phase 5 never
materialises one. Timing only the path generation would flatter Phases 3 and 4
by omitting work Phase 5 does inside its kernel.

================  ==========================================================
backend           pipeline
================  ==========================================================
PyTorch baseline  ``simulate_gbm`` -> MtM -> clamp -> mean   (all in PyTorch)
Phase 3           draw ``dW`` -> ``triton_simulate_gbm`` -> MtM -> reduce
Phase 4           ``philox_simulate_gbm`` (in-kernel RNG) -> MtM -> reduce
Phase 5           ``fused_expected_exposure``  (nothing of size M in HBM)
================  ==========================================================

``dW`` generation is timed *inside* the Phase 3 measurement. Phase 4 and 5 must
produce their increments too -- they simply do it in registers -- so excluding
the draw would treat Phase 3's input as free.

The OOM boundary
================
The headline question is where pure PyTorch stops being viable. A fixed sweep
only *brackets* that point: on a 16 GiB T4 the baseline survives 1e6 and dies at
5e6, which locates the threshold to within a factor of five. ``--find-oom``
bisects the interval to report the largest path count the baseline actually
completes, which is the number worth quoting.

Bisection is safe here because a genuine allocator OOM is catchable and
recoverable (``empty_cache`` restores the device). It would *not* be safe if the
failure mode were an illegal memory access -- that poisons the CUDA context --
which is why :func:`_harness.is_oom` matches only real OOM and lets everything
else propagate.

Usage
=====
    python benchmarks/bench_all_phases.py
    python benchmarks/bench_all_phases.py --find-oom
    python benchmarks/bench_all_phases.py --backward --markdown results.md
    python benchmarks/bench_all_phases.py --paths 100000 1000000 --repeats 5
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks._harness import (  # noqa: E402
    BYTES_PER_GIB,
    VRAM_SAFETY_FRACTION,
    Measurement,
    format_bytes,
    free_vram_bytes,
    markdown_table,
    measure,
    reset_cuda,
)
from src.csrc.triton_cva_fusion import (  # noqa: E402
    DEFAULT_MAX_PROGRAMS,
    build_affine_coefficients,
    fused_expected_exposure,
    select_fused_block_sizes,
)
from src.csrc.triton_gbm import HAS_TRITON, is_available, triton_simulate_gbm  # noqa: E402
from src.csrc.triton_philox_gbm import philox_simulate_gbm  # noqa: E402
from src.models.gbm import simulate_gbm  # noqa: E402
from src.pricer.options import SwapLeg  # noqa: E402

S0 = 100.0
MU = 0.03
RATE = 0.03
SIGMA = 0.20
MATURITY = 1.0

#: The sweep requested for the thesis table.
DEFAULT_PATHS = [100_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000]

BASELINE = "PyTorch baseline"
PHASE3 = "Phase 3 (Triton + dW)"
PHASE4 = "Phase 4 (in-kernel Philox)"
PHASE5 = "Phase 5 (fused reduction)"
BACKEND_ORDER = (BASELINE, PHASE3, PHASE4, PHASE5)


def portfolio() -> List[SwapLeg]:
    """The netting set used throughout: mixed signs, staggered maturity.

    Must be non-empty. With no legs the affine coefficients are identically
    zero, every exposure is zero, and the benchmark would time a no-op while
    reporting excellent numbers.
    """
    return [
        SwapLeg(notional=1.0, strike=100.0, maturity=MATURITY),
        SwapLeg(notional=-0.4, strike=110.0, maturity=MATURITY),
        SwapLeg(notional=0.7, strike=95.0, maturity=0.5),
    ]


# ==========================================================================
# Memory prediction
# ==========================================================================
def predict_peak_bytes(
    backend: str,
    n_paths: int,
    n_steps: int,
    element_size: int,
    *,
    include_backward: bool,
    max_programs: int = DEFAULT_MAX_PROGRAMS,
) -> int:
    """Predict a backend's peak allocation, counting only O(M*N) tensors.

    At these scales the :math:`O(MN)` terms dominate everything else by orders
    of magnitude, so the smaller terms are ignored deliberately rather than
    modelled badly.

    Args:
        backend: One of the module-level backend names.
        n_paths: Monte-Carlo paths :math:`M`.
        n_steps: Time steps :math:`N`.
        element_size: Bytes per element.
        include_backward: Whether the adjoint is included.
        max_programs: Phase 5 launch-grid cap.

    Returns:
        Predicted peak bytes.

    Raises:
        ValueError: On an unknown backend name.
    """
    paths_matrix = n_paths * (n_steps + 1) * element_size
    increments = n_paths * n_steps * element_size

    if backend == BASELINE:
        # simulate_gbm materialises ~5 M*N tensors (scale, add, scan, concat,
        # exp), then the MtM surface and its clamped copy. PyTorch frees as it
        # goes, but several are live simultaneously; 4x is a deliberate
        # under-estimate of the true peak so the guard errs toward attempting.
        total = increments + 4 * paths_matrix
        if include_backward:
            total += 2 * paths_matrix
        return total

    if backend == PHASE3:
        total = increments + paths_matrix + paths_matrix  # dW, out, exposure
        if include_backward:
            total += paths_matrix + increments  # grad_out, grad_dW
        return total

    if backend == PHASE4:
        total = paths_matrix + paths_matrix  # out, exposure (no dW)
        if include_backward:
            total += paths_matrix
        return total

    if backend == PHASE5:
        block_m, _ = select_fused_block_sizes(n_steps, element_size)
        n_programs = min(-(-n_paths // block_m), max_programs)
        # Partial buffer plus a handful of length-(N+1) vectors. No M term.
        return (n_programs + 8) * (n_steps + 1) * element_size

    raise ValueError(f"unknown backend {backend!r}")


# ==========================================================================
# Backends
# ==========================================================================
def make_backend(
    backend: str,
    n_paths: int,
    n_steps: int,
    *,
    dtype: torch.dtype,
    seed: int,
    include_backward: bool,
    max_programs: int,
) -> Callable[[], torch.Tensor]:
    """Build a zero-argument callable producing the EE profile.

    Every backend returns a tensor of shape ``(n_steps + 1,)``, so the
    comparison is like-for-like.

    Args:
        backend: One of the module-level backend names.
        n_paths: Monte-Carlo paths.
        n_steps: Time steps.
        dtype: Working precision.
        seed: RNG seed / Philox key.
        include_backward: Time forward + backward.
        max_programs: Phase 5 launch-grid cap.

    Returns:
        The callable.

    Raises:
        ValueError: On an unknown backend name.
    """
    device = torch.device("cuda")
    dt = MATURITY / n_steps
    legs = portfolio()
    times = torch.linspace(0.0, MATURITY, n_steps + 1, device=device, dtype=dtype)
    coeff_b, coeff_c = build_affine_coefficients(legs, times, RATE)

    def reduce_paths(paths: torch.Tensor) -> torch.Tensor:
        """Paths -> affine MtM -> exposure floor -> mean across paths."""
        mtm = coeff_b.reshape(1, -1) * paths - coeff_c.reshape(1, -1)
        return torch.clamp(mtm, min=0.0).mean(dim=0)

    def leaves() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(S0, device=device, dtype=dtype, requires_grad=True),
            torch.tensor(SIGMA, device=device, dtype=dtype, requires_grad=True),
        )

    if backend == BASELINE:
        def run() -> torch.Tensor:
            if include_backward:
                s0, sigma = leaves()
                generator = torch.Generator(device=device).manual_seed(seed)
                dW = torch.randn(
                    (n_paths, n_steps), device=device, dtype=dtype,
                    generator=generator,
                ) * math.sqrt(dt)
                profile = reduce_paths(simulate_gbm(s0, MU, sigma, dW, dt))
                profile.sum().backward()
                return profile.detach()
            with torch.no_grad():
                generator = torch.Generator(device=device).manual_seed(seed)
                dW = torch.randn(
                    (n_paths, n_steps), device=device, dtype=dtype,
                    generator=generator,
                ) * math.sqrt(dt)
                return reduce_paths(simulate_gbm(S0, MU, SIGMA, dW, dt))
        return run

    if backend == PHASE3:
        def run() -> torch.Tensor:
            if include_backward:
                s0, sigma = leaves()
                generator = torch.Generator(device=device).manual_seed(seed)
                dW = torch.randn(
                    (n_paths, n_steps), device=device, dtype=dtype,
                    generator=generator,
                ) * math.sqrt(dt)
                profile = reduce_paths(triton_simulate_gbm(s0, MU, sigma, dW, dt))
                profile.sum().backward()
                return profile.detach()
            with torch.no_grad():
                generator = torch.Generator(device=device).manual_seed(seed)
                dW = torch.randn(
                    (n_paths, n_steps), device=device, dtype=dtype,
                    generator=generator,
                ) * math.sqrt(dt)
                return reduce_paths(triton_simulate_gbm(S0, MU, SIGMA, dW, dt))
        return run

    if backend == PHASE4:
        def run() -> torch.Tensor:
            if include_backward:
                s0, sigma = leaves()
                paths = philox_simulate_gbm(
                    s0, MU, sigma, n_paths, n_steps, dt, seed=seed, dtype=dtype
                )
                profile = reduce_paths(paths)
                profile.sum().backward()
                return profile.detach()
            with torch.no_grad():
                paths = philox_simulate_gbm(
                    S0, MU, SIGMA, n_paths, n_steps, dt, seed=seed, dtype=dtype
                )
                return reduce_paths(paths)
        return run

    if backend == PHASE5:
        def run() -> torch.Tensor:
            if include_backward:
                s0, sigma = leaves()
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
        return run

    raise ValueError(f"unknown backend {backend!r}")


# ==========================================================================
# Sweep
# ==========================================================================
@dataclass
class SweepRow:
    """All backends measured at one path count."""

    n_paths: int
    results: Dict[str, Measurement] = field(default_factory=dict)

    def speedup(self, backend: str) -> Optional[float]:
        """Backend speedup over the PyTorch baseline, if both ran."""
        base = self.results.get(BASELINE)
        other = self.results.get(backend)
        if base is None or other is None or not (base.ok and other.ok):
            return None
        if not other.milliseconds:
            return None
        return base.milliseconds / other.milliseconds


def run_sweep(
    paths: Sequence[int],
    n_steps: int,
    *,
    dtype: torch.dtype,
    repeats: int,
    seed: int,
    include_backward: bool,
    max_programs: int,
    backends: Sequence[str],
) -> List[SweepRow]:
    """Measure every backend at every path count.

    Args:
        paths: Path counts to sweep.
        n_steps: Time steps.
        dtype: Working precision.
        repeats: Timed iterations per measurement.
        seed: RNG seed.
        include_backward: Time forward + backward.
        max_programs: Phase 5 launch-grid cap.
        backends: Which backends to run.

    Returns:
        One :class:`SweepRow` per path count.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    rows: List[SweepRow] = []

    for n_paths in paths:
        row = SweepRow(n_paths=n_paths)
        print(f"\n  M = {n_paths:,}")

        for backend in backends:
            reset_cuda()
            predicted = predict_peak_bytes(
                backend, n_paths, n_steps, element_size,
                include_backward=include_backward, max_programs=max_programs,
            )
            budget = int(VRAM_SAFETY_FRACTION * free_vram_bytes())

            if predicted > budget:
                row.results[backend] = Measurement(
                    predicted_oom=True,
                    predicted_bytes=predicted,
                    note=f"needs ~{format_bytes(predicted)}, budget {format_bytes(budget)}",
                )
                print(f"    {backend:<28} refused (needs ~{format_bytes(predicted)})")
                continue

            operation = make_backend(
                backend, n_paths, n_steps, dtype=dtype, seed=seed,
                include_backward=include_backward, max_programs=max_programs,
            )
            result, _ = measure(operation, repeats=repeats, keep_output=False)
            row.results[backend] = result
            reset_cuda()

            if result.ok:
                print(
                    f"    {backend:<28} {result.milliseconds:>9,.1f} ms   "
                    f"peak {format_bytes(result.peak_bytes)}"
                )
            else:
                print(f"    {backend:<28} OOM")

        rows.append(row)

    return rows


def find_oom_threshold(
    backend: str,
    n_steps: int,
    *,
    dtype: torch.dtype,
    seed: int,
    include_backward: bool,
    max_programs: int,
    lower: int,
    upper: int,
    tolerance: float = 0.02,
) -> tuple[Optional[int], Optional[int]]:
    """Bisect for the largest path count ``backend`` actually completes.

    A fixed sweep only brackets the boundary. This narrows it by binary search
    until the bracket is within ``tolerance`` relative width, which turns "dies
    somewhere between 1e6 and 5e6" into a number worth quoting.

    Args:
        backend: Backend to probe.
        n_steps: Time steps.
        dtype: Working precision.
        seed: RNG seed.
        include_backward: Whether the adjoint is included.
        max_programs: Phase 5 launch-grid cap.
        lower: A path count known (or assumed) to succeed.
        upper: A path count known (or assumed) to fail.
        tolerance: Stop when ``(upper - lower) / upper`` falls below this.

    Returns:
        ``(largest_success, smallest_failure)``. Either may be ``None`` if the
        initial bracket was wrong in that direction.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()

    def attempt(n_paths: int) -> bool:
        """Return True if the backend completes at this path count."""
        reset_cuda()
        predicted = predict_peak_bytes(
            backend, n_paths, n_steps, element_size,
            include_backward=include_backward, max_programs=max_programs,
        )
        if predicted > int(VRAM_SAFETY_FRACTION * free_vram_bytes()):
            return False
        operation = make_backend(
            backend, n_paths, n_steps, dtype=dtype, seed=seed,
            include_backward=include_backward, max_programs=max_programs,
        )
        # One repeat: this is a feasibility probe, not a timing.
        result, _ = measure(operation, repeats=1, keep_output=False)
        reset_cuda()
        return result.ok

    print(f"\n  Bisecting the OOM boundary for: {backend}")
    print(f"    bracket [{lower:,}, {upper:,}]")

    best_success: Optional[int] = None
    worst_failure: Optional[int] = None

    if attempt(lower):
        best_success = lower
    else:
        print(f"    {lower:,} already fails -- bracket lower bound is too high")
        return None, lower

    if attempt(upper):
        print(f"    {upper:,} succeeds -- bracket upper bound is too low")
        return upper, None
    worst_failure = upper

    while (worst_failure - best_success) / worst_failure > tolerance:
        midpoint = (best_success + worst_failure) // 2
        if midpoint in (best_success, worst_failure):
            break
        succeeded = attempt(midpoint)
        print(f"    {midpoint:>13,}  {'ok' if succeeded else 'OOM'}")
        if succeeded:
            best_success = midpoint
        else:
            worst_failure = midpoint

    return best_success, worst_failure


# ==========================================================================
# Reporting
# ==========================================================================
def build_markdown(
    rows: Sequence[SweepRow],
    backends: Sequence[str],
    *,
    n_steps: int,
    dtype: torch.dtype,
    repeats: int,
    include_backward: bool,
    oom_threshold: Optional[tuple[Optional[int], Optional[int]]] = None,
) -> str:
    """Assemble the full Markdown report.

    Args:
        rows: Completed sweep rows.
        backends: Backends that were run, in display order.
        n_steps: Time steps.
        dtype: Working precision.
        repeats: Timed iterations used.
        include_backward: Whether the adjoint was included.
        oom_threshold: Optional ``(largest_success, smallest_failure)`` from
            :func:`find_oom_threshold`.

    Returns:
        The report as Markdown.
    """
    properties = torch.cuda.get_device_properties(0)
    element_size = torch.tensor([], dtype=dtype).element_size()

    def cell(measurement: Optional[Measurement], kind: str) -> str:
        if measurement is None:
            return "not run"
        if measurement.predicted_oom:
            needed = format_bytes(measurement.predicted_bytes)
            return f"**OOM** (pred, ~{needed})"
        if measurement.failed_oom:
            return "**OOM**"
        if kind == "time":
            return f"{measurement.milliseconds:,.1f}"
        return format_bytes(measurement.peak_bytes)

    lines: List[str] = []
    lines.append("# XVA engine: PyTorch baseline vs Triton fused kernels")
    lines.append("")
    lines.append(
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_"
    )
    lines.append("")

    # ---- environment -------------------------------------------------
    lines.append("## Environment")
    lines.append("")
    lines.append(
        markdown_table(
            ["Item", "Value"],
            [
                ["GPU", f"{properties.name} ({properties.total_memory / BYTES_PER_GIB:,.1f} GiB)"],
                ["Compute capability", f"{properties.major}.{properties.minor}"],
                ["PyTorch", torch.__version__],
                ["CUDA", str(torch.version.cuda)],
                ["Triton", _triton_version()],
                ["Python", platform.python_version()],
                ["dtype", str(dtype).replace("torch.", "")],
                ["Time steps N", f"{n_steps:,}"],
                ["Repeats", f"{repeats} (minimum reported)"],
                ["Measuring", "forward + backward" if include_backward else "forward only"],
            ],
        )
    )
    lines.append("")

    # ---- what is being compared --------------------------------------
    lines.append("## What each backend does")
    lines.append("")
    lines.append(
        markdown_table(
            ["Backend", "Pipeline", "O(M*N) tensors in HBM"],
            [
                [BASELINE, "`simulate_gbm` -> MtM -> clamp -> mean", "~6"],
                [PHASE3, "draw `dW` -> `triton_simulate_gbm` -> reduce", "3"],
                [PHASE4, "`philox_simulate_gbm` (in-kernel RNG) -> reduce", "2"],
                [PHASE5, "`fused_expected_exposure`", "**0**"],
            ],
        )
    )
    lines.append("")
    lines.append(
        "All four produce the identical output: an expected-exposure profile of "
        f"shape `({n_steps + 1},)`. `dW` generation is timed inside the Phase 3 "
        "measurement, since Phases 4 and 5 must produce their increments too."
    )
    lines.append("")

    # ---- execution time ----------------------------------------------
    lines.append("## Execution time (ms)")
    lines.append("")
    lines.append(
        markdown_table(
            ["M"] + list(backends),
            [
                [f"{row.n_paths:,}"]
                + [cell(row.results.get(b), "time") for b in backends]
                for row in rows
            ],
        )
    )
    lines.append("")

    # ---- peak VRAM ---------------------------------------------------
    lines.append("## Peak VRAM")
    lines.append("")
    lines.append(
        markdown_table(
            ["M"] + list(backends),
            [
                [f"{row.n_paths:,}"]
                + [cell(row.results.get(b), "mem") for b in backends]
                for row in rows
            ],
        )
    )
    lines.append("")

    # ---- speedup -----------------------------------------------------
    accelerated = [b for b in backends if b != BASELINE]
    if accelerated:
        lines.append("## Speedup over PyTorch baseline")
        lines.append("")
        speed_rows = []
        for row in rows:
            cells = [f"{row.n_paths:,}"]
            for backend in accelerated:
                factor = row.speedup(backend)
                if factor is None:
                    base = row.results.get(BASELINE)
                    cells.append("n/a (baseline OOM)" if base and base.out_of_memory else "n/a")
                else:
                    cells.append(f"{factor:,.2f}x")
            speed_rows.append(cells)
        lines.append(markdown_table(["M"] + accelerated, speed_rows))
        lines.append("")
        lines.append(
            "_`n/a (baseline OOM)` marks path counts where no speedup is "
            "definable because the baseline cannot run at all -- which is the "
            "more important result._"
        )
        lines.append("")

    # ---- the OOM boundary --------------------------------------------
    lines.append("## Where pure PyTorch stops")
    lines.append("")

    # Survival matrix: the single clearest statement of the result. One row per
    # path count, one column per backend, so the crossover is visible at a
    # glance rather than inferred from two other tables.
    lines.append("### Survival by path count")
    lines.append("")
    survival_rows = []
    for row in rows:
        cells = [f"{row.n_paths:,}"]
        for backend in backends:
            measurement = row.results.get(backend)
            if measurement is None:
                cells.append("not run")
            elif measurement.ok:
                cells.append(f"ok ({measurement.milliseconds:,.0f} ms)")
            elif measurement.predicted_oom:
                cells.append("**OOM**")
            else:
                cells.append("**OOM**")
        survival_rows.append(cells)
    lines.append(markdown_table(["M"] + list(backends), survival_rows))
    lines.append("")

    first_baseline_oom = next(
        (
            row.n_paths
            for row in rows
            if (m := row.results.get(BASELINE)) is not None and m.out_of_memory
        ),
        None,
    )
    last_baseline_ok = None
    for row in rows:
        measurement = row.results.get(BASELINE)
        if measurement is not None and measurement.ok:
            last_baseline_ok = row.n_paths

    if first_baseline_oom is None:
        lines.append(
            "The PyTorch baseline completed at every path count in this sweep. "
            "Extend `--paths` upward to locate its ceiling."
        )
    else:
        lines.append(
            f"- **Largest M the baseline completed:** {last_baseline_ok:,}"
            if last_baseline_ok
            else "- The baseline failed at the smallest M in this sweep."
        )
        lines.append(f"- **First M where the baseline OOMs:** {first_baseline_oom:,}")

        if oom_threshold is not None:
            success, failure = oom_threshold
            lines.append("")
            if success is not None and failure is not None:
                lines.append(
                    f"Bisected boundary: the baseline completes at "
                    f"**{success:,}** paths and fails at **{failure:,}** "
                    f"(bracket width {100 * (failure - success) / failure:.1f}%)."
                )
            elif success is not None:
                lines.append(
                    f"Bisection found no failure up to {success:,} paths; "
                    "raise the upper bracket."
                )
            else:
                lines.append(
                    f"Bisection found failure even at the lower bracket "
                    f"({failure:,} paths); lower it."
                )

        # Which accelerated backends survive past that point?
        survivors = []
        for backend in accelerated:
            row = next((r for r in rows if r.n_paths == first_baseline_oom), None)
            if row is not None:
                measurement = row.results.get(backend)
                if measurement is not None and measurement.ok:
                    survivors.append(
                        f"{backend} ({measurement.milliseconds:,.1f} ms, "
                        f"peak {format_bytes(measurement.peak_bytes)})"
                    )
        if survivors:
            lines.append("")
            lines.append(
                f"At M = {first_baseline_oom:,}, where the baseline cannot run, "
                "these complete:"
            )
            lines.append("")
            for entry in survivors:
                lines.append(f"- {entry}")

    lines.append("")
    lines.append("### Analytic memory model")
    lines.append("")
    lines.append(
        f"At N = {n_steps}, {str(dtype).replace('torch.', '')}, one M x (N+1) "
        f"tensor is `M x {n_steps + 1} x {element_size}` bytes:"
    )
    lines.append("")
    model_rows = []
    for n_paths in (1_000_000, 5_000_000, 10_000_000, 50_000_000):
        one = n_paths * (n_steps + 1) * element_size
        model_rows.append(
            [
                f"{n_paths:,}",
                format_bytes(one),
                format_bytes(6 * one),
                format_bytes(2 * one),
                format_bytes(
                    predict_peak_bytes(
                        PHASE5, n_paths, n_steps, element_size,
                        include_backward=include_backward,
                    )
                ),
            ]
        )
    lines.append(
        markdown_table(
            ["M", "one M x (N+1)", "baseline (~6x)", "Phase 4 (~2x)", "Phase 5"],
            model_rows,
        )
    )
    lines.append("")
    lines.append(
        "Phase 5's column carries no M term at all: peak is "
        "`min(ceil(M / BLOCK_M), max_programs) x (N+1) x element_size`, which is "
        "constant above the grid-saturation point."
    )
    lines.append("")

    # ---- caveats -----------------------------------------------------
    lines.append("## Reading these numbers")
    lines.append("")
    lines.append(
        "- **Peak VRAM is `torch.cuda.max_memory_allocated`** -- what the "
        "PyTorch caching allocator handed out, not total process VRAM."
    )
    lines.append(
        "- **`OOM (pred)` means refused before launch** by a pre-flight check, "
        "not attempted and failed. Attempting a configuration that overruns the "
        "device risks an asynchronous abort that would poison the CUDA context "
        "and invalidate every later row in the sweep."
    )
    lines.append(
        "- **Timings are the minimum of "
        f"{repeats} repeats**, with a warm-up excluded so Triton JIT "
        "compilation is not billed to the first measurement."
    )
    lines.append(
        "- **Phases 4 and 5 draw different sample paths** from the baseline "
        "(different Philox addressing), so their exposure profiles agree only "
        "within Monte-Carlo error. Correctness is established in "
        "`tests/test_phase4.py` and `tests/test_phase5.py`, not here."
    )
    return "\n".join(lines)


def _triton_version() -> str:
    """Return the installed Triton version, or a marker if absent."""
    if not HAS_TRITON:
        return "not installed"
    try:
        import triton

        return str(triton.__version__)
    except Exception:  # noqa: BLE001
        return "unknown"


# ==========================================================================
# Entry point
# ==========================================================================
def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI."""
    parser = argparse.ArgumentParser(
        prog="bench_all_phases.py",
        description=(
            "Compare the PyTorch baseline against the Phase 3/4/5 Triton "
            "kernels across a path sweep, and emit a Markdown report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paths", type=int, nargs="+", default=DEFAULT_PATHS,
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
    parser.add_argument("--seed", type=int, default=20260826, help="RNG seed.")
    parser.add_argument(
        "--max-programs", type=int, default=DEFAULT_MAX_PROGRAMS,
        help="Phase 5 launch-grid cap.",
    )
    parser.add_argument(
        "--backward", action="store_true",
        help="Time forward + backward instead of forward only.",
    )
    parser.add_argument(
        "--find-oom", action="store_true",
        help="Bisect for the exact path count where the PyTorch baseline OOMs.",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=["baseline", "phase3", "phase4", "phase5"],
        help="Backends to omit.",
    )
    parser.add_argument(
        "--markdown", type=Path, default=None,
        help="Write the Markdown report to this path (also printed to stdout).",
    )
    return parser


def main() -> int:
    """Run the sweep and emit the report.

    Returns:
        ``0`` on success, ``1`` if the Triton path is unavailable, ``2`` on
        invalid arguments.
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
            "  This benchmark is GPU-only. Run it on a CUDA machine with "
            "Triton installed (e.g. Google Colab).\n",
            file=sys.stderr,
        )
        return 1

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    skip_map = {
        "baseline": BASELINE, "phase3": PHASE3,
        "phase4": PHASE4, "phase5": PHASE5,
    }
    skipped = {skip_map[name] for name in args.skip}
    backends = [b for b in BACKEND_ORDER if b not in skipped]

    properties = torch.cuda.get_device_properties(0)
    print()
    print("=" * 78)
    print("  CROSS-PHASE BENCHMARK  --  PyTorch baseline vs Triton fused kernels")
    print("=" * 78)
    print(f"  Device   : {properties.name} "
          f"({properties.total_memory / BYTES_PER_GIB:,.1f} GiB)")
    print(f"  torch    : {torch.__version__}   cuda {torch.version.cuda}   "
          f"triton {_triton_version()}")
    print(f"  dtype    : {args.dtype}   N = {args.steps:,}   "
          f"repeats {args.repeats}")
    print(f"  Measuring: {'forward + backward' if args.backward else 'forward only'}")
    print(f"  Backends : {', '.join(backends)}")

    rows = run_sweep(
        sorted(set(args.paths)), args.steps,
        dtype=dtype, repeats=args.repeats, seed=args.seed,
        include_backward=args.backward, max_programs=args.max_programs,
        backends=backends,
    )

    threshold = None
    if args.find_oom and BASELINE in backends:
        completed = [
            r.n_paths for r in rows
            if (m := r.results.get(BASELINE)) is not None and m.ok
        ]
        failed = [
            r.n_paths for r in rows
            if (m := r.results.get(BASELINE)) is not None and m.out_of_memory
        ]
        if completed and failed:
            threshold = find_oom_threshold(
                BASELINE, args.steps, dtype=dtype, seed=args.seed,
                include_backward=args.backward, max_programs=args.max_programs,
                lower=max(completed), upper=min(failed),
            )
        else:
            print(
                "\n  Skipping bisection: the sweep did not bracket the boundary "
                "(need at least one success and one OOM)."
            )

    report = build_markdown(
        rows, backends, n_steps=args.steps, dtype=dtype,
        repeats=args.repeats, include_backward=args.backward,
        oom_threshold=threshold,
    )

    print()
    print("=" * 78)
    print()
    print(report)
    print()

    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(report + "\n", encoding="utf-8")
        print(f"  Markdown report written to {args.markdown}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
