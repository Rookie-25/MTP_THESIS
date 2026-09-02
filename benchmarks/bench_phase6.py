r"""Phase 6 benchmark: local-volatility AAD, Triton kernel vs PyTorch autograd.

What is compared
================
Both backends compute the same estimator under the same state-dependent
volatility: parameters in, expected-exposure profile out, then a gradient with
respect to the surface parameters.

They do **not** produce identical numbers, and are not expected to. The kernel
draws its Brownian increments from Philox in registers; the baseline draws
``torch.randn``. Same estimator, independent streams, so the two profiles agree
only to Monte-Carlo error -- which the agreement check measures and reports
rather than assuming.

==================  =====================================================
backend             pipeline
==================  =====================================================
PyTorch autograd    ``reference_local_vol_ee`` -- a sequential Python time
                    loop over ``N`` steps, differentiated by the autograd
                    tape
Phase 6 Triton      ``fused_local_vol_ee`` -- sequential time loop in the
                    kernel, hand-written adjoint with sqrt(N) checkpointing
==================  =====================================================

Three quantities are reported separately, because they behave very differently
here:

* **forward time** -- under ``no_grad``, so no tape is built;
* **backward time** -- CUDA events wrapped around the ``.backward()`` call
  alone, with the forward run outside the timed region. Reporting
  ``total - forward`` instead would fold the tape-construction cost of the
  forward into the backward number;
* **peak VRAM**, measured separately for forward-only and for forward+backward.
  The gap between those two is the whole point of this phase.

Why the backward memory gap is the headline
===========================================
With constant volatility (Phases 3-5) the adjoint collapses to a suffix sum and
needs almost nothing kept from the forward. With :math:`\sigma(t, S_t)` it does
not: the reverse sweep needs :math:`\sigma_k` and
:math:`\partial\sigma_k/\partial X_k` at every step, in descending :math:`k`.

PyTorch's answer is to store the whole tape -- roughly five :math:`(M,)`
tensors per step across :math:`N` steps, so :math:`O(MN)` and growing with the
number of steps. The kernel's answer is sqrt(N) checkpointing: keep only
segment-entry states in SRAM and replay each segment on demand, for
:math:`O(\text{n\_programs} \times N)` total and no :math:`M` term at all.

So the expected result is not merely "the kernel is faster". It is that the
PyTorch baseline's *backward* runs out of memory at a path count where the
kernel's backward is still flat.

A caveat on the title of this benchmark
=======================================
This does **not** differentiate SSVI parameters, despite local volatility being
an SSVI concept elsewhere in this project. The Phase 6 kernel evaluates a
*parametric* surface

.. math::
    \sigma(t, x) = \sigma_0
                 + \sigma_{\text{skew}}\tanh(\kappa (x - x_{\text{ref}}))
                 + \sigma_{\text{term}} t

and its gradients are with respect to :math:`S_0`, :math:`\mu`,
:math:`\sigma_0` and :math:`\sigma_{\text{skew}}`. The SSVI surface in
:mod:`src.models.vol_surface` has entirely different parameters
(:math:`\rho, \eta, \gamma`, and the ATM variance term structure
:math:`\theta_T`), and **there is currently no code path connecting them** --
the kernel never sees an ``SSVISurface``.

Benchmarking "SSVI gradients" would therefore mean measuring something that
does not exist. What is measured instead is the honest comparison: identical
surfaces, identical outputs, identical gradient targets, on both backends.

Closing that gap needs a projection step -- fit
:math:`(\sigma_0, \sigma_{\text{skew}}, \kappa)` to the Dupire
:math:`\sigma_{LV}` implied by a calibrated SSVI surface, or replace the tanh
form with a Chebyshev expansion in :math:`(t, \log S)` whose coefficients are
themselves differentiable functions of the SSVI parameters. Either is real work
and neither is done.

Usage
=====
    python benchmarks/bench_phase6.py
    python benchmarks/bench_phase6.py --paths 10000 100000 1000000 5000000
    python benchmarks/bench_phase6.py --steps 252 --markdown phase6.md
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
    format_bytes,
    free_vram_bytes,
    is_oom,
    markdown_table,
    reset_cuda,
)
from src.csrc.triton_cva_fusion import build_affine_coefficients  # noqa: E402
from src.csrc.triton_gbm import HAS_TRITON  # noqa: E402
from src.csrc.triton_local_vol_cva import (  # noqa: E402
    LocalVolParams,
    fused_local_vol_ee,
    is_available,
    reference_local_vol_ee,
    select_local_vol_blocks,
)
from src.pricer.options import SwapLeg  # noqa: E402

SPOT = 100.0
DRIFT = 0.02
RATE = 0.03
BASE = 0.20
SKEW = 0.15
MATURITY = 1.0
DEFAULT_MAX_PROGRAMS = 4096

#: Strike deliberately off SPOT. At ``strike == SPOT`` the t=0 exposure is
#: exactly zero, which sits on the ``max(V, 0)`` kink where no derivative
#: exists -- see tests/test_phase6_kernel.py::TestAtTheMoneyKink. A benchmark
#: is not a correctness test, but there is no reason to time a degenerate point.
STRIKE = 95.0

BASELINE = "PyTorch autograd"
KERNEL = "Phase 6 Triton"
BACKENDS = (BASELINE, KERNEL)

#: Rough count of (M,) tensors the autograd tape retains per time step in
#: ``reference_local_vol_ee`` (tanh output, sigma, the squared term, the
#: increment, the new state), plus the stacked/exponentiated/clamped
#: (M, N+1) surfaces. Used only for the pre-flight refusal, so it is
#: deliberately an under-estimate: erring low means attempting a run that
#: might fail rather than refusing one that would have succeeded.
BASELINE_TAPE_TENSORS_PER_STEP = 8


def portfolio() -> List[SwapLeg]:
    """The single-leg netting set benchmarked throughout."""
    return [SwapLeg(notional=1.0, strike=STRIKE, maturity=MATURITY)]


def surface_params() -> LocalVolParams:
    """Parametric local volatility, skew centred on ``log(SPOT)``.

    Centring matters: an uncentred ``tanh`` saturates at ``log(100) ~ 4.6``,
    driving ``dsigma/dx`` to ~1e-5 and quietly reducing the whole thing to the
    constant-volatility case -- which would make this benchmark measure Phase 5
    while claiming to measure Phase 6.
    """
    return LocalVolParams(
        base=BASE, skew=SKEW, kappa=2.5, term=0.05, reference=math.log(SPOT)
    )


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
    """Predict peak allocation, counting only the ``O(M*N)`` terms.

    Args:
        backend: One of the module-level backend names.
        n_paths: Monte-Carlo paths :math:`M`.
        n_steps: Time steps :math:`N`.
        element_size: Bytes per element.
        include_backward: Whether the gradient pass is included.
        max_programs: Kernel launch-grid cap.

    Returns:
        Predicted peak bytes.

    Raises:
        ValueError: On an unknown backend name.
    """
    if backend == BASELINE:
        normals = n_paths * n_steps * element_size
        surfaces = 3 * n_paths * (n_steps + 1) * element_size
        if not include_backward:
            # no_grad: no tape, just the working surfaces.
            return normals + surfaces
        tape = BASELINE_TAPE_TENSORS_PER_STEP * n_paths * n_steps * element_size
        return normals + surfaces + tape

    if backend == KERNEL:
        block_m, _ = select_local_vol_blocks(n_steps, element_size)
        n_programs = min(-(-n_paths // block_m), max_programs)
        # Partial-sum buffer plus a handful of length-(N+1) vectors. The
        # checkpoint tiles live in SRAM and never appear in the allocator.
        partials = (n_programs + 8) * (n_steps + 1) * element_size
        if include_backward:
            partials += 4 * n_programs * element_size  # per-program gradients
        return partials

    raise ValueError(f"unknown backend {backend!r}")


# ==========================================================================
# Backends
# ==========================================================================
def make_forward(
    backend: str,
    n_paths: int,
    n_steps: int,
    *,
    dtype: torch.dtype,
    seed: int,
    max_programs: int,
    with_grad: bool,
) -> Callable[[], torch.Tensor]:
    """Build a callable returning the EE profile.

    Args:
        backend: Backend name.
        n_paths: Monte-Carlo paths.
        n_steps: Time steps.
        dtype: Working precision.
        seed: RNG seed / Philox key.
        max_programs: Kernel launch-grid cap.
        with_grad: Whether the returned profile must carry a graph. The
            parameters are created *inside* the callable so each invocation
            starts from fresh leaves.

    Returns:
        A zero-argument callable. When ``with_grad`` it returns
        ``(profile, leaves)``; otherwise just the profile.

    Raises:
        ValueError: On an unknown backend name.
    """
    device = torch.device("cuda")
    dt = MATURITY / n_steps
    legs = portfolio()
    params = surface_params()
    times = torch.linspace(0.0, MATURITY, n_steps + 1, device=device, dtype=dtype)
    coeff_b, coeff_c = build_affine_coefficients(legs, times, RATE)

    def leaves() -> Dict[str, torch.Tensor]:
        return {
            name: torch.tensor(
                value, device=device, dtype=dtype, requires_grad=with_grad
            )
            for name, value in (
                ("spot", SPOT), ("drift", DRIFT), ("base", BASE), ("skew", SKEW)
            )
        }

    if backend == BASELINE:
        def run():
            leaf = leaves()
            generator = torch.Generator(device=device).manual_seed(seed)
            # O(M*N) and unavoidable in the PyTorch formulation: the sequential
            # loop indexes normals[:, k] at every step, so the whole draw must
            # stay resident. The kernel generates its increments in registers.
            normals = torch.randn(
                (n_paths, n_steps), device=device, dtype=dtype, generator=generator
            )
            profile = reference_local_vol_ee(
                leaf["spot"], leaf["drift"], leaf["base"], leaf["skew"],
                normals, dt, coeff_b, coeff_c, params,
            )
            return (profile, leaf) if with_grad else profile
        return run

    if backend == KERNEL:
        def run():
            leaf = leaves()
            profile = fused_local_vol_ee(
                leaf["spot"], leaf["drift"], legs, times, RATE, n_paths, params,
                base=leaf["base"], skew=leaf["skew"],
                seed=seed, max_programs=max_programs,
            )
            return (profile, leaf) if with_grad else profile
        return run

    raise ValueError(f"unknown backend {backend!r}")


# ==========================================================================
# Measurement
# ==========================================================================
@dataclass
class Stage:
    """One timed stage (forward, or backward) with its own memory verdict.

    Forward and backward are guarded and recorded separately because they fail
    separately. The baseline's forward is ``O(M*N)`` for the normals alone; its
    backward adds the autograd tape on top of that. There are path counts where
    the forward completes comfortably and the backward cannot be attempted, and
    that gap *is* the Phase 6 result -- collapsing both into a single verdict
    would print a blank exactly where the finding should be.
    """

    ms: Optional[float] = None
    peak: Optional[int] = None
    failed_oom: bool = False
    predicted_oom: bool = False
    predicted_bytes: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.ms is not None

    @property
    def out_of_memory(self) -> bool:
        return self.failed_oom or self.predicted_oom


@dataclass
class PhaseSixMeasurement:
    """Both stages for one backend at one path count."""

    forward: Stage = field(default_factory=Stage)
    backward: Stage = field(default_factory=Stage)

    @property
    def ok(self) -> bool:
        return self.forward.ok and self.backward.ok

    @property
    def total_ms(self) -> Optional[float]:
        if not self.ok:
            return None
        return self.forward.ms + self.backward.ms


def _guard(
    backend: str,
    n_paths: int,
    n_steps: int,
    element_size: int,
    *,
    include_backward: bool,
    max_programs: int,
) -> Optional[Stage]:
    """Return a refusal :class:`Stage` if the prediction exceeds free VRAM.

    Returns:
        ``None`` when the stage is worth attempting, otherwise a populated
        refusal.
    """
    predicted = predict_peak_bytes(
        backend, n_paths, n_steps, element_size,
        include_backward=include_backward, max_programs=max_programs,
    )
    if predicted > VRAM_SAFETY_FRACTION * free_vram_bytes():
        return Stage(predicted_oom=True, predicted_bytes=predicted)
    return None


def measure_backend(
    backend: str,
    n_paths: int,
    n_steps: int,
    *,
    dtype: torch.dtype,
    repeats: int,
    seed: int,
    max_programs: int,
) -> PhaseSixMeasurement:
    """Measure the forward and backward stages independently.

    The backward is timed with CUDA events around the ``.backward()`` call
    alone, with the forward completed and synchronised first. Timing
    ``total - forward`` instead would attribute the forward's tape-building
    cost to the backward -- which is precisely the quantity under study.

    Args:
        backend: Backend name.
        n_paths: Monte-Carlo paths.
        n_steps: Time steps.
        dtype: Working precision.
        repeats: Timed iterations; the minimum is reported.
        seed: RNG seed.
        max_programs: Kernel launch-grid cap.

    Returns:
        A populated :class:`PhaseSixMeasurement`. Either stage may record an
        out-of-memory verdict while the other succeeds.
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    result = PhaseSixMeasurement()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # ---- stage 1: forward, under no_grad ----------------------------
    reset_cuda()
    refused = _guard(
        backend, n_paths, n_steps, element_size,
        include_backward=False, max_programs=max_programs,
    )
    if refused is not None:
        result.forward = refused
    else:
        try:
            forward_only = make_forward(
                backend, n_paths, n_steps, dtype=dtype, seed=seed,
                max_programs=max_programs, with_grad=False,
            )

            def run_forward() -> None:
                with torch.no_grad():
                    output = forward_only()
                del output

            run_forward()  # warm-up: Triton JIT, allocator growth
            torch.cuda.synchronize()
            reset_cuda()

            elapsed = math.inf
            for _ in range(repeats):
                start.record()
                run_forward()
                end.record()
                torch.cuda.synchronize()
                elapsed = min(elapsed, start.elapsed_time(end))
            result.forward = Stage(
                ms=elapsed, peak=torch.cuda.max_memory_allocated()
            )
        except Exception as error:  # noqa: BLE001 - OOM is an expected outcome
            if not is_oom(error):
                raise
            result.forward = Stage(failed_oom=True)
        finally:
            reset_cuda()

    # ---- stage 2: backward, timed in isolation ----------------------
    reset_cuda()
    refused = _guard(
        backend, n_paths, n_steps, element_size,
        include_backward=True, max_programs=max_programs,
    )
    if refused is not None:
        result.backward = refused
        return result

    try:
        with_graph = make_forward(
            backend, n_paths, n_steps, dtype=dtype, seed=seed,
            max_programs=max_programs, with_grad=True,
        )
        weights = torch.randn(
            n_steps + 1, device="cuda", dtype=dtype,
            generator=torch.Generator(device="cuda").manual_seed(seed + 7),
        )

        profile, _leaf = with_graph()  # warm-up
        (profile * weights).sum().backward()
        torch.cuda.synchronize()
        del profile, _leaf
        reset_cuda()

        elapsed = math.inf
        for _ in range(repeats):
            profile, _leaf = with_graph()
            loss = (profile * weights).sum()
            torch.cuda.synchronize()
            start.record()
            loss.backward()
            end.record()
            torch.cuda.synchronize()
            elapsed = min(elapsed, start.elapsed_time(end))
            del profile, loss, _leaf
        result.backward = Stage(
            ms=elapsed, peak=torch.cuda.max_memory_allocated()
        )
    except Exception as error:  # noqa: BLE001
        if not is_oom(error):
            raise
        result.backward = Stage(failed_oom=True)
    finally:
        reset_cuda()

    return result


def cross_check(
    n_paths: int,
    n_steps: int,
    *,
    dtype: torch.dtype,
    seed: int,
    max_programs: int,
) -> Optional[float]:
    r"""Max relative deviation between the two backends' EE profiles.

    A timing comparison only means something if both sides compute the same
    quantity. They use independent random streams, so exact agreement is
    neither expected nor asserted; the deviation should sit at Monte-Carlo
    scale, roughly :math:`O(1/\sqrt{M})`. A value far above that means the two
    backends are not pricing the same thing, and every timing below would be
    comparing different computations.

    Args:
        n_paths: Paths to compare at.
        n_steps: Time steps.
        dtype: Working precision.
        seed: RNG seed.
        max_programs: Kernel launch-grid cap.

    Returns:
        Max deviation relative to peak EE, or ``None`` if either backend could
        not run.
    """
    try:
        with torch.no_grad():
            profiles = [
                make_forward(
                    backend, n_paths, n_steps, dtype=dtype, seed=seed,
                    max_programs=max_programs, with_grad=False,
                )()
                for backend in BACKENDS
            ]
        scale = profiles[0].abs().max().clamp(min=1e-12)
        return float(((profiles[0] - profiles[1]).abs() / scale).max())
    except Exception as error:  # noqa: BLE001
        if not is_oom(error):
            raise
        return None
    finally:
        reset_cuda()


@dataclass
class SweepRow:
    """Both backends at one path count."""

    n_paths: int
    results: Dict[str, PhaseSixMeasurement] = field(default_factory=dict)

    def ratio(self, stage: str, attribute: str) -> Optional[float]:
        """Baseline-over-kernel ratio for one stage attribute."""
        base = self.results.get(BASELINE)
        kernel = self.results.get(KERNEL)
        if base is None or kernel is None:
            return None
        left = getattr(getattr(base, stage), attribute)
        right = getattr(getattr(kernel, stage), attribute)
        if not left or not right:
            return None
        return left / right


# ==========================================================================
# Reporting
# ==========================================================================
def _cell(
    measurement: Optional[PhaseSixMeasurement], stage: str, attribute: str
) -> str:
    """Render one table cell."""
    if measurement is None:
        return "not run"
    entry: Stage = getattr(measurement, stage)
    if entry.predicted_oom:
        return f"**OOM** (pred ~{format_bytes(entry.predicted_bytes)})"
    if entry.failed_oom:
        return "**OOM**"
    value = getattr(entry, attribute)
    if value is None:
        return "-"
    return f"{value:,.1f}" if attribute == "ms" else format_bytes(value)


def _ratio_cell(row: SweepRow, stage: str, attribute: str) -> str:
    """Render a speedup / memory-saving cell."""
    value = row.ratio(stage, attribute)
    if value is None:
        return "-"
    return f"{value:,.2f}x" if attribute == "ms" else f"{value:,.0f}x"


def build_markdown(
    rows: Sequence[SweepRow],
    *,
    n_steps: int,
    dtype: torch.dtype,
    repeats: int,
    deviation: Optional[float] = None,
    cross_check_paths: Optional[int] = None,
) -> str:
    """Assemble the Markdown report."""
    properties = torch.cuda.get_device_properties(0)
    element_size = torch.tensor([], dtype=dtype).element_size()
    block_m, block_ck = select_local_vol_blocks(n_steps, element_size)

    lines: List[str] = []
    lines.append("# Phase 6: local-volatility AAD, Triton vs PyTorch autograd")
    lines.append("")
    lines.append(
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_"
    )
    lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append(markdown_table(
        ["Item", "Value"],
        [
            ["GPU", f"{properties.name} "
                    f"({properties.total_memory / BYTES_PER_GIB:,.1f} GiB)"],
            ["PyTorch", torch.__version__],
            ["CUDA", str(torch.version.cuda)],
            ["Triton", _triton_version()],
            ["Python", platform.python_version()],
            ["dtype", str(dtype).replace("torch.", "")],
            ["Time steps N", f"{n_steps:,}"],
            ["Checkpointing", f"BLOCK_M={block_m}, BLOCK_CK={block_ck}, "
                              f"segments={-(-n_steps // block_ck)}"],
            ["Repeats", f"{repeats} (minimum reported)"],
        ],
    ))
    lines.append("")

    if deviation is not None and cross_check_paths:
        expected = 1.0 / math.sqrt(cross_check_paths)
        verdict = (
            "consistent with Monte-Carlo error"
            if deviation < 20.0 * expected
            else "**larger than Monte-Carlo error would explain -- investigate "
                 "before trusting the timings below**"
        )
        lines.append("## Agreement check")
        lines.append("")
        lines.append(
            f"At M = {cross_check_paths:,} the two backends' EE profiles differ "
            f"by at most **{deviation:.2%}** of peak EE, against a 1/sqrt(M) "
            f"scale of {expected:.2%} -- {verdict}. They draw independent "
            "random streams (Philox in-kernel vs `torch.randn`), so exact "
            "agreement is neither expected nor asserted; this confirms only "
            "that both sides price the same instrument under the same surface."
        )
        lines.append("")

    lines.append("## Forward time (ms)")
    lines.append("")
    lines.append(markdown_table(
        ["M", BASELINE, KERNEL, "speedup"],
        [[f"{row.n_paths:,}",
          _cell(row.results.get(BASELINE), "forward", "ms"),
          _cell(row.results.get(KERNEL), "forward", "ms"),
          _ratio_cell(row, "forward", "ms")] for row in rows],
    ))
    lines.append("")

    lines.append("## Backward time (ms)")
    lines.append("")
    lines.append(
        "Timed with CUDA events around `.backward()` alone, with the forward "
        "completed and synchronised first."
    )
    lines.append("")
    lines.append(markdown_table(
        ["M", BASELINE, KERNEL, "speedup"],
        [[f"{row.n_paths:,}",
          _cell(row.results.get(BASELINE), "backward", "ms"),
          _cell(row.results.get(KERNEL), "backward", "ms"),
          _ratio_cell(row, "backward", "ms")] for row in rows],
    ))
    lines.append("")

    lines.append("## Peak VRAM")
    lines.append("")
    lines.append(markdown_table(
        ["M",
         f"{BASELINE} fwd", f"{BASELINE} fwd+bwd",
         f"{KERNEL} fwd", f"{KERNEL} fwd+bwd",
         "saving (fwd+bwd)"],
        [[f"{row.n_paths:,}",
          _cell(row.results.get(BASELINE), "forward", "peak"),
          _cell(row.results.get(BASELINE), "backward", "peak"),
          _cell(row.results.get(KERNEL), "forward", "peak"),
          _cell(row.results.get(KERNEL), "backward", "peak"),
          _ratio_cell(row, "backward", "peak")] for row in rows],
    ))
    lines.append("")

    # ---- the headline ------------------------------------------------
    lines.append("## Where the autograd tape stops")
    lines.append("")
    tape_only = [
        row for row in rows
        if (base := row.results.get(BASELINE)) is not None
        and base.forward.ok and base.backward.out_of_memory
    ]
    both_gone = [
        row for row in rows
        if (base := row.results.get(BASELINE)) is not None
        and base.forward.out_of_memory
    ]
    kernel_alive = [
        row for row in rows
        if (kernel := row.results.get(KERNEL)) is not None and kernel.ok
    ]

    if tape_only or both_gone:
        lines.append(
            "PyTorch's backward retains the whole tape -- roughly "
            f"{BASELINE_TAPE_TENSORS_PER_STEP} `(M,)` tensors per step across "
            f"{n_steps} steps, so `O(M*N)`. The kernel replays sqrt(N) "
            "checkpoints held in SRAM, so its backward adds no `O(M*N)` term "
            "at all."
        )
        lines.append("")
    for row in tape_only:
        kernel = row.results.get(KERNEL)
        suffix = (
            f"the kernel's backward runs in {kernel.backward.ms:,.1f} ms at "
            f"{format_bytes(kernel.backward.peak)}"
            if kernel is not None and kernel.ok else "the kernel is unavailable"
        )
        lines.append(
            f"- **M = {row.n_paths:,}** -- the baseline forward completes "
            f"({row.results[BASELINE].forward.ms:,.1f} ms) but its **backward "
            f"cannot be attempted**; {suffix}. This is the cleanest statement "
            "of the result: the forward is not the problem, the tape is."
        )
    for row in both_gone:
        kernel = row.results.get(KERNEL)
        suffix = (
            "the kernel completes both stages in "
            f"{kernel.forward.ms + kernel.backward.ms:,.1f} ms at "
            f"{format_bytes(kernel.backward.peak)}"
            if kernel is not None and kernel.ok else "the kernel is unavailable"
        )
        lines.append(
            f"- **M = {row.n_paths:,}** -- the baseline is out of memory in the "
            f"forward already; {suffix}."
        )
    if not tape_only and not both_gone:
        lines.append(
            "Both backends completed every stage at every path count in this "
            "sweep. Extend `--paths` upward to find the baseline's ceiling."
        )
    if kernel_alive:
        peaks = {row.results[KERNEL].backward.peak for row in kernel_alive}
        lines.append("")
        lines.append(
            "The kernel's fwd+bwd peak across the whole sweep spans "
            f"{format_bytes(min(peaks))} to {format_bytes(max(peaks))} -- "
            "bounded by the launch-grid cap (`--max-programs`), not by M."
        )
    lines.append("")

    lines.append("## Reading these numbers")
    lines.append("")
    lines.append(
        "- **This does NOT differentiate SSVI parameters.** The kernel "
        "evaluates a parametric surface "
        "`sigma = base + skew*tanh(kappa*(x - x_ref)) + term*t`, and its "
        "gradients are w.r.t. `S0`, `mu`, `base` and `skew`. The SSVI surface "
        "in `src/models/vol_surface.py` has entirely different parameters "
        "(`rho`, `eta`, `gamma`, and the ATM variance term structure) and "
        "**no code path connects them to the kernel**. Closing that gap needs "
        "a projection step -- fitting the tanh parameters to the Dupire "
        "`sigma_LV` implied by a calibrated SSVI surface, or replacing the "
        "tanh with a Chebyshev expansion in `(t, log S)` whose coefficients "
        "depend differentiably on the SSVI parameters. Neither is implemented, "
        "so neither is benchmarked here."
    )
    lines.append(
        "- **Forward and backward are guarded separately.** A row can show a "
        "completed forward beside an OOM backward; that is a real result, not "
        "a missing measurement."
    )
    lines.append(
        "- **The strike is 95, not 100.** At `strike == spot` the t=0 exposure "
        "is exactly zero, sitting on the `max(V, 0)` kink where no derivative "
        "exists (see `tests/test_phase6_kernel.py::TestAtTheMoneyKink`). There "
        "is no reason to time a degenerate point."
    )
    lines.append(
        "- **`OOM (pred)` means refused before launch.** The baseline's tape "
        "size is estimated, deliberately on the low side, so the guard errs "
        "toward attempting a run rather than refusing a feasible one."
    )
    lines.append(
        "- **Peak VRAM is `torch.cuda.max_memory_allocated`** -- allocator "
        "scope, not total process VRAM. The kernel's checkpoint tiles live in "
        "SRAM and never appear in it at all, so its true working set is a "
        "little larger than the column shows."
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
        prog="bench_phase6.py",
        description=(
            "Compare the Phase 6 local-volatility Triton kernel against a "
            "PyTorch autograd baseline on forward time, backward time and "
            "peak VRAM."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--paths", type=int, nargs="+",
        default=[10_000, 100_000, 1_000_000, 5_000_000],
        help="Path counts M to sweep.",
    )
    parser.add_argument("--steps", type=int, default=252, help="Time steps N.")
    parser.add_argument(
        "--dtype", choices=["float32", "float64"], default="float32",
        help="Working precision.",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Timed iterations per measurement (minimum reported).",
    )
    parser.add_argument("--seed", type=int, default=20260901, help="RNG seed.")
    parser.add_argument(
        "--max-programs", type=int, default=DEFAULT_MAX_PROGRAMS,
        help="Kernel launch-grid cap.",
    )
    parser.add_argument(
        "--no-cross-check", action="store_true",
        help=(
            "Skip the backend agreement check. The check runs both forwards at "
            "the smallest path count and reports how far apart the EE profiles "
            "are, which is what makes the timings below meaningful."
        ),
    )
    parser.add_argument("--markdown", type=Path, default=None,
                        help="Write the report to this path.")
    return parser


def main() -> int:
    """Run the sweep and emit the report.

    Returns:
        ``0`` on success, ``1`` if the kernel is unavailable, ``2`` on invalid
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
            "\n  Cannot run: the Phase 6 Triton kernel is unavailable.\n"
            f"    triton installed : {HAS_TRITON}\n"
            f"    cuda available   : {torch.cuda.is_available()}\n\n"
            "  This benchmark is GPU-only.\n",
            file=sys.stderr,
        )
        return 1

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    element_size = torch.tensor([], dtype=dtype).element_size()
    block_m, block_ck = select_local_vol_blocks(args.steps, element_size)
    properties = torch.cuda.get_device_properties(0)

    print()
    print("=" * 80)
    print("  PHASE 6  --  local-volatility AAD: Triton kernel vs PyTorch autograd")
    print("=" * 80)
    print(f"  Device   : {properties.name} "
          f"({properties.total_memory / BYTES_PER_GIB:,.1f} GiB)")
    print(f"  torch    : {torch.__version__}   cuda {torch.version.cuda}   "
          f"triton {_triton_version()}")
    print(f"  dtype    : {args.dtype}   N = {args.steps:,}   "
          f"repeats {args.repeats}")
    print(f"  Kernel   : BLOCK_M={block_m}, BLOCK_CK={block_ck}, "
          f"segments={-(-args.steps // block_ck)}")
    print(f"  Surface  : parametric tanh (NOT SSVI -- see the report notes)")
    print()

    smallest = min(args.paths)
    deviation = None
    if not args.no_cross_check:
        print(f"  Agreement check at M = {smallest:,} ...", end=" ", flush=True)
        deviation = cross_check(
            smallest, args.steps, dtype=dtype, seed=args.seed,
            max_programs=args.max_programs,
        )
        if deviation is None:
            print("skipped (out of memory)")
        else:
            scale = 1.0 / math.sqrt(smallest)
            print(f"max deviation {deviation:.3%} of peak EE "
                  f"(1/sqrt(M) scale {scale:.3%})")
            if deviation >= 20.0 * scale:
                print("    WARNING: larger than Monte-Carlo error explains. "
                      "The two backends may not be pricing the same thing; "
                      "treat the timings below with suspicion.")
        print()

    rows: List[SweepRow] = []
    for n_paths in sorted(set(args.paths)):
        row = SweepRow(n_paths=n_paths)
        print(f"  M = {n_paths:,}")
        for backend in BACKENDS:
            result = measure_backend(
                backend, n_paths, args.steps, dtype=dtype,
                repeats=args.repeats, seed=args.seed,
                max_programs=args.max_programs,
            )
            row.results[backend] = result

            def describe(stage) -> str:
                """One-line verdict for a stage."""
                if stage.predicted_oom:
                    return f"refused (~{format_bytes(stage.predicted_bytes)})"
                if stage.failed_oom:
                    return "OOM"
                return f"{stage.ms:,.1f} ms / {format_bytes(stage.peak)}"

            print(f"    {backend:<20} fwd {describe(result.forward):<24} "
                  f"bwd {describe(result.backward)}")
        rows.append(row)

    report = build_markdown(
        rows, n_steps=args.steps, dtype=dtype, repeats=args.repeats,
        deviation=deviation,
        cross_check_paths=None if deviation is None else smallest,
    )
    print()
    print("=" * 80)
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
