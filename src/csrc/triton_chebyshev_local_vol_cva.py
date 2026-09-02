r"""Chebyshev local-volatility exposure kernel: fixes the tanh kernel's wing saturation.

Companion to :mod:`src.csrc.triton_local_vol_cva`. That kernel's
:math:`\sigma(t,x) = \sigma_0 + \sigma_{\text{skew}}\tanh(\kappa(x-x_{\text{ref}}))
+ \sigma_{\text{term}}t` is bounded by construction: past a few multiples of
:math:`1/\kappa` the tanh is flat, and a Dupire surface implied by SSVI has no
reason to flatten there too. Measured on an SSVI surface with
:math:`\rho=-0.35,\eta=1.2,\gamma=0.45` at a 3-sigma sampling width, the tanh
fit reaches only :math:`R^2=0.578` (relative RMSE 15.99%). This kernel replaces
the spatial term with a degree-:math:`K` Chebyshev expansion, which has no
saturation ceiling: the same surface fits to :math:`R^2=0.902` (relative RMSE
7.71%) at :math:`K=8`, verified in
``tests/test_chebyshev_local_vol.py::TestWingSaturationFix``.

.. math::
    \sigma(t, x) = \max\!\Bigl(\sum_{k=0}^{K} c_k\, T_k(u),\ \text{floor}\Bigr)
                 + \sigma_{\text{term}}\, t,
    \qquad u = \frac{x - x_{\text{ref}}}{w}

with :math:`T_k` the Chebyshev polynomials of the first kind, :math:`w` a
fixed domain half-width, and the floor a strictly-positive lower clamp on the
spatial term (never on the whole surface, so the linear-in-time correction
still applies past the floor -- the same convention as
:func:`src.models.vol_surface.evaluate_chebyshev_local_vol`, which this kernel
must reproduce exactly).

Two-tier verification, following this project's convention for every Triton
kernel it has written
(see ``src/csrc/triton_local_vol_cva.py``'s own module docstring for the
precedent):

============  ===========================================================
tier          status
============  ===========================================================
CPU (Tier 1)  Fully verified. :func:`reference_chebyshev_local_vol_ee` is
              plain ``torch``, differentiated by ``torch.autograd`` --
              ground truth. :func:`reference_chebyshev_local_vol_ee_adjoint`
              (full storage) and
              :func:`reference_checkpointed_chebyshev_ee_adjoint`
              (sqrt(N) checkpointing) are both hand-derived and checked
              against it in ``tests/test_chebyshev_local_vol.py``.
GPU (Tier 2)  Compiled once on a T4 (Triton 3.6.0) and **failed**, then
              fixed; see the note below. Written without a local Triton
              install, so every construct is chosen to match
              ``_fused_local_vol_forward_kernel`` /
              ``_fused_local_vol_backward_kernel`` primitive-for-primitive
              (``tl.where``, ``tl.exp``, ``tl.randn``, ``tl.load``,
              ``tl.maximum``, ``tl.sum`` -- nothing beyond what those two
              kernels already prove compiles on the target).
============  ===========================================================

What the first compile taught us: ``range`` is not unrolled
===========================================================
An earlier version of this file assumed that ``for k in range(...)`` inside
``@triton.jit`` unrolls at trace time and yields a Python ``int`` for ``k``
whenever the bound is a ``tl.constexpr``. **That is false.** Triton lowers
``range`` to a *runtime* loop and ``k`` becomes a ``tl.tensor``. Only
``tl.static_range`` unrolls. Two things broke as a result:

* ``float(k)`` raised ``TypeError: float() argument must be a string or a
  real number, not 'tensor'`` -- the reported compile failure.
* ``acc_coeff[k]``, a Python list of per-coefficient accumulators indexed by
  the loop variable, would have failed identically the moment the forward
  compiled.

Both are fixed without ``tl.static_range`` (unverifiable from the development
machine, and this project has already lost a round trip to assuming a
``tl.*`` symbol exists). Instead:

* the factor :math:`k` is folded into the coefficients host-side, so the
  derivative sum carries no :math:`k` arithmetic at all
  (:func:`_chebyshev_eval_and_deriv`);
* the accumulator list became a masked ``(BLOCK_M, BLOCK_DEG)`` tile, which
  is exactly the pattern the proven tanh kernel already uses for its
  ``checkpoints`` and ``replay`` tiles, and for the same reason;
* the value-only :func:`_chebyshev_eval` uses ``k`` for nothing but
  ``tl.load(coeff_ptr + k)``, which is legal with a runtime offset.

The remaining use of ``k`` -- ``offs_deg[None, :] == k`` -- is a tensor
comparison, which is the intended way to select a column by a runtime index.

Because the recursion, the checkpointing scheme, and the packed-parameter
convention are copied from a kernel already proven on a T4, a GPU-side
failure here is expected to be a Chebyshev-specific translation error, not a
scaffolding one -- the same argument that let every previous phase's
CPU/GPU split localise its own first-contact bugs quickly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.csrc.triton_gbm import HAS_TRITON, is_available
from src.csrc.triton_philox_gbm import validate_offset_scheme
from src.models.vol_surface import chebyshev_basis, chebyshev_basis_derivative
from src.xva.exposure import validate_uniform_grid

__all__ = [
    "ChebyshevLocalVolParams",
    "select_chebyshev_local_vol_blocks",
    "chebyshev_block_degree",
    "chebyshev_local_vol_and_state_derivative",
    "reference_chebyshev_local_vol_ee",
    "reference_chebyshev_local_vol_ee_adjoint",
    "reference_checkpointed_chebyshev_ee_adjoint",
    "FusedChebyshevLocalVolCVAFunction",
    "fused_chebyshev_local_vol_ee",
    "fused_chebyshev_local_vol_cva",
    "is_available",
]

try:  # pragma: no cover - depends on the host having Triton
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - reuse Phase 3's stubs
    from src.csrc.triton_gbm import tl, triton  # type: ignore[attr-defined]


#: Per-program SRAM budget, matching the tanh kernel's choice.
SRAM_TILE_BUDGET_BYTES = 16 * 1024

#: Widest Chebyshev degree the kernel will unroll. Not a mathematical limit --
#: a guard against launching a kernel whose unrolled instruction count would
#: make compilation impractical. 32 covers every degree exercised in testing
#: (fit quality plateaus well below this; see the module docstring's R^2
#: table in ``fit_local_vol_params``'s own docstring).
MAX_CHEBYSHEV_DEGREE = 32


@dataclass(frozen=True)
class ChebyshevLocalVolParams:
    r"""Parameters of the Chebyshev local-volatility surface.

    Produced by
    :meth:`src.models.vol_surface.LocalVolFit.to_local_vol_params` when the
    fit used ``basis="chebyshev"``.

    Attributes:
        coefficients: :math:`c_0, \dots, c_K`. Every coefficient is
            differentiable in principle; the kernel below differentiates all
            of them, unlike the tanh kernel which holds :math:`\kappa`
            constant -- there is no analogous "shape" parameter entering the
            Jacobian nonlinearly here, since :math:`\partial\sigma/\partial x`
            is linear in the coefficients.
        half_width: :math:`w`, the domain half-width standardising
            :math:`x` into the Chebyshev argument :math:`u`. Held constant --
            see :func:`src.models.vol_surface.fit_local_vol_params`.
        reference: :math:`x_{\text{ref}}`, pinned to :math:`\log S_0`, as in
            the tanh kernel.
        term: :math:`\sigma_{\text{term}}`, linear term-structure slope. Held
            constant, as in the tanh kernel.
        floor: Lower clamp on the spatial (Chebyshev) term before the linear
            time correction is added.
    """

    coefficients: Tuple[float, ...]
    half_width: float
    reference: float
    term: float = 0.0
    floor: float = 1e-4

    def __post_init__(self) -> None:
        if len(self.coefficients) == 0:
            raise ValueError("need at least one Chebyshev coefficient")
        if len(self.coefficients) - 1 > MAX_CHEBYSHEV_DEGREE:
            raise ValueError(
                f"degree {len(self.coefficients) - 1} exceeds "
                f"MAX_CHEBYSHEV_DEGREE={MAX_CHEBYSHEV_DEGREE}"
            )
        if not all(math.isfinite(c) for c in self.coefficients):
            raise ValueError("coefficients must be finite")
        if self.half_width <= 0.0:
            raise ValueError(f"half_width must be positive, got {self.half_width}")
        if self.floor <= 0.0:
            raise ValueError(f"floor must be positive, got {self.floor}")

    @property
    def degree(self) -> int:
        """Highest Chebyshev order :math:`K`."""
        return len(self.coefficients) - 1


def select_chebyshev_local_vol_blocks(
    n_steps: int, element_size: int, degree: int = 0
) -> Tuple[int, int]:
    r"""Choose ``(BLOCK_M, BLOCK_CK)``, identical rule to the tanh kernel.

    Kept as a separate function (rather than imported) so this module has no
    hard import-time dependency on the tanh kernel file; the selection rule
    itself must stay in sync by inspection, which is cheap since it is four
    lines of arithmetic on ``n_steps`` alone -- see
    ``triton_local_vol_cva.select_local_vol_blocks`` for the derivation.

    Args:
        n_steps: Number of time steps :math:`N`.
        element_size: Bytes per element.
        degree: Chebyshev degree, which sets the width of the two extra
            coefficient tiles the backward pass holds.

    Returns:
        ``(BLOCK_M, BLOCK_CK)``.
    """
    block_ck = 1
    while block_ck * block_ck < n_steps:
        block_ck *= 2
    block_ck = max(block_ck, 1)
    # Four (BLOCK_M, *) tiles are live in the backward pass: checkpoints and
    # replay (BLOCK_CK wide), plus the coefficient accumulator and the
    # Chebyshev basis tile (BLOCK_DEG wide). The tanh kernel budgeted for two;
    # under-counting here would silently cost occupancy rather than fail.
    block_deg = chebyshev_block_degree(degree)
    tile_bytes = 2 * (block_ck + block_deg) * element_size
    block_m = max(1, SRAM_TILE_BUDGET_BYTES // max(tile_bytes, 1))
    block_m = 1 << (block_m.bit_length() - 1) if block_m > 0 else 1
    return block_m, block_ck


def chebyshev_block_degree(degree: int) -> int:
    """Next power of two at or above ``degree + 1``.

    ``tl.arange`` requires a power-of-two length, so the coefficient axis is
    padded and the tail masked off -- the same treatment ``BLOCK_CK`` gets.

    Args:
        degree: Chebyshev degree :math:`K`.

    Returns:
        ``BLOCK_DEG``, at least 1.

    Raises:
        ValueError: If ``degree`` is negative.
    """
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    block_deg = 1
    while block_deg < degree + 1:
        block_deg *= 2
    return block_deg


# ==========================================================================
# CPU reference: forward
# ==========================================================================
def chebyshev_local_vol_and_state_derivative(
    log_spot: Tensor,
    coefficients: Tensor,
    half_width: float,
    reference: float,
    term: float,
    time: float,
    *,
    floor: float = 1e-4,
) -> Tuple[Tensor, Tensor]:
    r"""Evaluate :math:`\sigma(t,x)` and :math:`\partial\sigma/\partial x` together.

    Both quantities need the same Chebyshev basis and derivative evaluation,
    so they are computed once here rather than twice -- exactly the reason
    the tanh kernel's analogous function exists.

    .. math::
        \frac{\partial\sigma}{\partial x} = \frac{1}{w}\sum_{k=0}^{K} c_k T_k'(u)
        \quad\text{where the sum is active, else } 0

    (the floor clamp makes the spatial term locally constant wherever it
    binds, so its derivative there is zero -- not the polynomial's derivative,
    which is what a naive ``torch.autograd`` through the unclamped sum would
    give if the clamp were applied a
    fter differentiating instead of before).

    Args:
        log_spot: :math:`x`, any shape.
        coefficients: :math:`c_0, \dots, c_K`, shape ``(K + 1,)``.
        half_width: :math:`w`.
        reference: :math:`x_{\text{ref}}`.
        term: :math:`\sigma_{\text{term}}`.
        time: :math:`t`, broadcastable with ``log_spot``.
        floor: Lower clamp on the spatial term.

    Returns:
        ``(sigma, dsigma_dx)``, both the broadcast shape of ``log_spot``.
    """
    degree = coefficients.numel() - 1
    u = (log_spot - reference) / half_width
    basis = chebyshev_basis(u, degree)
    spatial = torch.tensordot(coefficients, basis, dims=([0], [0]))
    active = (spatial > floor).to(log_spot.dtype)

    derivative_basis = chebyshev_basis_derivative(u, degree)
    d_spatial_du = torch.tensordot(coefficients, derivative_basis, dims=([0], [0]))
    d_sigma_dx = active * d_spatial_du / half_width

    sigma = torch.clamp(spatial, min=floor) + term * time
    return sigma, d_sigma_dx


def reference_chebyshev_local_vol_ee(
    spot_zero: Tensor,
    drift: Tensor,
    coefficients: Tensor,
    term: Tensor,
    normals: Tensor,
    dt: float,
    coeff_b: Tensor,
    coeff_c: Tensor,
    half_width: float,
    reference: float,
    *,
    floor: float = 1e-4,
) -> Tensor:
    r"""Differentiable forward: EE profile under Chebyshev local volatility.

    Ordinary ``torch``, matching
    :func:`src.csrc.triton_local_vol_cva.reference_local_vol_ee`'s structure
    exactly except for the volatility formula, so this is the ground truth
    the hand-derived adjoints below are checked against.

    Args:
        spot_zero: :math:`S_0`, 0-dim.
        drift: :math:`\mu - q`, 0-dim.
        coefficients: :math:`c_0, \dots, c_K`, shape ``(K + 1,)``.
        term: :math:`\sigma_{\text{term}}`, 0-dim.
        normals: :math:`Z`, shape ``(n_paths, n_steps)``.
        dt: Step size.
        coeff_b: Affine payoff coefficient, shape ``(n_steps + 1,)``.
        coeff_c: Affine payoff coefficient, shape ``(n_steps + 1,)``.
        half_width: :math:`w`.
        reference: :math:`x_{\text{ref}}`.
        floor: Lower clamp on the spatial term.

    Returns:
        EE profile, shape ``(n_steps + 1,)``.
    """
    n_paths, n_steps = normals.shape
    sqrt_dt = math.sqrt(dt)
    degree = coefficients.numel() - 1

    log_spot = torch.log(spot_zero).expand(n_paths)
    columns = [log_spot]
    for step in range(n_steps):
        u = (log_spot - reference) / half_width
        basis = chebyshev_basis(u, degree)
        spatial = torch.tensordot(coefficients, basis, dims=([0], [0]))
        sigma = torch.clamp(spatial, min=floor) + term * (step * dt)
        log_spot = (
            log_spot
            + (drift - 0.5 * sigma * sigma) * dt
            + sigma * sqrt_dt * normals[:, step]
        )
        columns.append(log_spot)

    spots = torch.exp(torch.stack(columns, dim=1))
    mtm = coeff_b.reshape(1, -1) * spots - coeff_c.reshape(1, -1)
    return torch.clamp(mtm, min=0.0).mean(dim=0)


# ==========================================================================
# CPU reference: adjoints
# ==========================================================================
def _direct_chebyshev_state_adjoint(
    log_spot: Tensor,
    step_index: int,
    weight: Tensor,
    coeff_b: Tensor,
    coeff_c: Tensor,
) -> Tensor:
    r"""The EE output's direct adjoint of the state at one column.

    Identical in form to the tanh file's ``_direct_state_adjoint`` -- the
    payoff structure does not depend on the volatility model, so this part of
    the adjoint is shared conceptually (though re-implemented here to keep
    this module import-independent of the tanh one).
    """
    spot = torch.exp(log_spot)
    mtm = coeff_b[step_index] * spot - coeff_c[step_index]
    active = (mtm > 0.0).to(log_spot.dtype)
    return weight[step_index] * coeff_b[step_index] * spot * active


def reference_chebyshev_local_vol_ee_adjoint(
    grad_ee: Tensor,
    spot_zero: Tensor,
    drift: Tensor,
    coefficients: Tensor,
    term: Tensor,
    normals: Tensor,
    dt: float,
    coeff_b: Tensor,
    coeff_c: Tensor,
    half_width: float,
    reference: float,
    *,
    floor: float = 1e-4,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Sequential adjoint with the full trajectory stored.

    The reverse recursion is identical in *shape* to the tanh kernel's:
    :math:`a_k = \bar X_k + a_{k+1} J_k` with
    :math:`J_k = 1 + (\partial\sigma_k/\partial X_k)(\sqrt{\Delta t}Z_k -
    \sigma_k\Delta t)`. Only :math:`\partial\sigma_k/\partial X_k` and the
    parameter-gradient terms change, since :math:`\sigma` is now a Chebyshev
    sum in the coefficients rather than a tanh in ``base``/``skew``:
    :math:`\partial\sigma/\partial c_k = T_k(u)` wherever the floor does not
    bind, exactly analogous to :math:`\partial\sigma/\partial(\text{base})=1`
    in the tanh case.

    Args:
        grad_ee: Adjoint seed, shape ``(n_steps + 1,)``.
        spot_zero: :math:`S_0`, 0-dim.
        drift: :math:`\mu - q`, 0-dim.
        coefficients: :math:`c_0, \dots, c_K`, shape ``(K + 1,)``.
        term: :math:`\sigma_{\text{term}}`, 0-dim.
        normals: :math:`Z`, shape ``(n_paths, n_steps)``.
        dt: Step size.
        coeff_b: Affine payoff coefficient.
        coeff_c: Affine payoff coefficient.
        half_width: :math:`w`.
        reference: :math:`x_{\text{ref}}`.
        floor: Lower clamp on the spatial term.

    Returns:
        ``(d_s0, d_drift, d_coefficients, d_term)``.
    """
    n_paths, n_steps = normals.shape
    sqrt_dt = math.sqrt(dt)
    degree = coefficients.numel() - 1
    weight = grad_ee / n_paths

    log_spot = torch.log(spot_zero).expand(n_paths).clone()
    trajectory = [log_spot.clone()]
    for step in range(n_steps):
        sigma, _ = chebyshev_local_vol_and_state_derivative(
            log_spot, coefficients, half_width, reference, term, step * dt,
            floor=floor,
        )
        log_spot = (
            log_spot
            + (drift - 0.5 * sigma * sigma) * dt
            + sigma * sqrt_dt * normals[:, step]
        )
        trajectory.append(log_spot.clone())

    adjoint = _direct_chebyshev_state_adjoint(
        trajectory[n_steps], n_steps, weight, coeff_b, coeff_c
    )
    d_drift = torch.zeros_like(drift).expand(n_paths).clone()
    d_coefficients = torch.zeros(n_paths, degree + 1, dtype=coefficients.dtype)
    d_term = torch.zeros_like(term).expand(n_paths).clone()

    for step in range(n_steps - 1, -1, -1):
        state_k = trajectory[step]
        sigma, d_sigma_dx = chebyshev_local_vol_and_state_derivative(
            state_k, coefficients, half_width, reference, term, step * dt,
            floor=floor,
        )
        z = normals[:, step]
        vol_factor = sqrt_dt * z - sigma * dt

        u = (state_k - reference) / half_width
        basis = chebyshev_basis(u, degree)  # (K+1, n_paths)
        spatial = torch.tensordot(coefficients, basis, dims=([0], [0]))
        active = (spatial > floor).to(state_k.dtype)

        d_drift = d_drift + adjoint * dt
        # d_sigma/d_c_k = T_k(u) where the floor does not bind, else 0.
        d_coefficients = d_coefficients + (
            adjoint * vol_factor * active
        ).unsqueeze(-1) * basis.T
        d_term = d_term + adjoint * vol_factor * (step * dt)

        jacobian = 1.0 + d_sigma_dx * vol_factor
        direct = _direct_chebyshev_state_adjoint(
            state_k, step, weight, coeff_b, coeff_c
        )
        adjoint = direct + adjoint * jacobian

    d_s0 = (adjoint / spot_zero).sum()
    return (
        d_s0,
        d_drift.sum(),
        d_coefficients.sum(dim=0),
        d_term.sum(),
    )


def reference_checkpointed_chebyshev_ee_adjoint(
    grad_ee: Tensor,
    spot_zero: Tensor,
    drift: Tensor,
    coefficients: Tensor,
    term: Tensor,
    normals: Tensor,
    dt: float,
    coeff_b: Tensor,
    coeff_c: Tensor,
    half_width: float,
    reference: float,
    *,
    floor: float = 1e-4,
    checkpoint_stride: int = 8,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""The same adjoint, but replaying from sqrt(N)-spaced checkpoints.

    Mirrors ``reference_checkpointed_ee_adjoint`` in the tanh file: only
    segment-entry states are stored, and each segment is replayed forward
    before being walked backward. Checked against
    :func:`reference_chebyshev_local_vol_ee_adjoint` (full storage) in
    ``tests/test_chebyshev_local_vol.py`` -- agreement there is what licenses
    the Triton kernel's own checkpointing scheme, which uses the identical
    replay structure in SRAM instead of Python lists.

    Args:
        grad_ee: Adjoint seed.
        spot_zero: :math:`S_0`, 0-dim.
        drift: :math:`\mu - q`, 0-dim.
        coefficients: :math:`c_0, \dots, c_K`.
        term: :math:`\sigma_{\text{term}}`, 0-dim.
        normals: :math:`Z`, shape ``(n_paths, n_steps)``.
        dt: Step size.
        coeff_b: Affine payoff coefficient.
        coeff_c: Affine payoff coefficient.
        half_width: :math:`w`.
        reference: :math:`x_{\text{ref}}`.
        floor: Lower clamp on the spatial term.
        checkpoint_stride: Segment length (analogous to ``BLOCK_CK``).

    Returns:
        ``(d_s0, d_drift, d_coefficients, d_term)``.
    """
    n_paths, n_steps = normals.shape
    sqrt_dt = math.sqrt(dt)
    degree = coefficients.numel() - 1
    weight = grad_ee / n_paths

    def advance(state: Tensor, step: int) -> Tensor:
        sigma, _ = chebyshev_local_vol_and_state_derivative(
            state, coefficients, half_width, reference, term, step * dt,
            floor=floor,
        )
        return (
            state
            + (drift - 0.5 * sigma * sigma) * dt
            + sigma * sqrt_dt * normals[:, step]
        )

    # ---- pass 1: forward, recording only checkpoint-entry states ------
    checkpoints = {}
    state = torch.log(spot_zero).expand(n_paths).clone()
    for step in range(n_steps):
        if step % checkpoint_stride == 0:
            checkpoints[step] = state.clone()
        state = advance(state, step)
    terminal = state

    adjoint = _direct_chebyshev_state_adjoint(terminal, n_steps, weight, coeff_b, coeff_c)
    d_drift = torch.zeros_like(drift).expand(n_paths).clone()
    d_coefficients = torch.zeros(n_paths, degree + 1, dtype=coefficients.dtype)
    d_term = torch.zeros_like(term).expand(n_paths).clone()

    n_segments = -(-n_steps // checkpoint_stride)
    for segment in range(n_segments - 1, -1, -1):
        segment_start = segment * checkpoint_stride
        segment_end = min(segment_start + checkpoint_stride, n_steps)

        # ---- replay the segment forward from its checkpoint -----------
        replay = [checkpoints[segment_start]]
        state = checkpoints[segment_start]
        for step in range(segment_start, segment_end):
            state = advance(state, step)
            replay.append(state)

        # ---- walk the segment backward ---------------------------------
        for offset in range(segment_end - segment_start - 1, -1, -1):
            step = segment_start + offset
            state_k = replay[offset]
            sigma, d_sigma_dx = chebyshev_local_vol_and_state_derivative(
                state_k, coefficients, half_width, reference, term, step * dt,
                floor=floor,
            )
            z = normals[:, step]
            vol_factor = sqrt_dt * z - sigma * dt

            u = (state_k - reference) / half_width
            basis = chebyshev_basis(u, degree)
            spatial = torch.tensordot(coefficients, basis, dims=([0], [0]))
            active = (spatial > floor).to(state_k.dtype)

            d_drift = d_drift + adjoint * dt
            d_coefficients = d_coefficients + (
                adjoint * vol_factor * active
            ).unsqueeze(-1) * basis.T
            d_term = d_term + adjoint * vol_factor * (step * dt)

            jacobian = 1.0 + d_sigma_dx * vol_factor
            direct = _direct_chebyshev_state_adjoint(
                state_k, step, weight, coeff_b, coeff_c
            )
            adjoint = direct + adjoint * jacobian

    d_s0 = (adjoint / spot_zero).sum()
    return (
        d_s0,
        d_drift.sum(),
        d_coefficients.sum(dim=0),
        d_term.sum(),
    )


# ==========================================================================
# Triton device code -- NEVER COMPILED, see the module docstring
# ==========================================================================
@triton.jit
def _chebyshev_eval(u, coeff_ptr, degree: tl.constexpr):
    r"""Evaluate :math:`\sum_k c_k T_k(u)` only -- no derivative.

    Three of this file's four evaluation sites (the forward pass, and the
    backward pass's two forward replays) discard the derivative, so computing
    it there was pure waste: at :math:`N = 252` steps it ran the entire
    :math:`U` recurrence per step and threw the result away.

    Uses ``k`` for nothing but ``tl.load(coeff_ptr + k)``, which is legal with
    a runtime offset -- the same pointer arithmetic the proven tanh kernel
    does with ``tl.load(coeff_b_ptr + column)``. So this function contains no
    construct that depends on ``k`` being a compile-time value.

    Args:
        u: Standardised coordinate tile.
        coeff_ptr: Pointer to :math:`c_0, \dots, c_K`, contiguous.
        degree: :math:`K`, a compile-time constant.

    Returns:
        ``spatial``, same shape as ``u``.
    """
    # "ones" shaped like `u` via elementwise arithmetic on `u` itself: this
    # function is shape-agnostic, and Triton's zeros/ones constructors need an
    # explicit compile-time shape rather than "like another tensor".
    t_prev = 0.0 * u + 1.0  # T_0
    spatial = tl.load(coeff_ptr + 0) * t_prev

    if degree >= 1:
        t_curr = u  # T_1
        spatial = spatial + tl.load(coeff_ptr + 1) * t_curr

        for k in range(2, degree + 1):
            t_next = 2.0 * u * t_curr - t_prev
            spatial = spatial + tl.load(coeff_ptr + k) * t_next
            t_prev, t_curr = t_curr, t_next

    return spatial


@triton.jit
def _chebyshev_eval_and_deriv(
    u, coeff_ptr, deriv_coeff_ptr, degree: tl.constexpr
):
    r"""Evaluate :math:`\sum_k c_k T_k(u)` and its :math:`u`-derivative.

    The derivative uses :math:`T_k'(u) = k\,U_{k-1}(u)` with :math:`U` the
    Chebyshev polynomials of the second kind. The factor :math:`k` is **not
    computed here**: the host passes :math:`\hat c_k = k\,c_k` in
    ``deriv_coeff_ptr``, so the sum is :math:`\sum_k \hat c_k U_{k-1}(u)`
    with no :math:`k` arithmetic in the kernel at all.

    That matters for a concrete reason. ``for k in range(...)`` inside
    ``@triton.jit`` does **not** unroll at trace time -- Triton lowers it to a
    runtime loop and ``k`` becomes a ``tl.tensor``, so the original
    ``float(k)`` raised ``TypeError: float() argument must be ... not
    'tensor'`` on first compile. Folding :math:`k` into the coefficient
    sidesteps the question entirely rather than relying on int-to-float
    promotion, and removes one multiply per term. Verified on CPU against
    :func:`src.models.vol_surface.chebyshev_basis_derivative`.

    Args:
        u: Standardised coordinate tile.
        coeff_ptr: Pointer to :math:`c_0, \dots, c_K`, contiguous.
        deriv_coeff_ptr: Pointer to :math:`0, 1c_1, 2c_2, \dots, Kc_K`.
        degree: :math:`K`, a compile-time constant.

    Returns:
        ``(spatial, d_spatial_du)``, same shape as ``u``.
    """
    ones = 0.0 * u + 1.0
    t_prev = ones  # T_0
    spatial = tl.load(coeff_ptr + 0) * t_prev
    d_spatial_du = 0.0 * u  # T_0' = 0

    if degree >= 1:
        t_curr = u  # T_1
        spatial = spatial + tl.load(coeff_ptr + 1) * t_curr

        u_prev = ones  # U_0
        # T_1' = 1 * U_0, and the 1 is already folded into deriv_coeff[1].
        d_spatial_du = d_spatial_du + tl.load(deriv_coeff_ptr + 1) * u_prev

    if degree >= 2:
        u_curr = 2.0 * u  # U_1
        t_next = 2.0 * u * t_curr - t_prev  # T_2
        spatial = spatial + tl.load(coeff_ptr + 2) * t_next
        # T_2' = 2 * U_1; the 2 is folded into deriv_coeff[2].
        d_spatial_du = d_spatial_du + tl.load(deriv_coeff_ptr + 2) * u_curr
        t_prev, t_curr = t_curr, t_next

        for k in range(3, degree + 1):
            t_next = 2.0 * u * t_curr - t_prev
            spatial = spatial + tl.load(coeff_ptr + k) * t_next

            u_next = 2.0 * u * u_curr - u_prev
            d_spatial_du = (
                d_spatial_du + tl.load(deriv_coeff_ptr + k) * u_next
            )

            t_prev, t_curr = t_curr, t_next
            u_prev, u_curr = u_curr, u_next

    return spatial, d_spatial_du


@triton.jit
def _fused_chebyshev_local_vol_forward_kernel(
    params_ptr,
    coeff_ptr,
    coeff_b_ptr,
    coeff_c_ptr,
    partial_ee_ptr,
    seed,
    n_paths,
    n_columns,
    n_steps,
    pe_stride_p,
    pe_stride_k,
    DEGREE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    r"""Sequential-time forward under Chebyshev local volatility.

    Structurally identical to
    ``triton_local_vol_cva._fused_local_vol_forward_kernel`` -- same
    per-segment tile accumulation, same masked column writes -- with the
    volatility evaluation swapped for :func:`_chebyshev_eval_and_deriv`.

    Args:
        params_ptr: Packed ``[s0, drift, half_width, reference, term, floor,
            dt, sqrt_dt, log_s0]`` in the working dtype.
        coeff_ptr: Packed ``[c_0, ..., c_DEGREE]``, separate from
            ``params_ptr`` since its length varies with ``DEGREE``.
        coeff_b_ptr: Affine coefficient :math:`B`, shape ``(n_columns,)``.
        coeff_c_ptr: Affine coefficient :math:`C`, shape ``(n_columns,)``.
        partial_ee_ptr: Output partial sums, ``(n_programs, n_columns)``.
        seed: Base Philox key.
        n_paths: Number of paths :math:`M`.
        n_columns: ``n_steps + 1``.
        n_steps: Number of time steps :math:`N`.
        pe_stride_p: Program-row stride of ``partial_ee_ptr``.
        pe_stride_k: Column stride of ``partial_ee_ptr``.
        DEGREE: Chebyshev degree :math:`K`, compile-time.
        BLOCK_M: Paths per tile.
        BLOCK_CK: Segment length.
        DTYPE: Working element type.
    """
    pid = tl.program_id(axis=0)
    n_programs = tl.num_programs(axis=0)

    local_m = tl.arange(0, BLOCK_M)
    offs_ck = tl.arange(0, BLOCK_CK)

    s0 = tl.load(params_ptr + 0)
    drift = tl.load(params_ptr + 1)
    half_width = tl.load(params_ptr + 2)
    reference = tl.load(params_ptr + 3)
    term = tl.load(params_ptr + 4)
    floor = tl.load(params_ptr + 5)
    dt = tl.load(params_ptr + 6)
    sqrt_dt = tl.load(params_ptr + 7)
    log_s0 = tl.load(params_ptr + 8)
    zeros_m = tl.zeros([BLOCK_M], dtype=DTYPE)
    zeros_tile = tl.zeros([BLOCK_M, BLOCK_CK], dtype=DTYPE)

    n_segments = (n_steps + BLOCK_CK - 1) // BLOCK_CK
    n_blocks = (n_paths + BLOCK_M - 1) // BLOCK_M

    for block_index in range(pid, n_blocks, n_programs):
        offs_m = block_index * BLOCK_M + local_m
        mask_m = offs_m < n_paths
        program_seed = seed + block_index

        state = log_s0 + zeros_m

        b0 = tl.load(coeff_b_ptr)
        c0 = tl.load(coeff_c_ptr)
        spot0 = tl.exp(state)
        exposure0 = tl.where(mask_m, tl.maximum(b0 * spot0 - c0, 0.0), zeros_m)
        current0 = tl.load(partial_ee_ptr + pid.to(tl.int64) * pe_stride_p)
        tl.store(
            partial_ee_ptr + pid.to(tl.int64) * pe_stride_p,
            current0 + tl.sum(exposure0, axis=0),
        )

        for segment in range(0, n_segments):
            exposure_tile = zeros_tile

            for offset in range(0, BLOCK_CK):
                step = segment * BLOCK_CK + offset
                live = step < n_steps

                rng_offset = (local_m * n_columns + step).to(tl.int32)
                z = tl.randn(program_seed, rng_offset).to(DTYPE)

                u = (state - reference) / half_width
                spatial = _chebyshev_eval(u, coeff_ptr, DEGREE)
                sigma = tl.maximum(spatial, floor) + term * (step * dt)

                advanced = (
                    state
                    + (drift - 0.5 * sigma * sigma) * dt
                    + sigma * sqrt_dt * z
                )
                state = tl.where(live, advanced, state)

                column = step + 1
                coeff_b = tl.load(coeff_b_ptr + column, mask=live, other=0.0)
                coeff_c = tl.load(coeff_c_ptr + column, mask=live, other=0.0)
                spot = tl.exp(state)
                exposure = tl.maximum(coeff_b * spot - coeff_c, 0.0)
                exposure = tl.where(mask_m & live, exposure, zeros_m)

                exposure_tile = tl.where(
                    offs_ck[None, :] == offset, exposure[:, None], exposure_tile
                )

            segment_sum = tl.sum(exposure_tile, axis=0)
            columns = segment * BLOCK_CK + 1 + offs_ck
            mask_col = columns < n_columns
            addresses = (
                partial_ee_ptr
                + pid.to(tl.int64) * pe_stride_p
                + columns.to(tl.int64) * pe_stride_k
            )
            existing = tl.load(addresses, mask=mask_col, other=0.0)
            tl.store(addresses, existing + segment_sum, mask=mask_col)


@triton.jit
def _fused_chebyshev_local_vol_backward_kernel(
    params_ptr,
    coeff_ptr,
    deriv_coeff_ptr,
    coeff_b_ptr,
    coeff_c_ptr,
    weight_ptr,
    partial_s0_ptr,
    partial_drift_ptr,
    partial_coeff_ptr,
    partial_term_ptr,
    seed,
    n_paths,
    n_columns,
    n_steps,
    pc_stride_p,
    DEGREE: tl.constexpr,
    BLOCK_DEG: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_CK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    r"""Checkpointed adjoint under Chebyshev local volatility.

    Same two-pass structure (record checkpoints, then replay-and-walk-back
    per segment) as
    ``triton_local_vol_cva._fused_local_vol_backward_kernel``. The Jacobian
    and the parameter-gradient terms use
    :func:`_chebyshev_eval_and_deriv` in place of the tanh formulas; the
    per-coefficient gradient (one accumulator per :math:`c_k`, built as a
    Python list over ``range(DEGREE + 1)``) is the pattern flagged as
    untested in the module docstring.

    Args:
        params_ptr: Packed parameters, as in the forward kernel.
        coeff_ptr: Packed :math:`c_0, \dots, c_K`.
        deriv_coeff_ptr: Packed :math:`0, 1c_1, \dots, Kc_K` -- the factor
            :math:`k` folded in host-side, so no :math:`k` arithmetic happens
            in the kernel (see :func:`_chebyshev_eval_and_deriv`).
        coeff_b_ptr: Affine coefficient :math:`B`.
        coeff_c_ptr: Affine coefficient :math:`C`.
        weight_ptr: :math:`\omega_k = \texttt{grad\_ee}_k / M`.
        partial_s0_ptr: Per-program partials for ``dL/ds0``.
        partial_drift_ptr: Per-program partials for ``dL/ddrift``.
        partial_coeff_ptr: Per-program partials for ``dL/dc_k``, shape
            ``(n_programs, DEGREE + 1)``.
        partial_term_ptr: Per-program partials for ``dL/dterm``.
        seed: The **same** base key the forward used.
        n_paths: Number of paths :math:`M`.
        n_columns: ``n_steps + 1``.
        n_steps: Number of time steps :math:`N`.
        pc_stride_p: Program-row stride of ``partial_coeff_ptr``.
        DEGREE: Chebyshev degree, compile-time.
        BLOCK_DEG: Width of the coefficient axis, the next power of two at or
            above ``DEGREE + 1`` (``tl.arange`` requires a power of two).
        BLOCK_M: Paths per tile. MUST equal the forward's value.
        BLOCK_CK: Segment length.
        DTYPE: Working element type.
    """
    pid = tl.program_id(axis=0)
    n_programs = tl.num_programs(axis=0)

    local_m = tl.arange(0, BLOCK_M)
    offs_ck = tl.arange(0, BLOCK_CK)

    s0 = tl.load(params_ptr + 0)
    drift = tl.load(params_ptr + 1)
    half_width = tl.load(params_ptr + 2)
    reference = tl.load(params_ptr + 3)
    term = tl.load(params_ptr + 4)
    floor = tl.load(params_ptr + 5)
    dt = tl.load(params_ptr + 6)
    sqrt_dt = tl.load(params_ptr + 7)
    log_s0 = tl.load(params_ptr + 8)
    zeros_m = tl.zeros([BLOCK_M], dtype=DTYPE)
    zeros_tile = tl.zeros([BLOCK_M, BLOCK_CK], dtype=DTYPE)

    n_segments = (n_steps + BLOCK_CK - 1) // BLOCK_CK
    n_blocks = (n_paths + BLOCK_M - 1) // BLOCK_M

    acc_s0 = zeros_m
    acc_drift = zeros_m
    acc_term = zeros_m
    # One accumulator column per coefficient, as a TILE rather than a Python
    # list. `for k in range(...)` inside @triton.jit lowers to a runtime loop
    # and `k` becomes a tl.tensor, so `acc_coeff[k]` would index a Python list
    # with a tensor. Masked tile access is what the proven tanh kernel uses for
    # precisely this reason (see its `checkpoints`/`replay` handling).
    offs_deg = tl.arange(0, BLOCK_DEG)
    deg_live = offs_deg < (DEGREE + 1)
    acc_coeff = tl.zeros([BLOCK_M, BLOCK_DEG], dtype=DTYPE)

    for block_index in range(pid, n_blocks, n_programs):
        offs_m = block_index * BLOCK_M + local_m
        mask_m = offs_m < n_paths
        program_seed = seed + block_index

        checkpoints = zeros_tile
        state = log_s0 + zeros_m

        for segment in range(0, n_segments):
            checkpoints = tl.where(
                offs_ck[None, :] == segment, state[:, None], checkpoints
            )
            for offset in range(0, BLOCK_CK):
                step = segment * BLOCK_CK + offset
                live = step < n_steps
                rng_offset = (local_m * n_columns + step).to(tl.int32)
                z = tl.randn(program_seed, rng_offset).to(DTYPE)
                u = (state - reference) / half_width
                spatial = _chebyshev_eval(u, coeff_ptr, DEGREE)
                sigma = tl.maximum(spatial, floor) + term * (step * dt)
                advanced = (
                    state
                    + (drift - 0.5 * sigma * sigma) * dt
                    + sigma * sqrt_dt * z
                )
                state = tl.where(live, advanced, state)

        weight_n = tl.load(weight_ptr + n_steps)
        coeff_b_n = tl.load(coeff_b_ptr + n_steps)
        coeff_c_n = tl.load(coeff_c_ptr + n_steps)
        spot = tl.exp(state)
        active = (coeff_b_n * spot - coeff_c_n) > 0.0
        adjoint = tl.where(
            mask_m & active, weight_n * coeff_b_n * spot, zeros_m
        )

        for reverse_index in range(0, n_segments):
            segment = n_segments - 1 - reverse_index
            entry = tl.sum(
                tl.where(offs_ck[None, :] == segment, checkpoints, 0.0), axis=1
            )

            replay = zeros_tile
            state = entry
            for offset in range(0, BLOCK_CK):
                replay = tl.where(
                    offs_ck[None, :] == offset, state[:, None], replay
                )
                step = segment * BLOCK_CK + offset
                live = step < n_steps
                rng_offset = (local_m * n_columns + step).to(tl.int32)
                z = tl.randn(program_seed, rng_offset).to(DTYPE)
                u = (state - reference) / half_width
                spatial = _chebyshev_eval(u, coeff_ptr, DEGREE)
                sigma = tl.maximum(spatial, floor) + term * (step * dt)
                advanced = (
                    state
                    + (drift - 0.5 * sigma * sigma) * dt
                    + sigma * sqrt_dt * z
                )
                state = tl.where(live, advanced, state)

            for reverse_offset in range(0, BLOCK_CK):
                offset = BLOCK_CK - 1 - reverse_offset
                step = segment * BLOCK_CK + offset
                live = step < n_steps

                state_k = tl.sum(
                    tl.where(offs_ck[None, :] == offset, replay, 0.0), axis=1
                )
                rng_offset = (local_m * n_columns + step).to(tl.int32)
                z = tl.randn(program_seed, rng_offset).to(DTYPE)

                u = (state_k - reference) / half_width
                spatial, d_spatial_du = _chebyshev_eval_and_deriv(
                    u, coeff_ptr, deriv_coeff_ptr, DEGREE
                )
                active_floor = spatial > floor
                sigma = tl.maximum(spatial, floor) + term * (step * dt)
                d_sigma_dx = tl.where(
                    active_floor, d_spatial_du / half_width, 0.0 * u
                )
                vol_factor = sqrt_dt * z - sigma * dt

                gate = mask_m & live
                acc_drift = acc_drift + tl.where(gate, adjoint * dt, zeros_m)
                acc_term = acc_term + tl.where(
                    gate, adjoint * vol_factor * (step * dt), zeros_m
                )

                # d_sigma/d_c_k = T_k(u) wherever the floor does not bind.
                # Build the whole T_k basis into a (BLOCK_M, BLOCK_DEG) tile by
                # masked write -- column k receives T_k -- then accumulate every
                # coefficient's contribution in one vectorised update.
                basis_tile = tl.zeros([BLOCK_M, BLOCK_DEG], dtype=DTYPE)
                t_prev = 0.0 * u + 1.0  # T_0
                basis_tile = tl.where(
                    offs_deg[None, :] == 0, t_prev[:, None], basis_tile
                )
                if DEGREE >= 1:
                    t_curr = u  # T_1
                    basis_tile = tl.where(
                        offs_deg[None, :] == 1, t_curr[:, None], basis_tile
                    )
                    for k in range(2, DEGREE + 1):
                        t_next = 2.0 * u * t_curr - t_prev
                        basis_tile = tl.where(
                            offs_deg[None, :] == k, t_next[:, None], basis_tile
                        )
                        t_prev, t_curr = t_curr, t_next

                # The floor gate and the path gate are scalars per path, so
                # they factor out of the degree axis entirely.
                coeff_weight = tl.where(
                    gate & active_floor, adjoint * vol_factor, zeros_m
                )
                acc_coeff = acc_coeff + coeff_weight[:, None] * basis_tile

                jacobian = 1.0 + d_sigma_dx * vol_factor
                weight_k = tl.load(weight_ptr + step, mask=live, other=0.0)
                coeff_b_k = tl.load(coeff_b_ptr + step, mask=live, other=0.0)
                coeff_c_k = tl.load(coeff_c_ptr + step, mask=live, other=0.0)
                spot_k = tl.exp(state_k)
                active_k = (coeff_b_k * spot_k - coeff_c_k) > 0.0
                direct = tl.where(
                    gate & active_k, weight_k * coeff_b_k * spot_k, zeros_m
                )

                updated = direct + adjoint * jacobian
                adjoint = tl.where(live, updated, adjoint)

        acc_s0 = acc_s0 + tl.where(mask_m, adjoint / s0, zeros_m)

    tl.store(partial_s0_ptr + pid, tl.sum(acc_s0, axis=0))
    tl.store(partial_drift_ptr + pid, tl.sum(acc_drift, axis=0))
    tl.store(partial_term_ptr + pid, tl.sum(acc_term, axis=0))
    tl.store(
        partial_coeff_ptr + pid.to(tl.int64) * pc_stride_p + offs_deg,
        tl.sum(acc_coeff, axis=0),
        mask=deg_live,
    )


# ==========================================================================
# autograd.Function
# ==========================================================================
class FusedChebyshevLocalVolCVAFunction(torch.autograd.Function):
    """Autograd wrapper for the Chebyshev local-volatility exposure kernels.

    Mirrors ``FusedLocalVolCVAFunction`` exactly. Returns the expected-exposure
    profile only; the credit integral composes on top in PyTorch.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        spot_zero: Tensor,
        drift: Tensor,
        coefficients: Tensor,
        term: Tensor,
        coeff_b: Tensor,
        coeff_c: Tensor,
        n_paths: int,
        dt: float,
        seed: int,
        max_programs: int,
        half_width: float,
        reference: float,
        floor: float,
    ) -> Tensor:
        """Compute the EE profile without materialising any path matrix."""
        if not HAS_TRITON:
            raise RuntimeError(
                "Triton is not installed. The Chebyshev local-volatility "
                "kernel requires Triton and a CUDA device."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA device is visible. The Chebyshev local-volatility "
                "kernel is GPU-only."
            )

        device, dtype = spot_zero.device, spot_zero.dtype
        n_columns = int(coeff_b.numel())
        n_steps = n_columns - 1
        degree = coefficients.numel() - 1
        if degree > MAX_CHEBYSHEV_DEGREE:
            raise ValueError(
                f"degree {degree} exceeds MAX_CHEBYSHEV_DEGREE="
                f"{MAX_CHEBYSHEV_DEGREE}"
            )

        block_m, block_ck = select_chebyshev_local_vol_blocks(
            n_steps, coeff_b.element_size(), degree
        )
        validate_offset_scheme(block_m, n_columns)

        n_blocks = (n_paths + block_m - 1) // block_m
        n_programs = min(n_blocks, max_programs)

        params = torch.stack((
            spot_zero.reshape(()),
            drift.reshape(()),
            torch.as_tensor(half_width, device=device, dtype=dtype),
            torch.as_tensor(reference, device=device, dtype=dtype),
            term.reshape(()),
            torch.as_tensor(floor, device=device, dtype=dtype),
            torch.as_tensor(dt, device=device, dtype=dtype),
            torch.as_tensor(math.sqrt(dt), device=device, dtype=dtype),
            torch.log(spot_zero.reshape(())),
        )).contiguous()
        packed_coefficients = coefficients.reshape(-1).contiguous()

        partial_ee = torch.zeros((n_programs, n_columns), device=device, dtype=dtype)
        dtype_tl = tl.float64 if dtype == torch.float64 else tl.float32

        _fused_chebyshev_local_vol_forward_kernel[(n_programs,)](
            params, packed_coefficients, coeff_b, coeff_c, partial_ee,
            seed, n_paths, n_columns, n_steps,
            partial_ee.stride(0), partial_ee.stride(1),
            DEGREE=degree, BLOCK_M=block_m, BLOCK_CK=block_ck, DTYPE=dtype_tl,
        )

        expected_exposure = partial_ee.sum(dim=0) / n_paths

        ctx.save_for_backward(params, packed_coefficients, coeff_b, coeff_c)
        ctx.n_paths = n_paths
        ctx.n_columns = n_columns
        ctx.n_steps = n_steps
        ctx.seed = seed
        ctx.block_m = block_m
        ctx.block_ck = block_ck
        ctx.n_programs = n_programs
        ctx.degree = degree
        return expected_exposure

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_ee: Tensor):  # type: ignore[override]
        """Run the checkpointed adjoint kernel."""
        params, packed_coefficients, coeff_b, coeff_c = ctx.saved_tensors
        device, dtype = coeff_b.device, coeff_b.dtype

        weight = (grad_ee.contiguous() / ctx.n_paths).contiguous()
        n_programs = ctx.n_programs
        degree = ctx.degree

        partial_s0 = torch.empty(n_programs, device=device, dtype=dtype)
        partial_drift = torch.empty(n_programs, device=device, dtype=dtype)
        partial_term = torch.empty(n_programs, device=device, dtype=dtype)
        partial_coeff = torch.zeros(
            (n_programs, degree + 1), device=device, dtype=dtype
        )
        dtype_tl = tl.float64 if dtype == torch.float64 else tl.float32

        # d_sigma/d_u needs T_k'(u) = k * U_{k-1}(u). Folding the factor k
        # into the coefficient here means the kernel never does arithmetic on
        # its loop index -- which inside @triton.jit is a runtime tensor, not
        # a Python int (the original `float(k)` failed on exactly this).
        orders = torch.arange(
            degree + 1, device=device, dtype=dtype
        )
        deriv_coefficients = (orders * packed_coefficients).contiguous()

        _fused_chebyshev_local_vol_backward_kernel[(n_programs,)](
            params, packed_coefficients, deriv_coefficients,
            coeff_b, coeff_c, weight,
            partial_s0, partial_drift, partial_coeff, partial_term,
            ctx.seed, ctx.n_paths, ctx.n_columns, ctx.n_steps,
            partial_coeff.stride(0),
            DEGREE=degree, BLOCK_DEG=chebyshev_block_degree(degree),
            BLOCK_M=ctx.block_m, BLOCK_CK=ctx.block_ck,
            DTYPE=dtype_tl,
        )

        needs = ctx.needs_input_grad
        return (
            partial_s0.sum() if needs[0] else None,
            partial_drift.sum() if needs[1] else None,
            partial_coeff.sum(dim=0) if needs[2] else None,
            partial_term.sum() if needs[3] else None,
            None, None, None, None, None, None, None, None, None,
        )


# ==========================================================================
# User-facing helpers
# ==========================================================================
def _as_param(value, device: torch.device, dtype: torch.dtype, name: str) -> Tensor:
    """Coerce a scalar-like parameter to a 0-dim tensor, preserving grad."""
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"{name!r} must be scalar, got shape {tuple(value.shape)}"
            )
        tensor = value.to(device=device, dtype=dtype)
        return tensor if tensor.ndim == 0 else tensor.reshape(())
    return torch.as_tensor(float(value), device=device, dtype=dtype)


def fused_chebyshev_local_vol_ee(
    spot_zero,
    drift,
    legs: Sequence["object"],
    times: Tensor,
    rate: float,
    n_paths: int,
    params: ChebyshevLocalVolParams,
    *,
    coefficients=None,
    term=None,
    seed: int = 0,
    max_programs: int = 4096,
) -> Tensor:
    r"""Expected exposure under Chebyshev local volatility, O(1) in ``n_paths``.

    Signature mirrors ``fused_local_vol_ee``: ``coefficients``/``term`` may be
    overridden as differentiable tensors, letting a caller hold
    ``half_width``/``reference``/``floor`` fixed (structural, matching the
    tanh kernel's ``kappa``/``reference``) while differentiating the fitted
    coefficients.

    Args:
        spot_zero: :math:`S_0`.
        drift: :math:`\mu - q`.
        legs: The netting set.
        times: Observation grid, shape ``(n_steps + 1,)``.
        rate: Discount rate for the affine payoff coefficients.
        n_paths: Monte-Carlo paths.
        params: Structural parameters (``half_width``, ``reference``,
            ``floor``, and the default coefficients/term if not overridden).
        coefficients: Overrides ``params.coefficients`` when given.
        term: Overrides ``params.term`` when given.
        seed: Base Philox key.
        max_programs: Launch-grid cap.

    Returns:
        EE profile, shape ``(n_steps + 1,)``.
    """
    from src.csrc.triton_cva_fusion import build_affine_coefficients

    validate_uniform_grid(times)
    device, dtype = times.device, times.dtype
    dt = float((times[1] - times[0]).item())

    spot_zero = _as_param(spot_zero, device, dtype, "spot_zero")
    drift = _as_param(drift, device, dtype, "drift")
    resolved_coefficients = (
        torch.as_tensor(params.coefficients, device=device, dtype=dtype)
        if coefficients is None
        else coefficients.to(device=device, dtype=dtype)
    )
    resolved_term = (
        _as_param(params.term, device, dtype, "term")
        if term is None
        else _as_param(term, device, dtype, "term")
    )

    coeff_b, coeff_c = build_affine_coefficients(legs, times, rate)
    return FusedChebyshevLocalVolCVAFunction.apply(
        spot_zero, drift, resolved_coefficients, resolved_term,
        coeff_b, coeff_c, n_paths, dt, seed, max_programs,
        params.half_width, params.reference, params.floor,
    )


def fused_chebyshev_local_vol_cva(
    spot_zero,
    drift,
    legs: Sequence["object"],
    times: Tensor,
    rate: float,
    n_paths: int,
    hazard_rate,
    recovery_rate: float,
    params: ChebyshevLocalVolParams,
    *,
    coefficients=None,
    term=None,
    seed: int = 0,
    max_programs: int = 4096,
    convention: str = "endpoint",
) -> Tensor:
    """CVA on top of the Chebyshev kernel's EE profile, composed in PyTorch.

    Args:
        spot_zero: :math:`S_0`.
        drift: :math:`\\mu - q`.
        legs: The netting set.
        times: Observation grid.
        rate: Discount rate.
        n_paths: Monte-Carlo paths.
        hazard_rate: Flat counterparty intensity.
        recovery_rate: Counterparty recovery.
        params: Structural Chebyshev parameters.
        coefficients: Overrides ``params.coefficients``.
        term: Overrides ``params.term``.
        seed: Base Philox key.
        max_programs: Launch-grid cap.
        convention: ``"endpoint"`` or ``"average"``.

    Returns:
        0-dim CVA tensor.
    """
    from src.xva.cva import compute_unilateral_cva

    ee = fused_chebyshev_local_vol_ee(
        spot_zero, drift, legs, times, rate, n_paths, params,
        coefficients=coefficients, term=term, seed=seed,
        max_programs=max_programs,
    )
    return compute_unilateral_cva(
        ee, times, hazard_rate, recovery_rate, discount_rate=rate,
        convention=convention,
    )
