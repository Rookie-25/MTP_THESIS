r"""Phase 6: fused local-volatility exposure kernel with a checkpointed adjoint.

What changes from Phase 5, and why it is structural
===================================================
Phase 5's speed came from an accident of constant volatility: the log-Euler
recursion is *affine* in the state, so the adjoint collapses to a suffix sum
computable with one ``tl.cumsum``. With :math:`\sigma(t, S_t)` that collapse is
gone. The one-step Jacobian is no longer unity,

.. math::
    J_k = 1 + \frac{\partial\sigma_k}{\partial X_k}
              \left(\sqrt{\Delta t}\,Z_k - \sigma_k\Delta t\right),

so the reverse sweep is a genuine sequential recursion
:math:`a_k = \bar{X}_k + a_{k+1}J_k`. No parallel scan primitive applies, in
either direction. Both the forward and the backward become time loops.

Two Triton constraints that dictate the implementation
======================================================
**1. No dynamic indexing into a register tile.** Triton tiles are SSA values;
``tile[:, k]`` for a runtime ``k`` does not exist. This blocks the obvious
implementation of checkpointing twice over -- storing state into slot *k* and
reading slot *k* back. The workaround used throughout is masked access:

.. code-block:: python

    # write column j
    tile = tl.where(offs[None, :] == j, value[:, None], tile)
    # read column j
    value = tl.sum(tl.where(offs[None, :] == j, tile, 0.0), axis=1)

Each costs ``BLOCK`` lane-ops, so the tile must stay *narrow*. That is the real
reason :math:`\sqrt{N}` checkpointing is the right scheme here and not merely a
memory optimisation: at :math:`N = 252` it makes ``BLOCK_CK = 16``, so masked
access costs 16 ops rather than the 256 a full-trajectory tile would need.

**2. No branching on per-lane data.** Every step guard is arithmetic
(``tl.where``), never an ``if``, so warps never diverge.

The checkpointing scheme
========================
:math:`\lceil\sqrt{N}\rceil` segments of ``BLOCK_CK`` steps each. Storage is
one ``(BLOCK_M, BLOCK_CK)`` tile of segment-entry states plus one scratch tile
for the segment being replayed -- about 2 KiB at ``BLOCK_M=16``, entirely in
SRAM, with **no** :math:`O(MN)` or :math:`O(N)` HBM checkpoint buffer.

.. code-block:: text

    backward pass:
      forward once, recording only segment-entry states     (BLOCK_CK columns)
      for each segment, last to first:
          replay the segment forward into a scratch tile     (BLOCK_CK columns)
          walk that segment backwards, updating the adjoint

Total extra forward work is about one full pass -- the textbook uniform-
checkpointing trade (Griewank & Walther, Ch. 12).

Volatility model: parametric, not a grid
========================================
.. math::
    \sigma(t, x) = \sigma_0
                 + \sigma_{\text{skew}}\tanh\!\left(\kappa (x - x_{\text{ref}})\right)
                 + \sigma_{\text{term}}\, t

Chosen deliberately over a Dupire grid. A grid needs a data-dependent gather
per path per step, which Triton can only express as ``tl.load`` through the
cache hierarchy -- fine for a small surface, but it introduces memory
divergence for no benefit here. This form is closed-form in both
:math:`\partial\sigma/\partial x` and :math:`\partial\sigma/\partial\theta`,
evaluates entirely in registers, and has zero divergence.

The skew is centred on :math:`x_{\text{ref}} = \log S_0`. Not cosmetic: an
uncentred ``tanh(kappa * x)`` saturates at :math:`x \approx 4.6`, making
:math:`\operatorname{sech}^2 \approx 10^{-5}` and the surface *effectively
constant* -- which would silently reduce this kernel to Phase 5 while appearing
to test state dependence.

Connecting this to the calibrated SSVI surface of
:mod:`src.models.vol_surface` means fitting these three parameters to
:math:`\sigma_{LV}` (or extending to a Chebyshev expansion in
:math:`(t, \log S)`). That is follow-on work, not done here.

Gradients produced
==================
:math:`S_0` (Delta), :math:`\mu` (drift sensitivity), :math:`\sigma_0` and
:math:`\sigma_{\text{skew}}` (surface sensitivities). As in Phase 5 the kernel
stops at the expected-exposure profile and the :math:`O(N)` credit integral
stays in PyTorch autograd, so hazard-rate and recovery sensitivities come free
and exactly from :func:`src.xva.cva.compute_unilateral_cva`.

Verification status
===================
The **algorithm** is verified on CPU: :func:`reference_local_vol_ee_adjoint`
and :func:`reference_checkpointed_ee_adjoint` implement exactly what the kernels
implement, and ``tests/test_phase6_kernel.py`` checks both against
``torch.autograd``. The **kernels** are unverified until run on a GPU.
"""

# NOTE: `from __future__ import annotations` is deliberately NOT used. Triton
# reads `tl.constexpr` annotations as live objects; postponed annotations would
# demote compile-time constants to runtime arguments.

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.csrc.triton_gbm import HAS_TRITON, is_available
from src.csrc.triton_philox_gbm import validate_offset_scheme
from src.xva.exposure import validate_uniform_grid

__all__ = [
    "LocalVolParams",
    "SRAM_TILE_BUDGET_BYTES",
    "select_local_vol_blocks",
    "local_vol_and_state_derivative",
    "reference_local_vol_ee",
    "reference_local_vol_ee_adjoint",
    "reference_checkpointed_ee_adjoint",
    "FusedLocalVolCVAFunction",
    "fused_local_vol_ee",
    "fused_local_vol_cva",
    "is_available",
]

try:  # pragma: no cover - depends on the host having Triton
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - reuse Phase 3's stubs
    from src.csrc.triton_gbm import tl, triton  # type: ignore[attr-defined]


#: Per-program SRAM budget for the checkpoint + replay tiles.
SRAM_TILE_BUDGET_BYTES = 16 * 1024


@dataclass(frozen=True)
class LocalVolParams:
    r"""Parameters of the parametric local-volatility surface.

    Attributes:
        base: :math:`\sigma_0`, the level. Differentiable.
        skew: :math:`\sigma_{\text{skew}}`, the tanh amplitude. Differentiable.
        kappa: :math:`\kappa`, the tanh steepness. Held constant -- it enters
            :math:`\partial\sigma/\partial x` itself, so differentiating it
            would require a second-order term in the adjoint.
        term: :math:`\sigma_{\text{term}}`, linear term-structure slope. Held
            constant.
        reference: :math:`x_{\text{ref}}`, the log-spot the skew is centred on.
            Must be :math:`\log S_0` or the surface saturates.
    """

    base: float = 0.20
    skew: float = 0.15
    kappa: float = 2.5
    term: float = 0.05
    reference: float = math.log(100.0)

    def __post_init__(self) -> None:
        if self.base <= 0.0:
            raise ValueError(f"base must be positive, got {self.base}")
        if self.kappa <= 0.0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")


def select_local_vol_blocks(n_steps: int, element_size: int) -> Tuple[int, int]:
    r"""Choose ``(BLOCK_M, BLOCK_CK)`` for :math:`\sqrt{N}` checkpointing.

    ``BLOCK_CK`` is the next power of two at or above :math:`\lceil\sqrt{N}\rceil`
    -- the checkpoint count *and* the segment length, since uniform
    checkpointing with :math:`\sqrt{N}` of each minimises
    ``memory x recompute``. Keeping it small also keeps masked tile access cheap
    (see the module docstring).

    Args:
        n_steps: Number of time steps :math:`N`.
        element_size: Bytes per element.

    Returns:
        ``(BLOCK_M, BLOCK_CK)``, both powers of two.

    Raises:
        ValueError: On non-positive inputs, or if one path's two tiles already
            exceed the SRAM budget.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if element_size <= 0:
        raise ValueError(f"element_size must be positive, got {element_size}")

    target = max(1, math.ceil(math.sqrt(n_steps)))
    block_ck = 1 << max(0, (target - 1).bit_length())

    # Two tiles live at once: segment-entry checkpoints and the replay scratch.
    per_path = 2 * block_ck * element_size
    if per_path > SRAM_TILE_BUDGET_BYTES:
        raise ValueError(
            f"n_steps={n_steps} needs {per_path:,} bytes per path for its "
            f"checkpoint tiles, over the {SRAM_TILE_BUDGET_BYTES:,}-byte budget"
        )

    block_m = max(1, min(32, SRAM_TILE_BUDGET_BYTES // per_path))
    block_m = 1 << (block_m.bit_length() - 1)
    return block_m, block_ck


# ==========================================================================
# Pure-PyTorch reference: the algorithm the kernels implement
# ==========================================================================
def local_vol_and_state_derivative(
    time: float,
    log_spot: Tensor,
    params: LocalVolParams,
    base: Optional[Tensor] = None,
    skew: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Evaluate :math:`\sigma`, :math:`\partial\sigma/\partial x`, and the tanh.

    All three come from one ``tanh`` evaluation, which is what the kernel does
    too -- ``tanh`` dominates the cost, so recomputing it for the derivative
    would nearly double the surface evaluation.

    Args:
        time: :math:`t`.
        log_spot: :math:`x`, shape ``(n_paths,)``.
        params: Fixed surface parameters.
        base: Optional differentiable override for :math:`\sigma_0`.
        skew: Optional differentiable override for :math:`\sigma_{\text{skew}}`.

    Returns:
        ``(sigma, d_sigma_d_x, tanh_term)``. The third is
        :math:`\partial\sigma/\partial\sigma_{\text{skew}}`, needed by the
        adjoint.
    """
    level = params.base if base is None else base
    amplitude = params.skew if skew is None else skew
    tanh_term = torch.tanh(params.kappa * (log_spot - params.reference))
    sigma = level + amplitude * tanh_term + params.term * time
    d_sigma_d_x = amplitude * params.kappa * (1.0 - tanh_term * tanh_term)
    return sigma, d_sigma_d_x, tanh_term


def reference_local_vol_ee(
    spot_zero: Tensor,
    drift: Tensor,
    base: Tensor,
    skew: Tensor,
    normals: Tensor,
    dt: float,
    coeff_b: Tensor,
    coeff_c: Tensor,
    params: LocalVolParams,
) -> Tensor:
    r"""Differentiable forward: EE profile under state-dependent volatility.

    Ordinary PyTorch, so ``torch.autograd`` on this is the ground truth the
    hand-derived adjoints are checked against.

    Args:
        spot_zero: :math:`S_0`, 0-dim.
        drift: :math:`\mu - q`, 0-dim.
        base: :math:`\sigma_0`, 0-dim.
        skew: :math:`\sigma_{\text{skew}}`, 0-dim.
        normals: :math:`Z`, shape ``(n_paths, n_steps)``.
        dt: Step size.
        coeff_b: Affine payoff coefficient, shape ``(n_steps + 1,)``.
        coeff_c: Affine payoff coefficient, shape ``(n_steps + 1,)``.
        params: Fixed surface parameters.

    Returns:
        EE profile, shape ``(n_steps + 1,)``.
    """
    n_paths, n_steps = normals.shape
    sqrt_dt = math.sqrt(dt)

    log_spot = torch.log(spot_zero).expand(n_paths)
    columns = [log_spot]
    for step in range(n_steps):
        sigma, _, _ = local_vol_and_state_derivative(
            step * dt, log_spot, params, base=base, skew=skew
        )
        log_spot = (
            log_spot
            + (drift - 0.5 * sigma * sigma) * dt
            + sigma * sqrt_dt * normals[:, step]
        )
        columns.append(log_spot)

    spots = torch.exp(torch.stack(columns, dim=1))
    mtm = coeff_b.reshape(1, -1) * spots - coeff_c.reshape(1, -1)
    return torch.clamp(mtm, min=0.0).mean(dim=0)


def _direct_state_adjoint(
    log_spot: Tensor,
    step_index: int,
    weight: Tensor,
    coeff_b: Tensor,
    coeff_c: Tensor,
) -> Tensor:
    r"""The :math:`\bar{X}_k` term: adjoint of :math:`X_k` from the EE output.

    .. math::
        \bar{X}_k = \omega_k\,\mathbb{1}_{V_k > 0}\, B_k\, S_k,
        \qquad \omega_k = \frac{\bar{EE}_k}{M}

    This must be **added at every step**, not only the terminal one -- the EE
    profile reads the state at all :math:`k`, so every step has a direct
    contribution on top of the recursive one. Omitting it at interior steps is
    the single easiest error to make here, and it produces gradients that are
    plausible but wrong.

    Args:
        log_spot: :math:`X_k`, shape ``(n_paths,)``.
        step_index: :math:`k`.
        weight: :math:`\omega`, shape ``(n_steps + 1,)``.
        coeff_b: Affine coefficient.
        coeff_c: Affine coefficient.

    Returns:
        :math:`\bar{X}_k`, shape ``(n_paths,)``.
    """
    spot = torch.exp(log_spot)
    mtm = coeff_b[step_index] * spot - coeff_c[step_index]
    active = (mtm > 0.0).to(spot.dtype)
    return weight[step_index] * active * coeff_b[step_index] * spot


def reference_local_vol_ee_adjoint(
    grad_ee: Tensor,
    spot_zero: Tensor,
    drift: Tensor,
    base: Tensor,
    skew: Tensor,
    normals: Tensor,
    dt: float,
    coeff_b: Tensor,
    coeff_c: Tensor,
    params: LocalVolParams,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Sequential adjoint with the full trajectory stored -- simplest to verify.

    Implements
    :math:`a_k = \bar{X}_k + a_{k+1}J_k` with
    :math:`J_k = 1 + (\partial\sigma_k/\partial X_k)(\sqrt{\Delta t}Z_k - \sigma_k\Delta t)`,
    accumulating

    .. math::
        \frac{\partial\mathcal{L}}{\partial\theta}
            = \sum_k a_{k+1}\,\frac{\partial\sigma_k}{\partial\theta}
              \left(\sqrt{\Delta t}Z_k - \sigma_k\Delta t\right),
        \qquad
        \frac{\partial\mathcal{L}}{\partial S_0} = \frac{a_0}{S_0}.

    Args:
        grad_ee: Incoming adjoint of the profile, shape ``(n_steps + 1,)``.
        spot_zero: :math:`S_0`, 0-dim.
        drift: :math:`\mu - q`, 0-dim.
        base: :math:`\sigma_0`, 0-dim.
        skew: :math:`\sigma_{\text{skew}}`, 0-dim.
        normals: :math:`Z`, shape ``(n_paths, n_steps)``.
        dt: Step size.
        coeff_b: Affine coefficient.
        coeff_c: Affine coefficient.
        params: Fixed surface parameters.

    Returns:
        ``(grad_spot_zero, grad_drift, grad_base, grad_skew)``, all 0-dim.
    """
    n_paths, n_steps = normals.shape
    sqrt_dt = math.sqrt(dt)
    weight = grad_ee / n_paths

    with torch.no_grad():
        # Forward, storing everything.
        log_spot = torch.log(spot_zero.detach()).expand(n_paths).clone()
        states = [log_spot.clone()]
        for step in range(n_steps):
            sigma, _, _ = local_vol_and_state_derivative(
                step * dt, log_spot, params,
                base=base.detach(), skew=skew.detach(),
            )
            log_spot = (
                log_spot
                + (drift.detach() - 0.5 * sigma * sigma) * dt
                + sigma * sqrt_dt * normals[:, step]
            )
            states.append(log_spot.clone())

        # Reverse.
        adjoint = _direct_state_adjoint(
            states[n_steps], n_steps, weight, coeff_b, coeff_c
        )
        grad_drift = torch.zeros((), dtype=normals.dtype, device=normals.device)
        grad_base = torch.zeros_like(grad_drift)
        grad_skew = torch.zeros_like(grad_drift)

        for step in reversed(range(n_steps)):
            state = states[step]
            sigma, d_sigma_d_x, tanh_term = local_vol_and_state_derivative(
                step * dt, state, params,
                base=base.detach(), skew=skew.detach(),
            )
            vol_factor = sqrt_dt * normals[:, step] - sigma * dt

            grad_drift = grad_drift + (adjoint * dt).sum()
            grad_base = grad_base + (adjoint * vol_factor).sum()
            grad_skew = grad_skew + (adjoint * vol_factor * tanh_term).sum()

            jacobian = 1.0 + d_sigma_d_x * vol_factor
            adjoint = (
                _direct_state_adjoint(state, step, weight, coeff_b, coeff_c)
                + adjoint * jacobian
            )

        grad_spot_zero = adjoint.sum() / spot_zero.detach()

    return grad_spot_zero, grad_drift, grad_base, grad_skew


def reference_checkpointed_ee_adjoint(
    grad_ee: Tensor,
    spot_zero: Tensor,
    drift: Tensor,
    base: Tensor,
    skew: Tensor,
    normals: Tensor,
    dt: float,
    coeff_b: Tensor,
    coeff_c: Tensor,
    params: LocalVolParams,
    block_ck: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r""":math:`\sqrt{N}`-checkpointed adjoint -- mirrors the kernel exactly.

    Stores only segment-entry states and replays each segment forward on demand
    during the reverse walk. Must return the *same* values as
    :func:`reference_local_vol_ee_adjoint`; only which states are held versus
    recomputed differs. That equality is what
    ``tests/test_phase6_kernel.py`` asserts, and it is what makes this a
    trustworthy specification for the Triton translation.

    Args:
        grad_ee: Incoming adjoint of the profile.
        spot_zero: :math:`S_0`, 0-dim.
        drift: :math:`\mu - q`, 0-dim.
        base: :math:`\sigma_0`, 0-dim.
        skew: :math:`\sigma_{\text{skew}}`, 0-dim.
        normals: :math:`Z`, shape ``(n_paths, n_steps)``.
        dt: Step size.
        coeff_b: Affine coefficient.
        coeff_c: Affine coefficient.
        params: Fixed surface parameters.
        block_ck: Segment length. Defaults to
            :math:`\lceil\sqrt{N}\rceil` rounded up to a power of two.

    Returns:
        ``(grad_spot_zero, grad_drift, grad_base, grad_skew)``, all 0-dim.
    """
    n_paths, n_steps = normals.shape
    sqrt_dt = math.sqrt(dt)
    weight = grad_ee / n_paths

    if block_ck is None:
        _, block_ck = select_local_vol_blocks(n_steps, normals.element_size())
    n_segments = -(-n_steps // block_ck)

    detached_base, detached_skew = base.detach(), skew.detach()
    detached_drift = drift.detach()

    def advance(step: int, state: Tensor) -> Tensor:
        """One log-Euler step."""
        sigma, _, _ = local_vol_and_state_derivative(
            step * dt, state, params, base=detached_base, skew=detached_skew
        )
        return (
            state
            + (detached_drift - 0.5 * sigma * sigma) * dt
            + sigma * sqrt_dt * normals[:, step]
        )

    with torch.no_grad():
        # ---- forward: keep only segment-entry states -----------------
        log_spot = torch.log(spot_zero.detach()).expand(n_paths).clone()
        checkpoints = []
        for segment in range(n_segments):
            checkpoints.append(log_spot.clone())
            for offset in range(block_ck):
                step = segment * block_ck + offset
                if step < n_steps:
                    log_spot = advance(step, log_spot)
        terminal = log_spot

        # ---- reverse, segment by segment -----------------------------
        adjoint = _direct_state_adjoint(
            terminal, n_steps, weight, coeff_b, coeff_c
        )
        grad_drift = torch.zeros((), dtype=normals.dtype, device=normals.device)
        grad_base = torch.zeros_like(grad_drift)
        grad_skew = torch.zeros_like(grad_drift)

        for segment in reversed(range(n_segments)):
            # Replay this segment forward from its checkpoint.
            state = checkpoints[segment].clone()
            replay = []
            for offset in range(block_ck):
                step = segment * block_ck + offset
                if step >= n_steps:
                    break
                replay.append(state.clone())
                state = advance(step, state)

            # Walk it backwards.
            for offset in reversed(range(len(replay))):
                step = segment * block_ck + offset
                state = replay[offset]
                sigma, d_sigma_d_x, tanh_term = local_vol_and_state_derivative(
                    step * dt, state, params,
                    base=detached_base, skew=detached_skew,
                )
                vol_factor = sqrt_dt * normals[:, step] - sigma * dt

                grad_drift = grad_drift + (adjoint * dt).sum()
                grad_base = grad_base + (adjoint * vol_factor).sum()
                grad_skew = grad_skew + (adjoint * vol_factor * tanh_term).sum()

                jacobian = 1.0 + d_sigma_d_x * vol_factor
                adjoint = (
                    _direct_state_adjoint(state, step, weight, coeff_b, coeff_c)
                    + adjoint * jacobian
                )

        grad_spot_zero = adjoint.sum() / spot_zero.detach()

    return grad_spot_zero, grad_drift, grad_base, grad_skew


# ==========================================================================
# Triton kernels
# ==========================================================================
@triton.jit
def _tanh(x):
    r"""``tanh`` built from ``exp`` alone, for Triton-version portability.

    ``tl.math.tanh`` does not exist in Triton 3.6.0 (it moved between
    ``tl.math``, ``tl.extra.libdevice`` and top-level ``tl`` across releases),
    and guessing the right name is how this kernel failed its first compile.
    ``tl.exp`` and ``tl.where`` are used by the Phase 3-5 kernels, which compile
    and pass on the target, so building on them is guaranteed to work.

    Uses the overflow-free identity

    .. math:: \tanh(x) = \operatorname{sign}(x)\,\frac{1 - u}{1 + u},
              \qquad u = e^{-2|x|} \in (0, 1].

    The obvious form :math:`(e^{2x}-1)/(e^{2x}+1)` is **not** usable: at
    :math:`x \gtrsim 89` in float32 the exponential overflows to infinity and
    the ratio becomes ``inf/inf = nan``. Taking ``exp`` of a non-positive
    argument cannot overflow, so this form saturates cleanly to :math:`\pm 1`.

    Measured against ``torch.tanh`` over :math:`x \in [-40, 40]` plus extremes:
    max absolute error 1.1e-16 (float64) and 6.0e-08 (float32, ~0.5 ulp), all
    outputs finite and within :math:`[-1, 1]`.

    Args:
        x: Input tile of any shape.

    Returns:
        ``tanh(x)``, same shape and dtype.
    """
    positive = x >= 0.0
    magnitude = tl.where(positive, x, -x)
    decay = tl.exp(-2.0 * magnitude)
    folded = (1.0 - decay) / (1.0 + decay)
    return tl.where(positive, folded, -folded)


@triton.jit
def _fused_local_vol_forward_kernel(
    params_ptr,
    coeff_b_ptr,
    coeff_c_ptr,
    partial_ee_ptr,
    seed,
    n_paths,
    n_columns,
    n_steps,
    pe_stride_p,
    pe_stride_k,
    BLOCK_M: tl.constexpr,
    BLOCK_CK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    r"""Sequential-time forward: paths -> MtM -> exposure -> reduce, no path matrix.

    A time loop rather than a ``tl.cumsum``, because the volatility depends on
    the state. Exposure is accumulated one segment at a time into a narrow
    ``(BLOCK_M, BLOCK_CK)`` tile, then reduced and merged into this program's
    row of ``partial_ee``. Accumulating per *segment* rather than per *step*
    keeps masked tile writes ``BLOCK_CK``-wide (16) instead of ``n_columns``-wide
    (256) -- a 16x saving on the dominant bookkeeping cost.

    Args:
        params_ptr: Packed ``[s0, drift, base, skew, kappa, term, reference,
            dt, sqrt_dt]`` in the working dtype.
        coeff_b_ptr: Affine coefficient :math:`B`, shape ``(n_columns,)``.
        coeff_c_ptr: Affine coefficient :math:`C`, shape ``(n_columns,)``.
        partial_ee_ptr: Output partial sums, ``(n_programs, n_columns)``.
        seed: Base Philox key.
        n_paths: Number of paths :math:`M`.
        n_columns: ``n_steps + 1``.
        n_steps: Number of time steps :math:`N`.
        pe_stride_p: Program-row stride of ``partial_ee_ptr``.
        pe_stride_k: Column stride of ``partial_ee_ptr``.
        BLOCK_M: Paths per tile; part of the RNG addressing.
        BLOCK_CK: Segment length (approximately :math:`\sqrt{N}`).
        DTYPE: Working element type.
    """
    pid = tl.program_id(axis=0)
    n_programs = tl.num_programs(axis=0)

    local_m = tl.arange(0, BLOCK_M)
    offs_ck = tl.arange(0, BLOCK_CK)

    s0 = tl.load(params_ptr + 0)
    drift = tl.load(params_ptr + 1)
    base = tl.load(params_ptr + 2)
    skew = tl.load(params_ptr + 3)
    kappa = tl.load(params_ptr + 4)
    term = tl.load(params_ptr + 5)
    reference = tl.load(params_ptr + 6)
    dt = tl.load(params_ptr + 7)
    sqrt_dt = tl.load(params_ptr + 8)
    # log(S0) is computed host-side rather than with tl.log here: tl.log is
    # the only other primitive in this kernel not already exercised by the
    # Phase 3-5 kernels, and it is a single scalar, so there is nothing to
    # gain by taking the version risk.
    log_s0 = tl.load(params_ptr + 9)
    zeros_m = tl.zeros([BLOCK_M], dtype=DTYPE)
    zeros_tile = tl.zeros([BLOCK_M, BLOCK_CK], dtype=DTYPE)

    n_segments = (n_steps + BLOCK_CK - 1) // BLOCK_CK
    n_blocks = (n_paths + BLOCK_M - 1) // BLOCK_M

    for block_index in range(pid, n_blocks, n_programs):
        offs_m = block_index * BLOCK_M + local_m
        mask_m = offs_m < n_paths
        # Key on the ABSOLUTE block index so the stream is independent of the
        # launch grid; the backward can then use a different grid size.
        program_seed = seed + block_index

        state = log_s0 + zeros_m

        # ---- column 0 exposure ----------------------------------------
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

                # Philox: local_m keeps the counter inside int32 for any M.
                rng_offset = (local_m * n_columns + step).to(tl.int32)
                z = tl.randn(program_seed, rng_offset).to(DTYPE)

                tanh_term = _tanh(kappa * (state - reference))
                sigma = base + skew * tanh_term + term * (step * dt)

                advanced = (
                    state
                    + (drift - 0.5 * sigma * sigma) * dt
                    + sigma * sqrt_dt * z
                )
                # Arithmetic guard, never a branch: past the horizon the state
                # is frozen so the remaining unrolled iterations are inert.
                state = tl.where(live, advanced, state)

                # Exposure at the column this step *produced* (step + 1).
                column = step + 1
                coeff_b = tl.load(
                    coeff_b_ptr + column, mask=live, other=0.0
                )
                coeff_c = tl.load(
                    coeff_c_ptr + column, mask=live, other=0.0
                )
                spot = tl.exp(state)
                exposure = tl.maximum(coeff_b * spot - coeff_c, 0.0)
                exposure = tl.where(mask_m & live, exposure, zeros_m)

                # Masked write into the narrow segment tile.
                exposure_tile = tl.where(
                    offs_ck[None, :] == offset,
                    exposure[:, None],
                    exposure_tile,
                )

            # Reduce the segment and merge into this program's row. Only this
            # program touches row `pid`, so the read-modify-write is race-free.
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
def _fused_local_vol_backward_kernel(
    params_ptr,
    coeff_b_ptr,
    coeff_c_ptr,
    weight_ptr,
    partial_s0_ptr,
    partial_drift_ptr,
    partial_base_ptr,
    partial_skew_ptr,
    seed,
    n_paths,
    n_columns,
    n_steps,
    BLOCK_M: tl.constexpr,
    BLOCK_CK: tl.constexpr,
    DTYPE: tl.constexpr,
):
    r"""Checkpointed sequential adjoint, entirely in SRAM.

    Holds two ``(BLOCK_M, BLOCK_CK)`` tiles: segment-entry checkpoints and the
    replay scratch. No :math:`O(MN)` and no :math:`O(N)` HBM checkpoint buffer.

    The reverse recursion is
    :math:`a_k = \bar{X}_k + a_{k+1}J_k` with
    :math:`J_k = 1 + (\partial\sigma_k/\partial X_k)(\sqrt{\Delta t}Z_k - \sigma_k\Delta t)`.

    Args:
        params_ptr: Packed parameters, as in the forward kernel.
        coeff_b_ptr: Affine coefficient :math:`B`.
        coeff_c_ptr: Affine coefficient :math:`C`.
        weight_ptr: :math:`\omega_k = \texttt{grad\_ee}_k / M`, shape
            ``(n_columns,)``. The ``1/M`` is folded in on the host.
        partial_s0_ptr: Per-program partials for ``dL/ds0``.
        partial_drift_ptr: Per-program partials for ``dL/ddrift``.
        partial_base_ptr: Per-program partials for ``dL/dbase``.
        partial_skew_ptr: Per-program partials for ``dL/dskew``.
        seed: The **same** base key the forward used.
        n_paths: Number of paths :math:`M`.
        n_columns: ``n_steps + 1``.
        n_steps: Number of time steps :math:`N`.
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
    base = tl.load(params_ptr + 2)
    skew = tl.load(params_ptr + 3)
    kappa = tl.load(params_ptr + 4)
    term = tl.load(params_ptr + 5)
    reference = tl.load(params_ptr + 6)
    dt = tl.load(params_ptr + 7)
    sqrt_dt = tl.load(params_ptr + 8)
    # log(S0) is computed host-side rather than with tl.log here: tl.log is
    # the only other primitive in this kernel not already exercised by the
    # Phase 3-5 kernels, and it is a single scalar, so there is nothing to
    # gain by taking the version risk.
    log_s0 = tl.load(params_ptr + 9)
    zeros_m = tl.zeros([BLOCK_M], dtype=DTYPE)
    zeros_tile = tl.zeros([BLOCK_M, BLOCK_CK], dtype=DTYPE)

    n_segments = (n_steps + BLOCK_CK - 1) // BLOCK_CK
    n_blocks = (n_paths + BLOCK_M - 1) // BLOCK_M

    acc_s0 = zeros_m
    acc_drift = zeros_m
    acc_base = zeros_m
    acc_skew = zeros_m

    for block_index in range(pid, n_blocks, n_programs):
        offs_m = block_index * BLOCK_M + local_m
        mask_m = offs_m < n_paths
        program_seed = seed + block_index

        # ---- pass 1: forward, recording only segment-entry states -----
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
                tanh_term = _tanh(kappa * (state - reference))
                sigma = base + skew * tanh_term + term * (step * dt)
                advanced = (
                    state
                    + (drift - 0.5 * sigma * sigma) * dt
                    + sigma * sqrt_dt * z
                )
                state = tl.where(live, advanced, state)

        # ---- terminal direct adjoint ----------------------------------
        weight_n = tl.load(weight_ptr + n_steps)
        coeff_b_n = tl.load(coeff_b_ptr + n_steps)
        coeff_c_n = tl.load(coeff_c_ptr + n_steps)
        spot = tl.exp(state)
        active = (coeff_b_n * spot - coeff_c_n) > 0.0
        adjoint = tl.where(
            mask_m & active, weight_n * coeff_b_n * spot, zeros_m
        )

        # ---- pass 2: reverse, segment by segment ----------------------
        for reverse_index in range(0, n_segments):
            segment = n_segments - 1 - reverse_index

            # Masked read of the segment-entry checkpoint. Triton cannot index
            # a tile by a runtime value, so this is a masked reduction.
            entry = tl.sum(
                tl.where(offs_ck[None, :] == segment, checkpoints, 0.0), axis=1
            )

            # Replay the segment forward into the scratch tile.
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
                tanh_term = _tanh(kappa * (state - reference))
                sigma = base + skew * tanh_term + term * (step * dt)
                advanced = (
                    state
                    + (drift - 0.5 * sigma * sigma) * dt
                    + sigma * sqrt_dt * z
                )
                state = tl.where(live, advanced, state)

            # Walk the segment backwards.
            for reverse_offset in range(0, BLOCK_CK):
                offset = BLOCK_CK - 1 - reverse_offset
                step = segment * BLOCK_CK + offset
                live = step < n_steps

                state_k = tl.sum(
                    tl.where(offs_ck[None, :] == offset, replay, 0.0), axis=1
                )

                rng_offset = (local_m * n_columns + step).to(tl.int32)
                z = tl.randn(program_seed, rng_offset).to(DTYPE)

                tanh_term = _tanh(kappa * (state_k - reference))
                sigma = base + skew * tanh_term + term * (step * dt)
                d_sigma_d_x = skew * kappa * (1.0 - tanh_term * tanh_term)
                vol_factor = sqrt_dt * z - sigma * dt

                gate = mask_m & live
                acc_drift = acc_drift + tl.where(gate, adjoint * dt, zeros_m)
                acc_base = acc_base + tl.where(gate, adjoint * vol_factor, zeros_m)
                acc_skew = acc_skew + tl.where(
                    gate, adjoint * vol_factor * tanh_term, zeros_m
                )

                # THE JACOBIAN.
                jacobian = 1.0 + d_sigma_d_x * vol_factor

                # Direct adjoint of X_k from the EE output -- added at EVERY
                # step, since the profile reads the state at all k.
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

        # dX_0/ds0 = 1/s0.
        acc_s0 = acc_s0 + tl.where(mask_m, adjoint / s0, zeros_m)

    tl.store(partial_s0_ptr + pid, tl.sum(acc_s0, axis=0))
    tl.store(partial_drift_ptr + pid, tl.sum(acc_drift, axis=0))
    tl.store(partial_base_ptr + pid, tl.sum(acc_base, axis=0))
    tl.store(partial_skew_ptr + pid, tl.sum(acc_skew, axis=0))


# ==========================================================================
# autograd.Function
# ==========================================================================
class FusedLocalVolCVAFunction(torch.autograd.Function):
    """Autograd wrapper for the local-volatility exposure kernels.

    Returns the expected-exposure profile only; the :math:`O(N)` credit
    integral composes on top in PyTorch, so credit sensitivities come from
    autograd rather than from hand-written kernel code.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        spot_zero: Tensor,
        drift: Tensor,
        base: Tensor,
        skew: Tensor,
        coeff_b: Tensor,
        coeff_c: Tensor,
        n_paths: int,
        dt: float,
        seed: int,
        max_programs: int,
        kappa: float,
        term: float,
        reference: float,
    ) -> Tensor:
        """Compute the EE profile without materialising any path matrix."""
        if not HAS_TRITON:
            raise RuntimeError(
                "Triton is not installed. The local-volatility kernel requires "
                "Triton and a CUDA device."
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA device is visible. The local-volatility kernel is "
                "GPU-only."
            )

        device, dtype = spot_zero.device, spot_zero.dtype
        n_columns = int(coeff_b.numel())
        n_steps = n_columns - 1

        block_m, block_ck = select_local_vol_blocks(n_steps, coeff_b.element_size())
        validate_offset_scheme(block_m, n_columns)

        n_blocks = (n_paths + block_m - 1) // block_m
        n_programs = min(n_blocks, max_programs)

        params = torch.stack(
            (
                spot_zero.reshape(()),
                drift.reshape(()),
                base.reshape(()),
                skew.reshape(()),
                torch.as_tensor(kappa, device=device, dtype=dtype),
                torch.as_tensor(term, device=device, dtype=dtype),
                torch.as_tensor(reference, device=device, dtype=dtype),
                torch.as_tensor(dt, device=device, dtype=dtype),
                torch.as_tensor(math.sqrt(dt), device=device, dtype=dtype),
                torch.log(spot_zero.reshape(())),
            )
        ).contiguous()

        # The only O(n_programs * N) allocation; no M term.
        partial_ee = torch.zeros((n_programs, n_columns), device=device, dtype=dtype)
        dtype_tl = tl.float64 if dtype == torch.float64 else tl.float32

        _fused_local_vol_forward_kernel[(n_programs,)](
            params, coeff_b, coeff_c, partial_ee,
            seed, n_paths, n_columns, n_steps,
            partial_ee.stride(0), partial_ee.stride(1),
            BLOCK_M=block_m, BLOCK_CK=block_ck, DTYPE=dtype_tl,
        )

        expected_exposure = partial_ee.sum(dim=0) / n_paths

        ctx.save_for_backward(params, coeff_b, coeff_c)
        ctx.n_paths = n_paths
        ctx.n_columns = n_columns
        ctx.n_steps = n_steps
        ctx.seed = seed
        ctx.block_m = block_m
        ctx.block_ck = block_ck
        ctx.n_programs = n_programs
        return expected_exposure

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_ee: Tensor):  # type: ignore[override]
        """Run the checkpointed adjoint kernel."""
        params, coeff_b, coeff_c = ctx.saved_tensors
        device, dtype = coeff_b.device, coeff_b.dtype

        weight = (grad_ee.contiguous() / ctx.n_paths).contiguous()
        n_programs = ctx.n_programs

        partials = [
            torch.empty(n_programs, device=device, dtype=dtype) for _ in range(4)
        ]
        dtype_tl = tl.float64 if dtype == torch.float64 else tl.float32

        # BLOCK_M and seed reused verbatim: both are part of the RNG addressing.
        _fused_local_vol_backward_kernel[(n_programs,)](
            params, coeff_b, coeff_c, weight,
            partials[0], partials[1], partials[2], partials[3],
            ctx.seed, ctx.n_paths, ctx.n_columns, ctx.n_steps,
            BLOCK_M=ctx.block_m, BLOCK_CK=ctx.block_ck, DTYPE=dtype_tl,
        )

        needs = ctx.needs_input_grad
        return (
            partials[0].sum() if needs[0] else None,
            partials[1].sum() if needs[1] else None,
            partials[2].sum() if needs[2] else None,
            partials[3].sum() if needs[3] else None,
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


def fused_local_vol_ee(
    spot_zero,
    drift,
    legs: Sequence["object"],
    times: Tensor,
    rate: float,
    n_paths: int,
    params: LocalVolParams,
    *,
    base=None,
    skew=None,
    seed: int = 0,
    max_programs: int = 4096,
) -> Tensor:
    r"""Expected exposure under local volatility, with peak memory flat in ``n_paths``.

    Args:
        spot_zero: :math:`S_0`. Tensor with ``requires_grad`` for Delta.
        drift: :math:`\mu - q`. Tensor for the drift sensitivity.
        legs: Linear netting set (:class:`~src.pricer.options.SwapLeg`).
        times: Uniform observation grid on CUDA, shape ``(n_steps + 1,)``.
        rate: Flat discount rate, a plain float (see Raises).
        n_paths: Monte-Carlo paths :math:`M`.
        params: Fixed surface parameters. ``params.reference`` should be
            :math:`\log S_0`.
        base: Override for :math:`\sigma_0`; pass a tensor with
            ``requires_grad`` for its sensitivity. Defaults to ``params.base``.
        skew: Override for :math:`\sigma_{\text{skew}}`. Defaults to
            ``params.skew``.
        seed: Base Philox key.
        max_programs: Launch-grid cap; sets the memory ceiling.

    Returns:
        EE profile of shape ``(n_steps + 1,)``, differentiable w.r.t.
        ``spot_zero``, ``drift``, ``base`` and ``skew``.

    Raises:
        ValueError: On a non-CUDA or non-uniform ``times`` grid, non-positive
            ``n_paths``, or a ``rate`` that requires grad -- refused rather
            than silently returning half of Rho, since the discount factors are
            baked into constants before the kernel runs.
        RuntimeError: If Triton or CUDA is unavailable.
    """
    from src.csrc.triton_cva_fusion import build_affine_coefficients

    if isinstance(rate, Tensor) and rate.requires_grad:
        raise ValueError(
            "the fused path cannot produce total Rho: discount factors are "
            "compressed into constants before the kernel runs, so a gradient "
            "w.r.t. 'rate' would silently omit the discounting half."
        )
    if not isinstance(n_paths, int) or n_paths <= 0:
        raise ValueError(f"n_paths must be a positive int, got {n_paths!r}")
    if times.ndim != 1 or times.numel() < 2:
        raise ValueError("times must be a 1-D grid with at least two points")
    if not times.is_cuda:
        raise ValueError("times must be a CUDA tensor; this kernel is GPU-only")

    # Canonical horizon-relative uniformity check. A `rel_tol * dt` bound is
    # the wrong shape: linspace rounding is O(eps * T) and independent of N, so
    # dividing by dt = T/N injects a factor of N and the bound fails for large
    # N. See src/xva/exposure.py::_grid_step for the measurements.
    dt = validate_uniform_grid(times)

    device, dtype = times.device, times.dtype
    coeff_b, coeff_c = build_affine_coefficients(legs, times, float(rate))

    return FusedLocalVolCVAFunction.apply(
        _as_param(spot_zero, device, dtype, "spot_zero"),
        _as_param(drift, device, dtype, "drift"),
        _as_param(params.base if base is None else base, device, dtype, "base"),
        _as_param(params.skew if skew is None else skew, device, dtype, "skew"),
        coeff_b,
        coeff_c,
        n_paths,
        dt,
        int(seed),
        int(max_programs),
        float(params.kappa),
        float(params.term),
        float(params.reference),
    )


def fused_local_vol_cva(
    spot_zero,
    drift,
    legs: Sequence["object"],
    times: Tensor,
    rate: float,
    n_paths: int,
    params: LocalVolParams,
    *,
    base=None,
    skew=None,
    hazard_rate=0.02,
    recovery_rate: float = 0.4,
    seed: int = 0,
    max_programs: int = 4096,
    convention: str = "endpoint",
) -> Tuple[Tensor, Tensor]:
    r"""Fused local-volatility EE profile plus CVA.

    The :math:`O(MN)` reduction runs in the kernel; the :math:`O(N)` credit
    integral stays in PyTorch, so :math:`\partial CVA/\partial\lambda` and
    :math:`\partial CVA/\partial R` come from autograd exactly.

    Args:
        spot_zero: :math:`S_0`.
        drift: :math:`\mu - q`.
        legs: Linear netting set.
        times: Uniform grid on CUDA.
        rate: Flat discount rate (float).
        n_paths: Monte-Carlo paths.
        params: Fixed surface parameters.
        base: Override for :math:`\sigma_0`.
        skew: Override for :math:`\sigma_{\text{skew}}`.
        hazard_rate: Counterparty intensity; tensor for the credit sensitivity.
        recovery_rate: Counterparty recovery.
        seed: Base Philox key.
        max_programs: Launch-grid cap.
        convention: ``"endpoint"`` or ``"average"``.

    Returns:
        ``(ee, cva)``, both on the autograd tape.
    """
    from src.xva.cva import compute_unilateral_cva

    expected_exposure = fused_local_vol_ee(
        spot_zero, drift, legs, times, rate, n_paths, params,
        base=base, skew=skew, seed=seed, max_programs=max_programs,
    )
    cva = compute_unilateral_cva(
        expected_exposure, times, hazard_rate, recovery_rate,
        discount_rate=rate, convention=convention,
    )
    return expected_exposure, cva
