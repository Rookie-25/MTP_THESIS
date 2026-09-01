r"""Fused exposure reduction: O(N) memory XVA with a hand-derived adjoint.

The wall this removes
=====================
Phase 4 deleted the ``dW`` matrix but still materialised the
:math:`M \times (N+1)` path matrix, which became the new ceiling: 18.85 GiB at
:math:`M = 20`M, fp32. Peak was still :math:`O(MN)`.

Phase 5 consumes each path as it is produced. Paths are generated in SRAM,
converted to portfolio MtM, floored to exposure, and **reduced across paths
inside the kernel**. Nothing of size :math:`M` is ever written to HBM. Peak
memory becomes

.. math:: O(\texttt{n\_programs} \times N)\ \text{-- independent of } M,

which for the default 4096 programs and :math:`N=252` is about 4 MiB whether
:math:`M` is ten thousand or fifty million.

The affine collapse that makes this tractable
=============================================
A netting set of linear legs satisfies

.. math::
    V_{m,k} = \sum_i N_i\,\mathbb{1}_{t_k \le T_i}\,(S_{m,k} - K_i)\,
              e^{-r(T_i - t_k)}
            = B_k\, S_{m,k} - C_k,

with

.. math::
    B_k = \sum_i N_i\,\mathbb{1}_{t_k \le T_i}\, e^{-r(T_i-t_k)},\qquad
    C_k = \sum_i N_i\,\mathbb{1}_{t_k \le T_i}\, e^{-r(T_i-t_k)}\,K_i .

So a portfolio of **any** size compresses to two length-:math:`(N+1)` vectors,
computed once on the host. Two consequences matter:

1. The kernel signature is independent of the number of legs.
2. :math:`\partial V_{m,k}/\partial S_{m,k} = B_k` is a *per-step constant*,
   the same for every path -- which is what keeps the adjoint cheap.

:func:`build_affine_coefficients` reproduces
:func:`src.pricer.options.portfolio_swap_mtm` to floating-point rounding, and
``tests/test_phase5.py`` asserts it.

Where the CVA integral lives, and why not in the kernel
=======================================================
The credit integral

.. math:: CVA = (1-R)\sum_{k=1}^{N} EE(t_k)\, d\!PD_k\, DF(t_k)

is :math:`O(N)` -- roughly 252 multiply-adds. Fusing it into the kernel would
buy nothing measurable and would cost a great deal: :math:`\lambda` and
:math:`R` would need hand-written adjoints inside the kernel, and there would be
two implementations of the same integral to keep consistent.

Instead the kernel's job stops at the **expected exposure profile**, and the
credit integral is left in ordinary PyTorch autograd, reusing the already
verified :func:`src.xva.cva.compute_unilateral_cva`. This is strictly better:

* :math:`\partial CVA/\partial\lambda` and :math:`\partial CVA/\partial R` come
  free and exactly, from autograd on an :math:`O(N)` expression -- no kernel
  code, nothing new to validate.
* The fused EE composes with *any* downstream functional (PFE-weighted
  measures, DVA, collateralised variants from
  :mod:`src.xva.exposure`), not only this one CVA formula.
* One source of truth for the integral.

:func:`fused_cva` returns ``(ee, cva)`` so callers still get both, and every
sensitivity is available; only the :math:`O(MN)` part is hand-written.

The adjoint
===========
Write :math:`\omega_k = \bar{EE}_k / M` for the incoming adjoint of the profile,
where :math:`\bar{EE}` is ``grad_ee``. Since
:math:`EE_k = M^{-1}\sum_m (B_k S_{m,k} - C_k)^+`,

.. math::
    \frac{\partial\mathcal{L}}{\partial S_{m,k}}
        = \omega_k\, \mathbb{1}_{V_{m,k} > 0}\, B_k ,
    \qquad
    P_{m,k} := \frac{\partial\mathcal{L}}{\partial S_{m,k}}\, S_{m,k}.

From :math:`S_{m,k} = S_0 e^{L_{m,k}}` with
:math:`L_{m,k} = \sum_{j\le k}\iota_{m,j}` and
:math:`\iota_{m,j} = (\mu - \tfrac12\sigma^2)\Delta t + \sigma\sqrt{\Delta t}Z_{m,j}`:

.. math::
    Q_{m,j} = \sum_{k\ge j} P_{m,k}, \qquad
    \frac{\partial\mathcal{L}}{\partial S_0} = \frac{1}{S_0}\sum_{m,k} P_{m,k},

.. math::
    \frac{\partial\mathcal{L}}{\partial\mu} = \Delta t \sum_{m,j\ge1} Q_{m,j},
    \qquad
    \frac{\partial\mathcal{L}}{\partial\sigma}
        = \sum_{m,j\ge1} Q_{m,j}\left(\sqrt{\Delta t}\,Z_{m,j}
          - \sigma\,\Delta t\right).

Three subtleties, each a place this can silently go wrong:

* The :math:`j \ge 1` restriction is real. :math:`\iota_{m,0}` does not exist
  (:math:`L_{m,0} = 0` by definition), so the suffix sum at index 0 must be
  masked out of the :math:`\mu` and :math:`\sigma` accumulators. Including it
  double-counts the whole path.
* :math:`\partial\mathcal{L}/\partial S_0` **does** include :math:`k=0`, since
  :math:`\partial S_{m,0}/\partial S_0 = 1`.
* The :math:`-\sigma\Delta t` term is the Ito correction's contribution to Vega
  and is the single easiest term to drop.

Rematerialisation, twice over
=============================
The backward stores nothing of size :math:`M`, so it must recompute both the
random draws *and* the paths. Philox makes the first free (pure function of key
and counter). For the second, the kernel holds the entire time axis for a small
tile of paths in SRAM, so the forward walk and the reverse suffix scan happen in
the same tile with no HBM round trip.

That is why this kernel uses a **single time tile** rather than Phase 4's
chunked scan: the reverse suffix sum needs :math:`S_{m,k}` for *all* :math:`k`
at once, and re-deriving it chunk-by-chunk in reverse would cost
:math:`O(N^2/\text{BLOCK\_N})` work or an extra carry buffer. The price is a cap
on :math:`N` (see :func:`select_fused_block_sizes`); ~2500 daily steps fit in
fp32, which covers a ten-year horizon. Beyond that,
:mod:`src.csrc.triton_philox_gbm` remains available at :math:`O(MN)` memory.

Grid-stride and RNG addressing
==============================
The grid is **bounded** (default 4096 programs) and each program strides over
path blocks. This is what makes peak memory constant in :math:`M`: one partial
row per program, not per path block.

The Philox key is derived from the *absolute path-block index*, not from
``program_id``:

.. code-block:: text

    program_seed = seed + block_index      # block_index = absolute, not pid
    offset       = local_m * (n_steps + 1) + k

This is an improvement on Phase 4, where the key was ``seed + program_id`` and
therefore coupled to the grid size. Keying on the absolute block index makes the
random stream reproducible **regardless of how many programs are launched**, so
forward and backward may use different grid sizes and still rematerialise
identical draws. ``BLOCK_M`` remains part of the addressing and must match.

Offsets stay inside int32 by construction (``BLOCK_M * (N+1)``, at most ~16k),
while every *global pointer* offset is promoted to int64 -- the two 32-bit
hazards documented in :mod:`src.csrc.triton_philox_gbm`, which pull in opposite
directions.

Scope
=====
Supported: linear netting sets (forwards, swaps, any affine-in-:math:`S`
portfolio), Delta, drift sensitivity, Vega, and -- via the PyTorch tail --
credit sensitivities.

**Not** supported, deliberately and explicitly:

* *Total Rho.* The discount factors are baked into the precomputed
  :math:`B, C` constants, so differentiating the fused path w.r.t. ``rate``
  would return only the drift half and silently omit the discounting half.
  :func:`fused_expected_exposure` therefore **rejects** a ``rate`` that requires
  grad rather than returning a half-answer.
* *Non-linear payoffs.* A call-option MtM is not affine in :math:`S`, so the
  :math:`B_k S - C_k` collapse does not apply. Adding it means either
  Black-Scholes-on-path inside the kernel or a separate payoff branch.
* *Double backward.* ``once_differentiable``, as in Phases 3-4.
"""

# NOTE: `from __future__ import annotations` is deliberately NOT used here.
# Triton reads `tl.constexpr` annotations as live objects; postponed (string)
# annotations would demote compile-time constants to runtime arguments.

import math
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from src.csrc.triton_gbm import HAS_TRITON, is_available
from src.csrc.triton_philox_gbm import validate_offset_scheme
from src.xva.exposure import validate_uniform_grid

__all__ = [
    "DEFAULT_MAX_PROGRAMS",
    "SRAM_TILE_BUDGET_BYTES",
    "build_affine_coefficients",
    "select_fused_block_sizes",
    "FusedExposureFunction",
    "fused_expected_exposure",
    "fused_cva",
    "reference_fused_exposure",
    "reference_fused_exposure_backward",
    "is_available",
]

try:  # pragma: no cover - depends on the host having Triton
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - reuse Phase 3's stubs
    from src.csrc.triton_gbm import tl, triton  # type: ignore[attr-defined]


#: Upper bound on the launch grid. Peak memory is
#: ``n_programs * (n_steps + 1) * element_size``, so bounding the grid is what
#: makes memory constant in the path count. 4096 saturates any current GPU
#: while costing ~4 MiB at N=252, fp32.
DEFAULT_MAX_PROGRAMS = 4096

#: Per-program SRAM budget for the (BLOCK_M x BLOCK_T) tile. Conservative
#: enough to keep several programs resident per SM on every architecture.
SRAM_TILE_BUDGET_BYTES = 32 * 1024


def _require_runtime() -> None:
    """Raise a precise error when the fused path cannot run.

    Raises:
        RuntimeError: If Triton is missing or no CUDA device is present.
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed. The fused exposure kernel requires Triton "
            "and a CUDA device; use the PyTorch pipeline "
            "(src.models.gbm + src.xva) for the portable path."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device is visible to PyTorch. The fused exposure kernel is "
            "GPU-only; use the PyTorch pipeline on CPU."
        )


# ==========================================================================
# Host-side portfolio compression
# ==========================================================================
def build_affine_coefficients(
    legs: Sequence["object"],
    times: Tensor,
    rate: float,
) -> Tuple[Tensor, Tensor]:
    r"""Compress a linear netting set to the two vectors :math:`B, C`.

    Reproduces :func:`src.pricer.options.portfolio_swap_mtm` in the affine form
    :math:`V_{m,k} = B_k S_{m,k} - C_k`, so the kernel needs no per-leg data and
    its cost is independent of portfolio size.

    Args:
        legs: Sequence of :class:`~src.pricer.options.SwapLeg`, or anything with
            ``notional``, ``strike`` and ``maturity`` attributes.
        times: Observation grid of shape ``(n_steps + 1,)``.
        rate: Flat continuously compounded discount rate. A plain float: see the
            module docstring on why total Rho is not supported.

    Returns:
        ``(B, C)``, each of shape ``(n_steps + 1,)`` on ``times``' device and
        dtype. Both are detached constants.

    Raises:
        ValueError: If ``legs`` is empty or ``times`` is not 1-D.
    """
    if times.ndim != 1:
        raise ValueError(f"times must be 1-dimensional, got {tuple(times.shape)}")
    if not legs:
        raise ValueError("portfolio must contain at least one leg")

    with torch.no_grad():
        b = torch.zeros_like(times)
        c = torch.zeros_like(times)
        for leg in legs:
            maturity = float(leg.maturity)  # type: ignore[attr-defined]
            notional = float(leg.notional)  # type: ignore[attr-defined]
            strike = float(leg.strike)  # type: ignore[attr-defined]

            # Matches equity_forward_mtm exactly: settled legs contribute zero,
            # and the discount uses clamped time-to-maturity.
            alive = (times <= maturity + 1e-12).to(times.dtype)
            discount = torch.exp(
                -rate * torch.clamp(maturity - times, min=0.0)
            )
            weight = notional * alive * discount
            b = b + weight
            c = c + weight * strike

    return b.contiguous(), c.contiguous()


def select_fused_block_sizes(n_steps: int, element_size: int) -> Tuple[int, int]:
    """Choose ``(BLOCK_M, BLOCK_T)`` for the single-tile time axis.

    ``BLOCK_T`` must cover the whole time axis in one tile (``n_steps + 1``
    columns, rounded up to a power of two) because the reverse suffix scan needs
    every :math:`S_{m,k}` simultaneously. ``BLOCK_M`` is then whatever fits the
    SRAM budget.

    Args:
        n_steps: Number of time steps :math:`N`.
        element_size: Bytes per element (4 for float32, 8 for float64).

    Returns:
        ``(BLOCK_M, BLOCK_T)``, both powers of two.

    Raises:
        ValueError: On non-positive inputs, or if a single path's time axis
            already exceeds the SRAM budget -- in which case the horizon is too
            long for the fused kernel and the caller should fall back to
            :func:`src.csrc.triton_philox_gbm.philox_simulate_gbm`.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if element_size <= 0:
        raise ValueError(f"element_size must be positive, got {element_size}")

    columns = n_steps + 1
    block_t = 1 << max(0, (columns - 1).bit_length())

    row_bytes = block_t * element_size
    if row_bytes > SRAM_TILE_BUDGET_BYTES:
        raise ValueError(
            f"n_steps={n_steps} needs {row_bytes:,} bytes per path for a "
            f"single-tile time axis, over the {SRAM_TILE_BUDGET_BYTES:,}-byte "
            "SRAM budget. The fused reduction cannot hold this horizon; use "
            "src.csrc.triton_philox_gbm.philox_simulate_gbm (chunked, O(M*N) "
            "memory) instead."
        )

    block_m = SRAM_TILE_BUDGET_BYTES // row_bytes
    block_m = max(1, min(64, block_m))
    block_m = 1 << (block_m.bit_length() - 1)
    return block_m, block_t


# ==========================================================================
# Kernels
# ==========================================================================
@triton.jit
def _fused_exposure_forward_kernel(
    params_ptr,
    coeff_b_ptr,
    coeff_c_ptr,
    partial_ee_ptr,
    seed,
    n_paths,
    n_columns,
    pe_stride_p,
    pe_stride_k,
    BLOCK_M: tl.constexpr,
    BLOCK_T: tl.constexpr,
    DTYPE: tl.constexpr,
):
    r"""Generate paths, price, floor, and reduce across paths -- all in SRAM.

    Nothing of size :math:`M` is written. Each program keeps a register
    accumulator of length ``BLOCK_T`` and strides over path blocks, so the only
    output is one partial row per program.

    Args:
        params_ptr: Packed ``[s0, mu, sigma, dt, sqrt_dt]`` in the working
            dtype. A device tensor rather than scalar JIT arguments, which
            Triton would demote to float32.
        coeff_b_ptr: Affine coefficient :math:`B_k`, shape ``(n_columns,)``.
        coeff_c_ptr: Affine coefficient :math:`C_k`, shape ``(n_columns,)``.
        partial_ee_ptr: Output partial sums, shape ``(n_programs, n_columns)``.
            Summed over programs and divided by ``n_paths`` on the host.
        seed: Base Philox key.
        n_paths: Number of paths :math:`M`.
        n_columns: ``n_steps + 1``.
        pe_stride_p: Program-row stride of ``partial_ee_ptr``.
        pe_stride_k: Column stride of ``partial_ee_ptr``.
        BLOCK_M: Paths per tile. Part of the RNG addressing.
        BLOCK_T: Time-axis tile width, a power of two at least ``n_columns``.
        DTYPE: Working element type.
    """
    pid = tl.program_id(axis=0)
    n_programs = tl.num_programs(axis=0)

    offs_k = tl.arange(0, BLOCK_T)
    mask_k = offs_k < n_columns
    local_m = tl.arange(0, BLOCK_M)

    s0 = tl.load(params_ptr + 0)
    mu = tl.load(params_ptr + 1)
    sigma = tl.load(params_ptr + 2)
    dt = tl.load(params_ptr + 3)
    sqrt_dt = tl.load(params_ptr + 4)

    drift = (mu - 0.5 * sigma * sigma) * dt
    vol_step = sigma * sqrt_dt

    coeff_b = tl.load(coeff_b_ptr + offs_k, mask=mask_k, other=0.0)
    coeff_c = tl.load(coeff_c_ptr + offs_k, mask=mask_k, other=0.0)

    zeros_mt = tl.zeros([BLOCK_M, BLOCK_T], dtype=DTYPE)

    # Register accumulator: the entire point of the fusion. Lives across the
    # whole grid-stride loop, so no per-block HBM traffic.
    acc_ee = tl.zeros([BLOCK_T], dtype=DTYPE)

    n_blocks = (n_paths + BLOCK_M - 1) // BLOCK_M
    for block_index in range(pid, n_blocks, n_programs):
        offs_m = block_index * BLOCK_M + local_m
        mask_m = offs_m < n_paths
        mask = mask_m[:, None] & mask_k[None, :]

        # Key on the ABSOLUTE block index, not pid: this makes the stream
        # independent of the launch grid, so forward and backward may use
        # different grid sizes and still draw identical numbers.
        program_seed = seed + block_index

        # local_m keeps the Philox counter inside int32 for any n_paths.
        rng_offset = (local_m[:, None] * n_columns + offs_k[None, :]).to(tl.int32)
        z = tl.randn(program_seed, rng_offset).to(DTYPE)

        # Column 0 carries no increment: L[.,0] = 0 by definition.
        increments = tl.where(
            mask & (offs_k[None, :] >= 1), drift + vol_step * z, zeros_mt
        )
        log_path = tl.cumsum(increments, axis=1)
        paths = s0 * tl.exp(log_path)

        # Affine portfolio MtM, then the exposure floor.
        mtm = coeff_b[None, :] * paths - coeff_c[None, :]
        exposure = tl.where(mask, tl.maximum(mtm, 0.0), zeros_mt)

        acc_ee = acc_ee + tl.sum(exposure, axis=0)

    tl.store(
        partial_ee_ptr + pid.to(tl.int64) * pe_stride_p + offs_k.to(tl.int64) * pe_stride_k,
        acc_ee,
        mask=mask_k,
    )


@triton.jit
def _fused_exposure_backward_kernel(
    params_ptr,
    coeff_b_ptr,
    coeff_c_ptr,
    weight_ptr,
    partial_s0_ptr,
    partial_mu_ptr,
    partial_sigma_ptr,
    seed,
    n_paths,
    n_columns,
    BLOCK_M: tl.constexpr,
    BLOCK_T: tl.constexpr,
    DTYPE: tl.constexpr,
):
    r"""Adjoint of :func:`_fused_exposure_forward_kernel`.

    Rematerialises both the random draws and the paths, then contracts the
    suffix sum :math:`Q` against :math:`\partial\iota/\partial\theta`. Only
    three scalars per program leave the kernel.

    Args:
        params_ptr: Packed ``[s0, mu, sigma, dt, sqrt_dt]``.
        coeff_b_ptr: Affine coefficient :math:`B_k`.
        coeff_c_ptr: Affine coefficient :math:`C_k`.
        weight_ptr: :math:`\omega_k = \texttt{grad\_ee}_k / M`, shape
            ``(n_columns,)``. The ``1/M`` is folded in on the host so the kernel
            needs no float path count.
        partial_s0_ptr: Per-program partial sums for ``dL/ds0``.
        partial_mu_ptr: Per-program partial sums for ``dL/dmu``.
        partial_sigma_ptr: Per-program partial sums for ``dL/dsigma``.
        seed: The **same** base key the forward used.
        n_paths: Number of paths :math:`M`.
        n_columns: ``n_steps + 1``.
        BLOCK_M: Paths per tile. MUST equal the forward's value.
        BLOCK_T: Time-axis tile width.
        DTYPE: Working element type.
    """
    pid = tl.program_id(axis=0)
    n_programs = tl.num_programs(axis=0)

    offs_k = tl.arange(0, BLOCK_T)
    mask_k = offs_k < n_columns
    local_m = tl.arange(0, BLOCK_M)

    s0 = tl.load(params_ptr + 0)
    mu = tl.load(params_ptr + 1)
    sigma = tl.load(params_ptr + 2)
    dt = tl.load(params_ptr + 3)
    sqrt_dt = tl.load(params_ptr + 4)

    drift = (mu - 0.5 * sigma * sigma) * dt
    vol_step = sigma * sqrt_dt

    coeff_b = tl.load(coeff_b_ptr + offs_k, mask=mask_k, other=0.0)
    coeff_c = tl.load(coeff_c_ptr + offs_k, mask=mask_k, other=0.0)
    weight = tl.load(weight_ptr + offs_k, mask=mask_k, other=0.0)

    zeros_mt = tl.zeros([BLOCK_M, BLOCK_T], dtype=DTYPE)

    acc_s0 = tl.zeros([BLOCK_M], dtype=DTYPE)
    acc_mu = tl.zeros([BLOCK_M], dtype=DTYPE)
    acc_sigma = tl.zeros([BLOCK_M], dtype=DTYPE)

    n_blocks = (n_paths + BLOCK_M - 1) // BLOCK_M
    for block_index in range(pid, n_blocks, n_programs):
        offs_m = block_index * BLOCK_M + local_m
        mask_m = offs_m < n_paths
        mask = mask_m[:, None] & mask_k[None, :]

        program_seed = seed + block_index
        rng_offset = (local_m[:, None] * n_columns + offs_k[None, :]).to(tl.int32)
        z = tl.randn(program_seed, rng_offset).to(DTYPE)

        # ---- rematerialise the forward -------------------------------
        increments = tl.where(
            mask & (offs_k[None, :] >= 1), drift + vol_step * z, zeros_mt
        )
        log_path = tl.cumsum(increments, axis=1)
        paths = s0 * tl.exp(log_path)
        mtm = coeff_b[None, :] * paths - coeff_c[None, :]

        # ---- adjoint of the exposure floor ---------------------------
        # d/dS max(B*S - C, 0) = B * 1{B*S - C > 0}. The kink at exactly zero
        # is a null set under the GBM law; subgradient 0 there.
        active = (mtm > 0.0) & mask
        p = tl.where(active, weight[None, :] * coeff_b[None, :] * paths, zeros_mt)

        # ---- suffix sum over the time axis ---------------------------
        # Q[j] = sum_{k >= j} P[k], built from forward primitives only:
        #     suffix[j] = total - inclusive_prefix[j] + P[j]
        total = tl.sum(p, axis=1)
        suffix = total[:, None] - tl.cumsum(p, axis=1) + p

        # Column 0 has no increment (iota_0 does not exist), so it must be
        # excluded from the mu/sigma contractions. Including it would
        # double-count every path.
        increment_mask = mask & (offs_k[None, :] >= 1)
        q = tl.where(increment_mask, suffix, zeros_mt)

        # dS[.,0]/ds0 = 1 and dS[.,k]/ds0 = S/s0, so the s0 term keeps k=0.
        acc_s0 = acc_s0 + total / s0
        acc_mu = acc_mu + tl.sum(q, axis=1) * dt
        # sqrt(dt)*Z - sigma*dt : the second term is the Ito correction.
        dsigma = tl.where(increment_mask, sqrt_dt * z - sigma * dt, zeros_mt)
        acc_sigma = acc_sigma + tl.sum(q * dsigma, axis=1)

    tl.store(partial_s0_ptr + pid, tl.sum(acc_s0, axis=0))
    tl.store(partial_mu_ptr + pid, tl.sum(acc_mu, axis=0))
    tl.store(partial_sigma_ptr + pid, tl.sum(acc_sigma, axis=0))


# ==========================================================================
# Pure-PyTorch references (CPU-testable)
# ==========================================================================
def reference_fused_exposure(
    s0: Tensor,
    mu: Tensor,
    sigma: Tensor,
    z: Tensor,
    dt: float,
    coeff_b: Tensor,
    coeff_c: Tensor,
) -> Tensor:
    r"""Reference EE profile taking the normals :math:`Z` explicitly.

    Mirrors the kernel's parameterisation so the fused adjoint can be validated
    against autograd on a CPU-only machine.

    Args:
        s0: Initial spot, 0-dim tensor.
        mu: Drift, 0-dim tensor.
        sigma: Volatility, 0-dim tensor.
        z: Standard normals of shape ``(n_paths, n_columns)``. Column 0 is
            ignored, matching the kernel.
        dt: Time step.
        coeff_b: Affine coefficient :math:`B`, shape ``(n_columns,)``.
        coeff_c: Affine coefficient :math:`C`, shape ``(n_columns,)``.

    Returns:
        Expected exposure profile of shape ``(n_columns,)``.
    """
    increments = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
    # Column 0 carries no increment.
    increments = torch.cat(
        (torch.zeros_like(increments[:, :1]), increments[:, 1:]), dim=1
    )
    paths = s0 * torch.exp(torch.cumsum(increments, dim=1))
    mtm = coeff_b.reshape(1, -1) * paths - coeff_c.reshape(1, -1)
    return torch.clamp(mtm, min=0.0).mean(dim=0)


def reference_fused_exposure_backward(
    grad_ee: Tensor,
    s0: Tensor,
    mu: Tensor,
    sigma: Tensor,
    z: Tensor,
    dt: float,
    coeff_b: Tensor,
    coeff_c: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Closed-form adjoint of :func:`reference_fused_exposure`.

    Implements exactly the formulas the Triton backward implements, in ordinary
    differentiable PyTorch. Also supports double backward, unlike the kernel.

    Args:
        grad_ee: Incoming adjoint of the profile, shape ``(n_columns,)``.
        s0: Initial spot, 0-dim tensor. Must be non-zero.
        mu: Drift, 0-dim tensor.
        sigma: Volatility, 0-dim tensor.
        z: The normals used in the forward, shape ``(n_paths, n_columns)``.
        dt: Time step.
        coeff_b: Affine coefficient :math:`B`.
        coeff_c: Affine coefficient :math:`C`.

    Returns:
        ``(grad_s0, grad_mu, grad_sigma)``, all 0-dim.
    """
    n_paths = z.shape[0]

    increments = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
    increments = torch.cat(
        (torch.zeros_like(increments[:, :1]), increments[:, 1:]), dim=1
    )
    paths = s0 * torch.exp(torch.cumsum(increments, dim=1))
    mtm = coeff_b.reshape(1, -1) * paths - coeff_c.reshape(1, -1)

    weight = (grad_ee / n_paths).reshape(1, -1)
    active = (mtm > 0.0).to(paths.dtype)
    p = weight * coeff_b.reshape(1, -1) * paths * active

    # Suffix sum along time.
    q = torch.flip(torch.cumsum(torch.flip(p, dims=(1,)), dim=1), dims=(1,))
    # Column 0 has no increment.
    q_increments = q[:, 1:]
    z_increments = z[:, 1:]

    grad_s0 = p.sum() / s0
    grad_mu = q_increments.sum() * dt
    grad_sigma = (q_increments * (math.sqrt(dt) * z_increments - sigma * dt)).sum()
    return grad_s0, grad_mu, grad_sigma


# ==========================================================================
# autograd.Function
# ==========================================================================
class FusedExposureFunction(torch.autograd.Function):
    """Autograd wrapper for the fused, O(N)-memory exposure reduction.

    Returns only the expected-exposure profile. Everything downstream -- the
    credit integral, DVA, collateral adjustments -- composes on top in ordinary
    PyTorch, which is what keeps the hand-written surface minimal.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        s0: Tensor,
        mu: Tensor,
        sigma: Tensor,
        coeff_b: Tensor,
        coeff_c: Tensor,
        n_paths: int,
        dt: float,
        seed: int,
        max_programs: int,
    ) -> Tensor:
        """Compute the EE profile without materialising any path matrix.

        Args:
            ctx: Autograd context.
            s0: Initial spot, 0-dim CUDA tensor.
            mu: Drift, 0-dim CUDA tensor.
            sigma: Volatility, 0-dim CUDA tensor.
            coeff_b: Affine coefficient :math:`B`, shape ``(n_columns,)``.
            coeff_c: Affine coefficient :math:`C`, shape ``(n_columns,)``.
            n_paths: Number of paths :math:`M`.
            dt: Time step.
            seed: Base Philox key.
            max_programs: Launch-grid cap; bounds peak memory.

        Returns:
            Expected exposure profile of shape ``(n_columns,)``.
        """
        _require_runtime()

        device, dtype = s0.device, s0.dtype
        n_columns = int(coeff_b.numel())
        n_steps = n_columns - 1

        block_m, block_t = select_fused_block_sizes(n_steps, coeff_b.element_size())
        # Guard the Philox counter range: local_m * n_columns + k.
        validate_offset_scheme(block_m, n_columns)

        n_blocks = (n_paths + block_m - 1) // block_m
        n_programs = min(n_blocks, max_programs)

        params = torch.stack(
            (
                s0.reshape(()),
                mu.reshape(()),
                sigma.reshape(()),
                torch.as_tensor(dt, device=device, dtype=dtype),
                torch.as_tensor(math.sqrt(dt), device=device, dtype=dtype),
            )
        ).contiguous()

        # The ONLY O(n_programs * N) allocation. Independent of n_paths.
        partial_ee = torch.zeros(
            (n_programs, n_columns), device=device, dtype=dtype
        )

        dtype_tl = tl.float64 if dtype == torch.float64 else tl.float32

        _fused_exposure_forward_kernel[(n_programs,)](
            params,
            coeff_b,
            coeff_c,
            partial_ee,
            seed,
            n_paths,
            n_columns,
            partial_ee.stride(0),
            partial_ee.stride(1),
            BLOCK_M=block_m,
            BLOCK_T=block_t,
            DTYPE=dtype_tl,
        )

        expected_exposure = partial_ee.sum(dim=0) / n_paths

        ctx.save_for_backward(params, coeff_b, coeff_c)
        ctx.n_paths = n_paths
        ctx.n_columns = n_columns
        ctx.seed = seed
        ctx.block_m = block_m
        ctx.block_t = block_t
        ctx.n_programs = n_programs
        return expected_exposure

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(  # type: ignore[override]
        ctx, grad_ee: Tensor
    ) -> Tuple[
        Optional[Tensor], Optional[Tensor], Optional[Tensor],
        None, None, None, None, None, None,
    ]:
        """Run the rematerialising adjoint kernel.

        Args:
            ctx: Autograd context holding the packed params, coefficients, seed
                and launch configuration.
            grad_ee: Incoming adjoint of the profile, shape ``(n_columns,)``.

        Returns:
            Gradients for the nine forward arguments. ``coeff_b``/``coeff_c``
            are treated as constants (see the module docstring on Rho), and the
            four trailing scalars are structurally non-differentiable.
        """
        params, coeff_b, coeff_c = ctx.saved_tensors

        # Fold 1/M into the weight so the kernel never needs a float path count.
        weight = (grad_ee.contiguous() / ctx.n_paths).contiguous()

        n_programs = ctx.n_programs
        device, dtype = coeff_b.device, coeff_b.dtype
        partial_s0 = torch.empty(n_programs, device=device, dtype=dtype)
        partial_mu = torch.empty(n_programs, device=device, dtype=dtype)
        partial_sigma = torch.empty(n_programs, device=device, dtype=dtype)

        dtype_tl = tl.float64 if dtype == torch.float64 else tl.float32

        # BLOCK_M and seed reused verbatim: both are part of the RNG addressing.
        _fused_exposure_backward_kernel[(n_programs,)](
            params,
            coeff_b,
            coeff_c,
            weight,
            partial_s0,
            partial_mu,
            partial_sigma,
            ctx.seed,
            ctx.n_paths,
            ctx.n_columns,
            BLOCK_M=ctx.block_m,
            BLOCK_T=ctx.block_t,
            DTYPE=dtype_tl,
        )

        needs_s0, needs_mu, needs_sigma = ctx.needs_input_grad[:3]
        return (
            partial_s0.sum() if needs_s0 else None,
            partial_mu.sum() if needs_mu else None,
            partial_sigma.sum() if needs_sigma else None,
            None, None, None, None, None, None,
        )


# ==========================================================================
# User-facing helpers
# ==========================================================================
def _as_param(value, device: torch.device, dtype: torch.dtype, name: str) -> Tensor:
    """Coerce a scalar-like parameter to a 0-dim tensor, preserving grad.

    Args:
        value: Python float or scalar tensor.
        device: Target device.
        dtype: Target dtype.
        name: Parameter name for error messages.

    Returns:
        A 0-dim tensor on ``device`` with dtype ``dtype``.

    Raises:
        ValueError: If ``value`` is a non-scalar tensor.
    """
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"model parameter {name!r} must be scalar, got shape {tuple(value.shape)}"
            )
        tensor = value.to(device=device, dtype=dtype)
        return tensor if tensor.ndim == 0 else tensor.reshape(())
    return torch.as_tensor(float(value), device=device, dtype=dtype)


def fused_expected_exposure(
    s0,
    mu,
    sigma,
    legs: Sequence["object"],
    times: Tensor,
    rate: float,
    n_paths: int,
    *,
    seed: int = 0,
    max_programs: int = DEFAULT_MAX_PROGRAMS,
) -> Tensor:
    r"""Expected exposure profile with peak memory independent of ``n_paths``.

    Args:
        s0: Initial spot. Pass a tensor with ``requires_grad=True`` for Delta.
        mu: Drift. Pass a tensor for the drift sensitivity.
        sigma: Volatility. Pass a tensor for Vega.
        legs: Linear netting set (:class:`~src.pricer.options.SwapLeg`).
        times: Observation grid of shape ``(n_steps + 1,)`` on CUDA.
        rate: Flat discount rate, a plain **float**. See Raises.
        n_paths: Number of Monte-Carlo paths :math:`M`.
        seed: Base Philox key. Fixing it gives reproducibility and supplies
            common random numbers for finite-difference validation.
        max_programs: Launch-grid cap. Peak memory is
            ``max_programs * (n_steps + 1) * element_size``.

    Returns:
        Expected exposure of shape ``(n_steps + 1,)``, differentiable w.r.t.
        ``s0``, ``mu`` and ``sigma``.

    Raises:
        RuntimeError: If Triton or CUDA is unavailable.
        ValueError: On non-positive ``n_paths``/``dt``, a non-CUDA or
            non-uniform ``times`` grid, or a ``rate`` that requires grad --
            refused rather than silently returning half of Rho, since the
            discount factors are baked into precomputed constants.
    """
    _require_runtime()

    if isinstance(rate, Tensor) and rate.requires_grad:
        raise ValueError(
            "the fused path cannot produce total Rho: the discount factors are "
            "compressed into constant coefficients before the kernel runs, so a "
            "gradient w.r.t. 'rate' would silently omit the discounting half. "
            "Pass a float rate, and use the PyTorch pipeline if Rho is needed."
        )
    if not isinstance(n_paths, int) or n_paths <= 0:
        raise ValueError(f"n_paths must be a positive int, got {n_paths!r}")
    if times.ndim != 1 or times.numel() < 2:
        raise ValueError("times must be a 1-D grid with at least two points")
    if not times.is_cuda:
        raise ValueError("times must be a CUDA tensor; the fused kernel is GPU-only")

    # Canonical horizon-relative uniformity check. A `rel_tol * dt` bound is
    # the wrong shape: linspace rounding is O(eps * T) and independent of N, so
    # dividing by dt = T/N injects a factor of N and the bound fails for large
    # N. See src/xva/exposure.py::_grid_step for the measurements.
    dt = validate_uniform_grid(times)

    device, dtype = times.device, times.dtype
    coeff_b, coeff_c = build_affine_coefficients(legs, times, float(rate))

    return FusedExposureFunction.apply(
        _as_param(s0, device, dtype, "s0"),
        _as_param(mu, device, dtype, "mu"),
        _as_param(sigma, device, dtype, "sigma"),
        coeff_b,
        coeff_c,
        n_paths,
        dt,
        int(seed),
        int(max_programs),
    )


def fused_cva(
    s0,
    mu,
    sigma,
    legs: Sequence["object"],
    times: Tensor,
    rate: float,
    n_paths: int,
    *,
    hazard_rate=0.02,
    recovery_rate: float = 0.4,
    seed: int = 0,
    max_programs: int = DEFAULT_MAX_PROGRAMS,
    convention: str = "endpoint",
) -> Tuple[Tensor, Tensor]:
    r"""Fused EE profile plus CVA, with every sensitivity available.

    The :math:`O(MN)` reduction runs in the kernel; the :math:`O(N)` credit
    integral is left to :func:`src.xva.cva.compute_unilateral_cva` under
    ordinary autograd. So :math:`\partial CVA/\partial\lambda` and
    :math:`\partial CVA/\partial R` come from PyTorch exactly, with no kernel
    code, while Delta and Vega come from the hand-written adjoint.

    Args:
        s0: Initial spot; tensor with ``requires_grad`` for Delta.
        mu: Drift.
        sigma: Volatility; tensor with ``requires_grad`` for Vega.
        legs: Linear netting set.
        times: Uniform observation grid on CUDA, shape ``(n_steps + 1,)``.
        rate: Flat discount rate, a plain float.
        n_paths: Number of paths :math:`M`.
        hazard_rate: Counterparty intensity :math:`\lambda`; tensor with
            ``requires_grad`` for the credit sensitivity.
        recovery_rate: Counterparty recovery :math:`R`.
        seed: Base Philox key.
        max_programs: Launch-grid cap.
        convention: ``"endpoint"`` or ``"average"``, per
            :func:`src.xva.cva.compute_unilateral_cva`.

    Returns:
        ``(ee, cva)``. ``ee`` has shape ``(n_steps + 1,)``; ``cva`` is 0-dim.
        Both remain on the autograd tape.
    """
    from src.xva.cva import compute_unilateral_cva

    expected_exposure = fused_expected_exposure(
        s0, mu, sigma, legs, times, rate, n_paths,
        seed=seed, max_programs=max_programs,
    )
    cva = compute_unilateral_cva(
        expected_exposure,
        times,
        hazard_rate,
        recovery_rate,
        discount_rate=rate,
        convention=convention,
    )
    return expected_exposure, cva
