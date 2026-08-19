r"""In-kernel Philox GBM: zero-allocation Brownian increments with rematerialised AAD.

The bottleneck this removes
===========================
Phase 3 fused the arithmetic but still accepted a caller-supplied
:math:`M \times N` increment matrix. At the path counts XVA actually needs that
matrix *is* the memory wall:

===========  ===================  =====================  ==========================
paths        ``dW`` (fp32)        output (fp32)          Phase 3 peak
===========  ===================  =====================  ==========================
1M           0.94 GiB             0.94 GiB               1.88 GiB
5M           4.69 GiB             4.71 GiB               9.41 GiB
10M          9.39 GiB             9.42 GiB               18.81 GiB
20M          18.78 GiB            18.85 GiB              37.63 GiB
===========  ===================  =====================  ==========================

Phase 4 generates the increments *inside* the kernel from a counter-based
(Philox) RNG, so ``dW`` never exists in HBM. Peak drops to the output tensor
alone -- a clean 2x reduction over Phase 3, and roughly 6x over pure PyTorch.

Rematerialisation instead of storage
====================================
Reverse-mode AAD normally needs the forward's random draws again, because

.. math:: \frac{\partial\iota_{m,j}}{\partial\sigma}
          = \sqrt{\Delta t}\,Z_{m,j} - \sigma\,\Delta t

depends on :math:`Z`. Storing :math:`Z` would reinstate the very
:math:`M \times N` allocation we just deleted. Instead the backward kernel
**recomputes** :math:`Z` from the same ``(seed, offset)`` pair. Philox is a
*counter-based* generator: :math:`Z` is a pure function of its inputs, so
recomputation is bit-exact, and the cost is a few integer rounds per element --
far cheaper than a round trip to global memory. This is the classic
recompute-versus-store trade, and on modern GPUs recompute wins decisively for
an RNG.

Consequence for the gradient signature: ``dW`` is no longer a differentiable
input, so there is no ``grad_dW``. Only :math:`S_0`, :math:`\mu` and
:math:`\sigma` receive gradients. The adjoint is therefore *simpler* than Phase
3's, not more complex.

THE ADDRESSING TRAP (and why the offset scheme looks the way it does)
====================================================================
The obvious offset is a global linear index, ``offset = m * n_steps + j``. It is
also **silently catastrophic at scale**. Triton's Philox truncates its offset
argument to 32 bits, and

.. math:: M N > 2^{31}-1 \quad\Longleftrightarrow\quad M > 8.5\text{M (at } N=252)

so from roughly 10M paths onward the offset wraps, distinct
:math:`(m,j)` pairs collide, and the generator returns **the same increments for
different paths**. Nothing raises. The simulation still runs, the paths still
look log-normal, and the Monte-Carlo estimator is quietly biased because the
paths are correlated. This is precisely the failure mode a thesis benchmark at
:math:`M = 20`M would walk into.

The scheme used here keeps every offset tiny by moving the path identity into
the *key* rather than the counter:

.. code-block:: text

    program_seed = seed + program_id          # one Philox stream per path block
    offset       = local_m * n_steps + j      # local_m < BLOCK_M, so < BLOCK_M*N

With ``BLOCK_M <= 64`` and ``N = 252`` the offset never exceeds 16,128 --
comfortably int32 for **any** number of paths. Varying the key is exactly what
Random123 (Salmon et al., 2011) designed counter-based generators to support:
distinct keys yield statistically independent streams, which is the standard
parallel-RNG idiom.

Two properties of this scheme matter for correctness:

1. **The offset does not depend on** ``BLOCK_N``, because ``j`` is the *global*
   time index. The backward is therefore free to chunk time differently from
   the forward.
2. **The offset does depend on** ``BLOCK_M``, via ``local_m`` and
   ``program_id``. Forward and backward *must* use identical ``BLOCK_M`` and an
   identical grid, or rematerialisation silently returns different numbers and
   the gradients are wrong. ``BLOCK_M`` is therefore stored on the autograd
   context and reused verbatim, and
   ``tests/test_phase4.py::TestRematerialisation`` asserts it.

Precision note
==============
``tl.randn`` is a 32-bit Philox construction and returns float32 normals. Under
``float64`` the increments are cast up, so the *accumulation* is double
precision but the underlying normals carry ~7 significant digits. For Monte
Carlo this is irrelevant -- sampling error dwarfs it by orders of magnitude --
but it does mean Phase 4 float64 output will not match a float64-normal
simulation to full double precision. It also does **not** compromise the
finite-difference validation: :math:`Z` is held *fixed* across a bump, so its
precision cancels out of the difference quotient entirely.

What this does NOT fix
======================
The output path matrix is still :math:`O(MN)` and becomes the new ceiling:
18.85 GiB at :math:`M = 20`M. Eliminating ``dW`` roughly **doubles** the
attainable path count on a given device (on a 16 GiB card, ~7.9M paths becomes
~15.9M); it does not make memory independent of :math:`M N`. Reaching 20M+ on a
small GPU requires fusing the *payoff and exposure reduction* into the kernel so
paths are consumed as they are produced and never materialised, making peak
:math:`O(M)`. That is the natural Phase 5 step and is deliberately out of scope
here.

References
----------
Salmon, J. K., Moraes, M. A., Dror, R. O., Shaw, D. E. (2011). *Parallel Random
Numbers: As Easy as 1, 2, 3*. SC'11 -- the Philox/Random123 construction.
Griewank, A., Walther, A. (2008). *Evaluating Derivatives*, 2nd ed., SIAM --
Chapter 12 on checkpointing and recompute-versus-store in reverse mode.
"""

# NOTE: `from __future__ import annotations` is deliberately NOT used here.
# Triton inspects `tl.constexpr` annotations as live objects; postponed (string)
# annotations would demote compile-time constants to runtime arguments.

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from src.csrc.triton_gbm import (
    HAS_TRITON,
    is_available,
    select_block_sizes,
)

__all__ = [
    "MAX_PHILOX_OFFSET",
    "FusedPhiloxGBMFunction",
    "philox_simulate_gbm",
    "reference_philox_forward",
    "reference_philox_backward",
    "validate_offset_scheme",
    "is_available",
]

try:  # pragma: no cover - depends on the host having Triton
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - reuse Phase 3's stubs
    from src.csrc.triton_gbm import tl, triton  # type: ignore[attr-defined]


#: Philox truncates its offset argument to 32 bits. Any offset at or above this
#: aliases onto a smaller one, silently correlating paths.
MAX_PHILOX_OFFSET = 2**31 - 1


def _require_runtime() -> None:
    """Raise a precise error when the fused Philox path cannot run.

    Raises:
        RuntimeError: If Triton is missing or no CUDA device is present.
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "Triton is not installed. The in-kernel Philox GBM kernel requires "
            "Triton and a CUDA device; use src.models.gbm.simulate_gbm for the "
            "portable path."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device is visible to PyTorch. The in-kernel Philox GBM "
            "kernel is GPU-only; use src.models.gbm.simulate_gbm on CPU."
        )


def validate_offset_scheme(block_m: int, n_steps: int) -> None:
    """Assert the per-program offset range cannot alias under Philox truncation.

    This is the guard against the addressing trap described in the module
    docstring. It is cheap, it runs on every launch, and it converts a silent
    statistical corruption into a loud failure.

    Args:
        block_m: Paths per program.
        n_steps: Number of time steps :math:`N`.

    Raises:
        ValueError: If ``block_m * n_steps`` could exceed the 32-bit offset
            space, which would make distinct ``(path, step)`` pairs collide.
    """
    largest_offset = block_m * n_steps
    if largest_offset > MAX_PHILOX_OFFSET:
        raise ValueError(
            f"Philox offset range {largest_offset:,} (BLOCK_M={block_m} x "
            f"n_steps={n_steps}) exceeds the 32-bit limit "
            f"{MAX_PHILOX_OFFSET:,}. Offsets would wrap and different paths "
            "would share increments. Reduce BLOCK_M or n_steps."
        )


# ==========================================================================
# Kernels
# ==========================================================================
@triton.jit
def _philox_gbm_forward_kernel(
    params_ptr,
    out_ptr,
    seed,
    n_paths,
    n_steps,
    out_stride_m,
    out_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DTYPE: tl.constexpr,
):
    """Generate GBM paths with increments drawn inside the kernel.

    No ``dW`` pointer appears in the signature: that is the entire point. The
    increments are produced by ``tl.randn`` in registers, consumed by the scan,
    and discarded.

    Args:
        params_ptr: Packed ``[s0, mu, sigma, dt, sqrt_dt]`` in the working
            dtype. Passed as a device tensor because Python float arguments are
            demoted to float32 by Triton, which would destroy float64
            precision, and because reading a CUDA scalar into Python would force
            a host sync on every launch.
        out_ptr: Output paths, shape ``(n_paths, n_steps + 1)``.
        seed: Base Philox key. Program ``p`` uses ``seed + p`` so each path
            block draws from an independent stream.
        n_paths: Number of paths :math:`M`.
        n_steps: Number of time steps :math:`N`.
        out_stride_m: Row stride of ``out_ptr``.
        out_stride_n: Column stride of ``out_ptr``.
        BLOCK_M: Paths per program. Part of the RNG addressing, so it must
            match between forward and backward.
        BLOCK_N: Time steps per chunk. Not part of the RNG addressing.
        DTYPE: Working element type.
    """
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_paths

    # local_m stays INT32 on purpose: it feeds the Philox offset, which must
    # remain small (see validate_offset_scheme). Only the *global memory*
    # offsets below are widened to int64.
    local_m = tl.arange(0, BLOCK_M)

    # --- 64-bit global addressing (do not remove) ----------------------
    # tl.arange and tl.program_id yield int32, so `offs_m * stride` overflows
    # once n_paths * stride passes INT32_MAX. At N=252 that is 8,488,077 paths:
    # 10M x 253 = 2.53e9 wraps negative and surfaces as
    #   "CUDA error: an illegal memory access was encountered"
    # which poisons the CUDA context and cannot be caught. Promoting the row
    # index to int64 forces the whole offset expression to int64.
    row_out = offs_m.to(tl.int64) * out_stride_m

    s0 = tl.load(params_ptr + 0)
    mu = tl.load(params_ptr + 1)
    sigma = tl.load(params_ptr + 2)
    dt = tl.load(params_ptr + 3)
    sqrt_dt = tl.load(params_ptr + 4)

    drift = (mu - 0.5 * sigma * sigma) * dt
    vol_step = sigma * sqrt_dt

    # One independent Philox stream per path block. Keeping the path identity in
    # the key (not the counter) is what bounds the offset -- see the module
    # docstring on the addressing trap.
    program_seed = seed + pid

    zeros_m = tl.zeros([BLOCK_M], dtype=DTYPE)
    zeros_mn = tl.zeros([BLOCK_M, BLOCK_N], dtype=DTYPE)

    # Column 0 is exactly S0 on every path.
    tl.store(out_ptr + row_out, s0 + zeros_m, mask=mask_m)

    carry = zeros_m
    for start in range(0, n_steps, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_steps
        mask = mask_m[:, None] & mask_n[None, :]

        # local_m (not offs_m) keeps this inside int32 for any M. offs_n is the
        # GLOBAL time index, which makes the stream independent of BLOCK_N.
        rng_offset = (local_m[:, None] * n_steps + offs_n[None, :]).to(tl.int32)
        z = tl.randn(program_seed, rng_offset).to(DTYPE)

        # Masked lanes must contribute exactly zero: `drift + vol_step * 0` is
        # `drift`, which would leak spurious drift into the scan and the carry.
        increments = tl.where(mask, drift + vol_step * z, zeros_mn)

        log_path = tl.cumsum(increments, axis=1) + carry[:, None]
        tl.store(
            out_ptr
            + row_out[:, None]
            + (offs_n.to(tl.int64)[None, :] + 1) * out_stride_n,
            s0 * tl.exp(log_path),
            mask=mask,
        )
        carry = carry + tl.sum(increments, axis=1)


@triton.jit
def _philox_gbm_backward_kernel(
    grad_out_ptr,
    out_ptr,
    params_ptr,
    partial_s0_ptr,
    partial_mu_ptr,
    partial_sigma_ptr,
    seed,
    n_paths,
    n_steps,
    go_stride_m,
    go_stride_n,
    out_stride_m,
    out_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DTYPE: tl.constexpr,
):
    r"""Adjoint of :func:`_philox_gbm_forward_kernel` with rematerialised randoms.

    Walks time chunks in reverse to build the suffix sum
    :math:`Q_{m,j} = \sum_{k\ge j} G_{m,k} S_{m,k}`, regenerating :math:`Z` from
    the *same* ``(seed + pid, offset)`` pair the forward used. No ``dW`` is read
    and no ``grad_dW`` is written -- the increments are not differentiable
    inputs any more.

    Args:
        grad_out_ptr: Incoming adjoint, shape ``(n_paths, n_steps + 1)``.
        out_ptr: Forward output :math:`S`, shape ``(n_paths, n_steps + 1)``.
        params_ptr: Packed ``[s0, mu, sigma, dt, sqrt_dt]``.
        partial_s0_ptr: Per-program partial sums for ``dL/ds0``.
        partial_mu_ptr: Per-program partial sums for ``dL/dmu``.
        partial_sigma_ptr: Per-program partial sums for ``dL/dsigma``.
        seed: The **same** base key the forward used.
        n_paths: Number of paths :math:`M`.
        n_steps: Number of time steps :math:`N`.
        go_stride_m: Row stride of ``grad_out_ptr``.
        go_stride_n: Column stride of ``grad_out_ptr``.
        out_stride_m: Row stride of ``out_ptr``.
        out_stride_n: Column stride of ``out_ptr``.
        BLOCK_M: Paths per program. MUST equal the forward's value.
        BLOCK_N: Time steps per chunk.
        DTYPE: Working element type.
    """
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < n_paths

    # local_m stays INT32: it must reproduce the forward's Philox offset
    # bit-for-bit, and that offset is deliberately kept inside int32.
    local_m = tl.arange(0, BLOCK_M)

    # --- 64-bit global addressing (do not remove) ----------------------
    # tl.arange and tl.program_id yield int32, so `offs_m * stride` overflows
    # once n_paths * stride passes INT32_MAX. At N=252 that is 8,488,077 paths:
    # 10M x 253 = 2.53e9 wraps negative and surfaces as
    #   "CUDA error: an illegal memory access was encountered"
    # which poisons the CUDA context and cannot be caught. Promoting the row
    # index to int64 forces the whole offset expression to int64.
    row_go = offs_m.to(tl.int64) * go_stride_m
    row_out = offs_m.to(tl.int64) * out_stride_m

    s0 = tl.load(params_ptr + 0)
    sigma = tl.load(params_ptr + 2)
    dt = tl.load(params_ptr + 3)
    sqrt_dt = tl.load(params_ptr + 4)

    program_seed = seed + pid

    zeros_m = tl.zeros([BLOCK_M], dtype=DTYPE)
    zeros_mn = tl.zeros([BLOCK_M, BLOCK_N], dtype=DTYPE)

    # Column 0 depends on S0 alone, with dS/dS0 = 1.
    grad_column_zero = tl.load(
        grad_out_ptr + row_go, mask=mask_m, other=0.0
    )
    acc_s0 = tl.where(mask_m, grad_column_zero, zeros_m)
    acc_mu = zeros_m
    acc_sigma = zeros_m

    carry = zeros_m
    n_chunks = (n_steps + BLOCK_N - 1) // BLOCK_N

    for chunk in range(0, n_chunks):
        # Reverse traversal via a forward loop: negative-step ranges over
        # runtime bounds are not portable across Triton versions.
        start = (n_chunks - 1 - chunk) * BLOCK_N
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < n_steps
        mask = mask_m[:, None] & mask_n[None, :]

        offs_n_i64 = offs_n.to(tl.int64)
        grad_slice = tl.load(
            grad_out_ptr + row_go[:, None] + (offs_n_i64[None, :] + 1) * go_stride_n,
            mask=mask,
            other=0.0,
        )
        path_slice = tl.load(
            out_ptr + row_out[:, None] + (offs_n_i64[None, :] + 1) * out_stride_n,
            mask=mask,
            other=0.0,
        )

        p = tl.where(mask, grad_slice * path_slice, zeros_mn)
        chunk_total = tl.sum(p, axis=1)

        # Suffix sum from forward primitives only:
        #     suffix[j] = chunk_total - inclusive_prefix[j] + P[j]
        suffix = chunk_total[:, None] - tl.cumsum(p, axis=1) + p
        q = tl.where(mask, suffix + carry[:, None], zeros_mn)

        # REMATERIALISATION: byte-for-byte the forward's addressing.
        rng_offset = (local_m[:, None] * n_steps + offs_n[None, :]).to(tl.int32)
        z = tl.randn(program_seed, rng_offset).to(DTYPE)

        # d(increment)/d(sigma) = sqrt(dt)*Z - sigma*dt. The second term is the
        # Ito correction's contribution; omitting it yields a plausible-looking
        # and wrong Vega.
        dsigma = tl.where(mask, sqrt_dt * z - sigma * dt, zeros_mn)

        acc_s0 = acc_s0 + chunk_total / s0
        acc_mu = acc_mu + tl.sum(q, axis=1) * dt
        acc_sigma = acc_sigma + tl.sum(q * dsigma, axis=1)
        carry = carry + chunk_total

    tl.store(partial_s0_ptr + pid, tl.sum(acc_s0, axis=0))
    tl.store(partial_mu_ptr + pid, tl.sum(acc_mu, axis=0))
    tl.store(partial_sigma_ptr + pid, tl.sum(acc_sigma, axis=0))


# ==========================================================================
# Pure-PyTorch references (CPU-testable)
# ==========================================================================
def reference_philox_forward(
    s0: Tensor, mu: Tensor, sigma: Tensor, z: Tensor, dt: float
) -> Tensor:
    r"""Reference forward taking the standard normals :math:`Z` explicitly.

    Mirrors the kernel's parameterisation -- increments are
    :math:`a + \sigma\sqrt{\Delta t}\,Z` rather than :math:`a + \sigma\,dW` --
    so the Phase 4 adjoint can be validated on CPU without Triton or a GPU.

    Args:
        s0: Initial spot, 0-dim tensor.
        mu: Drift, 0-dim tensor.
        sigma: Volatility, 0-dim tensor.
        z: Standard normals of shape ``(n_paths, n_steps)``.
        dt: Time step.

    Returns:
        Paths of shape ``(n_paths, n_steps + 1)``.
    """
    increments = (mu - 0.5 * sigma * sigma) * dt + sigma * math.sqrt(dt) * z
    log_path = torch.cumsum(increments, dim=1)
    zero_column = log_path.new_zeros((log_path.shape[0], 1))
    return s0 * torch.exp(torch.cat((zero_column, log_path), dim=1))


def reference_philox_backward(
    grad_out: Tensor,
    paths: Tensor,
    z: Tensor,
    s0: Tensor,
    sigma: Tensor,
    dt: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    r"""Closed-form adjoint for the Phase 4 parameterisation.

    With :math:`P_{m,k} = G_{m,k}S_{m,k}` and
    :math:`Q_{m,j} = \sum_{k\ge j} P_{m,k}`:

    .. math::
        \frac{\partial\mathcal{L}}{\partial S_0}
            &= \sum_m G_{m,0} + \frac{1}{S_0}\sum_{m,k\ge1} P_{m,k}, \\
        \frac{\partial\mathcal{L}}{\partial \mu}
            &= \Delta t \sum_{m,j} Q_{m,j}, \\
        \frac{\partial\mathcal{L}}{\partial \sigma}
            &= \sum_{m,j} Q_{m,j}
               \left(\sqrt{\Delta t}\,Z_{m,j} - \sigma\,\Delta t\right).

    Unlike the kernel this is ordinary differentiable PyTorch, so it also
    supports double backward and serves as the second-order fallback.

    Args:
        grad_out: Incoming adjoint, shape ``(n_paths, n_steps + 1)``.
        paths: Forward output, shape ``(n_paths, n_steps + 1)``.
        z: The standard normals used in the forward, shape
            ``(n_paths, n_steps)``.
        s0: Initial spot, 0-dim tensor. Must be non-zero.
        sigma: Volatility, 0-dim tensor.
        dt: Time step.

    Returns:
        ``(grad_s0, grad_mu, grad_sigma)``, all 0-dim. There is deliberately no
        ``grad_z``: the increments are generated, not supplied, so they are not
        differentiable inputs.

    Raises:
        ValueError: On inconsistent shapes.
    """
    if grad_out.shape != paths.shape:
        raise ValueError(
            f"grad_out {tuple(grad_out.shape)} must match paths {tuple(paths.shape)}"
        )
    if z.shape[0] != paths.shape[0] or z.shape[1] != paths.shape[1] - 1:
        raise ValueError(
            f"z {tuple(z.shape)} inconsistent with paths {tuple(paths.shape)}"
        )

    p = grad_out[:, 1:] * paths[:, 1:]
    q = torch.flip(torch.cumsum(torch.flip(p, dims=(1,)), dim=1), dims=(1,))

    grad_s0 = grad_out[:, 0].sum() + p.sum() / s0
    grad_mu = q.sum() * dt
    grad_sigma = (q * (math.sqrt(dt) * z - sigma * dt)).sum()
    return grad_s0, grad_mu, grad_sigma


# ==========================================================================
# autograd.Function
# ==========================================================================
class FusedPhiloxGBMFunction(torch.autograd.Function):
    """Autograd wrapper for the zero-allocation Philox GBM kernels.

    The forward allocates only its output. The backward allocates only three
    tiny per-program partial buffers, regenerating the increments rather than
    reading them, so end-to-end AAD costs **no** extra :math:`O(MN)` memory.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        s0: Tensor,
        mu: Tensor,
        sigma: Tensor,
        n_paths: int,
        n_steps: int,
        dt: float,
        seed: int,
    ) -> Tensor:
        """Generate paths with in-kernel increments.

        Args:
            ctx: Autograd context.
            s0: Initial spot, 0-dim CUDA tensor.
            mu: Drift, 0-dim CUDA tensor.
            sigma: Volatility, 0-dim CUDA tensor.
            n_paths: Number of paths :math:`M`.
            n_steps: Number of time steps :math:`N`.
            dt: Time step.
            seed: Base Philox key.

        Returns:
            Paths of shape ``(n_paths, n_steps + 1)``.
        """
        _require_runtime()

        device, dtype = s0.device, s0.dtype
        out = torch.empty((n_paths, n_steps + 1), device=device, dtype=dtype)

        params = torch.stack(
            (
                s0.reshape(()),
                mu.reshape(()),
                sigma.reshape(()),
                torch.as_tensor(dt, device=device, dtype=dtype),
                torch.as_tensor(math.sqrt(dt), device=device, dtype=dtype),
            )
        ).contiguous()

        block_m, block_n = select_block_sizes(n_steps, out.element_size())
        # Fail loudly rather than aliasing the RNG. See the module docstring.
        validate_offset_scheme(block_m, n_steps)

        dtype_tl = tl.float64 if dtype == torch.float64 else tl.float32
        grid = (triton.cdiv(n_paths, block_m),)

        _philox_gbm_forward_kernel[grid](
            params,
            out,
            seed,
            n_paths,
            n_steps,
            out.stride(0),
            out.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            DTYPE=dtype_tl,
        )

        # `out` is the returned tensor, so saving it costs nothing extra. No Z
        # is saved -- that is the whole point.
        ctx.save_for_backward(out, params)
        ctx.dt = dt
        ctx.seed = seed
        ctx.n_paths = n_paths
        ctx.n_steps = n_steps
        ctx.block_m = block_m
        ctx.block_n = block_n
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(  # type: ignore[override]
        ctx, grad_out: Tensor
    ) -> Tuple[
        Optional[Tensor], Optional[Tensor], Optional[Tensor], None, None, None, None
    ]:
        """Run the adjoint, regenerating the increments on the fly.

        Args:
            ctx: Autograd context holding the forward's output, packed params,
                seed and launch configuration.
            grad_out: Incoming adjoint, shape ``(n_paths, n_steps + 1)``.

        Returns:
            Gradients for ``(s0, mu, sigma, n_paths, n_steps, dt, seed)``. The
            four non-tensor arguments are structurally non-differentiable and
            yield ``None``; so does any parameter that did not require grad.
        """
        out, params = ctx.saved_tensors
        grad_out = grad_out.contiguous()

        block_m, block_n = ctx.block_m, ctx.block_n
        n_programs = triton.cdiv(ctx.n_paths, block_m)

        partial_s0 = torch.empty(n_programs, device=out.device, dtype=out.dtype)
        partial_mu = torch.empty(n_programs, device=out.device, dtype=out.dtype)
        partial_sigma = torch.empty(n_programs, device=out.device, dtype=out.dtype)

        dtype_tl = tl.float64 if out.dtype == torch.float64 else tl.float32

        # BLOCK_M and seed are reused verbatim: they are part of the RNG
        # addressing, so any deviation would rematerialise different numbers.
        _philox_gbm_backward_kernel[(n_programs,)](
            grad_out,
            out,
            params,
            partial_s0,
            partial_mu,
            partial_sigma,
            ctx.seed,
            ctx.n_paths,
            ctx.n_steps,
            grad_out.stride(0),
            grad_out.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            DTYPE=dtype_tl,
        )

        needs_s0, needs_mu, needs_sigma = ctx.needs_input_grad[:3]
        return (
            partial_s0.sum() if needs_s0 else None,
            partial_mu.sum() if needs_mu else None,
            partial_sigma.sum() if needs_sigma else None,
            None,
            None,
            None,
            None,
        )


# ==========================================================================
# User-facing helper
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
        ValueError: If ``value`` is a tensor with more than one element.
    """
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(
                f"model parameter {name!r} must be scalar, got shape {tuple(value.shape)}"
            )
        tensor = value.to(device=device, dtype=dtype)
        return tensor if tensor.ndim == 0 else tensor.reshape(())
    return torch.as_tensor(float(value), device=device, dtype=dtype)


def philox_simulate_gbm(
    s0,
    mu,
    sigma,
    n_paths: int,
    n_steps: int,
    dt: float,
    *,
    seed: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    r"""Simulate GBM with zero-allocation, in-kernel Brownian increments.

    The API deliberately differs from
    :func:`~src.csrc.triton_gbm.triton_simulate_gbm`: there is no ``dW``
    argument, because not allocating it is the entire contribution of this
    phase. ``(n_paths, n_steps, seed)`` replace it.

    Args:
        s0: Initial spot :math:`S_0`. Pass a tensor with ``requires_grad=True``
            for Delta.
        mu: Drift :math:`\mu`. Equal to the risk-free rate under
            :math:`\mathbb{Q}`. Pass a tensor for Rho.
        sigma: Volatility :math:`\sigma`. Pass a tensor for Vega.
        n_paths: Number of Monte-Carlo paths :math:`M`.
        n_steps: Number of time steps :math:`N`.
        dt: Step size :math:`\Delta t`.
        seed: Base Philox key. Holding this fixed makes the whole simulation
            reproducible **and** supplies common random numbers for
            finite-difference validation.
        device: CUDA device. Defaults to the current one.
        dtype: ``float32`` (recommended) or ``float64``. Note the underlying
            normals are float32 either way -- see the module docstring.

    Returns:
        Paths of shape ``(n_paths, n_steps + 1)``. Column ``0`` is
        :math:`S_0` on every path.

    Raises:
        RuntimeError: If Triton or CUDA is unavailable.
        ValueError: On non-positive sizes, a non-positive ``dt``, an
            unsupported dtype, or an offset range that could alias.

    Example:
        >>> s0 = torch.tensor(100.0, device="cuda", requires_grad=True)
        >>> paths = philox_simulate_gbm(s0, 0.03, 0.2, 1_000_000, 252,
        ...                             1.0 / 252, seed=42)
        >>> paths.shape
        torch.Size([1000000, 253])
    """
    _require_runtime()

    if not isinstance(n_paths, int) or n_paths <= 0:
        raise ValueError(f"n_paths must be a positive int, got {n_paths!r}")
    if not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError(f"n_steps must be a positive int, got {n_steps!r}")
    if not isinstance(dt, (int, float)) or isinstance(dt, bool):
        raise ValueError(f"dt must be a real number, got {type(dt).__name__}")
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError(f"dt must be positive and finite, got {dt}")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"dtype must be float32 or float64, got {dtype}")
    if not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {type(seed).__name__}")

    resolved_device = device if device is not None else torch.device("cuda")
    s0_t = _as_param(s0, resolved_device, dtype, "s0")
    mu_t = _as_param(mu, resolved_device, dtype, "mu")
    sigma_t = _as_param(sigma, resolved_device, dtype, "sigma")

    return FusedPhiloxGBMFunction.apply(
        s0_t, mu_t, sigma_t, n_paths, n_steps, float(dt), int(seed)
    )
