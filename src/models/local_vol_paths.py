r"""State-dependent SDE simulation and its adjoint: the reference for Phase 6.

Why this module exists
======================
Phases 3-5 exploited a structural accident: with constant :math:`\sigma`, the
log-Euler recursion is *affine* in the state, so the reverse-mode adjoint
collapses to a **suffix sum** computable with one ``tl.cumsum``. That collapse is
the entire reason those kernels are fast and memory-flat.

State-dependent volatility destroys it. This module derives and validates the
replacement, in pure PyTorch, so the mathematics is verified before any Triton is
written.

The forward scheme
==================
With :math:`X_t = \log S_t`, the log-Euler discretisation of

.. math:: dS_t = (\mu - q)S_t\,dt + \sigma_{LV}(t, S_t)S_t\,dW_t

is

.. math::
    X_{k+1} = X_k
            + \left(\mu - q - \tfrac12\sigma_k^2\right)\Delta t
            + \sigma_k\sqrt{\Delta t}\,Z_k,
    \qquad \sigma_k = \sigma_{LV}(t_k, e^{X_k}).

Unlike GBM this is **not** exact: the local volatility is frozen over each step,
giving the usual Euler weak order 1. Discretisation bias is therefore a real
error source in Phase 6 where it was absent in Phases 1-5 -- a distinction worth
keeping straight when attributing benchmark discrepancies.

The adjoint is a sequential recursion, not a sum
================================================
Because :math:`\sigma_k` depends on :math:`X_k`, the one-step Jacobian is no
longer 1:

.. math::
    J_k := \frac{\partial X_{k+1}}{\partial X_k}
         = 1 + \frac{\partial\sigma_k}{\partial X_k}
               \left(\sqrt{\Delta t}\,Z_k - \sigma_k\,\Delta t\right).

Writing :math:`a_k = \partial\mathcal{L}/\partial X_k` and :math:`\bar{X}_k` for
the incoming adjoint of the state at step :math:`k`, the reverse sweep is

.. math:: a_k = \bar{X}_k + a_{k+1} J_k ,
          \qquad a_N = \bar{X}_N ,

and a surface parameter :math:`\theta` accumulates

.. math::
    \frac{\partial\mathcal{L}}{\partial\theta}
        = \sum_{k=0}^{N-1} a_{k+1}
          \frac{\partial\sigma_k}{\partial\theta}
          \left(\sqrt{\Delta t}\,Z_k - \sigma_k\,\Delta t\right),
    \qquad
    \frac{\partial\mathcal{L}}{\partial S_0} = \frac{a_0}{S_0}.

Setting :math:`\partial\sigma/\partial X = 0` gives :math:`J_k \equiv 1` and
:math:`a_k = \sum_{j\ge k}\bar{X}_j` -- precisely the Phase 3-5 suffix sum. So
the old kernels are the special case, not an approximation to be reused.

**Measured consequence of reusing the suffix sum anyway** (see
``tests/test_phase6.py``): on a smooth skewed local-vol surface with
:math:`N=252`, the suffix-sum shortcut misprices
:math:`\partial\mathcal{L}/\partial\theta` by roughly 1% at
:math:`\overline{|\partial\sigma/\partial x|} \approx 0.05`, rising with the
skew. One percent sounds survivable; it is not. The error is a *bias*, not
variance: it does not shrink as :math:`M \to \infty`, it is invisible to every
statistical check, and calibrating a surface by gradient descent on a
systematically wrong gradient converges to the wrong surface.

Memory: the reason Phase 6 cannot be free
=========================================
The reverse sweep needs :math:`\sigma_k` and
:math:`\partial\sigma_k/\partial X_k` at every :math:`k`, in **descending**
:math:`k`. The forward recursion only produces them in ascending order, and it
cannot be run backwards: inverting

.. math:: X_k = X_{k+1} - \left(\mu - q - \tfrac12\sigma_k^2\right)\Delta t
                        - \sigma_k\sqrt{\Delta t}Z_k

requires :math:`\sigma_k`, which depends on the unknown :math:`X_k`. It is an
implicit equation, so there is no cheap reverse reconstruction. Phases 3-5 dodged
this because :math:`S_{m,k}` was recoverable in closed form from a single
cumulative sum.

Three admissible strategies, with their costs per program tile
(:math:`B` = ``BLOCK_M``):

=========================  =====================  ==========================
strategy                   extra memory           extra forward work
=========================  =====================  ==========================
store full trajectory      :math:`O(BN)`          none
:math:`\sqrt{N}` checkpt   :math:`O(B\sqrt{N})`   ~1 extra forward pass
recompute from scratch     :math:`O(B)`           :math:`O(N/2)` passes
=========================  =====================  ==========================

At :math:`N=252`, :math:`B=16`, 4096 programs, fp32: full storage is ~66 MiB,
:math:`\sqrt{N}` checkpointing ~4 MiB, full recompute ~0.26 MiB but 126x the
forward work. **The recommendation is** :math:`\sqrt{N}` **checkpointing**: it
keeps memory in the same order as Phase 5's partial buffers while costing about
one extra forward pass, and it is the textbook optimum (Griewank & Walther,
Ch. 12). All three are implemented below so the trade can be measured rather
than argued.

Crucially, every one of these is :math:`O(1)` in the *path count* :math:`M` --
the Phase 5 headline survives. What changes is the constant, and the honest
claim becomes "flat in :math:`M`, linear in :math:`\sqrt{N}`" rather than
"flat".

References
----------
Griewank, A., Walther, A. (2008). *Evaluating Derivatives*, 2nd ed., SIAM --
Ch. 12, binomial and uniform checkpointing.
Giles, M., Glasserman, P. (2006). *Smoking adjoints: fast Monte Carlo Greeks*.
Risk 19(1) -- pathwise adjoints for SDE models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple

import torch
from torch import Tensor

__all__ = [
    "LocalVolFn",
    "StepDerivatives",
    "simulate_local_vol_paths",
    "local_vol_adjoint",
    "checkpointed_local_vol_adjoint",
    "suffix_sum_adjoint_incorrect",
    "checkpoint_schedule",
]


class LocalVolFn(Protocol):
    """A differentiable local-volatility callable.

    Must accept ``(time, log_spot)`` with ``log_spot`` of shape ``(n_paths,)``
    and return :math:`\\sigma` of the same shape, differentiably in both
    ``log_spot`` and any captured parameters.
    """

    def __call__(self, time: float, log_spot: Tensor) -> Tensor:  # pragma: no cover
        ...


@dataclass
class StepDerivatives:
    """Per-step quantities the reverse sweep needs.

    Attributes:
        sigma: :math:`\\sigma_k`, shape ``(n_paths,)``.
        d_sigma_d_log_spot: :math:`\\partial\\sigma_k/\\partial X_k`, shape
            ``(n_paths,)``. This is the term that is identically zero in
            Phases 1-5 and non-zero here.
    """

    sigma: Tensor
    d_sigma_d_log_spot: Tensor


def _evaluate_step(
    local_vol: LocalVolFn, time: float, log_spot: Tensor
) -> StepDerivatives:
    r"""Evaluate :math:`\sigma` and :math:`\partial\sigma/\partial X` at one step.

    Args:
        local_vol: The local-volatility callable.
        time: :math:`t_k`.
        log_spot: :math:`X_k`, shape ``(n_paths,)``.

    Returns:
        The :class:`StepDerivatives` for this step, detached.
    """
    with torch.enable_grad():
        leaf = log_spot.detach().clone().requires_grad_(True)
        sigma = local_vol(time, leaf)
        (derivative,) = torch.autograd.grad(sigma.sum(), leaf)
    return StepDerivatives(
        sigma=sigma.detach(), d_sigma_d_log_spot=derivative.detach()
    )


def simulate_local_vol_paths(
    spot_zero: Tensor,
    drift: float,
    local_vol: LocalVolFn,
    normals: Tensor,
    dt: float,
) -> Tensor:
    r"""Simulate log-Euler paths under state-dependent volatility.

    Written in ordinary differentiable PyTorch, so ``torch.autograd`` provides
    the ground truth the hand-derived adjoints below are checked against.

    Args:
        spot_zero: :math:`S_0`, 0-dim tensor. Pass ``requires_grad=True`` for
            Delta.
        drift: :math:`\mu - q`, the risk-neutral drift net of dividends.
        local_vol: Differentiable :math:`\sigma_{LV}(t, X)`.
        normals: Standard normals :math:`Z`, shape ``(n_paths, n_steps)``.
        dt: Step size :math:`\Delta t`.

    Returns:
        Log-spot trajectory of shape ``(n_paths, n_steps + 1)``. Column 0 is
        :math:`\log S_0` on every path.

    Raises:
        ValueError: If ``normals`` is not 2-D or ``dt`` is non-positive.
    """
    if normals.ndim != 2:
        raise ValueError(
            f"normals must be (n_paths, n_steps), got {tuple(normals.shape)}"
        )
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    n_paths, n_steps = normals.shape
    sqrt_dt = math.sqrt(dt)

    log_spot = torch.log(spot_zero).expand(n_paths)
    trajectory = [log_spot]
    for step in range(n_steps):
        sigma = local_vol(step * dt, log_spot)
        log_spot = (
            log_spot
            + (drift - 0.5 * sigma * sigma) * dt
            + sigma * sqrt_dt * normals[:, step]
        )
        trajectory.append(log_spot)
    return torch.stack(trajectory, dim=1)


def local_vol_adjoint(
    grad_log_spot: Tensor,
    spot_zero: Tensor,
    drift: float,
    local_vol: LocalVolFn,
    parameters: Tuple[Tensor, ...],
    normals: Tensor,
    dt: float,
) -> Tuple[Tensor, Tuple[Tensor, ...]]:
    r"""Sequential reverse-mode adjoint, storing the full trajectory.

    The reference implementation: simplest to verify, largest memory. Uses the
    recursion :math:`a_k = \bar{X}_k + a_{k+1}J_k` derived in the module
    docstring.

    Args:
        grad_log_spot: Incoming adjoint :math:`\bar{X}`, shape
            ``(n_paths, n_steps + 1)``.
        spot_zero: :math:`S_0`, 0-dim.
        drift: :math:`\mu - q`.
        local_vol: Differentiable :math:`\sigma_{LV}(t, X)`.
        parameters: Surface parameters to accumulate gradients for. Each must be
            a leaf that ``local_vol`` actually depends on.
        normals: Standard normals, shape ``(n_paths, n_steps)``.
        dt: Step size.

    Returns:
        ``(grad_spot_zero, grad_parameters)``. ``grad_spot_zero`` is 0-dim;
        ``grad_parameters`` matches ``parameters`` element-wise.

    Raises:
        ValueError: On inconsistent shapes.
    """
    n_paths, n_steps = normals.shape
    if grad_log_spot.shape != (n_paths, n_steps + 1):
        raise ValueError(
            f"grad_log_spot must be {(n_paths, n_steps + 1)}, "
            f"got {tuple(grad_log_spot.shape)}"
        )

    sqrt_dt = math.sqrt(dt)

    # ---- forward pass, recording what the reverse sweep needs ---------
    with torch.no_grad():
        log_spot = torch.log(spot_zero.detach()).expand(n_paths).clone()
        states = [log_spot.clone()]
        for step in range(n_steps):
            step_data = _evaluate_step(local_vol, step * dt, log_spot)
            log_spot = (
                log_spot
                + (drift - 0.5 * step_data.sigma**2) * dt
                + step_data.sigma * sqrt_dt * normals[:, step]
            )
            states.append(log_spot.clone())

    # ---- reverse sweep -----------------------------------------------
    grad_parameters = [torch.zeros_like(p) for p in parameters]
    adjoint = grad_log_spot[:, n_steps].clone()

    for step in reversed(range(n_steps)):
        state = states[step]
        # Re-evaluate sigma ON the graph so parameter gradients can flow.
        with torch.enable_grad():
            state_leaf = state.detach().clone().requires_grad_(True)
            sigma = local_vol(step * dt, state_leaf)
            # d(sigma)/dX for the Jacobian, and d(sigma)/d(theta) for the
            # parameter accumulation, from a single graph.
            vol_factor = sqrt_dt * normals[:, step] - sigma.detach() * dt
            weighted = (adjoint * vol_factor * sigma).sum()
            grads = torch.autograd.grad(
                weighted,
                (state_leaf,) + tuple(parameters),
                allow_unused=True,
                retain_graph=False,
            )
        # grads[0] is d/dX of (a * vf * sigma) = a * vf * dsigma/dX, since
        # a and vf are constants w.r.t. state_leaf here.
        for index, grad in enumerate(grads[1:]):
            if grad is not None:
                grad_parameters[index] = grad_parameters[index] + grad

        with torch.no_grad():
            step_data = _evaluate_step(local_vol, step * dt, state)
            jacobian = 1.0 + step_data.d_sigma_d_log_spot * (
                sqrt_dt * normals[:, step] - step_data.sigma * dt
            )
            adjoint = grad_log_spot[:, step] + adjoint * jacobian

    grad_spot_zero = adjoint.sum() / spot_zero.detach()
    return grad_spot_zero, tuple(grad_parameters)


def checkpoint_schedule(n_steps: int, n_checkpoints: Optional[int] = None) -> list:
    r"""Uniform checkpoint indices for :math:`\sqrt{N}` rematerialisation.

    Args:
        n_steps: Number of time steps :math:`N`.
        n_checkpoints: Number of stored states. Defaults to
            :math:`\lceil\sqrt{N}\rceil`, which minimises
            ``memory x recompute`` for uniform checkpointing.

    Returns:
        Sorted, unique step indices to checkpoint, always including ``0``.

    Raises:
        ValueError: If ``n_steps`` is non-positive or ``n_checkpoints < 1``.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if n_checkpoints is None:
        n_checkpoints = max(1, math.ceil(math.sqrt(n_steps)))
    if n_checkpoints < 1:
        raise ValueError(f"n_checkpoints must be >= 1, got {n_checkpoints}")

    stride = max(1, math.ceil(n_steps / n_checkpoints))
    return sorted({0, *range(0, n_steps, stride)})


def checkpointed_local_vol_adjoint(
    grad_log_spot: Tensor,
    spot_zero: Tensor,
    drift: float,
    local_vol: LocalVolFn,
    parameters: Tuple[Tensor, ...],
    normals: Tensor,
    dt: float,
    n_checkpoints: Optional[int] = None,
) -> Tuple[Tensor, Tuple[Tensor, ...]]:
    r"""Adjoint using :math:`\sqrt{N}` checkpointing -- the recommended scheme.

    Stores only :math:`O(\sqrt{N})` states per path instead of :math:`N`, and
    recomputes each segment forward on demand during the reverse sweep. Total
    extra forward work is about one full pass.

    This must produce **bitwise-comparable** results to
    :func:`local_vol_adjoint`; the only difference is which states are held in
    memory versus recomputed. ``tests/test_phase6.py`` asserts they agree.

    Args:
        grad_log_spot: Incoming adjoint, shape ``(n_paths, n_steps + 1)``.
        spot_zero: :math:`S_0`, 0-dim.
        drift: :math:`\mu - q`.
        local_vol: Differentiable :math:`\sigma_{LV}(t, X)`.
        parameters: Surface parameters to accumulate gradients for.
        normals: Standard normals, shape ``(n_paths, n_steps)``.
        dt: Step size.
        n_checkpoints: Number of stored states; defaults to
            :math:`\lceil\sqrt{N}\rceil`.

    Returns:
        ``(grad_spot_zero, grad_parameters)``, same contract as
        :func:`local_vol_adjoint`.
    """
    n_paths, n_steps = normals.shape
    sqrt_dt = math.sqrt(dt)
    checkpoints = checkpoint_schedule(n_steps, n_checkpoints)
    checkpoint_set = set(checkpoints)

    # ---- forward pass, storing ONLY the checkpoints -------------------
    with torch.no_grad():
        log_spot = torch.log(spot_zero.detach()).expand(n_paths).clone()
        stored = {}
        for step in range(n_steps):
            if step in checkpoint_set:
                stored[step] = log_spot.clone()
            step_data = _evaluate_step(local_vol, step * dt, log_spot)
            log_spot = (
                log_spot
                + (drift - 0.5 * step_data.sigma**2) * dt
                + step_data.sigma * sqrt_dt * normals[:, step]
            )

    def replay_segment(start: int, stop: int) -> list:
        """Recompute states ``[start, stop)`` from the checkpoint at ``start``."""
        with torch.no_grad():
            state = stored[start].clone()
            segment = [state.clone()]
            for step in range(start, stop - 1):
                step_data = _evaluate_step(local_vol, step * dt, state)
                state = (
                    state
                    + (drift - 0.5 * step_data.sigma**2) * dt
                    + step_data.sigma * sqrt_dt * normals[:, step]
                )
                segment.append(state.clone())
        return segment

    # ---- reverse sweep, segment by segment ---------------------------
    grad_parameters = [torch.zeros_like(p) for p in parameters]
    adjoint = grad_log_spot[:, n_steps].clone()

    boundaries = checkpoints + [n_steps]
    for index in reversed(range(len(checkpoints))):
        start, stop = boundaries[index], boundaries[index + 1]
        segment = replay_segment(start, stop)

        for step in reversed(range(start, stop)):
            state = segment[step - start]
            with torch.enable_grad():
                state_leaf = state.detach().clone().requires_grad_(True)
                sigma = local_vol(step * dt, state_leaf)
                vol_factor = sqrt_dt * normals[:, step] - sigma.detach() * dt
                weighted = (adjoint * vol_factor * sigma).sum()
                grads = torch.autograd.grad(
                    weighted, tuple(parameters), allow_unused=True
                )
            for position, grad in enumerate(grads):
                if grad is not None:
                    grad_parameters[position] = grad_parameters[position] + grad

            with torch.no_grad():
                step_data = _evaluate_step(local_vol, step * dt, state)
                jacobian = 1.0 + step_data.d_sigma_d_log_spot * (
                    sqrt_dt * normals[:, step] - step_data.sigma * dt
                )
                adjoint = grad_log_spot[:, step] + adjoint * jacobian

    grad_spot_zero = adjoint.sum() / spot_zero.detach()
    return grad_spot_zero, tuple(grad_parameters)


def suffix_sum_adjoint_incorrect(
    grad_log_spot: Tensor,
    spot_zero: Tensor,
    drift: float,
    local_vol: LocalVolFn,
    parameters: Tuple[Tensor, ...],
    normals: Tensor,
    dt: float,
) -> Tuple[Tensor, Tuple[Tensor, ...]]:
    r"""The Phase 3-5 suffix-sum adjoint, applied where it does **not** hold.

    Provided deliberately, and named accordingly, so the error from reusing the
    constant-volatility shortcut can be *measured* rather than asserted. It sets
    :math:`J_k \equiv 1`, which is exact only when
    :math:`\partial\sigma/\partial X = 0`.

    Do not use this for anything except the comparison test. It is a biased
    estimator: the error does not vanish as the path count grows, and no
    statistical check will reveal it.

    Args:
        grad_log_spot: Incoming adjoint, shape ``(n_paths, n_steps + 1)``.
        spot_zero: :math:`S_0`, 0-dim.
        drift: :math:`\mu - q`.
        local_vol: Local-volatility callable.
        parameters: Surface parameters.
        normals: Standard normals.
        dt: Step size.

    Returns:
        ``(grad_spot_zero, grad_parameters)`` -- systematically wrong whenever
        the volatility is state-dependent.
    """
    n_paths, n_steps = normals.shape
    sqrt_dt = math.sqrt(dt)

    with torch.no_grad():
        log_spot = torch.log(spot_zero.detach()).expand(n_paths).clone()
        states = [log_spot.clone()]
        for step in range(n_steps):
            step_data = _evaluate_step(local_vol, step * dt, log_spot)
            log_spot = (
                log_spot
                + (drift - 0.5 * step_data.sigma**2) * dt
                + step_data.sigma * sqrt_dt * normals[:, step]
            )
            states.append(log_spot.clone())

    # J_k == 1 makes the adjoint a plain suffix sum of the incoming weights.
    suffix = torch.flip(
        torch.cumsum(torch.flip(grad_log_spot, dims=(1,)), dim=1), dims=(1,)
    )

    grad_parameters = [torch.zeros_like(p) for p in parameters]
    for step in range(n_steps):
        with torch.enable_grad():
            state_leaf = states[step].detach().clone().requires_grad_(True)
            sigma = local_vol(step * dt, state_leaf)
            vol_factor = sqrt_dt * normals[:, step] - sigma.detach() * dt
            weighted = (suffix[:, step + 1] * vol_factor * sigma).sum()
            grads = torch.autograd.grad(
                weighted, tuple(parameters), allow_unused=True
            )
        for position, grad in enumerate(grads):
            if grad is not None:
                grad_parameters[position] = grad_parameters[position] + grad

    return suffix[:, 0].sum() / spot_zero.detach(), tuple(grad_parameters)
