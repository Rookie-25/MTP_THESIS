r"""Geometric Brownian Motion (GBM) path generation in PyTorch (CPU / CUDA).

Mathematical background
-----------------------
Under a risk-neutral measure :math:`\mathbb{Q}` the asset follows

.. math:: dS_t = \mu S_t\,dt + \sigma S_t\,dW_t,\qquad S_0 > 0,

which admits the *exact* strong solution

.. math::
    S_{t+\Delta} = S_t \exp\!\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta
                                   + \sigma\,\Delta W\right],
    \qquad \Delta W \sim \mathcal{N}(0, \Delta).

The log-Euler scheme implemented here therefore carries **no discretisation
bias**: increasing ``n_steps`` reduces neither bias nor variance of a terminal
payoff, it only refines the observation grid used later for XVA exposure
profiles.  This makes GBM the ideal Phase-1 baseline -- any error observed in
the Greeks is attributable to Monte-Carlo noise or to the differentiation
scheme, never to the SDE solver.

Design notes for reverse-mode AAD
---------------------------------
1. The Brownian increments are drawn **independently of the model parameters**
   and passed in explicitly.  This serves two purposes:

   * the pathwise derivative :math:`\partial S_T / \partial \theta` is the
     correct sensitivity of the *same* random outcome, and
   * finite-difference validation automatically uses **common random numbers**
     (CRN), without which bump-and-revalue Greeks are swamped by MC noise.

2. ``s0`` enters as a single multiplicative factor outside the ``cumsum``, so
   the adjoint of ``s0`` is one fused multiply rather than a chain through
   every time step.

3. Every operation is a standard differentiable ``torch`` op, so a single
   ``backward()`` yields sensitivities w.r.t. all of ``s0``, ``mu``, ``sigma``
   simultaneously.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
from torch import Tensor

__all__ = [
    "ScalarLike",
    "resolve_device",
    "draw_brownian_increments",
    "simulate_gbm",
    "GBMSimulator",
]

#: A model parameter may be supplied as a Python float or as a 0-dim tensor
#: (the latter being required when a gradient w.r.t. it is wanted).
ScalarLike = Union[float, Tensor]


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available compute device.

    Args:
        prefer_cuda: If ``True`` (default) return ``cuda`` whenever a CUDA
            device is visible to PyTorch, otherwise force ``cpu``.

    Returns:
        The selected :class:`torch.device`.
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _as_param(value: ScalarLike, device: torch.device, dtype: torch.dtype, name: str) -> Tensor:
    """Coerce a scalar-like model parameter to a 0-dim tensor, preserving grad.

    A tensor that is already on the requested device/dtype is returned
    unchanged (so it stays an autograd *leaf*); otherwise ``.to()`` is applied,
    through which gradients still propagate.

    Args:
        value: Python float or scalar tensor.
        device: Target device.
        dtype: Target floating dtype.
        name: Parameter name, used in error messages.

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


def draw_brownian_increments(
    n_paths: int,
    n_steps: int,
    dt: float,
    *,
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    antithetic: bool = False,
) -> Tensor:
    r"""Draw i.i.d. Brownian increments :math:`\Delta W \sim \mathcal{N}(0, \Delta t)`.

    The increments are deliberately decoupled from path generation so that the
    *same* draw can be reused across parameter perturbations (common random
    numbers), which is a prerequisite for meaningful finite-difference Greeks.

    Args:
        n_paths: Number of Monte-Carlo paths :math:`M`.
        n_steps: Number of time steps :math:`N`.
        dt: Step size :math:`\Delta t > 0`.
        device: Device on which to allocate the sample.
        dtype: Floating dtype of the sample. ``float64`` is recommended when
            the sample feeds a finite-difference comparison.
        seed: Convenience seed; a private :class:`torch.Generator` is created
            and seeded with it. Ignored when ``generator`` is supplied.
        generator: Explicit RNG. Must live on ``device``. Takes precedence
            over ``seed``.
        antithetic: If ``True``, draw :math:`M/2` normals and append their
            negation. This is an unbiased variance-reduction technique, but it
            makes the paths pairwise dependent, so sample standard errors
            computed as ``std / sqrt(M)`` become conservative.

    Returns:
        Tensor of shape ``(n_paths, n_steps)``.

    Raises:
        ValueError: On non-positive sizes, non-positive ``dt``, or an odd
            ``n_paths`` combined with ``antithetic=True``.
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    device = torch.device(device)
    if generator is None and seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError(f"antithetic sampling requires an even n_paths, got {n_paths}")
        half = torch.randn(
            (n_paths // 2, n_steps), device=device, dtype=dtype, generator=generator
        )
        normals = torch.cat((half, -half), dim=0)
    else:
        normals = torch.randn(
            (n_paths, n_steps), device=device, dtype=dtype, generator=generator
        )

    return normals * math.sqrt(dt)


def simulate_gbm(
    s0: ScalarLike,
    mu: ScalarLike,
    sigma: ScalarLike,
    dW: Tensor,
    dt: float,
) -> Tensor:
    r"""Simulate GBM paths from pre-drawn Brownian increments (exact scheme).

    Args:
        s0: Initial spot :math:`S_0`. Pass a tensor with ``requires_grad=True``
            to obtain Delta.
        mu: Drift :math:`\mu`. Equal to the risk-free rate under
            :math:`\mathbb{Q}`. Pass a tensor to obtain Rho's drift component.
        sigma: Volatility :math:`\sigma`. Pass a tensor to obtain Vega.
        dW: Brownian increments of shape ``(n_paths, n_steps)``, as produced by
            :func:`draw_brownian_increments`. Its device and dtype define those
            of the output.
        dt: Step size :math:`\Delta t` used to generate ``dW``.

    Returns:
        Paths of shape ``(n_paths, n_steps + 1)``. Column ``0`` is
        :math:`S_0` for every path; column ``k`` is :math:`S_{k\Delta t}`.

    Raises:
        ValueError: If ``dW`` is not 2-dimensional or ``dt`` is non-positive.
    """
    if dW.ndim != 2:
        raise ValueError(f"dW must be 2-dimensional (n_paths, n_steps), got {tuple(dW.shape)}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    device, dtype = dW.device, dW.dtype
    s0_t = _as_param(s0, device, dtype, "s0")
    mu_t = _as_param(mu, device, dtype, "mu")
    sigma_t = _as_param(sigma, device, dtype, "sigma")

    # Ito-corrected drift per step; broadcast against dW.
    log_drift = (mu_t - 0.5 * sigma_t * sigma_t) * dt
    log_increments = log_drift + sigma_t * dW
    log_path = torch.cumsum(log_increments, dim=1)

    # Prepend log(S_0 / S_0) = 0 so that column 0 evaluates to exactly S_0.
    zero_column = log_path.new_zeros((log_path.shape[0], 1))
    log_path = torch.cat((zero_column, log_path), dim=1)

    return s0_t * torch.exp(log_path)


@dataclass
class GBMSimulator:
    """Stateless configuration object for GBM Monte-Carlo simulation.

    The simulator owns the *time discretisation and execution context* only;
    model parameters are passed per call so that autograd leaves can be swapped
    freely between valuations (needed by both AAD and bump-and-revalue).

    Attributes:
        maturity: Simulation horizon :math:`T` in years.
        n_steps: Number of equal time steps :math:`N`.
        device: Execution device. Defaults to CUDA when available.
        dtype: Floating precision. ``float64`` is the default because Phase 1
            is a correctness baseline; switch to ``float32`` for GPU
            throughput benchmarks once the kernels land in Phase 3.
        antithetic: Whether :meth:`draw_increments` uses antithetic variates.
    """

    maturity: float
    n_steps: int
    device: torch.device = field(default_factory=resolve_device)
    dtype: torch.dtype = torch.float64
    antithetic: bool = False

    def __post_init__(self) -> None:
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        self.device = torch.device(self.device)

    @property
    def dt(self) -> float:
        """Step size :math:`\\Delta t = T / N`."""
        return self.maturity / self.n_steps

    def time_grid(self) -> Tensor:
        """Return the observation grid ``[0, dt, ..., T]`` of shape ``(N + 1,)``."""
        return torch.linspace(
            0.0, self.maturity, self.n_steps + 1, device=self.device, dtype=self.dtype
        )

    def draw_increments(
        self,
        n_paths: int,
        *,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Draw Brownian increments consistent with this simulator's grid.

        Args:
            n_paths: Number of Monte-Carlo paths.
            seed: Optional convenience seed (see
                :func:`draw_brownian_increments`).
            generator: Optional explicit RNG on ``self.device``.

        Returns:
            Tensor of shape ``(n_paths, n_steps)``.
        """
        return draw_brownian_increments(
            n_paths,
            self.n_steps,
            self.dt,
            device=self.device,
            dtype=self.dtype,
            seed=seed,
            generator=generator,
            antithetic=self.antithetic,
        )

    def simulate(
        self,
        s0: ScalarLike,
        mu: ScalarLike,
        sigma: ScalarLike,
        *,
        dW: Optional[Tensor] = None,
        n_paths: Optional[int] = None,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Generate GBM paths.

        Supply ``dW`` to reuse a fixed Brownian sample across valuations
        (required for common-random-number finite differences), or supply
        ``n_paths`` to draw a fresh one.

        Args:
            s0: Initial spot.
            mu: Drift.
            sigma: Volatility.
            dW: Pre-drawn increments of shape ``(n_paths, n_steps)``.
            n_paths: Number of paths to draw when ``dW`` is omitted.
            seed: Seed forwarded to :meth:`draw_increments`.
            generator: RNG forwarded to :meth:`draw_increments`.

        Returns:
            Paths of shape ``(n_paths, n_steps + 1)``.

        Raises:
            ValueError: If neither or both of ``dW`` and ``n_paths`` are given,
                or if ``dW`` has the wrong number of time steps.
        """
        if (dW is None) == (n_paths is None):
            raise ValueError("provide exactly one of 'dW' or 'n_paths'")
        if dW is None:
            dW = self.draw_increments(int(n_paths), seed=seed, generator=generator)
        elif dW.shape[1] != self.n_steps:
            raise ValueError(
                f"dW has {dW.shape[1]} time steps but simulator expects {self.n_steps}"
            )
        return simulate_gbm(s0, mu, sigma, dW, self.dt)
