r"""Arbitrage-free local volatility surface, differentiable end to end.

Scope
=====
Phases 1-5 assumed constant :math:`\sigma`. This module replaces it with a
state-dependent Dupire local volatility :math:`\sigma_{LV}(t, S_t)` derived from
a calibrated implied-volatility surface, while keeping the whole chain
differentiable so CVA sensitivities propagate back to raw surface parameters.

Parameterisation: SSVI
======================
The total implied variance :math:`w(k,T) = \sigma_{\text{imp}}^2(k,T)\,T`, with
:math:`k = \log(K/F_T)`, is represented by the *Surface SVI* form of Gatheral &
Jacquier (2014):

.. math::
    w(k, T) = \frac{\theta_T}{2}\left\{
        1 + \rho\,\varphi(\theta_T)\,k
        + \sqrt{\left(\varphi(\theta_T)k + \rho\right)^2 + 1 - \rho^2}
    \right\},

which satisfies :math:`w(0,T) = \theta_T`, so :math:`\theta_T` *is* the ATM
total variance. SSVI is chosen over a free-form spline for one decisive reason:
it has **known sufficient conditions for absence of arbitrage** in terms of its
parameters, so the penalty terms below are a safety net rather than the only
line of defence.

Two structural guarantees are built in rather than penalised:

* :math:`\theta_T` is constructed as a cumulative sum of ``softplus``
  increments, hence **non-decreasing in :math:`T` by construction**. This
  discharges the ATM part of the calendar condition exactly, with no penalty
  weight to tune.
* :math:`\rho \in (-1,1)` via ``tanh``, :math:`\eta > 0` via ``softplus``,
  :math:`\gamma \in (0,1)` via ``sigmoid``. The feasible set is the whole
  parameter space, so an optimiser can never step outside it.

Arbitrage conditions
====================
**Calendar spread.** A surface is free of calendar arbitrage iff total variance
is non-decreasing in maturity at fixed log-moneyness:

.. math:: \frac{\partial w(k,T)}{\partial T} \ge 0 \qquad \forall k, T.

**Butterfly.** Free of butterfly arbitrage iff the Gatheral-Jacquier function

.. math::
    g(k) = \left(1 - \frac{k\,\partial_k w}{2w}\right)^2
         - \frac{(\partial_k w)^2}{4}\left(\frac{1}{w} + \frac14\right)
         + \frac{\partial_k^2 w}{2}

is non-negative everywhere. Note the factor **2** in the first term's
denominator: :math:`k\,\partial_k w / (2w)`, not :math:`k\,\partial_k w / w`.
This is not cosmetic. The implied risk-neutral density of
:math:`\log(S_T/F_T)` is

.. math::
    p(k) = \frac{g(k)}{\sqrt{2\pi w(k)}}
           \exp\!\left(-\tfrac12 d_-(k)^2\right),
    \qquad d_-(k) = -\frac{k}{\sqrt{w}} - \frac{\sqrt{w}}{2},

and :math:`g \ge 0 \iff p \ge 0`. Dropping the 2 makes :math:`\int p\,dk`
differ from 1 -- measured at 0.9868 on a skewed SSVI slice
(:math:`\theta=0.04, \rho=-0.7, \varphi=1.5`), i.e. a 1.3% mass error, so the
"density" is not one. ``tests/test_phase6.py`` pins this down.

Dupire, and why the two conditions are the *same* condition
===========================================================
In total-variance coordinates Dupire's local variance is

.. math:: \sigma_{LV}^2(k, T) = \frac{\partial_T w(k,T)}{g(k,T)} .

So the numerator is exactly the calendar condition and the denominator is
exactly the butterfly function: **the two no-arbitrage constraints are precisely
the conditions for the local variance to be real, finite and non-negative.**
Enforcing them is not an extra requirement bolted onto calibration -- it is what
makes the local volatility well defined at all. That framing is worth stating
explicitly in any write-up.

Penalty design
==============
Both conditions are inequalities, and gradient descent needs them differentiable
*including at the boundary*. Two mechanisms, with different roles:

* **Smooth rectifier (softplus hinge)** --
  :math:`\beta^{-1}\log(1+e^{-\beta c})` for a constraint :math:`c \ge 0`. Finite
  and smooth everywhere, including at infeasible points, so it can *restore*
  feasibility from a bad initialisation. This is the default.
* **Log-barrier** -- :math:`-\log(c)`. Infinite gradient as :math:`c \to 0^+`,
  which enforces strict interiority, but it is undefined for :math:`c \le 0` so
  it cannot be used until the iterate is already feasible.

The recommended schedule is therefore hinge-first, barrier-second: run the hinge
to reach feasibility, then switch on a barrier with a decreasing weight to polish
while staying strictly inside. :class:`ArbitragePenalty` implements both and the
crossover.

References
----------
Gatheral, J., Jacquier, A. (2014). *Arbitrage-free SVI volatility surfaces*.
Quantitative Finance 14(1), 59-71. -- SSVI, :math:`g(k)`, and the sufficient
conditions in Theorems 4.1-4.2.
Gatheral, J. (2006). *The Volatility Surface*, Wiley. -- Dupire in
total-variance coordinates.
Dupire, B. (1994). *Pricing with a smile*. Risk 7(1), 18-20.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

__all__ = [
    "ATMTotalVariance",
    "SSVISurface",
    "butterfly_g",
    "implied_density",
    "calendar_slope",
    "ArbitragePenalty",
    "LocalVolatilitySurface",
    "SurfaceCalibrationResult",
    "calibrate_surface",
    "LocalVolFit",
    "local_vol_sampling_grid",
    "fit_local_vol_params",
    "evaluate_parametric_local_vol",
    "DEFAULT_FIT_SIGMA_WIDTH",
    "chebyshev_basis",
    "chebyshev_basis_derivative",
    "evaluate_chebyshev_local_vol",
]


# ==========================================================================
# ATM term structure -- monotone by construction
# ==========================================================================
class ATMTotalVariance(nn.Module):
    r"""Non-decreasing ATM total variance :math:`\theta_T`, by construction.

    Rather than penalising :math:`\partial_T\theta \ge 0`, the term structure is
    built as a cumulative sum of strictly positive increments:

    .. math:: \theta_{T_i} = \theta_{\min}
              + \sum_{j \le i} \operatorname{softplus}(u_j)\,\Delta T_j .

    The map from the unconstrained :math:`u` to a monotone :math:`\theta` is
    surjective onto the feasible set and smooth, so no optimiser step can leave
    it and no penalty weight needs tuning. Values off the knot grid come from
    linear interpolation in :math:`T`, which preserves monotonicity.

    Attributes:
        maturities: Knot maturities, shape ``(n_maturities,)``, strictly
            increasing.
        raw_increments: Unconstrained parameters, shape ``(n_maturities,)``.
        floor: Lower bound :math:`\theta_{\min} > 0` keeping the surface away
            from a degenerate zero-variance slice.
    """

    def __init__(
        self,
        maturities: Tensor,
        initial_total_variance: Optional[Tensor] = None,
        floor: float = 1e-6,
    ) -> None:
        """Initialise the term structure.

        Args:
            maturities: Strictly increasing knot maturities, shape ``(n,)``.
            initial_total_variance: Optional starting :math:`\\theta` at the
                knots, shape ``(n,)``. Must be non-decreasing and above
                ``floor``. Defaults to a flat 20% vol.
            floor: Strictly positive lower bound on :math:`\\theta`.

        Raises:
            ValueError: If ``maturities`` is not 1-D and increasing, if
                ``floor`` is non-positive, or if an initial term structure is
                supplied that is decreasing or below the floor.
        """
        super().__init__()
        if maturities.ndim != 1 or maturities.numel() < 1:
            raise ValueError("maturities must be a non-empty 1-D tensor")
        if maturities.numel() > 1 and bool((maturities[1:] <= maturities[:-1]).any()):
            raise ValueError("maturities must be strictly increasing")
        if floor <= 0.0:
            raise ValueError(f"floor must be positive, got {floor}")

        self.register_buffer("maturities", maturities.clone())
        self.floor = float(floor)

        gaps = torch.diff(maturities, prepend=maturities.new_zeros(1))
        if initial_total_variance is None:
            initial_total_variance = 0.04 * maturities  # 20% flat vol
        if initial_total_variance.shape != maturities.shape:
            raise ValueError("initial_total_variance must match maturities' shape")
        increments = torch.diff(
            initial_total_variance,
            prepend=initial_total_variance.new_full((1,), floor),
        )
        if bool((increments < 0).any()):
            raise ValueError("initial_total_variance must be non-decreasing")

        # Invert softplus to recover the unconstrained parameterisation.
        scaled = torch.clamp(increments / torch.clamp(gaps, min=1e-12), min=1e-8)
        self.raw_increments = nn.Parameter(scaled + torch.log(-torch.expm1(-scaled)))

    def knot_values(self) -> Tensor:
        r"""Return :math:`\theta` at the knot maturities, shape ``(n,)``."""
        gaps = torch.diff(
            self.maturities, prepend=self.maturities.new_zeros(1)
        )
        increments = torch.nn.functional.softplus(self.raw_increments) * gaps
        return self.floor + torch.cumsum(increments, dim=0)

    def forward(self, maturity: Tensor) -> Tensor:
        r"""Evaluate :math:`\theta_T` by linear interpolation in :math:`T`.

        Args:
            maturity: Maturities to evaluate, any shape.

        Returns:
            :math:`\theta_T` with the same shape as ``maturity``, differentiable
            w.r.t. the module parameters.
        """
        knots = self.knot_values()
        grid = self.maturities
        flat = maturity.reshape(-1)

        if grid.numel() == 1:
            # Single knot: scale linearly in T so theta(0) = 0 is respected.
            return (knots[0] * flat / torch.clamp(grid[0], min=1e-12)).reshape(
                maturity.shape
            )

        index = torch.clamp(
            torch.searchsorted(grid.contiguous(), flat.contiguous(), right=True),
            1, grid.numel() - 1,
        )
        left_t, right_t = grid[index - 1], grid[index]
        left_v, right_v = knots[index - 1], knots[index]
        weight = (flat - left_t) / torch.clamp(right_t - left_t, min=1e-12)
        return (left_v + weight * (right_v - left_v)).reshape(maturity.shape)


# ==========================================================================
# SSVI total variance surface
# ==========================================================================
class SSVISurface(nn.Module):
    r"""Surface SVI total variance with all parameters in their feasible sets.

    .. math::
        w(k,T) = \frac{\theta_T}{2}\left\{1 + \rho\varphi k
                 + \sqrt{(\varphi k + \rho)^2 + 1 - \rho^2}\right\},
        \qquad \varphi(\theta) = \frac{\eta}{\theta^{\gamma}(1+\theta)^{1-\gamma}} .

    Attributes:
        atm: The :class:`ATMTotalVariance` term structure.
        raw_rho: Unconstrained; :math:`\rho = \tanh(\text{raw})\in(-1,1)`.
        raw_eta: Unconstrained; :math:`\eta = \operatorname{softplus} > 0`.
        raw_gamma: Unconstrained; :math:`\gamma = \operatorname{sigmoid}\in(0,1)`.
    """

    def __init__(
        self,
        atm: ATMTotalVariance,
        rho: float = -0.3,
        eta: float = 1.0,
        gamma: float = 0.5,
    ) -> None:
        """Initialise SSVI.

        Args:
            atm: ATM total-variance term structure.
            rho: Initial correlation in ``(-1, 1)``.
            eta: Initial positive skew scale.
            gamma: Initial power-law exponent in ``(0, 1)``.

        Raises:
            ValueError: If any initial value is outside its feasible set.
        """
        super().__init__()
        if not -1.0 < rho < 1.0:
            raise ValueError(f"rho must lie in (-1, 1), got {rho}")
        if eta <= 0.0:
            raise ValueError(f"eta must be positive, got {eta}")
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must lie in (0, 1), got {gamma}")

        self.atm = atm
        self.raw_rho = nn.Parameter(torch.atanh(torch.tensor(float(rho))))
        self.raw_eta = nn.Parameter(
            torch.tensor(float(eta) + math.log(-math.expm1(-float(eta))))
        )
        self.raw_gamma = nn.Parameter(
            torch.logit(torch.tensor(float(gamma)))
        )

    #: Safety margin keeping the constrained parameters strictly interior.
    #:
    #: A bare ``tanh``/``sigmoid`` is NOT sufficient. Both saturate to exactly
    #: their endpoints in floating point once the raw parameter grows: at
    #: ``|raw| > ~19`` in float64, ``tanh`` returns exactly ``1.0``. That puts
    #: :math:`\rho` on the boundary, where :math:`1-\rho^2 = 0`, the SSVI square
    #: root degenerates to :math:`|\varphi k + \rho|` -- non-differentiable at
    #: :math:`\varphi k = -\rho` -- and both SSVI butterfly conditions collapse.
    #: Optimisers reach such raw values routinely during a bad line search, so
    #: this margin is a correctness requirement, not decoration.
    PARAMETER_MARGIN = 1e-6

    @property
    def rho(self) -> Tensor:
        r""":math:`\rho \in (-1+\epsilon,\, 1-\epsilon)`, strictly interior."""
        return (1.0 - self.PARAMETER_MARGIN) * torch.tanh(self.raw_rho)

    @property
    def eta(self) -> Tensor:
        r""":math:`\eta > 0`, bounded away from zero.

        ``softplus`` underflows to exactly ``0.0`` for very negative inputs,
        which would make :math:`\varphi \equiv 0` and flatten the smile to a
        degenerate slice with no skew at all.
        """
        return torch.nn.functional.softplus(self.raw_eta) + self.PARAMETER_MARGIN

    @property
    def gamma(self) -> Tensor:
        r""":math:`\gamma \in (\epsilon,\, 1-\epsilon)`, strictly interior."""
        margin = self.PARAMETER_MARGIN
        return margin + (1.0 - 2.0 * margin) * torch.sigmoid(self.raw_gamma)

    def phi(self, theta: Tensor) -> Tensor:
        r"""Power-law :math:`\varphi(\theta) = \eta\,\theta^{-\gamma}(1+\theta)^{\gamma-1}`."""
        safe_theta = torch.clamp(theta, min=1e-10)
        gamma = self.gamma
        return self.eta * safe_theta.pow(-gamma) * (1.0 + safe_theta).pow(gamma - 1.0)

    def forward(self, log_moneyness: Tensor, maturity: Tensor) -> Tensor:
        r"""Total implied variance :math:`w(k, T)`.

        Args:
            log_moneyness: :math:`k = \log(K/F_T)`, broadcastable with
                ``maturity``.
            maturity: :math:`T` in years.

        Returns:
            :math:`w(k,T)`, broadcast shape, strictly positive.
        """
        theta = self.atm(maturity)
        phi = self.phi(theta)
        scaled = phi * log_moneyness
        root = torch.sqrt(
            torch.clamp((scaled + self.rho) ** 2 + 1.0 - self.rho**2, min=1e-16)
        )
        return 0.5 * theta * (1.0 + self.rho * scaled + root)

    def ssvi_butterfly_margins(self, maturity: Tensor) -> Tuple[Tensor, Tensor]:
        r"""The two SSVI sufficient conditions, as slack values to keep positive.

        Gatheral & Jacquier (2014), Theorem 4.2: the slice is free of butterfly
        arbitrage if

        .. math::
            \theta\varphi(\theta)(1+|\rho|) < 4
            \quad\text{and}\quad
            \theta\varphi(\theta)^2(1+|\rho|) \le 4 .

        Args:
            maturity: Maturities to evaluate.

        Returns:
            ``(slack_linear, slack_quadratic)``, each ``4 - lhs``. Both must be
            positive. These are *sufficient*, not necessary, so they are a
            stronger requirement than :func:`butterfly_g` alone.
        """
        theta = self.atm(maturity)
        phi = self.phi(theta)
        factor = 1.0 + self.rho.abs()
        return 4.0 - theta * phi * factor, 4.0 - theta * phi**2 * factor


# ==========================================================================
# Arbitrage diagnostics
# ==========================================================================
def butterfly_g(
    log_moneyness: Tensor,
    total_variance: Tensor,
    first_derivative: Tensor,
    second_derivative: Tensor,
) -> Tensor:
    r"""Gatheral-Jacquier butterfly function :math:`g(k)`.

    .. math::
        g(k) = \left(1 - \frac{k\,w'}{2w}\right)^2
             - \frac{(w')^2}{4}\left(\frac{1}{w} + \frac14\right)
             + \frac{w''}{2}

    Args:
        log_moneyness: :math:`k`.
        total_variance: :math:`w(k)`, strictly positive.
        first_derivative: :math:`\partial_k w`.
        second_derivative: :math:`\partial_k^2 w`.

    Returns:
        :math:`g(k)`, same shape. Non-negative iff the slice is free of
        butterfly arbitrage.

    Note:
        The ``2w`` in the first term is essential -- with ``w`` instead, the
        implied density fails to normalise (measured 0.9868 on a skewed slice).
    """
    safe_w = torch.clamp(total_variance, min=1e-12)
    term_one = (1.0 - log_moneyness * first_derivative / (2.0 * safe_w)) ** 2
    term_two = (first_derivative**2 / 4.0) * (1.0 / safe_w + 0.25)
    return term_one - term_two + second_derivative / 2.0


def implied_density(
    log_moneyness: Tensor,
    total_variance: Tensor,
    first_derivative: Tensor,
    second_derivative: Tensor,
) -> Tensor:
    r"""Risk-neutral density of :math:`\log(S_T/F_T)` implied by a slice.

    .. math::
        p(k) = \frac{g(k)}{\sqrt{2\pi w}}
               \exp\!\left(-\tfrac12 d_-^2\right),
        \qquad d_- = -\frac{k}{\sqrt{w}} - \frac{\sqrt{w}}{2}

    Args:
        log_moneyness: :math:`k`.
        total_variance: :math:`w(k)`.
        first_derivative: :math:`\partial_k w`.
        second_derivative: :math:`\partial_k^2 w`.

    Returns:
        The density at each :math:`k`. Integrating it over a wide :math:`k`
        range is the sharpest single check on a slice, and is what
        ``tests/test_phase6.py`` uses to validate :func:`butterfly_g`.
    """
    safe_w = torch.clamp(total_variance, min=1e-12)
    g = butterfly_g(
        log_moneyness, safe_w, first_derivative, second_derivative
    )
    d_minus = -log_moneyness / torch.sqrt(safe_w) - 0.5 * torch.sqrt(safe_w)
    return g / torch.sqrt(2.0 * math.pi * safe_w) * torch.exp(-0.5 * d_minus**2)


def calendar_slope(
    surface: SSVISurface, log_moneyness: Tensor, maturity: Tensor
) -> Tensor:
    r"""Compute :math:`\partial_T w(k,T)` by automatic differentiation.

    Args:
        surface: The total-variance surface.
        log_moneyness: :math:`k`, broadcastable with ``maturity``.
        maturity: :math:`T`, will be differentiated against.

    Returns:
        :math:`\partial_T w`, same broadcast shape. Non-negative iff free of
        calendar-spread arbitrage.
    """
    maturity = maturity.detach().clone().requires_grad_(True)
    total_variance = surface(log_moneyness, maturity)
    (slope,) = torch.autograd.grad(
        total_variance.sum(), maturity, create_graph=torch.is_grad_enabled()
    )
    return slope


@dataclass(frozen=True)
class _PenaltyTerms:
    """Component penalties, kept separate so calibration logs are diagnosable."""

    calendar: Tensor
    butterfly: Tensor
    ssvi_linear: Tensor
    ssvi_quadratic: Tensor

    @property
    def total(self) -> Tensor:
        return (
            self.calendar + self.butterfly + self.ssvi_linear + self.ssvi_quadratic
        )


class ArbitragePenalty(nn.Module):
    r"""Differentiable penalty enforcing the no-arbitrage inequalities.

    Two mechanisms with different jobs:

    * ``mode="hinge"`` (default) -- smooth rectifier
      :math:`\beta^{-1}\log(1 + e^{-\beta c})`, finite and smooth for *all*
      :math:`c`, so it can pull an infeasible iterate back into the feasible
      set. Approaches :math:`\max(-c, 0)` as :math:`\beta \to \infty`.
    * ``mode="barrier"`` -- :math:`-\log(c/\text{scale})`, whose gradient blows
      up as :math:`c \to 0^+`, enforcing strict interiority. Undefined for
      :math:`c \le 0`, so it may only be enabled once feasible.

    Attributes:
        weight: Overall multiplier on the penalty.
        sharpness: :math:`\beta` for the hinge. Larger is a tighter
            approximation to the true hinge but a stiffer objective.
        margin: Required slack; the constraint enforced is
            :math:`c \ge \text{margin}`, which keeps the iterate off the
            boundary where the local variance would blow up.
        mode: ``"hinge"`` or ``"barrier"``.
    """

    def __init__(
        self,
        weight: float = 1.0,
        sharpness: float = 200.0,
        margin: float = 1e-4,
        mode: str = "hinge",
    ) -> None:
        """Initialise the penalty.

        Args:
            weight: Overall multiplier.
            sharpness: Hinge sharpness :math:`\\beta > 0`.
            margin: Required slack, :math:`\\ge 0`.
            mode: ``"hinge"`` or ``"barrier"``.

        Raises:
            ValueError: On a non-positive weight/sharpness, negative margin, or
                unknown mode.
        """
        super().__init__()
        if weight <= 0.0:
            raise ValueError(f"weight must be positive, got {weight}")
        if sharpness <= 0.0:
            raise ValueError(f"sharpness must be positive, got {sharpness}")
        if margin < 0.0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        if mode not in ("hinge", "barrier"):
            raise ValueError(f"mode must be 'hinge' or 'barrier', got {mode!r}")

        self.weight = float(weight)
        self.sharpness = float(sharpness)
        self.margin = float(margin)
        self.mode = mode

    def _apply_to(self, slack: Tensor) -> Tensor:
        """Penalise violations of ``slack >= margin``.

        Args:
            slack: Constraint values :math:`c`; feasible where
                ``c >= margin``.

        Returns:
            A scalar penalty, mean-reduced so the value is independent of how
            many sample points were supplied.
        """
        shifted = slack - self.margin
        if self.mode == "hinge":
            # softplus(-beta * c) / beta -- smooth, finite for all c.
            return (
                torch.nn.functional.softplus(-self.sharpness * shifted)
                / self.sharpness
            ).mean()
        # Barrier: only meaningful strictly inside the feasible set.
        return -torch.log(torch.clamp(shifted, min=1e-12)).mean()

    def forward(
        self,
        surface: SSVISurface,
        log_moneyness: Tensor,
        maturity: Tensor,
    ) -> _PenaltyTerms:
        r"""Evaluate all four penalty components on a sample grid.

        Args:
            surface: The surface under calibration.
            log_moneyness: Sample :math:`k` values, shape ``(n_k,)``.
            maturity: Sample :math:`T` values, shape ``(n_T,)``.

        Returns:
            A :class:`_PenaltyTerms` holding the calendar, butterfly and two
            SSVI-condition components separately, so a calibration log can show
            *which* constraint is binding rather than one opaque number.
        """
        grid_k = log_moneyness.reshape(1, -1).expand(maturity.numel(), -1)
        grid_t = maturity.reshape(-1, 1).expand(-1, log_moneyness.numel())

        # Calendar: d w / d T >= 0.
        maturity_leaf = grid_t.detach().clone().requires_grad_(True)
        total_variance = surface(grid_k, maturity_leaf)
        (slope,) = torch.autograd.grad(
            total_variance.sum(), maturity_leaf, create_graph=True
        )
        # Re-evaluate on the graph so the penalty reaches the parameters.
        calendar = self._apply_to(surface(grid_k, grid_t) * 0.0 + slope)

        # Butterfly: g(k) >= 0, with k-derivatives by autograd.
        moneyness_leaf = grid_k.detach().clone().requires_grad_(True)
        w_on_k = surface(moneyness_leaf, grid_t)
        (first,) = torch.autograd.grad(w_on_k.sum(), moneyness_leaf, create_graph=True)
        (second,) = torch.autograd.grad(first.sum(), moneyness_leaf, create_graph=True)
        butterfly = self._apply_to(
            butterfly_g(moneyness_leaf, w_on_k, first, second)
        )

        linear_slack, quadratic_slack = surface.ssvi_butterfly_margins(maturity)
        return _PenaltyTerms(
            calendar=self.weight * calendar,
            butterfly=self.weight * butterfly,
            ssvi_linear=self.weight * self._apply_to(linear_slack),
            ssvi_quadratic=self.weight * self._apply_to(quadratic_slack),
        )


# ==========================================================================
# Local volatility
# ==========================================================================
class LocalVolatilitySurface(nn.Module):
    r"""Dupire local volatility obtained by autodiff of the total-variance surface.

    .. math:: \sigma_{LV}^2(k,T) = \frac{\partial_T w(k,T)}{g(k,T)}

    All three derivatives of :math:`w` come from ``torch.autograd``, so the
    local variance is differentiable with respect to every surface parameter and
    CVA sensitivities reach the raw calibration inputs.

    Attributes:
        surface: The SSVI total-variance surface.
        rate: Risk-free rate :math:`\mu`, for the forward :math:`F_t`.
        dividend_yield: Continuous dividend yield :math:`q`.
        variance_floor: Lower clamp on the local variance. Necessary in
            practice: away from the calibrated region :math:`g` can approach
            zero and the ratio becomes numerically violent. Clamping is honest
            regularisation, not a silent fix -- :meth:`diagnostics` reports how
            often it binds.
    """

    def __init__(
        self,
        surface: SSVISurface,
        rate: float = 0.0,
        dividend_yield: float = 0.0,
        variance_floor: float = 1e-4,
    ) -> None:
        """Initialise the local volatility surface.

        Args:
            surface: Calibrated (or in-calibration) SSVI surface.
            rate: Risk-free rate :math:`\\mu`.
            dividend_yield: Dividend yield :math:`q`.
            variance_floor: Strictly positive floor on local variance.

        Raises:
            ValueError: If ``variance_floor`` is non-positive.
        """
        super().__init__()
        if variance_floor <= 0.0:
            raise ValueError(
                f"variance_floor must be positive, got {variance_floor}"
            )
        self.surface = surface
        self.rate = float(rate)
        self.dividend_yield = float(dividend_yield)
        self.variance_floor = float(variance_floor)

    def log_moneyness(self, spot: Tensor, time: Tensor, spot_zero: float) -> Tensor:
        r"""Map a simulated state to surface coordinates.

        The local volatility at :math:`(t, S_t)` is read off the surface at
        strike :math:`K = S_t`, so

        .. math::
            k = \log\frac{S_t}{F_t},
            \qquad F_t = S_0 e^{(\mu - q)t}.

        Getting this mapping wrong -- using :math:`\log(S_t/S_0)` instead of
        :math:`\log(S_t/F_t)` -- silently shifts the whole smile by the forward
        drift, which is a large error at long maturities and an invisible one at
        :math:`t=0`.

        Args:
            spot: :math:`S_t`, any shape.
            time: :math:`t`, broadcastable with ``spot``.
            spot_zero: :math:`S_0`.

        Returns:
            :math:`k`, broadcast shape.
        """
        forward = math.log(spot_zero) + (self.rate - self.dividend_yield) * time
        return torch.log(torch.clamp(spot, min=1e-12)) - forward

    def local_variance_from_coordinates(
        self, log_moneyness: Tensor, maturity: Tensor
    ) -> Tensor:
        r"""Local variance at surface coordinates, via autodiff of :math:`w`.

        Args:
            log_moneyness: :math:`k`, broadcastable with ``maturity``.
            maturity: :math:`T > 0`.

        Returns:
            :math:`\sigma_{LV}^2`, floored at ``variance_floor``.
        """
        create_graph = torch.is_grad_enabled()

        moneyness_leaf = log_moneyness.detach().clone().requires_grad_(True)
        maturity_leaf = maturity.detach().clone().requires_grad_(True)

        # Broadcast once so both leaves see the same shape.
        broadcast_k, broadcast_t = torch.broadcast_tensors(
            moneyness_leaf, maturity_leaf
        )
        total_variance = self.surface(broadcast_k, broadcast_t)

        (first,) = torch.autograd.grad(
            total_variance.sum(), moneyness_leaf, create_graph=True
        )
        (second,) = torch.autograd.grad(
            first.sum(), moneyness_leaf, create_graph=True
        )
        (slope,) = torch.autograd.grad(
            total_variance.sum(), maturity_leaf, create_graph=True
        )

        first_b, second_b, slope_b = torch.broadcast_tensors(
            first, second, slope
        )
        g = butterfly_g(broadcast_k, total_variance, first_b, second_b)

        # A vanishing g means the slice is at the edge of butterfly arbitrage;
        # clamping keeps the ratio finite. diagnostics() reports the frequency.
        variance = slope_b / torch.clamp(g, min=1e-8)
        result = torch.clamp(variance, min=self.variance_floor)
        return result if create_graph else result.detach()

    def local_volatility(
        self, spot: Tensor, time: Tensor, spot_zero: float
    ) -> Tensor:
        r"""Local volatility :math:`\sigma_{LV}(t, S_t)` at simulated states.

        Args:
            spot: :math:`S_t`, any shape.
            time: :math:`t`, broadcastable with ``spot``. Must be positive;
                the surface is undefined at :math:`T=0`.
            spot_zero: :math:`S_0`, defining the forward.

        Returns:
            :math:`\sigma_{LV}`, broadcast shape.
        """
        safe_time = torch.clamp(time, min=1e-8)
        moneyness = self.log_moneyness(spot, safe_time, spot_zero)
        return torch.sqrt(
            self.local_variance_from_coordinates(moneyness, safe_time)
        )

    def diagnostics(
        self, log_moneyness: Tensor, maturity: Tensor
    ) -> dict:
        """Report where the surface is straining, for calibration monitoring.

        Args:
            log_moneyness: Sample :math:`k`, shape ``(n_k,)``.
            maturity: Sample :math:`T`, shape ``(n_T,)``.

        Returns:
            A dict with the worst calendar slope, the worst :math:`g`, the
            fraction of grid points where the variance floor binds, and the two
            SSVI slacks. Every value is a plain float, ready for MLflow.
        """
        grid_k = log_moneyness.reshape(1, -1).expand(maturity.numel(), -1)
        grid_t = maturity.reshape(-1, 1).expand(-1, log_moneyness.numel())

        with torch.enable_grad():
            moneyness_leaf = grid_k.detach().clone().requires_grad_(True)
            maturity_leaf = grid_t.detach().clone().requires_grad_(True)
            total_variance = self.surface(moneyness_leaf, maturity_leaf)
            (first,) = torch.autograd.grad(
                total_variance.sum(), moneyness_leaf, create_graph=True
            )
            (second,) = torch.autograd.grad(
                first.sum(), moneyness_leaf, create_graph=True
            )
            (slope,) = torch.autograd.grad(total_variance.sum(), maturity_leaf)
            g = butterfly_g(
                moneyness_leaf, total_variance, first, second
            ).detach()

        variance = slope.detach() / torch.clamp(g, min=1e-8)
        linear_slack, quadratic_slack = self.surface.ssvi_butterfly_margins(maturity)

        return {
            "min_calendar_slope": float(slope.detach().min()),
            "min_butterfly_g": float(g.min()),
            "variance_floor_fraction": float(
                (variance < self.variance_floor).to(variance.dtype).mean()
            ),
            "min_ssvi_linear_slack": float(linear_slack.detach().min()),
            "min_ssvi_quadratic_slack": float(quadratic_slack.detach().min()),
            "rho": float(self.surface.rho.detach()),
            "eta": float(self.surface.eta.detach()),
            "gamma": float(self.surface.gamma.detach()),
        }


# ==========================================================================
# Calibration
# ==========================================================================
@dataclass
class SurfaceCalibrationResult:
    """Outcome of a calibration run, shaped for experiment tracking.

    Attributes:
        surface: The fitted SSVI surface.
        history: Per-iteration scalars (loss components and diagnostics).
        final_rmse: Root-mean-square implied-vol error, in vol points.
        iterations: Number of optimiser steps taken.
    """

    surface: SSVISurface
    history: list
    final_rmse: float
    iterations: int


def calibrate_surface(
    surface: SSVISurface,
    log_moneyness: Tensor,
    maturity: Tensor,
    implied_volatility: Tensor,
    *,
    weights: Optional[Tensor] = None,
    penalty: Optional[ArbitragePenalty] = None,
    iterations: int = 300,
    learning_rate: float = 5e-2,
    log_every: int = 25,
) -> SurfaceCalibrationResult:
    r"""Fit SSVI to an implied-volatility chain under arbitrage penalties.

    The objective is weighted squared error in *total variance* space plus the
    arbitrage penalties. Fitting in total variance rather than implied vol
    avoids a square root in the loss, which conditions the gradient better at
    short maturities where :math:`w \to 0`.

    Args:
        surface: Surface to fit, modified in place.
        log_moneyness: Observed :math:`k`, shape ``(n_quotes,)``.
        maturity: Observed :math:`T`, shape ``(n_quotes,)``.
        implied_volatility: Observed :math:`\sigma_{\text{imp}}`, shape
            ``(n_quotes,)``.
        weights: Optional per-quote weights, shape ``(n_quotes,)``. Defaults to
            uniform. Vega weights are the usual desk choice.
        penalty: Arbitrage penalty. Defaults to a hinge penalty.
        iterations: Optimiser steps.
        learning_rate: Adam learning rate.
        log_every: Record history every this many steps.

    Returns:
        A :class:`SurfaceCalibrationResult`.

    Raises:
        ValueError: On inconsistent input shapes or non-positive maturities.
    """
    if not (log_moneyness.shape == maturity.shape == implied_volatility.shape):
        raise ValueError("log_moneyness, maturity and implied_volatility must match")
    if bool((maturity <= 0).any()):
        raise ValueError("maturities must be strictly positive")
    if weights is None:
        weights = torch.ones_like(implied_volatility)
    if penalty is None:
        penalty = ArbitragePenalty()

    target_variance = implied_volatility**2 * maturity
    sample_k = torch.linspace(
        float(log_moneyness.min()) - 0.5,
        float(log_moneyness.max()) + 0.5,
        41,
        dtype=maturity.dtype,
        device=maturity.device,
    )
    sample_t = torch.linspace(
        float(maturity.min()), float(maturity.max()), 13,
        dtype=maturity.dtype, device=maturity.device,
    )

    optimiser = torch.optim.Adam(surface.parameters(), lr=learning_rate)
    history: list = []

    for step in range(iterations):
        optimiser.zero_grad(set_to_none=True)
        model_variance = surface(log_moneyness, maturity)
        fit_loss = (weights * (model_variance - target_variance) ** 2).mean()
        terms = penalty(surface, sample_k, sample_t)
        (fit_loss + terms.total).backward()
        optimiser.step()

        if step % log_every == 0 or step == iterations - 1:
            with torch.no_grad():
                model_vol = torch.sqrt(
                    torch.clamp(surface(log_moneyness, maturity) / maturity, min=0.0)
                )
                rmse = float(
                    torch.sqrt(((model_vol - implied_volatility) ** 2).mean())
                )
            history.append(
                {
                    "step": step,
                    "fit_loss": float(fit_loss.detach()),
                    "penalty_calendar": float(terms.calendar.detach()),
                    "penalty_butterfly": float(terms.butterfly.detach()),
                    "penalty_ssvi_linear": float(terms.ssvi_linear.detach()),
                    "penalty_ssvi_quadratic": float(terms.ssvi_quadratic.detach()),
                    "implied_vol_rmse": rmse,
                }
            )

    with torch.no_grad():
        model_vol = torch.sqrt(
            torch.clamp(surface(log_moneyness, maturity) / maturity, min=0.0)
        )
        final_rmse = float(torch.sqrt(((model_vol - implied_volatility) ** 2).mean()))

    return SurfaceCalibrationResult(
        surface=surface,
        history=history,
        final_rmse=final_rmse,
        iterations=iterations,
    )


# ==========================================================================
# Bridge: calibrated SSVI -> Phase 6 local-vol kernel parameters
# ==========================================================================
#: Default half-width of the sampling region, in standard deviations of
#: :math:`\log S_T`. Three sigma covers ~99.7% of the terminal distribution;
#: widening it trades accuracy where the diffusion actually goes for accuracy
#: in a tail the Monte Carlo barely visits.
DEFAULT_FIT_SIGMA_WIDTH = 3.0


def chebyshev_basis(u: Tensor, degree: int) -> Tensor:
    r"""Stack :math:`T_0(u), \dots, T_{\text{degree}}(u)` along a new leading axis.

    Uses the three-term recurrence
    :math:`T_k(u) = 2u\,T_{k-1}(u) - T_{k-2}(u)`, :math:`T_0 = 1`,
    :math:`T_1 = u` -- pure ``torch`` arithmetic, so it is differentiable
    end-to-end and needs no special-function library.

    Args:
        u: Standardised coordinate, any shape, nominally in :math:`[-1, 1]`
            (the recurrence is well-defined outside that range too, but the
            polynomials grow quickly there, so extrapolation should be used
            deliberately, not by accident).
        degree: Highest polynomial order, :math:`\ge 0`.

    Returns:
        Shape ``(degree + 1, *u.shape)``.

    Raises:
        ValueError: If ``degree`` is negative.
    """
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    terms = [torch.ones_like(u)]
    if degree >= 1:
        terms.append(u)
    previous, current = terms[0], u if degree >= 1 else None
    for _ in range(2, degree + 1):
        nxt = 2.0 * u * current - previous
        terms.append(nxt)
        previous, current = current, nxt
    return torch.stack(terms, dim=0)


def chebyshev_basis_derivative(u: Tensor, degree: int) -> Tensor:
    r"""Stack :math:`T_0{}'(u), \dots, T_{\text{degree}}{}'(u)`.

    Uses :math:`T_k{}'(u) = k\,U_{k-1}(u)` for :math:`k \ge 1` and
    :math:`T_0{}' = 0`, where :math:`U` is the Chebyshev polynomial of the
    second kind, itself built from :math:`U_0 = 1`, :math:`U_1 = 2u`,
    :math:`U_k = 2u\,U_{k-1} - U_{k-2}`.

    This closed form exists specifically for the hand-written Triton adjoint,
    which cannot call ``torch.autograd``. It is checked against
    ``torch.autograd`` of :func:`chebyshev_basis` directly in
    ``tests/test_chebyshev_local_vol.py`` -- the CPU forward path uses
    autograd and never calls this function, so that comparison is a genuine
    independent check, not a tautology.

    Args:
        u: Standardised coordinate, any shape.
        degree: Highest polynomial order, :math:`\ge 0`.

    Returns:
        Shape ``(degree + 1, *u.shape)``.

    Raises:
        ValueError: If ``degree`` is negative.
    """
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    derivatives = [torch.zeros_like(u)]
    if degree >= 1:
        u_prev = torch.ones_like(u)  # U_0
        derivatives.append(1.0 * u_prev)  # T_1' = 1 * U_0
    if degree >= 2:
        u_curr = 2.0 * u  # U_1
        derivatives.append(2.0 * u_curr)  # T_2' = 2 * U_1
        for k in range(3, degree + 1):
            u_next = 2.0 * u * u_curr - u_prev
            derivatives.append(float(k) * u_next)
            u_prev, u_curr = u_curr, u_next
    return torch.stack(derivatives, dim=0)


def evaluate_chebyshev_local_vol(
    coefficients: Tensor,
    half_width: float,
    reference: float,
    term: float,
    time: Tensor,
    log_spot: Tensor,
    *,
    floor: float = 1e-4,
) -> Tensor:
    r"""Evaluate the fitted Chebyshev surface, matching the kernel's arithmetic.

    .. math::
        \sigma(t, x) = \max\!\Bigl(\sum_{k=0}^{K} c_k\, T_k(u) ,\ \text{floor}\Bigr)
                       + \sigma_{\text{term}}\, t,
        \qquad u = \frac{x - x_{\text{ref}}}{w}

    The floor applies to the spatial (Chebyshev) term only, before the linear
    time correction is added -- so it guards the part of the surface that can
    actually go non-positive (an unconstrained polynomial has no sign
    guarantee), the same honest-regularisation pattern as
    :attr:`LocalVolatilitySurface.variance_floor`.

    Args:
        coefficients: :math:`c_0, \dots, c_K`, shape ``(K + 1,)``.
        half_width: :math:`w`, the domain half-width used to standardise
            ``log_spot``.
        reference: :math:`x_{\text{ref}}`.
        term: Linear time slope.
        time: :math:`t`, any shape.
        log_spot: :math:`x = \log S`, broadcastable with ``time``.
        floor: Lower clamp on the spatial term.

    Returns:
        :math:`\sigma(t, x)`, broadcast shape.
    """
    degree = coefficients.numel() - 1
    u = (log_spot - reference) / half_width
    basis = chebyshev_basis(u, degree)
    spatial = torch.tensordot(coefficients, basis, dims=([0], [0]))
    return torch.clamp(spatial, min=floor) + term * time


@dataclass(frozen=True)
class LocalVolFit:
    r"""A parametric local-vol surface fitted to a Dupire :math:`\sigma_{LV}`.

    The Phase 6 Triton kernel evaluates

    .. math::
        \sigma(t, x) = \sigma_0
                     + \sigma_{\text{skew}}\tanh(\kappa(x - x_{\text{ref}}))
                     + \sigma_{\text{term}}\, t,
        \qquad x = \log S,

    which is a *four-parameter* form. A Dupire surface implied by SSVI has no
    reason to lie in that family, so this is a projection, not an identity. The
    diagnostics are therefore part of the result rather than an afterthought:
    quoting kernel output without them would hide how much of the calibrated
    surface was discarded on the way in.

    Attributes:
        base: :math:`\sigma_0`, the level.
        skew: :math:`\sigma_{\text{skew}}`, the tanh amplitude.
        kappa: :math:`\kappa`, the tanh steepness in log-spot.
        term: :math:`\sigma_{\text{term}}`, the linear term-structure slope.
        reference: :math:`x_{\text{ref}}`, pinned to :math:`\log S_0`. Not
            fitted: the kernel's ``tanh`` saturates once
            :math:`|\kappa(x - x_{\text{ref}})| \gtrsim 3`, so an uncentred
            reference drives :math:`\partial\sigma/\partial x` to ~1e-5 and
            silently reduces the kernel to the constant-volatility case.
        rmse: Root-mean-square error in volatility units, density-weighted if
            the fit was.
        max_abs_error: Worst pointwise error over the sampled grid.
        relative_rmse: ``rmse`` divided by the mean target volatility.
        r_squared: Fraction of the target's weighted variance explained. Near
            zero means the tanh form captured the *level* only.
        n_samples: Grid points used.
        target_range: ``(min, max)`` of the sampled :math:`\sigma_{LV}`.
        fitted_range: ``(min, max)`` of the fitted surface on the same grid.
        variance_floor_fraction: Share of grid points where the Dupire
            variance floor was binding. A large value means the *target* was
            already regularisation, so the fit error understates true model
            error.
        density_weighted: Whether the lognormal transition density was used.
        basis: ``"tanh"`` (default) or ``"chebyshev"``. Determines which of
            ``base``/``skew``/``kappa`` versus ``chebyshev_coefficients``/
            ``chebyshev_half_width`` are meaningful; ``term`` and
            ``reference`` are used by both bases.
        chebyshev_coefficients: :math:`c_0, \dots, c_K` when
            ``basis == "chebyshev"``, else ``None``.
        chebyshev_half_width: The domain half-width :math:`w` used to
            standardise log-spot into the Chebyshev argument, when
            ``basis == "chebyshev"``, else ``None``. Fixed, not fitted -- see
            :func:`fit_local_vol_params`.
    """

    base: float
    skew: float
    kappa: float
    term: float
    reference: float
    rmse: float
    max_abs_error: float
    relative_rmse: float
    r_squared: float
    n_samples: int
    target_range: Tuple[float, float]
    fitted_range: Tuple[float, float]
    variance_floor_fraction: float
    density_weighted: bool
    basis: str = "tanh"
    chebyshev_coefficients: Optional[Tuple[float, ...]] = None
    chebyshev_half_width: Optional[float] = None

    def to_local_vol_params(self) -> "object":
        """Convert to the matching kernel-side parameter object.

        Imported lazily so ``src.models`` carries no hard dependency on
        ``src.csrc`` -- the models layer must stay importable on a machine with
        no Triton, which is where most of this project was developed.

        Returns:
            A :class:`src.csrc.triton_local_vol_cva.LocalVolParams` when
            ``basis == "tanh"``, or a
            :class:`src.csrc.triton_chebyshev_local_vol_cva.ChebyshevLocalVolParams`
            when ``basis == "chebyshev"``.

        Raises:
            ValueError: On an unrecognised ``basis``.
        """
        if self.basis == "chebyshev":
            from src.csrc.triton_chebyshev_local_vol_cva import (
                ChebyshevLocalVolParams,
            )

            return ChebyshevLocalVolParams(
                coefficients=self.chebyshev_coefficients,
                half_width=self.chebyshev_half_width,
                reference=self.reference,
                term=self.term,
            )
        if self.basis == "tanh":
            from src.csrc.triton_local_vol_cva import LocalVolParams

            return LocalVolParams(
                base=self.base,
                skew=self.skew,
                kappa=self.kappa,
                term=self.term,
                reference=self.reference,
            )
        raise ValueError(
            f"unknown basis {self.basis!r}; expected 'tanh' or 'chebyshev'"
        )

    @property
    def is_well_fitted(self) -> bool:
        """Whether the projection is tight enough to quote without caveat.

        The 2% relative-RMSE threshold is a judgement, not a theorem: it is
        roughly the bid-ask width on a liquid equity vol quote, so a fit inside
        it is not the dominant error in the pipeline. Read ``relative_rmse``
        directly rather than relying on this flag for anything quantitative.
        """
        return self.relative_rmse < 0.02

    def summary(self) -> str:
        """A human-readable report of the fit and its quality."""
        verdict = "OK" if self.is_well_fitted else "POOR -- see note below"
        if self.basis == "chebyshev":
            degree = len(self.chebyshev_coefficients) - 1
            terms = " + ".join(
                f"{c:+.5f}*T_{k}(u)" for k, c in enumerate(self.chebyshev_coefficients)
            )
            formula = (
                f"sigma(t,x) = max({terms}, floor) "
                f"{'+' if self.term >= 0 else '-'} {abs(self.term):.6f}*t   "
                f"[u = (x - {self.reference:.6f}) / {self.chebyshev_half_width:.6f}, "
                f"degree {degree}]"
            )
        else:
            formula = (
                f"sigma(t,x) = {self.base:.6f} "
                f"{'+' if self.skew >= 0 else '-'} {abs(self.skew):.6f}"
                f"*tanh({self.kappa:.6f}*(x - {self.reference:.6f})) "
                f"{'+' if self.term >= 0 else '-'} {abs(self.term):.6f}*t"
            )
        lines = [
            f"SSVI -> Phase 6 local-vol parameter fit ({self.basis})",
            f"  {formula}",
            "",
            f"  samples            : {self.n_samples:,}"
            f"{' (density-weighted)' if self.density_weighted else ''}",
            f"  RMSE               : {self.rmse:.6f} vol",
            f"  relative RMSE      : {self.relative_rmse:.3%}   [{verdict}]",
            f"  max abs error      : {self.max_abs_error:.6f} vol",
            f"  R^2                : {self.r_squared:.6f}",
            f"  target sigma range : "
            f"{self.target_range[0]:.4f} to {self.target_range[1]:.4f}",
            f"  fitted sigma range : "
            f"{self.fitted_range[0]:.4f} to {self.fitted_range[1]:.4f}",
            f"  variance floor hit : {self.variance_floor_fraction:.2%} "
            "of grid points",
        ]
        if not self.is_well_fitted:
            lines += [
                "",
                "  The four-parameter tanh form cannot represent this Dupire",
                "  surface to quoting accuracy. Kernel gradients will be with",
                "  respect to the FITTED surface, not the calibrated SSVI one.",
                "  Options: narrow the maturity window (the kernel's term",
                "  structure is linear, so a long window strains it), or extend",
                "  the kernel to a Chebyshev expansion in (t, log S).",
            ]
        if self.variance_floor_fraction > 0.05:
            lines += [
                "",
                f"  WARNING: the Dupire variance floor bound at "
                f"{self.variance_floor_fraction:.1%} of grid points, so the",
                "  target is partly regularisation rather than surface.",
            ]
        return "\n".join(lines)


def local_vol_sampling_grid(
    spot_zero: float,
    maturity: float,
    *,
    n_time: int = 24,
    n_space: int = 41,
    reference_vol: float = 0.2,
    sigma_width: float = DEFAULT_FIT_SIGMA_WIDTH,
    dtype: torch.dtype = torch.float64,
) -> Tuple[Tensor, Tensor]:
    r"""A :math:`(t, \log S)` grid covering where the diffusion actually goes.

    The half-width grows as :math:`\sigma\sqrt{t}`, so the grid is a *cone*,
    not a box. A box would spend most of its points on states unreachable at
    short maturities, and the fit would then trade accuracy near the money --
    where nearly all exposure sits -- for accuracy in a corner the simulation
    never visits.

    Time starts at ``maturity / n_time`` rather than zero: the Dupire surface
    is undefined at :math:`T = 0` and the transition density is degenerate
    there.

    Args:
        spot_zero: :math:`S_0`.
        maturity: Longest maturity to cover, in years.
        n_time: Time nodes.
        n_space: Log-spot nodes per time slice.
        reference_vol: Volatility scale setting the cone width.
        sigma_width: Half-width in standard deviations.
        dtype: Output dtype.

    Returns:
        ``(time, log_spot)``, both shape ``(n_time, n_space)``.

    Raises:
        ValueError: On non-positive or degenerate inputs.
    """
    if spot_zero <= 0.0:
        raise ValueError(f"spot_zero must be positive, got {spot_zero}")
    if maturity <= 0.0:
        raise ValueError(f"maturity must be positive, got {maturity}")
    if n_time < 1 or n_space < 2:
        raise ValueError(
            f"need n_time >= 1 and n_space >= 2, got {n_time}, {n_space}"
        )
    if reference_vol <= 0.0 or sigma_width <= 0.0:
        raise ValueError("reference_vol and sigma_width must be positive")

    time = torch.linspace(
        maturity / n_time, maturity, n_time, dtype=dtype
    ).reshape(-1, 1)
    unit = torch.linspace(-1.0, 1.0, n_space, dtype=dtype).reshape(1, -1)
    half_width = sigma_width * reference_vol * torch.sqrt(time)
    log_spot = math.log(spot_zero) + unit * half_width
    return time.expand(n_time, n_space), log_spot


def _sample_dupire_target(
    local_vol: LocalVolatilitySurface,
    spot_zero: float,
    maturity: float,
    *,
    drift: float,
    n_time: int,
    n_space: int,
    reference_vol: float,
    sigma_width: float,
    density_weighted: bool,
):
    r"""Sample the Dupire target and its calibration weights.

    Shared by every basis: the grid, the target evaluation, and the density
    weighting are basis-independent, so this exists once rather than being
    copied per fitting routine (a second copy is exactly how the two bases
    would silently drift apart on the sampling convention).

    Returns:
        ``(time, log_spot, reference, target, weights, floor_fraction)``.
    """
    time, log_spot = local_vol_sampling_grid(
        spot_zero,
        maturity,
        n_time=n_time,
        n_space=n_space,
        reference_vol=reference_vol,
        sigma_width=sigma_width,
        dtype=torch.float64,
    )
    reference = math.log(spot_zero)

    # The target is a fixed surface: this projects onto the kernel's family, it
    # does not co-calibrate SSVI. So the result is detached -- but it must NOT
    # be computed under no_grad. Dupire local variance is *defined* by
    # autodiff of the total-variance surface (d_T w / g), so suppressing grad
    # removes the derivative the target is made of and raises "element 0 of
    # tensors does not require grad". enable_grad is explicit rather than
    # implicit so that calling this from inside a no_grad block still works.
    with torch.enable_grad():
        spot = torch.exp(log_spot.detach().requires_grad_(False))
        target = (
            local_vol.local_volatility(spot, time, spot_zero).double().detach()
        )
    floor_fraction = float(
        (target <= math.sqrt(local_vol.variance_floor) + 1e-12)
        .to(target.dtype)
        .mean()
    )

    if density_weighted:
        # log S_t ~ N(log S0 + (mu - sigma^2/2) t, sigma^2 t)
        centre = reference + (drift - 0.5 * reference_vol**2) * time
        variance = (reference_vol**2) * time
        weights = torch.exp(
            -0.5 * (log_spot - centre) ** 2 / variance
        ) / torch.sqrt(variance)
    else:
        weights = torch.ones_like(target)
    weights = weights / weights.sum()

    return time, log_spot, reference, target, weights, floor_fraction


def _fit_diagnostics(
    weights: Tensor,
    target: Tensor,
    fitted: Tensor,
) -> Tuple[float, float, float, float]:
    """Shared error metrics: ``(rmse, relative_rmse, r_squared, max_abs)``."""
    residual = fitted - target
    rmse = float(torch.sqrt((weights * residual**2).sum()))
    mean_target = float((weights * target).sum())
    total_variance = float((weights * (target - mean_target) ** 2).sum())
    r_squared = 1.0 - (rmse**2) / total_variance if total_variance > 0.0 else 1.0
    relative_rmse = rmse / mean_target if mean_target > 0.0 else float("inf")
    return rmse, relative_rmse, r_squared, float(residual.abs().max())


def _fit_tanh_local_vol(
    local_vol: LocalVolatilitySurface,
    spot_zero: float,
    maturity: float,
    *,
    drift: float,
    n_time: int,
    n_space: int,
    reference_vol: float,
    sigma_width: float,
    density_weighted: bool,
    iterations: int,
    learning_rate: float,
) -> LocalVolFit:
    r"""The original four-parameter tanh fit -- copied verbatim as a helper.

    :math:`\sigma_0` and :math:`\kappa` are fitted through ``softplus`` so
    they stay strictly positive, which the kernel's ``LocalVolParams``
    validates. An unconstrained optimiser happily reaches a negative
    :math:`\kappa` -- an observationally equivalent surface with flipped skew
    -- which then fails validation downstream for no obvious reason.

    :math:`x_{\text{ref}}` is **not** fitted; it is pinned to
    :math:`\log S_0`. See :attr:`LocalVolFit.reference`.
    """
    time, log_spot, reference, target, weights, floor_fraction = (
        _sample_dupire_target(
            local_vol, spot_zero, maturity, drift=drift, n_time=n_time,
            n_space=n_space, reference_vol=reference_vol,
            sigma_width=sigma_width, density_weighted=density_weighted,
        )
    )

    # softplus-parameterised, so base > 0 and kappa > 0 by construction.
    raw_base = torch.tensor(
        math.log(math.expm1(max(float(target.mean()), 1e-3))),
        dtype=torch.float64,
        requires_grad=True,
    )
    raw_kappa = torch.tensor(
        math.log(math.expm1(2.5)), dtype=torch.float64, requires_grad=True
    )
    skew = torch.zeros((), dtype=torch.float64, requires_grad=True)
    term = torch.zeros((), dtype=torch.float64, requires_grad=True)

    optimiser = torch.optim.Adam(
        [raw_base, raw_kappa, skew, term], lr=learning_rate
    )
    centred = log_spot - reference

    def model() -> Tensor:
        base = torch.nn.functional.softplus(raw_base)
        kappa = torch.nn.functional.softplus(raw_kappa)
        return base + skew * torch.tanh(kappa * centred) + term * time

    # enable_grad again, and around the whole loop: an outer no_grad context
    # would leave `loss` without a grad_fn and backward() would fail. Callers
    # do wrap analysis code in no_grad, so this has to be robust to it.
    with torch.enable_grad():
        for _ in range(iterations):
            optimiser.zero_grad(set_to_none=True)
            loss = (weights * (model() - target) ** 2).sum()
            loss.backward()
            optimiser.step()

    with torch.no_grad():
        fitted = model()
        rmse, relative_rmse, r_squared, max_abs_error = _fit_diagnostics(
            weights, target, fitted
        )
        return LocalVolFit(
            basis="tanh",
            base=float(torch.nn.functional.softplus(raw_base)),
            skew=float(skew),
            kappa=float(torch.nn.functional.softplus(raw_kappa)),
            term=float(term),
            reference=reference,
            rmse=rmse,
            max_abs_error=max_abs_error,
            relative_rmse=relative_rmse,
            r_squared=r_squared,
            n_samples=int(target.numel()),
            target_range=(float(target.min()), float(target.max())),
            fitted_range=(float(fitted.min()), float(fitted.max())),
            variance_floor_fraction=floor_fraction,
            density_weighted=density_weighted,
        )


def _fit_chebyshev_local_vol(
    local_vol: LocalVolatilitySurface,
    spot_zero: float,
    maturity: float,
    *,
    drift: float,
    n_time: int,
    n_space: int,
    reference_vol: float,
    sigma_width: float,
    density_weighted: bool,
    iterations: int,
    learning_rate: float,
    degree: int,
    floor: float,
) -> LocalVolFit:
    r"""Fit a degree-``degree`` Chebyshev expansion in place of the tanh form.

    Fixes the wing-saturation failure mode of the tanh fit: tanh is bounded
    by construction, so past a few multiples of :math:`1/\kappa` it is flat
    and cannot track a Dupire surface that keeps rising into the wing. A
    Chebyshev polynomial has no such ceiling -- degree :math:`K` can represent
    any curvature the sampled points show, at the cost of :math:`K + 1`
    coefficients instead of two.

    **Solved in closed form, not by gradient descent.**
    :math:`\sigma = \sum_k c_k T_k(u) + \sigma_{\text{term}} t` is *linear*
    in :math:`(c_0, \dots, c_K, \sigma_{\text{term}})`, so the weighted
    least-squares problem has a unique global optimum reachable by one solve --
    no learning rate, no iteration count, no convergence question. An earlier
    version of this function used Adam, matching the tanh fit's pattern for
    consistency; verification caught it under-converging on a real SSVI target
    (:math:`R^2` plateaued at 0.916 at degree 12 with 1500 iterations, and the
    fitted range overshot the target range at both ends -- the signature of an
    under-converged fit, not a representational limit). Recognising the model
    is linear and solving it exactly removes that failure mode entirely rather
    than papering over it with more iterations. ``iterations`` and
    ``learning_rate`` are accepted for signature symmetry with the tanh path
    and ignored here.

    The domain half-width is **fixed** to ``sigma_width * reference_vol``, the
    same quantity that defines the sampling cone, rather than fitted. Letting
    it float would let the fit narrow the domain to match noise near the money
    instead of the region the kernel actually simulates over -- the same
    reasoning that keeps ``reference`` pinned in the tanh fit.

    The floor is applied to the raw polynomial sum only, after the unconstrained
    solve, before the linear time term is added -- see
    :func:`evaluate_chebyshev_local_vol`. It is therefore an approximation
    exactly to the extent the floor actually binds; :attr:`LocalVolFit`'s
    diagnostics report how often the *target's* own floor bound, and a fitted
    range far from the target range is the signal to check the analogous
    question for the fit itself.
    """
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    if floor <= 0.0:
        raise ValueError(f"floor must be positive, got {floor}")
    del iterations, learning_rate  # unused: closed-form, see docstring

    time, log_spot, reference, target, weights, floor_fraction = (
        _sample_dupire_target(
            local_vol, spot_zero, maturity, drift=drift, n_time=n_time,
            n_space=n_space, reference_vol=reference_vol,
            sigma_width=sigma_width, density_weighted=density_weighted,
        )
    )

    half_width = sigma_width * reference_vol
    u = (log_spot - reference) / half_width
    basis = chebyshev_basis(u, degree)  # (K+1, n_time, n_space)

    n_samples = target.numel()
    flat_basis = basis.reshape(degree + 1, n_samples).T  # (n_samples, K+1)
    flat_time = time.reshape(n_samples, 1)
    design = torch.cat([flat_basis, flat_time], dim=1)  # (n_samples, K+2)
    flat_target = target.reshape(n_samples, 1)

    # Fold the weights in by scaling rows with sqrt(w): least squares on the
    # scaled system minimises sum w_i * residual_i^2, the same objective the
    # tanh fit's Adam loop minimises.
    sqrt_weights = torch.sqrt(weights.reshape(n_samples, 1))
    solution = torch.linalg.lstsq(
        design * sqrt_weights, flat_target * sqrt_weights
    ).solution.reshape(-1)
    coefficients, term = solution[:-1], solution[-1]

    fitted = (
        torch.clamp(torch.tensordot(coefficients, basis, dims=([0], [0])), min=floor)
        + term * time
    )
    rmse, relative_rmse, r_squared, max_abs_error = _fit_diagnostics(
        weights, target, fitted
    )
    return LocalVolFit(
        basis="chebyshev",
        base=None,
        skew=None,
        kappa=None,
        term=float(term),
        reference=reference,
        chebyshev_coefficients=tuple(float(c) for c in coefficients),
        chebyshev_half_width=half_width,
        rmse=rmse,
        max_abs_error=max_abs_error,
        relative_rmse=relative_rmse,
        r_squared=r_squared,
        n_samples=n_samples,
        target_range=(float(target.min()), float(target.max())),
        fitted_range=(float(fitted.min()), float(fitted.max())),
        variance_floor_fraction=floor_fraction,
        density_weighted=density_weighted,
    )


def fit_local_vol_params(
    local_vol: LocalVolatilitySurface,
    spot_zero: float,
    maturity: float,
    *,
    drift: float = 0.0,
    n_time: int = 24,
    n_space: int = 41,
    reference_vol: float = 0.2,
    sigma_width: float = DEFAULT_FIT_SIGMA_WIDTH,
    density_weighted: bool = True,
    iterations: int = 600,
    learning_rate: float = 5e-2,
    basis: str = "tanh",
    degree: int = 8,
    floor: float = 1e-4,
) -> LocalVolFit:
    r"""Project a Dupire :math:`\sigma_{LV}` onto a Phase 6 kernel's form.

    This is the missing link between the calibrated SSVI surface and a Phase 6
    Triton kernel. Neither kernel reads an ``SSVISurface`` directly: each
    evaluates a closed-form surface in registers, because a gather from a
    Dupire grid would serialise every warp. SSVI therefore has to be
    *projected* onto one of these families, and the projection error is a real
    modelling cost that :class:`LocalVolFit` reports rather than hides.

    Two bases, selected by ``basis``:

    ``"tanh"`` (default, unchanged from the original bridge)
        :math:`\sigma_0 + \sigma_{\text{skew}}\tanh(\kappa(x - x_{\text{ref}}))
        + \sigma_{\text{term}} t`. Two free spatial parameters. Bounded by
        construction, so it **saturates** on a steep wing: fitted against an
        SSVI surface with :math:`\rho=-0.35, \eta=1.2, \gamma=0.45` at a
        :math:`3\sigma` sampling width, relative RMSE is 15.99% and
        :math:`R^2` only 0.578 -- the tanh simply cannot rise as fast as the
        target does in the wing.
    ``"chebyshev"``
        :math:`\max\bigl(\sum_{k=0}^{K} c_k T_k(u), \text{floor}\bigr)
        + \sigma_{\text{term}} t`, :math:`u = (x - x_{\text{ref}})/w`.
        :math:`K + 1` free spatial coefficients instead of two, with no
        saturation ceiling -- more terms track more curvature. Recommended
        whenever ``sigma_width`` is large enough for the tanh fit's own
        diagnostics to report ``relative_rmse`` above a percent or two.

    In both cases the fit is a weighted least squares in volatility units,
    with :math:`\omega_i` the lognormal transition density when
    ``density_weighted`` -- unweighted fitting treats a state three sigma out
    at :math:`t=0.1` as equally important as the money, and the result is
    visibly wrong exactly where the exposure is.

    :math:`x_{\text{ref}}` is **not** fitted in either basis; it is pinned to
    :math:`\log S_0` (see :attr:`LocalVolFit.reference`), and the Chebyshev
    domain half-width is likewise pinned to ``sigma_width * reference_vol``
    rather than fitted, for the same reason: letting either float lets the
    optimiser narrow the domain or the centring to fit noise instead of
    matching the region actually simulated.

    Args:
        local_vol: The Dupire surface built on a calibrated SSVI surface.
        spot_zero: :math:`S_0`.
        maturity: Longest maturity the kernel will simulate to.
        drift: :math:`\mu`, used for the density weights only.
        n_time: Time nodes in the sampling grid.
        n_space: Log-spot nodes per slice.
        reference_vol: Volatility scale for the sampling cone and the weights.
        sigma_width: Grid half-width in standard deviations.
        density_weighted: Weight by the lognormal transition density.
        iterations: Adam steps. Used only for ``basis="tanh"``; the
            Chebyshev fit is linear in its coefficients and is solved in
            closed form by weighted least squares, so this is ignored there.
        learning_rate: Adam learning rate. Also ignored for ``"chebyshev"``,
            for the same reason.
        basis: ``"tanh"`` or ``"chebyshev"``.
        degree: Chebyshev polynomial degree :math:`K`. Ignored for ``"tanh"``.
        floor: Positive lower clamp on the Chebyshev spatial term. Ignored for
            ``"tanh"`` (which is positivity-constrained via ``softplus``
            instead).

    Returns:
        A :class:`LocalVolFit` holding the parameters and the fit diagnostics.

    Raises:
        ValueError: On invalid grid or optimiser settings, an unknown
            ``basis``, a negative ``degree``, or a non-positive ``floor``.
    """
    if iterations < 1:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")

    if basis == "tanh":
        return _fit_tanh_local_vol(
            local_vol, spot_zero, maturity, drift=drift, n_time=n_time,
            n_space=n_space, reference_vol=reference_vol,
            sigma_width=sigma_width, density_weighted=density_weighted,
            iterations=iterations, learning_rate=learning_rate,
        )
    if basis == "chebyshev":
        return _fit_chebyshev_local_vol(
            local_vol, spot_zero, maturity, drift=drift, n_time=n_time,
            n_space=n_space, reference_vol=reference_vol,
            sigma_width=sigma_width, density_weighted=density_weighted,
            iterations=iterations, learning_rate=learning_rate,
            degree=degree, floor=floor,
        )
    raise ValueError(f"unknown basis {basis!r}; expected 'tanh' or 'chebyshev'")


def evaluate_parametric_local_vol(
    fit: LocalVolFit, time: Tensor, log_spot: Tensor
) -> Tensor:
    r"""Evaluate the fitted surface, matching the kernel's arithmetic exactly.

    Useful for plotting the projection against its Dupire target, and for
    checking on the host what the kernel computes on the device.

    Args:
        fit: A :class:`LocalVolFit`.
        time: :math:`t`, any shape.
        log_spot: :math:`x = \log S`, broadcastable with ``time``.

    Returns:
        :math:`\sigma(t, x)`, broadcast shape.
    """
    return (
        fit.base
        + fit.skew * torch.tanh(fit.kappa * (log_spot - fit.reference))
        + fit.term * time
    )
