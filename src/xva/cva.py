r"""Unilateral CVA / DVA under a flat hazard rate, end-to-end differentiable.

Credit model
------------
Default is modelled as the first jump of a Cox process whose intensity
(*hazard rate*) :math:`\lambda` is deterministic and constant. The survival
probability to time :math:`t` is then

.. math:: Q(t) = \mathbb{P}(\tau > t) = \exp(-\lambda t),

and the probability of default inside the interval :math:`(t_{i-1}, t_i]` --
the *marginal* or *incremental* default probability -- is

.. math::
    d\!PD_i = \mathbb{P}(t_{i-1} < \tau \le t_i)
            = Q(t_{i-1}) - Q(t_i)
            = e^{-\lambda t_{i-1}} - e^{-\lambda t_i}.

A flat intensity is the deliberate Phase 2 simplification: it is the
single-parameter model implied by quoting one CDS spread :math:`s` at one
tenor, via the classical credit triangle :math:`\lambda \approx s / (1 - R)`.
Bootstrapping a full piecewise-constant intensity curve from a term structure
of CDS quotes is a later-phase concern; the code below already accepts a
time-varying survival curve, so that upgrade is a change of *input*, not of
*algorithm*.

The CVA integral
----------------
Unilateral CVA is the risk-neutral expectation of the discounted loss given
the counterparty's default:

.. math::
    CVA = (1-R)\int_0^T \mathbb{E}^{\mathbb{Q}}
          \!\left[D(0,t)\, V_t^+ \,\middle|\, \tau = t\right] dQ(t),

which under the discretisation and independence assumptions below collapses to
the discrete sum this module implements:

.. math:: CVA = (1-R)\sum_{i=1}^{N} EE(t_i)\; d\!PD_i\; DF(t_i).

DVA is the exact mirror image, using our *own* hazard rate and recovery and
the Expected Negative Exposure:

.. math:: DVA = (1-R_{\text{own}})\sum_{i=1}^{N} ENE(t_i)\; d\!PD^{\text{own}}_i\; DF(t_i).

Assumptions made explicit
-------------------------
1. **No wrong-way risk.** Exposure and default time are assumed independent,
   which is what allows the joint expectation to factor into
   :math:`EE(t_i) \times d\!PD_i`. In reality a counterparty's credit quality is
   often correlated with the exposure it generates (a classic example: an oil
   producer selling oil forwards). Relaxing this requires either a stochastic
   intensity correlated with the market factors, or an :math:`\alpha`
   multiplier -- both out of Phase 2 scope, but the factorisation is exactly
   the line in the code where such a correction would enter.
2. **Deterministic discounting.** :math:`DF(t) = e^{-rt}` is independent of the
   exposure, so the discount factor can be pulled outside the expectation.
3. **Default at grid dates.** Losses are attributed to the grid point closing
   each interval (``convention="endpoint"``, matching the formula above) rather
   than to the true default time inside it. This is a first-order-accurate
   left/right-endpoint rule; ``convention="average"`` applies the trapezoidal
   correction and is second-order accurate, at no extra simulation cost.
4. **Zero collateral, no close-out netting beyond the portfolio.** The MtM
   surface handed in is already the netted position of one netting set.

Sensitivities
-------------
The headline Phase 2 result is that the whole chain

    parameters :math:`\to` GBM paths :math:`\to` portfolio MtM surface
    :math:`\to` exposure profile :math:`\to` CVA

is one unbroken autograd graph, so a **single** reverse sweep returns
:math:`\partial CVA/\partial S_0`, :math:`\partial CVA/\partial \sigma` and the
credit delta :math:`\partial CVA/\partial \lambda` simultaneously. Bump-and-
revalue needs :math:`2n+1 = 7` full Monte-Carlo revaluations for the same three
numbers, and the gap widens linearly with the number of risk factors -- which
is the entire argument for AAD on a real XVA book carrying hundreds of them.

References
----------
Brigo, D., Morini, M., Pallavicini, A. (2013). *Counterparty Credit Risk,
Collateral and Funding*, Wiley -- Chapters 1-4.
Gregory, J. (2020). *Counterparty Credit Risk, Funding, Collateral and
Capital*, 3rd ed., Wiley -- Chapter 14.
Green, A. (2015). *XVA: Credit, Funding and Capital Valuation Adjustments*,
Wiley -- Chapter 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
from torch import Tensor

from src.models.gbm import GBMSimulator, simulate_gbm
from src.pricer.greeks import GreekResult, aad_greeks, finite_difference_greeks
from src.pricer.options import PriceFn, SwapLeg, portfolio_swap_mtm, resolve_rate_and_drift
from src.xva.exposure import (
    ExposureProfile,
    ScalarLike,
    as_tensor_like,
    compute_exposure_profile,
    expected_exposure,
)

__all__ = [
    "XVAResult",
    "survival_probability",
    "marginal_default_probability",
    "discount_factors",
    "compute_unilateral_cva",
    "compute_unilateral_dva",
    "compute_xva",
    "make_cva_valuation_fn",
    "cva_aad_greeks",
    "cva_bump_and_revalue_greeks",
]

_VALID_CONVENTIONS = ("endpoint", "average")


@dataclass(frozen=True)
class XVAResult:
    """CVA, DVA and the exposure profile they were computed from.

    Attributes:
        cva: Credit Valuation Adjustment, a 0-dim differentiable tensor.
            Reported as a **positive magnitude**; it is a cost, so the
            credit-adjusted portfolio value is ``risk_free_value - cva``.
        dva: Debit Valuation Adjustment, a 0-dim differentiable tensor, also a
            positive magnitude. It is a *benefit*, so the bilateral adjusted
            value is ``risk_free_value - cva + dva``.
        profile: The :class:`~src.xva.exposure.ExposureProfile` underlying both.
        hazard_rate: Counterparty hazard rate used for the CVA leg.
        recovery_rate: Counterparty recovery rate used for the CVA leg.
    """

    cva: Tensor
    dva: Tensor
    profile: ExposureProfile
    hazard_rate: float
    recovery_rate: float

    @property
    def bilateral_adjustment(self) -> Tensor:
        """Net bilateral adjustment ``dva - cva`` (positive means net benefit)."""
        return self.dva - self.cva


def _validate_convention(convention: str) -> None:
    if convention not in _VALID_CONVENTIONS:
        raise ValueError(
            f"convention must be one of {_VALID_CONVENTIONS}, got {convention!r}"
        )


def _validate_rate_like(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")


def survival_probability(times: Tensor, hazard_rate: ScalarLike) -> Tensor:
    r"""Flat-intensity survival curve :math:`Q(t) = e^{-\lambda t}`.

    Args:
        times: Observation grid of shape ``(n_steps + 1,)`` in years, assumed
            non-negative and increasing.
        hazard_rate: Constant intensity :math:`\lambda \ge 0`. Pass a tensor
            with ``requires_grad=True`` to obtain the credit delta
            :math:`\partial CVA/\partial\lambda`.

    Returns:
        Tensor of shape ``(n_steps + 1,)`` with values in :math:`(0, 1]`,
        differentiable w.r.t. ``hazard_rate``.

    Note:
        No positivity constraint is enforced on ``hazard_rate``: a negative
        intensity is economically meaningless but must remain *representable*,
        because a central finite-difference bump of a near-zero hazard rate
        legitimately evaluates the curve at a negative value. Rejecting it here
        would break the bump-and-revalue validation path.
    """
    if times.ndim != 1:
        raise ValueError(f"times must be 1-dimensional, got shape {tuple(times.shape)}")
    lam = as_tensor_like(hazard_rate, times)
    return torch.exp(-lam * times)


def marginal_default_probability(times: Tensor, hazard_rate: ScalarLike) -> Tensor:
    r"""Per-interval default probabilities :math:`d\!PD_i = Q(t_{i-1}) - Q(t_i)`.

    Args:
        times: Observation grid of shape ``(n_steps + 1,)``.
        hazard_rate: Constant intensity :math:`\lambda`.

    Returns:
        Tensor of shape ``(n_steps,)``, differentiable w.r.t. ``hazard_rate``.
        Entries are non-negative whenever :math:`\lambda \ge 0` and ``times``
        is increasing, and they sum to :math:`Q(t_0) - Q(t_N)`, i.e. the total
        probability of defaulting somewhere in the window.

    Raises:
        ValueError: If fewer than two grid points are supplied.
    """
    if times.shape[0] < 2:
        raise ValueError("need at least two grid points to form default intervals")
    survival = survival_probability(times, hazard_rate)
    return survival[:-1] - survival[1:]


def discount_factors(times: Tensor, rate: ScalarLike) -> Tensor:
    r"""Deterministic discount curve :math:`DF(t) = e^{-rt}`.

    Args:
        times: Observation grid of shape ``(n_steps + 1,)``.
        rate: Continuously compounded risk-free rate :math:`r`.

    Returns:
        Tensor of shape ``(n_steps + 1,)``, differentiable w.r.t. ``rate`` so
        that a CVA Rho falls out of the same backward pass.
    """
    if times.ndim != 1:
        raise ValueError(f"times must be 1-dimensional, got shape {tuple(times.shape)}")
    return torch.exp(-as_tensor_like(rate, times) * times)


def _integrate_credit_leg(
    profile_values: Tensor,
    times: Tensor,
    hazard_rate: ScalarLike,
    recovery_rate: float,
    *,
    discount_rate: Optional[ScalarLike] = None,
    curve: Optional[Tensor] = None,
    convention: str = "endpoint",
) -> Tensor:
    r"""Shared discrete integrator for the CVA and DVA legs.

    Evaluates :math:`(1-R)\sum_i P(t_i)\, d\!PD_i\, DF(t_i)` where :math:`P` is
    either the EE profile (CVA) or the ENE profile (DVA).

    Args:
        profile_values: EE or ENE of shape ``(n_steps + 1,)``.
        times: Observation grid of shape ``(n_steps + 1,)``.
        hazard_rate: Intensity of the defaulting party.
        recovery_rate: Recovery rate :math:`R` of the defaulting party.
        discount_rate: Flat rate used to build :math:`DF`. Mutually exclusive
            with ``curve``.
        curve: Explicit discount factors of shape ``(n_steps + 1,)``. Mutually
            exclusive with ``discount_rate``.
        convention: ``"endpoint"`` uses the interval's closing values;
            ``"average"`` applies the trapezoidal rule to both the exposure and
            the discount factor.

    Returns:
        0-dim differentiable tensor.

    Raises:
        ValueError: On shape mismatch, an invalid convention or recovery rate,
            or an ambiguous/absent discount specification.
    """
    _validate_convention(convention)
    _validate_rate_like(recovery_rate, "recovery_rate")

    if profile_values.ndim != 1:
        raise ValueError(
            f"profile must be 1-dimensional, got shape {tuple(profile_values.shape)}"
        )
    if profile_values.shape != times.shape:
        raise ValueError(
            f"shape mismatch: profile {tuple(profile_values.shape)} "
            f"vs times {tuple(times.shape)}"
        )
    if (discount_rate is None) == (curve is None):
        raise ValueError("provide exactly one of 'discount_rate' or 'curve'")

    grid = times.to(device=profile_values.device, dtype=profile_values.dtype)
    if curve is None:
        factors = discount_factors(grid, discount_rate)
    else:
        if curve.shape != times.shape:
            raise ValueError(
                f"discount curve shape {tuple(curve.shape)} does not match "
                f"times {tuple(times.shape)}"
            )
        factors = curve.to(device=profile_values.device, dtype=profile_values.dtype)

    default_probabilities = marginal_default_probability(grid, hazard_rate)

    if convention == "endpoint":
        exposure_term = profile_values[1:]
        discount_term = factors[1:]
    else:  # "average" -- trapezoidal within each interval
        exposure_term = 0.5 * (profile_values[:-1] + profile_values[1:])
        discount_term = 0.5 * (factors[:-1] + factors[1:])

    loss_given_default = 1.0 - recovery_rate
    return loss_given_default * torch.sum(exposure_term * default_probabilities * discount_term)


def compute_unilateral_cva(
    ee: Tensor,
    times: Tensor,
    hazard_rate: ScalarLike,
    recovery_rate: float = 0.4,
    *,
    discount_rate: Optional[ScalarLike] = None,
    curve: Optional[Tensor] = None,
    convention: str = "endpoint",
) -> Tensor:
    r"""Unilateral CVA by discrete summation of the credit-loss integral.

    .. math:: CVA = (1-R)\sum_{i=1}^{N} EE(t_i)\; d\!PD_i\; DF(t_i)

    Args:
        ee: Expected Exposure profile of shape ``(n_steps + 1,)``, from
            :func:`src.xva.exposure.expected_exposure`.
        times: Observation grid of shape ``(n_steps + 1,)`` in years.
        hazard_rate: Counterparty intensity :math:`\lambda`. Pass a tensor with
            ``requires_grad=True`` for the credit delta.
        recovery_rate: Counterparty recovery :math:`R`, default ``0.4``
            (the standard senior-unsecured assumption).
        discount_rate: Flat continuously compounded rate for :math:`DF`.
            Mutually exclusive with ``curve``.
        curve: Explicit discount factors of shape ``(n_steps + 1,)``. Mutually
            exclusive with ``discount_rate``.
        convention: ``"endpoint"`` (default, matches the formula above) or
            ``"average"`` for the trapezoidal refinement.

    Returns:
        0-dim differentiable tensor holding a **positive** CVA magnitude.

    Note:
        CVA is returned as a positive cost. The credit-adjusted value of the
        portfolio is ``risk_free_value - cva``.
    """
    return _integrate_credit_leg(
        ee,
        times,
        hazard_rate,
        recovery_rate,
        discount_rate=discount_rate,
        curve=curve,
        convention=convention,
    )


def compute_unilateral_dva(
    ene: Tensor,
    times: Tensor,
    hazard_rate: ScalarLike,
    recovery_rate: float = 0.4,
    *,
    discount_rate: Optional[ScalarLike] = None,
    curve: Optional[Tensor] = None,
    convention: str = "endpoint",
) -> Tensor:
    r"""Unilateral DVA -- the mirror of :func:`compute_unilateral_cva`.

    .. math:: DVA = (1-R_{\text{own}})\sum_{i=1}^{N} ENE(t_i)\;
              d\!PD^{\text{own}}_i\; DF(t_i)

    Args:
        ene: Expected Negative Exposure of shape ``(n_steps + 1,)``.
        times: Observation grid of shape ``(n_steps + 1,)``.
        hazard_rate: **Our own** intensity, not the counterparty's.
        recovery_rate: Our own recovery rate.
        discount_rate: Flat rate for :math:`DF`. Mutually exclusive with
            ``curve``.
        curve: Explicit discount factors. Mutually exclusive with
            ``discount_rate``.
        convention: ``"endpoint"`` or ``"average"``.

    Returns:
        0-dim differentiable tensor holding a **positive** DVA magnitude,
        which enters the bilateral value with a ``+`` sign.
    """
    return _integrate_credit_leg(
        ene,
        times,
        hazard_rate,
        recovery_rate,
        discount_rate=discount_rate,
        curve=curve,
        convention=convention,
    )


def compute_xva(
    mtm: Tensor,
    times: Tensor,
    *,
    hazard_rate: ScalarLike,
    discount_rate: ScalarLike,
    recovery_rate: float = 0.4,
    own_hazard_rate: Optional[ScalarLike] = None,
    own_recovery_rate: Optional[float] = None,
    confidence_level: float = 0.95,
    convention: str = "endpoint",
) -> XVAResult:
    r"""Compute the exposure profile, CVA and DVA from an MtM surface in one pass.

    Args:
        mtm: Netted mark-to-market surface of shape ``(n_paths, n_steps + 1)``.
        times: Observation grid of shape ``(n_steps + 1,)``.
        hazard_rate: Counterparty intensity :math:`\lambda`.
        discount_rate: Flat continuously compounded discount rate.
        recovery_rate: Counterparty recovery, default ``0.4``.
        own_hazard_rate: Our own intensity for the DVA leg. Defaults to
            ``hazard_rate`` when omitted (a symmetric-credit assumption; state
            it explicitly in any write-up that relies on it).
        own_recovery_rate: Our own recovery for the DVA leg. Defaults to
            ``recovery_rate``.
        confidence_level: Quantile level for the PFE in the returned profile.
        convention: ``"endpoint"`` or ``"average"``.

    Returns:
        An :class:`XVAResult` whose tensors all remain on the autograd tape of
        ``mtm``.
    """
    profile = compute_exposure_profile(mtm, times, confidence_level=confidence_level)
    effective_own_hazard = hazard_rate if own_hazard_rate is None else own_hazard_rate
    effective_own_recovery = (
        recovery_rate if own_recovery_rate is None else own_recovery_rate
    )

    cva = compute_unilateral_cva(
        profile.ee,
        profile.times,
        hazard_rate,
        recovery_rate,
        discount_rate=discount_rate,
        convention=convention,
    )
    dva = compute_unilateral_dva(
        profile.ene,
        profile.times,
        effective_own_hazard,
        effective_own_recovery,
        discount_rate=discount_rate,
        convention=convention,
    )
    return XVAResult(
        cva=cva,
        dva=dva,
        profile=profile,
        hazard_rate=float(as_tensor_like(hazard_rate, profile.ee).detach()),
        recovery_rate=recovery_rate,
    )


def make_cva_valuation_fn(
    simulator: GBMSimulator,
    dW: Tensor,
    legs: Sequence[SwapLeg],
    *,
    recovery_rate: float = 0.4,
    rate: Optional[ScalarLike] = None,
    convention: str = "endpoint",
) -> PriceFn:
    r"""Build the single-graph CVA valuation closure used for AAD and bumping.

    The returned callable chains, without ever leaving the autograd tape:

    1. :func:`~src.models.gbm.simulate_gbm` -- GBM paths from the *captured*
       Brownian sample ``dW``,
    2. :func:`~src.pricer.options.portfolio_swap_mtm` -- netted MtM surface,
    3. :func:`~src.xva.exposure.expected_exposure` -- the EE profile,
    4. :func:`compute_unilateral_cva` -- the discrete credit integral.

    Capturing ``dW`` is what makes bump-and-revalue a legitimate oracle for
    this quantity: every perturbed revaluation differentiates the *same* Monte-
    Carlo realisation (common random numbers), so the AAD-vs-FD gap measures
    the differentiation scheme rather than resampling noise.

    Args:
        simulator: Configured :class:`~src.models.gbm.GBMSimulator` supplying
            the time grid and step size.
        dW: Fixed Brownian increments of shape ``(n_paths, n_steps)``.
        legs: Portfolio legs defining the netting set.
        recovery_rate: Counterparty recovery :math:`R`.
        rate: Default discount/drift rate, used unless ``params["rate"]`` is
            supplied at call time.
        convention: ``"endpoint"`` or ``"average"``.

    Returns:
        A :data:`~src.pricer.options.PriceFn` expecting keys ``"s0"``,
        ``"sigma"`` and ``"hazard_rate"``, optionally ``"rate"`` and ``"mu"``,
        and returning a 0-dim CVA tensor.

    Raises:
        ValueError: On an invalid convention or recovery rate.

    Example:
        >>> cva_fn = make_cva_valuation_fn(sim, dW, legs, rate=0.03)
        >>> greeks = aad_greeks(cva_fn, {"s0": 100.0, "sigma": 0.2,
        ...                              "hazard_rate": 0.02})
        >>> sorted(greeks.greeks)
        ['hazard_rate', 's0', 'sigma']
    """
    _validate_convention(convention)
    _validate_rate_like(recovery_rate, "recovery_rate")

    legs = tuple(legs)
    times = simulator.time_grid()

    def cva_fn(params: Mapping[str, Tensor]) -> Tensor:
        resolved_rate, drift = resolve_rate_and_drift(params, rate)
        paths = simulate_gbm(params["s0"], drift, params["sigma"], dW, simulator.dt)
        mtm = portfolio_swap_mtm(paths, times, legs, resolved_rate)
        ee = expected_exposure(mtm)
        return compute_unilateral_cva(
            ee,
            times,
            params["hazard_rate"],
            recovery_rate,
            discount_rate=resolved_rate,
            convention=convention,
        )

    return cva_fn


def cva_aad_greeks(
    cva_fn: PriceFn,
    params: Mapping[str, Tensor | float],
    *,
    wrt: Optional[Sequence[str]] = None,
) -> GreekResult:
    r"""CVA sensitivities via a single reverse-mode sweep.

    A thin, intention-revealing wrapper over
    :func:`src.pricer.greeks.aad_greeks`. The Phase 1 engine is entirely
    payoff-agnostic -- it differentiates any scalar-valued closure -- so no
    XVA-specific differentiation logic is needed here, and reusing it means the
    CVA Greeks inherit the tape-integrity guard that raises when a requested
    parameter turns out to be disconnected from the graph.

    Args:
        cva_fn: Closure from :func:`make_cva_valuation_fn`.
        params: Parameter values, typically ``{"s0": ..., "sigma": ...,
            "hazard_rate": ...}``.
        wrt: Subset of names to differentiate. Defaults to all of ``params``.

    Returns:
        A :class:`~src.pricer.greeks.GreekResult` with ``n_valuations == 1``,
        holding :math:`\partial CVA/\partial S_0`,
        :math:`\partial CVA/\partial\sigma` and
        :math:`\partial CVA/\partial\lambda`.
    """
    return aad_greeks(cva_fn, params, wrt=wrt)


def cva_bump_and_revalue_greeks(
    cva_fn: PriceFn,
    params: Mapping[str, Tensor | float],
    *,
    wrt: Optional[Sequence[str]] = None,
    rel_step: float = 1e-4,
    scheme: str = "central",
) -> GreekResult:
    r"""CVA sensitivities by finite differences -- the validation oracle.

    Args:
        cva_fn: Closure from :func:`make_cva_valuation_fn`, which must capture
            a fixed ``dW`` so that common random numbers apply.
        params: Parameter values.
        wrt: Subset of names to bump. Defaults to all of ``params``.
        rel_step: Relative bump size.
        scheme: ``"central"`` (:math:`O(h^2)`) or ``"forward"``
            (:math:`O(h)`).

    Returns:
        A :class:`~src.pricer.greeks.GreekResult` with ``n_valuations``
        equal to ``2n + 1`` for the central scheme.

    Note:
        Finite differences carry a bias here that AAD does not:
        :math:`EE` is a mean of :math:`\max(V_t, 0)`, and a bump of size
        :math:`h` flips the sign of :math:`V_t` on :math:`O(Mh)` paths. Each
        such crossing contributes a kink, so the difference quotient converges
        to the true derivative only at :math:`O(h)` rather than the
        :math:`O(h^2)` a smooth integrand would give. The effect is small at
        the tolerances used in Phase 2, but it is a genuine accuracy argument
        for AAD, not merely a speed one.
    """
    return finite_difference_greeks(
        cva_fn, params, wrt=wrt, rel_step=rel_step, scheme=scheme
    )
