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
    "PiecewiseHazard",
    "survival_probability",
    "piecewise_survival_probability",
    "piecewise_marginal_default_probability",
    "resolve_survival_curve",
    "marginal_default_probability",
    "discount_factors",
    "compute_unilateral_cva",
    "compute_unilateral_dva",
    "compute_xva",
    "make_cva_valuation_fn",
    "cva_credit_bucket_deltas",
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
        hazard_rate: Flat counterparty hazard rate used for the CVA leg, or
            ``None`` when a term-structure curve was supplied -- there is no
            single intensity to report in that case, and returning a summary
            number would invite it being quoted as if it were the input.
        recovery_rate: Counterparty recovery rate used for the CVA leg.
    """

    cva: Tensor
    dva: Tensor
    profile: ExposureProfile
    hazard_rate: Optional[float]
    recovery_rate: float

    @property
    def bilateral_adjustment(self) -> Tensor:
        """Net bilateral adjustment ``dva - cva`` (positive means net benefit)."""
        return self.dva - self.cva


@dataclass(frozen=True)
class PiecewiseHazard:
    r"""A piecewise-constant hazard curve held as torch tensors.

    The hazard is :math:`h_j` on :math:`(T_{j-1}, T_j]` with :math:`T_0 = 0`,
    held flat at :math:`h_J` beyond the last pillar -- the same convention as
    :class:`market_data.fetcher.CreditCurve`, so the two agree pointwise.

    Kept in torch rather than numpy for one reason: **gradients**. Reading
    survival probabilities out of a numpy curve yields a constant and silently
    destroys every credit sensitivity. Here :math:`Q` is rebuilt from
    ``hazard_rates``, so if that tensor carries ``requires_grad`` a single
    backward pass gives :math:`\partial CVA/\partial h_j` for every pillar --
    the bucketed credit deltas a desk hedges with, rather than one lumped
    sensitivity to a flat intensity.

    Attributes:
        pillar_times: :math:`T_1 < \dots < T_J`, shape ``(n_pillars,)``.
        hazard_rates: :math:`h_1, \dots, h_J`, shape ``(n_pillars,)``. Pass
            with ``requires_grad=True`` for per-pillar credit deltas.
    """

    pillar_times: Tensor
    hazard_rates: Tensor

    def __post_init__(self) -> None:
        if self.pillar_times.ndim != 1 or self.hazard_rates.ndim != 1:
            raise ValueError(
                "pillar_times and hazard_rates must both be 1-dimensional, got "
                f"{tuple(self.pillar_times.shape)} and "
                f"{tuple(self.hazard_rates.shape)}"
            )
        if self.pillar_times.shape != self.hazard_rates.shape:
            raise ValueError(
                f"pillar_times {tuple(self.pillar_times.shape)} and "
                f"hazard_rates {tuple(self.hazard_rates.shape)} must match"
            )
        if self.pillar_times.numel() == 0:
            raise ValueError("need at least one pillar")
        pillars = self.pillar_times.detach()
        if bool((pillars <= 0.0).any()):
            raise ValueError("pillar times must be positive")
        if pillars.numel() > 1 and bool((pillars[1:] <= pillars[:-1]).any()):
            raise ValueError("pillar times must be strictly increasing")

    @classmethod
    def from_credit_curve(
        cls,
        curve: "object",
        *,
        requires_grad: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> "PiecewiseHazard":
        """Adapt a bootstrapped :class:`market_data.fetcher.CreditCurve`.

        Duck-typed on ``pillar_times``/``hazard_rates`` so this module stays
        independent of ``market_data``: the dependency runs data-layer ->
        engine, and importing it here would reverse that.

        Args:
            curve: Any object exposing ``pillar_times`` and ``hazard_rates``
                as sequences of the same length.
            requires_grad: Make the hazard vector a differentiable leaf, giving
                per-pillar credit deltas from one backward pass.
            dtype: Target dtype, default ``torch.float64`` -- credit curves are
                bootstrapped in double and downcasting them here would throw
                away the precision that made the bootstrap reprice.
            device: Target device.

        Returns:
            An equivalent :class:`PiecewiseHazard`.

        Raises:
            TypeError: If the object lacks the required attributes.
        """
        for attribute in ("pillar_times", "hazard_rates"):
            if not hasattr(curve, attribute):
                raise TypeError(
                    f"{type(curve).__name__} has no {attribute!r}; expected a "
                    "CreditCurve-like object with 'pillar_times' and "
                    "'hazard_rates'"
                )
        resolved = torch.float64 if dtype is None else dtype
        pillars = torch.as_tensor(
            list(curve.pillar_times), dtype=resolved, device=device
        )
        hazards = torch.as_tensor(
            list(curve.hazard_rates), dtype=resolved, device=device
        )
        if requires_grad:
            hazards = hazards.clone().requires_grad_(True)
        return cls(pillar_times=pillars, hazard_rates=hazards)

    def survival_probability(self, times: Tensor) -> Tensor:
        """:math:`Q` on ``times``; see :func:`piecewise_survival_probability`."""
        return piecewise_survival_probability(
            times, self.pillar_times, self.hazard_rates
        )

    def marginal_default_probability(self, times: Tensor) -> Tensor:
        """Per-interval default probabilities on ``times``."""
        return piecewise_marginal_default_probability(
            times, self.pillar_times, self.hazard_rates
        )


def piecewise_survival_probability(
    times: Tensor, pillar_times: Tensor, hazard_rates: Tensor
) -> Tensor:
    r"""Survival under a piecewise-constant hazard, differentiable in :math:`h`.

    .. math::
        H(t) = \sum_j h_j \,\bigl[\min(t, T_j) - T_{j-1}\bigr]^+,
        \qquad Q(t) = e^{-H(t)}

    with :math:`T_0 = 0` and the final segment's upper limit taken as
    :math:`+\infty`, which extends the last hazard flat beyond the last pillar
    rather than letting :math:`Q` stop decaying.

    The bracket is exactly the overlap of :math:`[0, t]` with segment
    :math:`j`, so :math:`H` is *linear* in ``hazard_rates`` and one backward
    pass yields every :math:`\partial\,\cdot\,/\partial h_j`. Evaluating it
    this way -- rather than interpolating a precomputed cumulative hazard --
    is what keeps the curve on the autograd tape.

    Args:
        times: Observation grid of shape ``(n,)``, non-negative.
        pillar_times: :math:`T_1 < \dots < T_J`, shape ``(n_pillars,)``.
        hazard_rates: :math:`h_1 \dots h_J`, shape ``(n_pillars,)``.

    Returns:
        Shape ``(n,)``, values in :math:`(0, 1]` for non-negative hazards, with
        :math:`Q(0) = 1` exactly. Differentiable w.r.t. ``hazard_rates``.

    Raises:
        ValueError: On a non-1-D grid or mismatched pillar shapes.
    """
    if times.ndim != 1:
        raise ValueError(f"times must be 1-dimensional, got {tuple(times.shape)}")
    if pillar_times.shape != hazard_rates.shape:
        raise ValueError(
            f"pillar_times {tuple(pillar_times.shape)} and hazard_rates "
            f"{tuple(hazard_rates.shape)} must match"
        )

    pillars = pillar_times.to(device=times.device, dtype=times.dtype)
    hazards = hazard_rates.to(device=times.device, dtype=times.dtype)

    # Segment j spans (start_j, end_j]. The last segment is unbounded above so
    # the final hazard extrapolates flat.
    start = torch.cat([torch.zeros(1, dtype=times.dtype, device=times.device),
                       pillars[:-1]])
    end = torch.cat([
        pillars[:-1],
        torch.full((1,), float("inf"), dtype=times.dtype, device=times.device),
    ])

    overlap = torch.clamp(
        torch.minimum(times.unsqueeze(-1), end) - start, min=0.0
    )
    return torch.exp(-(overlap * hazards).sum(dim=-1))


def piecewise_marginal_default_probability(
    times: Tensor, pillar_times: Tensor, hazard_rates: Tensor
) -> Tensor:
    r"""Per-interval default probabilities :math:`Q(t_{i-1}) - Q(t_i)`.

    Same convention as :func:`marginal_default_probability`, so the flat and
    piecewise paths are interchangeable downstream.

    Args:
        times: Observation grid of shape ``(n + 1,)``.
        pillar_times: Pillar maturities.
        hazard_rates: Piecewise hazards.

    Returns:
        Shape ``(n,)``, differentiable w.r.t. ``hazard_rates``.

    Raises:
        ValueError: If fewer than two grid points are supplied.
    """
    if times.shape[0] < 2:
        raise ValueError("need at least two grid points to form default intervals")
    survival = piecewise_survival_probability(times, pillar_times, hazard_rates)
    return survival[:-1] - survival[1:]


def resolve_survival_curve(
    times: Tensor,
    *,
    hazard_rate: Optional[ScalarLike] = None,
    credit_curve: Optional["object"] = None,
    survival: Optional[Tensor] = None,
) -> Tensor:
    r"""Resolve the three credit specifications to a survival curve.

    Exactly one of the three must be supplied. Silently preferring one over
    another would make a caller who passed both believe the curve was used when
    it was not -- an error that shows up as a plausible-but-wrong CVA.

    Args:
        times: Observation grid of shape ``(n + 1,)``.
        hazard_rate: Flat intensity :math:`\lambda`.
        credit_curve: A :class:`PiecewiseHazard`, or a duck-typed
            ``CreditCurve`` exposing ``pillar_times``/``hazard_rates``.
        survival: Explicit :math:`Q` of shape ``(n + 1,)``.

    Returns:
        :math:`Q` on ``times``, shape ``(n + 1,)``, preserving whatever
        autograd graph the chosen input carried.

    Raises:
        ValueError: If not exactly one specification is given, or if an
            explicit ``survival`` has the wrong shape.
    """
    supplied = [
        name
        for name, value in (
            ("hazard_rate", hazard_rate),
            ("credit_curve", credit_curve),
            ("survival", survival),
        )
        if value is not None
    ]
    if len(supplied) != 1:
        raise ValueError(
            "provide exactly one credit specification -- 'hazard_rate' "
            f"(flat), 'credit_curve' or 'survival' -- got {supplied or 'none'}"
        )

    if hazard_rate is not None:
        return survival_probability(times, hazard_rate)

    if survival is not None:
        if survival.shape != times.shape:
            raise ValueError(
                f"survival shape {tuple(survival.shape)} does not match times "
                f"{tuple(times.shape)}"
            )
        return survival.to(device=times.device, dtype=times.dtype)

    curve = credit_curve
    if not isinstance(curve, PiecewiseHazard):
        curve = PiecewiseHazard.from_credit_curve(curve, dtype=times.dtype)
    return curve.survival_probability(times)


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
    hazard_rate: Optional[ScalarLike] = None,
    recovery_rate: float = 0.4,
    *,
    discount_rate: Optional[ScalarLike] = None,
    curve: Optional[Tensor] = None,
    credit_curve: Optional["object"] = None,
    survival: Optional[Tensor] = None,
    convention: str = "endpoint",
) -> Tensor:
    r"""Shared discrete integrator for the CVA and DVA legs.

    Evaluates :math:`(1-R)\sum_i P(t_i)\, d\!PD_i\, DF(t_i)` where :math:`P` is
    either the EE profile (CVA) or the ENE profile (DVA).

    Args:
        profile_values: EE or ENE of shape ``(n_steps + 1,)``.
        times: Observation grid of shape ``(n_steps + 1,)``.
        hazard_rate: Flat intensity of the defaulting party. Mutually
            exclusive with ``credit_curve`` and ``survival``.
        recovery_rate: Recovery rate :math:`R` of the defaulting party.
        discount_rate: Flat rate used to build :math:`DF`. Mutually exclusive
            with ``curve``.
        curve: Explicit discount factors of shape ``(n_steps + 1,)``. Mutually
            exclusive with ``discount_rate``.
        credit_curve: Piecewise-constant credit curve. Mutually exclusive with
            ``hazard_rate`` and ``survival``.
        survival: Explicit :math:`Q` of shape ``(n_steps + 1,)``. Mutually
            exclusive with ``hazard_rate`` and ``credit_curve``.
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

    resolved_survival = resolve_survival_curve(
        grid,
        hazard_rate=hazard_rate,
        credit_curve=credit_curve,
        survival=survival,
    )
    default_probabilities = resolved_survival[:-1] - resolved_survival[1:]

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
    hazard_rate: Optional[ScalarLike] = None,
    recovery_rate: float = 0.4,
    *,
    discount_rate: Optional[ScalarLike] = None,
    curve: Optional[Tensor] = None,
    credit_curve: Optional["object"] = None,
    survival: Optional[Tensor] = None,
    convention: str = "endpoint",
) -> Tensor:
    r"""Unilateral CVA by discrete summation of the credit-loss integral.

    .. math:: CVA = (1-R)\sum_{i=1}^{N} EE(t_i)\; d\!PD_i\; DF(t_i)

    Args:
        ee: Expected Exposure profile of shape ``(n_steps + 1,)``, from
            :func:`src.xva.exposure.expected_exposure`.
        times: Observation grid of shape ``(n_steps + 1,)`` in years.
        hazard_rate: Counterparty intensity :math:`\lambda` under a flat
            curve. Pass a tensor with ``requires_grad=True`` for the credit
            delta. Mutually exclusive with ``credit_curve`` and ``survival``.
        recovery_rate: Counterparty recovery :math:`R`, default ``0.4``
            (the standard senior-unsecured assumption).
        discount_rate: Flat continuously compounded rate for :math:`DF`.
            Mutually exclusive with ``curve``.
        curve: Explicit discount factors of shape ``(n_steps + 1,)``. Mutually
            exclusive with ``discount_rate``.
        credit_curve: A :class:`PiecewiseHazard`, or a bootstrapped
            :class:`market_data.fetcher.CreditCurve`, replacing the flat-hazard
            assumption with an observed term structure. Build it with
            ``PiecewiseHazard.from_credit_curve(curve, requires_grad=True)`` to
            get per-pillar credit deltas from one backward pass.
        survival: Explicit :math:`Q` of shape ``(n_steps + 1,)``, for a curve
            this module does not need to know the shape of. Differentiable if
            the caller built it that way.
        convention: ``"endpoint"`` (default, matches the formula above) or
            ``"average"`` for the trapezoidal refinement.

    Returns:
        0-dim differentiable tensor holding a **positive** CVA magnitude.

    Note:
        CVA is returned as a positive cost. The credit-adjusted value of the
        portfolio is ``risk_free_value - cva``.

    Example:
        Bucketed credit deltas against a bootstrapped curve::

            >>> hazards = PiecewiseHazard.from_credit_curve(
            ...     bootstrapped, requires_grad=True
            ... )
            >>> cva = compute_unilateral_cva(
            ...     ee, times, credit_curve=hazards, discount_rate=0.03
            ... )
            >>> cva.backward()
            >>> hazards.hazard_rates.grad  # one dCVA/dh_j per pillar
    """
    return _integrate_credit_leg(
        ee,
        times,
        hazard_rate,
        recovery_rate,
        discount_rate=discount_rate,
        curve=curve,
        credit_curve=credit_curve,
        survival=survival,
        convention=convention,
    )


def compute_unilateral_dva(
    ene: Tensor,
    times: Tensor,
    hazard_rate: Optional[ScalarLike] = None,
    recovery_rate: float = 0.4,
    *,
    discount_rate: Optional[ScalarLike] = None,
    curve: Optional[Tensor] = None,
    credit_curve: Optional["object"] = None,
    survival: Optional[Tensor] = None,
    convention: str = "endpoint",
) -> Tensor:
    r"""Unilateral DVA -- the mirror of :func:`compute_unilateral_cva`.

    .. math:: DVA = (1-R_{\text{own}})\sum_{i=1}^{N} ENE(t_i)\;
              d\!PD^{\text{own}}_i\; DF(t_i)

    Args:
        ene: Expected Negative Exposure of shape ``(n_steps + 1,)``.
        times: Observation grid of shape ``(n_steps + 1,)``.
        hazard_rate: **Our own** flat intensity, not the counterparty's.
        recovery_rate: Our own recovery rate.
        discount_rate: Flat rate for :math:`DF`. Mutually exclusive with
            ``curve``.
        curve: Explicit discount factors. Mutually exclusive with
            ``discount_rate``.
        credit_curve: **Our own** piecewise credit curve. Using the
            counterparty's here is a real and easy mistake -- DVA is driven by
            our own default.
        survival: Explicit **own** :math:`Q` of shape ``(n_steps + 1,)``.
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
        credit_curve=credit_curve,
        survival=survival,
        convention=convention,
    )


def compute_xva(
    mtm: Tensor,
    times: Tensor,
    *,
    hazard_rate: Optional[ScalarLike] = None,
    discount_rate: ScalarLike,
    recovery_rate: float = 0.4,
    own_hazard_rate: Optional[ScalarLike] = None,
    own_recovery_rate: Optional[float] = None,
    credit_curve: Optional["object"] = None,
    own_credit_curve: Optional["object"] = None,
    confidence_level: float = 0.95,
    convention: str = "endpoint",
) -> XVAResult:
    r"""Compute the exposure profile, CVA and DVA from an MtM surface in one pass.

    Args:
        mtm: Netted mark-to-market surface of shape ``(n_paths, n_steps + 1)``.
        times: Observation grid of shape ``(n_steps + 1,)``.
        hazard_rate: Flat counterparty intensity :math:`\lambda`. Mutually
            exclusive with ``credit_curve``.
        discount_rate: Flat continuously compounded discount rate.
        recovery_rate: Counterparty recovery, default ``0.4``.
        own_hazard_rate: Our own flat intensity for the DVA leg. Defaults to
            ``hazard_rate`` when omitted (a symmetric-credit assumption; state
            it explicitly in any write-up that relies on it).
        own_recovery_rate: Our own recovery for the DVA leg. Defaults to
            ``recovery_rate``.
        credit_curve: Counterparty piecewise credit curve, replacing the flat
            hazard. Mutually exclusive with ``hazard_rate``.
        own_credit_curve: Our own credit curve for the DVA leg. Defaults to
            ``credit_curve`` -- the same symmetric-credit assumption as
            ``own_hazard_rate``, and equally worth stating explicitly.
        confidence_level: Quantile level for the PFE in the returned profile.
        convention: ``"endpoint"`` or ``"average"``.

    Returns:
        An :class:`XVAResult` whose tensors all remain on the autograd tape of
        ``mtm``.
    """
    if (hazard_rate is None) == (credit_curve is None):
        raise ValueError(
            "provide exactly one of 'hazard_rate' (flat) or 'credit_curve' "
            "(term structure)"
        )

    profile = compute_exposure_profile(mtm, times, confidence_level=confidence_level)
    effective_own_recovery = (
        recovery_rate if own_recovery_rate is None else own_recovery_rate
    )

    # Own-credit defaults mirror the counterparty leg, keeping the flat and
    # curve paths symmetric. Passing an own_hazard_rate alongside a
    # counterparty credit_curve is accepted deliberately: a flat own curve
    # against a bootstrapped counterparty curve is a common desk setup.
    if own_credit_curve is not None:
        own_hazard_for_leg, own_curve_for_leg = None, own_credit_curve
    elif own_hazard_rate is not None:
        own_hazard_for_leg, own_curve_for_leg = own_hazard_rate, None
    else:
        own_hazard_for_leg, own_curve_for_leg = hazard_rate, credit_curve

    cva = compute_unilateral_cva(
        profile.ee,
        profile.times,
        hazard_rate,
        recovery_rate,
        discount_rate=discount_rate,
        credit_curve=credit_curve,
        convention=convention,
    )
    dva = compute_unilateral_dva(
        profile.ene,
        profile.times,
        own_hazard_for_leg,
        effective_own_recovery,
        discount_rate=discount_rate,
        credit_curve=own_curve_for_leg,
        convention=convention,
    )
    return XVAResult(
        cva=cva,
        dva=dva,
        profile=profile,
        hazard_rate=(
            None
            if hazard_rate is None
            else float(as_tensor_like(hazard_rate, profile.ee).detach())
        ),
        recovery_rate=recovery_rate,
    )


def make_cva_valuation_fn(
    simulator: GBMSimulator,
    dW: Tensor,
    legs: Sequence[SwapLeg],
    *,
    recovery_rate: float = 0.4,
    rate: Optional[ScalarLike] = None,
    credit_curve: Optional["object"] = None,
    survival: Optional[Tensor] = None,
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

    Args:
        simulator: Path generator supplying the time grid and ``dt``.
        dW: Fixed Brownian increments, so the returned function is a
            deterministic map from parameters to CVA (common random numbers).
        legs: The netting set.
        recovery_rate: Counterparty recovery.
        rate: Discount rate, if not supplied per-call in ``params``.
        credit_curve: Optional piecewise credit curve replacing the flat
            hazard. When given, the returned function no longer reads
            ``"hazard_rate"``; it accepts an optional scalar ``"hazard_shift"``
            instead (see below).
        survival: Optional explicit :math:`Q` on the simulator grid, replacing
            the flat hazard. Mutually exclusive with ``credit_curve``.
        convention: ``"endpoint"`` or ``"average"``.

    Returns:
        A :data:`~src.pricer.options.PriceFn` returning a 0-dim CVA tensor.
        Which keys it expects depends on the credit specification:

        * **flat** (default) -- ``"s0"``, ``"sigma"``, ``"hazard_rate"``,
          optionally ``"rate"``/``"mu"``;
        * **``credit_curve``** -- ``"s0"``, ``"sigma"``, and optionally
          ``"hazard_shift"``: a **parallel shift** added to every pillar
          hazard, defaulting to zero. It exists because
          :func:`~src.pricer.greeks.aad_greeks` reduces each gradient with
          ``float(grad.detach())`` and so is scalar-only -- it cannot carry a
          per-pillar delta vector. The shift is the scalar summary of curve
          risk; for the bucketed vector use
          :func:`cva_credit_bucket_deltas`.
        * **``survival``** -- ``"s0"``, ``"sigma"``; the curve is fixed.

    Raises:
        ValueError: On an invalid convention or recovery rate, or if both
            ``credit_curve`` and ``survival`` are supplied.

    Example:
        >>> cva_fn = make_cva_valuation_fn(sim, dW, legs, rate=0.03)
        >>> greeks = aad_greeks(cva_fn, {"s0": 100.0, "sigma": 0.2,
        ...                              "hazard_rate": 0.02})
        >>> sorted(greeks.greeks)
        ['hazard_rate', 's0', 'sigma']

        Against a bootstrapped curve, with the parallel credit sensitivity::

            >>> cva_fn = make_cva_valuation_fn(
            ...     sim, dW, legs, rate=0.03, credit_curve=hazards
            ... )
            >>> greeks = aad_greeks(cva_fn, {"s0": 100.0, "sigma": 0.2,
            ...                              "hazard_shift": 0.0})
    """
    _validate_convention(convention)
    _validate_rate_like(recovery_rate, "recovery_rate")
    if credit_curve is not None and survival is not None:
        raise ValueError("provide at most one of 'credit_curve' or 'survival'")

    legs = tuple(legs)
    times = simulator.time_grid()

    base_curve: Optional[PiecewiseHazard] = None
    if credit_curve is not None:
        base_curve = (
            credit_curve
            if isinstance(credit_curve, PiecewiseHazard)
            else PiecewiseHazard.from_credit_curve(
                credit_curve, dtype=times.dtype
            )
        )

    def cva_fn(params: Mapping[str, Tensor]) -> Tensor:
        resolved_rate, drift = resolve_rate_and_drift(params, rate)
        paths = simulate_gbm(params["s0"], drift, params["sigma"], dW, simulator.dt)
        mtm = portfolio_swap_mtm(paths, times, legs, resolved_rate)
        ee = expected_exposure(mtm)

        if base_curve is not None:
            # A parallel shift keeps the curve on the tape while presenting a
            # single scalar knob, so the existing greeks API applies unchanged.
            shift = params.get("hazard_shift")
            hazards = base_curve.hazard_rates
            if shift is not None:
                hazards = hazards + as_tensor_like(shift, hazards)
            shifted = PiecewiseHazard(
                pillar_times=base_curve.pillar_times, hazard_rates=hazards
            )
            return compute_unilateral_cva(
                ee, times, recovery_rate=recovery_rate,
                discount_rate=resolved_rate, credit_curve=shifted,
                convention=convention,
            )

        if survival is not None:
            return compute_unilateral_cva(
                ee, times, recovery_rate=recovery_rate,
                discount_rate=resolved_rate, survival=survival,
                convention=convention,
            )

        return compute_unilateral_cva(
            ee,
            times,
            params["hazard_rate"],
            recovery_rate,
            discount_rate=resolved_rate,
            convention=convention,
        )

    return cva_fn


def cva_credit_bucket_deltas(
    ee: Tensor,
    times: Tensor,
    credit_curve: "object",
    recovery_rate: float = 0.4,
    *,
    discount_rate: Optional[ScalarLike] = None,
    curve: Optional[Tensor] = None,
    convention: str = "endpoint",
) -> tuple:
    r"""Per-pillar credit deltas :math:`\partial CVA/\partial h_j`.

    One backward pass returns the whole vector, because :math:`H(t)` is linear
    in the hazards -- the same O(1)-in-parameter-count property that motivates
    AAD everywhere else in this engine. Bumping each pillar separately would
    need :math:`J+1` revaluations.

    This is separate from :func:`cva_aad_greeks` because that returns
    ``dict[str, float]`` via ``float(grad.detach())`` and cannot hold a vector.

    Args:
        ee: Expected Exposure of shape ``(n_steps + 1,)``.
        times: Observation grid of shape ``(n_steps + 1,)``.
        credit_curve: A :class:`PiecewiseHazard` or duck-typed
            ``CreditCurve``. Its hazards need not already require grad; a
            differentiable copy is made internally.
        recovery_rate: Counterparty recovery.
        discount_rate: Flat discount rate. Mutually exclusive with ``curve``.
        curve: Explicit discount factors. Mutually exclusive with
            ``discount_rate``.
        convention: ``"endpoint"`` or ``"average"``.

    Returns:
        ``(cva, deltas)`` -- the 0-dim CVA and a ``(n_pillars,)`` tensor of
        :math:`\partial CVA/\partial h_j`, both detached.
    """
    resolved = (
        credit_curve
        if isinstance(credit_curve, PiecewiseHazard)
        else PiecewiseHazard.from_credit_curve(credit_curve, dtype=times.dtype)
    )
    hazards = resolved.hazard_rates.detach().clone().requires_grad_(True)
    differentiable = PiecewiseHazard(
        pillar_times=resolved.pillar_times, hazard_rates=hazards
    )

    value = compute_unilateral_cva(
        ee.detach(),
        times,
        recovery_rate=recovery_rate,
        discount_rate=discount_rate,
        curve=curve,
        credit_curve=differentiable,
        convention=convention,
    )
    (deltas,) = torch.autograd.grad(value, hazards)
    return value.detach(), deltas.detach()


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
