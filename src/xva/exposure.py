r"""Counterparty credit exposure profiles from a differentiable mark-to-market surface.

Scope and definitions
---------------------
Let :math:`V_t` denote the netted mark-to-market of a portfolio with a single
counterparty, observed on a discrete grid :math:`0 = t_0 < t_1 < \dots < t_N = T`
and simulated over :math:`M` Monte-Carlo paths. The credit *exposure* is the
amount that would be lost if the counterparty defaulted at :math:`t` and
recovered nothing:

.. math:: E_t = \max(V_t, 0) \equiv V_t^+.

The positive part appears because, under a standard close-out netting
agreement (ISDA Master Agreement with CSA), a defaulting counterparty still
owes the surviving party any net positive balance, whereas a net *negative*
balance must still be paid in full by the survivor. Exposure is therefore
asymmetric even though the underlying MtM is not.

The three profiles implemented here are the industry-standard summaries of the
exposure distribution at each grid date:

**Expected Exposure (EE)** -- the first moment,

.. math:: EE(t) = \mathbb{E}^{\mathbb{Q}}\!\left[\max(V_t, 0)\right],

estimated by the sample mean :math:`\widehat{EE}(t) = M^{-1}\sum_j (V_t^{(j)})^+`.
This is the quantity that enters the CVA integral in :mod:`src.xva.cva`.

**Expected Negative Exposure (ENE)** -- the mirror image seen from the
counterparty's perspective,

.. math:: ENE(t) = \mathbb{E}^{\mathbb{Q}}\!\left[\max(-V_t, 0)\right],

which drives DVA (our own default is a *benefit* to us, since our liability is
extinguished at recovery).

**Potential Future Exposure (PFE)** -- a tail quantile rather than a moment,

.. math:: PFE_\alpha(t) = \inf\{x : \mathbb{P}(V_t^+ \le x) \ge \alpha\},

with :math:`\alpha = 0.95` by convention. PFE is a *limit-monitoring* measure
(does this trade breach the counterparty credit line?), not a pricing measure;
it deliberately ignores the shape of the distribution beyond the quantile.

Measure conventions
-------------------
Everything here is computed under the pricing measure :math:`\mathbb{Q}`,
because these profiles feed a *valuation adjustment*. Regulatory capital and
credit-limit calculations conventionally use the real-world measure
:math:`\mathbb{P}` instead; the code is measure-agnostic (it consumes whatever
MtM surface it is handed), so switching amounts to changing the drift passed to
the simulator, not changing this module.

Differentiability
-----------------
Every operation below is out-of-place and autograd-traceable, so an exposure
profile computed from parameter-dependent paths remains a node in the same tape
as :math:`S_0`, :math:`\sigma`, and the credit parameters. Two subtleties are
worth recording explicitly, since they matter for the AAD-vs-bump validation:

1. ``clamp(V, min=0)`` is not differentiable at :math:`V = 0`. PyTorch assigns
   subgradient :math:`0` there. Since :math:`\{V_t = 0\}` is a null set under
   the GBM law for :math:`t > 0`, the resulting pathwise derivative of ``EE``
   is almost surely correct and unbiased. This is precisely the regime where
   AAD beats bump-and-revalue: finite differences straddle the kink for the
   :math:`O(Mh)` paths that cross zero inside the bump, which injects an
   :math:`O(h)` bias that does *not* vanish as :math:`M \to \infty`.

2. ``PFE`` differentiates through an **order statistic**. The gradient is
   routed entirely to the single path (or the two interpolated paths) sitting
   at the quantile, which is a valid pathwise derivative but a high-variance
   estimator, and it is a *step* function of the parameters at finite
   :math:`M` (the identity of the quantile path changes discontinuously).
   Consequently PFE sensitivities should **not** be validated against finite
   differences at tight tolerance -- see the note in ``tests/test_phase2.py``.
   CVA depends only on ``EE``, so this does not contaminate the Phase 2
   acceptance criterion.

References
----------
Gregory, J. (2020). *Counterparty Credit Risk, Funding, Collateral and
Capital*, 3rd ed., Wiley -- Chapters 8-11.
Green, A. (2015). *XVA: Credit, Funding and Capital Valuation Adjustments*,
Wiley -- Chapter 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor

__all__ = [
    "ScalarLike",
    "ExposureProfile",
    "CSATerms",
    "as_tensor_like",
    "positive_exposure",
    "negative_exposure",
    "expected_exposure",
    "expected_negative_exposure",
    "exposure_standard_error",
    "differentiable_quantile",
    "potential_future_exposure",
    "expected_positive_exposure",
    "compute_exposure_profile",
    "GRID_UNIFORMITY_EPS_FACTOR",
    "validate_uniform_grid",
    "mpor_lag_steps",
    "collateral_required",
    "collateral_balance",
    "collateralized_exposure",
    "expected_collateralized_exposure",
    "compute_collateralized_exposure_profile",
]

#: How many machine epsilons of the *horizon* a grid may deviate from uniform.
#: 64 leaves ~2 decimal digits of headroom over the worst measured linspace
#: rounding (0.61 eps at float32, N=1000) while still rejecting a real
#: perturbation by orders of magnitude.
GRID_UNIFORMITY_EPS_FACTOR = 64.0

#: A scalar model input may be a Python float or a 0-dim tensor (the latter
#: when a gradient with respect to it is wanted).
ScalarLike = Union[float, Tensor]


def as_tensor_like(value: ScalarLike, reference: Tensor) -> Tensor:
    """Coerce a scalar-like value to a 0-dim tensor matching ``reference``.

    Gradients propagate through the ``.to()`` conversion, so a leaf tensor
    passed in as ``value`` remains connected to the tape.

    Args:
        value: Python float or scalar tensor.
        reference: Tensor supplying the target device and dtype.

    Returns:
        A 0-dim tensor on ``reference``'s device with ``reference``'s dtype.

    Raises:
        ValueError: If ``value`` is a tensor holding more than one element.
    """
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"expected a scalar, got shape {tuple(value.shape)}")
        converted = value.to(device=reference.device, dtype=reference.dtype)
        return converted if converted.ndim == 0 else converted.reshape(())
    return torch.as_tensor(float(value), device=reference.device, dtype=reference.dtype)


def _check_mtm(mtm: Tensor) -> None:
    """Validate the shape of a mark-to-market surface.

    Args:
        mtm: Candidate surface.

    Raises:
        ValueError: If ``mtm`` is not 2-dimensional or has no paths.
    """
    if mtm.ndim != 2:
        raise ValueError(
            f"mtm must have shape (n_paths, n_steps + 1), got {tuple(mtm.shape)}"
        )
    if mtm.shape[0] == 0:
        raise ValueError("mtm must contain at least one path")


@dataclass(frozen=True)
class ExposureProfile:
    r"""A complete set of exposure profiles on a fixed observation grid.

    All tensors remain attached to the autograd graph of the MtM surface they
    were derived from, so any of them can be differentiated with respect to the
    underlying market and model parameters.

    Attributes:
        times: Observation grid of shape ``(n_steps + 1,)``, in years.
        ee: Expected Exposure :math:`EE(t)`, shape ``(n_steps + 1,)``.
            Non-negative by construction.
        ene: Expected Negative Exposure :math:`ENE(t)`, shape
            ``(n_steps + 1,)``. Non-negative by construction.
        pfe: Potential Future Exposure :math:`PFE_\alpha(t)`, shape
            ``(n_steps + 1,)``. Non-negative by construction.
        confidence_level: The quantile level :math:`\alpha` used for ``pfe``.
        n_paths: Number of Monte-Carlo paths behind the estimates.
    """

    times: Tensor
    ee: Tensor
    ene: Tensor
    pfe: Tensor
    confidence_level: float
    n_paths: int

    @property
    def epe(self) -> Tensor:
        r"""Expected Positive Exposure: the time-average of :math:`EE(t)`.

        .. math:: EPE = \frac{1}{T}\int_0^T EE(t)\,dt

        evaluated by the trapezoidal rule on ``times``. This is the scalar
        summary Basel III uses as the basis for the EAD calculation.

        Returns:
            0-dim tensor, still differentiable.
        """
        return expected_positive_exposure(self.ee, self.times)

    @property
    def max_pfe(self) -> Tensor:
        """Peak PFE across the profile, the usual credit-line headline number."""
        return self.pfe.max()


def positive_exposure(mtm: Tensor) -> Tensor:
    r"""Path-wise positive exposure :math:`V_t^+ = \max(V_t, 0)`.

    Args:
        mtm: Netted mark-to-market surface of shape ``(n_paths, n_steps + 1)``.

    Returns:
        Tensor of the same shape, non-negative everywhere.

    Note:
        Implemented with out-of-place :func:`torch.clamp`; the in-place
        ``clamp_`` variant would corrupt the autograd tape of ``mtm``.
    """
    _check_mtm(mtm)
    return torch.clamp(mtm, min=0.0)


def negative_exposure(mtm: Tensor) -> Tensor:
    r"""Path-wise negative exposure :math:`\max(-V_t, 0) = (-V_t)^+`.

    This is the exposure *of the counterparty to us*, i.e. the amount we would
    fail to pay if we defaulted. It drives DVA.

    Args:
        mtm: Netted mark-to-market surface of shape ``(n_paths, n_steps + 1)``.

    Returns:
        Tensor of the same shape, non-negative everywhere.
    """
    _check_mtm(mtm)
    return torch.clamp(-mtm, min=0.0)


def expected_exposure(mtm: Tensor) -> Tensor:
    r"""Expected Exposure profile :math:`EE(t) = \mathbb{E}[\max(V_t, 0)]`.

    Args:
        mtm: Netted mark-to-market surface of shape ``(n_paths, n_steps + 1)``.

    Returns:
        Tensor of shape ``(n_steps + 1,)``, non-negative, differentiable.

    Note:
        The estimator is the plain sample mean across paths, which is unbiased.
        Its Monte-Carlo standard error at each date is
        :math:`\mathrm{sd}(V_t^+)/\sqrt{M}`; use
        :func:`exposure_standard_error` if that diagnostic is needed.
    """
    return positive_exposure(mtm).mean(dim=0)


def expected_negative_exposure(mtm: Tensor) -> Tensor:
    r"""Expected Negative Exposure :math:`ENE(t) = \mathbb{E}[\max(-V_t, 0)]`.

    Args:
        mtm: Netted mark-to-market surface of shape ``(n_paths, n_steps + 1)``.

    Returns:
        Tensor of shape ``(n_steps + 1,)``, non-negative, differentiable.
    """
    return negative_exposure(mtm).mean(dim=0)


def exposure_standard_error(mtm: Tensor) -> Tensor:
    r"""Monte-Carlo standard error of the :math:`EE(t)` estimator at each date.

    .. math:: \mathrm{se}\big(\widehat{EE}(t)\big)
        = \frac{\mathrm{sd}\!\left(V_t^+\right)}{\sqrt{M}}

    Args:
        mtm: Netted mark-to-market surface of shape ``(n_paths, n_steps + 1)``.

    Returns:
        Detached tensor of shape ``(n_steps + 1,)``. Detached deliberately: a
        standard error is a diagnostic about the estimator, not a term in the
        valuation, and letting it enter the tape would silently pollute
        sensitivities.

    Note:
        Under antithetic sampling the paired paths are negatively correlated,
        so this i.i.d. formula *overstates* the true error -- it is a
        conservative bound in that case, not an equality.
    """
    _check_mtm(mtm)
    exposure = torch.clamp(mtm.detach(), min=0.0)
    n_paths = exposure.shape[0]
    if n_paths < 2:
        return torch.zeros_like(exposure[0])
    return exposure.std(dim=0, unbiased=True) / math.sqrt(n_paths)


def differentiable_quantile(values: Tensor, q: float, dim: int = 0) -> Tensor:
    r"""Linear-interpolated sample quantile that preserves autograd.

    Reproduces the convention of :func:`torch.quantile` (and NumPy's default
    ``method="linear"``): for :math:`M` sorted observations
    :math:`x_{(0)} \le \dots \le x_{(M-1)}`, the level-:math:`q` quantile is

    .. math::
        h = q(M-1),\qquad
        \hat{Q}(q) = x_{(\lfloor h \rfloor)}
            + (h - \lfloor h \rfloor)\left(x_{(\lceil h \rceil)}
            - x_{(\lfloor h \rfloor)}\right).

    This is implemented via :func:`torch.sort` and :func:`torch.index_select`
    rather than :func:`torch.quantile` for two reasons: ``torch.quantile``
    imposes a hard limit of :math:`2^{24}` elements on its input (readily
    exceeded by a realistic ``(n_paths, n_steps)`` surface), and the explicit
    formulation makes the gradient routing auditable -- the adjoint flows only
    to the one or two order statistics that define the quantile.

    Args:
        values: Sample tensor.
        q: Quantile level in :math:`[0, 1]`.
        dim: Dimension along which to reduce (the path dimension).

    Returns:
        Tensor with ``dim`` removed, differentiable w.r.t. ``values``.

    Raises:
        ValueError: If ``q`` lies outside :math:`[0, 1]` or the reduced
            dimension is empty.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile level must lie in [0, 1], got {q}")
    n = values.shape[dim]
    if n == 0:
        raise ValueError("cannot take a quantile over an empty dimension")
    if n == 1:
        return values.squeeze(dim)

    sorted_values, _ = torch.sort(values, dim=dim)
    position = q * (n - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    weight = position - lower_index

    index_lower = torch.tensor([lower_index], device=values.device, dtype=torch.long)
    index_upper = torch.tensor([upper_index], device=values.device, dtype=torch.long)
    lower = sorted_values.index_select(dim, index_lower).squeeze(dim)
    upper = sorted_values.index_select(dim, index_upper).squeeze(dim)
    return lower + weight * (upper - lower)


def potential_future_exposure(mtm: Tensor, confidence_level: float = 0.95) -> Tensor:
    r"""Potential Future Exposure :math:`PFE_\alpha(t)`, the exposure quantile.

    Args:
        mtm: Netted mark-to-market surface of shape ``(n_paths, n_steps + 1)``.
        confidence_level: Quantile level :math:`\alpha`, ``0.95`` by
            convention.

    Returns:
        Tensor of shape ``(n_steps + 1,)``, non-negative, differentiable.

    Note:
        The quantile is taken of the *positive* exposure :math:`V_t^+`, not of
        :math:`V_t`. For :math:`\alpha \ge 0.5` and a portfolio that can be
        out-of-the-money the two differ: the quantile of :math:`V_t` may be
        negative, whereas exposure cannot be. Taking the quantile after
        flooring is the market-standard definition and guarantees
        :math:`PFE \ge 0`.
    """
    return differentiable_quantile(positive_exposure(mtm), confidence_level, dim=0)


def expected_positive_exposure(ee: Tensor, times: Tensor) -> Tensor:
    r"""Time-averaged Expected Exposure (EPE) by the trapezoidal rule.

    .. math:: EPE = \frac{1}{T}\int_0^T EE(t)\,dt
        \approx \frac{1}{t_N - t_0}\sum_{i=1}^{N}
        \frac{EE(t_{i-1}) + EE(t_i)}{2}\,(t_i - t_{i-1})

    Args:
        ee: Expected Exposure profile of shape ``(n_steps + 1,)``.
        times: Matching observation grid of shape ``(n_steps + 1,)``.

    Returns:
        0-dim differentiable tensor.

    Raises:
        ValueError: On shape mismatch, fewer than two grid points, or a
            degenerate (zero-length) time window.
    """
    if ee.ndim != 1 or times.ndim != 1:
        raise ValueError("ee and times must both be 1-dimensional")
    if ee.shape != times.shape:
        raise ValueError(f"shape mismatch: ee {tuple(ee.shape)} vs times {tuple(times.shape)}")
    if ee.shape[0] < 2:
        raise ValueError("need at least two grid points to integrate")

    grid = times.to(device=ee.device, dtype=ee.dtype)
    horizon = grid[-1] - grid[0]
    if float(horizon) <= 0.0:
        raise ValueError("times must span a positive interval")

    widths = grid[1:] - grid[:-1]
    trapezoids = 0.5 * (ee[:-1] + ee[1:]) * widths
    return trapezoids.sum() / horizon


def compute_exposure_profile(
    mtm: Tensor,
    times: Tensor,
    *,
    confidence_level: float = 0.95,
) -> ExposureProfile:
    r"""Compute EE, ENE and PFE from a mark-to-market surface in one pass.

    Args:
        mtm: Netted mark-to-market surface of shape ``(n_paths, n_steps + 1)``,
            typically produced by
            :func:`src.pricer.options.portfolio_swap_mtm`.
        times: Observation grid of shape ``(n_steps + 1,)`` in years, typically
            :meth:`src.models.gbm.GBMSimulator.time_grid`.
        confidence_level: Quantile level :math:`\alpha` for the PFE.

    Returns:
        A fully differentiable :class:`ExposureProfile`.

    Raises:
        ValueError: If ``mtm`` and ``times`` are inconsistently shaped, or the
            confidence level is outside :math:`[0, 1]`.

    Example:
        >>> paths = simulator.simulate(100.0, 0.03, 0.2, dW=dW)
        >>> mtm = portfolio_swap_mtm(paths, simulator.time_grid(), legs, 0.03)
        >>> profile = compute_exposure_profile(mtm, simulator.time_grid())
        >>> profile.ee.shape
        torch.Size([65])
    """
    _check_mtm(mtm)
    if times.ndim != 1 or times.shape[0] != mtm.shape[1]:
        raise ValueError(
            f"times must have shape ({mtm.shape[1]},) to match mtm, got {tuple(times.shape)}"
        )
    if not 0.0 <= confidence_level <= 1.0:
        raise ValueError(f"confidence_level must lie in [0, 1], got {confidence_level}")

    # Compute the two clamped surfaces once and reuse them, rather than calling
    # the public helpers twice and re-clamping.
    exposure = torch.clamp(mtm, min=0.0)
    counterparty_exposure = torch.clamp(-mtm, min=0.0)

    return ExposureProfile(
        times=times.to(device=mtm.device, dtype=mtm.dtype),
        ee=exposure.mean(dim=0),
        ene=counterparty_exposure.mean(dim=0),
        pfe=differentiable_quantile(exposure, confidence_level, dim=0),
        confidence_level=confidence_level,
        n_paths=int(mtm.shape[0]),
    )


# ==========================================================================
# Collateralised exposure -- variation margin under a CSA
# ==========================================================================
r"""
Collateral model
----------------
Under a Credit Support Annex the parties exchange **variation margin** to
neutralise the mark-to-market. Writing :math:`C_t` for the collateral balance
held by us at time :math:`t` (positive = we hold their collateral, negative =
we have posted ours), the exposure at default becomes the *uncollateralised*
shortfall

.. math:: E_t = \max\!\left(V_t - C_t,\; 0\right).

Three CSA frictions stop :math:`C_t` from simply equalling :math:`V_t`:

**Threshold** :math:`H` -- an unsecured allowance. No collateral is called
until the MtM exceeds it, so the required balance is the *soft-threshold*
(shrinkage) function

.. math::
    C^{\text{req}}(V) = \max(V - H_{\text{rec}}, 0) - \max(-V - H_{\text{post}}, 0),

which is zero on :math:`[-H_{\text{post}}, H_{\text{rec}}]` and parallel to
:math:`V` outside it. Setting :math:`H_{\text{post}} = \infty` models a one-way
CSA in which we never post.

**Minimum Transfer Amount** (MTA) -- a transfer only happens once the gap
between the required and the current balance exceeds the MTA, which stops
operationally pointless dust movements. This makes the balance *path
dependent*: :math:`C_{t_i}` depends on :math:`C_{t_{i-1}}`, so it must be
rolled forward step by step.

**Margin Period of Risk** (MPOR) -- the lag between the last effective margin
call and the actual close-out (dispute, notice, liquidation). It is modelled as
a fixed delay: the balance in force at :math:`t` is the one implied by the MtM
observed at :math:`t - \text{MPOR}`. Ten business days
(:math:`\approx 10/252` years) is the standard regulatory assumption for a
daily-margined netting set.

An important and deliberate consequence
---------------------------------------
With :math:`\text{MPOR} = 0` and non-negative thresholds the collateralised
pathwise exposure collapses to the exact identity

.. math:: \max(V_t - C_t, 0) = \min\!\left(V_t^+,\; H_{\text{rec}}\right),

so collateral can only ever *reduce* exposure, and
:math:`EE_{\text{collat}} \le EE_{\text{uncollat}}` holds pathwise.

With :math:`\text{MPOR} > 0` that guarantee is **lost**, and this is a genuine
feature of the model rather than a defect. If we posted collateral against a
deeply negative MtM and the market then reverses, we are exposed both to what
they now owe us *and* to the collateral we posted and cannot recall:
:math:`V_{t-\text{MPOR}} = -5` gives :math:`C = -5`, and a move to
:math:`V_t = +3` leaves an exposure of :math:`3 - (-5) = 8 > 3`. Collateral
still reduces exposure dramatically *in aggregate* for realistic parameters,
but the pathwise inequality is not a theorem once the margin period bites --
which is precisely why the MPOR is the dominant driver of collateralised CVA.

Differentiability
-----------------
Every step is out-of-place and autograd-traceable. The MTA roll-forward uses
:func:`torch.where` against a boolean mask, whose adjoint routes the gradient
to whichever branch was selected -- the correct pathwise derivative, since the
set where the transfer test binds exactly is a null set.
"""


@dataclass(frozen=True)
class CSATerms:
    r"""Credit Support Annex parameters governing variation margin.

    Attributes:
        threshold: Unsecured threshold :math:`H_{\text{rec}}` above which the
            counterparty must post to us. ``0.0`` means fully collateralised.
        minimum_transfer_amount: MTA -- the smallest collateral movement that
            will actually be transferred. ``0.0`` disables the stickiness and
            enables a fast vectorised path.
        margin_period_of_risk: MPOR in **years**. ``10.0 / 252.0`` is the usual
            ten-business-day regulatory assumption. ``0.0`` means instantaneous,
            frictionless margining.
        threshold_post: Our own threshold :math:`H_{\text{post}}` above which we
            post to them. Defaults to ``threshold`` (a symmetric CSA). Pass
            ``float("inf")`` for a one-way CSA under which we never post.
        initial_balance: Collateral held at :math:`t = 0`, before any margin
            call implied by the simulation has settled.
    """

    threshold: float = 0.0
    minimum_transfer_amount: float = 0.0
    margin_period_of_risk: float = 0.0
    threshold_post: Optional[float] = None
    initial_balance: float = 0.0

    def __post_init__(self) -> None:
        if self.threshold < 0.0:
            raise ValueError(f"threshold must be non-negative, got {self.threshold}")
        if self.minimum_transfer_amount < 0.0:
            raise ValueError(
                f"minimum_transfer_amount must be non-negative, "
                f"got {self.minimum_transfer_amount}"
            )
        if self.margin_period_of_risk < 0.0:
            raise ValueError(
                f"margin_period_of_risk must be non-negative, "
                f"got {self.margin_period_of_risk}"
            )
        if self.threshold_post is not None and self.threshold_post < 0.0:
            raise ValueError(
                f"threshold_post must be non-negative, got {self.threshold_post}"
            )

    @property
    def effective_threshold_post(self) -> float:
        """Our posting threshold, defaulting to a symmetric CSA."""
        return self.threshold if self.threshold_post is None else self.threshold_post

    @property
    def is_uncollateralised(self) -> bool:
        """``True`` when the terms are economically equivalent to no CSA."""
        return math.isinf(self.threshold) or (
            math.isinf(self.effective_threshold_post) and math.isinf(self.threshold)
        )


def mpor_lag_steps(margin_period_of_risk: float, dt: float) -> int:
    r"""Convert an MPOR in years to a whole number of grid steps.

    Args:
        margin_period_of_risk: MPOR in years.
        dt: Uniform grid step in years.

    Returns:
        The nearest non-negative integer number of steps.

    Raises:
        ValueError: If ``dt`` is non-positive or the MPOR is negative.

    Note:
        Rounding to the nearest step means a sub-step MPOR on a coarse grid
        collapses to zero lag. That is the honest behaviour -- a grid cannot
        resolve a delay finer than its own resolution -- but it does mean a
        realistic 10-business-day MPOR needs a grid at least that fine before
        it has any effect. Callers who care should check
        ``mpor_lag_steps(...) > 0``.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if margin_period_of_risk < 0.0:
        raise ValueError(
            f"margin_period_of_risk must be non-negative, got {margin_period_of_risk}"
        )
    return int(round(margin_period_of_risk / dt))


def collateral_required(
    mtm_observed: Tensor,
    threshold_receive: float,
    threshold_post: float,
) -> Tensor:
    r"""Required collateral balance implied by an observed mark-to-market.

    .. math::
        C^{\text{req}}(V) = \max(V - H_{\text{rec}}, 0)
                          - \max(-V - H_{\text{post}}, 0)

    Args:
        mtm_observed: MtM values of any shape.
        threshold_receive: :math:`H_{\text{rec}}`, their threshold. May be
            ``inf`` (they never post).
        threshold_post: :math:`H_{\text{post}}`, our threshold. May be ``inf``
            (we never post).

    Returns:
        Tensor of the same shape as ``mtm_observed``, differentiable through
        both branches.

    Note:
        Infinite thresholds are handled naturally: ``clamp(V - inf, min=0)``
        is identically zero, which is exactly "no collateral is ever called in
        that direction".
    """
    receive_leg = torch.clamp(mtm_observed - threshold_receive, min=0.0)
    post_leg = torch.clamp(-mtm_observed - threshold_post, min=0.0)
    return receive_leg - post_leg


def _grid_step(times: Tensor) -> float:
    """Return the uniform step of ``times``, validating uniformity.

    The uniformity test is **relative to the horizon, not to the step**.

    A grid built by ``torch.linspace`` is not exactly uniform in floating point:
    the time values are :math:`O(T)`, so each carries a rounding error of order
    :math:`\\epsilon T`, and the differences between consecutive steps inherit
    it. That error is *independent of* :math:`N`. Testing it against
    :math:`\\Delta t = T/N` therefore injects a spurious factor of :math:`N`,
    and any fixed ``rel_tol * dt`` bound fails once :math:`N` is large enough.

    Measured on ``torch.linspace(0, 1, N+1)``, worst deviation over step:

    ==========  ============  ============
    N           float32       float64
    ==========  ============  ============
    252         1.1e-05       2.4e-14
    1000        7.3e-05       1.1e-13
    2520        9.4e-05       2.8e-13
    10000       4.3e-04       1.0e-12
    ==========  ============  ============

    So a ``1e-6 * dt`` bound rejects a perfectly good float32 grid from
    :math:`N \\approx 100`, and even ``1e-4 * dt`` breaks by
    :math:`N \\approx 2700`. Bounding against :math:`\\epsilon T` instead is
    :math:`N`-independent and dtype-aware, and still rejects a genuinely
    non-uniform grid by many orders of magnitude.

    Args:
        times: Observation grid of shape ``(n_steps + 1,)``.

    Returns:
        The step size in years.

    Raises:
        ValueError: If the grid has fewer than two points, is not strictly
            increasing, or deviates from uniform by more than
            ``GRID_UNIFORMITY_EPS_FACTOR`` machine epsilons of the horizon.
    """
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("times must be a 1-D grid with at least two points")
    steps = (times[1:] - times[:-1]).detach()
    dt = float(steps[0])
    if dt <= 0.0:
        raise ValueError("times must be strictly increasing")

    horizon = abs(float(times[-1] - times[0]))
    epsilon = torch.finfo(times.dtype).eps
    tolerance = GRID_UNIFORMITY_EPS_FACTOR * epsilon * max(horizon, 1.0)
    deviation = float((steps - steps[0]).abs().max())
    if deviation > tolerance:
        raise ValueError(
            "a uniform time grid is required; worst step deviation "
            f"{deviation:.3e} exceeds {tolerance:.3e} "
            f"({GRID_UNIFORMITY_EPS_FACTOR} x eps x horizon for "
            f"{times.dtype})"
        )
    return dt


def validate_uniform_grid(times: Tensor) -> float:
    """Public alias of the canonical grid check.

    Args:
        times: Observation grid of shape ``(n_steps + 1,)``.

    Returns:
        The uniform step size in years.

    Raises:
        ValueError: If the grid is not uniform. See :func:`_grid_step`.
    """
    return _grid_step(times)


def collateral_balance(mtm: Tensor, times: Tensor, terms: CSATerms) -> Tensor:
    r"""Roll the variation-margin balance :math:`C_t` forward across the grid.

    Two code paths produce identical results; the split exists purely for speed:

    * **MTA = 0** -- the balance is memoryless, equal to
      :math:`C^{\text{req}}(V_{t-\text{MPOR}})`, so the whole surface is one
      shrinkage plus a column shift. Fully vectorised.
    * **MTA > 0** -- the transfer test compares against the *current* balance,
      making the recursion genuinely path dependent, so the balance is rolled
      forward one grid step at a time and stacked. Still fully differentiable:
      the loop builds graph nodes, it never writes in place.

    Args:
        mtm: Netted mark-to-market surface of shape ``(n_paths, n_steps + 1)``.
        times: Uniform observation grid of shape ``(n_steps + 1,)``.
        terms: The CSA parameters.

    Returns:
        Collateral balance surface of shape ``(n_paths, n_steps + 1)``.
        Positive entries mean we hold their collateral.

    Raises:
        ValueError: On shape mismatch or a non-uniform time grid.
    """
    _check_mtm(mtm)
    if times.ndim != 1 or times.shape[0] != mtm.shape[1]:
        raise ValueError(
            f"times must have shape ({mtm.shape[1]},) to match mtm, got {tuple(times.shape)}"
        )

    dt = _grid_step(times.to(device=mtm.device, dtype=mtm.dtype))
    lag = mpor_lag_steps(terms.margin_period_of_risk, dt)
    n_paths, n_columns = mtm.shape

    threshold_receive = terms.threshold
    threshold_post = terms.effective_threshold_post
    required = collateral_required(mtm, threshold_receive, threshold_post)

    # ---- Fast path: no MTA, so the balance has no memory -----------------
    if terms.minimum_transfer_amount == 0.0:
        if lag == 0:
            return required
        carried = min(lag, n_columns)
        # The pre-first-call columns hold the contractual opening balance, a
        # constant: `new_full` deliberately produces a non-differentiable
        # node, which is correct because nothing in the model links it to the
        # market parameters.
        head = mtm.new_full((n_paths, carried), terms.initial_balance)
        if carried == n_columns:
            return head
        return torch.cat((head, required[:, : n_columns - carried]), dim=1)

    # ---- General path: MTA makes the recursion path dependent ------------
    mta = terms.minimum_transfer_amount
    balance = mtm.new_full((n_paths,), terms.initial_balance)
    columns = []
    for index in range(n_columns):
        observed = index - lag
        if observed >= 0:
            gap = required[:, observed] - balance
            # A transfer occurs only when the gap clears the MTA. torch.where
            # keeps this on the tape: the mask is a constant selector and the
            # adjoint flows through whichever branch was taken.
            transfer = torch.where(gap.abs() >= mta, gap, torch.zeros_like(gap))
            balance = balance + transfer
        columns.append(balance)
    return torch.stack(columns, dim=1)


def collateralized_exposure(mtm: Tensor, times: Tensor, terms: CSATerms) -> Tensor:
    r"""Path-wise collateralised exposure :math:`\max(V_t - C_t, 0)`.

    Args:
        mtm: Netted mark-to-market surface of shape ``(n_paths, n_steps + 1)``.
        times: Uniform observation grid of shape ``(n_steps + 1,)``.
        terms: The CSA parameters.

    Returns:
        Tensor of shape ``(n_paths, n_steps + 1)``, non-negative, differentiable.

    Note:
        With ``margin_period_of_risk == 0`` and finite non-negative thresholds
        this equals ``min(max(mtm, 0), threshold)`` exactly. With a positive
        MPOR it can exceed the uncollateralised exposure on individual paths --
        see the module-level discussion; that is the modelled economics of a
        margin period, not an implementation artefact.
    """
    return torch.clamp(mtm - collateral_balance(mtm, times, terms), min=0.0)


def expected_collateralized_exposure(
    mtm: Tensor, times: Tensor, terms: CSATerms
) -> Tensor:
    r"""Collateralised Expected Exposure :math:`EE_{\text{collat}}(t)`.

    .. math:: EE_{\text{collat}}(t) = \mathbb{E}\!\left[\max(V_t - C_t, 0)\right]

    Args:
        mtm: Netted mark-to-market surface of shape ``(n_paths, n_steps + 1)``.
        times: Uniform observation grid of shape ``(n_steps + 1,)``.
        terms: The CSA parameters.

    Returns:
        Tensor of shape ``(n_steps + 1,)``, non-negative, differentiable. Feed
        it straight to :func:`src.xva.cva.compute_unilateral_cva` to obtain
        collateralised CVA.
    """
    return collateralized_exposure(mtm, times, terms).mean(dim=0)


def compute_collateralized_exposure_profile(
    mtm: Tensor,
    times: Tensor,
    terms: CSATerms,
    *,
    confidence_level: float = 0.95,
) -> ExposureProfile:
    r"""Full EE / ENE / PFE profile computed on the collateralised net position.

    The collateralised profile is simply the ordinary profile of the *net*
    surface :math:`V_t - C_t`, so this delegates to
    :func:`compute_exposure_profile` rather than duplicating the reduction
    logic. ``ene`` is then the counterparty's exposure to us net of the
    collateral we hold, which is the correct input for collateralised DVA.

    Args:
        mtm: Netted mark-to-market surface of shape ``(n_paths, n_steps + 1)``.
        times: Uniform observation grid of shape ``(n_steps + 1,)``.
        terms: The CSA parameters.
        confidence_level: Quantile level :math:`\alpha` for the PFE.

    Returns:
        A fully differentiable :class:`ExposureProfile` of the net position.
    """
    net_position = mtm - collateral_balance(mtm, times, terms)
    return compute_exposure_profile(net_position, times, confidence_level=confidence_level)
