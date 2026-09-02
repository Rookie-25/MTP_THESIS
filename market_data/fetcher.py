r"""Market data loading and credit-curve bootstrapping for the XVA engine.

Three independent capabilities, layered so the mathematics is testable without
a network:

===================  ======================================================
layer                contents
===================  ======================================================
pure (no I/O)        :class:`YieldCurve`, :class:`CreditCurve`,
                     :func:`bootstrap_hazard_rates`,
                     :func:`model_par_spread`, :func:`clean_option_chain`
thin I/O wrappers    :func:`fetch_spot`, :func:`fetch_implied_vol_surface`,
                     :func:`fetch_sofr_curve`, :func:`fetch_treasury_curve`,
                     :func:`fetch_discount_curve`
===================  ======================================================

``yfinance`` and ``pandas_datareader`` are imported *lazily*, inside the
functions that need them. The pure layer therefore imports and runs in CI with
neither library installed and no network access -- which is the only way the
bootstrapper can have meaningful unit tests, since a network fixture is neither
deterministic nor available offline.

Credit curve bootstrapping
==========================
Given par CDS spreads at pillars :math:`T_1 < \dots < T_J` (conventionally 1Y,
3Y, 5Y, 10Y), recover a piecewise-constant hazard :math:`h(t) = h_j` for
:math:`t \in (T_{j-1}, T_j]`, so that

.. math::
    Q(t) = \exp\left(-\int_0^t h(u)\,du\right)

reprices every input quote. The cumulative hazard is piecewise *linear*, so
:math:`Q` is continuous, strictly decreasing, and exactly representable at any
:math:`t` -- no interpolation error inside the pillar range.

A par CDS has zero value at inception, i.e. protection leg equals premium leg:

.. math::
    S_j \cdot A(T_j) = P(T_j)

with the risky annuity (premium leg per unit spread, including
accrual-on-default at the interval midpoint)

.. math::
    A(T) = \sum_i \Delta_i D(t_i)
           \left[ Q(t_i) + \tfrac{1}{2}\bigl(Q(t_{i-1}) - Q(t_i)\bigr) \right]

and the protection leg, with default placed at the interval midpoint,

.. math::
    P(T) = (1 - R) \sum_i D(\bar t_i)\bigl(Q(t_{i-1}) - Q(t_i)\bigr),
    \qquad \bar t_i = \tfrac{1}{2}(t_{i-1} + t_i).

The bootstrap is sequential: with :math:`h_1, \dots, h_{j-1}` already fixed by
the shorter pillars, :math:`h_j` affects only :math:`Q` beyond :math:`T_{j-1}`,
so each pillar is a one-dimensional root find. Raising :math:`h_j` increases
:math:`P` and decreases :math:`A`, hence increases the model par spread
strictly monotonically -- so the root is unique and bracketing is safe.

Why not just use the credit triangle
------------------------------------
:math:`h \approx S / (1 - R)` is exact only in the continuum limit with zero
rates and a flat spread curve. It cannot represent a *term structure*: a
1Y/3Y/5Y/10Y quote set carries forward credit information that a single flat
intensity discards. :func:`bootstrap_hazard_rates` is checked against the
triangle in exactly the degenerate case where they must agree (see
``tests/test_market_data.py``), and used for the initial bracket.

A caveat on "the SOFR curve" from FRED
======================================
**FRED does not publish a term SOFR swap (OIS) curve.** What it has is:

* ``SOFR`` -- the overnight rate;
* ``SOFR30DAYAVG``, ``SOFR90DAYAVG``, ``SOFR180DAYAVG`` -- backward-looking
  averages, so the SOFR family reaches only ~0.5Y.

Discounting a 10Y CDS needs the long end, which is why
:func:`fetch_discount_curve` defaults to splicing the SOFR short end onto
Treasury constant-maturity yields (``DGS1``..``DGS30``). Two approximations are
being made there, and both are real:

1. **Treasury yields are not SOFR.** The swap-Treasury basis is tens of basis
   points and time-varying. A production desk takes the SOFR OIS curve from
   ICE, Bloomberg or Refinitiv; FRED cannot supply it.
2. **CMT yields are par coupon yields, not zero rates.** Treating them as zero
   rates skips the par-curve bootstrap, worth a few basis points at 10Y for a
   normally-shaped curve.

Both are second-order for CVA, where the exposure profile dominates, but they
are approximations rather than the real curve and are labelled as such on the
returned object (:attr:`YieldCurve.label`).
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd
    import torch
    from torch import Tensor

__all__ = [
    "BASIS_POINT",
    "CDSQuote",
    "CreditCurve",
    "VolSurfaceData",
    "YieldCurve",
    "black_vega",
    "bootstrap_hazard_rates",
    "clean_option_chain",
    "fetch_discount_curve",
    "fetch_fred_series",
    "fetch_implied_vol_surface",
    "fetch_sofr_curve",
    "fetch_spot",
    "fetch_treasury_curve",
    "model_par_spread",
    "payment_schedule",
    "premium_leg_annuity",
    "protection_leg_pv",
]

#: One basis point as a decimal. CDS spreads are quoted in bp; every internal
#: computation is in decimals, and the conversion happens exactly once, in
#: :attr:`CDSQuote.spread`.
BASIS_POINT = 1.0e-4

#: Standard CDS premium frequency (quarterly, IMM-dated in practice).
DEFAULT_CDS_FREQUENCY = 4

#: Widest hazard rate the bootstrap will search. 5.0 corresponds to a ~99.3%
#: one-year default probability -- far beyond any traded name, so hitting this
#: bound means the quote is bad, not that the search was too narrow.
MAX_HAZARD_RATE = 5.0

#: Root-finder tolerance on the hazard rate itself.
HAZARD_TOLERANCE = 1.0e-12

#: FRED series for the SOFR complex, tenor in years -> series id. Backward
#: averages are treated as spot rates for a term equal to their window, which
#: is the standard approximation for curve construction at the short end.
SOFR_SERIES: Dict[float, str] = {
    1.0 / 360.0: "SOFR",
    30.0 / 360.0: "SOFR30DAYAVG",
    90.0 / 360.0: "SOFR90DAYAVG",
    180.0 / 360.0: "SOFR180DAYAVG",
}

#: FRED Treasury constant-maturity series, tenor in years -> series id.
TREASURY_SERIES: Dict[float, str] = {
    1.0 / 12.0: "DGS1MO",
    0.25: "DGS3MO",
    0.5: "DGS6MO",
    1.0: "DGS1",
    2.0: "DGS2",
    3.0: "DGS3",
    5.0: "DGS5",
    7.0: "DGS7",
    10.0: "DGS10",
    20.0: "DGS20",
    30.0: "DGS30",
}


# ==========================================================================
# Rate conventions
# ==========================================================================
def semiannual_to_continuous(par_yield: np.ndarray) -> np.ndarray:
    r"""Convert a semiannually-compounded yield to a continuous rate.

    :math:`r_c = 2\ln(1 + y/2)`. Treasury CMT yields are quoted
    bond-equivalent (semiannual), while every discount factor in this codebase
    is :math:`e^{-rt}`; skipping this conversion overstates discount factors by
    roughly :math:`y^2 t / 4`.

    Args:
        par_yield: Yields as decimals (0.043 for 4.3%).

    Returns:
        Continuously compounded rates, same shape.
    """
    return 2.0 * np.log1p(np.asarray(par_yield, dtype=float) / 2.0)


def simple_to_continuous(
    simple_rate: np.ndarray, tenor: np.ndarray
) -> np.ndarray:
    r"""Convert a simple (money-market) rate to a continuous rate.

    :math:`r_c = \ln(1 + r\tau)/\tau`. SOFR and its averages are simple
    ACT/360 money-market rates.

    Args:
        simple_rate: Rates as decimals.
        tenor: Accrual period in years, matching ``simple_rate`` elementwise.

    Returns:
        Continuously compounded rates, same shape.

    Raises:
        ValueError: If any tenor is non-positive.
    """
    rate = np.asarray(simple_rate, dtype=float)
    years = np.asarray(tenor, dtype=float)
    if np.any(years <= 0.0):
        raise ValueError("tenor must be positive to convert a simple rate")
    return np.log1p(rate * years) / years


# ==========================================================================
# Discount curve
# ==========================================================================
@dataclass(frozen=True)
class YieldCurve:
    r"""A zero curve with continuously compounded rates.

    Interpolation is **linear in the zero rate**, with flat extrapolation at
    both ends. Linear-in-zero-rate is the common desk default and keeps
    discount factors smooth and positive everywhere; flat extrapolation avoids
    the negative long-end forwards that linear extrapolation can produce.

    Attributes:
        tenors: Pillar maturities in years, strictly increasing, all positive.
        zero_rates: Continuously compounded zero rates at those pillars.
        as_of: Observation date, for provenance.
        label: Human-readable description, including any approximations made
            in construction. Carried so a downstream result can state which
            curve produced it.
    """

    tenors: np.ndarray
    zero_rates: np.ndarray
    as_of: Optional[_dt.date] = None
    label: str = "unspecified"

    def __post_init__(self) -> None:
        tenors = np.asarray(self.tenors, dtype=float).ravel()
        rates = np.asarray(self.zero_rates, dtype=float).ravel()
        if tenors.size == 0:
            raise ValueError("a yield curve needs at least one pillar")
        if tenors.shape != rates.shape:
            raise ValueError(
                f"tenors {tenors.shape} and zero_rates {rates.shape} must match"
            )
        if np.any(~np.isfinite(tenors)) or np.any(~np.isfinite(rates)):
            raise ValueError("curve contains non-finite values")
        if np.any(tenors <= 0.0):
            raise ValueError("all tenors must be positive")
        if tenors.size > 1 and np.any(np.diff(tenors) <= 0.0):
            raise ValueError("tenors must be strictly increasing")
        object.__setattr__(self, "tenors", tenors)
        object.__setattr__(self, "zero_rates", rates)

    # ---- construction ------------------------------------------------
    @classmethod
    def flat(
        cls, rate: float, *, horizon: float = 30.0, label: str = "flat"
    ) -> "YieldCurve":
        """A constant-rate curve, for tests and for isolating credit effects.

        Args:
            rate: Continuously compounded rate.
            horizon: Longest pillar, in years.
            label: Curve label.

        Returns:
            A two-pillar curve that is exactly flat everywhere.
        """
        return cls(
            tenors=np.array([min(1.0, horizon), horizon], dtype=float),
            zero_rates=np.array([rate, rate], dtype=float),
            label=f"{label} @ {rate:.4%}",
        )

    # ---- evaluation --------------------------------------------------
    def zero_rate(self, time: np.ndarray) -> np.ndarray:
        """Interpolated continuous zero rate.

        Args:
            time: Maturities in years; scalars and arrays both accepted.

        Returns:
            Zero rates, broadcast to the shape of ``time``.
        """
        query = np.asarray(time, dtype=float)
        # np.interp clamps outside the range, giving exactly the flat
        # extrapolation documented above.
        return np.interp(query, self.tenors, self.zero_rates)

    def discount_factor(self, time: np.ndarray) -> np.ndarray:
        r""":math:`DF(t) = e^{-r(t)\,t}`, with :math:`DF(0) = 1` exactly.

        Args:
            time: Maturities in years. Must be non-negative.

        Returns:
            Discount factors in :math:`(0, 1]` for non-negative rates.

        Raises:
            ValueError: If any time is negative.
        """
        query = np.asarray(time, dtype=float)
        if np.any(query < 0.0):
            raise ValueError("discount factors need non-negative times")
        return np.exp(-self.zero_rate(query) * query)

    def to_tensor(self, times: "Tensor") -> "Tensor":
        """Discount factors on a torch grid, for ``src.xva.cva``'s ``curve=``.

        ``compute_unilateral_cva`` accepts either a flat ``discount_rate`` or an
        explicit ``curve`` tensor; this produces the latter, so a real observed
        curve can be used in place of the flat-rate default.

        Args:
            times: 1-D grid of observation times, in years.

        Returns:
            Discount factors with the dtype and device of ``times``. The result
            is a constant with respect to autograd -- these are observed market
            data, not calibrated parameters, so no gradient flows through them.
        """
        import torch

        grid = times.detach().cpu().numpy().astype(float)
        factors = self.discount_factor(grid)
        return torch.as_tensor(factors, dtype=times.dtype, device=times.device)

    def __repr__(self) -> str:
        return (
            f"YieldCurve({self.tenors.size} pillars, "
            f"{self.tenors[0]:.3g}y-{self.tenors[-1]:.3g}y, "
            f"{self.zero_rates.min():.3%}-{self.zero_rates.max():.3%}, "
            f"label={self.label!r})"
        )


# ==========================================================================
# CDS quotes and the credit curve
# ==========================================================================
@dataclass(frozen=True)
class CDSQuote:
    """A single par CDS quote.

    Attributes:
        tenor: Maturity in years (1.0, 3.0, 5.0, 10.0 conventionally).
        spread_bps: Par spread in basis points.
    """

    tenor: float
    spread_bps: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.tenor) or self.tenor <= 0.0:
            raise ValueError(f"tenor must be positive and finite, got {self.tenor}")
        if not math.isfinite(self.spread_bps) or self.spread_bps <= 0.0:
            raise ValueError(
                f"spread_bps must be positive and finite, got {self.spread_bps}"
            )

    @property
    def spread(self) -> float:
        """Par spread as a decimal."""
        return self.spread_bps * BASIS_POINT


def payment_schedule(
    maturity: float, frequency: int = DEFAULT_CDS_FREQUENCY
) -> np.ndarray:
    """Premium payment dates for a CDS, in years from today.

    Built *backwards* from the maturity so that the maturity is always a
    payment date and only the first accrual period can be short. That matches
    market convention and, more importantly for a bootstrap, guarantees the
    schedule terminates exactly on the pillar being solved.

    Args:
        maturity: Contract maturity in years.
        frequency: Payments per year.

    Returns:
        Increasing times starting at (or just after) 0 and ending exactly at
        ``maturity``, shape ``(n_payments + 1,)``. The first entry is the
        accrual start, not a payment.

    Raises:
        ValueError: If maturity or frequency is non-positive.
    """
    if maturity <= 0.0:
        raise ValueError(f"maturity must be positive, got {maturity}")
    if frequency <= 0:
        raise ValueError(f"frequency must be positive, got {frequency}")

    n_payments = max(1, int(math.ceil(maturity * frequency - 1.0e-9)))
    times = maturity - np.arange(n_payments, -1, -1, dtype=float) / frequency
    times[0] = max(times[0], 0.0)
    # A maturity that is not a whole number of periods leaves a short first
    # stub; drop it if rounding made it degenerate.
    if times.size > 2 and times[1] - times[0] < 1.0e-9:
        times = times[1:]
    return times


@dataclass(frozen=True)
class CreditCurve:
    r"""A piecewise-constant hazard rate curve.

    The hazard is :math:`h_j` on :math:`(T_{j-1}, T_j]` with :math:`T_0 = 0`,
    and is held flat at :math:`h_J` beyond the last pillar. Cumulative hazard
    is therefore piecewise linear and :math:`Q(t) = e^{-H(t)}` is exact at
    every :math:`t`, with no interpolation error inside the pillar range.

    Attributes:
        pillar_times: :math:`T_1 < \dots < T_J` in years.
        hazard_rates: :math:`h_1, \dots, h_J`, non-negative.
        recovery_rate: The :math:`R` the curve was bootstrapped with. Stored
            because a hazard curve is meaningless without it -- the same
            spreads imply different hazards at different recoveries, and
            quoting :math:`Q(t)` without :math:`R` invites exactly that error.
        label: Provenance.
    """

    pillar_times: np.ndarray
    hazard_rates: np.ndarray
    recovery_rate: float = 0.4
    label: str = "unspecified"

    def __post_init__(self) -> None:
        pillars = np.asarray(self.pillar_times, dtype=float).ravel()
        hazards = np.asarray(self.hazard_rates, dtype=float).ravel()
        if pillars.size == 0:
            raise ValueError("a credit curve needs at least one pillar")
        if pillars.shape != hazards.shape:
            raise ValueError(
                f"pillar_times {pillars.shape} and hazard_rates "
                f"{hazards.shape} must match"
            )
        if np.any(~np.isfinite(pillars)) or np.any(~np.isfinite(hazards)):
            raise ValueError("credit curve contains non-finite values")
        if np.any(pillars <= 0.0):
            raise ValueError("pillar times must be positive")
        if pillars.size > 1 and np.any(np.diff(pillars) <= 0.0):
            raise ValueError("pillar times must be strictly increasing")
        if not 0.0 <= self.recovery_rate < 1.0:
            raise ValueError(
                f"recovery_rate must be in [0, 1), got {self.recovery_rate}"
            )
        object.__setattr__(self, "pillar_times", pillars)
        object.__setattr__(self, "hazard_rates", hazards)

    # ---- construction ------------------------------------------------
    @classmethod
    def flat(
        cls,
        hazard: float,
        *,
        horizon: float = 10.0,
        recovery_rate: float = 0.4,
        label: str = "flat",
    ) -> "CreditCurve":
        """A single-pillar curve, equivalent to ``src.xva.cva``'s flat model.

        Args:
            hazard: Constant intensity.
            horizon: Pillar location; the hazard is flat beyond it anyway.
            recovery_rate: Recovery assumption.
            label: Curve label.

        Returns:
            A one-pillar :class:`CreditCurve`.
        """
        return cls(
            pillar_times=np.array([horizon], dtype=float),
            hazard_rates=np.array([hazard], dtype=float),
            recovery_rate=recovery_rate,
            label=f"{label} h={hazard:.5f}",
        )

    # ---- evaluation --------------------------------------------------
    @property
    def _knots(self) -> np.ndarray:
        """Pillar times with a leading zero."""
        return np.concatenate(([0.0], self.pillar_times))

    @property
    def _cumulative_at_knots(self) -> np.ndarray:
        """Cumulative hazard at each knot, starting from ``H(0) = 0``."""
        knots = self._knots
        return np.concatenate(
            ([0.0], np.cumsum(self.hazard_rates * np.diff(knots)))
        )

    def cumulative_hazard(self, time: np.ndarray) -> np.ndarray:
        r""":math:`H(t) = \int_0^t h(u)\,du`, exact (piecewise linear).

        Args:
            time: Times in years, non-negative.

        Returns:
            Cumulative hazard, shape of ``time``.

        Raises:
            ValueError: If any time is negative.
        """
        query = np.asarray(time, dtype=float)
        if np.any(query < 0.0):
            raise ValueError("cumulative hazard needs non-negative times")
        knots = self._knots
        values = self._cumulative_at_knots
        result = np.interp(query, knots, values)
        # np.interp clamps; extend flat-hazard beyond the last pillar instead.
        beyond = query > knots[-1]
        if np.any(beyond):
            result = np.asarray(result, dtype=float).copy()
            result[beyond] = values[-1] + self.hazard_rates[-1] * (
                query[beyond] - knots[-1]
            )
        return result

    def hazard_at(self, time: np.ndarray) -> np.ndarray:
        """Instantaneous hazard, right-continuous at the pillars.

        Args:
            time: Times in years, non-negative.

        Returns:
            The piecewise-constant hazard, shape of ``time``.
        """
        query = np.asarray(time, dtype=float)
        # searchsorted with side="left" puts t == T_j in segment j, matching
        # the (T_{j-1}, T_j] convention.
        index = np.searchsorted(self.pillar_times, query, side="left")
        return self.hazard_rates[np.clip(index, 0, self.hazard_rates.size - 1)]

    def survival_probability(self, time: np.ndarray) -> np.ndarray:
        r""":math:`Q(t) = e^{-H(t)}`, with :math:`Q(0) = 1` exactly.

        Args:
            time: Times in years, non-negative.

        Returns:
            Survival probabilities in :math:`(0, 1]`, strictly decreasing
            wherever the hazard is strictly positive.
        """
        return np.exp(-self.cumulative_hazard(time))

    def marginal_default_probability(self, times: np.ndarray) -> np.ndarray:
        r"""Per-interval default probabilities :math:`Q(t_{i-1}) - Q(t_i)`.

        Matches the convention of
        :func:`src.xva.cva.marginal_default_probability` exactly, so the two
        are interchangeable on the same grid.

        Args:
            times: Increasing grid, shape ``(n + 1,)``.

        Returns:
            Shape ``(n,)``, non-negative, summing to :math:`Q(t_0) - Q(t_n)`.

        Raises:
            ValueError: If fewer than two grid points are given.
        """
        grid = np.asarray(times, dtype=float).ravel()
        if grid.size < 2:
            raise ValueError("need at least two grid points to form intervals")
        survival = self.survival_probability(grid)
        return survival[:-1] - survival[1:]

    def to_tensor(self, times: "Tensor") -> "Tensor":
        """Survival probabilities on a torch grid.

        Args:
            times: 1-D grid in years.

        Returns:
            :math:`Q` with the dtype and device of ``times``, as a constant
            with respect to autograd.
        """
        import torch

        grid = times.detach().cpu().numpy().astype(float)
        return torch.as_tensor(
            self.survival_probability(grid), dtype=times.dtype, device=times.device
        )

    def par_spread(
        self,
        maturity: float,
        discount_curve: YieldCurve,
        *,
        frequency: int = DEFAULT_CDS_FREQUENCY,
    ) -> float:
        """Model par spread at a maturity, as a decimal.

        Args:
            maturity: Contract maturity in years.
            discount_curve: Discount curve.
            frequency: Premium frequency.

        Returns:
            The breakeven spread this curve implies.
        """
        return model_par_spread(
            maturity,
            self,
            discount_curve,
            recovery_rate=self.recovery_rate,
            frequency=frequency,
        )

    def __repr__(self) -> str:
        pillars = ", ".join(
            f"{t:g}y={h:.5f}"
            for t, h in zip(self.pillar_times, self.hazard_rates)
        )
        return (
            f"CreditCurve({pillars}, R={self.recovery_rate:.2f}, "
            f"label={self.label!r})"
        )


# ==========================================================================
# CDS leg valuation
# ==========================================================================
def premium_leg_annuity(
    maturity: float,
    curve: CreditCurve,
    discount_curve: YieldCurve,
    *,
    frequency: int = DEFAULT_CDS_FREQUENCY,
    accrual_on_default: bool = True,
) -> float:
    r"""Risky annuity :math:`A(T)`: premium leg PV per unit of spread.

    .. math::
        A(T) = \sum_i \Delta_i D(t_i)
               \left[Q(t_i) + \tfrac{1}{2}(Q(t_{i-1}) - Q(t_i))\right]

    The bracketed term is survival to the payment date plus, when
    ``accrual_on_default``, the expected accrued coupon paid on a default
    inside the period -- approximated by placing that default at the midpoint.
    Dropping it biases the annuity low by roughly half a period's accrual on
    the default probability, which for a wide name is not negligible.

    Args:
        maturity: Contract maturity in years.
        curve: Credit curve supplying :math:`Q`.
        discount_curve: Discount curve supplying :math:`D`.
        frequency: Premium frequency.
        accrual_on_default: Include the midpoint accrual term.

    Returns:
        :math:`A(T)`, strictly positive.
    """
    schedule = payment_schedule(maturity, frequency)
    accruals = np.diff(schedule)
    pay_times = schedule[1:]

    survival = curve.survival_probability(schedule)
    discount = discount_curve.discount_factor(pay_times)

    survival_at_pay = survival[1:]
    interval_default = survival[:-1] - survival[1:]

    weight = survival_at_pay
    if accrual_on_default:
        weight = weight + 0.5 * interval_default
    return float(np.sum(accruals * discount * weight))


def protection_leg_pv(
    maturity: float,
    curve: CreditCurve,
    discount_curve: YieldCurve,
    *,
    recovery_rate: Optional[float] = None,
    frequency: int = DEFAULT_CDS_FREQUENCY,
) -> float:
    r"""Protection leg PV :math:`P(T)`, with default at the interval midpoint.

    .. math::
        P(T) = (1 - R)\sum_i D(\bar t_i)\,(Q(t_{i-1}) - Q(t_i))

    Discounting at the midpoint :math:`\bar t_i` rather than the period end is
    the more accurate placement for a default that can occur anywhere in the
    period, and it is what makes the flat-curve, zero-rate case reproduce the
    credit triangle to near machine precision.

    Args:
        maturity: Contract maturity in years.
        curve: Credit curve supplying :math:`Q`.
        discount_curve: Discount curve supplying :math:`D`.
        recovery_rate: Overrides the curve's own :math:`R` when given.
        frequency: Integration frequency; matched to the premium schedule so
            both legs share one partition.

    Returns:
        :math:`P(T)`, non-negative.
    """
    recovery = curve.recovery_rate if recovery_rate is None else recovery_rate
    schedule = payment_schedule(maturity, frequency)
    midpoints = 0.5 * (schedule[:-1] + schedule[1:])

    survival = curve.survival_probability(schedule)
    interval_default = survival[:-1] - survival[1:]
    discount = discount_curve.discount_factor(midpoints)

    return float((1.0 - recovery) * np.sum(discount * interval_default))


def model_par_spread(
    maturity: float,
    curve: CreditCurve,
    discount_curve: YieldCurve,
    *,
    recovery_rate: Optional[float] = None,
    frequency: int = DEFAULT_CDS_FREQUENCY,
    accrual_on_default: bool = True,
) -> float:
    r"""Breakeven spread :math:`S(T) = P(T) / A(T)`, as a decimal.

    Args:
        maturity: Contract maturity in years.
        curve: Credit curve.
        discount_curve: Discount curve.
        recovery_rate: Overrides the curve's own :math:`R` when given.
        frequency: Premium frequency.
        accrual_on_default: Include the midpoint accrual term.

    Returns:
        The par spread, as a decimal (not basis points).

    Raises:
        ZeroDivisionError: If the risky annuity vanishes, which requires a
            degenerate curve (certain immediate default).
    """
    annuity = premium_leg_annuity(
        maturity,
        curve,
        discount_curve,
        frequency=frequency,
        accrual_on_default=accrual_on_default,
    )
    if annuity <= 0.0:
        raise ZeroDivisionError(
            f"risky annuity is {annuity:g} at T={maturity}: the credit curve is "
            "degenerate (immediate certain default)"
        )
    protection = protection_leg_pv(
        maturity,
        curve,
        discount_curve,
        recovery_rate=recovery_rate,
        frequency=frequency,
    )
    return protection / annuity


# ==========================================================================
# The bootstrap
# ==========================================================================
def bootstrap_hazard_rates(
    quotes: Sequence[CDSQuote],
    discount_curve: YieldCurve,
    *,
    recovery_rate: float = 0.4,
    frequency: int = DEFAULT_CDS_FREQUENCY,
    accrual_on_default: bool = True,
    allow_negative_forward_hazard: bool = False,
    label: Optional[str] = None,
) -> CreditCurve:
    r"""Bootstrap a piecewise-constant hazard curve from par CDS spreads.

    Solves the pillars in ascending maturity order. At pillar :math:`j` the
    hazards :math:`h_1..h_{j-1}` are already fixed and affect :math:`Q` only up
    to :math:`T_{j-1}`, so :math:`h_j` alone determines the remaining survival
    and each pillar reduces to a one-dimensional root find.

    The objective is strictly increasing in :math:`h_j` -- a higher hazard both
    raises the protection leg and lowers the risky annuity -- so the root is
    unique and Brent's method on a bracket is guaranteed to converge.

    Args:
        quotes: Par CDS quotes, at least one. Sorted internally by tenor;
            duplicate tenors are rejected.
        discount_curve: Curve used to discount both legs.
        recovery_rate: :math:`R`, in :math:`[0, 1)`. Stored on the result --
            hazards are only meaningful alongside the recovery they assume.
        frequency: Premium frequency.
        accrual_on_default: Include the midpoint accrual in the annuity.
        allow_negative_forward_hazard: When ``True``, permit a negative
            :math:`h_j`. Defaults to ``False``: a negative forward hazard means
            survival *increases* over that period, which is arbitrageable and
            almost always a stale or crossed quote rather than a real signal.
        label: Provenance string for the result.

    Returns:
        A :class:`CreditCurve` with one pillar per quote, repricing every input
        quote to within the solver tolerance.

    Raises:
        ValueError: If ``quotes`` is empty, contains duplicate tenors, or has
            an invalid recovery rate.
        RuntimeError: If a pillar cannot be solved -- either because the quote
            implies a negative forward hazard (and that is disallowed) or
            because it exceeds :attr:`MAX_HAZARD_RATE`. The message names the
            offending pillar and what the model could reach, because a failure
            here is a data problem to be inspected, not a tolerance to loosen.
    """
    from scipy.optimize import brentq

    if not quotes:
        raise ValueError("need at least one CDS quote to bootstrap")
    if not 0.0 <= recovery_rate < 1.0:
        raise ValueError(f"recovery_rate must be in [0, 1), got {recovery_rate}")

    ordered = sorted(quotes, key=lambda quote: quote.tenor)
    tenors = [quote.tenor for quote in ordered]
    if len(set(tenors)) != len(tenors):
        raise ValueError(f"duplicate CDS tenors in quotes: {tenors}")

    pillars: List[float] = []
    hazards: List[float] = []

    for quote in ordered:
        target = quote.spread

        def mispricing(trial_hazard: float, _quote: CDSQuote = quote) -> float:
            """Model minus market spread, as a function of the new hazard."""
            trial = CreditCurve(
                pillar_times=np.array(pillars + [_quote.tenor], dtype=float),
                hazard_rates=np.array(hazards + [trial_hazard], dtype=float),
                recovery_rate=recovery_rate,
            )
            return (
                model_par_spread(
                    _quote.tenor,
                    trial,
                    discount_curve,
                    frequency=frequency,
                    accrual_on_default=accrual_on_default,
                )
                - target
            )

        # The credit triangle h ~ S/(1-R) is the exact answer in the flat,
        # zero-rate limit, so it is the natural centre for the bracket.
        guess = max(target / (1.0 - recovery_rate), 1.0e-8)
        low = -guess if allow_negative_forward_hazard else 0.0
        high = max(4.0 * guess, 1.0e-4)

        # Expand until the root is bracketed. The objective increases in the
        # trial hazard, so the sign at each end says which way to grow: both
        # ends below the target means the root is above `high`, both above
        # means it is below `low`. Expanding only upward would leave
        # allow_negative_forward_hazard inert, since the required hazard is
        # frequently more negative than the initial -guess.
        value_low = mispricing(low)
        value_high = mispricing(high)

        while value_high < 0.0 and high < MAX_HAZARD_RATE:
            high = min(high * 4.0, MAX_HAZARD_RATE)
            value_high = mispricing(high)

        if allow_negative_forward_hazard:
            while value_low > 0.0 and low > -MAX_HAZARD_RATE:
                low = max(low * 4.0, -MAX_HAZARD_RATE)
                value_low = mispricing(low)

        if value_low * value_high > 0.0:
            reachable = model_par_spread(
                quote.tenor,
                CreditCurve(
                    pillar_times=np.array(pillars + [quote.tenor], dtype=float),
                    hazard_rates=np.array(hazards + [low], dtype=float),
                    recovery_rate=recovery_rate,
                ),
                discount_curve,
                frequency=frequency,
                accrual_on_default=accrual_on_default,
            )
            if value_low > 0.0:
                raise RuntimeError(
                    f"cannot bootstrap the {quote.tenor:g}Y pillar: the quoted "
                    f"{quote.spread_bps:.1f}bp is below the "
                    f"{reachable / BASIS_POINT:.1f}bp this curve already "
                    f"implies at zero forward hazard. The spread term "
                    f"structure is inverted enough to require survival to "
                    f"*increase* over "
                    f"({pillars[-1] if pillars else 0.0:g}, {quote.tenor:g}]y, "
                    "which is arbitrageable. Check for a stale or crossed "
                    "quote, or pass allow_negative_forward_hazard=True to "
                    "accept it deliberately."
                )
            raise RuntimeError(
                f"cannot bootstrap the {quote.tenor:g}Y pillar: the quoted "
                f"{quote.spread_bps:.1f}bp exceeds what a hazard of "
                f"{MAX_HAZARD_RATE:g} can produce. Check the units -- spreads "
                "are expected in basis points."
            )

        solved = float(
            brentq(mispricing, low, high, xtol=HAZARD_TOLERANCE, maxiter=200)
        )
        pillars.append(quote.tenor)
        hazards.append(solved)

    described = ", ".join(
        f"{quote.tenor:g}Y={quote.spread_bps:g}bp" for quote in ordered
    )
    return CreditCurve(
        pillar_times=np.array(pillars, dtype=float),
        hazard_rates=np.array(hazards, dtype=float),
        recovery_rate=recovery_rate,
        label=label or f"bootstrapped from [{described}] @ R={recovery_rate:.0%}",
    )


# ==========================================================================
# Implied volatility surface
# ==========================================================================
def black_vega(
    forward: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    volatility: np.ndarray,
) -> np.ndarray:
    r"""Black-76 vega (undiscounted), :math:`F\sqrt{T}\,\phi(d_1)`.

    Used only to weight the calibration: at-the-money quotes carry the most
    vega and are the most reliable, and vega weighting is the standard desk
    choice for exactly that reason.

    Args:
        forward: Forward price.
        strike: Strike.
        maturity: Time to expiry in years.
        volatility: Implied volatility.

    Returns:
        Vega, non-negative, zero where the inputs are degenerate.
    """
    fwd = np.asarray(forward, dtype=float)
    k = np.asarray(strike, dtype=float)
    t = np.asarray(maturity, dtype=float)
    vol = np.asarray(volatility, dtype=float)

    # All four operands must broadcast together in ONE call. Broadcasting
    # them pairwise -- (forward, strike) and then (maturity, volatility) --
    # leaves a scalar maturity 0-dimensional while the mask is 1-D, and the
    # subsequent boolean index raises.
    fwd_v, k_v, t_v, vol_v = np.broadcast_arrays(fwd, k, t, vol)
    result = np.zeros(fwd_v.shape, dtype=float)

    mask = (fwd_v > 0.0) & (k_v > 0.0) & (t_v > 0.0) & (vol_v > 0.0)
    if not np.any(mask):
        return result

    # Evaluate on the valid subset only. Masking after the fact would still
    # take log() and divide by zero on the invalid entries first, and numpy
    # warnings there are noise that hides real ones.
    std = vol_v[mask] * np.sqrt(t_v[mask])
    d1 = (np.log(fwd_v[mask] / k_v[mask]) + 0.5 * std**2) / std
    density = np.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi)
    result[mask] = fwd_v[mask] * np.sqrt(t_v[mask]) * density
    return result


@dataclass(frozen=True)
class VolSurfaceData:
    """A cleaned implied-volatility chain, ready for SSVI calibration.

    Attributes:
        log_moneyness: :math:`k = \\ln(K/F)`, measured against the **forward**,
            not the spot. Using spot shifts every point by
            :math:`(r - q)T` and shows up as a spurious calendar-dependent
            skew, which then contaminates the fitted SSVI :math:`\\rho`.
        maturity: Time to expiry in years.
        implied_volatility: Mid implied volatility.
        weights: Per-quote calibration weights.
        strike: Original strikes, retained for diagnostics.
        forward: Forward used for each quote.
        spot: Spot at observation.
        as_of: Observation timestamp.
        label: Provenance.
    """

    log_moneyness: np.ndarray
    maturity: np.ndarray
    implied_volatility: np.ndarray
    weights: np.ndarray
    strike: np.ndarray
    forward: np.ndarray
    spot: float
    as_of: Optional[_dt.date] = None
    label: str = "unspecified"

    def __post_init__(self) -> None:
        arrays = {
            "log_moneyness": self.log_moneyness,
            "maturity": self.maturity,
            "implied_volatility": self.implied_volatility,
            "weights": self.weights,
            "strike": self.strike,
            "forward": self.forward,
        }
        shapes = {name: np.asarray(a).shape for name, a in arrays.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"ragged surface arrays: {shapes}")

    def __len__(self) -> int:
        return int(np.asarray(self.log_moneyness).size)

    @property
    def total_variance(self) -> np.ndarray:
        r"""Total variance :math:`w = \sigma^2 T`, the SSVI fitting space."""
        return self.implied_volatility**2 * self.maturity

    def to_calibration_inputs(
        self, *, dtype: Optional["torch.dtype"] = None
    ) -> Tuple["Tensor", "Tensor", "Tensor", "Tensor"]:
        """Tensors in the exact order ``calibrate_surface`` expects.

        Returns:
            ``(log_moneyness, maturity, implied_volatility, weights)``, ready
            to pass straight to :func:`src.models.vol_surface.calibrate_surface`.
        """
        import torch

        resolved = torch.float64 if dtype is None else dtype
        return (
            torch.as_tensor(self.log_moneyness, dtype=resolved),
            torch.as_tensor(self.maturity, dtype=resolved),
            torch.as_tensor(self.implied_volatility, dtype=resolved),
            torch.as_tensor(self.weights, dtype=resolved),
        )

    def __repr__(self) -> str:
        if len(self) == 0:
            return f"VolSurfaceData(empty, label={self.label!r})"
        return (
            f"VolSurfaceData({len(self)} quotes, "
            f"T {self.maturity.min():.3f}-{self.maturity.max():.3f}y, "
            f"k {self.log_moneyness.min():+.3f}..{self.log_moneyness.max():+.3f}, "
            f"iv {self.implied_volatility.min():.1%}-"
            f"{self.implied_volatility.max():.1%}, label={self.label!r})"
        )


def clean_option_chain(
    chain: "pd.DataFrame",
    spot: float,
    *,
    discount_curve: Optional[YieldCurve] = None,
    dividend_yield: float = 0.0,
    max_abs_log_moneyness: float = 0.35,
    min_implied_vol: float = 0.01,
    max_implied_vol: float = 3.0,
    require_two_sided: bool = True,
    max_relative_spread: float = 0.5,
    require_activity: bool = True,
    weight_scheme: str = "vega",
    as_of: Optional[_dt.date] = None,
    label: str = "cleaned chain",
) -> VolSurfaceData:
    r"""Filter a raw option chain into a calibration-ready surface.

    Pure function: takes a DataFrame and returns arrays, with no I/O, so every
    filter below is unit-testable.

    Raw yfinance chains are not usable as-is. The filters, and why each earns
    its place:

    * **``impliedVolatility`` outside a sane band.** yfinance reports exactly
      ``0.0`` for contracts its solver failed on, and occasionally values above
      500%. Both would otherwise dominate a least-squares fit.
    * **One-sided or crossed markets.** A zero bid means no one is buying at
      any price; the "mid" is then fictional.
    * **Very wide relative spreads.** A quote whose bid-ask straddles the mid by
      more than ``max_relative_spread`` carries almost no information about the
      mid.
    * **No trading activity.** Zero volume *and* zero open interest means the
      strike is quoted but not traded.
    * **Deep wings.** Beyond roughly 35% log-moneyness the inverted vol is
      dominated by the discreteness of the option price tick, and SSVI's wing
      behaviour is asymptotic rather than a fit target.

    Moneyness is measured against the **forward**
    :math:`F = S e^{(r - q)T}`, not the spot. This matters: with spot moneyness
    every expiry's smile is displaced by :math:`(r-q)T`, which a calibrator
    absorbs as a spurious maturity-dependent skew.

    Args:
        chain: Raw chain. Requires ``strike``, ``impliedVolatility`` and
            ``maturity`` (in years); uses ``bid``, ``ask``, ``volume`` and
            ``openInterest`` when present.
        spot: Spot price of the underlying.
        discount_curve: Curve for the forward. A flat 0% curve is used if
            omitted, which makes the forward equal the spot.
        dividend_yield: Continuous dividend yield :math:`q`.
        max_abs_log_moneyness: Wing cutoff on :math:`|k|`.
        min_implied_vol: Lower vol bound.
        max_implied_vol: Upper vol bound.
        require_two_sided: Drop quotes without a positive two-sided market.
        max_relative_spread: Maximum ``(ask - bid) / mid``.
        require_activity: Drop quotes with no volume and no open interest.
        weight_scheme: ``"vega"``, ``"uniform"`` or ``"spread"`` (inverse
            relative spread).
        as_of: Observation date.
        label: Provenance.

    Returns:
        A :class:`VolSurfaceData`, possibly empty if every quote was filtered.

    Raises:
        ValueError: On missing required columns, a non-positive spot, or an
            unknown weight scheme.
    """
    import pandas as pd

    if spot <= 0.0:
        raise ValueError(f"spot must be positive, got {spot}")
    if weight_scheme not in {"vega", "uniform", "spread"}:
        raise ValueError(
            f"unknown weight_scheme {weight_scheme!r}; expected 'vega', "
            "'uniform' or 'spread'"
        )

    required = {"strike", "impliedVolatility", "maturity"}
    missing = required - set(chain.columns)
    if missing:
        raise ValueError(f"option chain is missing columns: {sorted(missing)}")

    frame = chain.copy()
    curve = discount_curve or YieldCurve.flat(0.0, label="zero")

    # ---- numeric coercion; yfinance sometimes hands back object dtype ---
    for column in ("strike", "impliedVolatility", "maturity"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("bid", "ask", "volume", "openInterest"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    keep = (
        frame["strike"].notna()
        & frame["impliedVolatility"].notna()
        & frame["maturity"].notna()
        & (frame["strike"] > 0.0)
        & (frame["maturity"] > 0.0)
        & (frame["impliedVolatility"] >= min_implied_vol)
        & (frame["impliedVolatility"] <= max_implied_vol)
    )

    if require_two_sided and {"bid", "ask"} <= set(frame.columns):
        bid = frame["bid"].fillna(0.0)
        ask = frame["ask"].fillna(0.0)
        keep &= (bid > 0.0) & (ask >= bid)
        mid = 0.5 * (bid + ask)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.where(mid > 0.0, (ask - bid) / mid, np.inf)
        keep &= relative <= max_relative_spread

    if require_activity and {"volume", "openInterest"} <= set(frame.columns):
        activity = frame["volume"].fillna(0.0) + frame["openInterest"].fillna(0.0)
        keep &= activity > 0.0

    frame = frame.loc[keep]

    if frame.empty:
        empty = np.zeros(0, dtype=float)
        return VolSurfaceData(
            log_moneyness=empty, maturity=empty, implied_volatility=empty,
            weights=empty, strike=empty, forward=empty, spot=float(spot),
            as_of=as_of, label=f"{label} (all quotes filtered)",
        )

    maturity = frame["maturity"].to_numpy(dtype=float)
    strike = frame["strike"].to_numpy(dtype=float)
    vol = frame["impliedVolatility"].to_numpy(dtype=float)

    rate = curve.zero_rate(maturity)
    forward = spot * np.exp((rate - dividend_yield) * maturity)
    log_moneyness = np.log(strike / forward)

    band = np.abs(log_moneyness) <= max_abs_log_moneyness
    maturity, strike, vol = maturity[band], strike[band], vol[band]
    forward, log_moneyness = forward[band], log_moneyness[band]

    if weight_scheme == "uniform":
        weights = np.ones_like(vol)
    elif weight_scheme == "vega":
        weights = black_vega(forward, strike, maturity, vol)
    else:  # "spread"
        if {"bid", "ask"} <= set(frame.columns):
            bid = frame["bid"].fillna(0.0).to_numpy(dtype=float)[band]
            ask = frame["ask"].fillna(0.0).to_numpy(dtype=float)[band]
            mid = 0.5 * (bid + ask)
            with np.errstate(divide="ignore", invalid="ignore"):
                relative = np.where(mid > 0.0, (ask - bid) / mid, np.inf)
            weights = 1.0 / np.maximum(relative, 1.0e-4)
        else:
            weights = np.ones_like(vol)

    total = float(np.sum(weights))
    if total > 0.0:
        weights = weights * (weights.size / total)  # mean weight 1
    else:
        weights = np.ones_like(vol)

    return VolSurfaceData(
        log_moneyness=log_moneyness,
        maturity=maturity,
        implied_volatility=vol,
        weights=weights,
        strike=strike,
        forward=forward,
        spot=float(spot),
        as_of=as_of,
        label=label,
    )


# ==========================================================================
# Network layer -- yfinance
# ==========================================================================
def _require(module: str, package: str):
    """Import an optional dependency with an actionable error message.

    Args:
        module: Module name to import.
        package: pip package that provides it.

    Returns:
        The imported module.

    Raises:
        ImportError: With the install command, if unavailable.
    """
    try:
        return __import__(module)
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            f"{module} is required for this function but is not installed. "
            f"Install it with: pip install {package}"
        ) from error


def fetch_spot(ticker: str) -> float:
    """Latest spot price for an equity or index.

    Prefers the last daily close over the quote-endpoint fields, which are
    frequently ``None`` outside trading hours and differ between yfinance
    versions.

    Args:
        ticker: Yahoo symbol (``"AAPL"``, ``"^GSPC"``, ``"^SPX"``).

    Returns:
        The spot price.

    Raises:
        ImportError: If yfinance is not installed.
        RuntimeError: If no usable price comes back.
    """
    yfinance = _require("yfinance", "yfinance")

    handle = yfinance.Ticker(ticker)
    history = handle.history(period="5d", auto_adjust=False)
    if history is not None and not history.empty and "Close" in history:
        closes = history["Close"].dropna()
        if not closes.empty:
            return float(closes.iloc[-1])

    # Fall back to the quote endpoint only if history is unavailable.
    for attribute in ("last_price", "regularMarketPrice", "previousClose"):
        try:
            info = handle.fast_info if attribute == "last_price" else handle.info
            value = info.get(attribute) if hasattr(info, "get") else None
        except Exception:  # noqa: BLE001 - yfinance raises many shapes here
            value = None
        if value:
            return float(value)

    raise RuntimeError(
        f"no usable spot price for {ticker!r}. Check the symbol (indices need a "
        "caret, e.g. '^GSPC') and that the market data endpoint is reachable."
    )


def fetch_implied_vol_surface(
    ticker: str,
    *,
    max_expiries: int = 8,
    min_maturity_days: float = 7.0,
    max_maturity_days: float = 730.0,
    discount_curve: Optional[YieldCurve] = None,
    dividend_yield: float = 0.0,
    spot: Optional[float] = None,
    **clean_kwargs,
) -> VolSurfaceData:
    """Download and clean an implied-volatility surface.

    Takes only out-of-the-money contracts -- calls above the forward, puts
    below -- because OTM options are the liquid side at each strike and their
    inverted vols are the reliable ones. Including both wings at every strike
    would double-count the same information with the worse half of each pair.

    Args:
        ticker: Yahoo symbol.
        max_expiries: Cap on the number of expiries pulled; each is a separate
            HTTP request.
        min_maturity_days: Drop expiries nearer than this. Sub-week options are
            dominated by pin risk and discrete tick effects.
        max_maturity_days: Drop expiries beyond this, where quotes thin out.
        discount_curve: Curve for the forward. Defaults to flat 0%.
        dividend_yield: Continuous dividend yield.
        spot: Spot override; fetched if omitted.
        **clean_kwargs: Passed through to :func:`clean_option_chain`.

    Returns:
        A :class:`VolSurfaceData` pooled across expiries.

    Raises:
        ImportError: If yfinance or pandas is not installed.
        RuntimeError: If the ticker exposes no option expiries.
    """
    yfinance = _require("yfinance", "yfinance")
    pandas = _require("pandas", "pandas")

    handle = yfinance.Ticker(ticker)
    resolved_spot = fetch_spot(ticker) if spot is None else float(spot)

    expiries = list(getattr(handle, "options", ()) or ())
    if not expiries:
        raise RuntimeError(
            f"{ticker!r} exposes no option expiries. Indices often need the "
            "caret form (e.g. '^SPX'), and not every symbol has listed options."
        )

    today = _dt.date.today()
    frames: List["pd.DataFrame"] = []
    for expiry in expiries:
        if len(frames) >= max_expiries:
            break
        try:
            expiry_date = _dt.date.fromisoformat(expiry)
        except ValueError:  # pragma: no cover - defensive
            continue
        days = (expiry_date - today).days
        if not min_maturity_days <= days <= max_maturity_days:
            continue

        try:
            chain = handle.option_chain(expiry)
        except Exception:  # noqa: BLE001 - a single bad expiry must not abort
            continue

        maturity = days / 365.0
        rate = float(np.asarray(
            (discount_curve or YieldCurve.flat(0.0)).zero_rate(maturity)
        ))
        forward = resolved_spot * math.exp((rate - dividend_yield) * maturity)

        # OTM only: calls above the forward, puts below.
        for side, frame in (("call", chain.calls), ("put", chain.puts)):
            if frame is None or frame.empty:
                continue
            selected = frame[
                frame["strike"] >= forward if side == "call"
                else frame["strike"] < forward
            ].copy()
            if selected.empty:
                continue
            selected["maturity"] = maturity
            selected["side"] = side
            frames.append(selected)

    if not frames:
        empty = np.zeros(0, dtype=float)
        return VolSurfaceData(
            log_moneyness=empty, maturity=empty, implied_volatility=empty,
            weights=empty, strike=empty, forward=empty, spot=resolved_spot,
            as_of=today, label=f"{ticker} (no expiries in the maturity window)",
        )

    pooled = pandas.concat(frames, ignore_index=True)
    return clean_option_chain(
        pooled,
        resolved_spot,
        discount_curve=discount_curve,
        dividend_yield=dividend_yield,
        as_of=today,
        label=f"{ticker} implied vol surface",
        **clean_kwargs,
    )


# ==========================================================================
# Network layer -- FRED
# ==========================================================================
def fetch_fred_series(
    series: Iterable[str],
    *,
    start: Optional[_dt.date] = None,
    end: Optional[_dt.date] = None,
    lookback_days: int = 30,
) -> "pd.DataFrame":
    """Download FRED series via pandas_datareader.

    Args:
        series: FRED series ids.
        start: Start date; defaults to ``lookback_days`` before ``end``.
        end: End date; defaults to today.
        lookback_days: Window used when ``start`` is omitted. Needs to exceed
            the longest expected run of holidays, since daily rate series are
            ``NaN`` on non-business days and the most recent valid observation
            is what gets used.

    Returns:
        A DataFrame indexed by date, one column per series.

    Raises:
        ImportError: If pandas_datareader is not installed.
        RuntimeError: If FRED returns nothing.
    """
    _require("pandas_datareader", "pandas-datareader")
    from pandas_datareader import data as pdr

    identifiers = list(series)
    if not identifiers:
        raise ValueError("no FRED series requested")

    resolved_end = end or _dt.date.today()
    resolved_start = start or (resolved_end - _dt.timedelta(days=lookback_days))

    try:
        frame = pdr.DataReader(identifiers, "fred", resolved_start, resolved_end)
    except Exception as error:  # noqa: BLE001 - network/HTTP/parse all land here
        raise RuntimeError(
            f"FRED request failed for {identifiers}: {error}"
        ) from error

    if frame is None or frame.empty:
        raise RuntimeError(
            f"FRED returned no observations for {identifiers} between "
            f"{resolved_start} and {resolved_end}. Widen lookback_days if the "
            "window fell entirely on holidays."
        )
    return frame


def _latest_observations(frame: "pd.DataFrame") -> Dict[str, float]:
    """Most recent non-``NaN`` value for each column.

    Series are published on different schedules, so the last *row* is not the
    last observation for every column; taking a per-column ``last_valid_index``
    avoids silently dropping a whole tenor because it lagged by a day.

    Args:
        frame: A FRED DataFrame.

    Returns:
        Column name -> latest finite value, omitting entirely-empty columns.
    """
    latest: Dict[str, float] = {}
    for column in frame.columns:
        stamp = frame[column].last_valid_index()
        if stamp is None:
            continue
        value = float(frame.loc[stamp, column])
        if math.isfinite(value):
            latest[column] = value
    return latest


def fetch_sofr_curve(
    *,
    start: Optional[_dt.date] = None,
    end: Optional[_dt.date] = None,
    lookback_days: int = 30,
) -> YieldCurve:
    """Build the SOFR short end from FRED.

    Only reaches about 0.5Y: FRED publishes overnight SOFR and its 30/90/180-day
    backward averages, and **no term SOFR swap curve**. For anything requiring
    the long end -- a 5Y or 10Y CDS, for instance -- use
    :func:`fetch_discount_curve`, which splices this onto Treasury yields and
    labels the result accordingly.

    Args:
        start: Start date for the FRED query.
        end: End date for the FRED query.
        lookback_days: Window when ``start`` is omitted.

    Returns:
        A :class:`YieldCurve` covering the SOFR tenors, converted from simple
        ACT/360 money-market quotes to continuous compounding.

    Raises:
        RuntimeError: If no SOFR observation is available.
    """
    frame = fetch_fred_series(
        SOFR_SERIES.values(), start=start, end=end, lookback_days=lookback_days
    )
    latest = _latest_observations(frame)

    tenors: List[float] = []
    rates: List[float] = []
    for tenor, identifier in sorted(SOFR_SERIES.items()):
        if identifier not in latest:
            continue
        simple = latest[identifier] / 100.0  # FRED quotes percent
        tenors.append(tenor)
        rates.append(float(simple_to_continuous(simple, tenor)))

    if not tenors:
        raise RuntimeError(
            "no SOFR observations available from FRED in the requested window"
        )

    return YieldCurve(
        tenors=np.array(tenors, dtype=float),
        zero_rates=np.array(rates, dtype=float),
        as_of=end or _dt.date.today(),
        label=(
            "SOFR short end from FRED (overnight + 30/90/180d averages, "
            "simple ACT/360 converted to continuous). NOT a term SOFR swap "
            "curve -- FRED does not publish one."
        ),
    )


def fetch_treasury_curve(
    *,
    start: Optional[_dt.date] = None,
    end: Optional[_dt.date] = None,
    lookback_days: int = 30,
) -> YieldCurve:
    """Build a zero curve from Treasury constant-maturity yields on FRED.

    CMT yields are *par coupon* yields, not zero rates; they are converted from
    semiannual to continuous compounding but not bootstrapped through the par
    curve. For a normally-shaped curve that approximation is worth a few basis
    points at 10Y, which is second order for CVA -- but it is an approximation,
    and the returned label says so.

    Args:
        start: Start date for the FRED query.
        end: End date for the FRED query.
        lookback_days: Window when ``start`` is omitted.

    Returns:
        A :class:`YieldCurve` out to 30Y.

    Raises:
        RuntimeError: If no Treasury observation is available.
    """
    frame = fetch_fred_series(
        TREASURY_SERIES.values(), start=start, end=end, lookback_days=lookback_days
    )
    latest = _latest_observations(frame)

    tenors: List[float] = []
    rates: List[float] = []
    for tenor, identifier in sorted(TREASURY_SERIES.items()):
        if identifier not in latest:
            continue
        tenors.append(tenor)
        rates.append(float(semiannual_to_continuous(latest[identifier] / 100.0)))

    if not tenors:
        raise RuntimeError(
            "no Treasury observations available from FRED in the requested window"
        )

    return YieldCurve(
        tenors=np.array(tenors, dtype=float),
        zero_rates=np.array(rates, dtype=float),
        as_of=end or _dt.date.today(),
        label=(
            "Treasury CMT from FRED, semiannual par yields converted to "
            "continuous. Par yields treated as zero rates (no par bootstrap)."
        ),
    )


def fetch_discount_curve(
    *,
    source: str = "sofr_treasury",
    start: Optional[_dt.date] = None,
    end: Optional[_dt.date] = None,
    lookback_days: int = 30,
) -> YieldCurve:
    """Assemble the discount curve used for CVA and CDS valuation.

    Args:
        source: ``"sofr_treasury"`` splices the SOFR short end (to 0.5Y) onto
            Treasury pillars beyond it -- the default, because discounting a
            5Y or 10Y CDS needs a long end that the SOFR series do not reach.
            ``"sofr"`` returns the SOFR short end alone, whose flat
            extrapolation past 0.5Y is a poor long-end assumption.
            ``"treasury"`` returns Treasury alone.
        start: Start date for the FRED query.
        end: End date for the FRED query.
        lookback_days: Window when ``start`` is omitted.

    Returns:
        A :class:`YieldCurve` whose ``label`` records the approximations made.

    Raises:
        ValueError: On an unknown source.
    """
    if source == "sofr":
        return fetch_sofr_curve(start=start, end=end, lookback_days=lookback_days)
    if source == "treasury":
        return fetch_treasury_curve(
            start=start, end=end, lookback_days=lookback_days
        )
    if source != "sofr_treasury":
        raise ValueError(
            f"unknown source {source!r}; expected 'sofr_treasury', 'sofr' or "
            "'treasury'"
        )

    short = fetch_sofr_curve(start=start, end=end, lookback_days=lookback_days)
    long = fetch_treasury_curve(start=start, end=end, lookback_days=lookback_days)

    splice = float(short.tenors[-1])
    keep = long.tenors > splice
    tenors = np.concatenate([short.tenors, long.tenors[keep]])
    rates = np.concatenate([short.zero_rates, long.zero_rates[keep]])

    return YieldCurve(
        tenors=tenors,
        zero_rates=rates,
        as_of=short.as_of,
        label=(
            f"SOFR to {splice:.3g}y spliced onto Treasury CMT beyond. "
            "Approximation: Treasury is not SOFR (the swap-Treasury basis is "
            "tens of bp and time-varying), and CMT par yields are treated as "
            "zero rates. FRED publishes no term SOFR swap curve; a production "
            "desk sources it from ICE, Bloomberg or Refinitiv."
        ),
    )
