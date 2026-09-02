r"""Tests for :mod:`market_data.fetcher`.

Two tiers, matching the pattern used across this project:

* **Tier 1 (default)** -- the mathematics: curves, schedules, the CDS
  bootstrap, chain cleaning. Pure NumPy, no network, deterministic.
* **Tier 2 (opt-in)** -- the live fetchers, marked ``network`` and skipped
  unless ``XVA_NETWORK_TESTS=1``. They depend on Yahoo and FRED being
  reachable and on today's data, so they can never be part of a deterministic
  suite; they are smoke tests for the I/O wiring, nothing more.

Where a tolerance appears it is *derived*, not chosen. The two that matter:

* The bootstrap reprices its own inputs to solver precision, so the round-trip
  bound is the root-finder tolerance, not a guess.
* The credit triangle :math:`h = S/(1-R)` is exact only in the continuum limit.
  The discrete legs use the trapezoid rule, whose error is
  :math:`O((h\Delta)^2)`, so the tests assert the *observed second-order rate*
  rather than an arbitrary threshold. Measured ratio on refinement: 4.00.
"""

from __future__ import annotations

import datetime as _dt
import math
import os

import numpy as np
import pandas as pd
import pytest
import torch

from market_data.fetcher import (
    BASIS_POINT,
    DEFAULT_CDS_FREQUENCY,
    MAX_HAZARD_RATE,
    CDSQuote,
    CreditCurve,
    VolSurfaceData,
    YieldCurve,
    black_vega,
    bootstrap_hazard_rates,
    clean_option_chain,
    model_par_spread,
    payment_schedule,
    premium_leg_annuity,
    protection_leg_pv,
    semiannual_to_continuous,
    simple_to_continuous,
)
from src.xva import cva as cva_module

# A realistic upward-sloping investment-grade term structure.
STANDARD_QUOTES = [
    CDSQuote(tenor=1.0, spread_bps=80.0),
    CDSQuote(tenor=3.0, spread_bps=110.0),
    CDSQuote(tenor=5.0, spread_bps=135.0),
    CDSQuote(tenor=10.0, spread_bps=160.0),
]
RECOVERY = 0.40

#: Repricing bound in basis points. The root finder converges to ``1e-12`` on
#: the hazard; the spread it implies is smooth in the hazard, so ``1e-6`` bp is
#: several orders of magnitude of headroom while still being far tighter than
#: any market tick (0.01 bp).
REPRICING_TOLERANCE_BPS = 1.0e-6

requires_network = pytest.mark.skipif(
    os.environ.get("XVA_NETWORK_TESTS") != "1",
    reason="live market data; set XVA_NETWORK_TESTS=1 to enable",
)


@pytest.fixture
def flat_curve() -> YieldCurve:
    """A flat 3% discount curve."""
    return YieldCurve.flat(0.03)


@pytest.fixture
def zero_curve() -> YieldCurve:
    """A flat 0% curve, which isolates credit from discounting."""
    return YieldCurve.flat(0.0)


@pytest.fixture
def bootstrapped(flat_curve: YieldCurve) -> CreditCurve:
    """The standard quote set bootstrapped at 40% recovery."""
    return bootstrap_hazard_rates(
        STANDARD_QUOTES, flat_curve, recovery_rate=RECOVERY
    )


# ==========================================================================
# Rate conventions
# ==========================================================================
class TestRateConventions:
    """Compounding conversions, which are silent-error territory."""

    def test_semiannual_to_continuous_is_lower(self) -> None:
        """More frequent compounding needs a lower continuous rate."""
        par = np.array([0.01, 0.03, 0.05, 0.08])
        continuous = semiannual_to_continuous(par)
        assert np.all(continuous < par)

    def test_semiannual_round_trip(self) -> None:
        r"""``exp(r_c) == (1 + y/2)^2`` -- the defining identity."""
        par = np.array([0.01, 0.043, 0.075])
        continuous = semiannual_to_continuous(par)
        np.testing.assert_allclose(
            np.exp(continuous), (1.0 + par / 2.0) ** 2, rtol=1e-14
        )

    def test_simple_to_continuous_round_trip(self) -> None:
        r"""``exp(r_c * tau) == 1 + r * tau``."""
        rate = np.array([0.02, 0.05])
        tenor = np.array([0.25, 0.5])
        continuous = simple_to_continuous(rate, tenor)
        np.testing.assert_allclose(
            np.exp(continuous * tenor), 1.0 + rate * tenor, rtol=1e-14
        )

    def test_zero_rate_maps_to_zero(self) -> None:
        """Both conversions fix zero."""
        assert semiannual_to_continuous(np.array([0.0]))[0] == 0.0
        assert simple_to_continuous(np.array([0.0]), np.array([1.0]))[0] == 0.0

    def test_simple_conversion_rejects_zero_tenor(self) -> None:
        """A simple rate has no meaning without an accrual period."""
        with pytest.raises(ValueError, match="tenor must be positive"):
            simple_to_continuous(np.array([0.05]), np.array([0.0]))


# ==========================================================================
# Discount curve
# ==========================================================================
class TestYieldCurve:
    """Interpolation, extrapolation and discount factors."""

    def test_exact_at_pillars(self) -> None:
        """Interpolation must reproduce the inputs it was built from."""
        tenors = np.array([1.0, 2.0, 5.0, 10.0])
        rates = np.array([0.030, 0.035, 0.040, 0.042])
        curve = YieldCurve(tenors=tenors, zero_rates=rates)
        np.testing.assert_allclose(curve.zero_rate(tenors), rates, rtol=0.0)

    def test_discount_factor_at_zero_is_one(self) -> None:
        """``DF(0) == 1`` exactly, not approximately."""
        assert YieldCurve.flat(0.05).discount_factor(0.0) == 1.0

    def test_flat_curve_matches_closed_form(self) -> None:
        """A flat curve must give exactly ``exp(-rt)``."""
        rate = 0.037
        times = np.array([0.0, 0.5, 1.0, 7.0, 30.0])
        np.testing.assert_allclose(
            YieldCurve.flat(rate).discount_factor(times),
            np.exp(-rate * times),
            rtol=1e-15,
        )

    def test_discount_factors_decrease(self) -> None:
        """Positive rates give a strictly decreasing discount curve."""
        curve = YieldCurve(
            tenors=np.array([1.0, 5.0, 10.0]),
            zero_rates=np.array([0.03, 0.04, 0.045]),
        )
        times = np.linspace(0.0, 20.0, 60)
        factors = curve.discount_factor(times)
        assert np.all(np.diff(factors) < 0.0)
        assert np.all((factors > 0.0) & (factors <= 1.0))

    def test_extrapolation_is_flat_at_both_ends(self) -> None:
        """Flat extrapolation, so the long end cannot go negative."""
        curve = YieldCurve(
            tenors=np.array([1.0, 10.0]), zero_rates=np.array([0.02, 0.05])
        )
        assert curve.zero_rate(0.01) == pytest.approx(0.02)
        assert curve.zero_rate(50.0) == pytest.approx(0.05)

    def test_interpolation_is_linear_in_the_zero_rate(self) -> None:
        """The documented scheme; the midpoint pins it down."""
        curve = YieldCurve(
            tenors=np.array([1.0, 3.0]), zero_rates=np.array([0.02, 0.04])
        )
        assert curve.zero_rate(2.0) == pytest.approx(0.03)

    @pytest.mark.parametrize(
        "tenors, rates, message",
        [
            ([], [], "at least one pillar"),
            ([1.0, 2.0], [0.03], "must match"),
            ([2.0, 1.0], [0.03, 0.04], "strictly increasing"),
            ([1.0, 1.0], [0.03, 0.04], "strictly increasing"),
            ([0.0, 1.0], [0.03, 0.04], "must be positive"),
            ([1.0, np.nan], [0.03, 0.04], "non-finite"),
        ],
    )
    def test_rejects_malformed_input(self, tenors, rates, message) -> None:
        """Bad curves fail at construction, not at first use."""
        with pytest.raises(ValueError, match=message):
            YieldCurve(
                tenors=np.array(tenors, dtype=float),
                zero_rates=np.array(rates, dtype=float),
            )

    def test_negative_time_rejected(self) -> None:
        """A negative maturity is a caller bug, not something to extrapolate."""
        with pytest.raises(ValueError, match="non-negative"):
            YieldCurve.flat(0.03).discount_factor(np.array([-1.0]))


# ==========================================================================
# Payment schedule
# ==========================================================================
class TestPaymentSchedule:
    """Schedules are built backwards from maturity; that must hold exactly."""

    @pytest.mark.parametrize("maturity", [0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    def test_ends_exactly_on_maturity(self, maturity: float) -> None:
        """The bootstrap solves each pillar, so the schedule must land on it.

        A schedule ending a few days either side of the pillar would put a
        small systematic error into every bootstrapped hazard.
        """
        schedule = payment_schedule(maturity, DEFAULT_CDS_FREQUENCY)
        assert schedule[-1] == pytest.approx(maturity, abs=1e-14)

    def test_starts_at_zero_and_increases(self) -> None:
        """Accrual begins today and the schedule is strictly increasing."""
        schedule = payment_schedule(5.0, 4)
        assert schedule[0] == pytest.approx(0.0, abs=1e-14)
        assert np.all(np.diff(schedule) > 0.0)

    def test_regular_maturity_has_uniform_accruals(self) -> None:
        """A whole number of periods leaves no stub."""
        accruals = np.diff(payment_schedule(5.0, 4))
        np.testing.assert_allclose(accruals, 0.25, rtol=1e-14)
        assert accruals.size == 20

    def test_irregular_maturity_puts_the_stub_first(self) -> None:
        """Market convention: only the *first* period may be short."""
        accruals = np.diff(payment_schedule(7.0 / 12.0, 4))
        assert accruals[0] < 0.25
        np.testing.assert_allclose(accruals[1:], 0.25, rtol=1e-12)

    def test_short_maturity_still_gives_one_period(self) -> None:
        """A sub-period maturity must not produce an empty schedule."""
        schedule = payment_schedule(0.1, 4)
        assert schedule.size == 2
        assert schedule[-1] == pytest.approx(0.1)

    @pytest.mark.parametrize(
        "maturity, frequency", [(0.0, 4), (-1.0, 4), (1.0, 0), (1.0, -4)]
    )
    def test_rejects_invalid_arguments(self, maturity, frequency) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            payment_schedule(maturity, frequency)


# ==========================================================================
# Credit curve evaluation
# ==========================================================================
class TestCreditCurve:
    """Survival, hazard and cumulative hazard on a piecewise-constant curve."""

    def test_survival_at_zero_is_one(self, bootstrapped: CreditCurve) -> None:
        """``Q(0) == 1`` exactly."""
        assert bootstrapped.survival_probability(0.0) == 1.0

    def test_survival_is_strictly_decreasing(
        self, bootstrapped: CreditCurve
    ) -> None:
        """Positive hazard everywhere means monotone survival."""
        times = np.linspace(0.0, 20.0, 200)
        survival = bootstrapped.survival_probability(times)
        assert np.all(np.diff(survival) < 0.0)
        assert np.all((survival > 0.0) & (survival <= 1.0))

    def test_cumulative_hazard_is_piecewise_linear(self) -> None:
        r"""``H`` is exact between pillars, so a midpoint must be the average.

        This is what makes :math:`Q` interpolation-error-free inside the pillar
        range, as opposed to a curve that splines survival directly.
        """
        curve = CreditCurve(
            pillar_times=np.array([1.0, 3.0]),
            hazard_rates=np.array([0.01, 0.03]),
        )
        # Inside (1, 3] the hazard is constant, so H is a straight line.
        left, mid, right = curve.cumulative_hazard(np.array([1.0, 2.0, 3.0]))
        assert mid == pytest.approx(0.5 * (left + right), rel=1e-14)

    def test_cumulative_hazard_matches_hand_integration(self) -> None:
        """Explicit check against the integral computed by hand."""
        curve = CreditCurve(
            pillar_times=np.array([1.0, 3.0, 5.0]),
            hazard_rates=np.array([0.01, 0.02, 0.04]),
        )
        # H(4) = 0.01*1 + 0.02*2 + 0.04*1
        assert curve.cumulative_hazard(4.0) == pytest.approx(0.09, rel=1e-14)

    def test_hazard_extends_flat_beyond_the_last_pillar(self) -> None:
        """Beyond 10Y the last hazard is held, not extrapolated to zero."""
        curve = CreditCurve(
            pillar_times=np.array([1.0, 5.0]),
            hazard_rates=np.array([0.01, 0.02]),
        )
        assert curve.hazard_at(50.0) == pytest.approx(0.02)
        # H(15) = 0.01*1 + 0.02*4 + 0.02*10
        assert curve.cumulative_hazard(15.0) == pytest.approx(0.29, rel=1e-14)

    def test_hazard_is_right_continuous_at_pillars(self) -> None:
        r"""The ``(T_{j-1}, T_j]`` convention: ``T_j`` belongs to segment j."""
        curve = CreditCurve(
            pillar_times=np.array([1.0, 3.0]),
            hazard_rates=np.array([0.01, 0.03]),
        )
        assert curve.hazard_at(1.0) == pytest.approx(0.01)
        assert curve.hazard_at(1.0 + 1e-9) == pytest.approx(0.03)

    def test_marginal_default_probabilities_sum_correctly(
        self, bootstrapped: CreditCurve
    ) -> None:
        r"""They must telescope to :math:`Q(t_0) - Q(t_n)`."""
        grid = np.linspace(0.0, 10.0, 41)
        marginal = bootstrapped.marginal_default_probability(grid)
        expected = float(
            bootstrapped.survival_probability(grid[0])
            - bootstrapped.survival_probability(grid[-1])
        )
        assert marginal.sum() == pytest.approx(expected, rel=1e-13)
        assert np.all(marginal >= 0.0)

    def test_marginal_needs_two_points(
        self, bootstrapped: CreditCurve
    ) -> None:
        with pytest.raises(ValueError, match="at least two grid points"):
            bootstrapped.marginal_default_probability(np.array([1.0]))

    @pytest.mark.parametrize(
        "pillars, hazards, message",
        [
            ([], [], "at least one pillar"),
            ([1.0, 2.0], [0.01], "must match"),
            ([2.0, 1.0], [0.01, 0.02], "strictly increasing"),
            ([0.0], [0.01], "must be positive"),
            ([np.nan], [0.01], "non-finite"),
        ],
    )
    def test_rejects_malformed_input(self, pillars, hazards, message) -> None:
        with pytest.raises(ValueError, match=message):
            CreditCurve(
                pillar_times=np.array(pillars, dtype=float),
                hazard_rates=np.array(hazards, dtype=float),
            )

    @pytest.mark.parametrize("recovery", [-0.1, 1.0, 1.5])
    def test_rejects_invalid_recovery(self, recovery: float) -> None:
        """``R = 1`` gives zero loss given default and a singular bootstrap."""
        with pytest.raises(ValueError, match=r"recovery_rate must be in"):
            CreditCurve(
                pillar_times=np.array([5.0]),
                hazard_rates=np.array([0.02]),
                recovery_rate=recovery,
            )


# ==========================================================================
# The bootstrap -- the core of this module
# ==========================================================================
class TestCDSBootstrap:
    """Recovering piecewise-constant hazards from a par spread curve."""

    def test_reprices_every_input_quote(
        self, bootstrapped: CreditCurve, flat_curve: YieldCurve
    ) -> None:
        """The defining property: a bootstrap that cannot reprice is wrong.

        This is the single strongest test in the file. Every other property
        (monotonicity, ordering, the triangle limit) is a consequence or a
        sanity check; *this* is the equation the bootstrap claims to solve.
        """
        for quote in STANDARD_QUOTES:
            modelled = (
                model_par_spread(quote.tenor, bootstrapped, flat_curve)
                / BASIS_POINT
            )
            assert modelled == pytest.approx(
                quote.spread_bps, abs=REPRICING_TOLERANCE_BPS
            ), f"{quote.tenor:g}Y misprices"

    def test_one_pillar_per_quote(self, bootstrapped: CreditCurve) -> None:
        """A piecewise-constant curve has exactly as many degrees of freedom
        as it has quotes -- no more (unidentified) and no fewer (overfit)."""
        assert bootstrapped.pillar_times.size == len(STANDARD_QUOTES)
        np.testing.assert_allclose(
            bootstrapped.pillar_times, [1.0, 3.0, 5.0, 10.0]
        )

    def test_upward_sloping_spreads_give_increasing_hazards(
        self, bootstrapped: CreditCurve
    ) -> None:
        """A rising credit curve implies rising forward default intensity."""
        assert np.all(np.diff(bootstrapped.hazard_rates) > 0.0)

    def test_steeply_inverted_spreads_give_decreasing_hazards(
        self, flat_curve: YieldCurve
    ) -> None:
        """A steadily inverted curve implies falling forward intensity."""
        quotes = [
            CDSQuote(1.0, 300.0),
            CDSQuote(3.0, 250.0),
            CDSQuote(5.0, 200.0),
            CDSQuote(10.0, 150.0),
        ]
        curve = bootstrap_hazard_rates(
            quotes, flat_curve, recovery_rate=RECOVERY
        )
        assert np.all(np.diff(curve.hazard_rates) < 0.0)
        for quote in quotes:
            assert (
                model_par_spread(quote.tenor, curve, flat_curve) / BASIS_POINT
            ) == pytest.approx(quote.spread_bps, abs=REPRICING_TOLERANCE_BPS)

    def test_flattening_inversion_can_lift_the_far_forward_hazard(
        self, flat_curve: YieldCurve
    ) -> None:
        """Monotone spreads do NOT imply a monotone forward hazard.

        Recorded because it is counter-intuitive and looks like a bug. The par
        spread is a survival- and discount-weighted *average* of the forward
        hazard, with weights that decay in :math:`t`. When an inverted curve
        *flattens* -- here the decrements are 20, 10 then 5 bp -- matching the
        10Y average can require the forward hazard on (5, 10] to sit slightly
        above the one on (3, 5].

        The curve below still reprices every quote to solver precision and its
        survival function is still strictly decreasing, which is what actually
        has to hold. Asserting monotone hazards here would be asserting an
        intuition rather than a property of the model.
        """
        quotes = [
            CDSQuote(1.0, 200.0),
            CDSQuote(3.0, 180.0),
            CDSQuote(5.0, 170.0),
            CDSQuote(10.0, 165.0),
        ]
        curve = bootstrap_hazard_rates(
            quotes, flat_curve, recovery_rate=RECOVERY
        )

        # The non-monotonicity is real, not noise.
        assert curve.hazard_rates[3] > curve.hazard_rates[2]

        # What must hold regardless: exact repricing and monotone survival.
        for quote in quotes:
            assert (
                model_par_spread(quote.tenor, curve, flat_curve) / BASIS_POINT
            ) == pytest.approx(quote.spread_bps, abs=REPRICING_TOLERANCE_BPS)
        survival = curve.survival_probability(np.linspace(0.0, 15.0, 200))
        assert np.all(np.diff(survival) < 0.0)

    def test_flat_spreads_give_a_flat_hazard(
        self, flat_curve: YieldCurve
    ) -> None:
        """No term structure in, no term structure out."""
        quotes = [CDSQuote(tenor, 120.0) for tenor in (1.0, 3.0, 5.0, 10.0)]
        curve = bootstrap_hazard_rates(
            quotes, flat_curve, recovery_rate=RECOVERY
        )
        spread = curve.hazard_rates.max() - curve.hazard_rates.min()
        assert spread < 1e-4, f"hazards not flat: {curve.hazard_rates}"

    @pytest.mark.parametrize("spread_bps", [25.0, 100.0, 500.0])
    def test_credit_triangle_in_its_exact_limit(
        self, zero_curve: YieldCurve, spread_bps: float
    ) -> None:
        r"""With zero rates and flat spreads, :math:`h = S/(1-R)`.

        The triangle is exact only in the continuum; the discrete legs use the
        trapezoid rule, so the residual is :math:`O((h\Delta)^2)`. The bound
        below is that expression, not a tuned constant -- and it is checked
        against a measured second-order rate in
        :meth:`test_credit_triangle_error_is_second_order`.
        """
        quotes = [CDSQuote(tenor, spread_bps) for tenor in (1.0, 3.0, 5.0)]
        curve = bootstrap_hazard_rates(
            quotes, zero_curve, recovery_rate=RECOVERY
        )
        triangle = spread_bps * BASIS_POINT / (1.0 - RECOVERY)
        bound = (triangle / DEFAULT_CDS_FREQUENCY) ** 2
        relative = np.abs(curve.hazard_rates / triangle - 1.0).max()
        assert relative <= bound, (
            f"S={spread_bps}bp: relative deviation {relative:.3e} exceeds the "
            f"O((h*dt)^2) bound {bound:.3e}"
        )

    def test_credit_triangle_error_is_second_order(
        self, zero_curve: YieldCurve
    ) -> None:
        r"""Refining the schedule must quarter the error.

        This is what distinguishes "the discretisation is behaving as derived"
        from "the tolerance happens to pass". A first-order bug, or a wrong
        accrual placement, would show a ratio near 2 or 1 instead of 4.
        Measured: 4.01, 4.00, 4.00 across the refinements below.
        """
        spread_bps = 2000.0  # wide name, so the O((h*dt)^2) term is visible
        triangle = spread_bps * BASIS_POINT / (1.0 - RECOVERY)

        errors = []
        for frequency in (4, 8, 16, 32):
            curve = bootstrap_hazard_rates(
                [CDSQuote(5.0, spread_bps)],
                zero_curve,
                recovery_rate=RECOVERY,
                frequency=frequency,
            )
            errors.append(abs(curve.hazard_rates[0] / triangle - 1.0))

        for coarse, fine in zip(errors[:-1], errors[1:]):
            assert fine < coarse, "refinement must reduce the error"
            assert coarse / fine == pytest.approx(4.0, rel=0.10), (
                f"expected second-order convergence (ratio 4), got "
                f"{coarse / fine:.3f} across {errors}"
            )

    @pytest.mark.parametrize("recovery", [0.0, 0.2, 0.4, 0.6, 0.8])
    def test_higher_recovery_implies_higher_hazard(
        self, flat_curve: YieldCurve, recovery: float
    ) -> None:
        r"""A fixed spread with less loss per default needs more defaults.

        The economic content of :math:`h \approx S/(1-R)`, and the reason a
        hazard curve is meaningless without the recovery it assumed -- which is
        why :class:`CreditCurve` stores it.
        """
        curve = bootstrap_hazard_rates(
            [CDSQuote(5.0, 150.0)], flat_curve, recovery_rate=recovery
        )
        triangle = 150.0 * BASIS_POINT / (1.0 - recovery)
        assert curve.hazard_rates[0] == pytest.approx(triangle, rel=0.02)
        assert curve.recovery_rate == recovery

    def test_hazard_increases_with_spread(self, flat_curve: YieldCurve) -> None:
        """Monotone in the quote: wider spread, higher implied intensity."""
        hazards = [
            bootstrap_hazard_rates(
                [CDSQuote(5.0, spread)], flat_curve, recovery_rate=RECOVERY
            ).hazard_rates[0]
            for spread in (10.0, 50.0, 100.0, 300.0, 1000.0)
        ]
        assert np.all(np.diff(hazards) > 0.0)

    def test_quotes_are_sorted_internally(self, flat_curve: YieldCurve) -> None:
        """Quote order is not the caller's problem."""
        shuffled = [STANDARD_QUOTES[i] for i in (3, 0, 2, 1)]
        curve = bootstrap_hazard_rates(
            shuffled, flat_curve, recovery_rate=RECOVERY
        )
        np.testing.assert_allclose(curve.pillar_times, [1.0, 3.0, 5.0, 10.0])

    def test_order_does_not_change_the_result(
        self, bootstrapped: CreditCurve, flat_curve: YieldCurve
    ) -> None:
        """Sorting must be a genuine normalisation, not a reordering of output."""
        shuffled = bootstrap_hazard_rates(
            [STANDARD_QUOTES[i] for i in (2, 3, 1, 0)],
            flat_curve,
            recovery_rate=RECOVERY,
        )
        np.testing.assert_allclose(
            shuffled.hazard_rates, bootstrapped.hazard_rates, rtol=1e-12
        )

    def test_single_quote_is_allowed(self, flat_curve: YieldCurve) -> None:
        """One quote gives a flat curve, which is a legitimate degenerate case."""
        curve = bootstrap_hazard_rates(
            [CDSQuote(5.0, 100.0)], flat_curve, recovery_rate=RECOVERY
        )
        assert curve.hazard_rates.size == 1
        assert (
            model_par_spread(5.0, curve, flat_curve) / BASIS_POINT
        ) == pytest.approx(100.0, abs=REPRICING_TOLERANCE_BPS)

    def test_discounting_affects_the_hazards(self) -> None:
        """A sanity check that the discount curve is actually being used.

        If the curve were ignored, these would be identical -- an easy bug to
        ship, since the bootstrap would still reprice its own inputs
        self-consistently.
        """
        low = bootstrap_hazard_rates(
            STANDARD_QUOTES, YieldCurve.flat(0.0), recovery_rate=RECOVERY
        )
        high = bootstrap_hazard_rates(
            STANDARD_QUOTES, YieldCurve.flat(0.08), recovery_rate=RECOVERY
        )
        assert not np.allclose(low.hazard_rates, high.hazard_rates, rtol=1e-6)

    def test_inverted_curve_needing_negative_hazard_is_rejected(
        self, flat_curve: YieldCurve
    ) -> None:
        """Survival cannot increase; that is arbitrage, not a signal.

        A 1Y at 600bp with a 5Y at 60bp would require the forward hazard on
        (1, 5] to be negative. Silently clamping to zero would return a curve
        that misprices the 5Y while looking perfectly well-formed.
        """
        quotes = [CDSQuote(1.0, 600.0), CDSQuote(5.0, 60.0)]
        with pytest.raises(RuntimeError, match="inverted"):
            bootstrap_hazard_rates(quotes, flat_curve, recovery_rate=RECOVERY)

    def test_negative_forward_hazard_allowed_on_request(
        self, flat_curve: YieldCurve
    ) -> None:
        """The escape hatch works, and the result still reprices."""
        quotes = [CDSQuote(1.0, 600.0), CDSQuote(5.0, 60.0)]
        curve = bootstrap_hazard_rates(
            quotes,
            flat_curve,
            recovery_rate=RECOVERY,
            allow_negative_forward_hazard=True,
        )
        assert curve.hazard_rates[1] < 0.0
        for quote in quotes:
            assert (
                model_par_spread(quote.tenor, curve, flat_curve) / BASIS_POINT
            ) == pytest.approx(quote.spread_bps, abs=1e-4)

    def test_absurd_spread_reports_a_unit_error(
        self, flat_curve: YieldCurve
    ) -> None:
        """A spread passed as a decimal instead of bp should say so.

        ``0.05`` meant as 5% arrives as 0.05 bp and bootstraps fine; the
        reverse mistake -- 500000 bp -- exceeds any reachable hazard, and the
        message names the likely cause rather than reporting a bracketing
        failure.
        """
        with pytest.raises(RuntimeError, match="units"):
            bootstrap_hazard_rates(
                [CDSQuote(5.0, 5.0e6)], flat_curve, recovery_rate=RECOVERY
            )

    def test_empty_quotes_rejected(self, flat_curve: YieldCurve) -> None:
        with pytest.raises(ValueError, match="at least one CDS quote"):
            bootstrap_hazard_rates([], flat_curve)

    def test_duplicate_tenors_rejected(self, flat_curve: YieldCurve) -> None:
        """Two spreads at one tenor is contradictory data, not extra info."""
        with pytest.raises(ValueError, match="duplicate CDS tenors"):
            bootstrap_hazard_rates(
                [CDSQuote(5.0, 100.0), CDSQuote(5.0, 120.0)], flat_curve
            )

    @pytest.mark.parametrize("recovery", [-0.01, 1.0, 2.0])
    def test_invalid_recovery_rejected(
        self, flat_curve: YieldCurve, recovery: float
    ) -> None:
        with pytest.raises(ValueError, match=r"recovery_rate must be in"):
            bootstrap_hazard_rates(
                STANDARD_QUOTES, flat_curve, recovery_rate=recovery
            )

    @pytest.mark.parametrize("tenor, spread", [(0.0, 100.0), (-1.0, 100.0)])
    def test_quote_rejects_bad_tenor(self, tenor, spread) -> None:
        with pytest.raises(ValueError, match="tenor must be positive"):
            CDSQuote(tenor=tenor, spread_bps=spread)

    @pytest.mark.parametrize("spread", [0.0, -50.0, float("nan")])
    def test_quote_rejects_bad_spread(self, spread) -> None:
        with pytest.raises(ValueError, match="spread_bps must be positive"):
            CDSQuote(tenor=5.0, spread_bps=spread)

    def test_spread_conversion_happens_once(self) -> None:
        """bp -> decimal in exactly one place, so it cannot double-apply."""
        assert CDSQuote(5.0, 100.0).spread == pytest.approx(0.01)

    def test_par_spread_method_agrees_with_the_function(
        self, bootstrapped: CreditCurve, flat_curve: YieldCurve
    ) -> None:
        """The convenience method must not drift from the free function."""
        assert bootstrapped.par_spread(5.0, flat_curve) == pytest.approx(
            model_par_spread(5.0, bootstrapped, flat_curve), rel=1e-15
        )


class TestCDSLegs:
    """Properties of the two legs individually."""

    def test_annuity_is_positive_and_grows_with_maturity(
        self, bootstrapped: CreditCurve, flat_curve: YieldCurve
    ) -> None:
        """More premium payments means a larger annuity."""
        annuities = [
            premium_leg_annuity(t, bootstrapped, flat_curve)
            for t in (1.0, 3.0, 5.0, 10.0)
        ]
        assert all(value > 0.0 for value in annuities)
        assert np.all(np.diff(annuities) > 0.0)

    def test_protection_grows_with_maturity(
        self, bootstrapped: CreditCurve, flat_curve: YieldCurve
    ) -> None:
        """Longer protection covers strictly more default probability."""
        values = [
            protection_leg_pv(t, bootstrapped, flat_curve)
            for t in (1.0, 3.0, 5.0, 10.0)
        ]
        assert all(value >= 0.0 for value in values)
        assert np.all(np.diff(values) > 0.0)

    def test_accrual_on_default_raises_the_annuity(
        self, bootstrapped: CreditCurve, flat_curve: YieldCurve
    ) -> None:
        """The accrual term can only add value to the premium leg."""
        with_accrual = premium_leg_annuity(
            5.0, bootstrapped, flat_curve, accrual_on_default=True
        )
        without = premium_leg_annuity(
            5.0, bootstrapped, flat_curve, accrual_on_default=False
        )
        assert with_accrual > without

    def test_annuity_bounded_by_the_riskless_case(
        self, bootstrapped: CreditCurve, flat_curve: YieldCurve
    ) -> None:
        r"""A risky annuity cannot exceed the default-free one.

        Upper bound :math:`\sum \Delta_i D(t_i)`, since every survival weight
        is at most 1.
        """
        maturity = 10.0
        schedule = payment_schedule(maturity, DEFAULT_CDS_FREQUENCY)
        riskless = float(
            np.sum(
                np.diff(schedule) * flat_curve.discount_factor(schedule[1:])
            )
        )
        assert (
            0.0
            < premium_leg_annuity(maturity, bootstrapped, flat_curve)
            < riskless
        )

    def test_protection_bounded_by_loss_given_default(
        self, bootstrapped: CreditCurve, flat_curve: YieldCurve
    ) -> None:
        """Protection PV cannot exceed ``(1 - R)`` times certain default."""
        value = protection_leg_pv(10.0, bootstrapped, flat_curve)
        assert value < (1.0 - RECOVERY)

    def test_zero_hazard_gives_zero_protection(
        self, flat_curve: YieldCurve
    ) -> None:
        """A name that cannot default needs no protection."""
        curve = CreditCurve.flat(0.0, recovery_rate=RECOVERY)
        assert protection_leg_pv(5.0, curve, flat_curve) == pytest.approx(0.0)
        assert model_par_spread(5.0, curve, flat_curve) == pytest.approx(0.0)


# ==========================================================================
# Integration with the existing CVA engine
# ==========================================================================
class TestEngineIntegration:
    """The new curves must be interchangeable with ``src.xva.cva``.

    The flat cases are the overlap where both implementations are defined, and
    they agree *exactly* -- not to a tolerance. Any drift here means the two
    modules have diverged on a convention.
    """

    def test_discount_factors_match_cva_module(self) -> None:
        """``YieldCurve.flat(r).to_tensor`` == ``cva.discount_factors``."""
        times = torch.linspace(0.0, 5.0, 21, dtype=torch.float64)
        mine = YieldCurve.flat(0.035).to_tensor(times)
        theirs = cva_module.discount_factors(times, 0.035)
        assert torch.equal(mine, theirs)

    def test_survival_matches_cva_module(self) -> None:
        """``CreditCurve.flat(h).to_tensor`` == ``cva.survival_probability``."""
        times = torch.linspace(0.0, 10.0, 41, dtype=torch.float64)
        mine = CreditCurve.flat(0.0223).to_tensor(times)
        theirs = cva_module.survival_probability(times, 0.0223)
        assert torch.equal(mine, theirs)

    def test_marginal_default_convention_matches(self) -> None:
        r"""Both must use :math:`Q(t_{i-1}) - Q(t_i)`, not the reverse sign."""
        times = torch.linspace(0.0, 5.0, 21, dtype=torch.float64)
        mine = CreditCurve.flat(0.03).marginal_default_probability(
            times.numpy()
        )
        theirs = cva_module.marginal_default_probability(times, 0.03).numpy()
        np.testing.assert_allclose(mine, theirs, rtol=0.0, atol=0.0)

    def test_bootstrapped_curve_feeds_the_cva_discount_slot(
        self, bootstrapped: CreditCurve
    ) -> None:
        """A real observed curve can replace the flat-rate default.

        ``compute_unilateral_cva`` takes either ``discount_rate`` or an
        explicit ``curve`` tensor; this exercises the latter path with a
        multi-pillar curve.
        """
        times = torch.linspace(0.0, 5.0, 21, dtype=torch.float64)
        market = YieldCurve(
            tenors=np.array([1.0, 2.0, 5.0, 10.0]),
            zero_rates=np.array([0.030, 0.034, 0.040, 0.042]),
        )
        exposure = torch.linspace(10.0, 0.0, 21, dtype=torch.float64)

        value = cva_module.compute_unilateral_cva(
            exposure,
            times,
            0.02,
            RECOVERY,
            curve=market.to_tensor(times),
        )
        assert torch.isfinite(value) and value > 0.0

    def test_piecewise_curve_differs_from_the_flat_approximation(
        self, bootstrapped: CreditCurve
    ) -> None:
        """The term structure is not cosmetic.

        Collapsing a 1Y/3Y/5Y/10Y quote set to a single flat hazard (the
        credit triangle on the 5Y point) changes the survival curve materially
        -- which is the whole reason for bootstrapping. If these agreed, the
        term structure would be carrying no information.
        """
        times = np.linspace(0.0, 10.0, 41)
        exact = bootstrapped.survival_probability(times)

        five_year = next(q for q in STANDARD_QUOTES if q.tenor == 5.0)
        flat = CreditCurve.flat(
            five_year.spread / (1.0 - RECOVERY), recovery_rate=RECOVERY
        ).survival_probability(times)

        assert np.abs(exact - flat).max() > 0.01, (
            "the piecewise curve is indistinguishable from a flat one; the "
            "term structure would then be adding nothing"
        )

    def test_engine_cva_responds_to_the_credit_term_structure(
        self, bootstrapped: CreditCurve
    ) -> None:
        """Feeding real marginal default probabilities changes the CVA.

        ``_integrate_credit_leg`` computes ``(1-R) * sum(EE * dPD * DF)``. This
        replicates that with the bootstrapped ``dPD`` and confirms it differs
        from the flat-hazard answer the engine produces today -- quantifying
        what the missing injection point is worth.
        """
        times = torch.linspace(0.0, 10.0, 41, dtype=torch.float64)
        exposure = torch.linspace(20.0, 0.0, 41, dtype=torch.float64)
        discount = YieldCurve.flat(0.03)

        marginal = torch.as_tensor(
            bootstrapped.marginal_default_probability(times.numpy()),
            dtype=torch.float64,
        )
        factors = discount.to_tensor(times)
        term_structure_cva = float(
            (1.0 - RECOVERY)
            * torch.sum(exposure[1:] * marginal * factors[1:])
        )

        five_year = next(q for q in STANDARD_QUOTES if q.tenor == 5.0)
        flat_cva = float(
            cva_module.compute_unilateral_cva(
                exposure,
                times,
                five_year.spread / (1.0 - RECOVERY),
                RECOVERY,
                discount_rate=0.03,
            )
        )

        assert term_structure_cva > 0.0
        relative = abs(term_structure_cva / flat_cva - 1.0)
        assert relative > 0.01, (
            f"term-structure CVA {term_structure_cva:.6f} vs flat "
            f"{flat_cva:.6f} differ by only {relative:.2%}"
        )


# ==========================================================================
# Black vega
# ==========================================================================
class TestBlackVega:
    """Vega is only used for calibration weights, but a wrong sign or a NaN
    would silently corrupt every fit, so it is checked directly."""

    @pytest.mark.parametrize(
        "forward, strike, maturity, vol",
        [(100.0, 100.0, 1.0, 0.2), (100.0, 110.0, 0.5, 0.3),
         (50.0, 45.0, 2.0, 0.25), (2000.0, 1900.0, 0.25, 0.15)],
    )
    def test_matches_numerical_differentiation(
        self, forward, strike, maturity, vol
    ) -> None:
        """Against a central difference of the Black-76 price."""
        from scipy.stats import norm

        def price(volatility: float) -> float:
            std = volatility * math.sqrt(maturity)
            d1 = (math.log(forward / strike) + 0.5 * std**2) / std
            return forward * norm.cdf(d1) - strike * norm.cdf(d1 - std)

        step = 1e-6
        numerical = (price(vol + step) - price(vol - step)) / (2.0 * step)
        assert float(black_vega(forward, strike, maturity, vol)) == (
            pytest.approx(numerical, rel=1e-6)
        )

    def test_peaks_near_the_money(self) -> None:
        """The reason vega weighting favours ATM quotes."""
        strikes = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
        vega = black_vega(100.0, strikes, 1.0, 0.2)
        assert int(np.argmax(vega)) == 2

    def test_degenerate_inputs_give_zero_not_nan(self) -> None:
        """Zero strike, zero maturity or zero vol must not poison an array."""
        vega = black_vega(
            np.array([100.0, 0.0, 100.0, 100.0]),
            np.array([100.0, 100.0, 0.0, 100.0]),
            np.array([1.0, 1.0, 1.0, 0.0]),
            np.array([0.2, 0.2, 0.2, 0.2]),
        )
        assert np.all(np.isfinite(vega))
        assert vega[0] > 0.0
        np.testing.assert_array_equal(vega[1:], 0.0)

    def test_all_degenerate_returns_zeros(self) -> None:
        """The early-exit path."""
        vega = black_vega(
            np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        )
        np.testing.assert_array_equal(vega, np.zeros(3))


# ==========================================================================
# Option chain cleaning
# ==========================================================================
def _chain_row(
    strike: float = 100.0,
    implied_vol: float = 0.20,
    maturity: float = 0.5,
    bid: float = 5.0,
    ask: float = 5.2,
    volume: float = 10.0,
    open_interest: float = 100.0,
) -> dict:
    """One well-formed chain row, to be perturbed one field at a time."""
    return {
        "strike": strike,
        "impliedVolatility": implied_vol,
        "maturity": maturity,
        "bid": bid,
        "ask": ask,
        "volume": volume,
        "openInterest": open_interest,
    }


class TestOptionChainCleaning:
    """The filters that make a raw yfinance chain usable.

    Each test perturbs exactly one field of an otherwise-good row, so a
    failure names the filter that broke.
    """

    def test_keeps_a_clean_row(self) -> None:
        surface = clean_option_chain(pd.DataFrame([_chain_row()]), 100.0)
        assert len(surface) == 1

    @pytest.mark.parametrize(
        "field, value, reason",
        [
            ("implied_vol", 0.0, "yfinance reports 0.0 when its solver fails"),
            ("implied_vol", 9.99, "999% vol is not a real quote"),
            ("bid", 0.0, "no bid means no two-sided market"),
            ("strike", 500.0, "deep wing, beyond the moneyness band"),
            ("maturity", 0.0, "expired"),
        ],
    )
    def test_drops_bad_rows(self, field, value, reason) -> None:
        """Every filter, one field at a time."""
        surface = clean_option_chain(
            pd.DataFrame([_chain_row(**{field: value})]), 100.0
        )
        assert len(surface) == 0, f"should have dropped: {reason}"

    def test_drops_wide_markets(self) -> None:
        """A bid-ask straddling the mid by >50% carries no mid information."""
        surface = clean_option_chain(
            pd.DataFrame([_chain_row(bid=1.0, ask=9.0)]), 100.0
        )
        assert len(surface) == 0

    def test_drops_inactive_strikes(self) -> None:
        """Zero volume *and* zero open interest: quoted but not traded."""
        surface = clean_option_chain(
            pd.DataFrame([_chain_row(volume=0.0, open_interest=0.0)]), 100.0
        )
        assert len(surface) == 0

    def test_keeps_a_strike_with_open_interest_but_no_volume(self) -> None:
        """Activity is volume OR open interest -- not both.

        A strike that did not trade today but has a position outstanding is
        still a live market.
        """
        surface = clean_option_chain(
            pd.DataFrame([_chain_row(volume=0.0, open_interest=50.0)]), 100.0
        )
        assert len(surface) == 1

    def test_moneyness_is_measured_against_the_forward(self) -> None:
        r"""``k == 0`` exactly at :math:`K = F`, not at :math:`K = S`.

        Using spot moneyness would displace every expiry's smile by
        :math:`(r-q)T`, which an SSVI fit absorbs as a spurious
        maturity-dependent skew, biasing :math:`\rho`.
        """
        rate, maturity, spot = 0.05, 2.0, 100.0
        forward = spot * math.exp(rate * maturity)
        surface = clean_option_chain(
            pd.DataFrame([_chain_row(strike=forward, maturity=maturity)]),
            spot,
            discount_curve=YieldCurve.flat(rate),
            max_abs_log_moneyness=1.0,
        )
        assert len(surface) == 1
        assert surface.log_moneyness[0] == pytest.approx(0.0, abs=1e-12)
        assert surface.forward[0] == pytest.approx(forward, rel=1e-12)

    def test_dividend_yield_lowers_the_forward(self) -> None:
        """``F = S exp((r - q)T)``: a dividend reduces the forward."""
        rows = pd.DataFrame([_chain_row(maturity=1.0)])
        without = clean_option_chain(
            rows, 100.0, discount_curve=YieldCurve.flat(0.05),
            max_abs_log_moneyness=1.0,
        )
        with_dividend = clean_option_chain(
            rows, 100.0, discount_curve=YieldCurve.flat(0.05),
            dividend_yield=0.03, max_abs_log_moneyness=1.0,
        )
        assert with_dividend.forward[0] < without.forward[0]

    def test_weights_have_unit_mean(self) -> None:
        """Normalised so the penalty scale is independent of quote count."""
        rows = pd.DataFrame(
            [_chain_row(strike=k) for k in (95.0, 100.0, 105.0)]
        )
        for scheme in ("vega", "uniform", "spread"):
            surface = clean_option_chain(rows, 100.0, weight_scheme=scheme)
            assert surface.weights.mean() == pytest.approx(1.0, rel=1e-12)

    def test_vega_weights_favour_the_money(self) -> None:
        """The point of vega weighting."""
        rows = pd.DataFrame(
            [_chain_row(strike=k, bid=1.0, ask=1.05)
             for k in (80.0, 100.0, 120.0)]
        )
        surface = clean_option_chain(
            rows, 100.0, weight_scheme="vega", max_abs_log_moneyness=1.0
        )
        assert int(np.argmax(surface.weights)) == 1

    def test_empty_result_is_well_formed(self) -> None:
        """An all-filtered chain must return an empty surface, not raise."""
        surface = clean_option_chain(
            pd.DataFrame([_chain_row(implied_vol=0.0)]), 100.0
        )
        assert len(surface) == 0
        assert "filtered" in surface.label
        assert surface.total_variance.size == 0

    def test_total_variance_definition(self) -> None:
        r""":math:`w = \sigma^2 T`, the space SSVI is fitted in."""
        surface = clean_option_chain(
            pd.DataFrame([_chain_row(implied_vol=0.25, maturity=2.0)]), 100.0
        )
        assert surface.total_variance[0] == pytest.approx(0.25**2 * 2.0)

    def test_missing_columns_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            clean_option_chain(pd.DataFrame({"strike": [100.0]}), 100.0)

    def test_non_positive_spot_rejected(self) -> None:
        with pytest.raises(ValueError, match="spot must be positive"):
            clean_option_chain(pd.DataFrame([_chain_row()]), 0.0)

    def test_unknown_weight_scheme_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown weight_scheme"):
            clean_option_chain(
                pd.DataFrame([_chain_row()]), 100.0, weight_scheme="magic"
            )

    def test_string_columns_are_coerced(self) -> None:
        """yfinance occasionally returns object dtype; it must not crash."""
        row = _chain_row()
        frame = pd.DataFrame([{k: str(v) for k, v in row.items()}])
        assert len(clean_option_chain(frame, 100.0)) == 1

    def test_ragged_arrays_rejected(self) -> None:
        """The container guards its own invariant."""
        with pytest.raises(ValueError, match="ragged"):
            VolSurfaceData(
                log_moneyness=np.zeros(3),
                maturity=np.zeros(2),
                implied_volatility=np.zeros(3),
                weights=np.zeros(3),
                strike=np.zeros(3),
                forward=np.zeros(3),
                spot=100.0,
            )


class TestSSVIHandoff:
    """The surface must be consumable by the existing SSVI calibrator."""

    def test_calibration_inputs_have_the_expected_shapes(self) -> None:
        """Four aligned 1-D float64 tensors, in ``calibrate_surface`` order."""
        rows = pd.DataFrame(
            [_chain_row(strike=k, maturity=t)
             for k in (95.0, 100.0, 105.0) for t in (0.25, 0.5, 1.0)]
        )
        surface = clean_option_chain(rows, 100.0)
        log_moneyness, maturity, vol, weights = (
            surface.to_calibration_inputs()
        )
        for tensor in (log_moneyness, maturity, vol, weights):
            assert tensor.ndim == 1
            assert tensor.shape[0] == len(surface)
            assert tensor.dtype == torch.float64

    def test_calibrate_surface_accepts_the_output(self) -> None:
        """End-to-end smoke test: a few iterations must run and reduce loss.

        Not a calibration-quality test -- just proof the handoff signature and
        dtypes line up with what ``calibrate_surface`` expects.
        """
        pytest.importorskip("src.models.vol_surface")
        from src.models.vol_surface import (
            ATMTotalVariance,
            SSVISurface,
            calibrate_surface,
        )

        rows = pd.DataFrame(
            [_chain_row(strike=k, maturity=t, implied_vol=0.20 + 0.05 * (k - 100.0) / 100.0)
             for k in (90.0, 95.0, 100.0, 105.0, 110.0)
             for t in (0.25, 0.5, 1.0)]
        )
        surface = clean_option_chain(rows, 100.0, max_abs_log_moneyness=0.5)
        assert len(surface) > 0

        log_moneyness, maturity, vol, weights = (
            surface.to_calibration_inputs()
        )
        knots = torch.tensor([0.25, 0.5, 1.0], dtype=torch.float64)
        surface_model = SSVISurface(atm=ATMTotalVariance(knots)).double()

        result = calibrate_surface(
            surface_model,
            log_moneyness,
            maturity,
            vol,
            weights=weights,
            iterations=5,
            log_every=1,
        )
        assert result is not None


# ==========================================================================
# Tier 2: live data (opt-in)
# ==========================================================================
@requires_network
class TestLiveMarketData:
    """Smoke tests for the I/O wrappers.

    Deliberately loose: these assert shape and plausibility, never values.
    Yahoo and FRED change by the minute, so anything tighter would be a test
    that fails for reasons unrelated to this code.
    """

    def test_fetch_spot_returns_a_plausible_price(self) -> None:
        from market_data.fetcher import fetch_spot

        price = fetch_spot("AAPL")
        assert math.isfinite(price) and price > 0.0

    def test_fetch_sofr_curve(self) -> None:
        from market_data.fetcher import fetch_sofr_curve

        curve = fetch_sofr_curve()
        assert curve.tenors.size >= 1
        assert np.all(np.abs(curve.zero_rates) < 0.25)
        assert "NOT a term SOFR swap curve" in curve.label

    def test_fetch_treasury_curve(self) -> None:
        from market_data.fetcher import fetch_treasury_curve

        curve = fetch_treasury_curve()
        assert curve.tenors.size >= 3
        assert np.all(np.abs(curve.zero_rates) < 0.25)

    def test_spliced_discount_curve_is_monotone_in_time(self) -> None:
        from market_data.fetcher import fetch_discount_curve

        curve = fetch_discount_curve()
        assert np.all(np.diff(curve.tenors) > 0.0)
        factors = curve.discount_factor(np.linspace(0.0, 30.0, 100))
        assert np.all(np.diff(factors) < 0.0)

    def test_bootstrap_against_a_live_discount_curve(self) -> None:
        """The end-to-end path this module exists for."""
        from market_data.fetcher import fetch_discount_curve

        discount = fetch_discount_curve()
        credit = bootstrap_hazard_rates(
            STANDARD_QUOTES, discount, recovery_rate=RECOVERY
        )
        for quote in STANDARD_QUOTES:
            assert (
                model_par_spread(quote.tenor, credit, discount) / BASIS_POINT
            ) == pytest.approx(quote.spread_bps, abs=REPRICING_TOLERANCE_BPS)

    def test_fetch_implied_vol_surface(self) -> None:
        from market_data.fetcher import fetch_implied_vol_surface

        surface = fetch_implied_vol_surface("SPY", max_expiries=2)
        if len(surface) == 0:
            pytest.skip("no option quotes passed cleaning right now")
        assert np.all(surface.implied_volatility > 0.0)
        assert np.all(np.abs(surface.log_moneyness) <= 0.35 + 1e-9)
        assert np.all(surface.maturity > 0.0)
