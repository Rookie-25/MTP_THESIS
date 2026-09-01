r"""Phase 2 correctness suite: exposure profiles and CVA sensitivities.

Test strategy
-------------
Phase 2 has fewer closed-form anchors than Phase 1, so the suite leans on three
independent classes of check:

1. **Structural invariants** that must hold for *any* MtM surface -- exposure
   non-negativity, the identity :math:`V = V^+ - (-V)^+`, monotonicity of the
   survival curve, and the fact that marginal default probabilities telescope
   to :math:`Q(t_0) - Q(t_N)`. These catch sign errors and off-by-one slicing
   in the discrete integral.

2. **Analytic limits** where the general formula collapses to something
   checkable by hand: a deterministically in-the-money portfolio makes
   :math:`EE(t)` equal the deterministic MtM, which turns the CVA sum into a
   closed-form expression; a zero hazard rate must give exactly zero CVA; a
   100% recovery must give exactly zero CVA.

3. **AAD vs bump-and-revalue** under common random numbers -- the Phase 2
   acceptance criterion, mirroring Phase 1's methodology.

A note on what is deliberately *not* tested at tight tolerance
--------------------------------------------------------------
``PFE`` differentiates through an order statistic, so its finite-difference
derivative is a step function of the parameters at finite :math:`M` and does
not converge to the AAD value at any fixed bump size. That is a property of the
estimator, not a defect, and it is documented in
:mod:`src.xva.exposure`. CVA depends only on ``EE`` (a smooth sample mean), so
the acceptance criterion is unaffected. PFE is tested for *structure* here, not
for gradient agreement.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.gbm import GBMSimulator
from src.pricer.greeks import compare_greeks, format_comparison
from src.pricer.options import SwapLeg, portfolio_swap_mtm
from src.xva.cva import (
    compute_unilateral_cva,
    compute_unilateral_dva,
    compute_xva,
    cva_aad_greeks,
    cva_bump_and_revalue_greeks,
    discount_factors,
    make_cva_valuation_fn,
    marginal_default_probability,
    survival_probability,
)
from src.xva.exposure import (
    CSATerms,
    collateral_balance,
    collateral_required,
    collateralized_exposure,
    compute_collateralized_exposure_profile,
    compute_exposure_profile,
    differentiable_quantile,
    expected_collateralized_exposure,
    expected_exposure,
    expected_negative_exposure,
    expected_positive_exposure,
    mpor_lag_steps,
    negative_exposure,
    positive_exposure,
    potential_future_exposure,
)

# Shared market / contract / credit setup.
S0 = 100.0
STRIKE = 100.0
RATE = 0.03
SIGMA = 0.20
MATURITY = 1.0
N_STEPS = 48
N_PATHS = 100_000
SEED = 20260814

HAZARD_RATE = 0.02
RECOVERY_RATE = 0.4
CONFIDENCE_LEVEL = 0.95

# Phase 2 acceptance criterion: AAD CVA Greeks must match bump-and-revalue
# within this absolute tolerance.
GREEK_TOLERANCE = 1e-3

# Secondary, tighter check. Absolute tolerance alone is weak when CVA is small
# (a unit-notional CVA is ~0.06), so a relative bound is asserted as well to
# give the test real discriminating power. Kept loose enough to absorb the
# O(h) kink bias that finite differences carry through max(V, 0) -- see
# `cva_bump_and_revalue_greeks`.
GREEK_RELATIVE_TOLERANCE = 5e-3


@pytest.fixture(scope="module")
def simulator() -> GBMSimulator:
    """CPU, float64 simulator: correctness baseline, not a speed benchmark."""
    return GBMSimulator(maturity=MATURITY, n_steps=N_STEPS, device=torch.device("cpu"))


@pytest.fixture(scope="module")
def brownian_increments(simulator: GBMSimulator) -> torch.Tensor:
    """A single fixed Brownian sample shared by every AAD/FD comparison."""
    return simulator.draw_increments(N_PATHS, seed=SEED)


@pytest.fixture(scope="module")
def legs() -> list[SwapLeg]:
    """A two-leg netting set, so netting (not just a single trade) is exercised."""
    return [
        SwapLeg(notional=1.0, strike=STRIKE, maturity=MATURITY),
        SwapLeg(notional=-0.4, strike=110.0, maturity=MATURITY),
    ]


@pytest.fixture(scope="module")
def mtm_surface(
    simulator: GBMSimulator, brownian_increments: torch.Tensor, legs: list[SwapLeg]
) -> torch.Tensor:
    """Netted MtM surface of shape ``(n_paths, n_steps + 1)``."""
    paths = simulator.simulate(S0, RATE, SIGMA, dW=brownian_increments)
    return portfolio_swap_mtm(paths, simulator.time_grid(), legs, RATE)


class TestExposureStructure:
    """Invariants that must hold for any mark-to-market surface."""

    def test_exposure_profiles_have_positive_floor(self, mtm_surface: torch.Tensor) -> None:
        """EE, ENE and PFE can never be negative -- the core Phase 2 requirement."""
        ee = expected_exposure(mtm_surface)
        ene = expected_negative_exposure(mtm_surface)
        pfe = potential_future_exposure(mtm_surface, CONFIDENCE_LEVEL)

        assert torch.all(ee >= 0.0), f"EE has negative entries: min={float(ee.min())}"
        assert torch.all(ene >= 0.0), f"ENE has negative entries: min={float(ene.min())}"
        assert torch.all(pfe >= 0.0), f"PFE has negative entries: min={float(pfe.min())}"

    def test_pathwise_exposures_are_non_negative(self, mtm_surface: torch.Tensor) -> None:
        assert torch.all(positive_exposure(mtm_surface) >= 0.0)
        assert torch.all(negative_exposure(mtm_surface) >= 0.0)

    def test_positive_minus_negative_recovers_mtm(self, mtm_surface: torch.Tensor) -> None:
        r"""Identity :math:`V = \max(V,0) - \max(-V,0)` must hold exactly."""
        reconstructed = positive_exposure(mtm_surface) - negative_exposure(mtm_surface)
        assert torch.allclose(reconstructed, mtm_surface, atol=0.0, rtol=0.0)

    def test_profile_shapes_match_time_grid(
        self, mtm_surface: torch.Tensor, simulator: GBMSimulator
    ) -> None:
        profile = compute_exposure_profile(
            mtm_surface, simulator.time_grid(), confidence_level=CONFIDENCE_LEVEL
        )
        expected_shape = (N_STEPS + 1,)
        assert tuple(profile.ee.shape) == expected_shape
        assert tuple(profile.ene.shape) == expected_shape
        assert tuple(profile.pfe.shape) == expected_shape
        assert profile.n_paths == N_PATHS

    def test_pfe_dominates_ee(self, mtm_surface: torch.Tensor) -> None:
        """A 95th percentile must sit at or above the mean for these profiles.

        This is not a universal mathematical law, but for a right-skewed
        exposure distribution generated by GBM it holds comfortably, and a
        violation would signal that the quantile is reducing over the wrong
        dimension.
        """
        ee = expected_exposure(mtm_surface)
        pfe = potential_future_exposure(mtm_surface, CONFIDENCE_LEVEL)
        # t=0 is deterministic (all paths identical), so EE == PFE exactly there.
        assert torch.all(pfe[1:] > ee[1:])
        assert math.isclose(float(pfe[0]), float(ee[0]), rel_tol=1e-12)

    def test_exposure_at_time_zero_is_deterministic(
        self, mtm_surface: torch.Tensor, legs: list[SwapLeg]
    ) -> None:
        """Every path starts at S0, so the t=0 MtM column has zero dispersion."""
        column = mtm_surface[:, 0]
        assert float(column.std()) < 1e-12

        expected_v0 = sum(
            leg.notional * (S0 - leg.strike) * math.exp(-RATE * (leg.maturity - 0.0))
            for leg in legs
        )
        assert math.isclose(float(column[0]), expected_v0, rel_tol=1e-12)

    def test_epe_is_bounded_by_profile_extremes(
        self, mtm_surface: torch.Tensor, simulator: GBMSimulator
    ) -> None:
        """A time-average must lie between the min and max of what it averages."""
        ee = expected_exposure(mtm_surface)
        epe = expected_positive_exposure(ee, simulator.time_grid())
        assert float(ee.min()) <= float(epe) <= float(ee.max())


class TestDifferentiableQuantile:
    """The hand-rolled quantile must match torch.quantile and stay on the tape."""

    def test_matches_torch_quantile(self) -> None:
        generator = torch.Generator().manual_seed(7)
        values = torch.randn((5_000, 12), dtype=torch.float64, generator=generator)
        for q in (0.0, 0.05, 0.5, 0.95, 0.99, 1.0):
            ours = differentiable_quantile(values, q, dim=0)
            theirs = torch.quantile(values, q, dim=0)
            assert torch.allclose(ours, theirs, atol=1e-12, rtol=0.0), f"mismatch at q={q}"

    def test_preserves_gradients(self) -> None:
        values = torch.randn((1_000,), dtype=torch.float64, requires_grad=True)
        quantile = differentiable_quantile(values, 0.95, dim=0)
        quantile.backward()
        assert values.grad is not None
        # Exactly one order statistic defines the quantile when q*(M-1) is an
        # integer, so the adjoint must be a one-hot vector summing to 1.
        assert math.isclose(float(values.grad.sum()), 1.0, rel_tol=1e-12)

    def test_rejects_out_of_range_level(self) -> None:
        values = torch.randn((100,), dtype=torch.float64)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            differentiable_quantile(values, 1.5, dim=0)


class TestCreditCurve:
    """Flat-hazard survival and default probabilities."""

    def test_survival_starts_at_one_and_decays(self, simulator: GBMSimulator) -> None:
        times = simulator.time_grid()
        survival = survival_probability(times, HAZARD_RATE)
        assert math.isclose(float(survival[0]), 1.0, rel_tol=1e-12)
        assert torch.all(survival[1:] < survival[:-1])
        assert torch.all(survival > 0.0)

    def test_survival_matches_closed_form(self, simulator: GBMSimulator) -> None:
        times = simulator.time_grid()
        survival = survival_probability(times, HAZARD_RATE)
        expected = torch.exp(-HAZARD_RATE * times)
        assert torch.allclose(survival, expected, atol=1e-14)

    def test_default_probabilities_telescope(self, simulator: GBMSimulator) -> None:
        r""":math:`\sum_i d\!PD_i = Q(t_0) - Q(t_N)`."""
        times = simulator.time_grid()
        dpd = marginal_default_probability(times, HAZARD_RATE)
        total = float(dpd.sum())
        expected = 1.0 - math.exp(-HAZARD_RATE * MATURITY)
        assert math.isclose(total, expected, rel_tol=1e-12)
        assert torch.all(dpd > 0.0)
        assert dpd.shape == (N_STEPS,)

    def test_zero_hazard_gives_zero_default_probability(
        self, simulator: GBMSimulator
    ) -> None:
        dpd = marginal_default_probability(simulator.time_grid(), 0.0)
        assert torch.all(dpd == 0.0)

    def test_discount_factors_match_closed_form(self, simulator: GBMSimulator) -> None:
        times = simulator.time_grid()
        factors = discount_factors(times, RATE)
        assert torch.allclose(factors, torch.exp(-RATE * times), atol=1e-14)
        assert math.isclose(float(factors[0]), 1.0, rel_tol=1e-12)


class TestCVAAnalyticLimits:
    """Degenerate cases where the CVA sum has a hand-checkable answer."""

    def test_zero_hazard_rate_gives_zero_cva(
        self, mtm_surface: torch.Tensor, simulator: GBMSimulator
    ) -> None:
        ee = expected_exposure(mtm_surface)
        cva = compute_unilateral_cva(
            ee, simulator.time_grid(), 0.0, RECOVERY_RATE, discount_rate=RATE
        )
        assert float(cva) == 0.0

    def test_full_recovery_gives_zero_cva(
        self, mtm_surface: torch.Tensor, simulator: GBMSimulator
    ) -> None:
        ee = expected_exposure(mtm_surface)
        cva = compute_unilateral_cva(
            ee, simulator.time_grid(), HAZARD_RATE, 1.0, discount_rate=RATE
        )
        assert float(cva) == 0.0

    def test_cva_is_positive_for_positive_exposure(
        self, mtm_surface: torch.Tensor, simulator: GBMSimulator
    ) -> None:
        ee = expected_exposure(mtm_surface)
        cva = compute_unilateral_cva(
            ee, simulator.time_grid(), HAZARD_RATE, RECOVERY_RATE, discount_rate=RATE
        )
        assert float(cva) > 0.0

    def test_cva_increases_with_hazard_rate(
        self, mtm_surface: torch.Tensor, simulator: GBMSimulator
    ) -> None:
        ee = expected_exposure(mtm_surface)
        times = simulator.time_grid()
        low = compute_unilateral_cva(ee, times, 0.01, RECOVERY_RATE, discount_rate=RATE)
        high = compute_unilateral_cva(ee, times, 0.05, RECOVERY_RATE, discount_rate=RATE)
        assert float(high) > float(low)

    def test_deterministic_exposure_matches_hand_computed_sum(self) -> None:
        r"""Flat unit exposure reduces CVA to :math:`(1-R)\sum_i d\!PD_i DF(t_i)`.

        Bypassing the simulator entirely isolates the discrete integrator, so a
        slicing or convention error in the sum cannot hide behind MC noise.
        """
        n_steps = 4
        times = torch.linspace(0.0, 1.0, n_steps + 1, dtype=torch.float64)
        ee = torch.ones(n_steps + 1, dtype=torch.float64)

        cva = compute_unilateral_cva(
            ee, times, HAZARD_RATE, RECOVERY_RATE, discount_rate=RATE
        )

        expected = 0.0
        for i in range(1, n_steps + 1):
            t_prev, t_curr = float(times[i - 1]), float(times[i])
            dpd = math.exp(-HAZARD_RATE * t_prev) - math.exp(-HAZARD_RATE * t_curr)
            expected += dpd * math.exp(-RATE * t_curr)
        expected *= 1.0 - RECOVERY_RATE

        assert math.isclose(float(cva), expected, rel_tol=1e-12)

    def test_conventions_converge_as_the_grid_refines(self) -> None:
        r"""Endpoint and trapezoidal rules must agree in the limit :math:`\Delta t \to 0`.

        Both are consistent discretisations of the same integral, so their gap
        is a discretisation artefact of order :math:`O(\Delta t)`, driven by how
        much :math:`EE` moves within one interval. On a coarse grid the two can
        differ by several percent -- that is expected, not a defect -- but the
        gap must shrink roughly in proportion to the step size. Asserting the
        *convergence rate* rather than an arbitrary fixed bound is what actually
        pins down the correctness of both branches of the integrator.

        The exposure profile is the fixed continuous function
        :math:`EE(t) = 1 + t`, sampled on successively finer grids, so the only
        thing changing between runs is the discretisation.
        """
        gaps = []
        for n_steps in (8, 64, 512):
            times = torch.linspace(0.0, 1.0, n_steps + 1, dtype=torch.float64)
            ee = 1.0 + times

            endpoint = compute_unilateral_cva(
                ee, times, HAZARD_RATE, RECOVERY_RATE,
                discount_rate=RATE, convention="endpoint",
            )
            average = compute_unilateral_cva(
                ee, times, HAZARD_RATE, RECOVERY_RATE,
                discount_rate=RATE, convention="average",
            )
            assert float(endpoint) != float(average)
            gaps.append(abs(float(endpoint) - float(average)) / float(endpoint))

        # Strictly decreasing, and each 8x refinement cuts the gap by ~8x.
        assert gaps[0] > gaps[1] > gaps[2]
        for coarse, fine in zip(gaps, gaps[1:]):
            assert 4.0 < coarse / fine < 16.0, f"gap fell by {coarse / fine:.1f}x, expected ~8x"
        assert gaps[-1] < 1e-3

    def test_dva_uses_negative_exposure(
        self, mtm_surface: torch.Tensor, simulator: GBMSimulator
    ) -> None:
        """DVA on the ENE profile must be positive and distinct from CVA."""
        times = simulator.time_grid()
        cva = compute_unilateral_cva(
            expected_exposure(mtm_surface), times, HAZARD_RATE, RECOVERY_RATE,
            discount_rate=RATE,
        )
        dva = compute_unilateral_dva(
            expected_negative_exposure(mtm_surface), times, HAZARD_RATE, RECOVERY_RATE,
            discount_rate=RATE,
        )
        assert float(dva) > 0.0
        assert not math.isclose(float(cva), float(dva), rel_tol=1e-6)

    def test_compute_xva_is_consistent_with_standalone_calls(
        self, mtm_surface: torch.Tensor, simulator: GBMSimulator
    ) -> None:
        times = simulator.time_grid()
        result = compute_xva(
            mtm_surface,
            times,
            hazard_rate=HAZARD_RATE,
            discount_rate=RATE,
            recovery_rate=RECOVERY_RATE,
            confidence_level=CONFIDENCE_LEVEL,
        )
        standalone_cva = compute_unilateral_cva(
            expected_exposure(mtm_surface), times, HAZARD_RATE, RECOVERY_RATE,
            discount_rate=RATE,
        )
        assert math.isclose(float(result.cva), float(standalone_cva), rel_tol=1e-12)
        assert math.isclose(
            float(result.bilateral_adjustment),
            float(result.dva) - float(result.cva),
            rel_tol=1e-12,
        )


class TestGradientTracking:
    """The tape must survive the whole exposure -> CVA chain."""

    def test_exposure_profiles_track_gradients(
        self, simulator: GBMSimulator, brownian_increments: torch.Tensor,
        legs: list[SwapLeg],
    ) -> None:
        s0 = torch.tensor(S0, dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
        paths = simulator.simulate(s0, RATE, sigma, dW=brownian_increments)
        mtm = portfolio_swap_mtm(paths, simulator.time_grid(), legs, RATE)
        profile = compute_exposure_profile(mtm, simulator.time_grid())

        assert profile.ee.requires_grad
        assert profile.ene.requires_grad
        assert profile.pfe.requires_grad
        assert profile.epe.requires_grad

    def test_cva_backward_populates_all_leaf_grads(
        self, simulator: GBMSimulator, brownian_increments: torch.Tensor,
        legs: list[SwapLeg],
    ) -> None:
        s0 = torch.tensor(S0, dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
        hazard = torch.tensor(HAZARD_RATE, dtype=torch.float64, requires_grad=True)

        paths = simulator.simulate(s0, RATE, sigma, dW=brownian_increments)
        mtm = portfolio_swap_mtm(paths, simulator.time_grid(), legs, RATE)
        ee = expected_exposure(mtm)
        cva = compute_unilateral_cva(
            ee, simulator.time_grid(), hazard, RECOVERY_RATE, discount_rate=RATE
        )
        cva.backward()

        for name, leaf in (("s0", s0), ("sigma", sigma), ("hazard_rate", hazard)):
            assert leaf.grad is not None, f"{name} received no gradient"
            assert torch.isfinite(leaf.grad), f"{name} gradient is not finite"

    def test_no_in_place_corruption_on_repeated_backward(
        self, simulator: GBMSimulator, brownian_increments: torch.Tensor,
        legs: list[SwapLeg],
    ) -> None:
        """Two independent forward+backward passes must give identical grads.

        An in-place write on a tensor the tape still needs would either raise a
        version-counter error or silently change the second result.
        """
        grads = []
        for _ in range(2):
            s0 = torch.tensor(S0, dtype=torch.float64, requires_grad=True)
            paths = simulator.simulate(s0, RATE, SIGMA, dW=brownian_increments)
            mtm = portfolio_swap_mtm(paths, simulator.time_grid(), legs, RATE)
            cva = compute_unilateral_cva(
                expected_exposure(mtm), simulator.time_grid(), HAZARD_RATE,
                RECOVERY_RATE, discount_rate=RATE,
            )
            cva.backward()
            grads.append(float(s0.grad))
        assert grads[0] == grads[1]


class TestCVASensitivities:
    """The Phase 2 acceptance criterion: AAD CVA Greeks vs bump-and-revalue."""

    @staticmethod
    @pytest.fixture(scope="class")
    def cva_fn(
        simulator: GBMSimulator, brownian_increments: torch.Tensor, legs: list[SwapLeg]
    ):
        return make_cva_valuation_fn(
            simulator,
            brownian_increments,
            legs,
            recovery_rate=RECOVERY_RATE,
            rate=RATE,
        )

    @staticmethod
    def _params() -> dict[str, float]:
        return {"s0": S0, "sigma": SIGMA, "hazard_rate": HAZARD_RATE}

    def test_aad_matches_finite_difference(self, cva_fn) -> None:
        params = self._params()
        aad = cva_aad_greeks(cva_fn, params)
        fd = cva_bump_and_revalue_greeks(cva_fn, params, scheme="central")
        comparison = compare_greeks(fd, aad)

        # Surfaced on failure so the actual magnitudes are visible in the log.
        print("\n" + format_comparison(comparison))

        assert set(aad.greeks) == {"s0", "sigma", "hazard_rate"}
        assert comparison.max_absolute_error < GREEK_TOLERANCE, (
            f"AAD vs FD absolute mismatch exceeds {GREEK_TOLERANCE}: "
            f"{comparison.absolute_error}"
        )
        assert comparison.max_relative_error < GREEK_RELATIVE_TOLERANCE, (
            f"AAD vs FD relative mismatch exceeds {GREEK_RELATIVE_TOLERANCE}: "
            f"{comparison.relative_error}"
        )
        # Both differentiate the same MC realisation, so the base CVA must agree
        # to machine precision.
        assert comparison.price_absolute_error < 1e-12

    def test_aad_uses_one_valuation_fd_uses_seven(self, cva_fn) -> None:
        """The headline efficiency claim, asserted rather than merely stated."""
        params = self._params()
        aad = cva_aad_greeks(cva_fn, params)
        fd = cva_bump_and_revalue_greeks(cva_fn, params, scheme="central")

        assert aad.n_valuations == 1
        # 3 parameters, central scheme: base + 3*(up + down) = 7.
        assert fd.n_valuations == 7

    def test_greek_signs_are_economically_sensible(self, cva_fn) -> None:
        r"""Sign checks that would catch a transposed or negated term.

        * :math:`\partial CVA/\partial\lambda > 0` -- a riskier counterparty
          costs more.
        * :math:`\partial CVA/\partial\sigma > 0` -- more volatility fattens the
          right tail of :math:`V^+`, raising EE.
        * :math:`\partial CVA/\partial S_0 > 0` -- this netting set is net long
          the underlying, so a higher spot raises exposure.
        """
        aad = cva_aad_greeks(cva_fn, self._params())
        assert aad.greeks["hazard_rate"] > 0.0
        assert aad.greeks["sigma"] > 0.0
        assert aad.greeks["s0"] > 0.0

    def test_hazard_rate_sensitivity_matches_semi_analytic_value(
        self, cva_fn, simulator: GBMSimulator, brownian_increments: torch.Tensor,
        legs: list[SwapLeg],
    ) -> None:
        r"""Credit delta has a closed form once EE is treated as fixed.

        :math:`EE(t)` does not depend on :math:`\lambda`, so

        .. math::
            \frac{\partial CVA}{\partial\lambda}
            = (1-R)\sum_i EE(t_i) DF(t_i)
              \frac{\partial\,d\!PD_i}{\partial\lambda},
            \qquad
            \frac{\partial\,d\!PD_i}{\partial\lambda}
            = t_i e^{-\lambda t_i} - t_{i-1} e^{-\lambda t_{i-1}}.

        This is an independent derivation of one component, computed without
        touching autograd at all.
        """
        times = simulator.time_grid()
        paths = simulator.simulate(S0, RATE, SIGMA, dW=brownian_increments)
        mtm = portfolio_swap_mtm(paths, times, legs, RATE)
        ee = expected_exposure(mtm).detach()
        factors = discount_factors(times, RATE)

        d_dpd = (
            times[1:] * torch.exp(-HAZARD_RATE * times[1:])
            - times[:-1] * torch.exp(-HAZARD_RATE * times[:-1])
        )
        expected = float((1.0 - RECOVERY_RATE) * torch.sum(ee[1:] * factors[1:] * d_dpd))

        aad = cva_aad_greeks(cva_fn, self._params(), wrt=["hazard_rate"])
        assert math.isclose(aad.greeks["hazard_rate"], expected, rel_tol=1e-10)

    def test_partial_sensitivities_match_full_run(self, cva_fn) -> None:
        params = self._params()
        full = cva_aad_greeks(cva_fn, params)
        partial = cva_aad_greeks(cva_fn, params, wrt=["sigma"])

        assert set(partial.greeks) == {"sigma"}
        assert math.isclose(partial.greeks["sigma"], full.greeks["sigma"], rel_tol=1e-12)

    def test_forward_scheme_also_agrees_loosely(self, cva_fn) -> None:
        """Forward differences carry O(h) truncation, so only a loose bound."""
        params = self._params()
        aad = cva_aad_greeks(cva_fn, params)
        fd = cva_bump_and_revalue_greeks(cva_fn, params, scheme="forward")
        comparison = compare_greeks(fd, aad)
        assert comparison.max_absolute_error < 1e-2

    def test_missing_hazard_rate_raises(self, cva_fn) -> None:
        """A parameter dict lacking 'hazard_rate' must fail loudly, not silently."""
        with pytest.raises(KeyError):
            cva_aad_greeks(cva_fn, {"s0": S0, "sigma": SIGMA})


# Ten business days, the standard regulatory MPOR for a daily-margined
# netting set. On the N_STEPS=48 grid used by most tests this rounds to zero
# lag, so the collateral tests below use their own finer grid where the MPOR
# is actually resolvable.
MPOR_10BD = 10.0 / 252.0
CSA_N_STEPS = 252
CSA_N_PATHS = 20_000


@pytest.fixture(scope="module")
def csa_simulator() -> GBMSimulator:
    """Daily grid, so a 10-business-day MPOR spans a whole number of steps."""
    return GBMSimulator(maturity=MATURITY, n_steps=CSA_N_STEPS, device=torch.device("cpu"))


@pytest.fixture(scope="module")
def csa_mtm(csa_simulator: GBMSimulator, legs: list[SwapLeg]) -> torch.Tensor:
    """MtM surface on the daily grid used by the collateral tests."""
    dW = csa_simulator.draw_increments(CSA_N_PATHS, seed=SEED)
    paths = csa_simulator.simulate(S0, RATE, SIGMA, dW=dW)
    return portfolio_swap_mtm(paths, csa_simulator.time_grid(), legs, RATE)


# The CSA configurations swept by the inequality test. Each is a realistic
# desk arrangement rather than an arbitrary parameter soup.
CSA_SCENARIOS = {
    "perfect": CSATerms(),
    "threshold_only": CSATerms(threshold=5.0),
    "mpor_only": CSATerms(margin_period_of_risk=MPOR_10BD),
    "threshold_mta_mpor": CSATerms(
        threshold=5.0, minimum_transfer_amount=1.0, margin_period_of_risk=MPOR_10BD
    ),
    "one_way_we_never_post": CSATerms(
        threshold=0.0, threshold_post=float("inf"), margin_period_of_risk=MPOR_10BD
    ),
}


class TestCollateralisedExposure:
    """Variation margin: thresholds, MTA and the margin period of risk."""

    @pytest.mark.parametrize("scenario", sorted(CSA_SCENARIOS))
    def test_collateralised_ee_never_exceeds_uncollateralised(
        self, csa_mtm: torch.Tensor, csa_simulator: GBMSimulator, scenario: str
    ) -> None:
        r"""The headline CSA requirement: :math:`EE_{collat}(t) \le EE_{uncollat}(t)`.

        Asserted elementwise across the whole profile for every realistic CSA
        configuration. Note this is an *expectation-level* statement; see
        ``test_pathwise_inequality_can_break_under_mpor`` for why the
        corresponding pathwise claim is not a theorem once the MPOR is
        non-zero.
        """
        times = csa_simulator.time_grid()
        terms = CSA_SCENARIOS[scenario]

        uncollateralised = expected_exposure(csa_mtm)
        collateralised = expected_collateralized_exposure(csa_mtm, times, terms)

        excess = collateralised - uncollateralised
        assert torch.all(excess <= 1e-12), (
            f"[{scenario}] collateralised EE exceeds uncollateralised at "
            f"{int((excess > 1e-12).sum())} dates; worst excess "
            f"{float(excess.max()):+.6e}"
        )

    def test_perfect_collateralisation_eliminates_exposure(
        self, csa_mtm: torch.Tensor, csa_simulator: GBMSimulator
    ) -> None:
        """Zero threshold, zero MTA, zero MPOR must neutralise exposure exactly."""
        collateralised = expected_collateralized_exposure(
            csa_mtm, csa_simulator.time_grid(), CSATerms()
        )
        assert float(collateralised.abs().max()) == 0.0

    def test_zero_mpor_collapses_to_capped_exposure(
        self, csa_mtm: torch.Tensor, csa_simulator: GBMSimulator
    ) -> None:
        r"""With MPOR = 0, exposure is exactly :math:`\min(V^+, H)`.

        This closed-form identity is the cleanest available check on the
        shrinkage function: it pins down both the receive and post branches
        with no Monte-Carlo tolerance at all.
        """
        threshold = 5.0
        terms = CSATerms(threshold=threshold)
        actual = collateralized_exposure(csa_mtm, csa_simulator.time_grid(), terms)
        expected = torch.clamp(torch.clamp(csa_mtm, min=0.0), max=threshold)
        assert torch.allclose(actual, expected, atol=0.0, rtol=0.0)

    def test_cva_orders_correctly_across_a_ladder_of_csa_tightness(
        self, csa_mtm: torch.Tensor, csa_simulator: GBMSimulator
    ) -> None:
        """Tightening the CSA must monotonically reduce CVA, in the right order.

        The ladder runs from no protection to perfect protection:

            uncollateralised  >  threshold + MTA + MPOR  >  MPOR only  >  perfect

        Each rung isolates one friction, so the ordering is a genuine economic
        statement rather than a tuned constant: an unsecured threshold leaves
        more residual risk than margining with only a settlement lag, which in
        turn leaves more than frictionless margining. Asserting the *ordering*
        avoids baking in an arbitrary "CVA must fall by X%" bound, which
        depends entirely on how the threshold compares to the exposure scale.
        """
        times = csa_simulator.time_grid()

        def cva_for(ee: torch.Tensor) -> float:
            return float(
                compute_unilateral_cva(
                    ee, times, HAZARD_RATE, RECOVERY_RATE, discount_rate=RATE
                )
            )

        uncollateralised = cva_for(expected_exposure(csa_mtm))
        thresholded = cva_for(
            expected_collateralized_exposure(
                csa_mtm, times, CSA_SCENARIOS["threshold_mta_mpor"]
            )
        )
        margined = cva_for(
            expected_collateralized_exposure(csa_mtm, times, CSA_SCENARIOS["mpor_only"])
        )
        perfect = cva_for(
            expected_collateralized_exposure(csa_mtm, times, CSA_SCENARIOS["perfect"])
        )

        assert uncollateralised > thresholded > margined > perfect
        assert perfect == 0.0
        # Margining with only a settlement lag must remove most of the risk;
        # the residual is driven purely by drift over the margin period.
        assert margined < 0.5 * uncollateralised

    def test_exposure_increases_monotonically_with_mpor(
        self, csa_mtm: torch.Tensor, csa_simulator: GBMSimulator
    ) -> None:
        """A longer margin period can only expose us to more market drift."""
        times = csa_simulator.time_grid()
        peaks = []
        for mpor_days in (0.0, 5.0, 10.0, 20.0):
            terms = CSATerms(margin_period_of_risk=mpor_days / 252.0)
            profile = expected_collateralized_exposure(csa_mtm, times, terms)
            peaks.append(float(profile.max()))
        assert peaks == sorted(peaks), f"peak EE not monotone in MPOR: {peaks}"
        assert peaks[0] == 0.0  # zero MPOR with zero threshold is perfect margining

    def test_exposure_increases_monotonically_with_threshold(
        self, csa_mtm: torch.Tensor, csa_simulator: GBMSimulator
    ) -> None:
        """A larger unsecured allowance can only leave more exposure."""
        times = csa_simulator.time_grid()
        means = [
            float(expected_collateralized_exposure(
                csa_mtm, times, CSATerms(threshold=h)
            ).mean())
            for h in (0.0, 2.5, 5.0, 10.0)
        ]
        assert means == sorted(means), f"mean EE not monotone in threshold: {means}"

    def test_large_threshold_recovers_uncollateralised_exposure(
        self, csa_mtm: torch.Tensor, csa_simulator: GBMSimulator
    ) -> None:
        """A threshold above any attainable MtM must reproduce the unsecured profile."""
        times = csa_simulator.time_grid()
        huge = float(csa_mtm.abs().max()) * 2.0
        terms = CSATerms(threshold=huge, threshold_post=huge)
        assert torch.allclose(
            expected_collateralized_exposure(csa_mtm, times, terms),
            expected_exposure(csa_mtm),
            atol=0.0, rtol=0.0,
        )

    def test_one_way_csa_never_posts_collateral(
        self, csa_mtm: torch.Tensor, csa_simulator: GBMSimulator
    ) -> None:
        """With an infinite posting threshold the balance can never go negative."""
        terms = CSATerms(threshold=0.0, threshold_post=float("inf"))
        balance = collateral_balance(csa_mtm, csa_simulator.time_grid(), terms)
        assert torch.all(balance >= 0.0)
        assert torch.all(torch.isfinite(balance))

    def test_mta_makes_the_balance_sticky(
        self, csa_mtm: torch.Tensor, csa_simulator: GBMSimulator
    ) -> None:
        """A non-zero MTA must suppress some transfers relative to MTA = 0."""
        times = csa_simulator.time_grid()
        without = collateral_balance(csa_mtm, times, CSATerms(threshold=5.0))
        with_mta = collateral_balance(
            csa_mtm, times, CSATerms(threshold=5.0, minimum_transfer_amount=2.0)
        )
        assert not torch.allclose(without, with_mta)
        # Stickiness means the balance tracks the requirement less closely.
        required = collateral_required(csa_mtm, 5.0, 5.0)
        assert float((with_mta - required).abs().mean()) > float(
            (without - required).abs().mean()
        )

    def test_vectorised_and_looped_paths_agree(
        self, csa_mtm: torch.Tensor, csa_simulator: GBMSimulator
    ) -> None:
        """The MTA = 0 fast path must equal the general recursion in the limit.

        ``collateral_balance`` takes a vectorised shortcut when the MTA is
        exactly zero. Driving the general loop with a negligible MTA must
        reproduce it, which is what guarantees the optimisation is not silently
        changing the model.
        """
        times = csa_simulator.time_grid()
        terms_fast = CSATerms(threshold=5.0, margin_period_of_risk=MPOR_10BD)
        terms_loop = CSATerms(
            threshold=5.0, minimum_transfer_amount=1e-300, margin_period_of_risk=MPOR_10BD
        )
        assert torch.allclose(
            collateral_balance(csa_mtm, times, terms_fast),
            collateral_balance(csa_mtm, times, terms_loop),
            atol=1e-12, rtol=0.0,
        )

    def test_pathwise_inequality_can_break_under_mpor(self) -> None:
        r"""Documented limitation: with MPOR > 0 the *pathwise* bound can fail.

        If we posted collateral against a deeply negative MtM and the market
        then reverses inside the margin period, we are exposed both to what
        they now owe us and to the collateral we posted and cannot recall.
        Here :math:`V` moves from -5 to +3 with one step of lag, so exposure is
        :math:`3 - (-5) = 8`, against an uncollateralised 3.

        This test exists to pin the behaviour down deliberately: it is the
        modelled economics of a margin period, and it is exactly why MPOR
        dominates collateralised CVA. It does not contradict
        ``test_collateralised_ee_never_exceeds_uncollateralised``, which is an
        expectation-level statement over a diffusive book.
        """
        times = torch.tensor([0.0, 0.1, 0.2], dtype=torch.float64)
        mtm = torch.tensor([[-5.0, -5.0, 3.0]], dtype=torch.float64)
        terms = CSATerms(threshold=0.0, margin_period_of_risk=0.1)

        collateralised = collateralized_exposure(mtm, times, terms)
        uncollateralised = positive_exposure(mtm)

        assert float(collateralised[0, 2]) == pytest.approx(8.0)
        assert float(uncollateralised[0, 2]) == pytest.approx(3.0)
        assert float(collateralised[0, 2]) > float(uncollateralised[0, 2])

    def test_gradients_flow_through_both_code_paths(
        self, csa_simulator: GBMSimulator, legs: list[SwapLeg]
    ) -> None:
        """The tape must survive the shortcut *and* the sequential recursion."""
        times = csa_simulator.time_grid()
        dW = csa_simulator.draw_increments(2_000, seed=SEED)

        for label, terms in (
            ("vectorised", CSATerms(threshold=5.0, margin_period_of_risk=MPOR_10BD)),
            (
                "looped",
                CSATerms(
                    threshold=5.0, minimum_transfer_amount=1.0,
                    margin_period_of_risk=MPOR_10BD,
                ),
            ),
        ):
            s0 = torch.tensor(S0, dtype=torch.float64, requires_grad=True)
            sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
            paths = csa_simulator.simulate(s0, RATE, sigma, dW=dW)
            mtm = portfolio_swap_mtm(paths, times, legs, RATE)
            cva = compute_unilateral_cva(
                expected_collateralized_exposure(mtm, times, terms),
                times, HAZARD_RATE, RECOVERY_RATE, discount_rate=RATE,
            )
            cva.backward()

            for name, leaf in (("s0", s0), ("sigma", sigma)):
                assert leaf.grad is not None, f"[{label}] {name} got no gradient"
                assert torch.isfinite(leaf.grad), f"[{label}] {name} gradient not finite"

    def test_collateralised_profile_has_expected_shape_and_floors(
        self, csa_mtm: torch.Tensor, csa_simulator: GBMSimulator
    ) -> None:
        profile = compute_collateralized_exposure_profile(
            csa_mtm,
            csa_simulator.time_grid(),
            CSA_SCENARIOS["threshold_mta_mpor"],
            confidence_level=CONFIDENCE_LEVEL,
        )
        assert tuple(profile.ee.shape) == (CSA_N_STEPS + 1,)
        assert torch.all(profile.ee >= 0.0)
        assert torch.all(profile.ene >= 0.0)
        assert torch.all(profile.pfe >= 0.0)
        assert profile.n_paths == CSA_N_PATHS


class TestCSAPlumbing:
    """Parameter validation and unit conversions around the CSA terms."""

    def test_mpor_lag_conversion(self) -> None:
        assert mpor_lag_steps(0.0, 1.0 / 252.0) == 0
        assert mpor_lag_steps(10.0 / 252.0, 1.0 / 252.0) == 10
        # A sub-step MPOR on a coarse grid rounds away -- documented behaviour.
        assert mpor_lag_steps(10.0 / 252.0, 1.0 / 12.0) == 0

    def test_mpor_lag_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            mpor_lag_steps(0.04, 0.0)
        with pytest.raises(ValueError, match="non-negative"):
            mpor_lag_steps(-0.04, 1.0 / 252.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"threshold": -1.0},
            {"minimum_transfer_amount": -1.0},
            {"margin_period_of_risk": -1.0},
            {"threshold_post": -1.0},
        ],
    )
    def test_csa_terms_reject_negative_parameters(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            CSATerms(**kwargs)

    def test_symmetric_default_for_posting_threshold(self) -> None:
        assert CSATerms(threshold=7.0).effective_threshold_post == 7.0
        assert CSATerms(threshold=7.0, threshold_post=2.0).effective_threshold_post == 2.0

    def test_collateral_required_shrinkage_shape(self) -> None:
        """The required balance is zero inside the threshold band, linear outside."""
        mtm = torch.tensor([-12.0, -5.0, 0.0, 5.0, 12.0], dtype=torch.float64)
        required = collateral_required(mtm, 5.0, 5.0)
        expected = torch.tensor([-7.0, 0.0, 0.0, 0.0, 7.0], dtype=torch.float64)
        assert torch.allclose(required, expected)

    def test_non_uniform_grid_is_rejected(self, csa_mtm: torch.Tensor) -> None:
        """Collateral lag logic assumes a uniform grid and must say so."""
        bad_times = torch.linspace(0.0, MATURITY, CSA_N_STEPS + 1, dtype=torch.float64).clone()
        bad_times[3] = bad_times[3] + 0.001
        with pytest.raises(ValueError, match="uniform"):
            collateral_balance(csa_mtm, bad_times, CSATerms(margin_period_of_risk=MPOR_10BD))


class TestGridUniformityTolerance:
    """The uniformity check must be horizon-relative, not step-relative.

    Regression cover for a real bug. The original bound was ``1e-6 * dt``, which
    rejects a perfectly good ``torch.linspace`` grid in float32 from about
    N=100 -- linspace rounding is ``O(eps * T)`` and independent of N, so
    dividing by ``dt = T/N`` injects a spurious factor of N. It was patched to
    ``1e-4 * dt``, which merely moves the failure to about N=2700.
    """

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("n_steps", [2, 32, 100, 252, 1_000, 2_520, 10_000])
    @pytest.mark.parametrize("horizon", [0.25, 1.0, 10.0, 30.0])
    def test_linspace_grids_are_always_accepted(
        self, dtype: torch.dtype, n_steps: int, horizon: float
    ) -> None:
        """No legitimate linspace grid may be rejected, at any N, T or dtype."""
        from src.xva.exposure import validate_uniform_grid

        times = torch.linspace(0.0, horizon, n_steps + 1, dtype=dtype)
        step = validate_uniform_grid(times)
        assert step == pytest.approx(horizon / n_steps, rel=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_genuinely_non_uniform_grid_is_still_rejected(
        self, dtype: torch.dtype
    ) -> None:
        """Loosening the bound must not blind it to a real violation."""
        from src.xva.exposure import validate_uniform_grid

        times = torch.linspace(0.0, 1.0, 253, dtype=dtype).clone()
        times[5] = times[5] + 1e-4
        with pytest.raises(ValueError, match="uniform time grid"):
            validate_uniform_grid(times)

    def test_the_old_step_relative_bound_would_have_failed(self) -> None:
        """Document why the formulation changed, not just that it did.

        Asserts the arithmetic directly: a float32 linspace at N=252 deviates by
        more than ``1e-6 * dt``, so the original bound rejected valid input.
        """
        times = torch.linspace(0.0, 1.0, 253, dtype=torch.float32)
        steps = times[1:] - times[:-1]
        deviation = float((steps - steps[0]).abs().max())
        step = float(steps[0])

        assert deviation > 1e-6 * step, "the old bound would have passed here"
        # ...and the horizon-relative bound comfortably accepts it.
        assert deviation < 64 * torch.finfo(torch.float32).eps * 1.0

    def test_rejects_decreasing_and_degenerate_grids(self) -> None:
        from src.xva.exposure import validate_uniform_grid

        with pytest.raises(ValueError, match="at least two points"):
            validate_uniform_grid(torch.tensor([0.0]))
        with pytest.raises(ValueError, match="strictly increasing"):
            validate_uniform_grid(torch.tensor([1.0, 0.5, 0.0]))
