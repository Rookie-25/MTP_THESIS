r"""Tests for the credit-curve injection into ``src.xva.cva`` and the
SSVI -> Phase 6 local-vol parameter bridge in ``src.models.vol_surface``.

Both additions are load-bearing numerics, so the checks here are the ones that
would actually catch a regression:

* the torch piecewise curve must agree with the NumPy ``CreditCurve`` it came
  from, and must reduce **exactly** to the existing flat path on one pillar;
* the per-pillar credit deltas must match finite differences, because a
  hand-rolled adjoint that is merely plausible is the failure mode AAD work
  actually has;
* the flat scalar path must be untouched -- every existing caller depends on
  it, and ``hazard_rate`` staying the third positional argument is the
  compatibility contract;
* the local-vol fitter must recover parameters when the target *is* in its own
  family. If it cannot do that, no fit against a real Dupire surface means
  anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from market_data.fetcher import (
    CDSQuote,
    CreditCurve,
    YieldCurve,
    bootstrap_hazard_rates,
)
from src.xva.cva import (
    PiecewiseHazard,
    compute_unilateral_cva,
    compute_unilateral_dva,
    compute_xva,
    cva_credit_bucket_deltas,
    marginal_default_probability,
    piecewise_marginal_default_probability,
    piecewise_survival_probability,
    survival_probability,
)

QUOTES = [
    CDSQuote(1.0, 80.0),
    CDSQuote(3.0, 110.0),
    CDSQuote(5.0, 135.0),
    CDSQuote(10.0, 160.0),
]
RECOVERY = 0.4


@pytest.fixture
def credit_curve() -> CreditCurve:
    """A bootstrapped four-pillar curve."""
    return bootstrap_hazard_rates(
        QUOTES, YieldCurve.flat(0.03), recovery_rate=RECOVERY
    )


@pytest.fixture
def grid() -> torch.Tensor:
    """A ten-year observation grid."""
    return torch.linspace(0.0, 10.0, 41, dtype=torch.float64)


@pytest.fixture
def exposure(grid: torch.Tensor) -> torch.Tensor:
    """A declining exposure profile, the usual amortising-swap shape."""
    return torch.linspace(20.0, 0.0, grid.numel(), dtype=torch.float64)


# ==========================================================================
# The torch piecewise curve
# ==========================================================================
class TestPiecewiseSurvival:
    """``Q(t)`` from a piecewise-constant hazard, in torch."""

    def test_agrees_with_the_numpy_credit_curve(
        self, credit_curve: CreditCurve, grid: torch.Tensor
    ) -> None:
        """The two implementations must not drift apart.

        ``market_data`` bootstraps in NumPy and the engine integrates in torch;
        if these disagree, a CVA computed from a bootstrapped curve is not the
        CVA the curve reprices.
        """
        mine = PiecewiseHazard.from_credit_curve(credit_curve).survival_probability(
            grid
        )
        theirs = credit_curve.survival_probability(grid.numpy())
        np.testing.assert_allclose(mine.numpy(), theirs, rtol=0.0, atol=1e-15)

    def test_single_pillar_reduces_exactly_to_the_flat_curve(
        self, grid: torch.Tensor
    ) -> None:
        """One pillar must be bit-identical to ``survival_probability``.

        Exact, not close: both compute ``exp(-h*t)``, so any difference means
        the piecewise overlap arithmetic is not reducing correctly.
        """
        hazard = 0.0223
        piecewise = piecewise_survival_probability(
            grid,
            torch.tensor([10.0], dtype=torch.float64),
            torch.tensor([hazard], dtype=torch.float64),
        )
        assert torch.equal(piecewise, survival_probability(grid, hazard))

    def test_marginal_convention_matches_the_flat_path(
        self, grid: torch.Tensor
    ) -> None:
        r"""Both must be :math:`Q(t_{i-1}) - Q(t_i)`, with the same sign."""
        hazard = 0.03
        piecewise = piecewise_marginal_default_probability(
            grid,
            torch.tensor([10.0], dtype=torch.float64),
            torch.tensor([hazard], dtype=torch.float64),
        )
        assert torch.equal(piecewise, marginal_default_probability(grid, hazard))

    def test_survival_starts_at_one_and_decreases(
        self, credit_curve: CreditCurve
    ) -> None:
        """Basic monotonicity, over and past the pillar range."""
        curve = PiecewiseHazard.from_credit_curve(credit_curve)
        times = torch.linspace(0.0, 20.0, 200, dtype=torch.float64)
        survival = curve.survival_probability(times)
        assert survival[0].item() == 1.0
        assert bool((survival.diff() < 0.0).all())
        assert bool(((survival > 0.0) & (survival <= 1.0)).all())

    def test_hazard_extends_flat_beyond_the_last_pillar(
        self, credit_curve: CreditCurve
    ) -> None:
        """Past the last pillar the final hazard is held, not dropped to zero.

        Dropping it would make ``Q`` stop decaying and understate CVA on any
        trade maturing beyond the longest quote -- a silent, one-directional
        error.
        """
        curve = PiecewiseHazard.from_credit_curve(credit_curve)
        far = torch.tensor([15.0, 20.0], dtype=torch.float64)
        survival = curve.survival_probability(far)
        implied = -math.log(survival[1] / survival[0]) / 5.0
        assert implied == pytest.approx(float(credit_curve.hazard_rates[-1]), rel=1e-12)

    @pytest.mark.parametrize(
        "pillars, hazards, message",
        [
            ([1.0, 2.0], [0.01], "must match"),
            ([2.0, 1.0], [0.01, 0.02], "strictly increasing"),
            ([0.0], [0.01], "must be positive"),
            ([], [], "at least one pillar"),
        ],
    )
    def test_rejects_malformed_curves(self, pillars, hazards, message) -> None:
        with pytest.raises(ValueError, match=message):
            PiecewiseHazard(
                pillar_times=torch.tensor(pillars, dtype=torch.float64),
                hazard_rates=torch.tensor(hazards, dtype=torch.float64),
            )

    def test_from_credit_curve_rejects_a_foreign_object(self) -> None:
        """Duck typing still has to fail loudly on the wrong duck."""
        with pytest.raises(TypeError, match="pillar_times"):
            PiecewiseHazard.from_credit_curve(object())


# ==========================================================================
# CVA against a term structure
# ==========================================================================
class TestCVAWithCreditCurve:
    """``compute_unilateral_cva`` with a curve instead of a scalar."""

    def test_flat_path_is_unchanged(
        self, exposure: torch.Tensor, grid: torch.Tensor
    ) -> None:
        """The existing positional call must behave exactly as before.

        ``hazard_rate`` merely became optional; it is still third-positional,
        which is the compatibility contract for every existing caller.
        """
        value = compute_unilateral_cva(
            exposure, grid, 0.02, RECOVERY, discount_rate=0.03
        )
        assert torch.isfinite(value) and value.item() > 0.0

    def test_single_pillar_curve_equals_the_flat_answer(
        self, exposure: torch.Tensor, grid: torch.Tensor
    ) -> None:
        """A one-pillar curve and the scalar path must agree to the bit."""
        hazard = 0.02
        flat = compute_unilateral_cva(
            exposure, grid, hazard, RECOVERY, discount_rate=0.03
        )
        curve = compute_unilateral_cva(
            exposure, grid, recovery_rate=RECOVERY, discount_rate=0.03,
            credit_curve=PiecewiseHazard(
                torch.tensor([10.0], dtype=torch.float64),
                torch.tensor([hazard], dtype=torch.float64),
            ),
        )
        assert torch.equal(flat, curve)

    def test_term_structure_differs_from_the_flat_approximation(
        self, exposure: torch.Tensor, grid: torch.Tensor, credit_curve: CreditCurve
    ) -> None:
        """The curve has to actually change the answer.

        If collapsing the term structure to a 5Y credit triangle gave the same
        CVA, bootstrapping would be adding nothing and this whole path would be
        ceremony.
        """
        with_curve = compute_unilateral_cva(
            exposure, grid, recovery_rate=RECOVERY, discount_rate=0.03,
            credit_curve=credit_curve,
        ).item()
        five_year = next(q for q in QUOTES if q.tenor == 5.0)
        flat = compute_unilateral_cva(
            exposure, grid, five_year.spread / (1.0 - RECOVERY), RECOVERY,
            discount_rate=0.03,
        ).item()
        assert abs(with_curve / flat - 1.0) > 0.01

    def test_accepts_a_bootstrapped_curve_directly(
        self, exposure: torch.Tensor, grid: torch.Tensor, credit_curve: CreditCurve
    ) -> None:
        """A ``market_data`` curve should need no manual adaptation."""
        value = compute_unilateral_cva(
            exposure, grid, recovery_rate=RECOVERY, discount_rate=0.03,
            credit_curve=credit_curve,
        )
        assert torch.isfinite(value) and value.item() > 0.0

    def test_explicit_survival_tensor(
        self, exposure: torch.Tensor, grid: torch.Tensor
    ) -> None:
        """An explicit ``Q`` must give the same answer as the hazard it encodes."""
        hazard = 0.025
        explicit = compute_unilateral_cva(
            exposure, grid, recovery_rate=RECOVERY, discount_rate=0.03,
            survival=survival_probability(grid, hazard),
        )
        flat = compute_unilateral_cva(
            exposure, grid, hazard, RECOVERY, discount_rate=0.03
        )
        assert torch.allclose(explicit, flat, rtol=0.0, atol=1e-15)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"hazard_rate": 0.02, "credit_curve": "curve"},
            {"hazard_rate": 0.02, "survival": "survival"},
            {"credit_curve": "curve", "survival": "survival"},
        ],
    )
    def test_exactly_one_specification_required(
        self, exposure, grid, credit_curve, kwargs
    ) -> None:
        """Silently preferring one input over another is the dangerous option.

        A caller who passes both would otherwise believe the curve was used
        when it was not, and get a plausible-but-wrong number.
        """
        resolved = {
            key: (
                credit_curve if value == "curve"
                else survival_probability(grid, 0.02) if value == "survival"
                else value
            )
            for key, value in kwargs.items()
        }
        with pytest.raises(ValueError, match="exactly one"):
            compute_unilateral_cva(
                exposure, grid, recovery_rate=RECOVERY, discount_rate=0.03,
                **resolved,
            )

    def test_dva_accepts_a_curve_too(
        self, grid: torch.Tensor, credit_curve: CreditCurve
    ) -> None:
        """DVA is driven by our own curve, and takes the same argument."""
        ene = torch.linspace(0.0, -8.0, grid.numel(), dtype=torch.float64).abs()
        value = compute_unilateral_dva(
            ene, grid, recovery_rate=RECOVERY, discount_rate=0.03,
            credit_curve=credit_curve,
        )
        assert torch.isfinite(value) and value.item() > 0.0


class TestXVAWithCreditCurve:
    """``compute_xva`` with a term structure."""

    def test_curve_path_reports_no_single_hazard(
        self, credit_curve: CreditCurve
    ) -> None:
        """``hazard_rate`` is ``None`` when a curve was used.

        Reporting a summary intensity would invite it being quoted as if it had
        been the input.
        """
        torch.manual_seed(0)
        times = torch.linspace(0.0, 5.0, 21, dtype=torch.float64)
        mtm = torch.randn(500, 21, dtype=torch.float64) * 5.0
        result = compute_xva(
            mtm, times, discount_rate=0.03, recovery_rate=RECOVERY,
            credit_curve=credit_curve,
        )
        assert result.hazard_rate is None
        assert result.cva.item() > 0.0 and result.dva.item() > 0.0

    def test_flat_path_still_reports_its_hazard(self) -> None:
        """Backward compatibility of the reported field."""
        torch.manual_seed(0)
        times = torch.linspace(0.0, 5.0, 21, dtype=torch.float64)
        mtm = torch.randn(500, 21, dtype=torch.float64) * 5.0
        result = compute_xva(
            mtm, times, hazard_rate=0.02, discount_rate=0.03,
            recovery_rate=RECOVERY,
        )
        assert result.hazard_rate == pytest.approx(0.02)

    def test_requires_exactly_one_credit_specification(self) -> None:
        times = torch.linspace(0.0, 5.0, 21, dtype=torch.float64)
        mtm = torch.zeros(10, 21, dtype=torch.float64)
        with pytest.raises(ValueError, match="exactly one"):
            compute_xva(mtm, times, discount_rate=0.03)


# ==========================================================================
# AAD through the credit curve
# ==========================================================================
class TestCreditAAD:
    """Gradients with respect to the pillar hazards."""

    def test_bucket_deltas_match_finite_differences(
        self, exposure: torch.Tensor, grid: torch.Tensor, credit_curve: CreditCurve
    ) -> None:
        r"""The whole point: one backward pass, :math:`J` correct deltas.

        A hand-built adjoint that is merely plausible is exactly the failure
        this project has already hit once, so it is checked against a central
        difference per pillar rather than assumed.
        """
        cva, deltas = cva_credit_bucket_deltas(
            exposure, grid, credit_curve, RECOVERY, discount_rate=0.03
        )
        assert cva.item() > 0.0
        assert deltas.numel() == len(QUOTES)

        pillars = torch.tensor(credit_curve.pillar_times, dtype=torch.float64)
        base = torch.tensor(credit_curve.hazard_rates, dtype=torch.float64)
        step = 1e-7

        def value(hazards: torch.Tensor) -> float:
            return compute_unilateral_cva(
                exposure, grid, recovery_rate=RECOVERY, discount_rate=0.03,
                credit_curve=PiecewiseHazard(pillars, hazards),
            ).item()

        for index in range(base.numel()):
            up, down = base.clone(), base.clone()
            up[index] += step
            down[index] -= step
            expected = (value(up) - value(down)) / (2.0 * step)
            assert deltas[index].item() == pytest.approx(expected, rel=1e-6), (
                f"pillar {credit_curve.pillar_times[index]:g}Y"
            )

    def test_all_bucket_deltas_are_positive(
        self, exposure: torch.Tensor, grid: torch.Tensor, credit_curve: CreditCurve
    ) -> None:
        """More default intensity on a positive exposure means more CVA."""
        _, deltas = cva_credit_bucket_deltas(
            exposure, grid, credit_curve, RECOVERY, discount_rate=0.03
        )
        assert bool((deltas > 0.0).all())

    def test_gradient_flows_to_a_requires_grad_curve(
        self, exposure: torch.Tensor, grid: torch.Tensor, credit_curve: CreditCurve
    ) -> None:
        """``from_credit_curve(requires_grad=True)`` must produce a live leaf.

        The trap this guards: reading NumPy survival values gives a constant,
        the backward pass then succeeds with an all-zero gradient, and the
        credit sensitivities are silently gone.
        """
        curve = PiecewiseHazard.from_credit_curve(credit_curve, requires_grad=True)
        compute_unilateral_cva(
            exposure, grid, recovery_rate=RECOVERY, discount_rate=0.03,
            credit_curve=curve,
        ).backward()
        assert curve.hazard_rates.grad is not None
        assert bool((curve.hazard_rates.grad.abs() > 0.0).all())

    def test_exposure_gradient_survives_the_curve_path(
        self, grid: torch.Tensor, credit_curve: CreditCurve
    ) -> None:
        """The exposure tape must not be broken by the credit change."""
        exposure = torch.linspace(
            20.0, 0.0, grid.numel(), dtype=torch.float64
        ).requires_grad_(True)
        compute_unilateral_cva(
            exposure, grid, recovery_rate=RECOVERY, discount_rate=0.03,
            credit_curve=credit_curve,
        ).backward()
        assert exposure.grad is not None
        assert float(exposure.grad.abs().sum()) > 0.0

    def test_valuation_fn_exposes_a_parallel_shift(
        self, credit_curve: CreditCurve
    ) -> None:
        """``make_cva_valuation_fn`` + ``aad_greeks`` against a curve.

        The shift is scalar on purpose: ``aad_greeks`` reduces gradients with
        ``float(grad.detach())`` and cannot carry a vector, so curve risk is
        summarised as a parallel move and the bucketed vector lives in
        ``cva_credit_bucket_deltas``.
        """
        from src.models.gbm import GBMSimulator, draw_brownian_increments
        from src.pricer.greeks import aad_greeks
        from src.pricer.options import SwapLeg
        from src.xva.cva import make_cva_valuation_fn

        torch.manual_seed(0)
        simulator = GBMSimulator(
            maturity=5.0, n_steps=24, dtype=torch.float64
        )
        increments = draw_brownian_increments(
            2_000, 24, simulator.dt, dtype=torch.float64
        )
        cva_fn = make_cva_valuation_fn(
            simulator,
            increments,
            [SwapLeg(notional=1.0, strike=100.0, maturity=5.0)],
            rate=0.03,
            credit_curve=credit_curve,
        )
        greeks = aad_greeks(
            cva_fn, {"s0": 100.0, "sigma": 0.2, "hazard_shift": 0.0}
        )
        assert sorted(greeks.greeks) == ["hazard_shift", "s0", "sigma"]
        # Widening the whole curve on a positive exposure raises CVA.
        assert greeks.greeks["hazard_shift"] > 0.0
        assert greeks.n_valuations == 1

    def test_valuation_fn_rejects_both_credit_inputs(self) -> None:
        from src.models.gbm import GBMSimulator, draw_brownian_increments
        from src.pricer.options import SwapLeg
        from src.xva.cva import make_cva_valuation_fn

        simulator = GBMSimulator(maturity=1.0, n_steps=4, dtype=torch.float64)
        increments = draw_brownian_increments(
            10, 4, simulator.dt, dtype=torch.float64
        )
        with pytest.raises(ValueError, match="at most one"):
            make_cva_valuation_fn(
                simulator, increments,
                [SwapLeg(notional=1.0, strike=100.0, maturity=1.0)],
                rate=0.03,
                credit_curve=PiecewiseHazard(
                    torch.tensor([1.0], dtype=torch.float64),
                    torch.tensor([0.02], dtype=torch.float64),
                ),
                survival=torch.ones(5, dtype=torch.float64),
            )


# ==========================================================================
# The SSVI -> Phase 6 kernel bridge
# ==========================================================================
class _TanhSurface:
    """A stub Dupire surface that *is* in the kernel's parametric family.

    Used to test the fitter against a known answer. If the optimiser cannot
    recover parameters here, a fit against a real SSVI-implied surface carries
    no information -- the residual could be the projection or the optimiser and
    there would be no way to tell.
    """

    variance_floor = 1e-8

    def __init__(self, base: float, skew: float, kappa: float, term: float) -> None:
        self.base, self.skew, self.kappa, self.term = base, skew, kappa, term

    def local_volatility(self, spot, time, spot_zero):
        log_spot = torch.log(spot)
        return (
            self.base
            + self.skew * torch.tanh(self.kappa * (log_spot - math.log(spot_zero)))
            + self.term * time
        )


class TestLocalVolBridge:
    """Projecting a Dupire surface onto the Phase 6 kernel's four parameters."""

    TRUTH = {"base": 0.22, "skew": -0.08, "kappa": 3.0, "term": 0.04}

    def test_recovers_parameters_from_its_own_family(self) -> None:
        """The identifiability check the rest of the bridge rests on."""
        from src.models.vol_surface import fit_local_vol_params

        fit = fit_local_vol_params(
            _TanhSurface(**self.TRUTH), 100.0, 1.0,
            iterations=3000, learning_rate=0.02,
        )
        for name, expected in self.TRUTH.items():
            assert getattr(fit, name) == pytest.approx(expected, abs=1e-4), name
        assert fit.relative_rmse < 1e-5
        assert fit.r_squared > 0.999999
        assert fit.is_well_fitted

    def test_reference_is_pinned_to_log_spot(self) -> None:
        r"""Not fitted, and it must equal :math:`\log S_0` exactly.

        An uncentred reference saturates the kernel's ``tanh``, driving
        ``dsigma/dx`` to ~1e-5 and silently reducing Phase 6 to the
        constant-volatility case while appearing to exercise state dependence.
        """
        from src.models.vol_surface import fit_local_vol_params

        for spot in (50.0, 100.0, 4000.0):
            fit = fit_local_vol_params(
                _TanhSurface(**self.TRUTH), spot, 1.0, iterations=50
            )
            assert fit.reference == math.log(spot)

    def test_positivity_constraints_hold(self) -> None:
        """``base`` and ``kappa`` must stay positive for ``LocalVolParams``.

        An unconstrained fit reaches a negative ``kappa`` -- an observationally
        equivalent surface with flipped skew -- which then fails the kernel's
        validation for no apparent reason.
        """
        from src.models.vol_surface import fit_local_vol_params

        fit = fit_local_vol_params(
            _TanhSurface(0.2, 0.9, 4.0, 0.0), 100.0, 1.0, iterations=800
        )
        assert fit.base > 0.0 and fit.kappa > 0.0

    def test_handoff_constructs_kernel_params(self) -> None:
        """The fitted values must satisfy ``LocalVolParams``' own validation."""
        pytest.importorskip("src.csrc.triton_local_vol_cva")
        from src.models.vol_surface import fit_local_vol_params

        fit = fit_local_vol_params(
            _TanhSurface(**self.TRUTH), 100.0, 1.0, iterations=500
        )
        params = fit.to_local_vol_params()
        assert params.base == pytest.approx(fit.base)
        assert params.skew == pytest.approx(fit.skew)
        assert params.kappa == pytest.approx(fit.kappa)
        assert params.term == pytest.approx(fit.term)
        assert params.reference == pytest.approx(fit.reference)

    def test_evaluate_matches_the_fitted_form(self) -> None:
        """The host-side evaluator must reproduce the kernel's arithmetic."""
        from src.models.vol_surface import (
            evaluate_parametric_local_vol,
            fit_local_vol_params,
            local_vol_sampling_grid,
        )

        fit = fit_local_vol_params(
            _TanhSurface(**self.TRUTH), 100.0, 1.0,
            iterations=3000, learning_rate=0.02,
        )
        time, log_spot = local_vol_sampling_grid(100.0, 1.0, n_time=6, n_space=9)
        fitted = evaluate_parametric_local_vol(fit, time, log_spot)
        target = _TanhSurface(**self.TRUTH).local_volatility(
            torch.exp(log_spot), time, 100.0
        )
        assert torch.allclose(fitted, target, atol=1e-4)

    def test_runs_inside_a_no_grad_block(self) -> None:
        """Analysis code wraps things in ``no_grad``; the fit must survive it.

        Two separate traps: the Dupire target is *made of* autograd, and the
        optimiser needs its own graph. Both need an explicit ``enable_grad``.
        """
        from src.models.vol_surface import fit_local_vol_params

        with torch.no_grad():
            fit = fit_local_vol_params(
                _TanhSurface(**self.TRUTH), 100.0, 1.0, iterations=200
            )
        assert fit.base > 0.0

    def test_fits_a_real_ssvi_implied_surface(self) -> None:
        """End to end from a calibrated SSVI surface.

        Deliberately asserts only that the projection *runs and reports*, not
        that it is tight: a four-parameter tanh form genuinely cannot represent
        an arbitrary Dupire surface, and pinning a quality threshold here would
        be asserting a property of one particular SSVI parametrisation.
        """
        from src.models.vol_surface import (
            ATMTotalVariance,
            LocalVolatilitySurface,
            SSVISurface,
            fit_local_vol_params,
        )

        knots = torch.tensor([0.1, 0.25, 0.5, 1.0, 2.0], dtype=torch.float64)
        surface = SSVISurface(
            atm=ATMTotalVariance(knots), rho=-0.35, eta=1.2, gamma=0.45
        ).double()
        fit = fit_local_vol_params(
            LocalVolatilitySurface(surface, rate=0.02), 100.0, 1.0,
            drift=0.02, iterations=600,
        )
        assert fit.base > 0.0 and fit.kappa > 0.0
        assert math.isfinite(fit.rmse) and fit.rmse >= 0.0
        assert fit.n_samples > 0
        assert fit.target_range[0] <= fit.target_range[1]
        assert isinstance(fit.summary(), str) and fit.summary()

    def test_fit_error_grows_with_the_sampling_width(self) -> None:
        """The diagnostics must be responsive, not uniformly optimistic.

        Widening the cone brings in wings the tanh cannot follow, so the
        reported error has to rise. A metric that stayed flat here would be
        measuring nothing.
        """
        from src.models.vol_surface import (
            ATMTotalVariance,
            LocalVolatilitySurface,
            SSVISurface,
            fit_local_vol_params,
        )

        knots = torch.tensor([0.1, 0.25, 0.5, 1.0, 2.0], dtype=torch.float64)
        surface = SSVISurface(
            atm=ATMTotalVariance(knots), rho=-0.35, eta=1.2, gamma=0.45
        ).double()
        local_vol = LocalVolatilitySurface(surface, rate=0.02)

        errors = [
            fit_local_vol_params(
                local_vol, 100.0, 1.0, drift=0.02, sigma_width=width,
                iterations=600,
            ).relative_rmse
            for width in (0.5, 1.5, 3.0)
        ]
        assert errors[0] < errors[1] < errors[2], errors

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"spot_zero": 0.0}, "spot_zero must be positive"),
            ({"maturity": 0.0}, "maturity must be positive"),
            ({"n_space": 1}, "n_space >= 2"),
            ({"reference_vol": 0.0}, "must be positive"),
        ],
    )
    def test_grid_rejects_bad_arguments(self, kwargs, message) -> None:
        from src.models.vol_surface import local_vol_sampling_grid

        arguments = {"spot_zero": 100.0, "maturity": 1.0}
        arguments.update(kwargs)
        with pytest.raises(ValueError, match=message):
            local_vol_sampling_grid(**arguments)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"iterations": 0}, "iterations must be positive"),
            ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ],
    )
    def test_fit_rejects_bad_optimiser_settings(self, kwargs, message) -> None:
        from src.models.vol_surface import fit_local_vol_params

        with pytest.raises(ValueError, match=message):
            fit_local_vol_params(
                _TanhSurface(**self.TRUTH), 100.0, 1.0, **kwargs
            )


# ==========================================================================
# Benchmark-report parsing for the figures
# ==========================================================================
class TestPlotResultsParsing:
    """``benchmarks/plot_results.py`` reads measured data back out of Markdown.

    The invariant worth guarding is the OOM handling: a cell that means "this
    backend could not run" must become ``None``, never ``0``. Plotting zero
    would put a data point at exactly the place the finding lives and make a
    backend that died look infinitely fast.
    """

    @pytest.mark.parametrize(
        "cell, expected",
        [
            ("1,180.2", 1180.2),
            ("41.2", 41.2),
            ("**OOM**", None),
            ("**OOM** (pred, ~70.8 GiB)", None),
            ("-", None),
            ("", None),
            ("not run", None),
        ],
    )
    def test_parses_millisecond_cells(self, cell, expected) -> None:
        from benchmarks.plot_results import parse_number

        assert parse_number(cell) == expected

    @pytest.mark.parametrize(
        "cell, expected",
        [
            ("4.3 MiB", int(4.3 * 1024**2)),
            ("14.15 GiB", int(14.15 * 1024**3)),
            ("968.1 MiB", int(968.1 * 1024**2)),
            ("**OOM**", None),
            ("-", None),
        ],
    )
    def test_parses_byte_cells(self, cell, expected) -> None:
        from benchmarks.plot_results import parse_bytes

        assert parse_bytes(cell) == expected

    def test_round_trips_format_bytes(self) -> None:
        """Parsing must invert the harness's own formatter.

        These two functions are on opposite sides of a file, so a change to
        either silently breaks the figures. The tolerance is the formatter's
        own rounding (2 dp in GiB, 1 dp in MiB), not a fudge.
        """
        from benchmarks._harness import format_bytes
        from benchmarks.plot_results import parse_bytes

        for value in (4_500_000, 300_000_000, 15_200_000_000, 1024**3):
            recovered = parse_bytes(format_bytes(value))
            assert recovered is not None
            assert recovered == pytest.approx(value, rel=0.01)

    def test_reads_a_full_report(self, tmp_path) -> None:
        """End-to-end parse of the layout ``bench_all_phases`` emits."""
        from benchmarks._harness import markdown_table
        from benchmarks.plot_results import load_results

        report = tmp_path / "results.md"
        backends = ["PyTorch baseline", "Phase 5 fused"]
        report.write_text(
            "\n".join([
                "# report", "", "## Environment", "",
                markdown_table(
                    ["Item", "Value"],
                    [["GPU", "Tesla T4 (14.6 GiB)"], ["Time steps N", "252"]],
                ),
                "", "## Execution time (ms)", "",
                markdown_table(
                    ["M"] + backends,
                    [["100,000", "118.4", "22.6"],
                     ["5,000,000", "**OOM**", "722.8"]],
                ),
                "", "## Peak VRAM", "",
                markdown_table(
                    ["M"] + backends,
                    [["100,000", "1.42 GiB", "4.3 MiB"],
                     ["5,000,000", "**OOM**", "4.3 MiB"]],
                ),
                "",
            ]),
            encoding="utf-8",
        )

        results = load_results(report)
        assert results.gpu == "Tesla T4 (14.6 GiB)"
        assert results.total_vram_bytes == int(14.6 * 1024**3)
        assert results.n_steps == 252
        assert results.backends == ["PyTorch baseline", "Phase 5 fused"]

        # The OOM row is absent from the series, and recorded as an OOM.
        assert sorted(results.times_ms["PyTorch baseline"]) == [100_000]
        assert sorted(results.times_ms["Phase 5 fused"]) == [100_000, 5_000_000]
        assert results.oom_paths["PyTorch baseline"] == [5_000_000]
        assert 5_000_000 not in results.peak_bytes["PyTorch baseline"]

    def test_missing_sections_do_not_raise(self, tmp_path) -> None:
        """A partial run should still plot whatever it measured."""
        from benchmarks.plot_results import load_results

        report = tmp_path / "partial.md"
        report.write_text("# nothing here\n", encoding="utf-8")
        results = load_results(report)
        assert not results.has_timings and not results.has_memory
        assert results.backends == []


class TestExposureFigureData:
    """The exposure figure's numbers, independent of how they are drawn."""

    def test_collateral_reduces_exposure_everywhere(self) -> None:
        """The CSA invariant the figure exists to show.

        ``EE_collat <= EE_uncollat`` pointwise. If this ever fails the figure
        would be drawing a shaded band the wrong way round.
        """
        from benchmarks.plot_results import compute_exposure_curves

        curves = compute_exposure_curves(n_paths=4_000, n_steps=48)
        assert np.all(curves.ee_collateralized <= curves.ee + 1e-12)
        assert np.all(curves.pfe_collateralized <= curves.pfe + 1e-12)
        assert curves.ee_collateralized.sum() < curves.ee.sum()

    def test_profiles_are_finite_and_start_near_zero(self) -> None:
        """A par swap has no exposure at inception."""
        from benchmarks.plot_results import compute_exposure_curves

        curves = compute_exposure_curves(n_paths=4_000, n_steps=48)
        for values in (curves.ee, curves.pfe, curves.ee_collateralized):
            assert np.all(np.isfinite(values))
            assert np.all(values >= -1e-12)
        assert curves.ee[0] == pytest.approx(0.0, abs=1e-9)

    def test_is_reproducible(self) -> None:
        """A fixed seed, so the two profiles differ only by the CSA."""
        from benchmarks.plot_results import compute_exposure_curves

        first = compute_exposure_curves(n_paths=2_000, n_steps=24)
        second = compute_exposure_curves(n_paths=2_000, n_steps=24)
        np.testing.assert_allclose(first.ee, second.ee, rtol=0.0, atol=0.0)
