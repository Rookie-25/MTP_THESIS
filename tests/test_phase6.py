r"""Phase 6 suite: arbitrage-free local volatility and the non-linear adjoint.

Every test here runs on CPU. That is deliberate: Phase 6 introduces two
genuinely new mathematical objects -- an arbitrage-constrained surface and a
*sequential* adjoint -- and both must be nailed down before any Triton is
written, because a kernel cannot be debugged against a specification that is
itself unverified.

The four things being established
=================================
1. **The butterfly function is the right one.** ``g(k)`` is validated by
   integrating the implied density it induces. This catches the factor-of-2
   error in the first term, which produces a plausible-looking but non-
   normalising "density".
2. **The two arbitrage conditions are exactly the well-posedness conditions for
   Dupire.** :math:`\sigma_{LV}^2 = \partial_T w / g`, so calendar violation
   makes the numerator negative and butterfly violation makes the denominator
   vanish.
3. **The sequential adjoint is correct** -- checked against ``torch.autograd``,
   and against a :math:`\sqrt{N}`-checkpointed variant that must agree.
4. **The Phase 3-5 suffix-sum shortcut is measurably wrong** here, and the error
   is a bias rather than noise.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.local_vol_paths import (
    checkpoint_schedule,
    checkpointed_local_vol_adjoint,
    local_vol_adjoint,
    simulate_local_vol_paths,
    suffix_sum_adjoint_incorrect,
)
from src.models.vol_surface import (
    ATMTotalVariance,
    ArbitragePenalty,
    LocalVolatilitySurface,
    SSVISurface,
    butterfly_g,
    calibrate_surface,
    implied_density,
)

torch.set_default_dtype(torch.float64)

SPOT_ZERO = 100.0
DRIFT = 0.02
MATURITY = 1.0


def _surface(rho: float = -0.5, eta: float = 1.2, gamma: float = 0.4) -> SSVISurface:
    """A well-behaved SSVI surface for reuse across tests."""
    maturities = torch.tensor([0.25, 0.5, 1.0, 2.0])
    atm = ATMTotalVariance(maturities, initial_total_variance=0.04 * maturities)
    return SSVISurface(atm, rho=rho, eta=eta, gamma=gamma)


def _slice_derivatives(surface: SSVISurface, k: torch.Tensor, t: torch.Tensor):
    """Return ``(w, dw/dk, d2w/dk2)`` at fixed maturity via autograd."""
    leaf = k.detach().clone().requires_grad_(True)
    w = surface(leaf, t)
    (first,) = torch.autograd.grad(w.sum(), leaf, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), leaf, create_graph=True)
    return w.detach(), first.detach(), second.detach()


# ==========================================================================
# 1. The butterfly function
# ==========================================================================
class TestButterflyFunction:
    """``g(k)`` must be the function whose induced density normalises."""

    def test_flat_smile_gives_g_identically_one(self) -> None:
        """Black-Scholes has no skew or curvature, so ``g == 1`` exactly."""
        k = torch.linspace(-3.0, 3.0, 501)
        w = torch.full_like(k, 0.04)
        zero = torch.zeros_like(k)
        g = butterfly_g(k, w, zero, zero)
        assert torch.allclose(g, torch.ones_like(g), atol=0.0, rtol=0.0)

    def test_flat_smile_density_normalises_exactly(self) -> None:
        k = torch.linspace(-8.0, 8.0, 160001)
        w = torch.full_like(k, 0.04)
        zero = torch.zeros_like(k)
        total = float(implied_density(k, w, zero, zero).sum() * (k[1] - k[0]))
        assert math.isclose(total, 1.0, abs_tol=1e-8)

    def test_skewed_surface_density_normalises(self) -> None:
        """The decisive check: a curved, skewed slice must still integrate to 1."""
        surface = _surface(rho=-0.7, eta=1.5, gamma=0.5)
        k = torch.linspace(-8.0, 8.0, 160001)
        t = torch.tensor(1.0)
        w, first, second = _slice_derivatives(surface, k, t)
        total = float(implied_density(k, w, first, second).sum() * (k[1] - k[0]))
        assert math.isclose(total, 1.0, abs_tol=1e-5), (
            f"integral of implied density = {total:.8f}"
        )

    def test_the_factor_of_two_matters(self) -> None:
        r"""Using :math:`k w'/w` instead of :math:`k w'/(2w)` breaks normalisation.

        The variant is a plausible transcription of Gatheral-Jacquier and yields
        a smooth, positive, entirely reasonable-looking function. It is simply
        not the butterfly function, and the induced measure has the wrong mass.
        Pinned here so the correct form cannot be "simplified" later.
        """
        surface = _surface(rho=-0.7, eta=1.5, gamma=0.5)
        k = torch.linspace(-8.0, 8.0, 160001)
        t = torch.tensor(1.0)
        w, first, second = _slice_derivatives(surface, k, t)
        dk = float(k[1] - k[0])

        correct = float(implied_density(k, w, first, second).sum() * dk)

        # The wrong variant, reconstructed explicitly.
        wrong_g = (
            (1.0 - k * first / w) ** 2
            - 0.25 * (1.0 / w + 0.25) * first**2
            + 0.5 * second
        )
        d_minus = -k / torch.sqrt(w) - 0.5 * torch.sqrt(w)
        wrong = float(
            (
                wrong_g
                / torch.sqrt(2.0 * math.pi * w)
                * torch.exp(-0.5 * d_minus**2)
            ).sum()
            * dk
        )

        assert math.isclose(correct, 1.0, abs_tol=1e-5)
        assert not math.isclose(wrong, 1.0, abs_tol=1e-3), (
            f"the variant integrated to {wrong:.6f}, which is suspiciously close "
            "to 1 -- re-check that the test is exercising a real difference"
        )

    def test_well_behaved_surface_is_butterfly_arbitrage_free(self) -> None:
        surface = _surface()
        k = torch.linspace(-1.5, 1.5, 301)
        for maturity in (0.25, 0.5, 1.0, 2.0):
            w, first, second = _slice_derivatives(
                surface, k, torch.tensor(maturity)
            )
            g = butterfly_g(k, w, first, second)
            assert torch.all(g > 0.0), (
                f"T={maturity}: min g = {float(g.min()):.6f}"
            )


# ==========================================================================
# 2. Structural guarantees and Dupire
# ==========================================================================
class TestStructuralGuarantees:
    """Constraints discharged by construction rather than by penalty."""

    def test_atm_variance_is_monotone_for_any_parameters(self) -> None:
        """Random parameter values must still give a non-decreasing term structure."""
        maturities = torch.tensor([0.1, 0.3, 0.7, 1.5, 3.0])
        atm = ATMTotalVariance(maturities)
        generator = torch.Generator().manual_seed(5)
        for _ in range(20):
            with torch.no_grad():
                atm.raw_increments.copy_(
                    torch.randn(maturities.shape, generator=generator) * 5.0
                )
            knots = atm.knot_values()
            assert torch.all(torch.diff(knots) >= 0.0), knots
            assert torch.all(knots > 0.0)

    def test_atm_interpolation_is_monotone_off_grid(self) -> None:
        maturities = torch.tensor([0.25, 0.5, 1.0, 2.0])
        atm = ATMTotalVariance(maturities)
        fine = torch.linspace(0.25, 2.0, 200)
        values = atm(fine)
        assert torch.all(torch.diff(values) >= -1e-14)

    def test_ssvi_parameters_stay_in_feasible_sets(self) -> None:
        surface = _surface()
        generator = torch.Generator().manual_seed(7)
        for _ in range(50):
            with torch.no_grad():
                surface.raw_rho.copy_(torch.randn((), generator=generator) * 10)
                surface.raw_eta.copy_(torch.randn((), generator=generator) * 10)
                surface.raw_gamma.copy_(torch.randn((), generator=generator) * 10)
            # `.detach()` before the scalar conversion: rho/eta/gamma are
            # computed from nn.Parameters, so they carry requires_grad=True even
            # though nothing here needs a gradient. Converting a graph-attached
            # tensor to a Python float warns on newer PyTorch builds.
            assert -1.0 < surface.rho.detach().item() < 1.0
            assert surface.eta.detach().item() > 0.0
            assert 0.0 < surface.gamma.detach().item() < 1.0

    def test_calendar_slope_is_non_negative_on_a_calibrated_surface(self) -> None:
        surface = _surface()
        k = torch.linspace(-1.0, 1.0, 41).reshape(1, -1)
        t = torch.linspace(0.25, 2.0, 21).reshape(-1, 1)
        leaf = t.expand(-1, k.numel()).detach().clone().requires_grad_(True)
        w = surface(k.expand(t.numel(), -1), leaf)
        (slope,) = torch.autograd.grad(w.sum(), leaf)
        assert torch.all(slope >= -1e-12), f"min dw/dT = {float(slope.min()):.3e}"

    def test_local_variance_equals_calendar_over_butterfly(self) -> None:
        r"""The identity :math:`\sigma_{LV}^2 = \partial_T w / g` must hold."""
        surface = _surface()
        local = LocalVolatilitySurface(surface, rate=0.03, dividend_yield=0.01)

        k = torch.tensor([-0.3, -0.1, 0.0, 0.15, 0.4])
        t = torch.tensor(0.8)

        variance = local.local_variance_from_coordinates(k, t)

        # Independent reconstruction from the two ingredients.
        k_leaf = k.detach().clone().requires_grad_(True)
        t_leaf = t.detach().clone().requires_grad_(True)
        broadcast_k, broadcast_t = torch.broadcast_tensors(k_leaf, t_leaf)
        w = surface(broadcast_k, broadcast_t)
        (first,) = torch.autograd.grad(w.sum(), k_leaf, create_graph=True)
        (second,) = torch.autograd.grad(first.sum(), k_leaf, create_graph=True)
        (slope,) = torch.autograd.grad(w.sum(), t_leaf, create_graph=True)
        g = butterfly_g(broadcast_k, w, first, second)
        expected = slope.expand_as(g) / g

        assert torch.allclose(variance, expected, rtol=1e-10), (
            f"max |diff| = {float((variance - expected).abs().max()):.3e}"
        )

    def test_forward_mapping_uses_the_forward_not_the_spot(self) -> None:
        r"""``log_moneyness`` must be :math:`\log(S_t/F_t)`, not :math:`\log(S_t/S_0)`.

        At :math:`t=0` the two agree, so a test at inception cannot tell them
        apart. At :math:`t=2` with a 2% net drift they differ by 0.04 in
        log-moneyness -- a materially different point on the smile.
        """
        surface = _surface()
        local = LocalVolatilitySurface(surface, rate=0.05, dividend_yield=0.01)
        spot = torch.tensor([100.0])

        at_inception = local.log_moneyness(spot, torch.tensor(0.0), SPOT_ZERO)
        assert torch.allclose(at_inception, torch.zeros_like(at_inception))

        later = local.log_moneyness(spot, torch.tensor(2.0), SPOT_ZERO)
        assert math.isclose(float(later), -(0.05 - 0.01) * 2.0, abs_tol=1e-12)


# ==========================================================================
# 3. Arbitrage penalties
# ==========================================================================
class TestArbitragePenalty:
    """The penalty must be finite, smooth, and actually push toward feasibility."""

    def test_hinge_is_finite_and_differentiable_when_infeasible(self) -> None:
        """The whole point of the hinge: usable from an infeasible start."""
        penalty = ArbitragePenalty(mode="hinge")
        violated = torch.tensor([-5.0, -1.0, 0.0, 1.0], requires_grad=True)
        value = penalty._apply_to(violated)
        value.backward()
        assert torch.isfinite(value)
        assert torch.all(torch.isfinite(violated.grad))
        # Gradient must point toward increasing the slack.
        assert torch.all(violated.grad <= 0.0)

    def test_hinge_decreases_as_slack_increases(self) -> None:
        penalty = ArbitragePenalty(mode="hinge")
        values = [
            float(penalty._apply_to(torch.full((4,), slack)))
            for slack in (-1.0, -0.1, 0.0, 0.1, 1.0)
        ]
        assert values == sorted(values, reverse=True), values

    def test_barrier_blows_up_at_the_boundary(self) -> None:
        penalty = ArbitragePenalty(mode="barrier", margin=0.0)
        near = float(penalty._apply_to(torch.full((4,), 1e-6)))
        far = float(penalty._apply_to(torch.full((4,), 1.0)))
        assert near > far
        assert near > 10.0

    def test_penalty_reports_components_separately(self) -> None:
        """A calibration log needs to know WHICH constraint is binding."""
        surface = _surface()
        penalty = ArbitragePenalty()
        k = torch.linspace(-1.0, 1.0, 21)
        t = torch.linspace(0.25, 2.0, 7)
        terms = penalty(surface, k, t)
        for component in (
            terms.calendar, terms.butterfly,
            terms.ssvi_linear, terms.ssvi_quadratic,
        ):
            assert torch.isfinite(component)
        assert torch.isfinite(terms.total)

    def test_penalty_gradient_reaches_surface_parameters(self) -> None:
        surface = _surface()
        penalty = ArbitragePenalty()
        k = torch.linspace(-1.0, 1.0, 21)
        t = torch.linspace(0.25, 2.0, 7)
        penalty(surface, k, t).total.backward()
        for name, parameter in surface.named_parameters():
            assert parameter.grad is not None, f"{name} got no gradient"
            assert torch.all(torch.isfinite(parameter.grad)), name

    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError, match="weight must be positive"):
            ArbitragePenalty(weight=0.0)
        with pytest.raises(ValueError, match="sharpness must be positive"):
            ArbitragePenalty(sharpness=-1.0)
        with pytest.raises(ValueError, match="mode must be"):
            ArbitragePenalty(mode="quadratic")


class TestCalibration:
    """End-to-end: fit a surface and keep it arbitrage-free while doing so."""

    def test_recovers_a_surface_generated_by_itself(self) -> None:
        """Sanity: SSVI must be able to fit data drawn from SSVI."""
        truth = _surface(rho=-0.55, eta=1.3, gamma=0.45)
        k = torch.linspace(-0.6, 0.6, 13)
        t = torch.tensor([0.25, 0.5, 1.0, 2.0])
        grid_k = k.reshape(1, -1).expand(4, -1).reshape(-1)
        grid_t = t.reshape(-1, 1).expand(-1, 13).reshape(-1)
        with torch.no_grad():
            target_vol = torch.sqrt(truth(grid_k, grid_t) / grid_t)

        fitted = _surface(rho=0.0, eta=1.0, gamma=0.5)
        result = calibrate_surface(
            fitted, grid_k, grid_t, target_vol, iterations=400, learning_rate=8e-2
        )
        assert result.final_rmse < 5e-3, (
            f"implied-vol RMSE {result.final_rmse:.5f} in vol points"
        )

    def test_calibrated_surface_stays_arbitrage_free(self) -> None:
        truth = _surface(rho=-0.55, eta=1.3, gamma=0.45)
        k = torch.linspace(-0.6, 0.6, 13)
        t = torch.tensor([0.25, 0.5, 1.0, 2.0])
        grid_k = k.reshape(1, -1).expand(4, -1).reshape(-1)
        grid_t = t.reshape(-1, 1).expand(-1, 13).reshape(-1)
        with torch.no_grad():
            target_vol = torch.sqrt(truth(grid_k, grid_t) / grid_t)

        fitted = _surface(rho=0.0, eta=1.0, gamma=0.5)
        calibrate_surface(fitted, grid_k, grid_t, target_vol, iterations=300)

        local = LocalVolatilitySurface(fitted)
        diagnostics = local.diagnostics(
            torch.linspace(-1.0, 1.0, 41), torch.linspace(0.25, 2.0, 13)
        )
        assert diagnostics["min_calendar_slope"] >= -1e-10, diagnostics
        assert diagnostics["min_butterfly_g"] > 0.0, diagnostics

    def test_history_records_component_penalties(self) -> None:
        surface = _surface()
        k = torch.linspace(-0.4, 0.4, 9)
        t = torch.full((9,), 1.0)
        vol = torch.full((9,), 0.2)
        result = calibrate_surface(surface, k, t, vol, iterations=30, log_every=10)
        assert result.history
        for entry in result.history:
            for key in (
                "fit_loss", "penalty_calendar", "penalty_butterfly",
                "implied_vol_rmse",
            ):
                assert key in entry
                assert math.isfinite(entry[key])

    def test_rejects_mismatched_inputs(self) -> None:
        surface = _surface()
        with pytest.raises(ValueError, match="must match"):
            calibrate_surface(
                surface, torch.zeros(5), torch.ones(4), torch.full((5,), 0.2)
            )
        with pytest.raises(ValueError, match="strictly positive"):
            calibrate_surface(
                surface, torch.zeros(3), torch.zeros(3), torch.full((3,), 0.2)
            )


# ==========================================================================
# 4. The non-linear adjoint -- the core Phase 6 result
# ==========================================================================
def _skewed_local_vol(base: torch.Tensor, slope: torch.Tensor, steepness: float):
    r"""A smooth, genuinely state-dependent test volatility.

    :math:`\sigma(t, X) = \text{base} + \text{slope}\tanh(c(X - \log S_0))
    + 0.05\,t`.

    The skew is **centred at** :math:`\log S_0`. That detail is not cosmetic: an
    uncentred ``tanh(c*X)`` saturates at :math:`X \approx 4.6`, giving
    :math:`\operatorname{sech}^2 \approx 10^{-5}`, so
    :math:`\partial\sigma/\partial X \approx 0` and the surface is *effectively
    constant*. Any adjoint test built on it would silently pass while exercising
    nothing.
    """
    origin = math.log(SPOT_ZERO)

    def local_vol(time: float, log_spot: torch.Tensor) -> torch.Tensor:
        return base + slope * torch.tanh(steepness * (log_spot - origin)) + 0.05 * time

    return local_vol


class TestNonLinearAdjoint:
    """The sequential reverse sweep must reproduce autograd exactly."""

    @staticmethod
    def _setup(n_paths: int, n_steps: int, slope_value: float, steepness: float):
        generator = torch.Generator().manual_seed(20260821)
        normals = torch.randn((n_paths, n_steps), generator=generator)
        grad_log_spot = torch.randn((n_paths, n_steps + 1), generator=generator)
        base = torch.tensor(0.20, requires_grad=True)
        slope = torch.tensor(slope_value, requires_grad=True)
        return normals, grad_log_spot, base, slope, steepness

    @pytest.mark.parametrize(
        "n_paths,n_steps,slope,steepness",
        [(1, 1, 0.05, 1.0), (4, 8, 0.10, 2.0), (32, 64, 0.20, 3.0), (16, 252, 0.05, 1.0)],
    )
    def test_matches_autograd(
        self, n_paths: int, n_steps: int, slope: float, steepness: float
    ) -> None:
        normals, grad_log_spot, base, slope_p, steep = self._setup(
            n_paths, n_steps, slope, steepness
        )
        dt = MATURITY / n_steps
        local_vol = _skewed_local_vol(base, slope_p, steep)

        spot = torch.tensor(SPOT_ZERO, requires_grad=True)
        trajectory = simulate_local_vol_paths(spot, DRIFT, local_vol, normals, dt)
        (grad_log_spot * trajectory).sum().backward()
        truth = (float(spot.grad), float(base.grad), float(slope_p.grad))

        base_c = torch.tensor(0.20, requires_grad=True)
        slope_c = torch.tensor(slope, requires_grad=True)
        grad_spot, grad_params = local_vol_adjoint(
            grad_log_spot,
            torch.tensor(SPOT_ZERO),
            DRIFT,
            _skewed_local_vol(base_c, slope_c, steep),
            (base_c, slope_c),
            normals,
            dt,
        )
        manual = (float(grad_spot), float(grad_params[0]), float(grad_params[1]))

        for name, expected, got in zip(
            ("dL/dS0", "dL/dbase", "dL/dslope"), truth, manual
        ):
            assert math.isclose(expected, got, rel_tol=1e-9, abs_tol=1e-12), (
                f"{name}: autograd {expected!r} vs sequential {got!r}"
            )

    def test_checkpointed_agrees_with_full_storage(self) -> None:
        r""":math:`\sqrt{N}` checkpointing must change memory, never the answer."""
        normals, grad_log_spot, base, slope, steep = self._setup(24, 100, 0.15, 2.5)
        dt = MATURITY / 100

        full = local_vol_adjoint(
            grad_log_spot, torch.tensor(SPOT_ZERO), DRIFT,
            _skewed_local_vol(base, slope, steep), (base, slope), normals, dt,
        )
        base_c = torch.tensor(0.20, requires_grad=True)
        slope_c = torch.tensor(0.15, requires_grad=True)
        checkpointed = checkpointed_local_vol_adjoint(
            grad_log_spot, torch.tensor(SPOT_ZERO), DRIFT,
            _skewed_local_vol(base_c, slope_c, steep), (base_c, slope_c),
            normals, dt,
        )

        assert math.isclose(float(full[0]), float(checkpointed[0]), rel_tol=1e-12)
        for index in range(2):
            assert math.isclose(
                float(full[1][index]), float(checkpointed[1][index]), rel_tol=1e-12
            ), f"parameter {index}"

    @pytest.mark.parametrize("n_checkpoints", [1, 2, 5, 10, 100])
    def test_any_checkpoint_count_gives_the_same_answer(
        self, n_checkpoints: int
    ) -> None:
        normals, grad_log_spot, base, slope, steep = self._setup(8, 60, 0.12, 2.0)
        dt = MATURITY / 60
        reference = local_vol_adjoint(
            grad_log_spot, torch.tensor(SPOT_ZERO), DRIFT,
            _skewed_local_vol(base, slope, steep), (base, slope), normals, dt,
        )
        base_c = torch.tensor(0.20, requires_grad=True)
        slope_c = torch.tensor(0.12, requires_grad=True)
        candidate = checkpointed_local_vol_adjoint(
            grad_log_spot, torch.tensor(SPOT_ZERO), DRIFT,
            _skewed_local_vol(base_c, slope_c, steep), (base_c, slope_c),
            normals, dt, n_checkpoints=n_checkpoints,
        )
        assert math.isclose(float(reference[0]), float(candidate[0]), rel_tol=1e-11)
        assert math.isclose(
            float(reference[1][0]), float(candidate[1][0]), rel_tol=1e-11
        )

    def test_suffix_sum_shortcut_is_exact_only_for_constant_vol(self) -> None:
        """With zero skew the Phase 3-5 shortcut must reproduce the truth."""
        normals, grad_log_spot, base, _slope, steep = self._setup(32, 252, 0.0, 1.0)
        dt = MATURITY / 252
        zero_slope = torch.tensor(0.0, requires_grad=True)

        correct = local_vol_adjoint(
            grad_log_spot, torch.tensor(SPOT_ZERO), DRIFT,
            _skewed_local_vol(base, zero_slope, steep), (base,), normals, dt,
        )
        base_s = torch.tensor(0.20, requires_grad=True)
        shortcut = suffix_sum_adjoint_incorrect(
            grad_log_spot, torch.tensor(SPOT_ZERO), DRIFT,
            _skewed_local_vol(base_s, torch.tensor(0.0), steep), (base_s,),
            normals, dt,
        )
        assert math.isclose(
            float(correct[1][0]), float(shortcut[1][0]), rel_tol=1e-10
        ), "with no state dependence the suffix sum must be exact"

    @pytest.mark.parametrize("slope,steepness", [(0.05, 1.0), (0.10, 2.0), (0.20, 3.0)])
    def test_suffix_sum_shortcut_is_wrong_with_state_dependence(
        self, slope: float, steepness: float
    ) -> None:
        r"""Quantify the bias from reusing the constant-vol adjoint.

        This is the measurement that justifies the whole Phase 6 kernel redesign.
        The discrepancy is a *bias*: it does not shrink with more paths, and no
        statistical test on the exposure profile would reveal it.
        """
        normals, grad_log_spot, base, slope_p, steep = self._setup(
            64, 252, slope, steepness
        )
        dt = MATURITY / 252

        correct = local_vol_adjoint(
            grad_log_spot, torch.tensor(SPOT_ZERO), DRIFT,
            _skewed_local_vol(base, slope_p, steep), (base,), normals, dt,
        )
        base_s = torch.tensor(0.20, requires_grad=True)
        slope_s = torch.tensor(slope, requires_grad=True)
        shortcut = suffix_sum_adjoint_incorrect(
            grad_log_spot, torch.tensor(SPOT_ZERO), DRIFT,
            _skewed_local_vol(base_s, slope_s, steep), (base_s,), normals, dt,
        )

        relative = abs(
            float(shortcut[1][0]) - float(correct[1][0])
        ) / abs(float(correct[1][0]))
        assert relative > 1e-3, (
            f"slope={slope}: expected a measurable bias, got {relative:.2e}. "
            "Check that the test surface is genuinely state-dependent -- an "
            "uncentred tanh saturates and makes sigma effectively constant."
        )

    def test_gradients_flow_to_a_real_surface(self) -> None:
        """The adjoint must connect to the calibrated SSVI surface, not just toys."""
        surface = _surface()
        local = LocalVolatilitySurface(surface, rate=DRIFT, dividend_yield=0.0)

        def local_vol(time: float, log_spot: torch.Tensor) -> torch.Tensor:
            spot = torch.exp(log_spot)
            return local.local_volatility(
                spot, torch.tensor(max(time, 1e-6)), SPOT_ZERO
            )

        generator = torch.Generator().manual_seed(3)
        normals = torch.randn((16, 12), generator=generator)
        spot = torch.tensor(SPOT_ZERO, requires_grad=True)
        trajectory = simulate_local_vol_paths(
            spot, DRIFT, local_vol, normals, MATURITY / 12
        )
        trajectory.sum().backward()

        assert spot.grad is not None and torch.isfinite(spot.grad)
        for name, parameter in surface.named_parameters():
            assert parameter.grad is not None, f"{name} received no gradient"
            assert torch.all(torch.isfinite(parameter.grad)), name


class TestCheckpointSchedule:
    """The memory/recompute trade, as arithmetic."""

    @pytest.mark.parametrize("n_steps", [1, 4, 16, 100, 252, 2520])
    def test_default_is_sqrt_n(self, n_steps: int) -> None:
        schedule = checkpoint_schedule(n_steps)
        assert schedule[0] == 0
        assert len(schedule) <= math.ceil(math.sqrt(n_steps)) + 1
        assert schedule == sorted(set(schedule))
        assert all(0 <= index < n_steps for index in schedule)

    def test_memory_versus_recompute_tradeoff(self) -> None:
        r"""Document the three strategies at the realistic operating point."""
        n_steps, block_m, n_programs, element_size = 252, 16, 4096, 4
        per_state = block_m * n_programs * element_size

        full = n_steps * per_state
        sqrt_n = len(checkpoint_schedule(n_steps)) * per_state
        minimal = per_state

        assert full / 1024**2 > 60.0, "full storage should be tens of MiB"
        assert sqrt_n / 1024**2 < 6.0, "sqrt-N should be a few MiB"
        assert minimal / 1024**2 < 0.5
        # sqrt-N must be a large saving over full storage.
        assert full / sqrt_n > 10.0

    def test_rejects_invalid_arguments(self) -> None:
        with pytest.raises(ValueError, match="n_steps must be positive"):
            checkpoint_schedule(0)
        with pytest.raises(ValueError, match="n_checkpoints must be"):
            checkpoint_schedule(10, n_checkpoints=0)
