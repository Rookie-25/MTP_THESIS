r"""Tests for the Chebyshev local-volatility bridge and kernel.

Two tiers, matching this project's established pattern:

* **Tier 1 (CPU, always runs)** -- the mathematics. Basis identities, the
  hand-closed-form derivative checked against ``torch.autograd``, the
  hand-derived adjoint checked against ``torch.autograd`` (ground truth), the
  checkpointed adjoint checked against the full-storage adjoint, closed-form
  fit identifiability, and the actual before/after wing-saturation
  measurement against the tanh basis.
* **Tier 2 (GPU, opt-in)** -- the Triton kernel itself. Gated by
  ``@requires_triton`` like every other GPU tier in this repository. Because
  no Triton install or CUDA device was available while writing
  ``src/csrc/triton_chebyshev_local_vol_cva.py``, these tests have **never
  been run**; they exist so the first Colab session can execute them
  immediately, exactly as every prior Phase 6 GPU test did before its first
  run. See that module's docstring for the specific pattern flagged as
  highest-risk (a Python list of accumulators over a ``tl.constexpr``-length
  range).
"""

from __future__ import annotations

import math

import pytest
import torch

from src.csrc.triton_chebyshev_local_vol_cva import (
    ChebyshevLocalVolParams,
    chebyshev_local_vol_and_state_derivative,
    is_available,
    reference_chebyshev_local_vol_ee,
    reference_chebyshev_local_vol_ee_adjoint,
    reference_checkpointed_chebyshev_ee_adjoint,
)
from src.models.vol_surface import (
    ATMTotalVariance,
    LocalVolatilitySurface,
    SSVISurface,
    chebyshev_basis,
    chebyshev_basis_derivative,
    evaluate_chebyshev_local_vol,
    fit_local_vol_params,
)
from src.pricer.options import SwapLeg

requires_triton = pytest.mark.skipif(
    not is_available(), reason="Triton + CUDA runtime not present"
)

SEED = 20260904
SPOT = 100.0
DRIFT = 0.02
RATE = 0.03
MATURITY = 1.0


# ==========================================================================
# Chebyshev basis: pure math
# ==========================================================================
class TestChebyshevBasis:
    """Identities that pin down the recurrence independent of any fit."""

    def test_known_polynomials(self) -> None:
        """T_0=1, T_1=u, T_2=2u^2-1, T_3=4u^3-3u -- the textbook forms."""
        u = torch.tensor([-1.0, -0.5, 0.0, 0.37, 0.5, 1.0], dtype=torch.float64)
        basis = chebyshev_basis(u, 3)
        torch.testing.assert_close(basis[0], torch.ones_like(u))
        torch.testing.assert_close(basis[1], u)
        torch.testing.assert_close(basis[2], 2 * u**2 - 1)
        torch.testing.assert_close(basis[3], 4 * u**3 - 3 * u)

    def test_endpoint_values(self) -> None:
        """T_k(1) = 1 and T_k(-1) = (-1)^k for every k."""
        u = torch.tensor([-1.0, 1.0], dtype=torch.float64)
        basis = chebyshev_basis(u, 8)
        expected_at_one = torch.ones(9, dtype=torch.float64)
        expected_at_minus_one = torch.tensor(
            [(-1.0) ** k for k in range(9)], dtype=torch.float64
        )
        torch.testing.assert_close(basis[:, 1], expected_at_one)
        torch.testing.assert_close(basis[:, 0], expected_at_minus_one)

    def test_rejects_negative_degree(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            chebyshev_basis(torch.zeros(3, dtype=torch.float64), -1)

    @pytest.mark.parametrize("degree", [0, 1, 2, 3, 5, 8, 14])
    def test_derivative_matches_autograd(self, degree: int) -> None:
        """The closed form used by the hand-written adjoint, independently checked.

        This is the derivative the Triton kernel's backward pass needs
        analytically (it cannot call ``torch.autograd``); checking it against
        autograd here is what licenses using it un-verified inside a kernel
        that cannot itself be run in this environment.
        """
        u = torch.linspace(-1.3, 1.3, 41, dtype=torch.float64, requires_grad=True)
        basis = chebyshev_basis(u, degree)
        closed_form = chebyshev_basis_derivative(u.detach(), degree)

        if not basis.requires_grad:
            # degree=0: T_0 is torch.ones_like(u), a constant that never
            # touches u at all -- there is no graph for autograd to walk, and
            # the correct derivative (checked directly) is exactly zero.
            torch.testing.assert_close(closed_form, torch.zeros_like(closed_form))
            return

        rows = []
        for k in range(degree + 1):
            grad, = torch.autograd.grad(
                basis[k].sum(), u, retain_graph=True, allow_unused=True
            )
            rows.append(grad if grad is not None else torch.zeros_like(u))
        autograd_derivative = torch.stack(rows)
        torch.testing.assert_close(
            autograd_derivative, closed_form, rtol=1e-10, atol=1e-12
        )

    def test_rejects_negative_degree_in_derivative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            chebyshev_basis_derivative(torch.zeros(3, dtype=torch.float64), -1)


class TestEvaluateChebyshevLocalVol:
    """The host-side evaluator the fit, the CPU reference, and any plot share."""

    def test_matches_a_direct_tensordot(self) -> None:
        coefficients = torch.tensor([0.2, 0.05, -0.03], dtype=torch.float64)
        time = torch.tensor([0.5, 1.0], dtype=torch.float64)
        log_spot = torch.tensor([4.5, 4.7], dtype=torch.float64)
        result = evaluate_chebyshev_local_vol(
            coefficients, half_width=0.6, reference=4.6, term=0.04,
            time=time, log_spot=log_spot, floor=1e-6,
        )
        u = (log_spot - 4.6) / 0.6
        basis = chebyshev_basis(u, 2)
        expected = torch.clamp(
            torch.tensordot(coefficients, basis, dims=([0], [0])), min=1e-6
        ) + 0.04 * time
        torch.testing.assert_close(result, expected)

    def test_floor_binds_only_the_spatial_term(self) -> None:
        r"""A negative time term must not be swallowed by the floor.

        If the floor were applied to the *whole* surface rather than the
        Chebyshev sum alone, a negative ``term*t`` correction would be
        clamped away and time-structure sensitivity would silently vanish.
        """
        coefficients = torch.tensor([1e-8], dtype=torch.float64)  # near zero
        result = evaluate_chebyshev_local_vol(
            coefficients, half_width=1.0, reference=0.0, term=-0.5,
            time=torch.tensor([1.0], dtype=torch.float64),
            log_spot=torch.tensor([0.0], dtype=torch.float64),
            floor=1e-4,
        )
        # floor (1e-4) applied to the spatial term, then -0.5 subtracted.
        assert result.item() == pytest.approx(1e-4 - 0.5, abs=1e-12)


# ==========================================================================
# Fitting: identifiability and diagnostics
# ==========================================================================
class _ChebyshevStub:
    """A Dupire-surface stand-in that IS a Chebyshev sum, for identifiability."""

    variance_floor = 1e-8

    def __init__(self, coefficients, half_width: float, reference: float, term: float):
        self.coefficients = coefficients
        self.half_width = half_width
        self.reference = reference
        self.term = term

    def local_volatility(self, spot, time, spot_zero):
        log_spot = torch.log(spot)
        return evaluate_chebyshev_local_vol(
            torch.tensor(self.coefficients, dtype=torch.float64),
            self.half_width, self.reference, self.term, time, log_spot,
            floor=1e-6,
        )


class TestChebyshevFit:
    """``fit_local_vol_params(..., basis="chebyshev")``."""

    TRUTH = [0.20, 0.05, -0.03, 0.02, -0.01]

    def test_recovers_known_coefficients_exactly(self) -> None:
        """Closed-form least squares on an exactly-representable target.

        Not merely close: the model is linear in the coefficients and the
        target lies exactly in its span, so the residual should be at
        floating-point noise, not at any iteration-count-dependent tolerance.
        """
        half_width = 0.6
        stub = _ChebyshevStub(self.TRUTH, half_width, math.log(SPOT), 0.03)
        fit = fit_local_vol_params(
            stub, SPOT, MATURITY, basis="chebyshev", degree=len(self.TRUTH) - 1,
        )
        for truth, fitted in zip(self.TRUTH, fit.chebyshev_coefficients):
            assert fitted == pytest.approx(truth, abs=1e-8)
        assert fit.term == pytest.approx(0.03, abs=1e-8)
        assert fit.relative_rmse < 1e-6
        assert fit.r_squared > 1.0 - 1e-10
        assert fit.basis == "chebyshev"
        assert fit.base is None and fit.skew is None and fit.kappa is None

    def test_overparameterised_fit_zeros_the_extra_terms(self) -> None:
        """Fitting with degree above the truth must not corrupt the low terms."""
        half_width = 0.6
        stub = _ChebyshevStub(self.TRUTH, half_width, math.log(SPOT), 0.03)
        fit = fit_local_vol_params(stub, SPOT, MATURITY, basis="chebyshev", degree=10)
        for truth, fitted in zip(self.TRUTH, fit.chebyshev_coefficients):
            assert fitted == pytest.approx(truth, abs=1e-6)
        for extra in fit.chebyshev_coefficients[len(self.TRUTH):]:
            assert extra == pytest.approx(0.0, abs=1e-6)

    def test_reference_pinned_to_log_spot(self) -> None:
        stub = _ChebyshevStub(self.TRUTH, 0.6, math.log(SPOT), 0.0)
        for spot in (50.0, 100.0, 4000.0):
            fit = fit_local_vol_params(
                _ChebyshevStub(self.TRUTH, 0.6, math.log(spot), 0.0),
                spot, MATURITY, basis="chebyshev", degree=4,
            )
            assert fit.reference == math.log(spot)

    def test_half_width_matches_the_sampling_cone(self) -> None:
        """The domain is fixed to sigma_width * reference_vol, not fitted."""
        stub = _ChebyshevStub(self.TRUTH, 0.6, math.log(SPOT), 0.0)
        fit = fit_local_vol_params(
            stub, SPOT, MATURITY, basis="chebyshev", degree=4,
            reference_vol=0.25, sigma_width=2.0,
        )
        assert fit.chebyshev_half_width == pytest.approx(0.5)

    def test_runs_inside_a_no_grad_block(self) -> None:
        """Same Dupire-autograd trap as the tanh fit; must be guarded here too."""
        stub = _ChebyshevStub(self.TRUTH, 0.6, math.log(SPOT), 0.0)
        with torch.no_grad():
            fit = fit_local_vol_params(
                stub, SPOT, MATURITY, basis="chebyshev", degree=4
            )
        assert fit.chebyshev_coefficients is not None

    def test_is_closed_form_and_fast(self) -> None:
        """No Adam loop: this should be near-instant regardless of `iterations`."""
        import time as _time

        stub = _ChebyshevStub(self.TRUTH, 0.6, math.log(SPOT), 0.0)
        start = _time.perf_counter()
        fit_local_vol_params(
            stub, SPOT, MATURITY, basis="chebyshev", degree=8,
            iterations=999_999,  # would take hours if this were Adam
        )
        assert _time.perf_counter() - start < 2.0

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"basis": "chebyshev", "degree": -1}, "non-negative"),
            ({"basis": "chebyshev", "floor": 0.0}, "must be positive"),
            ({"basis": "spline"}, "unknown basis"),
        ],
    )
    def test_rejects_invalid_arguments(self, kwargs, message) -> None:
        stub = _ChebyshevStub(self.TRUTH, 0.6, math.log(SPOT), 0.0)
        with pytest.raises(ValueError, match=message):
            fit_local_vol_params(stub, SPOT, MATURITY, **kwargs)

    def test_to_local_vol_params_dispatches_to_chebyshev(self) -> None:
        stub = _ChebyshevStub(self.TRUTH, 0.6, math.log(SPOT), 0.03)
        fit = fit_local_vol_params(stub, SPOT, MATURITY, basis="chebyshev", degree=4)
        params = fit.to_local_vol_params()
        assert isinstance(params, ChebyshevLocalVolParams)
        assert params.degree == 4
        assert params.term == pytest.approx(fit.term)
        assert params.reference == pytest.approx(fit.reference)

    def test_summary_renders_for_chebyshev(self) -> None:
        stub = _ChebyshevStub(self.TRUTH, 0.6, math.log(SPOT), 0.03)
        fit = fit_local_vol_params(stub, SPOT, MATURITY, basis="chebyshev", degree=4)
        text = fit.summary()
        assert "chebyshev" in text
        assert "T_0" in text and f"T_{len(self.TRUTH) - 1}" in text

    def test_tanh_path_is_completely_unaffected(self) -> None:
        """The default basis must still be tanh, byte-for-byte as before.

        This is the regression guard for the refactor that split
        ``fit_local_vol_params`` into per-basis helpers: it must not have
        changed a single number on the path 43 pre-existing tests already
        pin down.
        """
        knots = torch.tensor([0.1, 0.25, 0.5, 1.0, 2.0], dtype=torch.float64)
        surface = SSVISurface(
            atm=ATMTotalVariance(knots), rho=-0.35, eta=1.2, gamma=0.45
        ).double()
        local_vol = LocalVolatilitySurface(surface, rate=0.02)
        fit = fit_local_vol_params(local_vol, SPOT, MATURITY, drift=0.02, iterations=200)
        assert fit.basis == "tanh"
        assert fit.base is not None and fit.kappa is not None
        assert fit.chebyshev_coefficients is None


# ==========================================================================
# THE ACTUAL FIX: measured before/after
# ==========================================================================
class TestWingSaturationFix:
    """Direct evidence the Chebyshev basis fixes what the tanh basis could not.

    Same SSVI surface, same sampling width, same evaluation grid -- only the
    basis differs. This is the comparison the whole file exists to support.
    """

    @classmethod
    @pytest.fixture(scope="class")
    def local_vol(cls) -> LocalVolatilitySurface:
        knots = torch.tensor([0.1, 0.25, 0.5, 1.0, 2.0], dtype=torch.float64)
        surface = SSVISurface(
            atm=ATMTotalVariance(knots), rho=-0.35, eta=1.2, gamma=0.45
        ).double()
        return LocalVolatilitySurface(surface, rate=0.02)

    def test_chebyshev_beats_tanh_at_three_sigma(
        self, local_vol: LocalVolatilitySurface
    ) -> None:
        r"""The headline number: relative RMSE roughly halves at degree 8.

        Measured once during development: tanh gives 15.99% / :math:`R^2`
        0.578; degree-8 Chebyshev gives 7.71% / 0.902. The assertions below
        use looser bounds than the exact measurement so the test is not
        fragile to the SSVI surface's own numerics changing by noise, while
        still failing if the fix regresses to roughly tanh-level error.
        """
        tanh_fit = fit_local_vol_params(
            local_vol, SPOT, MATURITY, drift=0.02, sigma_width=3.0,
            iterations=1500,
        )
        cheb_fit = fit_local_vol_params(
            local_vol, SPOT, MATURITY, drift=0.02, sigma_width=3.0,
            basis="chebyshev", degree=8,
        )

        assert tanh_fit.relative_rmse > 0.10, "tanh should still be struggling here"
        assert cheb_fit.relative_rmse < 0.5 * tanh_fit.relative_rmse, (
            f"expected roughly half the relative error; tanh="
            f"{tanh_fit.relative_rmse:.3%} chebyshev={cheb_fit.relative_rmse:.3%}"
        )
        assert cheb_fit.r_squared > tanh_fit.r_squared + 0.2

    def test_error_decreases_monotonically_with_degree(
        self, local_vol: LocalVolatilitySurface
    ) -> None:
        """More terms should never make a least-squares fit worse.

        A degree-K fit's optimum is a superset of degree-(K-1)'s (the extra
        coefficient can always be set to zero), so relative RMSE is
        monotonically non-increasing in K for the exact solver used here. A
        violation would indicate a bug in the closed-form solve, not noise.
        """
        errors = [
            fit_local_vol_params(
                local_vol, SPOT, MATURITY, drift=0.02, sigma_width=3.0,
                basis="chebyshev", degree=degree,
            ).relative_rmse
            for degree in (2, 4, 6, 8, 12, 16)
        ]
        for coarse, fine in zip(errors[:-1], errors[1:]):
            assert fine <= coarse + 1e-9, errors

    def test_fit_quality_degrades_gracefully_at_narrow_widths_too(
        self, local_vol: LocalVolatilitySurface
    ) -> None:
        """Chebyshev should dominate tanh at every sampling width, not just 3-sigma."""
        for width in (1.0, 2.0, 3.0):
            tanh_fit = fit_local_vol_params(
                local_vol, SPOT, MATURITY, drift=0.02, sigma_width=width,
                iterations=1000,
            )
            cheb_fit = fit_local_vol_params(
                local_vol, SPOT, MATURITY, drift=0.02, sigma_width=width,
                basis="chebyshev", degree=8,
            )
            assert cheb_fit.relative_rmse <= tanh_fit.relative_rmse, (
                f"width={width}: chebyshev did not improve on tanh"
            )


# ==========================================================================
# CPU-tier kernel reference: forward, adjoint, checkpointing
# ==========================================================================
@pytest.fixture
def netting_set():
    """A small off-the-money netting set and CRN normals, matching the
    at-the-money-kink avoidance already established for the tanh kernel:
    strike=95 keeps t=0 exposure off the max(V,0) kink."""
    from src.csrc.triton_cva_fusion import build_affine_coefficients

    n_steps, n_paths = 40, 2000
    dt = MATURITY / n_steps
    times = torch.linspace(0.0, MATURITY, n_steps + 1, dtype=torch.float64)
    legs = [SwapLeg(notional=1.0, strike=95.0, maturity=MATURITY)]
    coeff_b, coeff_c = build_affine_coefficients(legs, times, RATE)
    generator = torch.Generator().manual_seed(SEED + 7)
    normals = torch.randn((n_paths, n_steps), dtype=torch.float64, generator=generator)
    weights = torch.randn(n_steps + 1, dtype=torch.float64, generator=generator)
    return dict(dt=dt, coeff_b=coeff_b, coeff_c=coeff_c, normals=normals, weights=weights)


COEFFICIENTS = [0.20, 0.05, -0.03, 0.02, -0.01]
HALF_WIDTH = 0.6


class TestChebyshevKernelForward:
    """The pure-torch forward that everything else is checked against."""

    def test_finite_and_bounded_below(self, netting_set) -> None:
        s0 = torch.tensor(SPOT, dtype=torch.float64)
        drift = torch.tensor(DRIFT, dtype=torch.float64)
        coefficients = torch.tensor(COEFFICIENTS, dtype=torch.float64)
        term = torch.tensor(0.03, dtype=torch.float64)
        ee = reference_chebyshev_local_vol_ee(
            s0, drift, coefficients, term, netting_set["normals"], netting_set["dt"],
            netting_set["coeff_b"], netting_set["coeff_c"], HALF_WIDTH, math.log(SPOT),
        )
        assert torch.isfinite(ee).all()
        assert bool((ee >= 0.0).all())

    def test_state_derivative_matches_autograd(self) -> None:
        """The forward's per-step vol/derivative helper, checked directly."""
        log_spot = torch.linspace(4.0, 5.2, 30, dtype=torch.float64, requires_grad=True)
        coefficients = torch.tensor(COEFFICIENTS, dtype=torch.float64)
        term = torch.tensor(0.02, dtype=torch.float64)

        u = (log_spot - math.log(SPOT)) / HALF_WIDTH
        basis = chebyshev_basis(u, len(COEFFICIENTS) - 1)
        spatial = torch.tensordot(coefficients, basis, dims=([0], [0]))
        sigma_autograd = torch.clamp(spatial, min=1e-4) + term * 0.3
        grad, = torch.autograd.grad(sigma_autograd.sum(), log_spot)

        sigma, d_sigma_dx = chebyshev_local_vol_and_state_derivative(
            log_spot.detach(), coefficients, HALF_WIDTH, math.log(SPOT),
            float(term), 0.3, floor=1e-4,
        )
        torch.testing.assert_close(sigma, sigma_autograd.detach())
        torch.testing.assert_close(d_sigma_dx, grad)


class TestChebyshevKernelAdjoint:
    """The hand-derived adjoints, checked against autograd and each other."""

    def test_full_storage_adjoint_matches_autograd(self, netting_set) -> None:
        s0 = torch.tensor(SPOT, dtype=torch.float64, requires_grad=True)
        drift = torch.tensor(DRIFT, dtype=torch.float64, requires_grad=True)
        coefficients = torch.tensor(
            COEFFICIENTS, dtype=torch.float64, requires_grad=True
        )
        term = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)

        ee = reference_chebyshev_local_vol_ee(
            s0, drift, coefficients, term, netting_set["normals"], netting_set["dt"],
            netting_set["coeff_b"], netting_set["coeff_c"], HALF_WIDTH, math.log(SPOT),
        )
        (netting_set["weights"] * ee).sum().backward()

        d_s0, d_drift, d_coefficients, d_term = reference_chebyshev_local_vol_ee_adjoint(
            netting_set["weights"], s0.detach(), drift.detach(), coefficients.detach(),
            term.detach(), netting_set["normals"], netting_set["dt"],
            netting_set["coeff_b"], netting_set["coeff_c"], HALF_WIDTH, math.log(SPOT),
        )

        assert d_s0.item() == pytest.approx(s0.grad.item(), rel=1e-9)
        assert d_drift.item() == pytest.approx(drift.grad.item(), rel=1e-9)
        assert d_term.item() == pytest.approx(term.grad.item(), rel=1e-9)
        for k in range(len(COEFFICIENTS)):
            expected = coefficients.grad[k].item()
            got = d_coefficients[k].item()
            if abs(expected) < 1e-10:
                assert abs(got - expected) < 1e-8
            else:
                assert got == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("stride", [4, 8, 16, 40])
    def test_checkpointed_adjoint_matches_full_storage_exactly(
        self, netting_set, stride: int
    ) -> None:
        """Bitwise agreement, the same standard the tanh kernel's checkpointing met.

        Not "within tolerance": the checkpointed replay must reproduce the
        full-storage computation's floating-point path exactly, since both
        recompute the identical forward arithmetic -- only *when* the state
        is stored differs.
        """
        s0 = torch.tensor(SPOT, dtype=torch.float64)
        drift = torch.tensor(DRIFT, dtype=torch.float64)
        coefficients = torch.tensor(COEFFICIENTS, dtype=torch.float64)
        term = torch.tensor(0.03, dtype=torch.float64)

        full = reference_chebyshev_local_vol_ee_adjoint(
            netting_set["weights"], s0, drift, coefficients, term,
            netting_set["normals"], netting_set["dt"], netting_set["coeff_b"],
            netting_set["coeff_c"], HALF_WIDTH, math.log(SPOT),
        )
        checkpointed = reference_checkpointed_chebyshev_ee_adjoint(
            netting_set["weights"], s0, drift, coefficients, term,
            netting_set["normals"], netting_set["dt"], netting_set["coeff_b"],
            netting_set["coeff_c"], HALF_WIDTH, math.log(SPOT),
            checkpoint_stride=stride,
        )
        for full_value, checkpointed_value in zip(full, checkpointed):
            assert torch.equal(full_value, checkpointed_value)

    def test_at_the_money_kink_reproduces_the_known_discrepancy(self) -> None:
        r"""Sanity check that this file is subject to the same t=0 kink as tanh.

        ``strike == spot`` makes ``V[0] == 0`` exactly, sitting on the
        ``max(V,0)`` kink -- the same phenomenon
        ``tests/test_phase6_kernel.py::TestAtTheMoneyKink`` documents for the
        tanh kernel. This is not a Chebyshev-specific bug; it is inherited
        from the shared payoff structure, and this test exists so a future
        "gradient mismatch" investigation checks this cause first rather than
        re-discovering it.
        """
        from src.csrc.triton_cva_fusion import build_affine_coefficients

        n_steps, n_paths = 20, 1000
        dt = MATURITY / n_steps
        times = torch.linspace(0.0, MATURITY, n_steps + 1, dtype=torch.float64)
        legs = [SwapLeg(notional=1.0, strike=SPOT, maturity=MATURITY)]  # AT the money
        coeff_b, coeff_c = build_affine_coefficients(legs, times, RATE)
        generator = torch.Generator().manual_seed(SEED)
        normals = torch.randn((n_paths, n_steps), dtype=torch.float64, generator=generator)
        weights = torch.randn(n_steps + 1, dtype=torch.float64, generator=generator)

        s0 = torch.tensor(SPOT, dtype=torch.float64, requires_grad=True)
        drift = torch.tensor(DRIFT, dtype=torch.float64, requires_grad=True)
        coefficients = torch.tensor(
            COEFFICIENTS, dtype=torch.float64, requires_grad=True
        )
        term = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)

        ee = reference_chebyshev_local_vol_ee(
            s0, drift, coefficients, term, normals, dt, coeff_b, coeff_c,
            HALF_WIDTH, math.log(SPOT),
        )
        (weights * ee).sum().backward()

        d_s0, *_ = reference_chebyshev_local_vol_ee_adjoint(
            weights, s0.detach(), drift.detach(), coefficients.detach(),
            term.detach(), normals, dt, coeff_b, coeff_c, HALF_WIDTH,
            math.log(SPOT),
        )
        v0 = float(coeff_b[0]) * SPOT - float(coeff_c[0])
        assert v0 == 0.0, "this test's premise (exact ATM at t=0) must hold"
        # The AAD and hand-adjoint values need not agree with a naive
        # finite-difference oracle here (that is the documented kink
        # behaviour), but the hand adjoint MUST still agree with autograd,
        # since both take the identical subgradient convention at the kink.
        assert d_s0.item() == pytest.approx(s0.grad.item(), rel=1e-9)


class TestChebyshevLocalVolParams:
    """Validation on the kernel-facing parameter container."""

    def test_valid_construction(self) -> None:
        params = ChebyshevLocalVolParams(
            coefficients=(0.2, 0.05, -0.03), half_width=0.6, reference=4.6,
        )
        assert params.degree == 2

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"coefficients": ()}, "at least one"),
            ({"coefficients": (0.2,), "half_width": 0.0}, "half_width must be positive"),
            ({"coefficients": (0.2,), "floor": 0.0}, "floor must be positive"),
            ({"coefficients": (float("nan"),)}, "must be finite"),
        ],
    )
    def test_rejects_invalid_construction(self, kwargs, message) -> None:
        base = dict(coefficients=(0.2, 0.1), half_width=0.6, reference=4.6)
        base.update(kwargs)
        with pytest.raises(ValueError, match=message):
            ChebyshevLocalVolParams(**base)

    def test_rejects_degree_above_the_kernel_cap(self) -> None:
        from src.csrc.triton_chebyshev_local_vol_cva import MAX_CHEBYSHEV_DEGREE

        with pytest.raises(ValueError, match="MAX_CHEBYSHEV_DEGREE"):
            ChebyshevLocalVolParams(
                coefficients=tuple([0.1] * (MAX_CHEBYSHEV_DEGREE + 2)),
                half_width=0.6, reference=4.6,
            )


# ==========================================================================
# Tier 2: the actual Triton kernel (never run in this environment)
# ==========================================================================
@requires_triton
class TestChebyshevKernelGPU:
    """GPU-tier: the compiled kernel against the CPU reference.

    Skipped wherever Triton + CUDA are unavailable, which is everywhere this
    file has been developed. The first Colab run of this suite is also the
    first time this kernel has ever been compiled -- read
    ``src/csrc/triton_chebyshev_local_vol_cva.py``'s module docstring before
    debugging a failure here; it names the specific untested pattern most
    likely to be the cause.
    """

    def test_forward_matches_cpu_reference(self, netting_set) -> None:
        from src.csrc.triton_chebyshev_local_vol_cva import fused_chebyshev_local_vol_ee

        params = ChebyshevLocalVolParams(
            coefficients=tuple(COEFFICIENTS), half_width=HALF_WIDTH,
            reference=math.log(SPOT), term=0.03,
        )
        times = torch.linspace(
            0.0, MATURITY, netting_set["normals"].shape[1] + 1,
            device="cuda", dtype=torch.float64,
        )
        legs = [SwapLeg(notional=1.0, strike=95.0, maturity=MATURITY)]
        s0 = torch.tensor(SPOT, device="cuda", dtype=torch.float64)
        drift = torch.tensor(DRIFT, device="cuda", dtype=torch.float64)

        gpu_ee = fused_chebyshev_local_vol_ee(
            s0, drift, legs, times, RATE,
            n_paths=netting_set["normals"].shape[0], params=params, seed=SEED,
        )
        cpu_ee = reference_chebyshev_local_vol_ee(
            torch.tensor(SPOT, dtype=torch.float64),
            torch.tensor(DRIFT, dtype=torch.float64),
            torch.tensor(COEFFICIENTS, dtype=torch.float64),
            torch.tensor(0.03, dtype=torch.float64),
            netting_set["normals"], netting_set["dt"], netting_set["coeff_b"],
            netting_set["coeff_c"], HALF_WIDTH, math.log(SPOT),
        )
        # Independent random streams (Philox in-kernel vs CPU torch.randn),
        # so agreement is expected only at Monte-Carlo scale -- see
        # benchmarks/bench_phase6.py's own note on this exact point.
        assert torch.allclose(
            gpu_ee.cpu(), cpu_ee, rtol=0.05, atol=0.05
        )

    def test_gradients_match_finite_differences(self, netting_set) -> None:
        from src.csrc.triton_chebyshev_local_vol_cva import fused_chebyshev_local_vol_ee

        params = ChebyshevLocalVolParams(
            coefficients=tuple(COEFFICIENTS), half_width=HALF_WIDTH,
            reference=math.log(SPOT), term=0.03,
        )
        times = torch.linspace(
            0.0, MATURITY, netting_set["normals"].shape[1] + 1,
            device="cuda", dtype=torch.float64,
        )
        legs = [SwapLeg(notional=1.0, strike=95.0, maturity=MATURITY)]
        weights = netting_set["weights"].to("cuda")

        def value(spot: float, coefficients) -> torch.Tensor:
            s0 = torch.tensor(spot, device="cuda", dtype=torch.float64)
            drift = torch.tensor(DRIFT, device="cuda", dtype=torch.float64)
            ee = fused_chebyshev_local_vol_ee(
                s0, drift, legs, times, RATE,
                n_paths=netting_set["normals"].shape[0], params=params,
                coefficients=coefficients, seed=SEED,
            )
            return (weights * ee).sum()

        coefficients = torch.tensor(
            COEFFICIENTS, device="cuda", dtype=torch.float64, requires_grad=True
        )
        s0 = torch.tensor(SPOT, device="cuda", dtype=torch.float64, requires_grad=True)
        drift = torch.tensor(DRIFT, device="cuda", dtype=torch.float64, requires_grad=True)
        ee = fused_chebyshev_local_vol_ee(
            s0, drift, legs, times, RATE,
            n_paths=netting_set["normals"].shape[0], params=params,
            coefficients=coefficients, seed=SEED,
        )
        (weights * ee).sum().backward()

        step = 1e-6
        for k in range(len(COEFFICIENTS)):
            bumped_up = coefficients.detach().clone()
            bumped_up[k] += step
            bumped_down = coefficients.detach().clone()
            bumped_down[k] -= step
            finite = (
                value(SPOT, bumped_up).item() - value(SPOT, bumped_down).item()
            ) / (2 * step)
            assert coefficients.grad[k].item() == pytest.approx(finite, rel=5e-3)
