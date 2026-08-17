r"""Phase 1 correctness suite: MC pricing and AAD-vs-FD Greeks agreement.

Test strategy
-------------
Every test compares against an **independent oracle** wherever one exists:

* MC price vs. closed-form Black-Scholes / forward value (Monte-Carlo error
  only -- checks the simulator and the payoff, not the differentiation).
* AAD Greeks vs. bump-and-revalue Greeks under common random numbers (checks
  the differentiation scheme against a method with a different failure mode).
* AAD Greeks vs. closed-form Black-Scholes Greeks (checks both simultaneously,
  with a wider tolerance to absorb genuine MC sampling error).

All three must agree for the pipeline to be trustworthy: a bug that fools two
of the three checks but not the third is exactly the class of error this
project cannot afford once portfolio Greeks feed real hedging decisions.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.models.gbm import GBMSimulator
from src.pricer.analytic import (
    black_scholes_call,
    black_scholes_call_delta,
    black_scholes_call_vega,
    equity_forward_value,
)
from src.pricer.greeks import (
    aad_greeks,
    compare_greeks,
    finite_difference_greeks,
)
from src.pricer.options import (
    MCPrice,
    SwapLeg,
    european_call_price,
    make_european_call_price_fn,
    make_portfolio_swap_price_fn,
    portfolio_swap_price,
)

# Shared deterministic market / contract setup for every test in this module.
S0 = 100.0
STRIKE = 100.0
RATE = 0.03
SIGMA = 0.20
MATURITY = 1.0
N_STEPS = 64
N_PATHS = 200_000
SEED = 20260813

# AAD vs FD (and vs closed form) must agree within this absolute tolerance,
# per the Phase 1 acceptance criterion in the project brief.
GREEK_TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def simulator() -> GBMSimulator:
    """CPU, float64 simulator: correctness baseline, not a speed benchmark."""
    return GBMSimulator(maturity=MATURITY, n_steps=N_STEPS, device=torch.device("cpu"))


@pytest.fixture(scope="module")
def brownian_increments(simulator: GBMSimulator) -> torch.Tensor:
    """A single fixed Brownian sample shared by every AAD/FD comparison.

    Reusing one sample across the whole module is what makes bump-and-revalue
    a valid oracle: every price_fn built from it differentiates the *same* MC
    realisation, so AAD and FD agree up to true numerical error rather than
    up to independent sampling noise.
    """
    return simulator.draw_increments(N_PATHS, seed=SEED)


class TestMonteCarloPricing:
    """MC estimator vs. closed-form reference (no differentiation involved)."""

    def test_european_call_matches_black_scholes(
        self, simulator: GBMSimulator, brownian_increments: torch.Tensor
    ) -> None:
        paths = simulator.simulate(S0, RATE, SIGMA, dW=brownian_increments)
        mc: MCPrice = european_call_price(paths, STRIKE, RATE, MATURITY)
        analytic = float(black_scholes_call(S0, STRIKE, RATE, SIGMA, MATURITY))

        lo, hi = mc.confidence_interval
        assert lo - 5e-3 <= analytic <= hi + 5e-3, (
            f"BS price {analytic:.6f} outside MC 95% CI [{lo:.6f}, {hi:.6f}] "
            "(plus a small numerical-tolerance pad)"
        )
        assert math.isclose(float(mc.value), analytic, abs_tol=0.15)

    def test_portfolio_swap_matches_forward_value(
        self, simulator: GBMSimulator, brownian_increments: torch.Tensor
    ) -> None:
        legs = [SwapLeg(notional=1.0, strike=STRIKE, maturity=MATURITY)]
        paths = simulator.simulate(S0, RATE, SIGMA, dW=brownian_increments)
        times = simulator.time_grid()
        mc = portfolio_swap_price(paths, times, legs, RATE)
        analytic = float(equity_forward_value(S0, STRIKE, RATE, MATURITY))

        lo, hi = mc.confidence_interval
        assert lo - 5e-3 <= analytic <= hi + 5e-3
        assert math.isclose(float(mc.value), analytic, abs_tol=0.1)

    def test_multi_leg_portfolio_is_additive(
        self, simulator: GBMSimulator, brownian_increments: torch.Tensor
    ) -> None:
        """Netted 2-leg book equals the sum of two independently-priced legs."""
        legs = [
            SwapLeg(notional=1.5, strike=95.0, maturity=MATURITY),
            SwapLeg(notional=-0.5, strike=105.0, maturity=MATURITY),
        ]
        paths = simulator.simulate(S0, RATE, SIGMA, dW=brownian_increments)
        times = simulator.time_grid()
        netted = portfolio_swap_price(paths, times, legs, RATE)

        expected = sum(
            leg.notional
            * float(equity_forward_value(S0, leg.strike, RATE, MATURITY))
            for leg in legs
        )
        assert math.isclose(float(netted.value), expected, abs_tol=0.15)


class TestAADGreeks:
    """AAD sensitivities vs. the bump-and-revalue oracle and closed form."""

    @staticmethod
    @pytest.fixture(scope="class")
    def call_price_fn(simulator: GBMSimulator, brownian_increments: torch.Tensor):
        return make_european_call_price_fn(
            simulator, brownian_increments, STRIKE, rate=RATE
        )

    def test_call_delta_and_vega_match_finite_difference(self, call_price_fn) -> None:
        params = {"s0": S0, "sigma": SIGMA}
        aad = aad_greeks(call_price_fn, params)
        fd = finite_difference_greeks(call_price_fn, params, scheme="central")

        comparison = compare_greeks(fd, aad)
        assert comparison.max_absolute_error < GREEK_TOLERANCE, (
            f"AAD vs FD mismatch exceeds {GREEK_TOLERANCE}: "
            f"{comparison.absolute_error}"
        )
        # Same MC draw underlies both prices; they should match near-exactly.
        assert comparison.price_absolute_error < 1e-9

    def test_call_delta_and_vega_match_closed_form(self, call_price_fn) -> None:
        params = {"s0": S0, "sigma": SIGMA}
        aad = aad_greeks(call_price_fn, params)

        analytic_delta = float(black_scholes_call_delta(S0, STRIKE, RATE, SIGMA, MATURITY))
        analytic_vega = float(black_scholes_call_vega(S0, STRIKE, RATE, SIGMA, MATURITY))

        # Wider tolerance: this comparison also absorbs genuine MC sampling
        # error, unlike the CRN-based FD comparison above.
        assert math.isclose(aad.greeks["s0"], analytic_delta, abs_tol=5e-3)
        assert math.isclose(aad.greeks["sigma"], analytic_vega, abs_tol=0.5)

    def test_aad_matches_fd_forward_scheme_too(self, call_price_fn) -> None:
        params = {"s0": S0, "sigma": SIGMA}
        aad = aad_greeks(call_price_fn, params)
        fd_fwd = finite_difference_greeks(call_price_fn, params, scheme="forward")

        comparison = compare_greeks(fd_fwd, aad)
        # Forward differences carry O(h) truncation error, so allow more slack
        # than the central-difference test while still catching gross bugs.
        assert comparison.max_absolute_error < 1e-2

    def test_aad_uses_a_single_valuation_fd_uses_many(self, call_price_fn) -> None:
        params = {"s0": S0, "sigma": SIGMA}
        aad = aad_greeks(call_price_fn, params)
        fd = finite_difference_greeks(call_price_fn, params, scheme="central")

        assert aad.n_valuations == 1
        # 2 params, central scheme: base + 2*(up+down) = 5 valuations.
        assert fd.n_valuations == 5

    def test_linear_swap_delta_is_exactly_notional(
        self, simulator: GBMSimulator, brownian_increments: torch.Tensor
    ) -> None:
        """Closed-form regression: a forward's Delta is exactly notional.

        Vega is analytically zero (a linear leg does not depend on sigma at
        all), but the *finite-sample* MC estimator's sample-mean pathwise
        derivative is only zero in expectation -- its standard error scales as
        O(S0*sqrt(T)/sqrt(M)), which is non-negligible even at M=200,000. So
        instead of asserting a near-zero Vega, assert that AAD reproduces the
        *same* residual as bump-and-revalue under common random numbers: that
        is the actual invariant a broken tape would violate.
        """
        legs = [SwapLeg(notional=1.0, strike=STRIKE, maturity=MATURITY)]
        swap_price_fn = make_portfolio_swap_price_fn(
            simulator, brownian_increments, legs, rate=RATE
        )
        params = {"s0": S0, "sigma": SIGMA}
        aad = aad_greeks(swap_price_fn, params)
        fd = finite_difference_greeks(swap_price_fn, params, scheme="central")

        assert math.isclose(aad.greeks["s0"], 1.0, abs_tol=5e-3)
        comparison = compare_greeks(fd, aad)
        assert comparison.max_absolute_error < GREEK_TOLERANCE

    def test_partial_greeks_via_wrt(self, call_price_fn) -> None:
        """Requesting a subset of Greeks must not disturb the other leaves."""
        params = {"s0": S0, "sigma": SIGMA}
        full = aad_greeks(call_price_fn, params)
        partial = aad_greeks(call_price_fn, params, wrt=["s0"])

        assert set(partial.greeks) == {"s0"}
        assert math.isclose(partial.greeks["s0"], full.greeks["s0"], rel_tol=1e-9)

    def test_disconnected_parameter_raises(self, simulator: GBMSimulator) -> None:
        """A price_fn that ignores a requested parameter must fail loudly."""

        def broken_price_fn(params):
            # Uses only 's0'; 'sigma' never touches the graph.
            return params["s0"] * 0.0 + torch.as_tensor(1.0, dtype=torch.float64)

        with pytest.raises(ValueError, match="no gradient path"):
            aad_greeks(broken_price_fn, {"s0": S0, "sigma": SIGMA})


class TestGBMSimulator:
    """Sanity checks on the SDE solver independent of any pricer."""

    def test_first_column_equals_spot_exactly(self, simulator: GBMSimulator) -> None:
        dW = simulator.draw_increments(1_000, seed=SEED)
        paths = simulator.simulate(S0, RATE, SIGMA, dW=dW)
        assert torch.allclose(
            paths[:, 0], torch.full_like(paths[:, 0], S0), atol=0.0, rtol=0.0
        )

    def test_paths_stay_strictly_positive(self, simulator: GBMSimulator) -> None:
        dW = simulator.draw_increments(5_000, seed=SEED)
        paths = simulator.simulate(S0, RATE, SIGMA, dW=dW)
        assert torch.all(paths > 0.0)

    def test_terminal_mean_matches_risk_neutral_expectation(
        self, simulator: GBMSimulator
    ) -> None:
        dW = simulator.draw_increments(N_PATHS, seed=SEED)
        paths = simulator.simulate(S0, RATE, SIGMA, dW=dW)
        expected = S0 * math.exp(RATE * MATURITY)
        sample_mean = float(paths[:, -1].mean())
        sample_std = float(paths[:, -1].std())
        std_error = sample_std / math.sqrt(N_PATHS)
        assert abs(sample_mean - expected) < 6.0 * std_error

    def test_dt_matches_maturity_over_steps(self, simulator: GBMSimulator) -> None:
        assert math.isclose(simulator.dt, MATURITY / N_STEPS)

    def test_rejects_mismatched_dw_step_count(self, simulator: GBMSimulator) -> None:
        wrong_steps = simulator.draw_increments(10, seed=SEED)[:, :-1]
        with pytest.raises(ValueError, match="time steps"):
            simulator.simulate(S0, RATE, SIGMA, dW=wrong_steps)
