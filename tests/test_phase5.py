r"""Phase 5 correctness suite: fused O(N)-memory exposure reduction.

What has to be proved
=====================
Phase 5 makes three claims, and each needs a different kind of evidence:

**1. The affine collapse is exact.** The kernel prices a whole netting set as
:math:`V = B_k S - C_k`. If :math:`B, C` disagree with
:func:`src.pricer.options.portfolio_swap_mtm`, every downstream number is wrong
in a way no statistical test would flag. This is checked to floating-point
rounding on CPU, against the Phase 2 implementation, for multi-leg portfolios
including staggered maturities and mixed signs.

**2. Peak memory is independent of M.** Asserted directly with
``torch.cuda.max_memory_allocated`` across ``M in {10k, 100k, 1M}``. The honest
form of this claim is *bounded by a constant*, not *bitwise identical*: the
launch grid is ``min(ceil(M/BLOCK_M), max_programs)``, so a small M launches
fewer programs and allocates slightly **less**. What must never happen is growth
with M -- at 1M paths an O(MN) design would need ~1 GiB, so a constant-bounded
peak of a few MiB is unambiguous.

**3. The fused adjoint is correct.** Verified two ways: against
``torch.autograd`` on a CPU reference implementing the same formulas, and
against central finite differences on the GPU under common random numbers.

Tiering
-------
Tier 1 runs anywhere and covers the affine collapse and the adjoint algebra --
the two places a silent error would hide. Tier 2 needs Triton + CUDA.

Why exact agreement with Phase 4 is not asserted
------------------------------------------------
Phase 4 and Phase 5 use *different* Philox addressing (Phase 4 keys on
``program_id``, Phase 5 on the absolute path-block index), so they draw
different sample paths. Their EE profiles agree only within Monte-Carlo
sampling error, and the cross-check below is written as a statistical
comparison with a standard-error-derived tolerance rather than a tight
elementwise one.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.csrc.triton_cva_fusion import (
    DEFAULT_MAX_PROGRAMS,
    SRAM_TILE_BUDGET_BYTES,
    build_affine_coefficients,
    fused_cva,
    fused_expected_exposure,
    is_available,
    reference_fused_exposure,
    reference_fused_exposure_backward,
    select_fused_block_sizes,
)
from src.models.gbm import GBMSimulator
from src.pricer.options import SwapLeg, portfolio_swap_mtm
from src.xva.cva import compute_unilateral_cva
from src.xva.exposure import expected_exposure

S0 = 100.0
MU = 0.03
RATE = 0.03
SIGMA = 0.20
MATURITY = 1.0
N_STEPS = 252
DT = MATURITY / N_STEPS
SEED = 20260820
HAZARD_RATE = 0.02
RECOVERY_RATE = 0.4

requires_triton = pytest.mark.skipif(
    not is_available(),
    reason="fused exposure kernel requires Triton and a CUDA device",
)


def _legs() -> list[SwapLeg]:
    """A multi-leg netting set: mixed signs and a staggered maturity."""
    return [
        SwapLeg(notional=1.0, strike=100.0, maturity=MATURITY),
        SwapLeg(notional=-0.4, strike=110.0, maturity=MATURITY),
        SwapLeg(notional=0.7, strike=95.0, maturity=0.5),
    ]


# ==========================================================================
# Tier 1 -- the affine collapse (CPU)
# ==========================================================================
class TestAffineCollapse:
    """``V = B*S - C`` must reproduce Phase 2's portfolio MtM exactly.

    This is the highest-leverage check in Phase 5. The kernel never sees a leg;
    it sees two vectors. If those vectors are wrong the exposure profile is
    wrong everywhere, and because the result still *looks* like a plausible
    exposure profile no statistical test would catch it.
    """

    @staticmethod
    def _setup(n_steps: int = 32, n_paths: int = 128):
        simulator = GBMSimulator(
            maturity=MATURITY, n_steps=n_steps, device=torch.device("cpu")
        )
        times = simulator.time_grid()
        dW = simulator.draw_increments(n_paths, seed=SEED)
        paths = simulator.simulate(S0, MU, SIGMA, dW=dW)
        return times, paths

    @pytest.mark.parametrize(
        "legs",
        [
            [SwapLeg(1.0, 100.0, MATURITY)],
            [SwapLeg(1.0, 100.0, MATURITY), SwapLeg(-0.4, 110.0, MATURITY)],
            [
                SwapLeg(1.0, 100.0, MATURITY),
                SwapLeg(-0.4, 110.0, MATURITY),
                SwapLeg(0.7, 95.0, 0.5),
            ],
            [SwapLeg(-2.5, 80.0, 0.25), SwapLeg(3.0, 120.0, 0.75)],
        ],
    )
    def test_matches_portfolio_swap_mtm(self, legs: list[SwapLeg]) -> None:
        times, paths = self._setup()
        reference = portfolio_swap_mtm(paths, times, legs, RATE)

        coeff_b, coeff_c = build_affine_coefficients(legs, times, RATE)
        affine = coeff_b.reshape(1, -1) * paths - coeff_c.reshape(1, -1)

        # Summation order differs, so compare to float64 rounding rather than
        # demanding bitwise equality.
        assert torch.allclose(affine, reference, rtol=1e-13, atol=1e-12), (
            f"max |diff| = {float((affine - reference).abs().max()):.3e}"
        )

    def test_settled_legs_stop_contributing(self) -> None:
        """A leg past its maturity must drop out of both B and C."""
        times, _ = self._setup(n_steps=8)
        legs = [SwapLeg(1.0, 100.0, 0.5)]
        coeff_b, coeff_c = build_affine_coefficients(legs, times, RATE)

        past_maturity = times > 0.5 + 1e-12
        assert torch.all(coeff_b[past_maturity] == 0.0)
        assert torch.all(coeff_c[past_maturity] == 0.0)
        assert torch.all(coeff_b[~past_maturity] > 0.0)

    def test_coefficients_are_detached_constants(self) -> None:
        """B and C must never carry grad: Rho is deliberately unsupported."""
        times, _ = self._setup(n_steps=8)
        coeff_b, coeff_c = build_affine_coefficients(_legs(), times, RATE)
        assert not coeff_b.requires_grad
        assert not coeff_c.requires_grad

    def test_b_is_the_mtm_derivative(self) -> None:
        r"""``B[k]`` must equal :math:`\partial V_{m,k}/\partial S_{m,k}`.

        Checked against autograd, because ``B`` is exactly what the kernel's
        adjoint contracts against -- an error here corrupts Delta and Vega
        while leaving the forward EE perfectly correct.
        """
        n_steps = 16
        simulator = GBMSimulator(
            maturity=MATURITY, n_steps=n_steps, device=torch.device("cpu")
        )
        times = simulator.time_grid()
        legs = _legs()
        coeff_b, _ = build_affine_coefficients(legs, times, RATE)

        paths = torch.full((1, n_steps + 1), S0, dtype=torch.float64, requires_grad=True)
        mtm = portfolio_swap_mtm(paths, times, legs, RATE)
        mtm.sum().backward()
        assert torch.allclose(paths.grad.reshape(-1), coeff_b, rtol=1e-13, atol=1e-13)

    def test_rejects_empty_portfolio(self) -> None:
        times, _ = self._setup(n_steps=4)
        with pytest.raises(ValueError, match="at least one leg"):
            build_affine_coefficients([], times, RATE)


# ==========================================================================
# Tier 1 -- the fused adjoint algebra (CPU)
# ==========================================================================
class TestFusedAdjointOnCPU:
    """Validate the fused adjoint against autograd, no GPU required.

    The fused adjoint is *not* the Phase 4 adjoint: the exposure floor
    introduces an indicator, and the reduction folds a ``1/M`` and the ``B[k]``
    coefficient into the per-step weight. That makes it a third independent
    derivation deserving its own verification.
    """

    @staticmethod
    def _inputs(n_paths: int, n_steps: int):
        generator = torch.Generator().manual_seed(SEED)
        n_columns = n_steps + 1
        z = torch.randn((n_paths, n_columns), dtype=torch.float64, generator=generator)
        times = torch.linspace(0.0, MATURITY, n_columns, dtype=torch.float64)
        coeff_b, coeff_c = build_affine_coefficients(_legs(), times, RATE)
        grad_ee = torch.randn(n_columns, dtype=torch.float64, generator=generator)
        return z, coeff_b, coeff_c, grad_ee

    @pytest.mark.parametrize("n_paths,n_steps", [(1, 1), (3, 4), (17, 9), (256, 64)])
    def test_reference_adjoint_matches_autograd(self, n_paths: int, n_steps: int) -> None:
        z, coeff_b, coeff_c, grad_ee = self._inputs(n_paths, n_steps)
        dt = MATURITY / n_steps

        s0 = torch.tensor(S0, dtype=torch.float64, requires_grad=True)
        mu = torch.tensor(MU, dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)

        profile = reference_fused_exposure(s0, mu, sigma, z, dt, coeff_b, coeff_c)
        profile.backward(grad_ee)

        grad_s0, grad_mu, grad_sigma = reference_fused_exposure_backward(
            grad_ee,
            torch.tensor(S0, dtype=torch.float64),
            torch.tensor(MU, dtype=torch.float64),
            torch.tensor(SIGMA, dtype=torch.float64),
            z,
            dt,
            coeff_b,
            coeff_c,
        )

        assert math.isclose(float(grad_s0), float(s0.grad), rel_tol=1e-10, abs_tol=1e-14)
        assert math.isclose(float(grad_mu), float(mu.grad), rel_tol=1e-10, abs_tol=1e-14)
        assert math.isclose(
            float(grad_sigma), float(sigma.grad), rel_tol=1e-10, abs_tol=1e-14
        )

    def test_column_zero_must_be_excluded_from_mu_and_sigma(self) -> None:
        r"""Including :math:`\iota_0` would double-count every path.

        :math:`L_{m,0} = 0` by definition, so there is no increment at column 0
        and the suffix sum there must not enter the drift or vol contractions.
        The buggy variant is constructed explicitly and shown to differ, so a
        future "simplification" that drops the mask fails loudly.
        """
        z, coeff_b, coeff_c, grad_ee = self._inputs(128, 32)
        dt = MATURITY / 32
        s0 = torch.tensor(S0, dtype=torch.float64)
        sigma = torch.tensor(SIGMA, dtype=torch.float64)

        _, correct_mu, _ = reference_fused_exposure_backward(
            grad_ee, s0, torch.tensor(MU, dtype=torch.float64), sigma,
            z, dt, coeff_b, coeff_c,
        )

        # Reconstruct the buggy version that keeps column 0.
        increments = (MU - 0.5 * SIGMA**2) * dt + SIGMA * math.sqrt(dt) * z
        increments = torch.cat(
            (torch.zeros_like(increments[:, :1]), increments[:, 1:]), dim=1
        )
        paths = S0 * torch.exp(torch.cumsum(increments, dim=1))
        mtm = coeff_b.reshape(1, -1) * paths - coeff_c.reshape(1, -1)
        weight = (grad_ee / z.shape[0]).reshape(1, -1)
        p = weight * coeff_b.reshape(1, -1) * paths * (mtm > 0.0).to(paths.dtype)
        q = torch.flip(torch.cumsum(torch.flip(p, dims=(1,)), dim=1), dims=(1,))
        buggy_mu = float(q.sum() * dt)  # keeps column 0

        assert not math.isclose(float(correct_mu), buggy_mu, rel_tol=1e-9)
        # The discrepancy is exactly the column-0 suffix term.
        assert math.isclose(buggy_mu - float(correct_mu), float(q[:, 0].sum() * dt),
                            rel_tol=1e-10)

    def test_vega_retains_the_ito_correction(self) -> None:
        r"""The :math:`-\sigma\Delta t` term must be present."""
        z, coeff_b, coeff_c, grad_ee = self._inputs(128, 32)
        dt = MATURITY / 32
        s0 = torch.tensor(S0, dtype=torch.float64)
        sigma = torch.tensor(SIGMA, dtype=torch.float64)

        _, _, correct = reference_fused_exposure_backward(
            grad_ee, s0, torch.tensor(MU, dtype=torch.float64), sigma,
            z, dt, coeff_b, coeff_c,
        )
        # Rebuild without the Ito term.
        increments = (MU - 0.5 * SIGMA**2) * dt + SIGMA * math.sqrt(dt) * z
        increments = torch.cat(
            (torch.zeros_like(increments[:, :1]), increments[:, 1:]), dim=1
        )
        paths = S0 * torch.exp(torch.cumsum(increments, dim=1))
        mtm = coeff_b.reshape(1, -1) * paths - coeff_c.reshape(1, -1)
        weight = (grad_ee / z.shape[0]).reshape(1, -1)
        p = weight * coeff_b.reshape(1, -1) * paths * (mtm > 0.0).to(paths.dtype)
        q = torch.flip(torch.cumsum(torch.flip(p, dims=(1,)), dim=1), dims=(1,))
        without_ito = float((q[:, 1:] * (math.sqrt(dt) * z[:, 1:])).sum())

        assert not math.isclose(float(correct), without_ito, rel_tol=1e-9)

    def test_exposure_floor_kills_gradient_where_out_of_the_money(self) -> None:
        """Deeply out-of-the-money exposure must give exactly zero gradient."""
        n_columns = 17
        times = torch.linspace(0.0, MATURITY, n_columns, dtype=torch.float64)
        # A strike far above any attainable spot: exposure is identically zero.
        legs = [SwapLeg(1.0, 1.0e9, MATURITY)]
        coeff_b, coeff_c = build_affine_coefficients(legs, times, RATE)

        generator = torch.Generator().manual_seed(SEED)
        z = torch.randn((64, n_columns), dtype=torch.float64, generator=generator)
        grad_ee = torch.ones(n_columns, dtype=torch.float64)

        profile = reference_fused_exposure(
            torch.tensor(S0, dtype=torch.float64),
            torch.tensor(MU, dtype=torch.float64),
            torch.tensor(SIGMA, dtype=torch.float64),
            z, MATURITY / (n_columns - 1), coeff_b, coeff_c,
        )
        assert float(profile.abs().max()) == 0.0

        grads = reference_fused_exposure_backward(
            grad_ee,
            torch.tensor(S0, dtype=torch.float64),
            torch.tensor(MU, dtype=torch.float64),
            torch.tensor(SIGMA, dtype=torch.float64),
            z, MATURITY / (n_columns - 1), coeff_b, coeff_c,
        )
        for grad in grads:
            assert float(grad) == 0.0

    def test_reference_forward_matches_the_phase2_pipeline(self) -> None:
        """The fused reference must equal expected_exposure(portfolio_swap_mtm).

        Closes the loop on Tier 1: if the fused reference agrees with the
        already-verified Phase 2 pipeline on identical normals, then the only
        thing left to validate on GPU is the kernel itself.
        """
        n_paths, n_steps = 512, 32
        n_columns = n_steps + 1
        dt = MATURITY / n_steps
        times = torch.linspace(0.0, MATURITY, n_columns, dtype=torch.float64)
        legs = _legs()

        generator = torch.Generator().manual_seed(SEED)
        z = torch.randn((n_paths, n_columns), dtype=torch.float64, generator=generator)

        coeff_b, coeff_c = build_affine_coefficients(legs, times, RATE)
        fused = reference_fused_exposure(
            torch.tensor(S0, dtype=torch.float64),
            torch.tensor(MU, dtype=torch.float64),
            torch.tensor(SIGMA, dtype=torch.float64),
            z, dt, coeff_b, coeff_c,
        )

        # Same normals through the Phase 2 route: dW = sqrt(dt) * Z[:, 1:].
        from src.models.gbm import simulate_gbm

        paths = simulate_gbm(S0, MU, SIGMA, math.sqrt(dt) * z[:, 1:], dt)
        unfused = expected_exposure(portfolio_swap_mtm(paths, times, legs, RATE))

        assert torch.allclose(fused, unfused, rtol=1e-12, atol=1e-12), (
            f"max |diff| = {float((fused - unfused).abs().max()):.3e}"
        )


class TestBlockSizeSelection:
    """Single-tile launch configuration, testable without a GPU."""

    @pytest.mark.parametrize("n_steps", [1, 15, 31, 252, 511, 1_023, 2_047])
    @pytest.mark.parametrize("element_size", [4, 8])
    def test_tile_covers_time_axis_and_fits_budget(
        self, n_steps: int, element_size: int
    ) -> None:
        block_m, block_t = select_fused_block_sizes(n_steps, element_size)
        assert block_t >= n_steps + 1, "tile must span the whole time axis"
        assert block_t & (block_t - 1) == 0
        assert block_m & (block_m - 1) == 0
        assert block_m >= 1
        assert block_m * block_t * element_size <= SRAM_TILE_BUDGET_BYTES

    def test_philox_counter_stays_in_int32(self) -> None:
        """BLOCK_M * n_columns must never approach the 32-bit counter limit."""
        for n_steps in (252, 511, 2_047):
            block_m, _ = select_fused_block_sizes(n_steps, 4)
            assert block_m * (n_steps + 1) < 2**31 - 1

    def test_rejects_horizons_too_long_for_a_single_tile(self) -> None:
        """A very long horizon must fail loudly and name the fallback."""
        with pytest.raises(ValueError, match="philox_simulate_gbm"):
            select_fused_block_sizes(100_000, 4)

    def test_rejects_invalid_arguments(self) -> None:
        with pytest.raises(ValueError, match="n_steps must be positive"):
            select_fused_block_sizes(0, 4)
        with pytest.raises(ValueError, match="element_size must be positive"):
            select_fused_block_sizes(252, 0)


class TestGracefulDegradation:
    """Import must succeed and failure must be actionable without Triton/CUDA."""

    def test_module_imports_regardless(self) -> None:
        from src.csrc import triton_cva_fusion

        assert isinstance(triton_cva_fusion.DEFAULT_MAX_PROGRAMS, int)
        assert isinstance(is_available(), bool)

    @pytest.mark.skipif(is_available(), reason="a working Triton+CUDA runtime is present")
    def test_helper_raises_actionable_error(self) -> None:
        times = torch.linspace(0.0, MATURITY, N_STEPS + 1, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Triton is not installed|No CUDA device"):
            fused_expected_exposure(
                S0, MU, SIGMA, _legs(), times, RATE, 1_024, seed=SEED
            )


class TestMemoryScalingArithmetic:
    """The O(N) claim as arithmetic, verifiable without a GPU."""

    def test_predicted_peak_is_independent_of_path_count(self) -> None:
        """``max_programs * (N+1) * element_size`` has no M in it."""
        n_steps, element_size = 252, 4
        predicted = DEFAULT_MAX_PROGRAMS * (n_steps + 1) * element_size
        assert predicted < 8 * 1024 * 1024, "should be a few MiB"

        # What an O(M*N) design would have needed at the same sizes.
        for n_paths in (1_000_000, 10_000_000, 50_000_000):
            unfused = n_paths * (n_steps + 1) * element_size
            assert unfused / predicted > 100, (
                f"M={n_paths:,}: fused {predicted / 1024**2:.1f} MiB vs "
                f"unfused {unfused / 1024**3:.1f} GiB"
            )

    def test_fifty_million_paths_would_not_fit_unfused(self) -> None:
        """The headline Phase 5 claim, as arithmetic."""
        n_steps, element_size = 252, 4
        sixteen_gib = 16 * 1024**3
        unfused_50m = 50_000_000 * (n_steps + 1) * element_size
        fused = DEFAULT_MAX_PROGRAMS * (n_steps + 1) * element_size

        assert unfused_50m > sixteen_gib, "50M paths unfused must exceed 16 GiB"
        assert fused < 0.001 * sixteen_gib, "fused must be a rounding error"


# ==========================================================================
# Tier 2 -- the kernel (GPU only)
# ==========================================================================
@requires_triton
class TestFusedAgainstUnfused:
    """Fused EE and CVA must match the unfused pipeline within sampling error."""

    @staticmethod
    def _times(n_steps: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        return torch.linspace(
            0.0, MATURITY, n_steps + 1, device="cuda", dtype=dtype
        )

    def test_ee_matches_unfused_within_sampling_error(self) -> None:
        """Different Philox addressing means different draws, so compare
        statistically rather than elementwise.
        """
        n_paths, n_steps = 400_000, 64
        times = self._times(n_steps)
        legs = _legs()

        fused = fused_expected_exposure(
            S0, MU, SIGMA, legs, times, RATE, n_paths, seed=SEED
        ).detach()

        simulator = GBMSimulator(
            maturity=MATURITY, n_steps=n_steps,
            device=torch.device("cuda"), dtype=torch.float64,
        )
        dW = simulator.draw_increments(n_paths, seed=SEED + 1)
        paths = simulator.simulate(S0, MU, SIGMA, dW=dW)
        mtm = portfolio_swap_mtm(paths, times, legs, RATE)
        unfused = expected_exposure(mtm).detach()
        # Per-date standard error of the unfused estimator.
        std_error = torch.clamp(mtm, min=0.0).std(dim=0) / math.sqrt(n_paths)

        deviation = (fused - unfused).abs()
        # Independent samples, so the difference has ~sqrt(2) times the SE of one.
        tolerance = 6.0 * math.sqrt(2.0) * std_error + 1e-9
        assert torch.all(deviation <= tolerance), (
            f"worst deviation {float((deviation / (std_error + 1e-30)).max()):.2f} "
            "sample standard errors"
        )

    def test_cva_matches_unfused_within_sampling_error(self) -> None:
        n_paths, n_steps = 400_000, 64
        times = self._times(n_steps)
        legs = _legs()

        _, fused = fused_cva(
            S0, MU, SIGMA, legs, times, RATE, n_paths,
            hazard_rate=HAZARD_RATE, recovery_rate=RECOVERY_RATE, seed=SEED,
        )

        simulator = GBMSimulator(
            maturity=MATURITY, n_steps=n_steps,
            device=torch.device("cuda"), dtype=torch.float64,
        )
        dW = simulator.draw_increments(n_paths, seed=SEED + 1)
        paths = simulator.simulate(S0, MU, SIGMA, dW=dW)
        unfused = compute_unilateral_cva(
            expected_exposure(portfolio_swap_mtm(paths, times, legs, RATE)),
            times, HAZARD_RATE, RECOVERY_RATE, discount_rate=RATE,
        )

        # CVA is a weighted average over dates, so its relative MC error is far
        # smaller than any single EE point's. 1% is generous.
        assert math.isclose(float(fused), float(unfused), rel_tol=1e-2), (
            f"fused {float(fused):.8f} vs unfused {float(unfused):.8f}"
        )

    def test_profile_shape_and_floor(self) -> None:
        times = self._times(N_STEPS)
        profile = fused_expected_exposure(
            S0, MU, SIGMA, _legs(), times, RATE, 100_000, seed=SEED
        )
        assert tuple(profile.shape) == (N_STEPS + 1,)
        assert torch.all(profile >= 0.0)
        assert torch.all(torch.isfinite(profile))

    def test_reproducible_across_runs(self) -> None:
        times = self._times(64)
        first = fused_expected_exposure(
            S0, MU, SIGMA, _legs(), times, RATE, 50_000, seed=SEED
        )
        second = fused_expected_exposure(
            S0, MU, SIGMA, _legs(), times, RATE, 50_000, seed=SEED
        )
        assert torch.equal(first, second)

    def test_grid_size_does_not_change_the_result(self) -> None:
        """Keying Philox on the absolute block index must decouple the result
        from the launch grid -- the improvement over Phase 4's scheme.
        """
        times = self._times(64)
        small_grid = fused_expected_exposure(
            S0, MU, SIGMA, _legs(), times, RATE, 50_000, seed=SEED, max_programs=64
        )
        large_grid = fused_expected_exposure(
            S0, MU, SIGMA, _legs(), times, RATE, 50_000, seed=SEED, max_programs=4096
        )
        # Partial-sum ordering differs, so allow float rounding but nothing more.
        assert torch.allclose(small_grid, large_grid, rtol=1e-11, atol=1e-12)

    def test_rejects_rate_requiring_grad(self) -> None:
        """Refusing a half-answer for Rho, rather than silently giving one."""
        times = self._times(32)
        rate = torch.tensor(RATE, device="cuda", dtype=torch.float64, requires_grad=True)
        with pytest.raises(ValueError, match="total Rho"):
            fused_expected_exposure(
                S0, MU, SIGMA, _legs(), times, rate, 1_024, seed=SEED
            )

    def test_rejects_non_uniform_grid(self) -> None:
        times = self._times(32).clone()
        times[5] += 0.001
        with pytest.raises(ValueError, match="uniform"):
            fused_expected_exposure(
                S0, MU, SIGMA, _legs(), times, RATE, 1_024, seed=SEED
            )


@requires_triton
class TestMemoryIndependenceOfM:
    """Peak allocation must not grow with the path count."""

    @pytest.mark.parametrize("n_paths", [10_000, 100_000, 1_000_000])
    def test_peak_memory_stays_bounded(self, n_paths: int) -> None:
        n_steps = N_STEPS
        times = torch.linspace(
            0.0, MATURITY, n_steps + 1, device="cuda", dtype=torch.float32
        )

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        profile = fused_expected_exposure(
            S0, MU, SIGMA, _legs(), times, RATE, n_paths, seed=SEED
        )
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()

        # The fused ceiling: partial buffer + a handful of length-N vectors.
        ceiling = DEFAULT_MAX_PROGRAMS * (n_steps + 1) * 4 * 4  # 4x headroom
        assert peak < ceiling, (
            f"M={n_paths:,}: peak {peak / 1024**2:,.2f} MiB exceeds the "
            f"M-independent ceiling {ceiling / 1024**2:,.2f} MiB"
        )

        # And it must be nowhere near what an O(M*N) design would need.
        unfused = n_paths * (n_steps + 1) * 4
        if n_paths >= 100_000:
            assert peak < 0.2 * unfused, (
                f"M={n_paths:,}: peak {peak / 1024**2:,.1f} MiB is not clearly "
                f"below the O(M*N) requirement {unfused / 1024**2:,.1f} MiB"
            )
        del profile

    def test_peak_is_flat_across_two_decades_of_paths(self) -> None:
        """The direct statement of the claim: 100x more paths, same memory.

        Peaks are compared as a *ratio*, which is the meaningful form. They are
        not asserted bitwise identical: the launch grid is
        ``min(ceil(M/BLOCK_M), max_programs)``, so a small M legitimately
        launches fewer programs and allocates slightly less.
        """
        n_steps = N_STEPS
        times = torch.linspace(
            0.0, MATURITY, n_steps + 1, device="cuda", dtype=torch.float32
        )
        peaks = {}
        for n_paths in (10_000, 100_000, 1_000_000):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            profile = fused_expected_exposure(
                S0, MU, SIGMA, _legs(), times, RATE, n_paths, seed=SEED
            )
            torch.cuda.synchronize()
            peaks[n_paths] = torch.cuda.max_memory_allocated()
            del profile

        largest, smallest = max(peaks.values()), min(peaks.values())
        # assert largest / smallest < 15*1024**2, (
        #     "peak memory scales with M: "
        #     + ", ".join(f"M={m:,}->{p / 1024**2:.2f} MiB" for m, p in peaks.items())
        # )
        assert largest < 15 * 1024**2, (
            f"Memory is not O(1)! Peaked at {largest / 1024**2:.2f} MiB for M={max(peaks.keys())}"
        )
        # A 100x path increase must not cost anything like 100x memory.
        assert peaks[1_000_000] < 10 * peaks[10_000]


@requires_triton
class TestFusedGreeks:
    """AAD Greeks from the fused kernel vs central finite differences."""

    N_PATHS = 200_000
    N_STEPS = 64
    STEP = MATURITY / 64

    @classmethod
    def _cva(cls, s0, mu, sigma, hazard, times) -> torch.Tensor:
        _, cva = fused_cva(
            s0, mu, sigma, _legs(), times, RATE, cls.N_PATHS,
            hazard_rate=hazard, recovery_rate=RECOVERY_RATE, seed=SEED,
        )
        return cva

    @classmethod
    def _times(cls) -> torch.Tensor:
        return torch.linspace(
            0.0, MATURITY, cls.N_STEPS + 1, device="cuda", dtype=torch.float64
        )

    def test_greeks_match_central_differences(self) -> None:
        r"""Delta, drift sensitivity, Vega and credit sensitivity vs FD.

        The seed pins the sample, so a bump redraws identical normals and FD is
        an exact oracle. The exposure floor makes CVA only piecewise smooth, so
        finite differences carry an O(h) kink bias -- hence a relative tolerance
        rather than machine precision.
        """
        times = self._times()

        s0 = torch.tensor(S0, device="cuda", dtype=torch.float64, requires_grad=True)
        mu = torch.tensor(MU, device="cuda", dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(
            SIGMA, device="cuda", dtype=torch.float64, requires_grad=True
        )
        hazard = torch.tensor(
            HAZARD_RATE, device="cuda", dtype=torch.float64, requires_grad=True
        )

        self._cva(s0, mu, sigma, hazard, times).backward()
        aad = {
            "s0": float(s0.grad),
            "mu": float(mu.grad),
            "sigma": float(sigma.grad),
            "hazard": float(hazard.grad),
        }

        base = {"s0": S0, "mu": MU, "sigma": SIGMA, "hazard": HAZARD_RATE}
        for name in ("s0", "mu", "sigma", "hazard"):
            step = 1e-5 * max(abs(base[name]), 1.0)
            with torch.no_grad():
                up, down = dict(base), dict(base)
                up[name] = base[name] + step
                down[name] = base[name] - step
                value_up = float(
                    self._cva(up["s0"], up["mu"], up["sigma"], up["hazard"], times)
                )
                value_down = float(
                    self._cva(
                        down["s0"], down["mu"], down["sigma"], down["hazard"], times
                    )
                )
            finite_difference = (value_up - value_down) / (2.0 * step)

            assert math.isclose(
                aad[name], finite_difference, rel_tol=2e-3, abs_tol=1e-9
            ), f"{name}: AAD {aad[name]!r} vs FD {finite_difference!r}"

    def test_all_greeks_are_nonzero(self) -> None:
        """Guard the guard: a zero Greek would make the FD check vacuous."""
        times = self._times()
        s0 = torch.tensor(S0, device="cuda", dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(
            SIGMA, device="cuda", dtype=torch.float64, requires_grad=True
        )
        hazard = torch.tensor(
            HAZARD_RATE, device="cuda", dtype=torch.float64, requires_grad=True
        )
        self._cva(s0, MU, sigma, hazard, times).backward()

        assert abs(float(s0.grad)) > 1e-8
        assert abs(float(sigma.grad)) > 1e-8
        assert abs(float(hazard.grad)) > 1e-8

    def test_credit_sensitivity_matches_closed_form(self) -> None:
        r"""``dCVA/dlambda`` has a closed form once EE is held fixed.

        :math:`EE` does not depend on :math:`\lambda`, so
        :math:`\partial CVA/\partial\lambda
        = (1-R)\sum_k EE_k DF_k (t_k e^{-\lambda t_k}
        - t_{k-1} e^{-\lambda t_{k-1}})`.
        Because the credit integral is left in PyTorch rather than fused into
        the kernel, this is an exact check on the whole tail.
        """
        times = self._times()
        profile = fused_expected_exposure(
            S0, MU, SIGMA, _legs(), times, RATE, self.N_PATHS, seed=SEED
        ).detach()

        hazard = torch.tensor(
            HAZARD_RATE, device="cuda", dtype=torch.float64, requires_grad=True
        )
        compute_unilateral_cva(
            profile, times, hazard, RECOVERY_RATE, discount_rate=RATE
        ).backward()

        discount = torch.exp(-RATE * times)
        derivative = (
            times[1:] * torch.exp(-HAZARD_RATE * times[1:])
            - times[:-1] * torch.exp(-HAZARD_RATE * times[:-1])
        )
        expected = float(
            (1.0 - RECOVERY_RATE) * torch.sum(profile[1:] * discount[1:] * derivative)
        )
        assert math.isclose(float(hazard.grad), expected, rel_tol=1e-10)

    def test_partial_gradients_are_respected(self) -> None:
        times = self._times()
        s0 = torch.tensor(S0, device="cuda", dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(SIGMA, device="cuda", dtype=torch.float64)  # no grad
        self._cva(s0, MU, sigma, HAZARD_RATE, times).backward()

        assert s0.grad is not None
        assert sigma.grad is None

    def test_backward_is_deterministic(self) -> None:
        times = self._times()
        results = []
        for _ in range(2):
            s0 = torch.tensor(
                S0, device="cuda", dtype=torch.float64, requires_grad=True
            )
            sigma = torch.tensor(
                SIGMA, device="cuda", dtype=torch.float64, requires_grad=True
            )
            self._cva(s0, MU, sigma, HAZARD_RATE, times).backward()
            results.append((float(s0.grad), float(sigma.grad)))
        assert results[0] == results[1]

    def test_backward_memory_is_also_independent_of_m(self) -> None:
        """Rematerialisation means the adjoint allocates nothing of size M."""
        n_steps = self.N_STEPS
        times = torch.linspace(
            0.0, MATURITY, n_steps + 1, device="cuda", dtype=torch.float32
        )
        ceiling = DEFAULT_MAX_PROGRAMS * (n_steps + 1) * 4 * 6  # generous headroom

        for n_paths in (100_000, 2_000_000):
            s0 = torch.tensor(
                S0, device="cuda", dtype=torch.float32, requires_grad=True
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            profile = fused_expected_exposure(
                s0, MU, SIGMA, _legs(), times, RATE, n_paths, seed=SEED
            )
            profile.sum().backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()

            assert peak < ceiling, (
                f"M={n_paths:,}: backward peak {peak / 1024**2:,.2f} MiB exceeds "
                f"the M-independent ceiling {ceiling / 1024**2:,.2f} MiB"
            )

    def test_double_backward_raises_rather_than_lying(self) -> None:
        times = self._times()
        s0 = torch.tensor(S0, device="cuda", dtype=torch.float64, requires_grad=True)
        profile = fused_expected_exposure(
            s0, MU, SIGMA, _legs(), times, RATE, 1_024, seed=SEED
        )
        (grad_s0,) = torch.autograd.grad(profile.sum(), s0, create_graph=True)
        with pytest.raises(RuntimeError):
            torch.autograd.grad(grad_s0, s0)
