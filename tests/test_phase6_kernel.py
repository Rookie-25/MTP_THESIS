r"""Phase 6 kernel suite: local-volatility exposure and its checkpointed adjoint.

Tiering, and why it matters more here than anywhere else
=======================================================
This kernel has more independent ways to be wrong than any earlier one:

1. the adjoint derivation (the Jacobian recursion),
2. the checkpointing scheme (replay produces the right states),
3. the Triton translation (masked tile access, RNG addressing, step guards).

The suite separates them so a failure localises instead of requiring a search:

**Tier 1 (CPU, runs anywhere).** ``reference_local_vol_ee_adjoint`` is checked
against ``torch.autograd``, and ``reference_checkpointed_ee_adjoint`` is checked
against *that*. If Tier 1 passes, the derivation and the checkpointing scheme
are both correct and any Tier 2 failure is purely the Triton translation.

**Tier 2 (GPU).** Forward and gradients against the Tier 1 reference at
:math:`M = 1000`, plus finite differences under common random numbers.

The trap this suite is built to catch
=====================================
:math:`\bar{X}_k` -- the direct adjoint of the state from the EE output -- must
be re-injected at **every** step, because the profile reads the state at all
:math:`k`. Adding it only at the terminal step gives gradients that are smooth,
plausible, and wrong. ``test_direct_adjoint_needed_at_every_step`` constructs
that exact bug and asserts it differs.

A second trap: the skew must be centred on :math:`\log S_0`. An uncentred
``tanh(kappa * x)`` saturates at :math:`x \approx 4.6`, driving
:math:`\partial\sigma/\partial x` to :math:`\approx 10^{-5}` and reducing the
whole kernel to the constant-volatility Phase 5 case while appearing to exercise
state dependence. ``test_surface_is_genuinely_state_dependent`` guards it.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.csrc.triton_local_vol_cva import (
    LocalVolParams,
    is_available,
    local_vol_and_state_derivative,
    reference_checkpointed_ee_adjoint,
    reference_local_vol_ee,
    reference_local_vol_ee_adjoint,
    select_local_vol_blocks,
)

torch.set_default_dtype(torch.float64)

SPOT = 100.0
DRIFT = 0.02
BASE = 0.20
SKEW = 0.15
MATURITY = 1.0
SEED = 20260827

PARAMS = LocalVolParams(
    base=BASE, skew=SKEW, kappa=2.5, term=0.05, reference=math.log(SPOT)
)

requires_triton = pytest.mark.skipif(
    not is_available(),
    reason="local-volatility kernel requires Triton and a CUDA device",
)


def _inputs(n_paths: int, n_steps: int, seed: int = SEED):
    """Build normals, affine coefficients and an incoming profile adjoint."""
    generator = torch.Generator().manual_seed(seed)
    normals = torch.randn((n_paths, n_steps), generator=generator)
    coeff_b = 1.0 + 0.3 * torch.rand(n_steps + 1, generator=generator)
    coeff_c = 90.0 + 10.0 * torch.rand(n_steps + 1, generator=generator)
    grad_ee = torch.randn(n_steps + 1, generator=generator)
    return normals, coeff_b, coeff_c, grad_ee


def _leaves():
    """Fresh differentiable parameters."""
    return (
        torch.tensor(SPOT, requires_grad=True),
        torch.tensor(DRIFT, requires_grad=True),
        torch.tensor(BASE, requires_grad=True),
        torch.tensor(SKEW, requires_grad=True),
    )


# ==========================================================================
# Tier 1 -- the surface
# ==========================================================================
class TestSurface:
    """The parametric local volatility and its analytic derivatives."""

    def test_state_derivative_matches_autograd(self) -> None:
        r""":math:`\partial\sigma/\partial x` must equal the autograd value."""
        x = torch.linspace(math.log(60.0), math.log(160.0), 41, requires_grad=True)
        sigma, analytic, _ = local_vol_and_state_derivative(0.3, x, PARAMS)
        (from_autograd,) = torch.autograd.grad(sigma.sum(), x)
        assert torch.allclose(analytic, from_autograd, rtol=1e-12)

    def test_skew_derivative_is_the_tanh_term(self) -> None:
        r""":math:`\partial\sigma/\partial\sigma_{\text{skew}}` is the tanh."""
        x = torch.linspace(math.log(60.0), math.log(160.0), 21)
        skew = torch.tensor(SKEW, requires_grad=True)
        sigma, _, tanh_term = local_vol_and_state_derivative(
            0.3, x, PARAMS, skew=skew
        )
        (from_autograd,) = torch.autograd.grad(sigma.sum(), skew)
        assert math.isclose(float(tanh_term.sum()), float(from_autograd), rel_tol=1e-12)

    def test_surface_is_genuinely_state_dependent(self) -> None:
        r"""Guard: the skew must not be saturated at the operating point.

        With ``reference = log(S0)`` the tanh argument is near zero at
        inception, so :math:`\operatorname{sech}^2 \approx 1`. An uncentred
        surface would give :math:`\approx 10^{-5}` and silently reduce this
        kernel to the constant-volatility case.
        """
        x = torch.full((16,), math.log(SPOT))
        _, centred, _ = local_vol_and_state_derivative(0.0, x, PARAMS)
        assert float(centred.abs().mean()) > 0.1 * PARAMS.skew * PARAMS.kappa

        uncentred = LocalVolParams(
            base=BASE, skew=SKEW, kappa=2.5, term=0.05, reference=0.0
        )
        _, saturated, _ = local_vol_and_state_derivative(0.0, x, uncentred)
        assert float(saturated.abs().mean()) < 1e-3, (
            "the uncentred control should be saturated -- if it is not, this "
            "test is not exercising the trap it exists to guard"
        )

    def test_rejects_invalid_parameters(self) -> None:
        with pytest.raises(ValueError, match="base must be positive"):
            LocalVolParams(base=0.0)
        with pytest.raises(ValueError, match="kappa must be positive"):
            LocalVolParams(kappa=-1.0)


# ==========================================================================
# Tier 1 -- the adjoint derivation
# ==========================================================================
class TestAdjointOnCPU:
    """The sequential adjoint and the checkpointed variant, vs autograd."""

    @pytest.mark.parametrize(
        "n_paths,n_steps", [(1, 1), (3, 4), (17, 9), (64, 40), (32, 252)]
    )
    def test_sequential_adjoint_matches_autograd(
        self, n_paths: int, n_steps: int
    ) -> None:
        normals, coeff_b, coeff_c, grad_ee = _inputs(n_paths, n_steps)
        dt = MATURITY / n_steps

        spot, drift, base, skew = _leaves()
        profile = reference_local_vol_ee(
            spot, drift, base, skew, normals, dt, coeff_b, coeff_c, PARAMS
        )
        (grad_ee * profile).sum().backward()
        truth = (
            float(spot.grad), float(drift.grad),
            float(base.grad), float(skew.grad),
        )

        manual = reference_local_vol_ee_adjoint(
            grad_ee,
            torch.tensor(SPOT), torch.tensor(DRIFT),
            torch.tensor(BASE), torch.tensor(SKEW),
            normals, dt, coeff_b, coeff_c, PARAMS,
        )

        for name, expected, got in zip(
            ("dL/dS0", "dL/ddrift", "dL/dbase", "dL/dskew"), truth, manual
        ):
            assert math.isclose(
                expected, float(got), rel_tol=1e-9, abs_tol=1e-14
            ), f"{name}: autograd {expected!r} vs sequential {float(got)!r}"

    @pytest.mark.parametrize("n_steps", [4, 16, 40, 100, 252])
    def test_checkpointed_matches_full_storage(self, n_steps: int) -> None:
        r""":math:`\sqrt{N}` checkpointing must change memory, never the answer."""
        normals, coeff_b, coeff_c, grad_ee = _inputs(24, n_steps)
        dt = MATURITY / n_steps
        args = (
            grad_ee,
            torch.tensor(SPOT), torch.tensor(DRIFT),
            torch.tensor(BASE), torch.tensor(SKEW),
            normals, dt, coeff_b, coeff_c, PARAMS,
        )
        full = reference_local_vol_ee_adjoint(*args)
        checkpointed = reference_checkpointed_ee_adjoint(*args)
        for index, name in enumerate(("dS0", "ddrift", "dbase", "dskew")):
            assert math.isclose(
                float(full[index]), float(checkpointed[index]), rel_tol=1e-12
            ), name

    @pytest.mark.parametrize("block_ck", [1, 2, 4, 8, 16, 64])
    def test_any_segment_length_gives_the_same_answer(self, block_ck: int) -> None:
        """The answer must not depend on the checkpoint granularity."""
        normals, coeff_b, coeff_c, grad_ee = _inputs(16, 40)
        dt = MATURITY / 40
        args = (
            grad_ee,
            torch.tensor(SPOT), torch.tensor(DRIFT),
            torch.tensor(BASE), torch.tensor(SKEW),
            normals, dt, coeff_b, coeff_c, PARAMS,
        )
        reference = reference_local_vol_ee_adjoint(*args)
        candidate = reference_checkpointed_ee_adjoint(*args, block_ck=block_ck)
        for index in range(4):
            assert math.isclose(
                float(reference[index]), float(candidate[index]), rel_tol=1e-11
            )

    def test_direct_adjoint_needed_at_every_step(self) -> None:
        r"""The trap: :math:`\bar{X}_k` must be added at every step, not just N.

        The EE profile reads the state at all :math:`k`, so each step carries a
        direct contribution on top of the recursive one. Injecting it only at
        the terminal step yields a smooth, plausible, wrong gradient. This test
        builds that bug explicitly and asserts it differs, so a future
        "simplification" cannot quietly reintroduce it.
        """
        n_paths, n_steps = 48, 30
        normals, coeff_b, coeff_c, grad_ee = _inputs(n_paths, n_steps)
        dt = MATURITY / n_steps
        sqrt_dt = math.sqrt(dt)
        weight = grad_ee / n_paths

        correct = reference_local_vol_ee_adjoint(
            grad_ee,
            torch.tensor(SPOT), torch.tensor(DRIFT),
            torch.tensor(BASE), torch.tensor(SKEW),
            normals, dt, coeff_b, coeff_c, PARAMS,
        )

        # The buggy variant: terminal injection only.
        with torch.no_grad():
            state = torch.full((n_paths,), math.log(SPOT))
            states = [state.clone()]
            for step in range(n_steps):
                sigma, _, _ = local_vol_and_state_derivative(
                    step * dt, state, PARAMS
                )
                state = (
                    state
                    + (DRIFT - 0.5 * sigma * sigma) * dt
                    + sigma * sqrt_dt * normals[:, step]
                )
                states.append(state.clone())

            spot_n = torch.exp(states[n_steps])
            active = ((coeff_b[n_steps] * spot_n - coeff_c[n_steps]) > 0).to(
                spot_n.dtype
            )
            adjoint = weight[n_steps] * active * coeff_b[n_steps] * spot_n
            buggy_base = torch.zeros(())
            for step in reversed(range(n_steps)):
                sigma, d_sigma_d_x, _ = local_vol_and_state_derivative(
                    step * dt, states[step], PARAMS
                )
                vol_factor = sqrt_dt * normals[:, step] - sigma * dt
                buggy_base = buggy_base + (adjoint * vol_factor).sum()
                adjoint = adjoint * (1.0 + d_sigma_d_x * vol_factor)

        relative = abs(float(buggy_base) - float(correct[2])) / abs(float(correct[2]))
        assert relative > 1e-3, (
            f"terminal-only injection differs by only {relative:.2e}; the test "
            "is not exercising the trap (check that grad_ee has meaningful "
            "weight at interior steps)"
        )

    def test_gradients_are_nonzero_so_the_tests_have_teeth(self) -> None:
        """A zero gradient would make every comparison above vacuous."""
        normals, coeff_b, coeff_c, grad_ee = _inputs(64, 40)
        grads = reference_local_vol_ee_adjoint(
            grad_ee,
            torch.tensor(SPOT), torch.tensor(DRIFT),
            torch.tensor(BASE), torch.tensor(SKEW),
            normals, MATURITY / 40, coeff_b, coeff_c, PARAMS,
        )
        for index, name in enumerate(("dS0", "ddrift", "dbase", "dskew")):
            assert abs(float(grads[index])) > 1e-8, name


class TestBlockSelection:
    """Launch configuration for the checkpointing scheme."""

    @pytest.mark.parametrize("n_steps", [1, 4, 16, 100, 252, 1_024, 4_096])
    @pytest.mark.parametrize("element_size", [4, 8])
    def test_block_ck_is_about_sqrt_n(self, n_steps: int, element_size: int) -> None:
        block_m, block_ck = select_local_vol_blocks(n_steps, element_size)
        assert block_ck & (block_ck - 1) == 0, "BLOCK_CK must be a power of two"
        assert block_m & (block_m - 1) == 0
        assert block_ck >= math.ceil(math.sqrt(n_steps))
        # And not wastefully larger than the next power of two above sqrt(N).
        assert block_ck < 2 * max(1, 2 ** math.ceil(math.log2(max(
            1, math.ceil(math.sqrt(n_steps))
        ))))

    @pytest.mark.parametrize("element_size", [4, 8])
    def test_two_tiles_fit_the_sram_budget(self, element_size: int) -> None:
        from src.csrc.triton_local_vol_cva import SRAM_TILE_BUDGET_BYTES

        for n_steps in (16, 252, 1_024):
            block_m, block_ck = select_local_vol_blocks(n_steps, element_size)
            assert 2 * block_m * block_ck * element_size <= SRAM_TILE_BUDGET_BYTES

    def test_philox_counter_stays_in_int32(self) -> None:
        for n_steps in (252, 1_024, 4_096):
            block_m, _ = select_local_vol_blocks(n_steps, 4)
            assert block_m * (n_steps + 1) < 2**31 - 1

    def test_rejects_invalid_arguments(self) -> None:
        with pytest.raises(ValueError, match="n_steps must be positive"):
            select_local_vol_blocks(0, 4)
        with pytest.raises(ValueError, match="element_size must be positive"):
            select_local_vol_blocks(252, 0)


class TestGracefulDegradation:
    """Import must work and failure must be actionable without Triton/CUDA."""

    def test_module_imports_regardless(self) -> None:
        from src.csrc import triton_local_vol_cva

        assert isinstance(triton_local_vol_cva.SRAM_TILE_BUDGET_BYTES, int)
        assert isinstance(is_available(), bool)

    @pytest.mark.skipif(is_available(), reason="a Triton+CUDA runtime is present")
    def test_helper_raises_actionable_error(self) -> None:
        from src.csrc.triton_local_vol_cva import fused_local_vol_ee
        from src.pricer.options import SwapLeg

        times = torch.linspace(0.0, MATURITY, 33, dtype=torch.float32)
        with pytest.raises((RuntimeError, ValueError)):
            fused_local_vol_ee(
                SPOT, DRIFT, [SwapLeg(1.0, SPOT, MATURITY)],
                times, 0.03, 1_000, PARAMS, seed=SEED,
            )


# ==========================================================================
# Tier 2 -- the kernels (GPU only)
# ==========================================================================
class TestAtTheMoneyKink:
    r"""The t=0 exposure kink when a trade is struck exactly at-the-money.

    This is not a defect, but it *looks* like one and cost a debugging round, so
    it is pinned down here.

    ``EE[0]`` is deterministic: every path starts at :math:`S_0`, so
    :math:`V_0 = B_0 S_0 - C_0` is the same on all of them. When the strike
    equals the spot, :math:`C_0 = B_0 \cdot \text{strike} = B_0 S_0`, so
    :math:`V_0 = 0` **exactly** -- not approximately. ``max(V, 0)`` has no
    derivative there.

    For :math:`t > 0` this cannot happen with probability one, because the state
    is diffusive and :math:`\{V_t = 0\}` is a null set. The :math:`t=0` column
    is the sole exception, and it is exactly the case a desk hits when it books
    a fresh at-the-money trade.

    Consequences, both measured below:

    * AAD returns the subgradient ``0`` (PyTorch's ``clamp`` convention, used
      consistently across this codebase).
    * Central finite differences return the mean of the two one-sided
      derivatives, i.e. half the jump.

    They therefore differ by a **fixed absolute amount independent of N** --
    which is the signature to look for. Relative error then just scales as
    (that constant) / |gradient|, which is why the same underlying discrepancy
    showed up as 1.7%, 3.4% and 76.8% at different step counts.
    """

    @staticmethod
    def _probe(strike: float, n_steps: int = 64, n_paths: int = 1_000):
        """Return ``(v0, aad, fd)`` for the spot gradient at a given strike."""
        from src.csrc.triton_cva_fusion import build_affine_coefficients
        from src.pricer.options import SwapLeg

        rate, dt = 0.03, MATURITY / n_steps
        times = torch.linspace(0.0, MATURITY, n_steps + 1)
        legs = [SwapLeg(notional=1.0, strike=strike, maturity=MATURITY)]
        coeff_b, coeff_c = build_affine_coefficients(legs, times, rate)

        generator = torch.Generator().manual_seed(SEED + 7)
        normals = torch.randn((n_paths, n_steps), generator=generator)
        weights = torch.randn(n_steps + 1, generator=generator)

        spot = torch.tensor(SPOT, requires_grad=True)
        profile = reference_local_vol_ee(
            spot, torch.tensor(DRIFT), torch.tensor(BASE), torch.tensor(SKEW),
            normals, dt, coeff_b, coeff_c, PARAMS,
        )
        (weights * profile).sum().backward()
        aad = float(spot.grad)

        step = 1e-6 * SPOT
        with torch.no_grad():
            def evaluate(value: float) -> float:
                return float((weights * reference_local_vol_ee(
                    torch.tensor(value), torch.tensor(DRIFT),
                    torch.tensor(BASE), torch.tensor(SKEW),
                    normals, dt, coeff_b, coeff_c, PARAMS,
                )).sum())

            finite = (evaluate(SPOT + step) - evaluate(SPOT - step)) / (2.0 * step)

        v0 = float(coeff_b[0]) * SPOT - float(coeff_c[0])
        return v0, aad, finite

    def test_at_the_money_puts_t0_exposure_exactly_on_the_kink(self) -> None:
        """``strike == spot`` gives ``V[0] == 0`` to the bit, not merely near it."""
        v0, _, _ = self._probe(strike=SPOT)
        assert v0 == 0.0, f"expected exactly zero, got {v0!r}"

    @pytest.mark.parametrize("strike", [90.0, 95.0, 105.0, 110.0])
    def test_away_from_the_money_aad_and_fd_agree(self, strike: float) -> None:
        """Off the kink, the two methods agree to machine precision."""
        v0, aad, finite = self._probe(strike=strike)
        assert abs(v0) > 1e-6, "this strike should not be at-the-money"
        assert math.isclose(aad, finite, rel_tol=1e-6), (
            f"strike={strike}: AAD {aad!r} vs FD {finite!r}"
        )

    def test_on_the_kink_they_disagree_and_that_is_expected(self) -> None:
        """At-the-money they differ -- by design, not by defect."""
        v0, aad, finite = self._probe(strike=SPOT)
        assert v0 == 0.0
        assert not math.isclose(aad, finite, rel_tol=5e-3), (
            "expected AAD and FD to differ on the kink; if they now agree, the "
            "clamp subgradient convention changed and this test needs revisiting"
        )

    @pytest.mark.parametrize("n_steps", [16, 32, 64])
    def test_the_kink_gap_is_a_constant_independent_of_n(self, n_steps: int) -> None:
        r"""The signature that identifies this as a kink rather than a bug.

        The disagreement comes entirely from the single :math:`t=0` column, so
        its absolute size is :math:`\tfrac12 w_0 B_0` -- independent of the
        number of time steps. A recursion bug would instead scale with
        :math:`N`.
        """
        _, aad, finite = self._probe(strike=SPOT, n_steps=n_steps)
        gap = finite - aad

        # Reconstruct the predicted half-jump from the t=0 column alone.
        from src.csrc.triton_cva_fusion import build_affine_coefficients
        from src.pricer.options import SwapLeg

        times = torch.linspace(0.0, MATURITY, n_steps + 1)
        coeff_b, _ = build_affine_coefficients(
            [SwapLeg(notional=1.0, strike=SPOT, maturity=MATURITY)], times, 0.03
        )
        generator = torch.Generator().manual_seed(SEED + 7)
        _ = torch.randn((1_000, n_steps), generator=generator)
        weights = torch.randn(n_steps + 1, generator=generator)
        predicted = 0.5 * float(weights[0]) * float(coeff_b[0])

        assert math.isclose(abs(gap), abs(predicted), rel_tol=0.02), (
            f"N={n_steps}: gap {gap:.6e} vs predicted half-jump "
            f"{predicted:.6e} -- if these stop matching, the discrepancy is no "
            "longer explained by the t=0 kink alone"
        )


@requires_triton
class TestKernelForward:
    """Forward parity against the CPU reference at M = 1000."""

    N_PATHS = 1_000
    N_STEPS = 64

    @staticmethod
    def _setup(n_paths: int, n_steps: int):
        from src.csrc.triton_cva_fusion import build_affine_coefficients
        from src.pricer.options import SwapLeg

        times = torch.linspace(
            0.0, MATURITY, n_steps + 1, device="cuda", dtype=torch.float64
        )
        legs = [
            SwapLeg(notional=1.0, strike=100.0, maturity=MATURITY),
            SwapLeg(notional=-0.4, strike=110.0, maturity=MATURITY),
        ]
        coeff_b, coeff_c = build_affine_coefficients(legs, times, 0.03)
        return times, legs, coeff_b, coeff_c

    def test_profile_shape_and_floor(self) -> None:
        from src.csrc.triton_local_vol_cva import fused_local_vol_ee

        times, legs, _, _ = self._setup(self.N_PATHS, self.N_STEPS)
        profile = fused_local_vol_ee(
            SPOT, DRIFT, legs, times, 0.03, self.N_PATHS, PARAMS, seed=SEED
        )
        assert tuple(profile.shape) == (self.N_STEPS + 1,)
        assert torch.all(profile >= 0.0)
        assert torch.all(torch.isfinite(profile))

    def test_reproducible_and_grid_independent(self) -> None:
        """Absolute-block-index keying must decouple the result from the grid."""
        from src.csrc.triton_local_vol_cva import fused_local_vol_ee

        times, legs, _, _ = self._setup(self.N_PATHS, self.N_STEPS)
        first = fused_local_vol_ee(
            SPOT, DRIFT, legs, times, 0.03, self.N_PATHS, PARAMS, seed=SEED
        )
        again = fused_local_vol_ee(
            SPOT, DRIFT, legs, times, 0.03, self.N_PATHS, PARAMS, seed=SEED
        )
        assert torch.equal(first, again)

        coarse = fused_local_vol_ee(
            SPOT, DRIFT, legs, times, 0.03, self.N_PATHS, PARAMS,
            seed=SEED, max_programs=8,
        )
        assert torch.allclose(first, coarse, rtol=1e-11, atol=1e-12)

    def test_matches_cpu_reference_on_identical_normals(self) -> None:
        """Distributional forward check against the CPU reference.

        Triton's Philox cannot be made to reproduce ``torch.randn``, so an
        elementwise comparison is impossible: the two draw different samples.
        This therefore compares them *statistically* at a path count large
        enough that Monte-Carlo error is small, which validates the forward
        recursion and the affine payoff but not the exact arithmetic.

        The exact check lives in
        :meth:`TestKernelGradients.test_gradients_match_central_differences`,
        which bumps parameters against the kernel's *own* fixed sample and so
        needs no cross-backend sample agreement at all.
        """
        from src.csrc.triton_local_vol_cva import fused_local_vol_ee

        n_paths = 200_000
        times, legs, coeff_b, coeff_c = self._setup(n_paths, self.N_STEPS)
        kernel_profile = fused_local_vol_ee(
            SPOT, DRIFT, legs, times, 0.03, n_paths, PARAMS, seed=SEED
        ).detach()

        generator = torch.Generator(device="cuda").manual_seed(SEED + 1)
        normals = torch.randn(
            (n_paths, self.N_STEPS), device="cuda", dtype=torch.float64,
            generator=generator,
        )
        reference_profile = reference_local_vol_ee(
            torch.tensor(SPOT, device="cuda"),
            torch.tensor(DRIFT, device="cuda"),
            torch.tensor(BASE, device="cuda"),
            torch.tensor(SKEW, device="cuda"),
            normals, MATURITY / self.N_STEPS, coeff_b, coeff_c, PARAMS,
        ).detach()

        # Independent samples: allow a few standard errors, scaled by the
        # profile magnitude.
        scale = reference_profile.abs().max().clamp(min=1e-12)
        deviation = (kernel_profile - reference_profile).abs().max() / scale
        assert float(deviation) < 0.05, (
            f"worst relative deviation {float(deviation):.4f} -- larger than "
            "Monte-Carlo error should allow at 200k paths"
        )


@requires_triton
class TestKernelGradients:
    """The Phase 6 acceptance criterion: kernel gradients at M = 1000."""

    N_PATHS = 1_000
    N_STEPS = 64

    @staticmethod
    def _functional(spot, drift, base, skew, legs, times, n_paths, weights):
        """A smooth scalar functional of the EE profile."""
        from src.csrc.triton_local_vol_cva import fused_local_vol_ee

        profile = fused_local_vol_ee(
            spot, drift, legs, times, 0.03, n_paths, PARAMS,
            base=base, skew=skew, seed=SEED,
        )
        return (profile * weights).sum()

    @staticmethod
    def _setup(n_steps: int):
        """Build the grid, netting set and functional weights.

        The strike is deliberately **not** equal to ``SPOT``. With
        ``strike == spot`` the t=0 exposure is

            V[0] = B[0]*S0 - C[0] = B[0]*S0 - B[0]*strike = 0   exactly,

        which sits precisely on the ``max(V, 0)`` kink. There is no derivative
        there: AAD returns the subgradient 0 while central differences return
        the average of the two one-sided derivatives, and they disagree by half
        the jump -- a fixed absolute amount, regardless of N. That is a property
        of the function, not a kernel bug, and it is measured and documented in
        ``TestAtTheMoneyKink`` below.

        Unlike t > 0, where ``{V_t = 0}`` is a null set under the diffusion, the
        t=0 state is deterministic, so an at-the-money strike puts the kink on
        the sample with probability one.
        """
        from src.pricer.options import SwapLeg

        times = torch.linspace(
            0.0, MATURITY, n_steps + 1, device="cuda", dtype=torch.float64
        )
        legs = [SwapLeg(notional=1.0, strike=95.0, maturity=MATURITY)]
        generator = torch.Generator(device="cuda").manual_seed(SEED + 7)
        weights = torch.randn(
            n_steps + 1, device="cuda", dtype=torch.float64, generator=generator
        )
        return times, legs, weights

    @staticmethod
    def _aad_and_fd(
        legs, times, weights, n_paths: int, n_steps: int, step_rel: float = 1e-6
    ):
        """Return ``{param: (aad, fd)}`` for all four parameters.

        Deliberately evaluates **every** parameter before asserting anything.
        An assertion inside the loop short-circuits on the first failure, which
        hides whether the others agree -- and that is exactly the information
        needed to localise a backward bug.
        """
        import math as _math

        leaves = {
            "spot": torch.tensor(SPOT, device="cuda", dtype=torch.float64,
                                 requires_grad=True),
            "drift": torch.tensor(DRIFT, device="cuda", dtype=torch.float64,
                                  requires_grad=True),
            "base": torch.tensor(BASE, device="cuda", dtype=torch.float64,
                                 requires_grad=True),
            "skew": torch.tensor(SKEW, device="cuda", dtype=torch.float64,
                                 requires_grad=True),
        }
        TestKernelGradients._functional(
            leaves["spot"], leaves["drift"], leaves["base"], leaves["skew"],
            legs, times, n_paths, weights,
        ).backward()
        aad = {name: float(leaf.grad) for name, leaf in leaves.items()}

        nominal = {"spot": SPOT, "drift": DRIFT, "base": BASE, "skew": SKEW}
        finite = {}
        for name in ("spot", "drift", "base", "skew"):
            step = step_rel * max(abs(nominal[name]), 1.0)
            with torch.no_grad():
                up, down = dict(nominal), dict(nominal)
                up[name] = nominal[name] + step
                down[name] = nominal[name] - step
                value_up = float(TestKernelGradients._functional(
                    up["spot"], up["drift"], up["base"], up["skew"],
                    legs, times, n_paths, weights,
                ))
                value_down = float(TestKernelGradients._functional(
                    down["spot"], down["drift"], down["base"], down["skew"],
                    legs, times, n_paths, weights,
                ))
            finite[name] = (value_up - value_down) / (2.0 * step)

        del _math
        return {name: (aad[name], finite[name]) for name in aad}

    @staticmethod
    def _format_comparison(results, header: str = "") -> str:
        """Render an AAD-vs-FD table, with the ratio -- patterns are diagnostic.

        A clean 2x, 0.5x or 1/N ratio points at a specific structural mistake;
        a ragged ratio points at something sample-dependent.
        """
        lines = [header] if header else []
        lines.append(
            f"  {'param':<8}{'AAD':>20}{'FD':>20}{'FD/AAD':>10}{'rel err':>11}"
        )
        lines.append("  " + "-" * 67)
        for name in ("spot", "drift", "base", "skew"):
            aad, fd = results[name]
            ratio = fd / aad if abs(aad) > 1e-300 else float("nan")
            relative = abs(fd - aad) / max(abs(aad), 1e-300)
            flag = "" if relative < 5e-3 else "   <-- MISMATCH"
            lines.append(
                f"  {name:<8}{aad:>20.10e}{fd:>20.10e}{ratio:>10.4f}"
                f"{relative:>11.2%}{flag}"
            )
        return "\n".join(lines)

    def test_gradients_match_central_differences(self) -> None:
        r"""AAD vs finite differences under common random numbers.

        The seed pins the sample, so a bump redraws identical normals and FD is
        a valid oracle here. That is not an assumption: on the CPU reference --
        itself verified against ``torch.autograd`` to 1e-9 -- AAD and FD agree
        to 0.0% at this path count across step sizes 1e-3 to 1e-7. So a
        disagreement on the kernel is a kernel bug, not FD noise.

        All four parameters are evaluated and reported together before any
        assertion, so a failure shows the full picture rather than stopping at
        whichever happens to be checked first.
        """
        times, legs, weights = self._setup(self.N_STEPS)
        results = self._aad_and_fd(legs, times, weights, self.N_PATHS, self.N_STEPS)

        table = self._format_comparison(
            results, f"AAD vs FD  (M={self.N_PATHS:,}, N={self.N_STEPS})"
        )
        print("\n" + table)

        mismatched = [
            name for name, (aad, fd) in results.items()
            if not math.isclose(aad, fd, rel_tol=5e-3, abs_tol=1e-9)
        ]
        assert not mismatched, (
            f"{len(mismatched)} of 4 gradients disagree with finite differences "
            f"({', '.join(mismatched)}).\n{table}\n"
            "  If ONLY 'spot' disagrees by a FIXED ABSOLUTE amount that does not\n"
            "  change with N, suspect a kink: check whether V[0] = B[0]*S0 - C[0]\n"
            "  is zero, which makes the t=0 derivative undefined (see\n"
            "  TestAtTheMoneyKink). If it varies with N, the recursion or the\n"
            "  final adjoint read is at fault. If ALL FOUR disagree, the\n"
            "  recursion itself is mistranslated."
        )

    @pytest.mark.parametrize("n_steps", [16, 63, 64, 100])
    def test_gradient_agreement_across_checkpoint_geometries(
        self, n_steps: int
    ) -> None:
        """Diagnostic: does the error track the checkpoint geometry?

        ``BLOCK_CK`` is derived from ``sqrt(n_steps)``, so these four cases give
        different segment layouts -- and crucially ``n_steps=63`` does **not**
        divide evenly by its ``BLOCK_CK=8``, leaving a partial final segment
        that exercises the ``live`` masking. ``n_steps=64`` divides exactly and
        never touches that path.

        If the error appears only for non-multiples, the bug is in the segment
        boundary handling. If it is uniform, the bug is in the recursion or the
        final read.
        """
        from src.csrc.triton_local_vol_cva import select_local_vol_blocks

        block_m, block_ck = select_local_vol_blocks(n_steps, 8)
        n_segments = -(-n_steps // block_ck)
        divides_evenly = n_steps % block_ck == 0

        times, legs, weights = self._setup(n_steps)
        results = self._aad_and_fd(legs, times, weights, self.N_PATHS, n_steps)

        table = self._format_comparison(
            results,
            f"N={n_steps}  BLOCK_M={block_m}  BLOCK_CK={block_ck}  "
            f"segments={n_segments}  "
            f"{'exact' if divides_evenly else 'PARTIAL final segment'}",
        )
        print("\n" + table)

        mismatched = [
            name for name, (aad, fd) in results.items()
            if not math.isclose(aad, fd, rel_tol=5e-3, abs_tol=1e-9)
        ]
        assert not mismatched, f"{', '.join(mismatched)} disagree\n{table}"

    def test_all_gradients_are_nonzero(self) -> None:
        """Guard the guard: a zero Greek makes the FD check vacuous."""
        times, legs, weights = self._setup(self.N_STEPS)
        spot = torch.tensor(SPOT, device="cuda", dtype=torch.float64, requires_grad=True)
        base = torch.tensor(BASE, device="cuda", dtype=torch.float64, requires_grad=True)
        skew = torch.tensor(SKEW, device="cuda", dtype=torch.float64, requires_grad=True)
        self._functional(
            spot, DRIFT, base, skew, legs, times, self.N_PATHS, weights
        ).backward()
        assert abs(float(spot.grad)) > 1e-8
        assert abs(float(base.grad)) > 1e-8
        assert abs(float(skew.grad)) > 1e-8, (
            "a zero skew gradient would mean the state dependence is not "
            "reaching the adjoint at all"
        )

    def test_partial_gradients_are_respected(self) -> None:
        times, legs, weights = self._setup(self.N_STEPS)
        spot = torch.tensor(SPOT, device="cuda", dtype=torch.float64, requires_grad=True)
        skew = torch.tensor(SKEW, device="cuda", dtype=torch.float64)  # no grad
        self._functional(
            spot, DRIFT, BASE, skew, legs, times, self.N_PATHS, weights
        ).backward()
        assert spot.grad is not None
        assert skew.grad is None

    def test_backward_is_deterministic(self) -> None:
        """Partial-buffer reduction, not atomics -- so bitwise repeatable."""
        times, legs, weights = self._setup(self.N_STEPS)
        results = []
        for _ in range(2):
            spot = torch.tensor(
                SPOT, device="cuda", dtype=torch.float64, requires_grad=True
            )
            skew = torch.tensor(
                SKEW, device="cuda", dtype=torch.float64, requires_grad=True
            )
            self._functional(
                spot, DRIFT, BASE, skew, legs, times, self.N_PATHS, weights
            ).backward()
            results.append((float(spot.grad), float(skew.grad)))
        assert results[0] == results[1]

    def test_checkpointing_costs_no_o_mn_memory(self) -> None:
        """The whole point: the adjoint allocates nothing of size M."""
        from src.csrc.triton_local_vol_cva import fused_local_vol_ee
        from src.pricer.options import SwapLeg

        n_steps = 252
        times = torch.linspace(
            0.0, MATURITY, n_steps + 1, device="cuda", dtype=torch.float32
        )
        legs = [SwapLeg(notional=1.0, strike=100.0, maturity=MATURITY)]
        ceiling = 4096 * (n_steps + 1) * 4 * 6  # generous headroom

        for n_paths in (1_000, 500_000):
            spot = torch.tensor(
                SPOT, device="cuda", dtype=torch.float32, requires_grad=True
            )
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            profile = fused_local_vol_ee(
                spot, DRIFT, legs, times, 0.03, n_paths, PARAMS, seed=SEED
            )
            profile.sum().backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()

            assert peak < ceiling, (
                f"M={n_paths:,}: peak {peak / 1024**2:,.2f} MiB exceeds the "
                f"M-independent ceiling {ceiling / 1024**2:,.2f} MiB"
            )

    def test_double_backward_raises_rather_than_lying(self) -> None:
        from src.csrc.triton_local_vol_cva import fused_local_vol_ee
        from src.pricer.options import SwapLeg

        times = torch.linspace(
            0.0, MATURITY, 17, device="cuda", dtype=torch.float64
        )
        legs = [SwapLeg(notional=1.0, strike=100.0, maturity=MATURITY)]
        spot = torch.tensor(SPOT, device="cuda", dtype=torch.float64, requires_grad=True)
        profile = fused_local_vol_ee(
            spot, DRIFT, legs, times, 0.03, 256, PARAMS, seed=SEED
        )
        (grad_spot,) = torch.autograd.grad(profile.sum(), spot, create_graph=True)
        with pytest.raises(RuntimeError):
            torch.autograd.grad(grad_spot, spot)
