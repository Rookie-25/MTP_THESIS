r"""Phase 3 correctness suite: fused Triton GBM vs the pure-PyTorch simulator.

Test strategy
-------------
A custom kernel with a hand-written adjoint has two independent ways to be
wrong -- the kernel can compute the wrong thing, or the derived gradient
formulas can be wrong. The suite separates them so a failure localises
immediately:

**Tier 1 -- runs anywhere, including CPU-only machines.**
:func:`~src.csrc.triton_gbm.reference_gbm_backward` is a pure-PyTorch
transcription of exactly the analytic formulas the Triton adjoint implements.
Comparing it against ``torch.autograd`` on the reference simulator validates the
*mathematics* with no GPU in sight. If this tier fails, the derivation is wrong
and no amount of kernel debugging will help.

**Tier 2 -- requires Triton and CUDA, skipped otherwise.**
Forward parity against ``simulate_gbm``, backward parity against both autograd
and the Tier 1 reference, and ``gradcheck`` in float64. If Tier 1 passes and
Tier 2 fails, the derivation is right and the kernel plumbing is wrong
(strides, masking, chunk carry, block sizes).

This split is deliberate: Phase 3 is developed on Colab but the CPU test suite
must stay meaningful on the local machine, where Triton has no wheel.

Why exact forward equality is *not* asserted
--------------------------------------------
The kernel and PyTorch both compute a cumulative sum along time, but they
associate the additions differently -- PyTorch runs one contiguous scan, the
kernel runs a chunked scan with a carry. Floating-point addition is not
associative, so the two agree only to within accumulated round-off. The tests
therefore assert agreement to a tolerance appropriate to the dtype, and use
float64 for the tight comparisons. Demanding bitwise equality would be
demanding that two different (both correct) summation orders coincide.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.csrc.triton_gbm import (
    HAS_TRITON,
    FusedGBMFunction,
    is_available,
    reference_gbm_backward,
    select_block_sizes,
    triton_simulate_gbm,
)
from src.models.gbm import draw_brownian_increments, simulate_gbm

# Shared market setup.
S0 = 100.0
MU = 0.03
SIGMA = 0.20
MATURITY = 1.0
N_STEPS = 252
DT = MATURITY / N_STEPS
SEED = 20260818

# Skip marker for everything that needs a real GPU + Triton.
requires_triton = pytest.mark.skipif(
    not is_available(),
    reason=(
        "fused Triton kernels require Triton and a CUDA device "
        f"(triton installed: {HAS_TRITON}, cuda: {torch.cuda.is_available()})"
    ),
)


def _make_leaves(
    device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fresh ``(s0, mu, sigma)`` autograd leaves."""
    return (
        torch.tensor(S0, device=device, dtype=dtype, requires_grad=True),
        torch.tensor(MU, device=device, dtype=dtype, requires_grad=True),
        torch.tensor(SIGMA, device=device, dtype=dtype, requires_grad=True),
    )


# ==========================================================================
# Tier 1 -- the adjoint mathematics, verified on CPU
# ==========================================================================
class TestAnalyticAdjointOnCPU:
    """Validate the derived gradient formulas against autograd, no GPU needed.

    These tests are the reason a CPU-only machine can still make progress on
    Phase 3: they pin down the derivation that the Triton kernel implements.
    """

    @staticmethod
    def _autograd_grads(
        grad_out: torch.Tensor, dW: torch.Tensor
    ) -> tuple[float, float, float, torch.Tensor]:
        """Differentiate the reference simulator with autograd."""
        s0, mu, sigma = _make_leaves(dW.device, dW.dtype)
        dw_leaf = dW.detach().clone().requires_grad_(True)
        paths = simulate_gbm(s0, mu, sigma, dw_leaf, DT)
        paths.backward(grad_out)
        return float(s0.grad), float(mu.grad), float(sigma.grad), dw_leaf.grad

    @staticmethod
    def _reference_grads(
        grad_out: torch.Tensor, dW: torch.Tensor
    ) -> tuple[float, float, float, torch.Tensor]:
        """Evaluate the closed-form adjoint the kernel implements."""
        s0 = torch.tensor(S0, dtype=dW.dtype)
        sigma = torch.tensor(SIGMA, dtype=dW.dtype)
        paths = simulate_gbm(S0, MU, SIGMA, dW, DT).detach()
        grad_s0, grad_mu, grad_sigma, grad_dw = reference_gbm_backward(
            grad_out, paths, dW, s0, sigma, DT
        )
        return float(grad_s0), float(grad_mu), float(grad_sigma), grad_dw

    @pytest.mark.parametrize("n_paths,n_steps", [(1, 1), (1, 8), (7, 3), (64, 252)])
    def test_reference_adjoint_matches_autograd(self, n_paths: int, n_steps: int) -> None:
        """The derived formulas must reproduce autograd on assorted shapes."""
        dt = MATURITY / n_steps
        dW = draw_brownian_increments(
            n_paths, n_steps, dt, dtype=torch.float64, seed=SEED
        )
        generator = torch.Generator().manual_seed(SEED + 1)
        grad_out = torch.randn(
            (n_paths, n_steps + 1), dtype=torch.float64, generator=generator
        )

        # Autograd on the reference simulator.
        s0, mu, sigma = _make_leaves(dW.device, dW.dtype)
        dw_leaf = dW.detach().clone().requires_grad_(True)
        paths = simulate_gbm(s0, mu, sigma, dw_leaf, dt)
        paths.backward(grad_out)

        # The closed-form adjoint.
        grad_s0, grad_mu, grad_sigma, grad_dw = reference_gbm_backward(
            grad_out,
            paths.detach(),
            dW,
            torch.tensor(S0, dtype=torch.float64),
            torch.tensor(SIGMA, dtype=torch.float64),
            dt,
        )

        assert math.isclose(float(grad_s0), float(s0.grad), rel_tol=1e-11)
        assert math.isclose(float(grad_mu), float(mu.grad), rel_tol=1e-11)
        assert math.isclose(float(grad_sigma), float(sigma.grad), rel_tol=1e-11)
        assert torch.allclose(grad_dw, dw_leaf.grad, rtol=1e-11, atol=1e-13)

    def test_vega_ito_correction_is_present(self) -> None:
        r"""Dropping the :math:`-\sigma\Delta t` term must visibly break Vega.

        This is the single most plausible transcription error in the adjoint:
        :math:`\partial\iota/\partial\sigma = \Delta W - \sigma\Delta t`, and the
        second term (from the Ito correction inside the drift) is easy to omit.
        The naive version is *close enough to look right*, so this test asserts
        the two are measurably different -- guarding against a future
        "simplification" that silently deletes it.
        """
        dW = draw_brownian_increments(256, 64, DT, dtype=torch.float64, seed=SEED)
        paths = simulate_gbm(S0, MU, SIGMA, dW, DT)
        grad_out = torch.ones_like(paths)

        _, _, correct_vega, _ = reference_gbm_backward(
            grad_out,
            paths.detach(),
            dW,
            torch.tensor(S0, dtype=torch.float64),
            torch.tensor(SIGMA, dtype=torch.float64),
            DT,
        )

        # Reconstruct the naive (wrong) version: dW only, no Ito term.
        p = grad_out[:, 1:] * paths.detach()[:, 1:]
        q = torch.flip(torch.cumsum(torch.flip(p, dims=(1,)), dim=1), dims=(1,))
        naive_vega = (q * dW).sum()

        assert not math.isclose(float(correct_vega), float(naive_vega), rel_tol=1e-6)
        # And the discrepancy must equal exactly the omitted term.
        assert math.isclose(
            float(naive_vega - correct_vega),
            float((q * SIGMA * DT).sum()),
            rel_tol=1e-10,
        )

    def test_reference_adjoint_is_double_differentiable(self) -> None:
        """Unlike the kernel, the PyTorch reference supports second order.

        This is what makes it the fallback for Gamma / Hessian-vector work.
        """
        dW = draw_brownian_increments(32, 16, DT, dtype=torch.float64, seed=SEED)
        sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)
        paths = simulate_gbm(S0, MU, sigma, dW, DT)
        grad_out = torch.ones_like(paths)

        _, _, grad_sigma, _ = reference_gbm_backward(
            grad_out,
            paths,
            dW,
            torch.tensor(S0, dtype=torch.float64),
            sigma,
            DT,
        )
        second = torch.autograd.grad(grad_sigma, sigma)[0]
        assert torch.isfinite(second)

    def test_reference_rejects_inconsistent_shapes(self) -> None:
        dW = draw_brownian_increments(8, 4, DT, dtype=torch.float64, seed=SEED)
        paths = simulate_gbm(S0, MU, SIGMA, dW, DT)
        s0 = torch.tensor(S0, dtype=torch.float64)
        sigma = torch.tensor(SIGMA, dtype=torch.float64)

        with pytest.raises(ValueError, match="must match paths"):
            reference_gbm_backward(torch.ones(8, 3), paths, dW, s0, sigma, DT)
        with pytest.raises(ValueError, match="inconsistent"):
            reference_gbm_backward(
                torch.ones_like(paths), paths, torch.ones(8, 9), s0, sigma, DT
            )


class TestBlockSizeSelection:
    """Launch-configuration heuristics, testable without a GPU."""

    @pytest.mark.parametrize("n_steps", [1, 2, 17, 64, 252, 1_000, 10_000])
    @pytest.mark.parametrize("element_size", [4, 8])
    def test_blocks_are_powers_of_two(self, n_steps: int, element_size: int) -> None:
        block_m, block_n = select_block_sizes(n_steps, element_size)
        assert block_m >= 1 and block_n >= 1
        assert block_m & (block_m - 1) == 0, f"BLOCK_M={block_m} not a power of two"
        assert block_n & (block_n - 1) == 0, f"BLOCK_N={block_n} not a power of two"

    @pytest.mark.parametrize("element_size", [4, 8])
    def test_tile_stays_within_sram_budget(self, element_size: int) -> None:
        """A tile must not blow past the SRAM budget the heuristic targets."""
        for n_steps in (16, 64, 252, 4_096):
            block_m, block_n = select_block_sizes(n_steps, element_size)
            assert block_m * block_n * element_size <= 32 * 1024

    def test_float64_uses_smaller_tiles_than_float32(self) -> None:
        """Doubling element size must not double SRAM usage."""
        m32, n32 = select_block_sizes(252, 4)
        m64, n64 = select_block_sizes(252, 8)
        assert m64 * n64 * 8 <= m32 * n32 * 4

    def test_rejects_invalid_arguments(self) -> None:
        with pytest.raises(ValueError, match="n_steps must be positive"):
            select_block_sizes(0, 4)
        with pytest.raises(ValueError, match="element_size must be positive"):
            select_block_sizes(252, 0)


class TestGracefulDegradation:
    """Without Triton/CUDA the module must import and fail with a clear message."""

    def test_module_imports_regardless(self) -> None:
        """Importing must never raise, so the CPU suite keeps collecting."""
        from src.csrc import triton_gbm

        assert isinstance(triton_gbm.HAS_TRITON, bool)
        assert isinstance(is_available(), bool)

    @pytest.mark.skipif(is_available(), reason="a working Triton+CUDA runtime is present")
    def test_helper_raises_actionable_error_when_unavailable(self) -> None:
        dW = torch.randn(4, 8, dtype=torch.float32)
        with pytest.raises(RuntimeError, match="Triton is not installed|No CUDA device"):
            triton_simulate_gbm(S0, MU, SIGMA, dW, DT)


# ==========================================================================
# Tier 2 -- the kernels themselves (GPU only)
# ==========================================================================
@requires_triton
class TestFusedForward:
    """Forward parity against the pure-PyTorch simulator."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("n_paths,n_steps", [(1, 1), (3, 5), (1_024, 252), (5_000, 100)])
    def test_matches_reference_simulator(
        self, dtype: torch.dtype, n_paths: int, n_steps: int
    ) -> None:
        dt = MATURITY / n_steps
        dW = draw_brownian_increments(
            n_paths, n_steps, dt, device="cuda", dtype=dtype, seed=SEED
        )
        expected = simulate_gbm(S0, MU, SIGMA, dW, dt)
        actual = triton_simulate_gbm(S0, MU, SIGMA, dW, dt)

        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        # Chunked vs contiguous summation orders differ; compare to dtype-
        # appropriate tolerance rather than demanding bitwise equality.
        rtol = 1e-11 if dtype == torch.float64 else 2e-5
        assert torch.allclose(actual, expected, rtol=rtol, atol=0.0)

    def test_first_column_is_exactly_spot(self) -> None:
        dW = draw_brownian_increments(
            512, N_STEPS, DT, device="cuda", dtype=torch.float64, seed=SEED
        )
        paths = triton_simulate_gbm(S0, MU, SIGMA, dW, DT)
        assert torch.all(paths[:, 0] == S0)

    def test_paths_stay_strictly_positive(self) -> None:
        dW = draw_brownian_increments(
            2_048, N_STEPS, DT, device="cuda", dtype=torch.float64, seed=SEED
        )
        paths = triton_simulate_gbm(S0, MU, SIGMA, dW, DT)
        assert torch.all(paths > 0.0)

    def test_is_deterministic_across_repeated_calls(self) -> None:
        """Same inputs must give bitwise-identical output on every launch."""
        dW = draw_brownian_increments(
            4_096, N_STEPS, DT, device="cuda", dtype=torch.float64, seed=SEED
        )
        first = triton_simulate_gbm(S0, MU, SIGMA, dW, DT)
        second = triton_simulate_gbm(S0, MU, SIGMA, dW, DT)
        assert torch.equal(first, second)

    def test_handles_non_contiguous_increments(self) -> None:
        """Strides are passed explicitly, so a sliced view must still work."""
        wide = draw_brownian_increments(
            256, 2 * N_STEPS, DT, device="cuda", dtype=torch.float64, seed=SEED
        )
        view = wide[:, :N_STEPS]
        assert not view.is_contiguous()
        assert torch.allclose(
            triton_simulate_gbm(S0, MU, SIGMA, view, DT),
            simulate_gbm(S0, MU, SIGMA, view, DT),
            rtol=1e-11,
        )

    def test_rejects_cpu_input(self) -> None:
        dW = torch.randn(4, 8, dtype=torch.float32)
        with pytest.raises(ValueError, match="GPU-only"):
            triton_simulate_gbm(S0, MU, SIGMA, dW, DT)

    @pytest.mark.parametrize("bad_dt", [0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_invalid_dt(self, bad_dt: float) -> None:
        dW = torch.randn(4, 8, device="cuda", dtype=torch.float32)
        with pytest.raises(ValueError, match="dt must be positive"):
            triton_simulate_gbm(S0, MU, SIGMA, dW, bad_dt)

    def test_rejects_wrong_dimensionality(self) -> None:
        with pytest.raises(ValueError, match="2-dimensional"):
            triton_simulate_gbm(
                S0, MU, SIGMA, torch.randn(4, 8, 2, device="cuda"), DT
            )


@requires_triton
class TestFusedBackward:
    """The hand-written adjoint must reproduce PyTorch's autograd exactly."""

    @pytest.mark.parametrize("n_paths,n_steps", [(1, 1), (3, 7), (1_024, 252)])
    def test_greeks_match_autograd(self, n_paths: int, n_steps: int) -> None:
        """Delta, Rho and Vega from the kernel vs from native autograd."""
        dt = MATURITY / n_steps
        dW = draw_brownian_increments(
            n_paths, n_steps, dt, device="cuda", dtype=torch.float64, seed=SEED
        )
        generator = torch.Generator(device="cuda").manual_seed(SEED + 2)
        grad_out = torch.randn(
            (n_paths, n_steps + 1), device="cuda", dtype=torch.float64,
            generator=generator,
        )

        # Reference: pure PyTorch autograd.
        ref_s0, ref_mu, ref_sigma = _make_leaves(dW.device, dW.dtype)
        ref_dw = dW.detach().clone().requires_grad_(True)
        simulate_gbm(ref_s0, ref_mu, ref_sigma, ref_dw, dt).backward(grad_out)

        # Candidate: the fused kernel.
        fused_s0, fused_mu, fused_sigma = _make_leaves(dW.device, dW.dtype)
        fused_dw = dW.detach().clone().requires_grad_(True)
        triton_simulate_gbm(fused_s0, fused_mu, fused_sigma, fused_dw, dt).backward(
            grad_out
        )

        assert math.isclose(float(fused_s0.grad), float(ref_s0.grad), rel_tol=1e-9), (
            f"Delta mismatch: fused {float(fused_s0.grad)!r} vs "
            f"autograd {float(ref_s0.grad)!r}"
        )
        assert math.isclose(float(fused_mu.grad), float(ref_mu.grad), rel_tol=1e-9), (
            f"Rho mismatch: fused {float(fused_mu.grad)!r} vs "
            f"autograd {float(ref_mu.grad)!r}"
        )
        assert math.isclose(
            float(fused_sigma.grad), float(ref_sigma.grad), rel_tol=1e-9
        ), (
            f"Vega mismatch: fused {float(fused_sigma.grad)!r} vs "
            f"autograd {float(ref_sigma.grad)!r}"
        )
        assert torch.allclose(fused_dw.grad, ref_dw.grad, rtol=1e-9, atol=1e-12)

    def test_matches_the_cpu_verified_reference_adjoint(self) -> None:
        """Close the loop: kernel == Tier 1 reference == autograd.

        Tier 1 already proved the reference adjoint equals autograd on CPU. If
        the kernel also equals the reference, the chain is complete and any
        residual difference is pure kernel plumbing.
        """
        dW = draw_brownian_increments(
            512, N_STEPS, DT, device="cuda", dtype=torch.float64, seed=SEED
        )
        paths = simulate_gbm(S0, MU, SIGMA, dW, DT).detach()
        generator = torch.Generator(device="cuda").manual_seed(SEED + 3)
        grad_out = torch.randn_like(paths)
        del generator

        expected = reference_gbm_backward(
            grad_out,
            paths,
            dW,
            torch.tensor(S0, device="cuda", dtype=torch.float64),
            torch.tensor(SIGMA, device="cuda", dtype=torch.float64),
            DT,
        )

        s0, mu, sigma = _make_leaves(dW.device, dW.dtype)
        dw_leaf = dW.detach().clone().requires_grad_(True)
        triton_simulate_gbm(s0, mu, sigma, dw_leaf, DT).backward(grad_out)

        assert math.isclose(float(s0.grad), float(expected[0]), rel_tol=1e-9)
        assert math.isclose(float(mu.grad), float(expected[1]), rel_tol=1e-9)
        assert math.isclose(float(sigma.grad), float(expected[2]), rel_tol=1e-9)
        assert torch.allclose(dw_leaf.grad, expected[3], rtol=1e-9, atol=1e-12)

    def test_gradcheck_float64(self) -> None:
        """Numerical gradcheck on a small problem, the strictest single check."""
        n_paths, n_steps = 6, 12
        dt = MATURITY / n_steps
        dW = draw_brownian_increments(
            n_paths, n_steps, dt, device="cuda", dtype=torch.float64, seed=SEED
        )
        s0 = torch.tensor(S0, device="cuda", dtype=torch.float64, requires_grad=True)
        mu = torch.tensor(MU, device="cuda", dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(
            SIGMA, device="cuda", dtype=torch.float64, requires_grad=True
        )
        dw_leaf = dW.detach().clone().requires_grad_(True)

        assert torch.autograd.gradcheck(
            FusedGBMFunction.apply,
            (s0, mu, sigma, dw_leaf, dt),
            eps=1e-6,
            atol=1e-7,
            rtol=1e-5,
        )

    def test_partial_gradients_are_respected(self) -> None:
        """Only leaves that require grad should receive one."""
        dW = draw_brownian_increments(
            128, 32, DT, device="cuda", dtype=torch.float64, seed=SEED
        )
        s0 = torch.tensor(S0, device="cuda", dtype=torch.float64, requires_grad=True)
        mu = torch.tensor(MU, device="cuda", dtype=torch.float64)  # no grad
        sigma = torch.tensor(
            SIGMA, device="cuda", dtype=torch.float64, requires_grad=True
        )

        paths = triton_simulate_gbm(s0, mu, sigma, dW, DT)
        paths.sum().backward()

        assert s0.grad is not None
        assert sigma.grad is not None
        assert mu.grad is None

    def test_double_backward_raises_rather_than_lying(self) -> None:
        """Second order is unsupported; it must error, not return garbage."""
        dW = draw_brownian_increments(
            32, 8, DT, device="cuda", dtype=torch.float64, seed=SEED
        )
        s0 = torch.tensor(S0, device="cuda", dtype=torch.float64, requires_grad=True)
        paths = triton_simulate_gbm(s0, MU, SIGMA, dW, DT)
        (grad_s0,) = torch.autograd.grad(paths.sum(), s0, create_graph=True)

        with pytest.raises(RuntimeError):
            torch.autograd.grad(grad_s0, s0)

    def test_backward_is_deterministic(self) -> None:
        """Partial-buffer reduction (not atomics) must give bitwise repeatability."""
        dW = draw_brownian_increments(
            4_096, N_STEPS, DT, device="cuda", dtype=torch.float64, seed=SEED
        )
        results = []
        for _ in range(2):
            s0, mu, sigma = _make_leaves(dW.device, dW.dtype)
            triton_simulate_gbm(s0, mu, sigma, dW, DT).sum().backward()
            results.append((float(s0.grad), float(mu.grad), float(sigma.grad)))
        assert results[0] == results[1]


@requires_triton
class TestPipelineIntegration:
    """The fused kernel must slot into the Phase 1-2 pipeline unchanged."""

    def test_cva_greeks_match_between_backends(self) -> None:
        """End-to-end: CVA sensitivities via PyTorch paths vs via fused paths.

        This is the test that actually matters for the thesis -- it proves the
        kernel swap is invisible to everything downstream (payoff, exposure
        profile, credit integral).
        """
        from src.pricer.options import SwapLeg, portfolio_swap_mtm
        from src.xva.cva import compute_unilateral_cva
        from src.xva.exposure import expected_exposure

        n_paths, n_steps = 4_096, 64
        dt = MATURITY / n_steps
        dW = draw_brownian_increments(
            n_paths, n_steps, dt, device="cuda", dtype=torch.float64, seed=SEED
        )
        times = torch.linspace(
            0.0, MATURITY, n_steps + 1, device="cuda", dtype=torch.float64
        )
        legs = [SwapLeg(notional=1.0, strike=S0, maturity=MATURITY)]

        def cva_for(simulator_fn) -> tuple[float, float, float]:
            s0, mu, sigma = _make_leaves(dW.device, dW.dtype)
            paths = simulator_fn(s0, mu, sigma, dW, dt)
            mtm = portfolio_swap_mtm(paths, times, legs, MU)
            cva = compute_unilateral_cva(
                expected_exposure(mtm), times, 0.02, 0.4, discount_rate=MU
            )
            cva.backward()
            return float(cva), float(s0.grad), float(sigma.grad)

        reference = cva_for(simulate_gbm)
        fused = cva_for(triton_simulate_gbm)

        assert math.isclose(fused[0], reference[0], rel_tol=1e-9), "CVA differs"
        assert math.isclose(fused[1], reference[1], rel_tol=1e-8), "dCVA/dS0 differs"
        assert math.isclose(fused[2], reference[2], rel_tol=1e-8), "dCVA/dsigma differs"
