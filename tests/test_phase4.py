r"""Phase 4 correctness suite: in-kernel Philox RNG with rematerialised AAD.

Why this suite is shaped differently from Phase 3
=================================================
Phase 3 could validate its kernel against ``simulate_gbm`` element by element,
because both consumed the *same* ``dW``. Phase 4 generates its own increments
from Philox, which will never bitwise-match ``torch.randn``. There is no
reference trajectory to diff against, so correctness has to be established
three other ways:

**1. Distributional correctness (statistical moments).** If the kernel's
increments really are i.i.d. :math:`\mathcal{N}(0,\Delta t)`, then the terminal
law is log-normal with known moments:

.. math::
    \mathbb{E}[S_T] = S_0 e^{\mu T}, \qquad
    \operatorname{Var}[S_T] = S_0^2 e^{2\mu T}\!\left(e^{\sigma^2 T} - 1\right),

.. math::
    \mathbb{E}\!\left[\log\tfrac{S_T}{S_0}\right]
        = \left(\mu - \tfrac{\sigma^2}{2}\right)T, \qquad
    \operatorname{Var}\!\left[\log\tfrac{S_T}{S_0}\right] = \sigma^2 T .

Every moment test below derives its tolerance from the *sample* standard error
rather than a hardcoded epsilon, so the tests scale correctly with path count
instead of being tuned to one configuration.

**2. Gradient correctness (finite differences with common random numbers).**
Because the seed pins the entire random sample, bumping a parameter re-draws the
*identical* increments. That makes bump-and-revalue a legitimate oracle -- and
critically, it is the only test that can catch a **rematerialisation bug**: if
the backward regenerated different randoms than the forward, Vega would be
wrong while Delta and Rho stayed right, because only Vega touches :math:`Z`.
That asymmetry is the diagnostic signature, and
:class:`TestRematerialisation` leans on it directly.

Note the float32 normals do **not** weaken this test. :math:`Z` is held fixed
across the bump, so its precision cancels out of the difference quotient
entirely; the accumulation is float64 and that is what governs FD accuracy.

**3. Independence of the parallel RNG streams.** The offset scheme puts path
identity in the Philox *key* (``seed + program_id``). If those streams were
correlated -- or worse, if offsets aliased -- paths would repeat and the
estimator would be silently biased with no error raised. Tested explicitly.

Tiering
-------
As in Phase 3, the adjoint *mathematics* is verified on CPU via
:func:`~src.csrc.triton_philox_gbm.reference_philox_backward`, so a machine with
no GPU still validates the derivation. Kernel-level tests skip cleanly.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.csrc.triton_philox_gbm import (
    MAX_PHILOX_OFFSET,
    is_available,
    philox_simulate_gbm,
    reference_philox_backward,
    reference_philox_forward,
    validate_offset_scheme,
)

S0 = 100.0
MU = 0.03
SIGMA = 0.20
MATURITY = 1.0
N_STEPS = 252
DT = MATURITY / N_STEPS
SEED = 20260819

# Moment tests need enough paths that the sample standard error is small
# relative to the quantity, but not so many that the suite crawls.
MOMENT_PATHS = 500_000

# Number of sample standard errors allowed. 5 gives a ~6e-7 false-failure rate
# per assertion under normality -- tight enough to catch a real bias, loose
# enough that the suite is not flaky.
SIGMA_TOLERANCE = 5.0

requires_triton = pytest.mark.skipif(
    not is_available(),
    reason="in-kernel Philox kernels require Triton and a CUDA device",
)


# ==========================================================================
# Tier 1 -- adjoint mathematics, verified on CPU
# ==========================================================================
class TestPhiloxAdjointOnCPU:
    """Validate the Phase 4 adjoint formulas against autograd without a GPU.

    Phase 4 reparameterises the increment as
    :math:`a + \\sigma\\sqrt{\\Delta t}\\,Z` rather than
    :math:`a + \\sigma\\,dW`, which changes the Vega term. That makes this an
    independent derivation from Phase 3's, deserving its own verification.
    """

    @pytest.mark.parametrize("n_paths,n_steps", [(1, 1), (1, 8), (5, 3), (64, 252)])
    def test_reference_adjoint_matches_autograd(self, n_paths: int, n_steps: int) -> None:
        dt = MATURITY / n_steps
        generator = torch.Generator().manual_seed(SEED)
        z = torch.randn((n_paths, n_steps), dtype=torch.float64, generator=generator)
        grad_out = torch.randn(
            (n_paths, n_steps + 1), dtype=torch.float64, generator=generator
        )

        s0 = torch.tensor(S0, dtype=torch.float64, requires_grad=True)
        mu = torch.tensor(MU, dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)

        paths = reference_philox_forward(s0, mu, sigma, z, dt)
        paths.backward(grad_out)

        grad_s0, grad_mu, grad_sigma = reference_philox_backward(
            grad_out,
            paths.detach(),
            z,
            torch.tensor(S0, dtype=torch.float64),
            torch.tensor(SIGMA, dtype=torch.float64),
            dt,
        )

        assert math.isclose(float(grad_s0), float(s0.grad), rel_tol=1e-11)
        assert math.isclose(float(grad_mu), float(mu.grad), rel_tol=1e-11)
        assert math.isclose(float(grad_sigma), float(sigma.grad), rel_tol=1e-11)

    def test_vega_differs_from_the_phase3_parameterisation(self) -> None:
        r"""Guard against copy-pasting Phase 3's Vega.

        Phase 3 used :math:`\partial\iota/\partial\sigma = dW - \sigma\Delta t`;
        Phase 4 uses :math:`\sqrt{\Delta t}Z - \sigma\Delta t`. Since
        :math:`dW = \sqrt{\Delta t}Z` these agree *only* when the caller scaled
        :math:`Z` correctly. This test pins down that the
        :math:`\sqrt{\Delta t}` factor is present, which a naive port would drop.
        """
        n_paths, n_steps = 128, 32
        dt = MATURITY / n_steps
        generator = torch.Generator().manual_seed(SEED)
        z = torch.randn((n_paths, n_steps), dtype=torch.float64, generator=generator)

        paths = reference_philox_forward(
            torch.tensor(S0, dtype=torch.float64),
            torch.tensor(MU, dtype=torch.float64),
            torch.tensor(SIGMA, dtype=torch.float64),
            z,
            dt,
        )
        grad_out = torch.ones_like(paths)

        _, _, correct = reference_philox_backward(
            grad_out,
            paths,
            z,
            torch.tensor(S0, dtype=torch.float64),
            torch.tensor(SIGMA, dtype=torch.float64),
            dt,
        )

        p = grad_out[:, 1:] * paths[:, 1:]
        q = torch.flip(torch.cumsum(torch.flip(p, dims=(1,)), dim=1), dims=(1,))
        # The bug: forgetting sqrt(dt) on the Z term.
        unscaled = float((q * (z - SIGMA * dt)).sum())

        assert not math.isclose(float(correct), unscaled, rel_tol=1e-6)

    def test_reference_is_double_differentiable(self) -> None:
        """The PyTorch reference supports second order; the kernel does not."""
        generator = torch.Generator().manual_seed(SEED)
        z = torch.randn((32, 16), dtype=torch.float64, generator=generator)
        sigma = torch.tensor(SIGMA, dtype=torch.float64, requires_grad=True)

        paths = reference_philox_forward(
            torch.tensor(S0, dtype=torch.float64),
            torch.tensor(MU, dtype=torch.float64),
            sigma,
            z,
            DT,
        )
        _, _, grad_sigma = reference_philox_backward(
            torch.ones_like(paths),
            paths,
            z,
            torch.tensor(S0, dtype=torch.float64),
            sigma,
            DT,
        )
        second = torch.autograd.grad(grad_sigma, sigma)[0]
        assert torch.isfinite(second)

    def test_reference_rejects_inconsistent_shapes(self) -> None:
        z = torch.randn((8, 4), dtype=torch.float64)
        paths = reference_philox_forward(
            torch.tensor(S0, dtype=torch.float64),
            torch.tensor(MU, dtype=torch.float64),
            torch.tensor(SIGMA, dtype=torch.float64),
            z,
            DT,
        )
        s0 = torch.tensor(S0, dtype=torch.float64)
        sigma = torch.tensor(SIGMA, dtype=torch.float64)

        with pytest.raises(ValueError, match="must match paths"):
            reference_philox_backward(torch.ones(8, 3), paths, z, s0, sigma, DT)
        with pytest.raises(ValueError, match="inconsistent"):
            reference_philox_backward(
                torch.ones_like(paths), paths, torch.ones(8, 9), s0, sigma, DT
            )


class TestOffsetSchemeGuard:
    """The int32 aliasing guard -- the highest-consequence check in Phase 4.

    A wrapped Philox offset does not raise; it silently makes different paths
    share increments, correlating the sample and biasing every downstream
    estimator. These tests run anywhere.
    """

    @pytest.mark.parametrize("block_m", [1, 16, 32, 64])
    @pytest.mark.parametrize("n_steps", [1, 252, 1_000, 10_000])
    def test_realistic_configurations_are_safe(self, block_m: int, n_steps: int) -> None:
        validate_offset_scheme(block_m, n_steps)  # must not raise

    def test_rejects_configurations_that_would_alias(self) -> None:
        with pytest.raises(ValueError, match="exceeds the 32-bit limit"):
            validate_offset_scheme(64, MAX_PHILOX_OFFSET)

    def test_the_naive_global_scheme_would_have_overflowed(self) -> None:
        """Document precisely why per-program keys are used, not global offsets.

        The rejected design was ``offset = path_index * n_steps + step``. At the
        path counts this phase targets that exceeds int32, which is the entire
        reason path identity lives in the Philox *key* instead.
        """
        n_steps = 252
        assert 8_000_000 * n_steps < MAX_PHILOX_OFFSET, "8M paths would have fit"
        assert 10_000_000 * n_steps > MAX_PHILOX_OFFSET, "10M paths would alias"
        assert 20_000_000 * n_steps > MAX_PHILOX_OFFSET, "20M paths would alias badly"

        # The scheme actually used stays microscopic regardless of path count.
        for block_m in (16, 32, 64):
            assert block_m * n_steps < 20_000
            validate_offset_scheme(block_m, n_steps)


class TestGracefulDegradation:
    """Import must succeed and failure must be actionable without Triton/CUDA."""

    def test_module_imports_regardless(self) -> None:
        from src.csrc import triton_philox_gbm

        assert isinstance(triton_philox_gbm.MAX_PHILOX_OFFSET, int)
        assert isinstance(is_available(), bool)

    @pytest.mark.skipif(is_available(), reason="a working Triton+CUDA runtime is present")
    def test_helper_raises_actionable_error(self) -> None:
        with pytest.raises(RuntimeError, match="Triton is not installed|No CUDA device"):
            philox_simulate_gbm(S0, MU, SIGMA, 1_024, N_STEPS, DT, seed=SEED)


# ==========================================================================
# Tier 2 -- distributional correctness of the in-kernel RNG (GPU only)
# ==========================================================================
@requires_triton
class TestTerminalDistribution:
    """The generated paths must follow the theoretical log-normal law."""

    @staticmethod
    def _terminal(dtype: torch.dtype = torch.float64) -> torch.Tensor:
        paths = philox_simulate_gbm(
            S0, MU, SIGMA, MOMENT_PATHS, N_STEPS, DT, seed=SEED, dtype=dtype
        )
        return paths[:, -1].detach()

    def test_terminal_mean_matches_theory(self) -> None:
        r""":math:`\mathbb{E}[S_T] = S_0 e^{\mu T}`."""
        terminal = self._terminal()
        expected = S0 * math.exp(MU * MATURITY)
        sample_mean = float(terminal.mean())
        std_error = float(terminal.std()) / math.sqrt(MOMENT_PATHS)

        assert abs(sample_mean - expected) < SIGMA_TOLERANCE * std_error, (
            f"E[S_T] = {sample_mean:.6f}, theory {expected:.6f}, "
            f"se {std_error:.6f} ({abs(sample_mean - expected) / std_error:.2f} sigma)"
        )

    def test_terminal_variance_matches_theory(self) -> None:
        r""":math:`\operatorname{Var}[S_T] = S_0^2 e^{2\mu T}(e^{\sigma^2 T}-1)`."""
        terminal = self._terminal()
        expected = (
            S0**2
            * math.exp(2.0 * MU * MATURITY)
            * (math.exp(SIGMA**2 * MATURITY) - 1.0)
        )
        sample_variance = float(terminal.var(unbiased=True))
        # se(sample variance) ~ variance * sqrt(2 / M) for a near-normal sample;
        # the log-normal has heavier tails, so allow a wider band.
        std_error = expected * math.sqrt(2.0 / MOMENT_PATHS)

        assert abs(sample_variance - expected) < 10.0 * std_error, (
            f"Var[S_T] = {sample_variance:.4f}, theory {expected:.4f}"
        )

    def test_log_return_mean_matches_theory(self) -> None:
        r""":math:`\mathbb{E}[\log(S_T/S_0)] = (\mu - \sigma^2/2)T`."""
        log_returns = torch.log(self._terminal() / S0)
        expected = (MU - 0.5 * SIGMA**2) * MATURITY
        sample_mean = float(log_returns.mean())
        std_error = float(log_returns.std()) / math.sqrt(MOMENT_PATHS)

        assert abs(sample_mean - expected) < SIGMA_TOLERANCE * std_error, (
            f"E[log S_T/S_0] = {sample_mean:.8f}, theory {expected:.8f}"
        )

    def test_log_return_variance_matches_theory(self) -> None:
        r""":math:`\operatorname{Var}[\log(S_T/S_0)] = \sigma^2 T`.

        The sharpest single test of the increment generator: it pins down the
        variance of the underlying normals, and would catch a missing
        :math:`\sqrt{\Delta t}` scaling immediately.
        """
        log_returns = torch.log(self._terminal() / S0)
        expected = SIGMA**2 * MATURITY
        sample_variance = float(log_returns.var(unbiased=True))
        std_error = expected * math.sqrt(2.0 / MOMENT_PATHS)

        assert abs(sample_variance - expected) < 10.0 * std_error, (
            f"Var[log S_T/S_0] = {sample_variance:.8f}, theory {expected:.8f}"
        )

    def test_log_returns_are_not_skewed(self) -> None:
        """Log returns must be symmetric; skew would signal a broken normal map."""
        log_returns = torch.log(self._terminal() / S0)
        centred = log_returns - log_returns.mean()
        skew = float((centred**3).mean() / centred.std() ** 3)
        # se(skew) ~ sqrt(6/M) for a normal sample.
        assert abs(skew) < 6.0 * math.sqrt(6.0 / MOMENT_PATHS), f"skew {skew:.5f}"

    def test_paths_are_strictly_positive_and_finite(self) -> None:
        paths = philox_simulate_gbm(
            S0, MU, SIGMA, 100_000, N_STEPS, DT, seed=SEED, dtype=torch.float64
        )
        assert torch.all(paths > 0.0)
        assert torch.all(torch.isfinite(paths))

    def test_first_column_is_exactly_spot(self) -> None:
        paths = philox_simulate_gbm(
            S0, MU, SIGMA, 10_000, N_STEPS, DT, seed=SEED, dtype=torch.float64
        )
        assert torch.all(paths[:, 0] == S0)

    def test_intermediate_marginal_also_matches_theory(self) -> None:
        r"""Not only :math:`S_T`: the law at an interior date must be right too.

        A carry bug in the chunked scan could leave the terminal column correct
        while corrupting the middle of the path, so an interior marginal is
        checked independently.
        """
        paths = philox_simulate_gbm(
            S0, MU, SIGMA, MOMENT_PATHS, N_STEPS, DT, seed=SEED, dtype=torch.float64
        ).detach()
        mid_index = N_STEPS // 2
        t_mid = mid_index * DT

        log_returns = torch.log(paths[:, mid_index] / S0)
        expected_mean = (MU - 0.5 * SIGMA**2) * t_mid
        expected_var = SIGMA**2 * t_mid
        std_error = float(log_returns.std()) / math.sqrt(MOMENT_PATHS)

        assert abs(float(log_returns.mean()) - expected_mean) < SIGMA_TOLERANCE * std_error
        assert abs(float(log_returns.var(unbiased=True)) - expected_var) < 10.0 * (
            expected_var * math.sqrt(2.0 / MOMENT_PATHS)
        )


@requires_triton
class TestRandomStreamQuality:
    """The per-program key scheme must give reproducible, independent streams."""

    def test_same_seed_reproduces_bitwise(self) -> None:
        first = philox_simulate_gbm(S0, MU, SIGMA, 50_000, 64, DT, seed=SEED)
        second = philox_simulate_gbm(S0, MU, SIGMA, 50_000, 64, DT, seed=SEED)
        assert torch.equal(first, second)

    def test_different_seeds_give_different_paths(self) -> None:
        first = philox_simulate_gbm(S0, MU, SIGMA, 50_000, 64, DT, seed=SEED)
        second = philox_simulate_gbm(S0, MU, SIGMA, 50_000, 64, DT, seed=SEED + 1)
        assert not torch.allclose(first, second)

    def test_no_duplicate_paths_across_program_boundaries(self) -> None:
        """Aliased offsets would make distinct paths identical -- the silent bug.

        With ``BLOCK_M`` paths per Philox stream, an addressing error would most
        likely repeat paths at a stride of ``BLOCK_M``. Comparing every path
        against the one ``BLOCK_M`` positions away catches exactly that.
        """
        n_paths = 8_192
        paths = philox_simulate_gbm(
            S0, MU, SIGMA, n_paths, 32, DT, seed=SEED, dtype=torch.float64
        ).detach()

        terminal = paths[:, -1]
        # No two paths should coincide to double precision.
        assert torch.unique(terminal).numel() == n_paths, "duplicate terminal values"

        for stride in (16, 32, 64):
            left, right = terminal[:-stride], terminal[stride:]
            assert not torch.allclose(left, right), (
                f"paths repeat at stride {stride}: Philox offsets are aliasing"
            )

    def test_streams_are_uncorrelated_across_blocks(self) -> None:
        """Adjacent Philox keys must not produce correlated increments."""
        n_paths = 65_536
        paths = philox_simulate_gbm(
            S0, MU, SIGMA, n_paths, 64, DT, seed=SEED, dtype=torch.float64
        ).detach()
        log_returns = torch.log(paths[:, -1] / S0)

        half = n_paths // 2
        first, second = log_returns[:half], log_returns[half:]
        centred_first = first - first.mean()
        centred_second = second - second.mean()
        correlation = float(
            (centred_first * centred_second).mean()
            / (centred_first.std() * centred_second.std())
        )
        # se(correlation) ~ 1/sqrt(half) under independence.
        assert abs(correlation) < 5.0 / math.sqrt(half), f"correlation {correlation:.5f}"


# ==========================================================================
# Tier 3 -- rematerialised gradients (GPU only)
# ==========================================================================
@requires_triton
class TestRematerialisation:
    """Strict finite-difference validation of the rematerialised adjoint.

    The seed fixes the entire sample, so a bump re-draws identical increments
    and bump-and-revalue is an exact oracle. This is the only place a
    rematerialisation bug can be caught: only Vega reads :math:`Z`, so a
    mismatch between the forward's and backward's randoms shows up as a wrong
    Vega alongside a *correct* Delta and Rho.
    """

    N_PATHS = 20_000
    N_STEPS = 64
    STEP = MATURITY / 64

    @classmethod
    def _functional(
        cls, s0, mu, sigma, weights: torch.Tensor
    ) -> torch.Tensor:
        """A smooth scalar functional of the whole path matrix.

        Fixed weights over every column exercise the full time axis, and
        smoothness (no ``max``/kink) means the finite-difference truncation
        error is the only error present -- so a tight tolerance is meaningful.
        """
        paths = philox_simulate_gbm(
            s0, mu, sigma, cls.N_PATHS, cls.N_STEPS, cls.STEP,
            seed=SEED, dtype=torch.float64,
        )
        return (paths * weights).sum() / cls.N_PATHS

    @classmethod
    def _weights(cls) -> torch.Tensor:
        generator = torch.Generator(device="cuda").manual_seed(SEED + 7)
        return torch.randn(
            (1, cls.N_STEPS + 1), device="cuda", dtype=torch.float64,
            generator=generator,
        )

    def test_aad_greeks_match_central_differences(self) -> None:
        weights = self._weights()

        s0 = torch.tensor(S0, device="cuda", dtype=torch.float64, requires_grad=True)
        mu = torch.tensor(MU, device="cuda", dtype=torch.float64, requires_grad=True)
        sigma = torch.tensor(
            SIGMA, device="cuda", dtype=torch.float64, requires_grad=True
        )
        self._functional(s0, mu, sigma, weights).backward()
        aad = {
            "delta": float(s0.grad),
            "rho": float(mu.grad),
            "vega": float(sigma.grad),
        }

        base = {"s0": S0, "mu": MU, "sigma": SIGMA}
        finite_difference = {}
        for name, key in (("delta", "s0"), ("rho", "mu"), ("vega", "sigma")):
            step = 1e-6 * max(abs(base[key]), 1.0)
            with torch.no_grad():
                up = dict(base)
                up[key] = base[key] + step
                down = dict(base)
                down[key] = base[key] - step
                value_up = float(
                    self._functional(up["s0"], up["mu"], up["sigma"], weights)
                )
                value_down = float(
                    self._functional(down["s0"], down["mu"], down["sigma"], weights)
                )
            finite_difference[name] = (value_up - value_down) / (2.0 * step)

        for name in ("delta", "rho", "vega"):
            assert math.isclose(
                aad[name], finite_difference[name], rel_tol=1e-6, abs_tol=1e-9
            ), (
                f"{name}: AAD {aad[name]!r} vs FD {finite_difference[name]!r}. "
                "A wrong Vega with a correct Delta/Rho means the backward "
                "rematerialised different random numbers than the forward."
            )

    def test_vega_is_nonzero_so_the_test_has_teeth(self) -> None:
        """Guard the guard: a zero Vega would make the FD check vacuous."""
        weights = self._weights()
        sigma = torch.tensor(
            SIGMA, device="cuda", dtype=torch.float64, requires_grad=True
        )
        self._functional(S0, MU, sigma, weights).backward()
        assert abs(float(sigma.grad)) > 1e-6

    def test_delta_matches_closed_form(self) -> None:
        r"""For a linear functional, :math:`\partial/\partial S_0` is exact.

        Since :math:`S_{m,k} = S_0 e^{L_{m,k}}` is homogeneous of degree one in
        :math:`S_0`, any linear functional :math:`F` satisfies
        :math:`\partial F/\partial S_0 = F/S_0` exactly -- no Monte-Carlo error.
        A rare closed-form check on a kernel gradient.
        """
        weights = self._weights()
        s0 = torch.tensor(S0, device="cuda", dtype=torch.float64, requires_grad=True)
        value = self._functional(s0, MU, SIGMA, weights)
        value.backward()
        assert math.isclose(float(s0.grad), float(value) / S0, rel_tol=1e-10)

    def test_partial_gradients_are_respected(self) -> None:
        weights = self._weights()
        s0 = torch.tensor(S0, device="cuda", dtype=torch.float64, requires_grad=True)
        mu = torch.tensor(MU, device="cuda", dtype=torch.float64)  # no grad
        sigma = torch.tensor(
            SIGMA, device="cuda", dtype=torch.float64, requires_grad=True
        )
        self._functional(s0, mu, sigma, weights).backward()

        assert s0.grad is not None
        assert sigma.grad is not None
        assert mu.grad is None

    def test_backward_is_deterministic(self) -> None:
        weights = self._weights()
        results = []
        for _ in range(2):
            s0 = torch.tensor(
                S0, device="cuda", dtype=torch.float64, requires_grad=True
            )
            sigma = torch.tensor(
                SIGMA, device="cuda", dtype=torch.float64, requires_grad=True
            )
            self._functional(s0, MU, sigma, weights).backward()
            results.append((float(s0.grad), float(sigma.grad)))
        assert results[0] == results[1]

    def test_double_backward_raises_rather_than_lying(self) -> None:
        s0 = torch.tensor(S0, device="cuda", dtype=torch.float64, requires_grad=True)
        paths = philox_simulate_gbm(
            s0, MU, SIGMA, 1_024, 16, DT, seed=SEED, dtype=torch.float64
        )
        (grad_s0,) = torch.autograd.grad(paths.sum(), s0, create_graph=True)
        with pytest.raises(RuntimeError):
            torch.autograd.grad(grad_s0, s0)


@requires_triton
class TestMemoryFootprint:
    """The whole point of Phase 4: AAD must cost no extra O(MN) allocation."""

    def test_peak_memory_is_close_to_the_output_tensor_alone(self) -> None:
        """Forward peak should approach ``M*(N+1)*element_size``, nothing more."""
        n_paths, n_steps = 200_000, 252
        expected_bytes = n_paths * (n_steps + 1) * 4  # float32

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        paths = philox_simulate_gbm(
            S0, MU, SIGMA, n_paths, n_steps, DT, seed=SEED, dtype=torch.float32
        )
        peak = torch.cuda.max_memory_allocated()

        # Allow 25% headroom for the partial buffers, the packed params and
        # allocator rounding. A regression that reinstated a dW-sized buffer
        # would blow straight past this.
        assert peak < 1.25 * expected_bytes, (
            f"peak {peak / 1024**2:,.1f} MiB exceeds output-only budget "
            f"{expected_bytes / 1024**2:,.1f} MiB by more than 25%"
        )
        del paths

    def test_backward_adds_no_o_mn_allocation(self) -> None:
        """Rematerialisation means backward allocates no Z and no grad_dW."""
        n_paths, n_steps = 200_000, 252
        element_bytes = n_paths * (n_steps + 1) * 4

        s0 = torch.tensor(S0, device="cuda", dtype=torch.float32, requires_grad=True)
        sigma = torch.tensor(
            SIGMA, device="cuda", dtype=torch.float32, requires_grad=True
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        paths = philox_simulate_gbm(
            s0, MU, sigma, n_paths, n_steps, DT, seed=SEED, dtype=torch.float32
        )
        paths.sum().backward()
        peak = torch.cuda.max_memory_allocated()

        # Output plus the incoming adjoint is 2 x O(MN); a stored Z or a
        # grad_dW would push this to 3-4x.
        assert peak < 2.6 * element_bytes, (
            f"backward peak {peak / 1024**2:,.1f} MiB suggests an extra "
            f"O(M*N) buffer (budget {2.6 * element_bytes / 1024**2:,.1f} MiB)"
        )
