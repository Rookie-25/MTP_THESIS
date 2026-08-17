# PROJECT_STATE.md

**Project:** GPU-Native Differentiable AAD Monte-Carlo XVA & Greeks Engine with Neural-SDE / Rough-Volatility Calibration
**Last updated:** 2026-08-17

---

## Current Phase & Module

**Phase 1 (Month 1) — Baseline CPU/GPU Autodiff Simulator: COMPLETE.**

**Phase 2 (Month 2) — Exposure Profiling, Collateral & CVA/DVA: COMPLETE.**

**Phase 3 (Month 3) — Custom CUDA/Triton kernels: READY TO START, on Google Colab.**

All Phase 2 deliverables are implemented, tested and benchmarked: exposure profiles (EE/ENE/PFE/EPE), collateralised exposure under a CSA (threshold, MTA, MPOR), unilateral CVA/DVA, end-to-end AAD sensitivities through the full chain, a manual inspection sandbox, and an empirical O(1)-vs-O(n) scaling benchmark. **74/74 tests passing.**

## Files created/modified

```
xva-cuda-engine/
├── PROJECT_STATE.md
├── pyproject.toml                 # pytest config (pythonpath=["."]), deps: torch>=2.4, numpy>=1.26
├── data/                           # empty, placeholder for market data fetchers
├── benchmarks/                     # empty, placeholder for Phase 3 CPU-vs-GPU benchmarks
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── gbm.py                 # GBMSimulator, simulate_gbm, draw_brownian_increments, resolve_device
│   ├── pricer/
│   │   ├── __init__.py
│   │   ├── analytic.py            # Black-Scholes call price/delta/vega/rho, equity_forward_value (closed-form oracle)
│   │   ├── options.py             # european_call_payoff/price, SwapLeg, portfolio_swap_mtm/price,
│   │   │                          # resolve_rate_and_drift (MADE PUBLIC in Phase 2, was _resolve_rate_and_drift),
│   │   │                          # make_european_call_price_fn / make_portfolio_swap_price_fn (CRN closures)
│   │   └── greeks.py              # aad_greeks, finite_difference_greeks (bump_and_revalue_greeks alias),
│   │                              # compare_greeks, format_comparison, GreekResult/GreekComparison
│   ├── xva/                        # ---------- PHASE 2 ----------
│   │   ├── __init__.py            # re-exports the full Phase 2 public API
│   │   ├── exposure.py            # EE / ENE / PFE / EPE profiles from an MtM surface:
│   │   │                          #   positive_exposure, negative_exposure,
│   │   │                          #   expected_exposure, expected_negative_exposure,
│   │   │                          #   differentiable_quantile, potential_future_exposure,
│   │   │                          #   expected_positive_exposure, exposure_standard_error,
│   │   │                          #   compute_exposure_profile -> ExposureProfile dataclass,
│   │   │                          #   as_tensor_like (shared scalar->tensor coercion helper)
│   │   │                          # COLLATERAL / CSA (variation margin):
│   │   │                          #   CSATerms (threshold, MTA, MPOR, threshold_post, initial_balance),
│   │   │                          #   mpor_lag_steps, collateral_required (soft-threshold shrinkage),
│   │   │                          #   collateral_balance (vectorised when MTA=0, sequential when MTA>0),
│   │   │                          #   collateralized_exposure, expected_collateralized_exposure,
│   │   │                          #   compute_collateralized_exposure_profile
│   │   └── cva.py                 # flat-hazard credit model + discrete CVA/DVA integral:
│   │                              #   survival_probability, marginal_default_probability,
│   │                              #   discount_factors, compute_unilateral_cva/dva,
│   │                              #   compute_xva -> XVAResult, make_cva_valuation_fn (single-graph
│   │                              #   GBM->MtM->EE->CVA closure), cva_aad_greeks,
│   │                              #   cva_bump_and_revalue_greeks
│   ├── csrc/__init__.py           # empty stub, Phase 3 target
│   └── api/__init__.py            # empty stub, Phase 5 target
├── manual_sandbox.py               # interactive CLI inspection harness: nudge market/credit
│                                   # params, print CVA + AAD Greeks, write sandbox_exposures.png
├── benchmarks/
│   └── bench_phase2.py             # AAD O(1) vs FD O(n) scaling sweep, ASCII table + optional CSV
└── tests/
    ├── conftest.py                 # repo-root sys.path fallback, seeds torch, `device` fixture (cpu/cuda param)
    ├── test_phase1.py              # 15 tests, all passing
    └── test_phase2.py              # 59 tests, all passing (33 core + 26 collateral/CSA)
```

**Full suite: 74/74 passing** (`python -m pytest tests/ -q`, ~31s on CPU).

## Verified math / working components

### Phase 1 (unchanged, re-verified after the `resolve_rate_and_drift` refactor)

- `simulate_gbm` uses the *exact* log-Euler GBM scheme (no discretisation bias); terminal mean matches `S0*exp(rT)` within 6 MC standard errors.
- MC call price inside the Black-Scholes 95% CI; portfolio swap matches `equity_forward_value`; multi-leg netting additive.
- AAD vs FD Greeks agree to < `1e-3` absolute (Delta, Vega), central and forward schemes.
- AAD `n_valuations == 1` vs FD `2n + 1`.

### Phase 2 (new)

- **Exposure non-negativity** — `EE(t)`, `ENE(t)`, `PFE(t)` all verified `>= 0` elementwise (the explicit Phase 2 requirement), and the identity `V = max(V,0) - max(-V,0)` holds exactly (bitwise).
- **`differentiable_quantile`** matches `torch.quantile` to `1e-12` at levels `{0, 0.05, 0.5, 0.95, 0.99, 1}`. Implemented via `sort` + `index_select` rather than `torch.quantile` because the latter hard-caps input at 2^24 elements (readily exceeded by a realistic `(n_paths, n_steps)` surface). Gradient verified to be a one-hot adjoint summing to 1.
- **Credit curve** — `Q(t) = exp(-λt)` verified against closed form to `1e-14`; `Q(0) = 1`; strictly decaying; marginal default probabilities telescope to `1 - exp(-λT)` to `1e-12`; `λ = 0` gives exactly zero.
- **CVA analytic limits** — `λ = 0` → CVA exactly `0`; `R = 1` → CVA exactly `0`; CVA strictly increasing in `λ`; deterministic flat unit EE reproduces a hand-computed `(1-R)Σ dPD_i·DF(t_i)` to `1e-12` (isolates the discrete integrator from MC noise).
- **Discretisation conventions** — `"endpoint"` (default, matches the brief's formula) and `"average"` (trapezoidal) verified to converge to each other at `O(Δt)`: refining the grid 8x cuts their relative gap ~8x across `n_steps ∈ {8, 64, 512}`. This pins down both branches of the integrator; a fixed-percentage bound would not have.
- **Gradient tracking through the full chain** — `EE`, `ENE`, `PFE`, `EPE` all carry `requires_grad`; `cva.backward()` populates finite grads on `s0`, `sigma`, `hazard_rate` leaves; two independent forward+backward passes give bitwise-identical grads (proves no in-place tape corruption).
- **AAD CVA sensitivities vs bump-and-revalue (the Phase 2 acceptance criterion)** — measured agreement, `M = 100,000` paths, `N = 48` steps, float64, common random numbers:

  | param | fd-central | aad | abs err | rel err |
  |---|---|---|---|---|
  | `hazard_rate` | 3.5808348250 | 3.5808348181 | 6.96e-09 | 1.94e-09 |
  | `s0` | 0.0054433583 | 0.0054433514 | 6.84e-09 | 1.26e-06 |
  | `sigma` | 0.1498183334 | 0.1498186394 | 3.06e-07 | 2.04e-06 |

  Base CVA `0.0724199432` identical to machine precision between methods (`|diff| = 0`). **Max abs error `3.06e-07`, comfortably inside the `1e-3` requirement** — and inside a `5e-3` *relative* bound that the test also asserts, since absolute tolerance alone is weak when CVA is ~0.07.
- **Efficiency claim asserted, not just stated** — AAD `n_valuations == 1` vs FD `n_valuations == 7` for 3 risk factors. Measured wall-clock on CPU: **AAD 315 ms vs FD 1038 ms = 3.30x speedup**, and the gap grows linearly with the number of risk factors (FD needs `2n+1` revaluations, AAD needs 1 regardless of `n`).
- **Independent derivation cross-check** — `∂CVA/∂λ` verified to `1e-10` relative against a hand-derived semi-analytic formula `(1-R)Σ EE(t_i)·DF(t_i)·[t_i e^{-λt_i} - t_{i-1} e^{-λt_{i-1}}]`, computed without touching autograd at all.
- **Economic sign checks** — `∂CVA/∂λ > 0`, `∂CVA/∂σ > 0`, `∂CVA/∂S0 > 0` for the net-long test netting set.

### Phase 2 — collateral / CSA (new)

- **`EE_collat(t) ≤ EE_uncollat(t)` verified elementwise** across the full profile for all five CSA configurations tested: perfect margining, threshold-only, MPOR-only, threshold+MTA+MPOR, and a one-way CSA where we never post.
- **Exact closed-form identity at MPOR = 0**: collateralised pathwise exposure equals `min(V⁺, H)` **bitwise** (`max|diff| = 0.0`). This pins down both branches of the shrinkage function with no Monte-Carlo tolerance at all.
- **Perfect margining** (H=0, MTA=0, MPOR=0) drives exposure to *exactly* zero, and hence CVA to exactly zero.
- **Monotonicity** verified in both frictions: peak EE is non-decreasing in MPOR (0/5/10/20 business days) and mean EE is non-decreasing in threshold (0/2.5/5/10).
- **Limiting case** — a threshold above any attainable MtM reproduces the uncollateralised profile bitwise.
- **CVA ladder ordering** — `uncollateralised > threshold+MTA+MPOR > MPOR-only > perfect (= 0)`. Measured on a daily grid, 20k paths: peak EE falls **9.81 → 1.75** with MPOR-only margining and **9.81 → 2.80** with a threshold of 5 plus MTA 1.
- **Two code paths proven equivalent** — `collateral_balance` takes a vectorised shortcut when MTA = 0; driving the general sequential recursion with a negligible MTA (1e-300) reproduces it to `1e-12`, so the optimisation provably does not change the model.
- **Gradients survive both paths** — `cva.backward()` produces finite grads on `s0` and `sigma` through the vectorised shortcut *and* through the 253-step sequential MTA roll-forward (no in-place writes; `torch.where` against a constant mask routes the adjoint to the selected branch).
- **One-way CSA invariant** — with `threshold_post = inf` the balance is provably `>= 0` everywhere.

### Phase 2 — AAD scaling benchmark (new)

`benchmarks/bench_phase2.py`. Measured on CPU, 5,000 paths × 32 steps, float64, median of 2 repeats — a small smoke sweep; the defaults (20k paths, 64 steps, n up to 100) give cleaner numbers:

| n | forward (ms) | AAD (ms) | FD (ms) | speedup | AAD/fwd | AAD vals | FD vals | max abs err |
|---|---|---|---|---|---|---|---|---|
| 1 | 4.36 | 8.96 | 10.23 | 1.14x | 2.05x | 1 | 3 | 1.11e-06 |
| 5 | 3.18 | 8.65 | 37.66 | 4.36x | 2.71x | 1 | 11 | 4.69e-07 |
| 10 | 3.08 | 8.46 | 86.45 | 10.22x | 2.74x | 1 | 21 | 3.06e-07 |

**Over a 10× increase in risk factors, AAD wall time grew 0.94× (i.e. flat) while FD grew 8.45×.** The `AAD/fwd` column holds at ~2.0–2.7× — squarely inside the 2–5× constant that reverse-mode theory predicts, and independent of `n`. `forward (ms)` also stays flat, confirming the multi-factor mock adds risk factors without inflating the simulation, so the comparison is fair. Both methods agree to ~1e-6 throughout, so the timing comparison is between two methods that actually produce the same answer.

## Known bugs / technical blockers

1. **BLOCKER #1 — BYPASSED, not fixed. Local GPU unusable; Phase 3 moves to Google Colab.** `Get-PnpDevice` on the RTX 3050 Laptop GPU reports `CM_PROB_FAILED_POST_START`; `nvidia-smi` fails with a permissions error; `torch.cuda.device_count() == 0` even though PyTorch 2.4.0+cu121 is CUDA-built (`torch.backends.cuda.is_built() == True`) and `nvcuda.dll` loads fine. This is an NVIDIA driver/device fault, not a PyTorch install issue.

   **Strategic decision (2026-08-17): stop trying to repair the local driver and migrate Phase 3 to a cloud GPU (Google Colab).** Rationale: the thesis needs a *known-good, reproducible* CUDA environment for kernel development and for benchmark numbers that go into the write-up; debugging a laptop hybrid-graphics driver fault is unbounded work that produces no thesis output. Colab also gives a newer datacentre-class GPU (T4/L4/A100 depending on tier) with `nvcc` and Triton preinstalled, which is strictly better for the Phase 3 kernel work than a 4 GB laptop 3050 would have been.

   **Consequences to manage:**
   - All code is already device-agnostic (`resolve_device()` picks CUDA automatically), so **no source changes are needed** to run on Colab — only an environment bootstrap.
   - Colab sessions are ephemeral: the repo must be cloned (or mounted from Drive) per session, and any benchmark output must be written back out before the session dies.
   - Colab's Python/PyTorch versions differ from the local 3.11.5 / 2.4.0+cu121. Pin and record the Colab versions alongside any published benchmark number so results stay reproducible.
   - The local machine remains the CPU correctness baseline; keep running the full test suite locally, and treat Colab as the GPU performance environment.
   - If the local driver is ever repaired it becomes a bonus second datapoint, not a dependency.
2. **RESOLVED (2026-08-14).** A stray `random.py` in `D:\MTP\AMEX` used to shadow the stdlib `random` module and break `import torch`. User renamed it to `dsa_random.py`; confirmed `import torch` now works from `D:\MTP\AMEX` directly.
3. **Design note, not a bug — PFE gradients are deliberately not FD-validated.** `PFE` differentiates through an *order statistic*, so its finite-difference derivative is a step function of the parameters at finite `M` and will not converge to the AAD value at any fixed bump size. This is a property of the estimator, documented in `src/xva/exposure.py`. CVA depends only on `EE` (a smooth sample mean), so the Phase 2 acceptance criterion is unaffected. If PFE sensitivities are ever needed for hedging, they require smoothing (e.g. a kernel-smoothed quantile or a sorting-network relaxation) — flag for the thesis write-up.
4. **Known FD bias worth writing up.** `EE` is a mean of `max(V_t, 0)`, so a bump of size `h` flips the sign of `V_t` on `O(Mh)` paths. Each crossing contributes a kink, so central differences converge at `O(h)` rather than `O(h²)` here. AAD returns the unbiased pathwise derivative. This is an *accuracy* argument for AAD on top of the speed argument, and is documented in `cva_bump_and_revalue_greeks`.
5. **DOCUMENTED MODEL LIMITATION — the pathwise collateral inequality fails under MPOR.** `EE_collat ≤ EE_uncollat` holds at the *expectation* level (verified across all five CSA scenarios), but **not pathwise** once `MPOR > 0`. If we posted collateral against a deeply negative MtM and the market reverses inside the margin period, we are exposed both to what they now owe us and to the collateral we posted and cannot recall: `V = -5 → C = -5`, then `V → +3` gives exposure `3 - (-5) = 8 > 3`. This is the modelled economics of a margin period, not a bug — and it is exactly why MPOR dominates collateralised CVA. Locked in by `test_pathwise_inequality_can_break_under_mpor`. **Worth a paragraph in the thesis write-up.**
6. **Simplifications currently baked in** (all deliberate, all documented): flat hazard rate (no CDS-curve bootstrap), no wrong-way risk (exposure ⊥ default), deterministic discounting, default attributed to grid dates, no initial margin (variation margin only), no trade flows during the MPOR, uniform time grid required by the collateral lag logic.
7. Dependencies (`torch`, `numpy`, `pytest`, `matplotlib`) are installed into the system Python 3.11.5, not a virtualenv. Consider a `venv` before the dependency list grows (torchsde, FastAPI, Redis, MLflow, yfinance/FRED).

## Next immediate task for the upcoming session

**Phase 3 — custom CUDA/Triton kernels for the SDE hot path, on Google Colab.**

Session-zero bootstrap (do this first, it is the only new thing):

1. Get the repo onto Colab — either `git clone` (once it is pushed to a remote; **it is not currently a git repository**, so `git init` + push is a prerequisite) or mount Google Drive and sync the folder.
2. Record the Colab environment: `torch.__version__`, `torch.version.cuda`, `nvidia-smi` output, GPU model. Pin these next to any benchmark number.
3. Re-run the full suite on Colab GPU to confirm the device-agnostic code path works end to end: the `device` fixture in `tests/conftest.py` already parametrises over CUDA and currently skips — it should start running.
4. Re-run `benchmarks/bench_phase2.py` on GPU to get the CPU-vs-GPU baseline *before* any kernel work, so the Phase 3 speedup has an honest reference point.

Then the actual Phase 3 work:

5. Profile the hot path. `simulate_gbm` is currently `cumsum` + `exp` over an `(M, N)` tensor — memory-bandwidth bound, and it materialises the full path matrix. That materialisation is the real target.
6. Write a fused Triton kernel for GBM path generation + payoff accumulation that avoids materialising all `M × N` intermediates. Start with Triton (portable, no `nvcc` toolchain fights) before dropping to raw CUDA C++ via `torch.utils.cpp_extension`.
7. **Critical constraint:** any custom kernel must supply a matching backward via `torch.autograd.Function`, or the entire AAD pipeline breaks. Validate every kernel against `torch.autograd.gradcheck` *and* against the existing CPU path — the Phase 1/2 test suites are the regression net.
8. Benchmark CPU vs GPU vs fused-kernel across path counts (10k → 10M), targeting the 10x–100x claim in the project objectives.

Deferred Phase 2 nice-to-haves (not blocking Phase 3):

- `data/` — real market data fetchers (yfinance spot/vol, FRED discount curve) to replace hardcoded `S0=100, σ=0.2`.
- Bootstrap a piecewise-constant hazard curve from a CDS term structure. `_integrate_credit_leg` already accepts an explicit `curve` argument, so this is an input change, not an algorithm change.
- Initial margin (SIMM-style) on top of the existing variation margin.
