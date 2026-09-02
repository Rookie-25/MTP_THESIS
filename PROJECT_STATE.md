# PROJECT_STATE.md

**Project:** GPU-Native Differentiable AAD Monte-Carlo XVA & Greeks Engine with Neural-SDE / Rough-Volatility Calibration
**Status source of truth:** this file. It carries no date on purpose -- a
date invites trusting it past its shelf life. Verify against the repo with
`git log --oneline -10` and `python -m pytest tests/ -q` before relying on
any claim below.

---

## Current Phase & Module

**Phase 1 (Month 1) — Baseline CPU/GPU Autodiff Simulator: COMPLETE.**

**Phase 2 (Month 2) — Exposure Profiling, Collateral & CVA/DVA: COMPLETE.**

**Phase 3 (Month 3) — Custom Triton kernels: COMPLETE AND GPU-VERIFIED.**

All Phase 2 deliverables are implemented, tested and benchmarked: exposure profiles (EE/ENE/PFE/EPE), collateralised exposure under a CSA (threshold, MTA, MPOR), unilateral CVA/DVA, end-to-end AAD sensitivities through the full chain, a manual inspection sandbox, and an empirical O(1)-vs-O(n) scaling benchmark.

**Phase 4 (Month 3-4) — In-kernel Philox RNG + rematerialisation: COMPLETE AND GPU-VERIFIED.**

**Phase 5 (Month 4) — Fused payoff/exposure reduction, O(1)-in-M memory: COMPLETE AND GPU-VERIFIED.**

**Phase 6 (Month 5) — Local volatility + non-linear adjoint: MATHS CPU-VERIFIED; KERNEL WRITTEN, NEVER EXECUTED.**

Phase 3 has a fused Triton GBM kernel with a hand-derived adjoint. Phase 4 removes the caller-supplied `dW` matrix entirely by generating increments in-kernel from a counter-based (Philox) RNG, and rematerialises them in the backward pass instead of storing them.

**MILESTONE 2026-08-21: the GPU tier executed for the first time, on Colab (Tesla T4, Python 3.12.13, pytest 8.4.2).**

Result, first run: 280 passed, 3 skipped, 1 failed of 284. The 3 skips are the inverse-condition `test_helper_raises_actionable_error` guards (correctly skipped *because* Triton is present). The 1 failure was a bug in one of my tests, not in any kernel (see "GPU findings" below) — fixed, re-run confirmed **282 passed, 3 skipped, 0 failed of 285** (one test added by the fix).

**RESOLVED — graph-attached scalar conversions.** Three sites converted a `requires_grad` tensor to a Python scalar, warning on Colab's PyTorch but silent on the local 2.4.0 build. That version gap made them cost three separate round-trips to find. A static scan flagged 115 candidate conversions across `tests/`, but a runtime detector showed only **3** genuinely involved a `requires_grad` tensor — blanket-detaching the rest would have been churn.

Two guards now make the class unrepeatable: `pyproject.toml` promotes PyTorch's own warning to an *error*, and `tests/conftest.py` adds an opt-in, version-independent detector (`STRICT_TENSOR_SCALAR=1`) that patches `Tensor.__float__`/`.item` and reports file, line and source for every graph-attached conversion. Verified by reintroducing the bug and confirming it is flagged.

**RESOLVED — the `d4998d2` grid tolerance.** Investigated and re-formulated; see "FIXED: the grid-uniformity tolerance was mis-formulated" below. Short version: the loosening was a *necessary* bug fix (the original bound rejected valid float32 grids from N~100), but `1e-4 * dt` was itself the wrong shape and would have broken again by N~2700. The check is now horizon-relative, N-independent, and lives in exactly one place.

What this converts from conjecture to measurement:

- **Phase 3 kernels** — forward parity vs `simulate_gbm` at all 8 shape/dtype combinations; `gradcheck_float64` PASSED; backward parity vs autograd at (1,1), (3,7), (1024,252); bitwise determinism; non-contiguous stride handling; `test_cva_greeks_match_between_backends` (full pipeline swap invisible downstream).
- **Phase 4 in-kernel Philox** — all five distributional moment tests PASSED (terminal mean/variance, log-return mean/variance, skew), interior marginal, `test_no_duplicate_paths_across_program_boundaries` and `test_streams_are_uncorrelated_across_blocks` (so the per-program-key scheme really does give independent streams), `test_aad_greeks_match_central_differences` (the rematerialisation canary — a wrong Vega with correct Delta/Rho would have meant the backward regenerated different randoms), and both `TestMemoryFootprint` tests.
- **The int64 addressing fix works.** `TestGlobalPointerAddressing` all green, and the >8.5M-path runs that previously died with `illegal memory access` now complete.
- **Phase 5 fused reduction** — `test_ee_matches_unfused_within_sampling_error`, `test_cva_matches_unfused_within_sampling_error`, `test_grid_size_does_not_change_the_result` (confirming the absolute-block-index Philox keying decouples results from grid size), `TestFusedGreeks::test_greeks_match_central_differences`, `test_credit_sensitivity_matches_closed_form`, and `test_backward_memory_is_also_independent_of_m`.

Local suite (CPU tier only, no Triton on Windows): **321 passed, 75 skipped**.
The 75 skips are GPU-tier tests; they run and pass on Colab except the 9
Phase 6 kernel tests, which have never executed anywhere.

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
│   ├── csrc/                       # ---------- PHASE 3 ----------
│   │   ├── __init__.py
│   │   └── triton_gbm.py          # fused Triton GBM kernel + hand-written adjoint:
│   │                              #   _fused_gbm_forward_kernel  (chunked time-axis scan with
│   │                              #     running carry; increment scale + cumsum + exp + S0 fused)
│   │                              #   _fused_gbm_backward_kernel (reverse chunk walk building the
│   │                              #     suffix sum via total-minus-prefix; per-program partial
│   │                              #     buffers reduced host-side for bitwise determinism)
│   │                              #   reference_gbm_backward     (pure-PyTorch transcription of the
│   │                              #     same adjoint -- CPU-testable AND double-differentiable)
│   │                              #   FusedGBMFunction (autograd.Function, once_differentiable)
│   │                              #   triton_simulate_gbm (drop-in for simulate_gbm)
│   │                              #   select_block_sizes, is_available, HAS_TRITON
│   │       triton_local_vol_cva.py # -------- PHASE 6 -------- local-vol kernel (UNRUN):
│   │                              #   _fused_local_vol_forward_kernel  (sequential time loop,
│   │                              #     NOT tl.cumsum -- sigma is state-dependent)
│   │                              #   _fused_local_vol_backward_kernel (sqrt(N) checkpointing
│   │                              #     in SRAM; masked tile access, no dynamic indexing)
│   │                              #   LocalVolParams, select_local_vol_blocks
│   │                              #   reference_local_vol_ee[_adjoint] (CPU-verified)
│   │                              #   reference_checkpointed_ee_adjoint (CPU-verified)
│   │                              #   FusedLocalVolCVAFunction, fused_local_vol_ee/_cva
│   │       triton_cva_fusion.py   # ---------- PHASE 5 ---------- fused O(N)-memory reduction:
│   │                              #   _fused_exposure_forward_kernel  (paths->MtM->floor->reduce,
│   │                              #     register accumulator, bounded grid-stride; NO path matrix)
│   │                              #   _fused_exposure_backward_kernel (rematerialises Z AND paths)
│   │                              #   build_affine_coefficients (portfolio -> B, C vectors)
│   │                              #   select_fused_block_sizes, FusedExposureFunction
│   │                              #   fused_expected_exposure, fused_cva
│   │                              #   reference_fused_exposure[_backward] (CPU-testable)
│   │       triton_philox_gbm.py   # ---------- PHASE 4 ---------- in-kernel RNG:
│   │                              #   _philox_gbm_forward_kernel  (tl.randn in SRAM; NO dW ptr)
│   │                              #   _philox_gbm_backward_kernel (rematerialises Z from the same
│   │                              #     (seed+pid, offset); no grad_dW, no stored Z)
│   │                              #   reference_philox_forward/backward (CPU-testable, 2nd-order OK)
│   │                              #   FusedPhiloxGBMFunction, philox_simulate_gbm
│   │                              #   validate_offset_scheme, MAX_PHILOX_OFFSET (int32 alias guard)
│   └── api/__init__.py            # empty stub, Phase 5 target
├── manual_sandbox.py               # interactive CLI inspection harness: nudge market/credit
│                                   # params, print CVA + AAD Greeks, write sandbox_exposures.png
├── benchmarks/
│   ├── _harness.py                 # SHARED measurement primitives -- Measurement, measure(),
│   │                               # reset_cuda, is_oom, free_vram_bytes, markdown_table.
│   │                               # cuda.Event timing exists ONLY here; the four bench
│   │                               # scripts each had their own copy before, which risked
│   │                               # cross-script numbers not being comparable.
│   ├── bench_all_phases.py         # WRAPPER: PyTorch baseline vs Phase 3/4/5 over an M sweep.
│   │                               # Markdown report (time / peak VRAM / speedup / survival
│   │                               # matrix) + --find-oom bisection for the exact baseline
│   │                               # OOM threshold. RUN on a T4 -- numbers below.
│   ├── profile_scaling.py          # single-kernel M sweep, rows labelled ramp-up/saturated
│   ├── bench_phase2.py             # AAD O(1) vs FD O(n) scaling sweep, ASCII table + optional CSV
│   ├── bench_phase3.py             # fused-vs-PyTorch time + peak VRAM sweep; torch.cuda.Event
│   │                               # timing, max_memory_allocated, OOM captured as a result
│   ├── bench_phase5.py             # Phase 4 (materialised paths) vs Phase 5 (fused);
│   │                               # M to 50M, pre-flight VRAM guard, O(N) memory evidence
│   ├── bench_phase4.py             # Phase 3 (dW in HBM) vs Phase 4 (in-kernel Philox);
│   │                               # M up to 20M, reports the OOM crossover + ceiling analysis
│   ├── plot_results.py             # publication figures: time vs M, VRAM vs M,
│   │                               # EE/PFE with and without collateral.
│   │                               # Validated palette; refuses to invent data.
│   └── bench_phase6.py             # Phase 6 Triton local-vol AAD vs PyTorch autograd.
│                                   # Forward time, backward time (CUDA events around
│                                   # .backward() ALONE), and peak VRAM -- each stage
│                                   # guarded SEPARATELY so a row can show a completed
│                                   # forward beside an OOM backward. Includes a
│                                   # cross-backend agreement check. NEVER RUN.
├── market_data/                    # NEW: market data + credit curve bootstrapping
│   ├── __init__.py                 # re-exports the public surface
│   └── fetcher.py                  # YieldCurve (linear-in-zero-rate, flat extrap),
│                                   # CreditCurve (piecewise-constant hazard),
│                                   # bootstrap_hazard_rates (Brent per pillar),
│                                   # clean_option_chain (forward moneyness),
│                                   # fetch_* wrappers (yfinance / FRED, lazy import)
└── tests/
    ├── conftest.py                 # repo-root sys.path fallback, seeds torch, `device` fixture (cpu/cuda param)
    ├── test_phase1.py              # 15 tests, all passing
    ├── test_phase2.py              # 59 tests, all passing (33 core + 26 collateral/CSA)
    ├── test_phase3.py              # 54 tests: 27 CPU-tier passing, 27 GPU-tier skipped locally
    ├── test_phase4.py              # 57 tests: 37 CPU-tier passing, 20 GPU-tier skipped locally
    ├── test_phase5.py              # 54 tests: 37 CPU-tier passing, 18 GPU-tier skipped locally
    ├── test_phase6.py              # 44 tests: surface, arbitrage penalties, non-linear adjoint
    │                               # (CPU only -- the maths was settled before any Triton)
    └── test_phase6_kernel.py       # 51 tests: 42 CPU-tier passing, 9 GPU-tier NEVER RUN
    ├── test_market_data.py         # 130 tests: 124 passing, 6 network-tier opt-in
    └── test_credit_curve_integration.py  # 61 tests: piecewise credit AAD,
                                    # SSVI->kernel bridge, figure parsing
```

**Full suite: 219 passed, 65 skipped** (`python -m pytest tests/ -q`, ~40s on CPU).

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

### Phase 3 — fused Triton kernel (new, PARTIALLY VERIFIED)

**Verified on CPU today (27 tests passing):**

- **The adjoint derivation is correct.** `reference_gbm_backward` — a pure-PyTorch transcription of exactly the formulas the Triton kernel implements — matches `torch.autograd` to `rel_tol=1e-11` on shapes `(1,1)`, `(1,8)`, `(7,3)`, `(64,252)`. This is the hard part of the phase and it is *done*, independent of any GPU.
- **The Ito correction in Vega is guarded by a dedicated test.** `∂ι/∂σ = ΔW − σΔt`; the second term is the most plausible thing to drop, and `test_vega_ito_correction_is_present` asserts the naive `ΔW`-only version differs, and that the discrepancy equals exactly `σΔt·ΣQ`.
- **The reference adjoint is double-differentiable**, so second-order work (Gamma, Hessian-vector products) has a validated fallback that the kernel path cannot provide.
- **Block-size heuristics**: `BLOCK_M`/`BLOCK_N` always powers of two, tile always ≤ 32 KiB SRAM budget, float64 tiles no larger than float32 tiles.
- **Graceful degradation**: the module imports cleanly with no Triton installed (stub `triton`/`tl` objects), so the CPU suite keeps collecting; calling the fused path raises an actionable `RuntimeError`.

**NOT yet verified — requires Colab:**

- The forward kernel's chunked scan (masking, carry propagation, stride handling).
- The backward kernel's reverse chunk walk and partial-buffer reduction.
- `gradcheck`, forward/backward parity vs PyTorch, determinism, non-contiguous input handling, and the end-to-end CVA-Greeks parity test.
- **No performance number has been measured.** `bench_phase3.py` has never executed. Do not quote a speedup or memory saving until it has.

**Design decisions worth recording:**

- **Determinism over speed in the reduction.** Scalar gradients use per-program partial buffers summed in PyTorch, *not* `tl.atomic_add`. Atomics make the result depend on program completion order, so two runs could differ in the last bits — unacceptable when validating against a finite-difference oracle, and float64 atomic support in Triton is patchy anyway.
- **Suffix sum without a reverse-scan primitive.** The backward uses `suffix[j] = chunk_total − inclusive_prefix[j] + P[j]`, needing only forward `cumsum`/`sum`. Avoids depending on `tl.flip`, which is not in every Triton version.
- **Scalars passed as a device tensor, not JIT scalar args.** Python floats get demoted to float32 by Triton, silently destroying float64 precision. Packing `[s0, mu, sigma, dt]` into a device tensor also avoids a host sync per call (which would otherwise show up in the benchmark).
- **`from __future__ import annotations` deliberately omitted** in `triton_gbm.py`: Triton inspects `tl.constexpr` annotations as live objects, and postponed (string) annotations would silently demote compile-time constants to runtime arguments.
- **No double backward** (`once_differentiable`). Raises rather than returning silent garbage. The PyTorch path retains double-backward support.

### Phase 4 — in-kernel Philox RNG + rematerialisation (new, PARTIALLY VERIFIED)

**Verified on CPU today (27 tests passing):**

- **The Phase 4 adjoint derivation is correct.** `reference_philox_backward` matches `torch.autograd` to `rel_tol=1e-11` on shapes `(1,1)`, `(1,8)`, `(5,3)`, `(64,252)`. This is an *independent* derivation from Phase 3's: the increment is reparameterised as `a + σ√Δt·Z` rather than `a + σ·dW`, so the Vega term changes to `√Δt·Z − σΔt`. A dedicated test asserts the `√Δt` factor is present, since a naive port of Phase 3's Vega would drop it.
- **The int32 offset-aliasing guard is tested.** `validate_offset_scheme` rejects any configuration whose offset range could wrap, and `test_the_naive_global_scheme_would_have_overflowed` documents the arithmetic: 8M paths fit in int32, 10M and 20M do not.
- **`reference_philox_backward` is double-differentiable**, preserving the second-order fallback.
- **Graceful degradation**: module imports with no Triton (reuses Phase 3's stubs); the helper raises an actionable `RuntimeError`.

**NOT yet verified — requires Colab:**

- The forward kernel's `tl.randn` usage and the distributional correctness of the generated increments.
- The backward kernel's rematerialisation (does it regenerate *identical* `Z`?).
- All 20 GPU-tier tests: terminal/interior moment tests, stream-independence tests, the finite-difference gradient check, and the peak-memory assertions.
- **No performance or memory number has been measured.** `bench_phase4.py` has never executed.

**ARCHITECTURAL DECISIONS (logged)**

1. **Path identity lives in the Philox *key*, not the counter.** This is the single most important decision in Phase 4. The obvious offset `m*n_steps + j` exceeds int32 at `M > ~8.5M` (N=252), and Triton's Philox **truncates its offset to 32 bits** — so at the 10M/20M path counts this phase targets, distinct `(path, step)` pairs would collide and the RNG would return *the same increments for different paths*. Nothing raises; the paths still look log-normal; the Monte-Carlo estimator is silently biased by correlation. The scheme adopted instead is `program_seed = seed + program_id`, `offset = local_m * n_steps + j`, keeping every offset under `BLOCK_M * N ≤ 16,128` for **any** M. Varying the key is the standard Random123 parallel-RNG idiom (Salmon et al. 2011). `validate_offset_scheme()` runs on every launch to convert this class of corruption into a loud failure.
2. **The offset is deliberately independent of `BLOCK_N` but dependent on `BLOCK_M`.** Using the *global* time index `j` means the backward may chunk time differently from the forward. But `local_m` and `program_id` tie the stream to `BLOCK_M` and the grid, so **forward and backward must use identical `BLOCK_M`** or rematerialisation silently returns different numbers. `BLOCK_M` is stored on the autograd context and reused verbatim.
3. **Recompute rather than store.** Storing `Z` for the backward would reinstate the exact `M×N` allocation Phase 4 exists to delete. Philox is a pure function of `(key, counter)`, so recomputation is bit-exact and costs a few integer rounds per element — far cheaper than an HBM round trip. Classic checkpointing trade (Griewank & Walther Ch. 12); recompute wins decisively for an RNG.
4. **`dW` is no longer a differentiable input**, so there is no `grad_dW`. The Phase 4 adjoint is therefore *simpler* than Phase 3's, and the backward allocates only three tiny per-program partial buffers.
5. **Non-tensor arguments in `autograd.Function`.** `(n_paths, n_steps, dt, seed)` are structurally non-differentiable and return `None` from `backward`.
6. **float32 normals under float64 accumulation.** `tl.randn` is a 32-bit Philox construction, so the normals carry ~7 significant digits even when the scan runs in float64. Irrelevant for Monte Carlo (sampling error dominates by orders of magnitude) and — importantly — it does **not** weaken the finite-difference validation, because `Z` is held *fixed* across a bump so its precision cancels out of the difference quotient.
7. **Statistical validation replaces bitwise validation.** Triton's RNG will never match `torch.randn` bitwise, so there is no reference trajectory to diff. Correctness is established via theoretical log-normal moments (with tolerances derived from *sample* standard errors, not hardcoded epsilons), stream-independence checks, and FD gradient agreement.
8. **Vega is the rematerialisation canary.** Only Vega reads `Z`. So a backward that regenerates the wrong randoms produces a **wrong Vega alongside a correct Delta and Rho** — that asymmetry is the diagnostic signature, and the FD test's failure message says so explicitly.

**HONEST SCOPE OF THE MEMORY CLAIM**

Eliminating `dW` **halves** peak memory and roughly **doubles** the attainable path count; it does not make memory independent of `M·N`, because the output path matrix remains:

| paths (fp32) | Phase 3 (dW + out) | Phase 4 (out only) | fits in 16 GiB? |
|---|---|---|---|
| 1M | 1.88 GiB | 0.94 GiB | both |
| 5M | 9.41 GiB | 4.71 GiB | both |
| 10M | 18.81 GiB | 9.42 GiB | **Phase 4 only** |
| 20M | 37.63 GiB | 18.85 GiB | neither |

On a 16 GiB card the practical limit moves from ~7.9M to ~15.9M paths. **M=20M needs an 80 GiB device** (A100) — on a T4/L4 both designs OOM at 20M, and the benchmark reports that rather than hiding it. Making peak `O(M)` requires fusing the payoff/exposure reduction into the kernel so paths are consumed as produced; that is Phase 5 and is deliberately not attempted here.

### Phase 4 — FIXED 2026-08-19: int32 global pointer overflow (crashed on Colab)

**Symptom:** `CUDA error: an illegal memory access was encountered` at M=10,000,000, N=252 during the first Colab run of `bench_phase4.py`.

**Root cause:** *not* the Philox offset (which `validate_offset_scheme` already guarded) but the **global memory** pointer arithmetic. `tl.arange` and `tl.program_id` yield **int32**, so `offs_m * out_stride_m` overflowed once `n_paths * (n_steps + 1)` passed INT32_MAX. Exact threshold: **8,488,077 paths** at N=252. At 10M it computes 2.53e9, wraps negative, and dereferences out of bounds.

**Two distinct 32-bit hazards exist in these kernels and must not be conflated:**

| hazard | symptom | fix |
|---|---|---|
| Philox *counter* overflow | **silent** path correlation, no error | keep offset small via per-program keys (`seed + pid`) |
| Global *pointer* overflow | **loud** illegal memory access, context poisoned | promote row index to `tl.int64` |

The first is statistical and silent; the second is fatal and immediate. They pull in *opposite* directions — global offsets must be widened to int64, while the Philox counter must stay int32 and small. Both invariants are now asserted by source-level tests.

**Fix applied (14 edits across both kernel files):** row indices promoted once per kernel into `row_out` / `row_dw` / `row_go` / `row_gdw` via `offs_m.to(tl.int64) * stride`, and column offsets via `offs_n.to(tl.int64)`. Every global load/store now derives from those int64 bases. `local_m` and `rng_offset` deliberately remain int32.

**Note this affected Phase 3 too** (`triton_gbm.py`), which had the identical bug in four places — it just had not been run at >8.5M paths yet. Both are fixed.

**Regression cover added (`TestGlobalPointerAddressing`, 10 tests, all CPU-runnable):** the exact break-even path count is pinned at 8,488,077; int64 headroom is verified to 1e9 paths; and two **source-level static checks** assert the `offs_m.to(tl.int64)` casts are present and the pre-fix int32 patterns have not returned. Static checks were chosen deliberately — reproducing the real failure needs a >8.5M-path GPU run, far too expensive for a normal test, so this fails locally in milliseconds instead of crashing an hour into a Colab benchmark.

**Benchmark hardening (`bench_phase4.py`):** a pre-flight VRAM check now refuses any configuration needing more than 90% of *free* memory (`torch.cuda.mem_get_info`), reporting `OOM (pred)` plus what it would have needed, without launching. This matters beyond tidiness: an illegal memory access **poisons the CUDA context for the whole process and cannot be caught**, so one bad size would abort the entire sweep. `_reset_cuda()` now also synchronises before dropping the cache and clears accumulated stats, and the budget is re-read between backends.

### Phase 5 — fused exposure reduction, O(N) memory (new, PARTIALLY VERIFIED)

`src/csrc/triton_cva_fusion.py`, `tests/test_phase5.py` (54 tests: **37 CPU-tier passing**, 18 GPU-tier skipped), `benchmarks/bench_phase5.py`.

**Verified on CPU today (37 tests passing):**

- **The affine collapse is exact.** `V[m,k] = B[k]*S[m,k] - C[k]` reproduces `portfolio_swap_mtm` to 4e-14 across four portfolios (single-leg, two-leg netted, three-leg with staggered maturity, mixed-sign). Verified separately that `B[k]` equals `dV/dS` via autograd — that matters because `B` is exactly what the adjoint contracts against, so an error there would corrupt Delta and Vega while leaving the forward EE perfectly correct.
- **The fused adjoint is correct.** `reference_fused_exposure_backward` matches autograd to `rel_tol=1e-10` on shapes `(1,1)`, `(3,4)`, `(17,9)`, `(256,64)`. This is a **third independent derivation** (Phase 3 used `dW`; Phase 4 used `sqrt(dt)*Z`; Phase 5 adds the exposure indicator, the `1/M`, and the `B[k]` factor).
- **Loop closed:** `reference_fused_exposure` equals `expected_exposure(portfolio_swap_mtm(...))` to 1e-12 on identical normals. So the only thing left unverified is the kernel itself.
- **Two specific bug-shapes locked in by dedicated tests:** the column-0 exclusion (including the non-existent increment at k=0 double-counts every path — the test asserts the buggy variant differs AND that the gap equals exactly the column-0 suffix term) and the Ito correction in Vega.
- Exposure floor kills gradients exactly where out-of-the-money; single-tile block selection covers the time axis, fits the 32 KiB SRAM budget, and keeps the Philox counter inside int32; long horizons fail loudly naming the Phase 4 fallback.

**NOT yet verified — requires Colab:** every kernel-level claim. The 18 GPU tests cover fused-vs-unfused EE/CVA agreement within sampling error, peak-memory independence of M, FD Greeks, grid-size invariance, and backward memory. **No performance or memory number has been measured.**

**ARCHITECTURAL DECISIONS (logged)**

1. **The affine collapse is what makes the fusion tractable.** A linear netting set of *any* size compresses to two length-(N+1) vectors: `B[k] = sum_i N_i*1{t<=T_i}*exp(-r(T_i-t))` and `C[k] = sum_i N_i*1{t<=T_i}*exp(-r(T_i-t))*K_i`. So the kernel signature is independent of portfolio size, and `dV/dS = B[k]` is a per-step constant identical across paths — which is what keeps the adjoint cheap.
2. **DEVIATION FROM BRIEF: the CVA integral is NOT fused into the kernel.** It is O(N) (~252 multiply-adds), so fusing it buys nothing measurable while costing hand-written adjoints for lambda and R inside the kernel plus a second implementation of the same integral to keep consistent. The kernel stops at the EE profile; `fused_cva()` composes the credit integral in PyTorch using the already-verified Phase 2 `compute_unilateral_cva`. Consequences, all favourable: `dCVA/dlambda` and `dCVA/dR` come from autograd exactly with zero new kernel code; the fused EE composes with any downstream functional (DVA, PFE measures, the Phase 2 collateral machinery), not just one CVA formula; one source of truth. Callers still get `(ee, cva)` and every sensitivity.
3. **Single time tile, not Phase 4's chunked scan.** The reverse suffix scan needs `S[m,k]` for *all* k simultaneously; re-deriving it chunk-by-chunk in reverse would cost O(N^2/BLOCK_N) work or an extra carry buffer. Price is a cap on N (~2500 daily steps in fp32, i.e. a ten-year horizon); beyond that `select_fused_block_sizes` raises and names `philox_simulate_gbm` as the O(M*N) fallback.
4. **Bounded grid + register accumulator.** Peak is `n_programs*(N+1)*element_size` — the grid is capped (default 4096) and each program strides over path blocks with a `BLOCK_T`-wide register accumulator, so there is no per-block HBM traffic and no M term anywhere in the memory formula.
5. **Philox keyed on the ABSOLUTE path-block index, not `program_id`.** An improvement on Phase 4, where the key was grid-coupled. Keying on the absolute block index makes the random stream reproducible regardless of how many programs launch, so forward and backward may use different grid sizes and still rematerialise identical draws. A GPU test asserts `max_programs=64` and `max_programs=4096` agree.
6. **Both 32-bit hazards handled, in opposite directions** (as established in Phase 4): the Philox counter stays int32 and small (`BLOCK_M*(N+1)` <= ~16k), while every global pointer offset is promoted to int64.
7. **Total Rho is REFUSED, not half-answered.** Discount factors are baked into the constant `B, C` before the kernel runs, so differentiating w.r.t. `rate` would silently return only the drift half. `fused_expected_exposure` raises if `rate` requires grad. Explicit refusal beats a silently partial Greek.
8. **Non-linear payoffs out of scope.** A call MtM is not affine in S, so the `B*S - C` collapse does not apply.

**HONEST SCOPE OF THE MEMORY CLAIM**

| paths (fp32, N=252) | Phase 4 path matrix | Phase 5 partials |
|---|---|---|
| 1M | 0.94 GiB | ~4 MiB |
| 5M | 4.71 GiB | ~4 MiB |
| 10M | 9.42 GiB | ~4 MiB |
| 50M | 47.13 GiB | ~4 MiB |

The claim is **bounded by a constant, not bitwise identical across M**: the grid is `min(ceil(M/BLOCK_M), max_programs)`, so a *small* M launches fewer programs and uses slightly *less*. What must never happen is growth with M. 50M paths is unreachable for Phase 4 on any current single GPU and routine for Phase 5 on a 16 GiB card — that is the headline, and it is the first phase where the memory result is qualitative rather than a constant factor.

### Phase 6 — local volatility + non-linear adjoint (new, MATH VERIFIED / KERNEL NOT WRITTEN)

`src/models/vol_surface.py`, `src/models/local_vol_paths.py`, `tests/test_phase6.py` (**44 tests, all passing on CPU** — no GPU tier, deliberately: the maths had to be settled before any Triton was written).

Blueprint covering Modules 1/2/4/5: https://claude.ai/code/artifact/ebeb0229-c0db-43e3-be68-140a2ba8693f

**THE CENTRAL FINDING — the Phase 3-5 adjoint structure does not survive state-dependent volatility.**

Phases 3-5 exploited an accident of constant sigma: the log-Euler recursion is *affine* in the state, so the adjoint collapses to a **suffix sum** computable with one `tl.cumsum`. That collapse is the entire reason those kernels are fast and memory-flat. With `sigma(t,S)` the one-step Jacobian is no longer 1:

    J_k = dX_{k+1}/dX_k = 1 + (dsigma_k/dX_k) * (sqrt(dt)*Z_k - sigma_k*dt)

so the reverse sweep is a **sequential recursion** `a_k = Xbar_k + a_{k+1}*J_k`. No parallel scan primitive applies. Measured error from reusing the suffix-sum shortcut anyway, N=252, smooth centred skew:

| mean abs(dsigma/dx) | sequential | suffix-sum | rel error |
|---|---|---|---|
| 0.0000 | 5.450408e+01 | 5.450408e+01 | 0.00% (Phase 3-5 case: exact) |
| 0.0489 | 5.389047e+01 | 5.438165e+01 | 0.91% |
| 0.1849 | 5.354359e+01 | 5.408721e+01 | 1.02% |
| 1.1795 | 5.137390e+01 | 5.202345e+01 | 1.26% |

**1% is not survivable**: it is a *bias*, not variance. It does not shrink as M grows, no statistical test on the exposure profile reveals it, and calibrating a surface by gradient descent on a systematically wrong gradient converges to a systematically wrong surface. The zero row is exactly what makes this easy to miss when porting.

**Memory consequence.** The reverse sweep needs sigma_k and dsigma_k/dX_k in *descending* k, but the forward produces them ascending and cannot be inverted (solving for X_k needs sigma_k, which depends on X_k — implicit). Phases 3-5 escaped this because S was recoverable in closed form from one cumsum. Three strategies, per tile (BLOCK_M=16, 4096 programs, N=252, fp32):

| strategy | extra memory | at N=252 | extra forward work |
|---|---|---|---|
| store full trajectory | O(B*N) | 66 MiB | none |
| **sqrt(N) checkpointing (RECOMMENDED)** | **O(B*sqrt(N))** | **4 MiB** | **~1 extra pass** |
| recompute from scratch | O(B) | 0.26 MiB | ~126x |

All three implemented in `local_vol_paths.py`; the checkpointed variant is asserted to agree with full storage to 1e-12 across checkpoint counts {1,2,5,10,100}. **All are O(1) in M — the Phase 5 headline survives. What changes is the constant: the defensible claim becomes "flat in M, linear in sqrt(N)", not "flat".**

**Verified on CPU (44 tests):**

- **CORRECTION TO THE BRIEF's g(k).** The brief's first term is `(1 - (y/w)*dw/dy)^2`; Gatheral-Jacquier is `(1 - k*w'/(2w))^2` — a factor 2 in the denominator. Not cosmetic: g converts a total-variance slice into the risk-neutral density, so `g >= 0 <=> p >= 0`. Integrating the induced density over k in [-8,8] at 160,001 points on a skewed SSVI slice: **correct form 1.00000000, brief's form 0.98676357** (1.3% mass error, so it is not a density). Pinned by `test_the_factor_of_two_matters`.
- **The two arbitrage conditions ARE the Dupire well-posedness conditions.** `sigma_LV^2 = (dw/dT)/g`, so calendar violation makes the numerator negative and butterfly violation makes the denominator vanish. Enforcing them is not an extra requirement bolted onto calibration — it is what makes sigma_LV well defined at all. Identity verified to rel_tol 1e-10.
- **Sequential adjoint == autograd** to rel_tol 1e-9 on shapes (1,1), (4,8), (32,64), (16,252), for dL/dS0, dL/dbase, dL/dslope.
- **Constraints discharged by construction, not by penalty**: ATM total variance is a cumsum of softplus increments, hence monotone for *any* parameter values (verified over 20 random draws); SSVI rho/eta/gamma map into their feasible sets.
- **A test caught a real numerical bug**: bare `tanh` saturates to *exactly* 1.0 in float64 once |raw| >~ 19, putting rho on the boundary where `1-rho^2 = 0`, the SSVI root degenerates to `|phi*k + rho|` (non-differentiable), and both butterfly conditions collapse. Optimisers reach such raw values routinely in a bad line search. Fixed with `PARAMETER_MARGIN = 1e-6` on all three constrained parameters.
- Penalty design: softplus hinge (finite and smooth for infeasible c, so it can *restore* feasibility) vs log-barrier (undefined for c<=0, enforces interiority). Schedule is hinge-then-barrier. Components reported separately so a calibration log shows *which* constraint binds.
- Forward-mapping guard: local vol must be read at `k = log(S_t/F_t)`, not `log(S_t/S_0)`. The two agree at t=0, so an inception-only test cannot distinguish them; at t=2 with 4% net drift they differ by 0.08 in log-moneyness.

**ARCHITECTURAL NOTES (corrections to the brief's Module 2 framing)**

1. **Triton has no explicit shared-memory staging.** There is no primitive for "stage a tile in SRAM, then gather from it with computed indices". `tl.make_block_ptr` describes structured *strided* tiles, not data-dependent gathers, and a loaded `tl.tensor` cannot be indexed by a runtime tensor. What is achievable is `tl.load(surface_ptr + computed_offsets)`, which is **L1/L2 resident** because a realistic grid is tiny (24x50x4B ~ 4.7 KiB). The mechanism is cache residency, not SRAM staging — worth describing accurately in a thesis.
2. **The stronger design removes the gather entirely**: calibrate the grid/spline in PyTorch, then compress to a parametric form (SSVI coefficients, or a low-order Chebyshev expansion in (t, log S)). 20-50 coefficients broadcast identically to every lane — pure register arithmetic, no memory access, no divergence at all.
3. **Warp divergence and memory divergence are different problems.** Warp divergence is fully avoidable here (clamp with `tl.minimum`/`tl.maximum`, select with `tl.where`, never `if` on per-lane data). Memory divergence is unavoidable with a grid gather and is mitigated only by cache residency or by removing the gather.
4. **BLOCK_M must shrink** from Phase 5's 32 to ~16: the surface tile competes with the path tile for the same SRAM budget.
5. **Discretisation bias is now a real error source.** Log-Euler with frozen vol is weak order 1, whereas Phases 1-5 used the *exact* GBM solution. Keep it separate from MC error when attributing benchmark discrepancies.
6. **Bandwidth utilisation is a misleading headline metric for a fused kernel.** "% of theoretical peak GB/s" will read *low* precisely because the fusion worked. Report arithmetic intensity and roofline position instead; keep the bandwidth figure as clearly-labelled supporting evidence that HBM traffic collapsed.

**NOT DONE:** `_fused_local_vol_cva_kernel` is specified in the blueprint but **not written**. Gaps flagged there: the `acc_ee` accumulator sketch is written for clarity not efficiency; whether `tl.randn` accepts a 1-D offset inside a nested loop without per-step recompilation is unresolved; whether the N=252 sequential loop unrolls acceptably needs first-contact testing.

### MEASURED: peak memory has two regimes, and only one is flat

The single Colab failure was `test_peak_is_flat_across_two_decades_of_paths`, asserting `peaks[1M] < 10 * peaks[10k]`. Observed **325,120 B at M=10k vs 4,476,928 B at M=1M — a 13.8x ratio.**

**This was my test being wrong, not the kernel.** The launch grid is `min(ceil(M / BLOCK_M), max_programs)`, which creates two regimes:

| M | n_blocks | n_programs | predicted partial | observed peak |
|---|---|---|---|---|
| 10,000 | 313 | 313 | 0.30 MiB | 0.31 MiB |
| 100,000 | 3,125 | 3,125 | 3.02 MiB | — |
| **131,072** | **4,096** | **4,096** | **3.95 MiB** | — (saturation point) |
| 1,000,000 | 31,250 | 4,096 | 3.95 MiB | 4.27 MiB |
| 5,000,000 | 156,250 | 4,096 | 3.95 MiB | — |
| 50,000,000 | 1,562,500 | 4,096 | 3.95 MiB | — |

At N=252/fp32, `BLOCK_M=32` and the grid saturates at `4096 x 32 = 131,072` paths. Below that the grid grows with M, so the partial buffer grows with M. Above it the grid is pinned and peak is exactly `max_programs x (N+1) x element_size`. The observed 13.8x is precisely the program-count ratio 4096/313 = 13.1x plus allocator granularity — **it is the ramp, not a scaling violation.** My test compared a ramp-up point against a saturated point and read the difference as a failure.

Note `test_peak_memory_stays_bounded[10000/100000/1000000]` — which asserts against the constant *ceiling* rather than a ratio — **passed at every M**. That formulation was correct; the ratio formulation was not.

**Fixed:** `test_peak_is_flat_once_the_grid_saturates` now samples only inside the saturated regime (4x, 16x, 64x the saturation point) and asserts the peak ratio is < 1.10 across a 16x path increase, plus the predicted ceiling. A new `test_ramp_up_region_is_bounded_by_the_saturated_ceiling` documents the ramp explicitly — non-decreasing in M, bounded by the same ceiling — so the behaviour is captured rather than merely avoided.

**Consequence for the thesis claim.** The defensible statement is *"peak memory is independent of M above a fixed saturation point of ~131k paths, at `max_programs x (N+1) x element_size`"* — measured at 4.27 MiB on a T4, flat from 131k paths to 50M. Not "flat at all M": below saturation it is smaller and grows. Both halves should appear in any figure; `benchmarks/profile_scaling.py` labels each row with its regime for exactly this reason.

### Two notebook errors (user profiling cell, not repo bugs)

1. `ValueError: portfolio must contain at least one leg` — the cell passed `legs=[]`. The guard is working as designed: with no legs, B and C are identically zero, every exposure is zero, and the kernel would report a flat-zero profile with perfect-looking timings. Silently profiling a no-op is worse than failing.
2. `Import "src.models.portfolio" could not be resolved` — that module does not exist. `SwapLeg` lives in **`src.pricer.options`**.

The `Import "src.csrc.triton_cva_fusion" could not be resolved` warning is a Pylance/editor path issue, not a runtime error — the tests import it fine. Harmless.

Replaced by `benchmarks/profile_scaling.py`, which uses a real 3-leg portfolio, imports from the right module, and marks each row's regime.

### Phase 6 kernel — written, CPU-verified maths, NEVER EXECUTED

`src/csrc/triton_local_vol_cva.py` + `tests/test_phase6_kernel.py` (**42 CPU tests passing, 9 GPU tests never run**).

**Verified on CPU:**

- **Sequential adjoint == `torch.autograd`** to `rel_tol=1e-9` across shapes (1,1), (3,4), (17,9), (64,40), (32,252). The recursion is `a_k = Xbar_k + a_{k+1} * J_k` with `J_k = 1 + (dsigma_k/dX_k)(sqrt(dt) Z_k - sigma_k dt)`.
- **sqrt(N) checkpointing == full storage to EXACTLY `0.00e+00`** — bitwise identical, across N in {4,16,40,100,252} and BLOCK_CK in {1,2,4,8,16,64}. This is the strongest available evidence the checkpointing scheme is right: not "within tolerance", but the same floating-point values.
- **Two traps caught by dedicated tests.** (a) `Xbar_k`, the direct state adjoint from the EE output, must be injected at *every* step, not just the terminal one — terminal-only injection gives a smooth, plausible gradient that is **63.3% wrong**. (b) The skew must be centred on `log(S0)`; an uncentred `tanh(kappa*x)` saturates at `x ~ 4.6`, driving `dsigma/dx` to ~1e-5 and silently reducing the kernel to the constant-vol Phase 5 case while appearing to exercise state dependence.

**Design decisions forced by Triton, worth recording:**

1. **Triton cannot index a register tile by a runtime value.** `tile[:, k]` does not exist. This blocks checkpointing twice over — storing state into slot k, and reading it back. The workaround is masked write / masked reduction, each costing `BLOCK` lane-ops. **That is the real reason sqrt(N) is the right scheme here, not merely a memory optimisation:** at N=252 it makes `BLOCK_CK=16`, so masked access costs 16 ops instead of the 256 a full-trajectory tile would need.
2. **No branching on per-lane data.** Every step guard past the horizon is `tl.where`, never `if`, so warps do not diverge.
3. **Parametric volatility, not a Dupire grid**: `sigma = base + skew*tanh(kappa*(x - x_ref)) + term*t`. Closed-form in both `dsigma/dx` and `dsigma/dtheta`, register-only, no gather, no memory divergence. Connecting it to the calibrated SSVI surface means fitting these parameters to `sigma_LV` — follow-on work, not done.

**Expected first-contact failures on Colab** (in likelihood order): `tl.math.tanh` may not exist in the installed Triton version; the nested loops over a runtime `n_segments` may not unroll acceptably; scalar `tl.load(ptr + k)` with a runtime `k` inside the loop. The CPU/GPU tier split means any Tier 2 failure is *purely* translation — the maths and the checkpointing are settled.

### Market data layer — NEW, CPU-VERIFIED (124 tests passing)

`market_data/fetcher.py` + `market_data/__init__.py` + `tests/test_market_data.py`
(**124 passing, 6 network-tier skipped by default**).

Layered so the mathematics is testable with no network and no optional
libraries: `yfinance` and `pandas_datareader` are imported *lazily*, inside the
functions that use them. The pure layer (`YieldCurve`, `CreditCurve`,
`bootstrap_hazard_rates`, `clean_option_chain`, `black_vega`) imports and runs
in CI regardless. Network tests are marked `network` and skipped unless
`XVA_NETWORK_TESTS=1` — a live-data fixture can never be deterministic.

**CDS hazard bootstrapping (the substantive content).** Piecewise-constant
hazard `h_j` on `(T_{j-1}, T_j]` recovered from par spreads at 1Y/3Y/5Y/10Y by
sequential one-dimensional root finds (Brent). Premium leg includes
accrual-on-default at the interval midpoint; protection leg places default at
the midpoint too. The objective is strictly increasing in `h_j`, so the root is
unique and bracketing is safe.

Verified numerically:

- **Round trip: every input quote reprices to ≤ 1.4e-11 bp.** This is the
  equation the bootstrap claims to solve; everything else is a consequence.
- **Credit triangle recovered in its exact limit.** With zero rates and a flat
  spread curve, `h = S/(1-R)` to 9.0e-08 relative at 25bp.
- **The residual is second order, as derived.** The discrete legs use the
  trapezoid rule, error `O((h·dt)^2)`. Measured convergence ratio on schedule
  refinement: **4.01, 4.00, 4.00** — so the tests assert the *rate*, not a
  tuned threshold. At 2000bp the deviation is 5.79e-04 against a predicted
  `(h·dt)^2/12 = 5.8e-04`, matching to two digits.
- **Exact agreement with `src/xva/cva.py` on the flat overlap** —
  `YieldCurve.flat(r).to_tensor` vs `cva.discount_factors`, and
  `CreditCurve.flat(h).to_tensor` vs `cva.survival_probability`, both to
  **exactly 0.0** (`torch.equal`), as does the `Q(t_{i-1}) - Q(t_i)` marginal
  convention.
- **`black_vega` matches numerical differentiation of the Black-76 price to
  1e-10** across four strike/maturity/vol combinations.

**Two counter-intuitive findings recorded as tests:**

1. **Monotone spreads do NOT imply a monotone forward hazard.** A 200/180/170/165
   bp inverted-but-*flattening* curve bootstraps to hazards
   `[0.0332, 0.0281, 0.0254, 0.0263]` — the far forward hazard ticks *up*. The
   par spread is a survival- and discount-weighted average of the forward
   hazard with weights decaying in `t`, so matching the 10Y average requires
   this. The curve still reprices to 1.4e-11 bp with `Q` strictly decreasing,
   which is what actually has to hold.
   (`test_flattening_inversion_can_lift_the_far_forward_hazard`)
2. **A steeply inverted curve can require a negative forward hazard**, i.e.
   survival *increasing* — arbitrageable, and almost always a stale quote.
   Rejected by default with a message naming the pillar and what the curve
   already implies at zero forward hazard; `allow_negative_forward_hazard=True`
   overrides.

**FRED cannot supply a SOFR term curve.** FRED publishes overnight `SOFR` plus
30/90/180-day backward averages, so the SOFR family reaches only ~0.5Y — there
is no term SOFR swap (OIS) curve on FRED. `fetch_discount_curve` therefore
defaults to splicing the SOFR short end onto Treasury CMT (`DGS1`..`DGS30`),
and the returned `YieldCurve.label` records both approximations: Treasury is
not SOFR (the basis is tens of bp and time-varying), and CMT par yields are
treated as zero rates without a par bootstrap. Both are second order for CVA
but they are approximations, not the real curve.

**Option chain cleaning.** Moneyness is measured against the **forward**
`F = S·exp((r-q)T)`, not the spot: spot moneyness displaces every expiry's
smile by `(r-q)T`, which an SSVI fit absorbs as a spurious maturity-dependent
skew and biases `rho`. Filters drop yfinance's `impliedVolatility == 0.0`
solver failures, one-sided/crossed markets, spreads wider than 50% of mid,
strikes with neither volume nor open interest, and the deep wings beyond 35%
log-moneyness. Output feeds `calibrate_surface` directly (verified end to end).

**Two real bugs found and fixed during testing:**

1. `black_vega` broadcast `(forward, strike)` and `(maturity, volatility)` in
   *separate* `np.broadcast_arrays` calls, so a scalar maturity stayed
   0-dimensional while the mask was 1-D — `IndexError` on any mixed
   scalar/array call, which is the common case. Now broadcasts all four in one
   call.
2. `allow_negative_forward_hazard=True` was **inert**: the bracket expanded
   only upward, so when the required hazard was more negative than the initial
   `-guess` the root was never bracketed and the function raised the very error
   the flag exists to suppress. Now expands in whichever direction the sign at
   each end indicates.

**GAP CLOSED (see the credit term structure section below).** The text that
follows describes why it existed; the injection point now exists in
`src/xva/cva.py` as `credit_curve=`/`survival=`, with per-pillar AAD.

**Original note — the bootstrapped curve could not reach the CVA engine.**
`src/xva/cva.py::_integrate_credit_leg` takes `hazard_rate` as a **scalar** and
builds the flat-intensity curve internally via
`marginal_default_probability(grid, hazard_rate)`. There is no parameter for an
externally supplied survival curve, so a piecewise-constant `Q(t)` has nowhere
to go. `CreditCurve` deliberately exposes `survival_probability`,
`marginal_default_probability` and `to_tensor` in exactly the engine's
conventions (verified bit-identical on the flat overlap), so closing the gap is
a small additive change to `cva.py`: accept an optional
`survival_curve`/`marginal_default` tensor alongside the existing scalar path.
Not done here — it is a different module than the one requested.

Measured cost of the gap: on a declining 20→0 exposure profile over 10Y, the
bootstrapped term structure gives a CVA differing from the 5Y credit-triangle
flat-hazard approximation by more than 1%
(`test_engine_cva_responds_to_the_credit_term_structure`).

### Credit term structure now reaches the CVA engine — CLOSED (61 tests)

`src/xva/cva.py`. The gap recorded in the market-data section is closed:
`_integrate_credit_leg` no longer builds the flat curve internally. Three
credit specifications, exactly one required:

| argument | meaning |
|---|---|
| `hazard_rate=` | flat intensity — the pre-existing path, **unchanged** |
| `credit_curve=` | `PiecewiseHazard`, or a duck-typed `market_data` `CreditCurve` |
| `survival=` | an explicit `Q(t)` tensor on the grid |

`hazard_rate` is still the third positional argument (it merely became
optional), so every existing call site is untouched — confirmed by the 454
pre-existing tests still passing.

**AAD was the substance, not the plumbing.** Reading NumPy survival values out
of a bootstrapped curve yields a *constant*: the backward pass then succeeds
with an all-zero gradient and every credit sensitivity is silently gone. So
`PiecewiseHazard` keeps the hazard vector in torch and rebuilds `Q` from it:

    H(t) = sum_j h_j * clamp(min(t, T_j) - T_{j-1}, min=0)

which is *linear* in `h`, so one backward pass yields every
`dCVA/dh_j` — the bucketed credit deltas a desk hedges with, not one lumped
sensitivity to a flat intensity.

Verified:

- **Bucket deltas match central finite differences to 2e-11 relative**, per
  pillar (`cva_credit_bucket_deltas`).
- **Torch piecewise `Q` == NumPy `CreditCurve.survival_probability`** to
  1.1e-16.
- **A one-pillar curve is bit-identical to the flat path** (`torch.equal`) for
  both survival and the marginal-default convention. Exact, not close — both
  compute `exp(-h t)`, so any difference would mean the overlap arithmetic is
  not reducing.
- Flat extrapolation past the last pillar holds the final hazard, verified by
  recovering it from `Q(15)`/`Q(20)`. Dropping it instead would make `Q` stop
  decaying and understate CVA on any trade maturing beyond the longest quote —
  a silent, one-directional error.
- Term structure vs the 5Y credit triangle differs by >1% on a declining
  exposure, so the curve is not cosmetic.

**One constraint hit, worth recording:** `aad_greeks` reduces every gradient
with `float(grad.detach())` and is therefore **scalar-only** — it cannot carry
a per-pillar delta vector. Rather than change `greeks.py` (and the tests
resting on it), `make_cva_valuation_fn(credit_curve=...)` exposes a scalar
`"hazard_shift"`: a parallel shift of the whole curve, which is a quantity
desks quote anyway, so it composes with the existing greeks API unchanged. The
bucketed vector lives in `cva_credit_bucket_deltas`, where it can be returned
as a tensor. `compute_xva` gained `credit_curve=`/`own_credit_curve=`, and
`XVAResult.hazard_rate` is now `Optional[float]` — `None` under a curve,
because reporting a summary intensity would invite it being quoted as if it
had been the input.

`cva.py` deliberately does **not** import `market_data`: the dependency runs
data-layer -> engine, and reversing it would make the engine depend on its own
data layer. `CreditCurve` is duck-typed on `(pillar_times, hazard_rates)`.

### SSVI -> Phase 6 kernel bridge — WRITTEN, and it is a lossy projection

`src/models/vol_surface.py`: `LocalVolFit`, `fit_local_vol_params`,
`local_vol_sampling_grid`, `evaluate_parametric_local_vol`.

Fits the kernel's four-parameter surface
`sigma(t,x) = base + skew*tanh(kappa*(x - x_ref)) + term*t` to the Dupire
`sigma_LV` implied by a calibrated `SSVISurface`, by weighted least squares in
volatility units. `to_local_vol_params()` hands the result straight to the
Phase 6 kernel (lazy import, so `src.models` keeps no hard dependency on
`src.csrc`).

**This is a projection, not an identity, and the loss is material.** Measured
against an SSVI surface with `rho=-0.35, eta=1.2, gamma=0.45`, relative RMSE
against sampling width:

| cone half-width | rel. RMSE | R^2 | target sigma range |
|---|---|---|---|
| 0.5 sigma | 3.47% | 0.879 | 0.167-0.237 |
| 1.0 sigma | 6.81% | 0.831 | 0.167-0.286 |
| 1.5 sigma | 10.66% | 0.729 | 0.167-0.333 |
| 2.0 sigma | 13.59% | 0.646 | 0.167-0.374 |
| 3.0 sigma | 15.99% | 0.578 | 0.167-0.447 |

The tanh saturates, so it cannot follow a steep Dupire wing: at 3 sigma the
target reaches 0.447 while the fitted surface tops out near 0.287. **Phase 6
gradients are therefore with respect to the FITTED surface, not the calibrated
SSVI one**, and `LocalVolFit.summary()` says so whenever relative RMSE exceeds
2%. Quoting kernel output without those diagnostics would hide how much of the
calibrated surface was discarded on the way in. A tighter coupling needs the
Chebyshev extension already noted for the kernel.

Identifiability is checked first: against a target that *is* in the tanh family
the fitter recovers `(base, skew, kappa, term)` to **8e-6**. Without that check
a residual against a real surface could be the projection or the optimiser with
no way to tell.

Design points: the sampling grid is a **cone** (half-width growing as
`sigma*sqrt(t)`), not a box, so points are not spent on states unreachable at
short maturities; quotes are weighted by the lognormal transition density
(unweighted fitting moved `base` from 0.218 to 0.263 and worsened relative RMSE
from 15.9% to 21.1%); `base` and `kappa` are fitted through `softplus` because
an unconstrained optimiser reaches a negative `kappa` — an observationally
equivalent surface with flipped skew — which then fails `LocalVolParams`
validation for no apparent reason; `x_ref` is **pinned** to `log(S0)`, never
fitted, because an uncentred tanh saturates and silently reduces Phase 6 to the
constant-vol case.

**Bug found and fixed while wiring it:** the target evaluation was wrapped in
`torch.no_grad()`. Dupire local variance is *defined* by autodiff of the
total-variance surface (`d_T w / g`), so suppressing grad removed the
derivative the target is made of, raising `element 0 of tensors does not
require grad`. Fixed with explicit `torch.enable_grad()` — needed **twice**,
once for the target and once around the optimisation loop, since an outer
`no_grad` also leaves the loss without a `grad_fn`.

### Figures — `benchmarks/plot_results.py`

Three publication-ready figures, PNG + PDF + a companion CSV of the plotted
values (a figure alone is not accessible).

1. **Execution time vs M** — log-log, one line per backend.
2. **Peak VRAM vs M** — log-log, with the device capacity as a reference line,
   so the OOM cliff is visibly the intersection of an `O(M*N)` line with a
   fixed ceiling rather than lines stopping for no reason.
3. **Exposure profiles** — EE and PFE with and without a CSA. Computed **live
   on CPU** from `src.xva`, so it needs no GPU and no benchmark file: it is a
   property of the model, not the hardware.

Figures 1 and 2 parse the Markdown `bench_all_phases.py --markdown` writes.
**`results.md` does not exist yet**, so they are skipped with the command that
produces them. No placeholder data is ever plotted — a benchmark figure showing
invented numbers is worse than a missing figure. An OOM cell parses to `None`,
never `0`: plotting zero would put a point at exactly the place the finding
lives and make a backend that died look infinitely fast.

Palette chosen by the project's data-viz validator, not by taste. The obvious
fourth series colour, yellow, **fails** the all-pairs normal-vision gate
against orange (Delta E 13.7, below the floor of 15) — readers with ordinary
colour vision cannot reliably separate them. Blue / orange / aqua / violet
passes both adjacent and all-pairs gates (worst all-pairs normal Delta E 16.3,
CVD 9.2). Aqua sits at 2.82:1 on white, below the 3:1 bar, so every series also
carries a direct end-of-line label — the documented relief, which doubles as
the redundant encoding that keeps the figures readable in greyscale print.
Colour follows the entity, so a backend keeps its hue across both figures, with
line style and marker varying too. Figure 3 encodes the metric in colour and
the CSA in line style rather than giving four series four hues, because the
comparison a reader needs is within each metric. **No dual-axis figure** — time
and memory get one axis each.

Layout defects were found by rendering and looking, not by reading the code:
end-of-line labels clipped at the right spine (fixed with reserved x headroom),
the legend colliding with the capacity annotation (moved below the axes), the
OOM marker landing on top of the series label (folded into one text object),
and a "PFE peak" annotation that was both clipped and redundant — the profile
rises monotonically to maturity, so the peak is always the last point (replaced
with terminal-value labels).

Measured on the default single-swap netting set (20,000 paths, 5Y, threshold 5,
MTA 1, MPOR 10bd): the CSA removes **83% of aggregate EE and 88% of peak PFE**,
with `EE_collat <= EE_uncollat` pointwise as Phase 2 asserts.

### Figures — GENERATED from real Colab measurements (bench_all_phases.py ran)

`results.md` now holds the actual `bench_all_phases.py` output measured on a
Tesla T4 (torch 2.11.0+cu128, CUDA 12.8, Triton 3.6.0) — saved verbatim from
that run rather than regenerated, since inventing a "reproduction" would defeat
the point. `benchmarks/plot_results.py` ran against it and `figures/` now holds
all three deliverables in PNG (300 DPI) + PDF, plus `figure_data.csv`.

**Bug found and fixed while running it for the first time:**
`BACKEND_STYLE` was keyed on placeholder names (`"Phase 3 Triton"`) invented
before any real report existed. The actual `bench_all_phases.py` headers carry
a parenthetical detail (`"Phase 3 (Triton + dW)"`), so the exact-match lookup
missed on every non-baseline backend and all three silently fell back to one
generic grey style with the same marker — invisible in code review, obvious
the instant the figure was rendered. Fixed by matching on the stable `"Phase
N"` prefix (`BACKEND_STYLE_BY_PREFIX` + `backend_style()`), which survives the
parenthetical wording changing again.

**Two more defects found by rendering and looking, not by reading code:**
the "device capacity" annotation collided with per-series end labels at
upper-right (moved to upper-left, which is reliably empty since every series
starts small); and two end labels (PyTorch baseline vs Phase 3 at M=1e6: 4.71
vs 3.78 GiB) sat close enough in log-space to overlap text. Fixed with
`_place_end_labels` — a collision-avoidance pass using the real
`transData`-to-points mapping, run *after* `set_xscale`/`set_ylim` are
finalised (a before-limits pass gives the wrong pixel positions). A residual
case — Phase 4's own label sitting almost exactly on the capacity dash
(14.15 vs 14.6 GiB, a data coincidence no repositioning avoids) — is handled
with a white halo (`bbox`) behind the text rather than movement.

**Measured headline numbers, now on the actual figures:** Phase 5 is 7.97x
faster than the PyTorch baseline at 1M paths and its peak VRAM is flat at
4.0–4.3 MiB from 100K through 50M paths (a 500x span) while every other
backend is OOM by 5–10M. The baseline's OOM boundary bisects to 2,750,000
paths. The exposure figure (computed live, no GPU needed) shows the CSA
removing 83% of aggregate EE and 88% of peak PFE on the default single-swap
netting set.

### plot_results.py CUDA fix

`compute_exposure_curves` called `.numpy()` on tensors from a `GBMSimulator`
that defaults to CUDA when available, so the exposure-figure tests failed with
`TypeError: can't convert cuda:0 device type tensor to numpy` the first time
they ran on a GPU instance rather than this CPU-only dev machine (515/515 had
passed locally, masking it entirely). Fixed at the root: the simulator is now
pinned to `device=torch.device("cpu")` explicitly, since this figure
illustrates a model property and must render identically with or without a
GPU present. `.cpu()` was also added before every `.numpy()` call as
defense-in-depth, in case a future change reintroduces a CUDA tensor upstream.

### Chebyshev kernel — FIRST COMPILE on T4, two bugs found and fixed

The GPU tier ran for the first time on Colab (Tesla T4, Triton 3.6.0) and
surfaced exactly the class of bug the two-tier split exists to isolate. Both
are fixed; local suite is now **569 passing**.

**Bug 1 — `for k in range(...)` inside `@triton.jit` does NOT unroll.**
The kernel assumed a `tl.constexpr` bound makes the loop index a Python
`int` at trace time. It does not: Triton lowers `range` to a *runtime* loop
and `k` becomes a `tl.tensor`. Only `tl.static_range` unrolls. Two failures
followed from the one misconception:

- `float(k)` -> `TypeError: float() argument must be a string or a real
  number, not 'tensor'` (the reported compile error);
- `acc_coeff[k]`, a Python list of per-coefficient accumulators indexed by
  the loop variable, would have failed identically the moment the forward
  compiled. The module docstring had flagged this pattern as the top risk but
  gave the **wrong reason** ("unrolled into a separate SSA value at trace
  time"); that claim is now corrected in place rather than left standing.

Fixed without `tl.static_range` — unverifiable from the dev machine, and this
project already lost a round trip to assuming `tl.math.tanh` existed:

1. **Split the device function.** Three of four call sites discard the
   derivative, so `_chebyshev_eval` (value only) now serves them. It uses `k`
   for nothing but `tl.load(coeff_ptr + k)`, which is legal with a runtime
   offset. Side benefit: the forward had been running the entire `U`
   recurrence and throwing it away at all 252 steps.
2. **Folded `k` into the coefficients host-side.** Since
   `d/du sum_k c_k T_k(u) = sum_k (k*c_k) U_{k-1}(u)`, the host passes
   `dc_k = k*c_k` and the kernel does no `k` arithmetic at all. Verified on
   CPU against `chebyshev_basis_derivative` across degrees 0-14.
3. **Replaced the accumulator list with a masked `(BLOCK_M, BLOCK_DEG)`
   tile**, accumulated vectorised and written with one masked `tl.store`.
   This is precisely the pattern the proven tanh kernel already uses for its
   `checkpoints`/`replay` tiles, and for the same reason. `BLOCK_DEG` is the
   next power of two at or above `degree+1` (`tl.arange` requires one), with
   the tail masked; padding columns verified to stay exactly zero.
   `select_chebyshev_local_vol_blocks` now budgets for the two extra tiles,
   so `BLOCK_M` drops 64 -> 32 at float64/degree-8 rather than silently
   costing occupancy.

All three rewritten pieces were checked by a **line-for-line Python mirror**
against the verified basis functions before being trusted in code that cannot
be compiled locally: value-only exact (0.0), derivative and accumulator at
float64 recurrence noise, padding columns exactly zero.

**Bug 2 — a pytest version trap of my own making.** A class-scoped fixture
defined as an instance method is deprecated in pytest 10, and the workaround
I used (`@classmethod` stacked over `@pytest.fixture`) is silently
version-dependent: pytest 9.1.1 (dev machine) unwraps the descriptor to find
the fixture marker, pytest 8.4.2 (Colab) does not and reports
`fixture 'local_vol' not found`. So a warning that only the *newer* pytest
emits was silenced in a way that broke the *older* one — invisible locally.
Replaced with a plain module-level fixture: no descriptor, same caching,
identical behaviour on every pytest version.

**New regression guards (`TestRuntimeLoopVariableGuards`).** Neither bug is
catchable by running anything without a GPU, so they are now *static source
checks* over the AST of every `@triton.jit` function: no `float(...)` call,
and no Python container subscripted by a `for`-loop target. Same rationale
and precedent as `tests/test_phase4.py::TestGlobalPointerAddressing`, which
guards the int64 pointer fix the same way. A meta-test asserts the guards
actually find the kernels, so an import rename cannot silently empty them.

**Still unverified:** the corrected kernel has not been compiled either. The
fixes remove every construct identifiable as version- or semantics-dependent,
but a second first-contact failure remains possible.

### Chebyshev local-vol kernel — fixes the tanh kernel's wing saturation

`src/models/vol_surface.py` (`chebyshev_basis`, `chebyshev_basis_derivative`,
`evaluate_chebyshev_local_vol`, `LocalVolFit` extended with `basis=`,
`fit_local_vol_params(..., basis="chebyshev", degree=K)`) +
`src/csrc/triton_chebyshev_local_vol_cva.py` (NEW) +
`tests/test_chebyshev_local_vol.py` (**42 CPU tests passing, 2 GPU tests
NEVER RUN**).

**The problem, quantified.** The tanh bridge's own diagnostics showed it:
against an SSVI surface (`rho=-0.35, eta=1.2, gamma=0.45`) at a 3-sigma
sampling width, relative RMSE was 15.99% and R^2 only 0.578 — tanh is bounded
by construction and cannot follow a Dupire surface that keeps rising into the
wing.

**The fix and its measured effect.** Replacing the spatial term with a
degree-K Chebyshev expansion removes the saturation ceiling entirely.
Verified on the identical surface, identical sampling width:

| degree K | relative RMSE | R^2 |
|---|---|---|
| tanh (baseline) | 15.99% | 0.578 |
| 4 | 9.42% | 0.854 |
| 8 | **7.71%** | **0.902** |
| 12 | 7.14% | 0.916 |
| 16 | 6.90% | 0.922 |
| 24 | 6.73% | 0.925 |

Error decreases monotonically with degree (asserted directly in
`TestWingSaturationFix::test_error_decreases_monotonically_with_degree` — a
least-squares fit's optimum at degree K is a superset of degree K-1's, so this
must hold exactly, and a violation would mean the solver is buggy, not that
the model is a bad fit).

**Solved in closed form, not by gradient descent — a real correction made
during verification.** The Chebyshev sum is *linear* in its coefficients, so
it has a unique global optimum reachable by one weighted least-squares solve.
An earlier version used Adam (matching the tanh fit's pattern for
consistency); verification caught it under-converging against the real SSVI
target — R^2 plateaued at 0.916 at degree 12 after 1500 iterations, with the
fitted range overshooting the target range at both ends, the signature of an
under-converged fit rather than a representational limit. Recognising the
model is linear and solving it exactly removed the failure mode rather than
papering over it with more iterations: identifiability against a target
constructed *as* a Chebyshev sum now recovers all coefficients to **1.9e-16**
(machine precision), and the fit is ~1000x faster (0.004s vs ~1-2s).

**A finding recorded but not chased:** letting each coefficient vary linearly
in time (`c_k(t) = a_k + b_k*t`, still linear in all parameters, still one LS
solve) measured R^2=0.951 at degree 8 vs 0.902 for the flat-term model — a
real further improvement. Not implemented: it doubles the kernel's per-step
parameter-gradient bookkeeping for a second, unverified change, and the
flat-term result already closes most of the gap. Noted here as the next
concrete step if degree-8 Chebyshev alone is not enough.

**Kernel status — CPU-verified, GPU tier NEVER RUN**, following this
project's established two-tier pattern for every Triton kernel it has
written:

- Forward, full-storage adjoint, and sqrt(N)-checkpointed adjoint are all
  plain-torch / hand-derived-Python, matching the tanh kernel's own reference
  structure exactly.
- **Full-storage adjoint vs `torch.autograd`: exact to float64 machine
  epsilon** (2.22e-16) on every one of `(s0, drift, term, c_0..c_4)`.
- **Checkpointed adjoint vs full-storage adjoint: `torch.equal` — exactly
  0.0 — across checkpoint strides {4, 8, 16, 40}.**
- The Triton device-function logic (`_chebyshev_eval_and_deriv`, evaluating
  the Chebyshev sum and its derivative via the standard T_k/U_k recurrences)
  was translated to a line-for-line Python mirror and checked against the
  verified `chebyshev_basis`/`chebyshev_basis_derivative` — bit-identical
  across degrees 0-14 — *before* being trusted inside the untestable `.jit`
  kernel.
- **No Triton install or CUDA device was available while writing the kernel
  file**, so `_fused_chebyshev_local_vol_forward_kernel` /
  `_..._backward_kernel` have never been compiled. They mirror
  `triton_local_vol_cva.py`'s two kernels primitive-for-primitive (same
  `tl.where`/`tl.exp`/`tl.randn`/`tl.load`/`tl.maximum`/`tl.sum`, same
  sqrt(N)-checkpointing structure), so the risk is concentrated in one
  genuinely new pattern flagged explicitly in the module docstring: a Python
  list of per-coefficient gradient accumulators built over a
  `range(DEGREE + 1)` loop where `DEGREE` is a `tl.constexpr`. That is the
  first thing to inspect on a Colab compile failure.
- 2 GPU-tier tests (`TestChebyshevKernelGPU`, `@requires_triton`-gated,
  matching every prior phase's convention) are written and will run the
  moment Triton is available; they compare the compiled kernel against the
  CPU reference and against finite differences.

### Phase 6 benchmark — written, NEVER RUN

`benchmarks/bench_phase6.py`. Compares the Phase 6 Triton kernel against a
PyTorch autograd baseline (`reference_local_vol_ee`, a sequential Python time
loop) on forward time, backward time and peak VRAM, sweeping M in {1e4, 1e5,
1e6, 5e6} at N=252.

**The request asked for gradients w.r.t. SSVI surface parameters. That is not
what this measures, because it does not exist.** The kernel evaluates the
parametric tanh surface and differentiates `S0`, `mu`, `base`, `skew`. The SSVI
surface in `src/models/vol_surface.py` is a separate object with separate
parameters (`rho`, `eta`, `gamma`, and the ATM variance term structure) and
**there is no code path from those to the kernel**. Benchmarking "SSVI
gradients" would mean timing something unimplemented. The benchmark measures
the honest comparison instead — identical surface, identical estimator, on both
backends — and says so prominently in its own report.

Design points worth keeping:

- **Forward and backward are guarded and recorded independently.** A single
  guard sized on forward+backward would blank the forward column at exactly
  the path counts where the finding lives. The interesting row is
  "baseline forward completes, baseline backward cannot be attempted" — the
  forward is not the problem, the `O(M*N)` tape is.
- **The backward is timed with CUDA events around `.backward()` alone**, with
  the forward completed and synchronised first. Reporting `total - forward`
  would fold the forward's tape-construction cost into the backward number,
  which is the very quantity under study.
- **The two backends draw independent random streams** (Philox in-kernel vs
  `torch.randn`), so they agree only to Monte-Carlo error. An agreement check
  runs both forwards at the smallest M and reports the deviation against the
  `1/sqrt(M)` scale, rather than assuming the comparison is like-for-like.
- **Strike 95, not 100** — at `strike == spot` the t=0 exposure is exactly zero,
  on the `max(V,0)` kink. No reason to time a degenerate point.

Analytic predictions on a 14.6 GiB T4, N=252, fp32 (from `predict_peak_bytes`;
the baseline tape term is deliberately under-estimated so the guard errs toward
attempting a run):

| M | baseline fwd | baseline fwd+bwd | kernel fwd+bwd |
|---|---|---|---|
| 10,000 | 38.6 MiB | 115.5 MiB | 0.3 MiB |
| 100,000 | 385.7 MiB | 1.13 GiB | 3.1 MiB |
| 1,000,000 | 3.77 GiB | 11.28 GiB | 4.0 MiB |
| 5,000,000 | 18.83 GiB | 56.38 GiB | 4.0 MiB |

So expect the baseline's backward to die around M=1M while its forward still
runs, and both its stages to be gone at 5M — with the kernel flat at 4 MiB
throughout, bounded by `--max-programs`, not by M.

Verified on CPU (the kernel itself still needs a GPU): module imports, CLI
parses, every reporting branch renders from fabricated results, and the
baseline wiring produces finite non-zero gradients for all four parameters
(`spot 2.36e+00`, `drift 1.38e+02`, `base 9.54e+01`, `skew -3.92e+00` at
M=4000, N=32, float64, with `EE[0]=4.85` safely off the kink).

### FIXED: the grid-uniformity tolerance was mis-formulated

Commit `d4998d2` loosened the uniform-grid check in `fused_expected_exposure` from `1e-6 * dt` to `1e-4 * dt` with no recorded reason. Investigated:

**The loosening was a genuine bug fix — but `1e-4` was a patch on a mis-formulated test.**

`torch.linspace` rounding is `O(eps * T)` and **independent of N**, because the time values are `O(T)`. Testing that deviation against `dt = T/N` injects a spurious factor of N, so the ratio grows linearly with N. Measured on `linspace(0, 1, N+1)`:

| N | float32 dev/dt | float64 dev/dt |
|---|---|---|
| 252 | 1.1e-05 | 2.4e-14 |
| 1000 | 7.3e-05 | 1.1e-13 |
| 2520 | 9.4e-05 | 2.8e-13 |
| 10000 | 4.3e-04 | 1.0e-12 |

So `1e-6 * dt` rejects a perfectly valid float32 grid from about **N=100** — the original bound was broken, and the loosening was necessary. But `1e-4 * dt` only buys headroom to about **N=2700**; at N=2520 the margin was already down to 6%.

**Fix applied:** bound the deviation against the *horizon*, not the step — `max|s_i - s_0| <= 64 * eps * T`. This is N-independent and dtype-aware. Validated across N in {2 … 50,000}, T in {0.25, 1, 10, 30}, both dtypes: every legitimate grid passes, and a real 1e-4 perturbation is still rejected by orders of magnitude.

The check now lives in one place, `src/xva/exposure.py::_grid_step` (public alias `validate_uniform_grid`), used by `triton_cva_fusion.py` and `triton_local_vol_cva.py`. Previously there were **three** inline copies with three different tolerances (`1e-4`, `1e-4`, `1e-9`) — the `1e-9` one in `exposure.py` would have failed on float32 too. 60 regression tests added in `tests/test_phase2.py::TestGridUniformityTolerance`, including one that asserts *why* the formulation changed.

## Known bugs / technical blockers

1. **RESOLVED 2026-08-21 — the Colab bypass worked; the GPU tier now runs and passes there.** Kept for the record. Local GPU remains unusable: `Get-PnpDevice` on the RTX 3050 Laptop GPU reports `CM_PROB_FAILED_POST_START`; `nvidia-smi` fails with a permissions error; `torch.cuda.device_count() == 0` even though PyTorch 2.4.0+cu121 is CUDA-built (`torch.backends.cuda.is_built() == True`) and `nvcuda.dll` loads fine. This is an NVIDIA driver/device fault, not a PyTorch install issue.

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

**Priority 1: finish collecting benchmark numbers.**

`bench_all_phases.py` HAS now run on a Colab Tesla T4 (torch 2.11.0+cu128, CUDA
12.8, Triton 3.6.0, Python 3.13.15). Measured, not predicted:

- **Phase 5 is 8.16x faster than the PyTorch baseline at 1M paths.**
- **Phase 5 peak VRAM is flat at 4.3 MiB from 1M through 50M paths** — the
  O(1)-in-M claim, demonstrated over a 50x range.
- **The PyTorch baseline's OOM boundary bisects to 2,750,000 paths** (it fails
  at 2,781,250). The analytic model had predicted "between 2M and 3M" — a
  genuine prediction, confirmed.
- At M=5M: baseline and Phase 3 OOM; Phase 4 survives at 14.15 GiB; Phase 5 at
  4.3 MiB. At M=50M **only Phase 5 survives** (821 ms, 4.3 MiB).

Still outstanding: `profile_scaling.py`, the per-phase detail runs, and
`bench_phase6.py` (never executed).

1. `python benchmarks/bench_all_phases.py --find-oom --markdown results.md` — the headline table. PyTorch baseline vs Phase 3/4/5 across M in {1e5, 1e6, 5e6, 1e7, 5e7}, with time, peak VRAM, speedup, a survival matrix, and a bisected OOM threshold. The analytic model puts the baseline's OOM between **2M and 3M paths** on a 14.7 GiB T4; expect Phase 4 to OOM around 10M and Phase 5 alone to survive at 50M.
2. `python benchmarks/profile_scaling.py` — the O(1)-memory evidence, with each row labelled ramp-up or saturated.
3. `bench_phase3.py`, `bench_phase4.py`, `bench_phase5.py` for the per-phase detail. On a 16 GiB T4 use `--paths 1000000 5000000 10000000` for Phase 4; 20M predict-OOMs for both designs there.

Record the device name, driver, Triton and PyTorch versions alongside every number.

**Priority 2: re-run the Phase 6 kernel tests after the ATM-kink fix.**

`python -m pytest tests/test_phase6_kernel.py -v`. Two GPU-tier defects have
been found and fixed since the kernel was written:

1. `tl.math.tanh` does not exist in Triton 3.6.0 — replaced with an exp-only
   device function; `tl.log` moved host-side. Both were the only primitives in
   the kernel not already proven by Phases 3-5.
2. The gradient test sampled `strike == spot`, putting `V[0]` exactly on the
   `max(V,0)` kink where no derivative exists, so AAD (subgradient 0) and
   central FD (half the jump) disagreed by a fixed absolute amount. The kernel
   was never wrong. Test moved to `strike=95`; `TestAtTheMoneyKink` now pins
   the behaviour down.

**Priority 3: run `benchmarks/bench_phase6.py`.** Start with
`--paths 10000 100000` to confirm the agreement check is clean before spending
time on the large sizes.

**Priority 4 (open design gap): connect SSVI to the kernel.** Nothing currently
links the calibrated `SSVISurface` to the local-vol kernel. Closing it means
either fitting `(base, skew, kappa)` to the Dupire `sigma_LV` implied by a
calibrated surface, or replacing the tanh with a Chebyshev expansion in
`(t, log S)` whose coefficients depend differentiably on `rho`, `eta`, `gamma`.
Until then, no result in this project is a gradient with respect to an SSVI
parameter, and none should be described as one.

**Priority 3 (optional):** connect the Phase 6 parametric surface to the calibrated SSVI surface, and add a GPU tier to `tests/test_phase6.py`, which currently has none.

Deferred Phase 2 nice-to-haves (not blocking Phase 3):

- `data/` — real market data fetchers (yfinance spot/vol, FRED discount curve) to replace hardcoded `S0=100, σ=0.2`.
- Bootstrap a piecewise-constant hazard curve from a CDS term structure. `_integrate_credit_leg` already accepts an explicit `curve` argument, so this is an input change, not an algorithm change.
- Initial margin (SIMM-style) on top of the existing variation margin.
