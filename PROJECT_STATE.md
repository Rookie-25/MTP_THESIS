# PROJECT_STATE.md

**Project:** GPU-Native Differentiable AAD Monte-Carlo XVA & Greeks Engine with Neural-SDE / Rough-Volatility Calibration
**Last updated:** 2026-08-20 (Phase 5 fused reduction)

---

## Current Phase & Module

**Phase 1 (Month 1) — Baseline CPU/GPU Autodiff Simulator: COMPLETE.**

**Phase 2 (Month 2) — Exposure Profiling, Collateral & CVA/DVA: COMPLETE.**

**Phase 3 (Month 3) — Custom Triton kernels: IN PROGRESS. Code written; GPU validation pending on Colab.**

All Phase 2 deliverables are implemented, tested and benchmarked: exposure profiles (EE/ENE/PFE/EPE), collateralised exposure under a CSA (threshold, MTA, MPOR), unilateral CVA/DVA, end-to-end AAD sensitivities through the full chain, a manual inspection sandbox, and an empirical O(1)-vs-O(n) scaling benchmark.

**Phase 4 (Month 3-4) — In-kernel Philox RNG + rematerialisation: IN PROGRESS. Code written; GPU validation pending on Colab.**

**Phase 5 (Month 4) — Fused payoff/exposure reduction, O(N) memory: IN PROGRESS. Code written; GPU validation pending on Colab.**

Phase 3 has a fused Triton GBM kernel with a hand-derived adjoint. Phase 4 removes the caller-supplied `dW` matrix entirely by generating increments in-kernel from a counter-based (Philox) RNG, and rematerialises them in the backward pass instead of storing them.

**For both Phase 3 and Phase 4: the adjoint mathematics is verified on CPU today; the kernels themselves are unverified and must be run on Colab before any claim is made about them.** Suite status: **175 passed, 65 skipped** — the 47 skips are the GPU tier, which cannot execute locally (no Triton wheel on Windows, no working CUDA device).

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
│   ├── bench_phase2.py             # AAD O(1) vs FD O(n) scaling sweep, ASCII table + optional CSV
│   ├── bench_phase3.py             # fused-vs-PyTorch time + peak VRAM sweep; torch.cuda.Event
│   │                               # timing, max_memory_allocated, OOM captured as a result
│   ├── bench_phase5.py             # Phase 4 (materialised paths) vs Phase 5 (fused);
│   │                               # M to 50M, pre-flight VRAM guard, O(N) memory evidence
│   └── bench_phase4.py             # Phase 3 (dW in HBM) vs Phase 4 (in-kernel Philox);
│                                   # M up to 20M, reports the OOM crossover + ceiling analysis
└── tests/
    ├── conftest.py                 # repo-root sys.path fallback, seeds torch, `device` fixture (cpu/cuda param)
    ├── test_phase1.py              # 15 tests, all passing
    ├── test_phase2.py              # 59 tests, all passing (33 core + 26 collateral/CSA)
    ├── test_phase3.py              # 54 tests: 27 CPU-tier passing, 27 GPU-tier skipped locally
    ├── test_phase4.py              # 57 tests: 37 CPU-tier passing, 20 GPU-tier skipped locally
    └── test_phase5.py              # 54 tests: 37 CPU-tier passing, 18 GPU-tier skipped locally
```

**Full suite: 175 passed, 65 skipped** (`python -m pytest tests/ -q`, ~32s on CPU).

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

**Validate the Phase 3 AND Phase 4 kernels on Colab. Nothing in either phase can be trusted until this is done.**

1. Bootstrap Colab (see below), then `python -m pytest tests/test_phase3.py -v`. The 27 skipped GPU tests must run and pass. Expect to iterate — the kernels have never executed. Likely failure points, in order: `tl.cumsum` axis semantics on the 2-D tile, the `range(0, n_steps, BLOCK_N)` dynamic loop bound, `tl.store` of a 0-d reduction into the partial buffers, float64 `tl.exp`.
2. Then `python -m pytest tests/test_phase4.py -v` (20 GPU tests), then `tests/test_phase5.py -v` (18 GPU tests). Additional Phase 4 failure points: the exact `tl.randn(seed, offset)` signature and whether it accepts a 2-D offset tile; whether `tl.randn` returns float32 (the `.to(DTYPE)` cast assumes it does); and whether `seed + pid` is accepted as a runtime scalar key.
3. **Run the moment tests before the gradient tests.** If the increments are not really `N(0, Δt)`, every gradient test is meaningless. `test_log_return_variance_matches_theory` is the sharpest single check — it would catch a missing `√Δt`.
4. **If Vega fails FD while Delta and Rho pass, the bug is rematerialisation**, not the adjoint (the CPU tier already proved the formulas). Check that `BLOCK_M` and `seed` reaching the backward are identical to the forward's.
5. Only once tests pass, run `bench_phase3.py`, `bench_phase4.py`, then `bench_phase5.py` and record the numbers. Phase 5 likely failure points, additional to Phase 4's: `tl.num_programs`, the grid-stride `range(pid, n_blocks, n_programs)` with runtime bounds, `tl.maximum` on a 2-D tile, and whether a `BLOCK_T`-wide register accumulator survives the loop without spilling. **Do not put any speedup or memory figure in the write-up before this.** Note `bench_phase4.py` defaults to M up to 20M, which needs ~19 GiB just for the output — on a T4/L4 expect OOM at 20M and possibly 10M; that is reported, not hidden. Use `--paths 1000000 5000000 10000000` on a smaller card.
6. Re-run the full suite on GPU (`python -m pytest tests/ -q`) to confirm nothing regressed.

Colab bootstrap (prerequisite for all of the above):

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
