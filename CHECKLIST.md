Here is your master project checklist. Keep this in your repository as **`CHECKLIST.md`** alongside `PROJECT_STATE.md`.

Whenever you finish a session, simply check off the items (`[x]`) and verify the specific acceptance criteria before moving to the next phase.

---

### **Master Project Execution & Verification Checklist**

#### **Phase 1: Baseline Simulator & Autodiff Validation**

* [x] **Setup Repository Core**: Package layout, `pyproject.toml`, pytest configuration.
* [x] **Path Generation**: PyTorch exact-scheme GBM simulator (`src/models/gbm.py`).
* [x] **Pricing Engine**: Vectorized European Call & Portfolio Swap payoffs (`src/pricer/options.py`).
* [x] **AAD Sensitivity Engine**: Reverse-mode autograd Greeks (`aad_greeks`) vs. Finite Difference (`src/pricer/greeks.py`).
* [x] **Verification Goal**: Run `python -m pytest tests/test_phase1.py -v`.
* **Pass Condition**: 15/15 tests pass with zero warnings; AAD Delta and Vega match Finite Difference within $10^{-3}$ tolerance on CPU/CUDA.



---

#### **Phase 2: Exposure Profiling & Counterparty Credit Risk (CVA/DVA)**

* [x] **Exposure Engine**: Build `compute_exposure_profiles` in `src/xva/exposure.py` for $EE(t)$ (Expected Exposure) and $PFE(t)$ (Potential Future Exposure).
* [x] **CVA/DVA Pricing**: Build `compute_unilateral_cva` in `src/xva/cva.py` with flat hazard rate integration.
* [x] **CVA Sensitivities**: Extend `aad_greeks` to differentiate CVA w.r.t. $S_0$, $\sigma$, and hazard rate $\lambda$ in a single backward pass.
* [x] **Verification Goal**: Run `python -m pytest tests/test_phase2.py -v`.
* **Pass Condition**: Exposure profile tensors maintain positive floors; CVA matches single-forward analytical approximations; CVA AAD gradients match bump-and-revalue gradients within $10^{-3}$ tolerance.



---

#### **Phase 3: The GPU Deep Dive (Custom CUDA / Triton Kernels)**

* [x] **Cloud GPU Migration**: Set up Google Colab / Kaggle environment with active T4 GPU.
* [x] **Custom Path Generator**: Write C++/CUDA extension (`src/csrc/euler_maruyama.cu`) or Triton kernel for parallel SDE path generation.
* [x] **PyTorch Binding**: Bind custom CUDA kernel to PyTorch autograd graph using `torch.utils.cpp_extension`.
* [x] **Benchmarking Harness**: Write execution timing scripts in `benchmarks/` comparing CPU vs. PyTorch GPU vs. Custom CUDA.
* [x] **Verification Goal**: Run `python benchmarks/bench_phase3.py`.
* **Pass Condition**: Custom CUDA kernel executes Monte Carlo trajectory generation at least **10x–50x faster** than CPU baseline without breaking AAD gradient backward passes.



---

#### **Phase 4: RNG Adjoint & Memory Rematerialization (Completed)**

* [x] **Philox State Management:** Implement robust global pointer addressing to avoid int32 overflow during high-scale path generation.
* [x] **Rematerialization:** Compute backward pass gradients via recomputation rather than storing massive intermediate trajectory tensors.
* [x] **Statistical Quality:** Ensure terminal distributions, mean, and variance match theoretical log-normal moments.
* [x] **Verification Goal:** Adjoint gradients match PyTorch autograd; peak memory footprint drops significantly compared to standard backpropagation.

#### **Phase 5: $O(1)$ Memory Fused Exposure Reduction (Completed)**

* [x] **Kernel Fusion:** Combine spot simulation, affine payoff collapse, and exposure reduction into a single Triton kernel.
* [x] **Hardware Optimization:** Optimize block size selection to stay strictly within SRAM limits.
* [x] **Performance Scaling:** Prove memory usage remains perfectly flat regardless of Monte Carlo path count (up to 50,000,000+ paths).
* [x] **Verification Goal:** Peak memory ratio stays bounded; CVA and Expected Exposure (EE) match unfused implementations within Monte Carlo error tolerances.

#### **Phase 6: Arbitrage-Free Volatility Modeling (Next)**

* [ ] **Market Data Ingestion:** Build pipelines to ingest real-world options chain data for calibration.
* [ ] **Surface Formulation:** Implement a SABR or Local Volatility (Dupire) model to map dynamic $\sigma(t, S_t)$.
* [ ] **Kernel Integration:** Modify the Triton $O(1)$ engine to accept dynamic volatility grids or state equations instead of a scalar parameter.
* [ ] **Verification Goal:** The generated volatility surface strictly satisfies no-calendar and no-butterfly arbitrage conditions; calibration loss converges.

#### **Phase 7: SOTA Quant Inference & MLOps (Future)**

* [ ] **Experiment Tracking:** Integrate MLflow or Weights & Biases to track volatility surface calibration metrics and hyperparameters.
* [ ] **High-Performance Serving:** Wrap the PyTorch/Triton engine in a production-ready inference server (e.g., NVIDIA Triton Inference Server or FastAPI with Redis caching).
* [ ] **Verification Goal:** API successfully accepts a portfolio payload and market data, calibrates the surface, and returns $O(1)$ CVA sensitivities with sub-second latency.
---

### How to use this going forward:

When starting any session with your AI coding assistant, simply copy the specific phase task from this checklist into your prompt, run the **Verification Goal** command, and check the box once it passes.