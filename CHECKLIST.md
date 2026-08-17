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

* [ ] **Exposure Engine**: Build `compute_exposure_profiles` in `src/xva/exposure.py` for $EE(t)$ (Expected Exposure) and $PFE(t)$ (Potential Future Exposure).
* [ ] **CVA/DVA Pricing**: Build `compute_unilateral_cva` in `src/xva/cva.py` with flat hazard rate integration.
* [ ] **CVA Sensitivities**: Extend `aad_greeks` to differentiate CVA w.r.t. $S_0$, $\sigma$, and hazard rate $\lambda$ in a single backward pass.
* [ ] **Verification Goal**: Run `python -m pytest tests/test_phase2.py -v`.
* **Pass Condition**: Exposure profile tensors maintain positive floors; CVA matches single-forward analytical approximations; CVA AAD gradients match bump-and-revalue gradients within $10^{-3}$ tolerance.



---

#### **Phase 3: The GPU Deep Dive (Custom CUDA / Triton Kernels)**

* [ ] **Cloud GPU Migration**: Set up Google Colab / Kaggle environment with active T4 GPU.
* [ ] **Custom Path Generator**: Write C++/CUDA extension (`src/csrc/euler_maruyama.cu`) or Triton kernel for parallel SDE path generation.
* [ ] **PyTorch Binding**: Bind custom CUDA kernel to PyTorch autograd graph using `torch.utils.cpp_extension`.
* [ ] **Benchmarking Harness**: Write execution timing scripts in `benchmarks/` comparing CPU vs. PyTorch GPU vs. Custom CUDA.
* [ ] **Verification Goal**: Run `python benchmarks/bench_phase3.py`.
* **Pass Condition**: Custom CUDA kernel executes Monte Carlo trajectory generation at least **10x–50x faster** than CPU baseline without breaking AAD gradient backward passes.



---

#### **Phase 4: Neural-SDE Market Generator & Calibration**

* [ ] **Data Pipeline**: Build market data pullers (`src/data/fetcher.py`) using `yfinance` / Deribit for option implied volatility chains.
* [ ] **Neural Surrogate Model**: Implement `torchsde` Neural-SDE architecture to model rough/turbulent market paths (`src/models/neural_sde.py`).
* [ ] **Calibration Loop**: Calibrate Neural-SDE parameters against historical option chains.
* [ ] **Verification Goal**: Run `python -m pytest tests/test_phase4.py -v`.
* **Pass Condition**: Neural-SDE loss converges during training; calibrated generator reconstructs option surfaces accurately; AAD sensitivities remain stable through the neural SDE solver.



---

#### **Phase 5: Production Microservice & MLOps Infrastructure**

* [ ] **REST API**: Wrap pricing and XVA engine in a FastAPI application (`src/api/app.py`) with `/price`, `/xva`, and `/greeks` endpoints.
* [ ] **Caching Layer**: Integrate Redis to cache generated exposure profiles for identical netting sets.
* [ ] **Experiment Tracking**: Log model calibration runs, SDE parameters, and loss curves using Weights & Biases / MLflow.
* [ ] **Containerization**: Write a working `Dockerfile` for the API service.
* [ ] **Verification Goal**: Run `pytest tests/test_api.py` and ping local/cloud endpoints.
* **Pass Condition**: API returns sub-second XVA and Greeks JSON responses for incoming portfolio payload requests.



---

#### **Phase 6: Benchmarking, Thesis Writing & Defense Prep**

* [ ] **Speedup Profiling**: Produce final latency and throughput comparison charts (CPU vs Native GPU vs Custom CUDA).
* [ ] **Sensitivity Bounds**: Plot AAD error bounds vs Finite Difference bump-and-revalue times.
* [ ] **Thesis Draft**: Complete full LaTeX draft including background, methodology, system architecture, and empirical benchmarks.
* [ ] **Verification Goal**: Codebase fully documented, repository public on GitHub, thesis document submitted for review.

---

### How to use this going forward:

When starting any session with your AI coding assistant, simply copy the specific phase task from this checklist into your prompt, run the **Verification Goal** command, and check the box once it passes.