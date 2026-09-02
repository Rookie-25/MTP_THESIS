# XVA engine: PyTorch baseline vs Triton fused kernels

_Generated 2026-09-02 21:17 UTC_

## Environment

| Item               | Value                |
|--------------------|----------------------|
| GPU                | Tesla T4 (14.6 GiB)  |
| Compute capability | 7.5                  |
| PyTorch            | 2.11.0+cu128         |
| CUDA               | 12.8                 |
| Triton             | 3.6.0                |
| Python             | 3.13.15              |
| dtype              | float32              |
| Time steps N       | 252                  |
| Repeats            | 3 (minimum reported) |
| Measuring          | forward only         |

## What each backend does

| Backend                    | Pipeline                                        | O(M*N) tensors in HBM |
|----------------------------|--------------------------------------------------|------------------------|
| PyTorch baseline           | `simulate_gbm` -> MtM -> clamp -> mean          | ~6                    |
| Phase 3 (Triton + dW)      | draw `dW` -> `triton_simulate_gbm` -> reduce    | 3                     |
| Phase 4 (in-kernel Philox) | `philox_simulate_gbm` (in-kernel RNG) -> reduce | 2                     |
| Phase 5 (fused reduction)  | `fused_expected_exposure`                       | **0**                 |

All four produce the identical output: an expected-exposure profile of shape `(253,)`. `dW` generation is timed inside the Phase 3 measurement, since Phases 4 and 5 must produce their increments too.

## Execution time (ms)

| M          | PyTorch baseline            | Phase 3 (Triton + dW)       | Phase 4 (in-kernel Philox) | Phase 5 (fused reduction) |
|------------|------------------------------|------------------------------|-----------------------------|----------------------------|
| 100,000    | 10.5                        | 8.2                         | 9.8                        | 4.0                       |
| 1,000,000  | 137.4                       | 76.8                        | 82.6                       | 17.2                      |
| 5,000,000  | **OOM** (pred, ~23.54 GiB)  | **OOM** (pred, ~14.12 GiB)  | 408.7                      | 87.1                      |
| 10,000,000 | **OOM** (pred, ~47.09 GiB)  | **OOM** (pred, ~28.24 GiB)  | **OOM** (pred, ~18.85 GiB) | 171.9                     |
| 50,000,000 | **OOM** (pred, ~235.44 GiB) | **OOM** (pred, ~141.19 GiB) | **OOM** (pred, ~94.25 GiB) | 864.6                     |

## Peak VRAM

| M          | PyTorch baseline            | Phase 3 (Triton + dW)       | Phase 4 (in-kernel Philox) | Phase 5 (fused reduction) |
|------------|------------------------------|------------------------------|-----------------------------|----------------------------|
| 100,000    | 482.2 MiB                   | 386.4 MiB                   | 290.3 MiB                  | 3.0 MiB                   |
| 1,000,000  | 4.71 GiB                    | 3.78 GiB                    | 2.84 GiB                   | 4.3 MiB                   |
| 5,000,000  | **OOM** (pred, ~23.54 GiB)  | **OOM** (pred, ~14.12 GiB)  | 14.15 GiB                  | 4.3 MiB                   |
| 10,000,000 | **OOM** (pred, ~47.09 GiB)  | **OOM** (pred, ~28.24 GiB)  | **OOM** (pred, ~18.85 GiB) | 4.3 MiB                   |
| 50,000,000 | **OOM** (pred, ~235.44 GiB) | **OOM** (pred, ~141.19 GiB) | **OOM** (pred, ~94.25 GiB) | 4.3 MiB                   |

## Speedup over PyTorch baseline

| M          | Phase 3 (Triton + dW) | Phase 4 (in-kernel Philox) | Phase 5 (fused reduction) |
|------------|------------------------|------------------------------|-----------------------------|
| 100,000    | 1.29x                 | 1.07x                      | 2.63x                     |
| 1,000,000  | 1.79x                 | 1.66x                      | 7.97x                     |
| 5,000,000  | n/a (baseline OOM)    | n/a (baseline OOM)         | n/a (baseline OOM)        |
| 10,000,000 | n/a (baseline OOM)    | n/a (baseline OOM)         | n/a (baseline OOM)        |
| 50,000,000 | n/a (baseline OOM)    | n/a (baseline OOM)         | n/a (baseline OOM)        |

_`n/a (baseline OOM)` marks path counts where no speedup is definable because the baseline cannot run at all -- which is the more important result._

## Where pure PyTorch stops

### Survival by path count

| M          | PyTorch baseline | Phase 3 (Triton + dW) | Phase 4 (in-kernel Philox) | Phase 5 (fused reduction) |
|------------|-------------------|--------------------------|------------------------------|-----------------------------|
| 100,000    | ok (11 ms)       | ok (8 ms)               | ok (10 ms)                  | ok (4 ms)                 |
| 1,000,000  | ok (137 ms)      | ok (77 ms)              | ok (83 ms)                  | ok (17 ms)                |
| 5,000,000  | **OOM**          | **OOM**                 | ok (409 ms)                 | ok (87 ms)                |
| 10,000,000 | **OOM**          | **OOM**                 | **OOM**                     | ok (172 ms)               |
| 50,000,000 | **OOM**          | **OOM**                 | **OOM**                     | ok (865 ms)               |

- **Largest M the baseline completed:** 1,000,000
- **First M where the baseline OOMs:** 5,000,000

Bisected boundary: the baseline completes at **2,750,000** paths and fails at **2,781,250** (bracket width 1.1%).

At M = 5,000,000, where the baseline cannot run, these complete:

- Phase 4 (in-kernel Philox) (408.7 ms, peak 14.15 GiB)
- Phase 5 (fused reduction) (87.1 ms, peak 4.3 MiB)

### Analytic memory model

At N = 252, float32, one M x (N+1) tensor is `M x 253 x 4` bytes:

| M          | one M x (N+1) | baseline (~6x) | Phase 4 (~2x) | Phase 5 |
|------------|----------------|-----------------|-----------------|---------|
| 1,000,000  | 965.1 MiB     | 5.65 GiB        | 1.88 GiB        | 4.0 MiB |
| 5,000,000  | 4.71 GiB      | 28.27 GiB       | 9.42 GiB        | 4.0 MiB |
| 10,000,000 | 9.42 GiB      | 56.55 GiB       | 18.85 GiB       | 4.0 MiB |
| 50,000,000 | 47.12 GiB     | 282.75 GiB      | 94.25 GiB       | 4.0 MiB |

Phase 5's column carries no M term at all: peak is `min(ceil(M / BLOCK_M), max_programs) x (N+1) x element_size`, which is constant above the grid-saturation point.

## Reading these numbers

- **Peak VRAM is `torch.cuda.max_memory_allocated`** -- what the PyTorch caching allocator handed out, not total process VRAM.
- **`OOM (pred)` means refused before launch** by a pre-flight check, not attempted and failed. Attempting a configuration that overruns the device risks an asynchronous abort that would poison the CUDA context and invalidate every later row in the sweep.
- **Timings are the minimum of 3 repeats**, with a warm-up excluded so Triton JIT compilation is not billed to the first measurement.
- **Phases 4 and 5 draw different sample paths** from the baseline (different Philox addressing), so their exposure profiles agree only within Monte-Carlo error. Correctness is established in `tests/test_phase4.py` and `tests/test_phase5.py`, not here.
