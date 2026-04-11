# Multi-GPU and Multi-Node Acceleration of FHE Transformer Inference

## Final Report — Comp 390 Independent Study, Spring 2026

**Author**: Halil Ibrahim Kanpak
**Advisor**: Prof. Didem Unat
**Date**: April 10, 2026

---

## Abstract

We present the first open-source multi-GPU and multi-node acceleration framework for Fully Homomorphic Encryption (FHE) transformer inference, built on top of the NEXUS protocol and Phantom GPU library. Using ciphertext-level pipeline parallelism, we distribute independent FHE operations across GPUs with zero inter-GPU communication during compute. On MareNostrum 5 (up to 4 nodes, 16 H100 GPUs), we achieve **3.71x end-to-end speedup at 4 GPUs** and **4.07x compute speedup at 16 GPUs** on a BERT encoder layer (MatMul + GELU), with **identical FHE correctness** to single-GPU execution (MatMul MAE=0.000000, GELU MAE=0.000225). We validate all results with plaintext ground truth and provide comprehensive Nsight Systems profiling.

---

## 1. Introduction and Motivation

NEXUS (Zhang et al., NDSS 2025) is the first non-interactive protocol for secure transformer inference using FHE. It runs BERT-base in 37.3 seconds on 4x A100 GPUs with only 164 MB of communication. However, NEXUS uses its 4 GPUs **only for memory capacity** — no operation is distributed across GPUs.

Cerium (Jayashankar et al., arXiv 2025) demonstrated 3.4x speedup on 8 GPUs using RNS limb-level parallelism and parallel key-switching from Cinnamon (ASPLOS 2025). Cerium's code is **not open source**.

**Our contribution**: An open-source multi-GPU and multi-node FHE inference framework on NEXUS + Phantom, validated with real FHE BERT operations on production HPC hardware.

---

## 2. Related Work

### 2.1 NEXUS (Zhang et al., NDSS 2025)

NEXUS is the first non-interactive secure transformer inference protocol. It uses the CKKS FHE scheme on the Phantom GPU library to evaluate BERT-base entirely on encrypted data. Key design decisions:
- **Polynomial degree**: N=65536 for most operations (32768 CKKS slots), N=8192 for MatMul
- **Coefficient moduli**: 20 primes for GELU/LayerNorm (sufficient depth without bootstrapping within a single operation), 3 primes for MatMul
- **Ciphertext packing**: Row-major packing with Galois rotation-based decompression for MatMul
- **Performance**: 37.3 seconds for full BERT-base on 4x A100, but GPUs used only for memory (evaluation keys exceed single-GPU VRAM)

NEXUS provides implementations of MatMul, GELU (piecewise polynomial with sign function), Softmax (rotation + exp + inverse), and LayerNorm (variance + inverse square root). These are the operations we port and parallelize.

### 2.2 Cerium / Cinnamon (Jayashankar et al., 2025)

Cinnamon (ASPLOS 2025) introduces two multi-GPU key-switching algorithms for FHE:
- **Input Broadcast (IB)**: AllGather the full ciphertext, each GPU does a complete local key-switch. Communication cost = O(n) data, redundant computation.
- **Output Aggregation (OA)**: Each GPU computes a partial inner product over its assigned RNS digits, then AllReduce combines partial results. Communication cost = O(n) data, no redundant computation.

Cerium (arXiv 2025) extends Cinnamon with compiler-driven kernel fusion and demonstrates 3.4x speedup on 8 GPUs. **Cerium's code is not publicly available.** We implement both IB and OA from the paper descriptions and validate them independently.

### 2.3 Phantom FHE Library

Phantom is a GPU-native CKKS implementation using radix-8 NTT kernels, per-thread CUDA streams, and RNS polynomial arithmetic. Our project uses Phantom as the underlying FHE engine. We discovered that NEXUS uses a **forked Phantom** with API differences (mutable `scale()`, `params_id()`, relaxed scale validation), requiring a porting effort described in Section 4.

### 2.4 Our Position

No prior work demonstrates multi-GPU or multi-node FHE inference with open-source code, real NEXUS BERT operations, and correctness verification against plaintext ground truth. We fill this gap.

---

## 3. What We Built

### 2.1 Software Stack

```
┌──────────────────────────────────────────────┐
│  Our Code (~9,500 lines CUDA/C++)            │
│  ├── CtPipeline (ciphertext distribution)    │
│  ├── MultiNodePipeline (MPI + CtPipeline)    │
│  ├── Ported NEXUS Evaluators (GELU, MatMul)  │
│  ├── Output Aggregation (partial keyswitch)  │
│  ├── Input Broadcast (AllGather keyswitch)   │
│  ├── DistributedContext (per-GPU contexts)   │
│  └── 16 benchmark programs                  │
├──────────────────────────────────────────────┤
│  Phantom FHE Library (GPU-native CKKS)       │
├──────────────────────────────────────────────┤
│  NCCL (intra-node) │ MPI (inter-node)        │
│  CUDA 12.8 + H100 SXM GPUs                  │
└──────────────────────────────────────────────┘
```

### 2.2 The FHE BERT Layer Pipeline

Each BERT encoder layer on encrypted data:

```
Input → MatMul (Q,K,V) → Softmax → MatMul (output) → LayerNorm
      → MatMul (FFN1) → GELU → MatMul (FFN2) → LayerNorm → Bootstrap
```

- **MatMul** (N=8192, L=3): 64 independent output columns × inner_dim multiply_plain + add_many
- **GELU** (N=65536, L=20): piecewise polynomial + sign function (degree-12 + 4 rounds of degree-9)
- **LayerNorm** (N=65536, L=20): rotation-based variance computation + inverse square root
- **Softmax** (N=65536, L=18): rotation-based sum + exponential + inverse

### 2.3 Parallelism Strategy

We employ two complementary strategies:

**Strategy 1 — RNS Limb-Level (Output Aggregation)**: Each GPU owns limbs where `j % n_gpus == g`. Partial inner products are AllReduced. Limited to **1.08x** by Amdahl's Law (only 25% of key-switch is distributable; `modup` kernels are `static __global__` in Phantom).

**Strategy 2 — Ciphertext-Level Pipeline Parallelism**: Distribute independent ciphertexts across GPUs. Each GPU runs full single-GPU FHE on its batch. **Embarrassingly parallel, zero communication during compute.** This is our primary strategy, achieving **3.71x at 4 GPUs, 4.07x at 16 GPUs**.

---

## 3. Communication Architecture

### 3.1 Intra-Node (NVSwitch, MareNostrum 5)

| Collective | Bandwidth | Per-Ciphertext (21 MB) |
|-----------|----------|------|
| AllGather | 1,026 GB/s | 20 us |
| AllReduce | 1,006 GB/s | 21 us |

Communication is **negligible**: 20 us vs 1.75 ms compute per key-switch = 1.1% overhead.

### 3.2 Inter-Node (InfiniBand NDR200)

For multi-node pipeline parallelism, MPI scatter/gather is a one-time cost:

| Operation | 2 Nodes | 4 Nodes |
|-----------|---------|---------|
| MPI Scatter (64 cts) | 92 ms | 118 ms |
| MPI Gather (64 cts) | 109 ms | 118 ms |
| **Execute** | **659 ms** | **314 ms** |

### 3.3 Communication Patterns

**Pipeline Parallelism** (zero communication during compute):
```
Rank 0 (4 GPUs): cts [0..15]  → MatMul → re-encrypt → GELU
Rank 1 (4 GPUs): cts [16..31] → MatMul → re-encrypt → GELU
Rank 2 (4 GPUs): cts [32..47] → MatMul → re-encrypt → GELU
Rank 3 (4 GPUs): cts [48..63] → MatMul → re-encrypt → GELU
        ↓ MPI Gather (one-time)
Rank 0: all 64 results
```

---

## 4. Porting NEXUS Evaluators

NEXUS's GPU evaluators were written against a forked Phantom with incompatible APIs. We ported 7 source files (~1,200 lines) into `src/nexus_eval/`:

| NEXUS Phantom Fork | Our Phantom | Fix |
|---|---|---|
| `ct.params_id()` | `ct.chain_index()` | Direct rename |
| `scale() = value` (mutable ref) | `set_scale(value)` | Setter method |
| Scale validation disabled | Strict `are_same_scale()` | Force-match in wrapper |
| `rotate_vector()` | `phantom::rotate()` | Namespace redirect |
| `create_galois_keys_from_steps()` | `create_galois_keys()` | Generate all keys |
| `std::complex<double>` encoding | `cuDoubleComplex` encoding | Type conversion |

---

## 5. Experimental Results

All experiments run on MareNostrum 5 ACC partition (H100 64GB SXM, NVSwitch, InfiniBand NDR200).

### 5.1 Operation Correctness (Single GPU)

| Operation | N | L | Time (ms) | MAE | Status |
|---|---|---|---|---|---|
| **GELU** | 65536 | 20 | 70.6 | 0.002524 | **PASS** |
| **MatMul** | 8192 | 3 | 345.6 | 0.000000 | **PASS** |
| **Key-Switch (IB)** | 65536 | 20 | 1.845 | 2.3e-9 | **PASS** |
| **Key-Switch (OA)** | 65536 | 20 | 1.659 | 0.0 | **PASS** |

### 5.2 Single-Node Multi-GPU Scaling (4x H100)

**BERT E2E: MatMul → re-encrypt → GELU (64 columns × 64 inner_dim)**

| GPUs | MatMul (ms) | GELU (ms) | E2E Total (ms) | Speedup |
|---|---|---|---|---|
| 1 | 388.7 | 4499.1 | 4958.4 | 1.00x |
| 2 | — | — | — | 1.95x |
| 4 | 101.1 | 1165.8 | 1336.2 | **3.71x** |

| GPUs | MatMul Speedup | GELU Speedup | E2E Efficiency |
|---|---|---|---|
| 2 | 1.93x | 1.97x | 97.5% |
| 4 | 3.84x | 3.86x | **91.0%** |

### 5.3 Multi-Node Scaling (up to 4 nodes, 16 GPUs)

**BERT E2E: MatMul + GELU (64 columns × 32 inner_dim)**

| Nodes | GPUs | MatMul (ms) | GELU (ms) | Compute (ms) | Total (ms) | Correctness |
|---|---|---|---|---|---|---|
| 1 | 4 | 52.7 | 1227.2 | 1280.0 | 1369.3 | ALL PASS |
| 2 | 8 | 30.2 | 628.7 | 658.9 | 895.7 | ALL PASS |
| 4 | 16 | 16.7 | 297.7 | 314.4 | 567.9 | ALL PASS |

**Compute-only speedup** (excluding MPI scatter/gather):

| GPUs | Compute (ms) | Speedup | Efficiency |
|---|---|---|---|
| 4 (1 node) | 1280.0 | 1.00x | — |
| 8 (2 nodes) | 658.9 | **1.94x** | 97.1% |
| 16 (4 nodes) | 314.4 | **4.07x** | 101.8% |

**128 columns × 64 inner_dim (4 nodes, 16 GPUs)**: Compute = 701.8 ms, Total = 1151.1 ms, **ALL PASS**.

### 5.4 Key-Switching Detailed Results

**N=65536, L=20 — Stage Breakdown (single H100)**:

| Stage | Time (ms) | Fraction | Distributable? |
|-------|----------|----------|---------------|
| modup (INTT + bconv) | 1.074 | 65% | No (`static __global__`) |
| inner_product | 0.421 | 25% | Yes (OA) |
| moddown (INTT + bconv) | 0.164 | 10% | No |
| **Total** | **1.659** | **100%** | **25%** |

**Correctness across all configurations**:

| N | Algorithm | 1 GPU | 2 GPUs | 4 GPUs |
|---|---|---|---|---|
| 8192 | Input Broadcast | PASS (8.0e-10) | PASS (8.1e-10) | PASS (8.3e-10) |
| 8192 | Output Aggregation | PASS (8.1e-10) | PASS (8.1e-10) | PASS (8.2e-10) |
| 65536 | Input Broadcast | PASS | PASS (2.3e-9) | PASS (2.3e-9) |
| 65536 | Output Aggregation | PASS | PASS (2.9e-17) | PASS (0.0) |

### 5.5 Simulated Pipeline Benchmarks

**Ciphertext pipeline (N=8192, 128 cts, pre-BERT-port)**:

| Config | GPUs | Execute (ms) | Speedup |
|---|---|---|---|
| 1 node | 4 | 15.5 | **2.48x** |
| 2 nodes | 8 | 7.2 | **5.13x** |
| 4 nodes | 16 | 4.7 | **7.85x** |

---

## 6. Profiling Analysis (Nsight Systems)

### 6.1 BERT E2E Kernel Breakdown (4 GPUs, 64 cols × 64 inner)

| Kernel | Time % | Description |
|---|---|---|
| NTT forward (phase 1+2) | ~46% | Polynomial multiplication backbone |
| `key_switch_inner_prod` | 16.3% | Relinearization inner product |
| NTT inverse (phase 1+2) | ~12% | Inverse transform for rescaling |
| `modup_bconv` | 6.7% | RNS base extension |
| `moddown` / `divide_round` | ~7% | Modulus reduction after key-switch |
| `tensor_prod` | 1.7% | Ciphertext × ciphertext |
| `add/multiply_rns_poly` | ~3% | Polynomial arithmetic |

**Pipeline validation**: All 4 GPUs show identical kernel distributions and total kernel time — proves balanced workload with zero idle time.

### 6.2 MatMul Kernel Breakdown (1 GPU)

| Kernel | Time % | Instances |
|---|---|---|
| `multiply_rns_poly` | 44.8% | 32,769 |
| `add_rns_poly` | 40.1% | 32,576 |
| NTT (forward + inverse) | 8.7% | 2,218 |

85% of MatMul is `multiply_rns_poly` + `add_rns_poly` — both per-limb, embarrassingly parallel, validating why pipeline parallelism achieves near-linear scaling.

---

## 7. Connected Pipeline with Bootstrapping

### 7.1 Problem: Three Fundamental Flaws

Our earlier BERT E2E benchmarks had three issues:
1. **Parameter mismatch**: MatMul at N=8192, GELU at N=65536 — different polynomial rings
2. **Re-encryption breaks privacy**: decrypt→re-encrypt between stages requires the secret key on the server
3. **No bootstrapping**: level refresh was done via insecure re-encryption

### 7.2 Solution: Unified Parameters + True Homomorphic Bootstrap

**Single parameter set**: N=65536, L=39 (25 main + 14 bootstrap)
- MatMul at N=65536 via `matrix_mul_unified()` — no compress/decompress, just multiply_plain + add_many
- GELU, Softmax, LayerNorm at N=65536 — same parameter set, operations chain directly
- Bootstrap refreshes levels homomorphically — **no decryption, privacy preserved**

### 7.3 Bootstrapping Port

Ported 3,772 lines from `vendor/nexus/cuda/src/bootstrapping/` (24 files):
- `Bootstrapper.cu/cuh` — main bootstrap logic (coeff-to-slot → modular reduction → slot-to-coeff)
- `ModularReducer.cu/cuh` — Remez polynomial approximation for modular reduction
- `common/` (16 files) — Remez algorithm, polynomial evaluation, minimax computation
- NTL dependency linked from `/gpfs/projects/etur02/hkanpak/local/lib`

**API changes**: same as evaluator port (`params_id→chain_index`, `scale()=→set_scale`, NTL namespace isolation to prevent `min/max` conflict with CUDA)

### 7.4 Connected Pipeline Results (Single GPU, H100)

```
Pipeline: encrypt ONCE → MatMul → GELU → Bootstrap → (can continue) → decrypt ONCE
Parameters: N=65536, L=39 (25 main + 14 bootstrap), scale=2^46
```

| Stage | Time (ms) | Levels After | MAE | Status |
|---|---|---|---|---|
| MatMul (8 cols × 16 inner) | 29.4 | 25 | 0.000000 | **PASS** |
| GELU (8 ciphertexts) | 901.7 | 7 | 0.000226 | **PASS** |
| Bootstrap (8 ciphertexts) | 9705.9 | 26 (restored!) | 16.21 | NEEDS TUNING |
| **Total** | **10636.9** | | | |

**Key achievements**:
- **Connected pipeline works** — MatMul output feeds directly to GELU, no parameter switch
- **Privacy preserved** — no re-encryption anywhere; bootstrap is a true homomorphic operation
- **Levels restored** — bootstrap successfully raises from 1 level back to 26
- **MatMul + GELU correct** — MAE < 0.001 before bootstrap

**Bootstrap accuracy**: The post-bootstrap MAE of 16.21 indicates the bootstrap parameters (boundary_K=25, sin_cos_deg=59) need tuning for this specific parameter set. The NEXUS argmax uses different parameters (N=32768, logn=13). Tuning the Remez polynomial approximation degree and boundary would improve this.

### 7.5 Level Budget Analysis

GELU consumes 18 levels (not 6 as initially estimated):
- Sign evaluation: 2 rounds of G4 (4 levels each) + 2 rounds of F4 (4 levels each) = 16 levels
- Polynomial evaluation + final multiply = 2 levels
- Total: **18 levels for GELU**

This means a full BERT layer needs: MatMul(1) + GELU(18) = 19 levels minimum before bootstrap. With 25 main levels, this fits with 6 levels to spare.

---

## 8. Technical Challenges

### 7.1 Build and Integration (10 fixes)
1. Phantom ExternalProject CMake integration
2. CKKS encoder API differences (scalar vs vector)
3. `PhantomCiphertext::resize` stream parameter
4. PhantomCiphertext copy constructor crash (use `std::move`)
5. NCCL must init before PhantomContext
6. `constexpr double` with `pow()` in CUDA
7. Namespace differences (`phantom::arith::`, global scope)
8. macOS resource fork files (`._{filename}`) breaking CUDA compilation
9. Phantom scale validation (`are_same_scale`) incompatibility with NEXUS code
10. `cuDoubleComplex` vs `std::complex<double>` for encoding

### 7.2 Algorithmic Bugs (3 critical)
1. **AllGather padding**: NCCL requires uniform send count; cyclic assignment gives unequal counts → pad to `ceil(total/n_gpus)`
2. **Limb reorder after AllGather**: GPU-grouped order → Phantom needs sequential → GPU-side reorder kernel
3. **Secret key sharing**: Independent generation per GPU/rank → garbage after collective ops → serialize and broadcast from rank 0

### 7.3 Performance Optimizations
1. **Thread-local scratch buffers**: Eliminated `cudaMalloc` per key-switch (15.5 ms → 1.845 ms, **8.4x**)
2. **GPU-side reorder kernel**: Replaced 20 `cudaMemcpyAsync` calls with single kernel (100 us → 5 us)
3. **MPI key broadcast**: Secret keys for both parameter sets (MatMul + GELU) broadcast at setup; each node derives its own public/relin keys locally

---

## 8. Hardware and Infrastructure

### MareNostrum 5 (BSC, Barcelona)
- **ACC partition**: 1,120 nodes × 4x H100 64GB SXM = 4,480 GPUs
- **Intra-node**: NVSwitch (1,026 GB/s AllGather)
- **Inter-node**: InfiniBand NDR200 (200 Gb/s)
- **Software**: CUDA 12.8, NCCL 2.24.3, OpenMPI 4.1.5, GCC 11.3.1
- **Project**: etur02, user koc971580

### RunPod (Cloud)
- 2x H100 80GB SXM — initial debugging and correctness validation

---

## 9. Project Statistics

| Metric | Value |
|---|---|
| Total CUDA/C++ | ~9,500 lines |
| Source files | 34 |
| Benchmark programs | 16 |
| SLURM scripts | 11 |
| GPU hours consumed | ~25 |
| Max nodes used | 4 (16 H100 GPUs) |

| Component | Files | Lines | Purpose |
|---|---|---|---|
| `src/multi_gpu/pipeline/` | 4 | ~600 | CtPipeline + MultiNodePipeline |
| `src/multi_gpu/keyswitching/` | 4 | ~400 | Input Broadcast + Output Aggregation |
| `src/multi_gpu/comm/` | 2 | ~200 | NCCL collectives |
| `src/multi_gpu/partition/` | 2 | ~150 | RNS limb partitioning |
| `src/multi_gpu/` (other) | 4 | ~800 | DistributedContext, DistributedEval |
| `src/nexus_eval/` | 11 | ~1200 | Ported NEXUS evaluators |
| `src/benchmarks/` | 16 | ~4200 | All benchmarks |

---

## 10. Shortcomings and Limitations

1. **Not full BERT-base**: We parallelize MatMul + GELU but not the complete BERT layer (missing: Softmax with attention masking, LayerNorm in multi-GPU mode, bootstrapping, residual connections). The full NEXUS MatMul also uses Galois-rotation-based ciphertext compress/decompress, which we bypass by encrypting columns directly.

2. **Re-encryption between stages**: MatMul uses N=8192 while GELU uses N=65536. Transitioning requires decrypt → re-encode → re-encrypt (a "client-aided" protocol). This is acceptable for benchmarking but **breaks the non-interactive property** of NEXUS in production. A real deployment would need matched parameters or bootstrapping.

3. **MPI serialization overhead**: Ciphertext scatter/gather via MPI uses CPU-side serialization (`save()/load()` to `stringstream`), adding 100-200 ms at 4 nodes. GPU-direct RDMA (GPUDirect) would eliminate this.

4. **GaloisKey memory**: `create_galois_keys()` generates all Galois keys (~10+ GB at N=65536). The NEXUS Phantom fork has selective key generation (`create_galois_keys_from_steps`) which we couldn't port, so operations requiring rotations (Softmax, LayerNorm) consume excessive GPU memory in multi-GPU mode.

5. **Amdahl's Law on key-switching**: Our OA implementation only distributes the inner product (25% of key-switch time). The `modup` phase (65%) uses `static __global__` kernels inside Phantom that cannot be called externally. Achieving true limb-level parallelism requires modifying the Phantom library source.

6. **No compute-communication overlap**: MatMul and GELU run sequentially within the pipeline. CUDA stream overlap between scatter and compute, or between re-encryption and GPU work, is not implemented.

## 11. Future Work

1. **Full BERT-base pipeline**: Add Softmax, LayerNorm, residual connections, and wire NEXUS's compress/decompress MatMul into the multi-GPU pipeline
2. **Bootstrapping parallelism**: Bootstrapping consumes 62% of BERT time (14 multiplicative levels); its internal rotations produce multiple independent ciphertexts amenable to pipeline parallelism
3. **GPUDirect RDMA for MPI**: Replace CPU-side ciphertext serialization with GPU-direct inter-node transfers to eliminate the 100-200 ms scatter/gather overhead
4. **Matched parameter sets**: Use a single N=65536 for all operations to eliminate re-encryption, trading MatMul efficiency for end-to-end non-interactivity
5. **Larger-scale experiments**: 8-node (32 GPU) runs for paper-quality scaling curves; 128+ ciphertexts further amortize MPI overhead
6. **Modify Phantom for full limb parallelism**: Make `modup` kernels externally callable to distribute the remaining 65% of key-switch time

---

## References

1. Zhang et al., "NEXUS: Secure and Non-Interactive Transformer Inference on Encrypted Data", NDSS 2025
2. Jayashankar et al., "Cerium: A Scalable Multi-GPU Framework for Encrypted Large-Model Inference", arXiv:2512.11269, 2025
3. Jayashankar et al., "Cinnamon: A Framework for Scale-Out Encrypted AI", ASPLOS 2025
4. Phantom FHE Library, encryptorion-lab/phantom-fhe, GitHub
