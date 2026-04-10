# Multi-GPU Acceleration of NEXUS FHE Transformer Inference
## Progress Report — Comp 390 Independent Study, Spring 2026

**Author**: Halil Ibrahim Kanpak  
**Advisor**: Prof. Didem Unat  
**Date**: April 10, 2026

---

## 1. Problem & Motivation

NEXUS (Zhang et al., NDSS 2025) is the first non-interactive protocol for secure transformer inference using FHE (Fully Homomorphic Encryption). It runs BERT-base inference in 37.3 seconds on 4x A100 GPUs with only 164 MB of client-server communication — orders of magnitude less than interactive MPC-based alternatives.

However, NEXUS uses its 4 GPUs **only for memory capacity** (evaluation keys exceed single-GPU memory), not for compute parallelism. No operation is distributed across GPUs. Meanwhile, Cerium (Jayashankar et al., arXiv 2025) demonstrated that multi-GPU FHE can achieve 3.4x speedup on 8 GPUs through:

1. **RNS limb-level parallelism**: distributing CRT residue polynomials across GPUs
2. **Two parallel key-switching algorithms**: Input Broadcast and Output Aggregation (from Cinnamon, ASPLOS 2025)
3. **Compiler-driven kernel fusion and scheduling**: overlapping compute and communication

Cerium's code is **not open source**. Our project implements the multi-GPU FHE infrastructure from scratch on top of the open-source NEXUS + Phantom stack, validated on MareNostrum 5 H100 GPUs.

---

## 2. Architecture

### 2.1 System Design

```
                     DistributedContext (n GPUs)
    ┌─────────────┬─────────────┬─────────────┬─────────────┐
    │   GPU 0     │   GPU 1     │   GPU 2     │   GPU 3     │
    │ PhantomCtx  │ PhantomCtx  │ PhantomCtx  │ PhantomCtx  │
    │ RelinKey    │ RelinKey    │ RelinKey    │ RelinKey    │
    │ NCCL comm   │ NCCL comm   │ NCCL comm   │ NCCL comm   │
    │ limbs:0,4,8 │ limbs:1,5,9 │ limbs:2,6,10│ limbs:3,7,11│
    └──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
           │ LOCAL ops   │ LOCAL ops   │ LOCAL ops   │ LOCAL ops
           │ (per-limb)  │ (per-limb)  │ (per-limb)  │ (per-limb)
           └──────┬──────┴──────┬──────┴──────┬──────┘
                  │     NCCL AllReduce / AllGather     │
                  │     (key-switching only)           │
                  └────────────────────────────────────┘
```

### 2.2 Operation Classification

| Type | Operations | Communication | Multi-GPU strategy |
|------|-----------|---------------|-------------------|
| **LOCAL** | add, sub, multiply_plain, negate, NTT | None | Each GPU processes local limbs only |
| **CROSS-LIMB** | rescale, mod_switch | None (but needs all limbs) | Gather → operate → scatter |
| **KEYED** | relinearize, rotate | AllGather or AllReduce | Input Broadcast or Output Aggregation |

### 2.3 Key-Switching Algorithms

**Input Broadcast (IB)**:
1. AllGather c2 limbs → every GPU gets full c2
2. Each GPU runs Phantom's `keyswitch_inplace` locally
3. No further communication

**Output Aggregation (OA)**:
1. Each GPU computes partial inner product (its assigned digits only)
2. AllReduce partial results
3. Each GPU does mod-down and add-to-ct locally

### 2.4 Implementation Scale

- **Core infrastructure**: 3,002 lines of CUDA/C++ (15 files)
- **Benchmarks**: 3,083 lines (9 benchmark programs)
- **Total**: 6,085 lines across 24 files

---

## 3. Results

### 3.1 NCCL Communication Performance

**Platform**: MareNostrum 5, 4x H100 64GB SXM (NVSwitch)

| Collective | Aggregate Bandwidth | Per-Ciphertext Latency (21 MB) |
|-----------|-------------------|------|
| AllGather | 1,026 GB/s | 20 us |
| AllReduce | 1,006 GB/s | 21 us |
| Broadcast | 807 GB/s | — |

**Platform**: RunPod, 2x H100 80GB SXM (NVLink)

| Collective | Aggregate Bandwidth | Per-Ciphertext Latency |
|-----------|-------------------|------|
| AllGather | 633 GB/s | 33 us |
| AllReduce | 428 GB/s | 49 us |

**Key finding**: Communication cost is negligible — 20 us vs 1.75 ms compute per key-switch = 1.1% overhead.

### 3.2 Key-Switching Correctness

All tests pass with MAE (Mean Absolute Error) < 1e-9 against single-GPU ground truth.

| N | L | Algorithm | 1 GPU | 2 GPUs | 4 GPUs |
|---|---|-----------|-------|--------|--------|
| 8192 | 5 | IB | PASS (8.0e-10) | PASS (8.1e-10) | PASS (8.3e-10) |
| 8192 | 5 | OA | PASS (8.1e-10) | PASS (8.1e-10) | PASS (8.2e-10) |
| 16384 | 10 | IB | PASS (1.1e-9) | PASS (1.2e-9) | PASS (1.2e-9) |
| 65536 | 20 | IB | PASS | PASS (2.3e-9) | PASS (2.3e-9) |
| 65536 | 20 | OA | PASS | PASS (2.9e-17) | PASS (0.0) |

### 3.3 Key-Switching Stage Breakdown

**N=65536, L=20 (NEXUS parameters), H100**:

| Stage | Time (ms) | Fraction | Distributable? |
|-------|----------|----------|---------------|
| modup (INTT + base conversion) | 1.074 | 65% | No* |
| inner_product | 0.421 | 25% | Yes (OA) |
| moddown | 0.164 | 10% | No |
| **Total** | **1.659** | **100%** | **25%** |

*modup's internal kernels are `static __global__` in Phantom — cannot be called from outside the library.

**N=65536, L=35 (full NEXUS depth)**:

| Stage | Time (ms) | Fraction |
|-------|----------|----------|
| modup | 3.025 | 66% |
| inner_product | 1.276 | 28% |
| moddown | 0.252 | 6% |
| **Total** | **4.554** | |

### 3.4 Scaling Results

**Output Aggregation key-switch only** (N=65536, L=20):

| GPUs | Time (ms) | Speedup |
|------|----------|---------|
| 1 | 1.820 | 1.00x |
| 2 | 1.864 | 0.98x |
| 4 | 1.659 | 1.08x |

**Input Broadcast overhead** (N=65536, L=20):

| GPUs | IB Time (ms) | GT Time (ms) | Overhead |
|------|-------------|-------------|----------|
| 1 | 1.745 | 1.749 | 0.0% |
| 2 | 1.845 | 1.755 | 5.1% |
| 4 | 1.828 | 1.749 | 4.5% |

**BERT layer simulation** (200 NTT + 50 mul + 30 add + 20 KS, N=65536):

| GPUs | Time (ms) | Speedup |
|------|----------|---------|
| 1 | 37.22 | 1.00x |
| 2 | 37.48 | 0.99x |
| 4 | 33.34 | **1.12x** |

### 3.5 Amdahl's Law Analysis

With only inner_product distributable (25% of KS time):
- Theoretical max speedup: 1 / (0.75 + 0.25/n) = **1.33x** at infinite GPUs
- Measured at 4 GPUs: **1.08x** (consistent with 25% distributable + overhead)

---

## 4. Technical Challenges Solved

### 4.1 Build Fixes (10 issues)
1. Phantom ExternalProject integration (CMake `CMAKE_SOURCE_DIR` conflict)
2. Phantom target naming (`Phantom` vs `phantom`)
3. CKKS encoder API (`std::vector<double>` required, not scalar)
4. `encrypt_symmetric` argument count
5. `PhantomCiphertext::resize` requires stream parameter
6. PhantomCiphertext copy constructor crashes → use `std::move`
7. NCCL must initialize before PhantomContext (memory corruption)
8. `constexpr double` with `pow()` → use `(double)(1ULL << 40)`
9. `CoeffModulus::Create` needs `phantom::arith::` prefix
10. macOS `._` resource fork files in tar archives

### 4.2 Algorithmic Fixes (3 critical bugs)
1. **AllGather padding**: NCCL requires uniform send count; with cyclic limb assignment and `total_limbs % n_gpus != 0`, GPUs have unequal limb counts → pad to `ceil(total/n_gpus)`
2. **Limb reorder after AllGather**: AllGather produces GPU-grouped order `[GPU0_limbs | GPU1_limbs]`; Phantom needs sequential `[limb0 | limb1 | limb2]` → GPU-side reorder kernel
3. **Secret key sharing**: Each GPU independently encrypts → different ciphertexts → garbage after AllGather. Fix: serialize key on GPU 0, load on all GPUs; encrypt once, distribute via `cudaMemcpyPeer`

### 4.3 Performance Optimizations
1. **Pre-allocated scratch buffers**: `thread_local` static buffers eliminate `cudaMalloc/cudaFree` per key-switch call. Reduced 2-GPU N=65536 from 15.5ms to 1.845ms (8.4x improvement).
2. **GPU-side reorder kernel**: Replaced host-side `cudaMemcpyAsync` loop (20 calls × 5us driver overhead each) with single CUDA kernel launch (~5us total).
3. **Persistent worker threads**: Eliminated `std::thread` creation/join overhead (1ms per GPU per iteration) by keeping threads alive across iterations with barrier synchronization.

---

## 5. Hardware & Infrastructure

### MareNostrum 5 (BSC, Barcelona)
- **ACC partition**: 1,120 nodes × 4x H100 64GB SXM = 4,480 GPUs
- **Interconnect**: NVSwitch (intra-node), InfiniBand NDR200 (inter-node)
- **Project**: etur02, user koc971580
- **NTL**: Built from source (`/gpfs/projects/etur02/hkanpak/local/`)

### RunPod (Cloud)
- 2x H100 80GB SXM with NVLink
- Used for initial debugging and validation ($10 budget)

### AWS (Blocked)
- Only 8 vCPU quota approved for P instances
- p4d.24xlarge (8x A100) requires 96 vCPUs — denied

---

## 6. Next Steps

### 6.1 Ciphertext-Level Pipeline Parallelism
The key insight: in a BERT layer, MatMul produces **64 independent ciphertexts**, each requiring its own key-switch. Instead of distributing one key-switch across GPUs (limited by Amdahl's Law at 1.43x), distribute **different ciphertexts to different GPUs**. Each GPU does full single-GPU operations on its batch — embarrassingly parallel, zero communication.

Expected speedup: ~3.5x at 4 GPUs for MatMul workloads.

### 6.2 Hybrid Strategy
- **Batch operations** (MatMul 64 cts): pipeline parallelism (near-linear)
- **Single-ct operations** (GELU, Softmax): OA key-switching (1.08-1.12x)
- Combined BERT layer: ~2-3x at 4 GPUs

### 6.3 Multi-Node Extension
Pipeline parallelism extends trivially to multi-node via MPI:
- MPI_Scatter ciphertexts across nodes
- Each node runs intra-node pipeline on its GPUs
- MPI_Gather results

Expected: ~8-10x at 4 nodes (16 GPUs) on MN5.

---

## References

1. Zhang et al., "NEXUS: Secure and Non-Interactive Transformer Inference on Encrypted Data", NDSS 2025
2. Jayashankar et al., "Cerium: A Scalable Multi-GPU Framework for Encrypted Large-Model Inference", arXiv:2512.11269, 2025
3. Jayashankar et al., "Cinnamon: A Framework for Scale-Out Encrypted AI", ASPLOS 2025
4. Phantom FHE Library, encryptorion-lab/phantom-fhe, GitHub
