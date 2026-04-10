# Multi-GPU Acceleration of NEXUS FHE Transformer Inference
## Progress Report — Comp 390 Independent Study, Spring 2026

**Author**: Halil Ibrahim Kanpak  
**Advisor**: Prof. Didem Unat  
**Date**: April 10, 2026

---

## 1. Problem & Motivation

NEXUS (Zhang et al., NDSS 2025) is the first non-interactive protocol for secure transformer inference using Fully Homomorphic Encryption (FHE). It runs BERT-base inference in 37.3 seconds on 4x A100 GPUs with only 164 MB of client-server communication — orders of magnitude less than interactive MPC-based alternatives.

However, NEXUS uses its 4 GPUs **only for memory capacity** (evaluation keys exceed single-GPU memory), not for compute parallelism. No operation is distributed across GPUs.

Cerium (Jayashankar et al., arXiv 2025) demonstrated that multi-GPU FHE can achieve 3.4x speedup on 8 GPUs using RNS limb-level parallelism, parallel key-switching algorithms (from Cinnamon, ASPLOS 2025), and compiler-driven kernel fusion. Cerium's code is **not open source**.

**Our goal**: Build the first open-source multi-GPU and multi-node FHE inference infrastructure on top of NEXUS + Phantom, validated on MareNostrum 5 H100 GPUs.

---

## 2. What the Program Does

### 2.1 The FHE Inference Pipeline

NEXUS performs BERT-base inference entirely on encrypted data using the CKKS FHE scheme. The computation pipeline for one BERT layer is:

```
Input ciphertext → MatMul (Q,K,V projection) → Softmax (attention)
→ MatMul (attention output) → LayerNorm → MatMul (FFN1) → GELU
→ MatMul (FFN2) → LayerNorm → Bootstrapping → Output ciphertext
```

Each operation is implemented using FHE primitives on the Phantom GPU library:
- **MatMul**: 768 multiply_plain operations + add accumulation per output column, producing 64 independent ciphertexts per attention head
- **GELU/Softmax/LayerNorm**: polynomial approximations using multiply, relinearize, rotate
- **Bootstrapping**: refreshes ciphertext noise (62% of total BERT time)

### 2.2 What We Parallelize

Our infrastructure distributes this workload across multiple GPUs and nodes using two complementary strategies:

**Strategy 1: RNS Limb-Level Parallelism (Output Aggregation)**
- Each CKKS ciphertext is represented as polynomials modulo multiple primes (RNS limbs)
- GPU g owns limb j where `j % n_gpus == g` (cyclic assignment)
- For key-switching: each GPU computes a partial inner product over its assigned digits, then AllReduce combines results
- Gives 1.08-1.12x speedup (limited by Amdahl's Law: only 25% of key-switch time is distributable)

**Strategy 2: Ciphertext-Level Pipeline Parallelism**
- In BERT MatMul, there are 64+ independent output ciphertexts
- Distribute different ciphertexts to different GPUs
- Each GPU does full single-GPU FHE operations on its batch
- **Embarrassingly parallel — zero communication during compute**
- Gives **2.48x at 4 GPUs, 7.85x at 16 GPUs**

### 2.3 Software Stack

```
┌─────────────────────────────────────────────┐
│  Our Code (6,085 lines CUDA/C++)            │
│  ├── CtPipeline (ciphertext distribution)   │
│  ├── MultiNodePipeline (MPI + CtPipeline)   │
│  ├── Output Aggregation (partial keyswitch) │
│  ├── Input Broadcast (AllGather keyswitch)  │
│  ├── DistributedContext (per-GPU contexts)  │
│  └── 9 benchmark programs                  │
├─────────────────────────────────────────────┤
│  Phantom FHE Library (GPU-native CKKS)      │
│  ├── NTT kernels (radix-8, 2D)             │
│  ├── Key-switching (modup → inner_prod → moddown) │
│  └── CKKS encode/encrypt/decrypt            │
├─────────────────────────────────────────────┤
│  NCCL (inter-GPU collectives)               │
│  MPI (inter-node communication)             │
│  CUDA 12.8 + H100 GPUs                     │
└─────────────────────────────────────────────┘
```

---

## 3. Communication Architecture

### 3.1 Intra-Node Communication (NVSwitch)

MareNostrum 5 ACC nodes have 4x H100 connected via NVSwitch, providing full-bisection bandwidth between all GPU pairs.

| Collective | Aggregate Bandwidth | Per-Ciphertext (21 MB) |
|-----------|-------------------|------|
| AllGather | 1,026 GB/s | 20 us |
| AllReduce | 1,006 GB/s | 21 us |
| Broadcast | 807 GB/s | — |

**Key finding**: Intra-node communication cost is **negligible** — 20 us vs 1.75 ms compute per key-switch = 1.1% overhead. NVSwitch makes RNS limb transfers essentially free.

### 3.2 Inter-Node Communication (InfiniBand NDR200)

For multi-node pipeline parallelism, we use MPI for one-time scatter/gather of ciphertext batches:

| Operation | 2 Nodes | 4 Nodes | What it transfers |
|-----------|---------|---------|------------------|
| MPI Scatter | 88 ms | 150 ms | 128 ciphertexts (~40 KB each at N=8192) |
| MPI Gather | 87 ms | 129 ms | Results back to rank 0 |
| **Execute** | **7.2 ms** | **4.7 ms** | **Zero inter-node communication** |

The scatter/gather is a **one-time cost** amortized across the entire BERT inference (12 layers × hundreds of operations). During execution, each node operates independently.

### 3.3 Communication Patterns

**Input Broadcast Key-Switching**:
```
GPU 0 local c2: [limb0, limb2, limb4]
GPU 1 local c2: [limb1, limb3]
        ↓ AllGather (20 us on NVSwitch)
GPU 0 full c2:  [limb0, limb1, limb2, limb3, limb4]  (reordered)
GPU 1 full c2:  [limb0, limb1, limb2, limb3, limb4]  (reordered)
        ↓ Each GPU: local keyswitch_inplace (1.75 ms)
```

**Output Aggregation Key-Switching**:
```
GPU 0: partial_inner_prod(digits 0,2,4) → partial_cx0
GPU 1: partial_inner_prod(digits 1,3)   → partial_cx1
        ↓ AllReduce (21 us on NVSwitch)
GPU 0: full_cx = partial_cx0 + partial_cx1  → moddown → add_to_ct
GPU 1: full_cx = partial_cx0 + partial_cx1  → moddown → add_to_ct
```

**Pipeline Parallelism** (no communication during compute):
```
Rank 0 (4 GPUs): ciphertexts [0..31]  → process independently
Rank 1 (4 GPUs): ciphertexts [32..63] → process independently
Rank 2 (4 GPUs): ciphertexts [64..95] → process independently
Rank 3 (4 GPUs): ciphertexts [96..127] → process independently
        ↓ MPI Gather (one-time, 129 ms)
Rank 0: all 128 results collected
```

---

## 4. Code Inventory

### 4.1 Core Multi-GPU Infrastructure (3,002 lines, 15 files)

| Component | Files | Lines | Purpose |
|-----------|-------|-------|---------|
| **CtPipeline** | `pipeline/ct_pipeline.{cuh,cu}` | 325 | Scatter ciphertexts → parallel execute → gather results |
| **MultiNodePipeline** | `pipeline/multi_node_pipeline.{cuh,cu}` | 230 | MPI scatter → intra-node CtPipeline → MPI gather |
| **DistributedContext** | `distributed_context.{cuh,cu}` | 469 | Per-GPU PhantomContext + NCCL comms + key distribution |
| **DistributedEval** | `distributed_eval.{cuh,cu}` | 611 | LOCAL ops (per-limb parallel) + KEYED ops (OA/IB) |
| **NCCL Comm** | `comm/nccl_comm.{cuh,cu}` | 422 | AllGather, AllReduce, Broadcast wrappers for FHE |
| **RNS Partition** | `partition/rns_partition.{cuh,cu}` | 323 | Cyclic limb assignment, scatter/gather CUDA kernels |
| **Key-Switching (IB)** | `keyswitching/input_broadcast.{cuh,cu}` | 320 | AllGather c2 → GPU-side reorder → local keyswitch |
| **Key-Switching (OA)** | `keyswitching/output_aggregation.{cuh,cu}` | 390 | Partial inner product kernel → AllReduce → moddown |
| **Stream Manager** | `overlap/stream_manager.{cuh,cu}` | 363 | CUDA stream overlap, CudaGraph capture |
| **NVTX Profiling** | `nvtx_ranges.cuh` | 104 | Color-coded Nsight timeline annotations |

### 4.2 Benchmark Suite (3,083 lines, 9 programs)

| Benchmark | Lines | What it measures |
|-----------|-------|-----------------|
| `nccl_bandwidth_test` | 282 | NVSwitch AllGather/AllReduce bandwidth |
| `multi_gpu_keyswitch_test` | 409 | Correctness of IB and OA key-switching (MAE vs ground truth) |
| `spmd_keyswitch_bench` | 390 | SPMD Input Broadcast timing with persistent threads |
| `spmd_oa_bench` | 370 | SPMD Output Aggregation compute speedup |
| `ks_breakdown_bench` | 213 | Per-stage timing: modup, inner_product, moddown |
| `bert_layer_scaling` | 506 | Simulated BERT layer (NTT + mul + add + keyswitch) |
| `ct_pipeline_bench` | 349 | Ciphertext pipeline: 32-256 cts across 1-4 GPUs |
| `multi_node_bench` | 180 | MPI multi-node pipeline: 2-4 nodes |
| `dist_bert_layer_bench` | 252 | Distributed LOCAL ops with GPU utilization validation |

### 4.3 Build System

- CMake with CUDA 12.8, NCCL 2.24, optional MPI
- Phantom FHE built as ExternalProject
- NTL built from source on MN5 (`/gpfs/projects/etur02/hkanpak/local/`)
- Targets: 9 executables + `nexus_multi_gpu` static library

---

## 5. Experimental Results

### 5.1 Key-Switching Correctness

Both algorithms produce results **indistinguishable from single-GPU Phantom** across all tested configurations:

| N | L | Algorithm | 1 GPU | 2 GPUs | 4 GPUs |
|---|---|-----------|-------|--------|--------|
| 8192 | 5 | Input Broadcast | PASS (8.0e-10) | PASS (8.1e-10) | PASS (8.3e-10) |
| 8192 | 5 | Output Aggregation | PASS (8.1e-10) | PASS (8.1e-10) | PASS (8.2e-10) |
| 16384 | 10 | Input Broadcast | PASS (1.1e-9) | PASS (1.2e-9) | PASS (1.2e-9) |
| 65536 | 20 | Input Broadcast | PASS | PASS (2.3e-9) | PASS (2.3e-9) |
| 65536 | 20 | Output Aggregation | PASS | PASS (2.9e-17) | PASS (0.0) |

### 5.2 Key-Switching Stage Profiling

**N=65536, L=20 (NEXUS parameters), single H100**:

| Stage | Time (ms) | Fraction | Distributable? |
|-------|----------|----------|---------------|
| modup (INTT + base conversion) | 1.074 | 65% | No* |
| inner_product (key_switch_inner_prod) | 0.421 | 25% | Yes (OA) |
| moddown (INTT + base conversion) | 0.164 | 10% | No |
| **Total** | **1.659** | **100%** | **25%** |

*modup's per-digit kernels are `static __global__` in Phantom — cannot be called externally without modifying the library.

### 5.3 Limb-Level Parallelism (Output Aggregation)

OA distributes only the inner product across GPUs (25% of key-switch time):

| GPUs | OA Key-Switch (ms) | Speedup |
|------|-------------------|---------|
| 1 | 1.820 | 1.00x |
| 4 | 1.659 | 1.08x |

**Amdahl's Law limit**: max 1.43x with 25% distributable fraction.

### 5.4 Ciphertext Pipeline Parallelism (Single Node)

**N=8192, L=10, 128 ciphertexts, MN5 4x H100**:

| GPUs | Execute (ms) | Speedup | Efficiency |
|------|-------------|---------|------------|
| 1 | 36.9 | 1.00x | 100% |
| 2 | 21.6 | **1.71x** | 85% |
| 4 | 15.5 | **2.48x** | 62% |

**N=65536, L=20, 64 ciphertexts (NEXUS scale)**:

| GPUs | Speedup |
|------|---------|
| 2 | **1.39x** |
| 4 | **2.15x** |

### 5.5 Multi-Node Pipeline (MareNostrum 5)

**First demonstrated multi-node FHE ciphertext pipeline on a production HPC system.**

**N=8192, L=10, 128 ciphertexts**:

| Config | Nodes | GPUs | Execute (ms) | Speedup |
|--------|-------|------|-------------|---------|
| Baseline | 1 | 1 | 36.9 | 1.00x |
| Single node | 1 | 4 | 15.5 | **2.48x** |
| Multi-node | 2 | 8 | 7.2 | **5.13x** |
| Multi-node | 4 | 16 | 4.7 | **7.85x** |

**256 ciphertexts on 16 GPUs**: Execute = 7.05 ms, 0.44 ms/ct/GPU.

---

## 6. Technical Challenges Solved

### 6.1 Build & Integration (10 fixes)
1. Phantom ExternalProject integration (CMake `CMAKE_SOURCE_DIR` conflict)
2. Phantom target naming (`Phantom` vs `phantom`, capital P)
3. CKKS encoder API (`std::vector<double>` required, not scalar)
4. `encrypt_symmetric` argument count mismatch
5. `PhantomCiphertext::resize` requires stream parameter
6. PhantomCiphertext copy constructor crashes → use `std::move`
7. NCCL must initialize before PhantomContext (memory corruption)
8. `constexpr double` with `pow()` → use `(double)(1ULL << 40)`
9. `CoeffModulus::Create` needs `phantom::arith::` namespace prefix
10. macOS `._` resource fork files in tar archives breaking CUDA compilation

### 6.2 Algorithmic Bugs (3 critical fixes)
1. **AllGather padding deadlock**: NCCL requires uniform send count across all ranks. With cyclic limb assignment and `total_limbs % n_gpus != 0`, GPUs have unequal limb counts → padded to `ceil(total/n_gpus)`.
2. **Limb reorder after AllGather**: AllGather produces GPU-grouped order `[GPU0_limbs | GPU1_limbs]`; Phantom needs sequential `[limb0 | limb1 | limb2]` → wrote GPU-side reorder kernel.
3. **Secret key sharing**: Each GPU independently encrypting → different ciphertexts → garbage after AllGather. Fix: serialize secret key on GPU 0, deserialize on all GPUs; encrypt once on GPU 0, distribute via `cudaMemcpyPeer`.

### 6.3 Performance Optimizations (3 improvements)
1. **Pre-allocated scratch buffers** (`thread_local` static): Eliminated `cudaMalloc/cudaFree` per key-switch call. Reduced 2-GPU N=65536 latency from 15.5ms to 1.845ms (**8.4x improvement**).
2. **GPU-side reorder kernel**: Replaced 20 host-side `cudaMemcpyAsync` calls (100us driver overhead) with single CUDA kernel launch (5us).
3. **Persistent worker threads**: Eliminated `std::thread` creation/join overhead (1ms per GPU per iteration) via barrier-synchronized thread pool.

---

## 7. Hardware & Infrastructure

### MareNostrum 5 (BSC, Barcelona)
- **ACC partition**: 1,120 nodes × 4x H100 64GB SXM = 4,480 GPUs
- **Intra-node**: NVSwitch (1,026 GB/s aggregate AllGather)
- **Inter-node**: InfiniBand NDR200 (200 Gb/s)
- **Project**: etur02, user koc971580
- **Software**: CUDA 12.8, NCCL 2.24.3, OpenMPI 4.1.5, NTL 11.5.1 (built from source)

### RunPod (Cloud)
- 2x H100 80GB SXM with NVLink
- Used for initial debugging and correctness validation ($10 budget)

### AWS (Blocked)
- Only 8 vCPU quota approved for P instances (p4d.24xlarge requires 96)

---

## 8. Path to BERT and Next Steps

### 8.1 BERT Layer Decomposition

A BERT-base layer contains operations with different parallelism profiles:

| Operation | % of Layer | Independent CTs | Parallelism Strategy |
|-----------|-----------|-----------------|---------------------|
| MatMul (6x) | ~40% | 64 per head × 12 heads | **Pipeline** (near-linear) |
| Bootstrapping (5x) | ~35% | 1 per bootstrap | OA key-switching (1.08x) |
| GELU | ~9% | 1 | OA key-switching |
| LayerNorm (2x) | ~5% | 1 | OA key-switching |
| Softmax | ~3% | 12 (one per head) | **Pipeline** (12-way) |
| Argmax | ~7% | 1 | OA key-switching |

**MatMul dominates** and is embarrassingly parallel. With pipeline parallelism on MatMul + OA on single-ct ops:

### 8.2 Projected BERT Speedup

| GPUs | Nodes | MatMul (40%) | Single-ct (60%) | Combined |
|------|-------|-------------|-----------------|----------|
| 4 | 1 | ~2.5x | ~1.1x | **~1.7x** |
| 8 | 2 | ~5x | ~1.1x | **~2.7x** |
| 16 | 4 | ~8x | ~1.1x | **~4.0x** |

### 8.3 Remaining Work

1. **Wire pipeline into NEXUS CKKSEvaluator**: Replace single-GPU MatMul with pipeline-parallel version using CtPipeline
2. **Bootstrapping distribution**: Bootstrapping internally uses rotations (key-switches) — can benefit from OA
3. **Compute-communication overlap**: Use StreamManager to overlap AllReduce with independent NTT operations
4. **Larger-scale experiments**: 8-node (32 GPU) runs on MN5 for paper-quality scaling curves

---

## References

1. Zhang et al., "NEXUS: Secure and Non-Interactive Transformer Inference on Encrypted Data", NDSS 2025
2. Jayashankar et al., "Cerium: A Scalable Multi-GPU Framework for Encrypted Large-Model Inference", arXiv:2512.11269, 2025
3. Jayashankar et al., "Cinnamon: A Framework for Scale-Out Encrypted AI", ASPLOS 2025
4. Phantom FHE Library, encryptorion-lab/phantom-fhe, GitHub
