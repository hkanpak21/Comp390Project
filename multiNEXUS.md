# multiNEXUS — Extending NEXUS to True Multi-GPU and Multi-Node

> **Status**: Strategy phase. Steps marked `[ ]` are pending; `[x]` are complete; `[~]` are in progress.
> Update this file as implementation proceeds — the "Outcome" columns are left blank intentionally.

---

## 1. NEXUS GPU Backbone — What We Now Understand

### 1.1 Three-N Architecture

NEXUS is not a single-parameter-set framework. It runs each class of operation at a different polynomial degree N, re-encrypting between them:

```
MatMul         →  N = 8,192   (MM_LOG_N = 13)
Non-linear ops →  N = 65,536  (N = 1 << 16)
Bootstrap      →  N = 32,768  (logN = 15, downgraded from 16)
```

Source evidence:
- `vendor/nexus/cuda/src/main.cu` line 24-26: `N = 1ULL << 16`, `MM_LOG_N = 13`
- `vendor/nexus/cuda/src/bootstrapping.cu` line 28: `long logN = 15; // 16 -> 15`
- The comment is explicit: *"adjusted to satisfy the memory constraints of an A100 GPU"*

Re-encryption between parameter sets requires the secret key on the compute server.
This breaks true client-server non-interactivity, but NEXUS accepts this as a protocol design choice.

### 1.2 Galois Key Memory at Each N

Key size formula (RNS-CKKS):
```
key_size_bytes = total_moduli^2 × 2 × N × 8
```

| Operation | N | Total moduli | Key size | Keys needed | Total GPU memory |
|---|---|---|---|---|---|
| MatMul rotation | 8,192 | 3 | ~3 MB | 13 | **~40 MB** |
| Softmax/LN sum-reduce | 65,536 | 19 | ~379 MB | 7 | **~2.7 GB** |
| Bootstrap (NEXUS) | 32,768 | 31 | ~504 MB | ~20 sparse | **~10 GB** |
| Bootstrap (our N=65536) | 65,536 | 31 | ~1.28 GB | ~50 sparse | **~64 GB** |

**Why NEXUS's 4×A100 works**: At N=32768 with sparse slots, bootstrap keys = ~10 GB.
Each A100 (40 GB) holds one complete copy. Four bootstraps per BERT layer run in parallel, one per GPU.
This is genuine compute parallelism — not memory distribution.

**Why we need CPU streaming**: At N=65536, bootstrap keys = ~64 GB. No single GPU (even H100 80 GB)
holds a full copy after accounting for context and ciphertexts. We stream from CPU pinned RAM.

### 1.3 NEXUS 4-GPU Bootstrap Parallelism (Illustrated)

```
BERT layer has 4 bootstraps: BS1, BS2, BS3, BS4

NEXUS (N=32768, keys fit):
  GPU 0 ──── BS1 ────────────────────────
  GPU 1 ──── BS2 ────────────────────────
  GPU 2 ──── BS3 ────────────────────────
  GPU 3 ──── BS4 ────────────────────────
  Wall time = 1 × bootstrap_time = 5.6s

Our current (N=65536, CPU streaming):
  GPU 0 ──── BS1(stream) ──── BS2(stream) ──── BS3(stream) ──── BS4(stream) ────
  Wall time = 4 × bootstrap_time = 4 × 10.7s = 42.8s   (per head)
```

The 4-GPU head parallelism we implemented distributes 12 heads across GPUs,
but within each GPU the 4 bootstraps are still sequential with streaming overhead.

---

## 2. The Problem We Are Solving

### 2.1 What "true multi-GPU" means here

Our existing multi-GPU work parallelizes at the **head level**:
12 heads split across GPUs, each GPU runs its own independent head. This is embarrassingly parallel
and requires no inter-GPU communication except at gather time. It does NOT:

- Reduce bootstrap time (each GPU still does 4 × 10.7s of streaming bootstrap)
- Reduce memory pressure (each GPU still needs to stream ~64 GB of keys)
- Scale bootstrap computation across GPUs

**True multi-GPU extension** means: **a single bootstrap (or rotation) distributes its
work across multiple GPUs**, so that:
- Memory per GPU scales as `total_key_memory / num_GPUs`
- Compute time scales as `bootstrap_time / num_GPUs` (minus communication overhead)

### 2.2 The Root Cause: Key-Switching is the Bottleneck

Bootstrap = CoeffToSlot (~25 rotations) + ModReduce (polynomial eval) + SlotToCoeff (~25 rotations).
Each rotation = one Galois key-switch. Each key-switch dominates bootstrap time.

Key-switching formula for a ciphertext `(c0, c1)` under Galois element `g`:

```
KS(c1, ksk_g) = sum_{j=0}^{L-1}  c1_j  ⊗  ksk_g[j]    (mod q_0 ... q_L)
```

Where `L = total_moduli` and `ksk_g[j]` is the j-th row of the key-switching key.
This is an inner product over L RNS limbs — **each limb's contribution is independent**.

**This is the parallelization hook**: distribute limbs across GPUs, each GPU computes its
partial sum, then AllReduce.

---

## 3. Distributed Key-Switching Strategy (DKS)

### 3.1 Core Idea (from Jayashankar et al. and Cerium lineage)

Attributed concept from Jayashankar et al. (do not use their code — re-implement from scratch):

> Partition the RNS limbs of each key-switching key across P GPUs.
> Each GPU holds 1/P of the limbs of every key.
> Each GPU computes its partial product independently.
> An AllReduce (sum) across GPUs reconstructs the full key-switch output.

Memory implication:
- Key memory per GPU: `total_key_memory / P`
- At N=65536, 50 bootstrap keys, 4 GPUs: `64 GB / 4 = 16 GB` per GPU → fits H100!
- At 8 GPUs: `8 GB` per GPU → fits A100 40 GB!

Communication cost:
- One AllReduce per rotation: `2 × ciphertext_size × (P-1)/P` data moved
- Ciphertext at N=65536, L=31 limbs: `2 × 65536 × 31 × 8 = ~32 MB`
- NVLink bandwidth (H100 SXM): ~900 GB/s bidirectional
- AllReduce cost: ~32 MB at ~450 GB/s effective = **~0.07 ms per rotation**
- 50 rotations in bootstrap: ~3.5 ms communication overhead total

This is negligible compared to 10.7s bootstrap compute.

### 3.2 Architecture Diagram

```
Bootstrap on P=4 GPUs, each holds limbs {0..7}, {8..15}, {16..23}, {24..30}

For each rotation step:
  GPU 0: compute partial_ks[0..7]   ─────┐
  GPU 1: compute partial_ks[8..15]  ──── AllReduce(sum) ──→ final_ks  → all GPUs
  GPU 2: compute partial_ks[16..23] ────┘
  GPU 3: compute partial_ks[24..30] ────┘

Result: each GPU has the full rotated ciphertext, continues to next rotation.
```

### 3.3 Multi-Node Extension

For multi-node (P GPUs across K nodes):
- Intra-node: NVLink AllReduce (NCCL)
- Inter-node: InfiniBand/Slingshot AllReduce (NCCL cross-node)
- MPI used only for initial key distribution and ciphertext scatter, not for the hot path

```
Node 0 (GPUs 0-3): limbs {0..7}
Node 1 (GPUs 4-7): limbs {8..15}
Node 2 (GPUs 8-11): limbs {16..23}
Node 3 (GPUs 12-15): limbs {24..30}

AllReduce spans all nodes via NCCL (NCCL handles NVLink intra-node + IB inter-node natively).
```

---

## 4. Target Results

### 4.1 Bootstrap Time vs GPU Count (single bootstrap operation, N=65536)

Expected numbers based on roofline estimates. **Outcome column filled as experiments run.**

| Config | GPUs | Memory/GPU | Expected bootstrap time | Actual outcome | Notes |
|---|---|---|---|---|---|
| Baseline (CPU stream) | 1 × H100 | 64 GB (CPU RAM) | 10,730 ms | — | Current implementation |
| DKS 2-GPU | 2 × H100 | ~32 GB | ~5,400 ms | — | 2× compute + comm overhead |
| DKS 4-GPU | 4 × H100 | ~16 GB | ~2,800 ms | — | Target: match NEXUS N=32768 |
| DKS 8-GPU | 8 × H100 | ~8 GB | ~1,500 ms | — | Better than NEXUS |
| DKS 4-node/16-GPU | 16 × H100 | ~4 GB | ~800 ms | — | Multi-node target |
| NEXUS baseline | 4 × A100 | ~10 GB (N=32768) | 5,600 ms | N/A | Reference point |

### 4.2 Full BERT Layer Time (12 heads × 4 bootstraps)

| Config | GPUs | Expected total time | Actual outcome |
|---|---|---|---|
| Current (head parallel, CPU stream) | 4 × H100 | ~249 s | 249.6 s (measured) |
| DKS 4-GPU per bootstrap + 12 head parallel | 4 × H100 | ~85 s | — |
| DKS 4-GPU per bootstrap, 8-node | 32 × H100 | ~25 s | — |
| NEXUS 4-GPU (N=32768, re-encrypt) | 4 × A100 | ~34.9 s | N/A |

### 4.3 Memory Scaling (bootstrap keys only, N=65536, 50 keys)

| GPU count | Key memory per GPU | Fits A100 (40 GB)? | Fits H100 (80 GB)? |
|---|---|---|---|
| 1 | ~64 GB | No | No (barely, with system overhead) |
| 2 | ~32 GB | No | Yes |
| 4 | ~16 GB | Yes | Yes |
| 8 | ~8 GB | Yes | Yes |
| 16 | ~4 GB | Yes | Yes |

---

## 5. Communication Libraries

### 5.1 NCCL (intra-node and inter-node AllReduce)

**Why NCCL, not raw CUDA P2P**:
- NVLink topology is non-trivial to exploit manually (H100 NVSwitch topologies vary)
- NCCL auto-detects NVLink vs PCIe vs IB and picks optimal algorithm
- NCCL AllReduce is ring or tree — both better than naïve point-to-point
- Already in our codebase: `src/multi_gpu/nccl_comm.cuh`

**Usage in DKS**: one `ncclAllReduce` per rotation, on the partial key-switch output
(2 × N × L × 8 bytes ≈ 32 MB at N=65536).

```cpp
// After local partial key-switch on each GPU:
ncclAllReduce(d_partial_ct, d_full_ct,
              2 * N * L,       // element count (uint64_t)
              ncclUint64,
              ncclSum,
              nccl_comm, stream);
```

### 5.2 MPI (inter-node ciphertext scatter, key fragment distribution)

**Why MPI stays**:
- Initial key fragment distribution (key gen on rank 0, scatter limb shards to all ranks)
- Ciphertext scatter at inference start (client → server nodes)
- Ciphertext gather at inference end
- These are one-time or low-frequency ops → MPI overhead acceptable

**Why NOT MPI for the hot path (AllReduce inside bootstrap)**:
- MPI puts data through CPU; NCCL operates GPU-to-GPU natively
- For 50 rotations × 32 MB = 1.6 GB of AllReduce per bootstrap, CPU staging is too slow
- NCCL + CUDA-aware MPI (or NCCL standalone) is the right choice

### 5.3 CUDA Streams

- Each GPU uses its own Phantom default stream (already thread-local in our implementation)
- DKS adds: a separate NCCL stream for AllReduce, synchronized with compute stream via events
- Key insight: local partial compute and AllReduce can overlap with next rotation's key prefetch

---

## 6. Code Changes Required

### 6.1 Phantom modifications (key generation side)

**File**: `vendor/phantom/include/util/encryptionparams.h` and `vendor/phantom/src/secretkey.cu`

| Change | Description | Status |
|---|---|---|
| `gen_galois_key_shard(key, gpu_id, num_gpus)` | Generate only limbs `[gpu_id * L/P .. (gpu_id+1) * L/P - 1]` of a Galois key | `[ ]` |
| `set_key_shard_config(gpu_id, num_gpus)` | Tell the context which limb range this GPU owns | `[ ]` |
| Shard-aware key storage | `PhantomGaloisKey` stores only its shard in GPU memory | `[ ]` |

**Approach**: In Phantom's key generation, the Galois key rows are already indexed by limb `j`.
We generate only the subset `j ∈ [start, end)` and store those. The shard is complete and
self-contained — no modification to how keys are stored on disk.

### 6.2 Phantom modifications (key-switching operation)

**File**: `vendor/phantom/src/evaluator.cu` — `keyswitch_inplace()` or equivalent

| Change | Description | Status |
|---|---|---|
| Partial key-switch kernel | Compute `sum_{j=start}^{end} c1_j ⊗ ksk[j]` for local limbs only | `[ ]` |
| Output: partial ciphertext | Result is a partial sum (not valid ciphertext yet) | `[ ]` |
| AllReduce hook | After partial compute, call NCCL AllReduce to sum across GPUs | `[ ]` |
| Full ciphertext output | After AllReduce, each GPU has the valid rotated ciphertext | `[ ]` |

The kernel modification is localized: existing key-switch code loops `j = 0..L-1`.
We simply change the loop bounds to `j = start..end` and add an AllReduce after the loop.

### 6.3 GaloisKeyStore — distributed variant

**File**: `src/nexus_eval/galois_key_store.cuh` (new variant: `dist_galois_key_store.cuh`)

| Change | Description | Status |
|---|---|---|
| `DistGaloisKeyStore` class | Stores only local limb shard in GPU memory | `[ ]` |
| `generate_shard(rot_idx)` | Generates key shard for rotation `rot_idx`, stores in GPU | `[ ]` |
| Remove CPU streaming path | With shards fitting in GPU, no PCIe streaming needed | `[ ]` |
| Pre-generate all shards | Bootstrap's ~50 keys pre-loaded at startup | `[ ]` |

### 6.4 CKKS Evaluator hook

**File**: `src/nexus_eval/ckks_evaluator.cuh`

| Change | Description | Status |
|---|---|---|
| Replace `ensure_key_loaded()` | Remove CPU→GPU streaming; key shard already in GPU | `[ ]` |
| `distributed_rotate()` | Calls partial key-switch + NCCL AllReduce | `[ ]` |
| `distributed_bootstrap()` | Bootstrap using `distributed_rotate()` for all 75 rotations | `[ ]` |

### 6.5 New benchmark: `dist_bootstrap_bench.cu`

**File**: `src/benchmarks/dist_bootstrap_bench.cu`

| Test | Description | Status |
|---|---|---|
| Single rotation DKS | One rotation across P GPUs, verify correctness + time | `[ ]` |
| Full bootstrap DKS | 50-rotation bootstrap across P GPUs | `[ ]` |
| Scaling sweep | 1/2/4/8 GPUs, report bootstrap time | `[ ]` |
| MAE check | Ensure bootstrap accuracy unchanged vs single-GPU | `[ ]` |

### 6.6 BERT layer with DKS

**File**: `src/benchmarks/bert_dks_multigpu.cu`

- Replace per-head GPU assignment with per-bootstrap DKS assignment
- Option: hybrid — DKS for bootstrap, head-parallel for MatMul/GELU
- Target: use all 4 GPUs for each bootstrap (not 3 heads per GPU)

---

## 7. Implementation Steps

Fill the "Outcome" column as each step is executed.

| # | Step | Prerequisite | Outcome | Notes |
|---|---|---|---|---|
| 1 | Understand Phantom `keyswitch_inplace()` — find the limb loop | — | — | Read `vendor/phantom/src/evaluator.cu` |
| 2 | Implement single-GPU partial key-switch (local limb range) | Step 1 | — | No AllReduce yet; verify partial output |
| 3 | Implement NCCL AllReduce for ciphertext-size buffer | — | — | Test standalone: random uint64 buffer allreduce |
| 4 | Wire partial key-switch + AllReduce into one `distributed_rotate()` | Steps 2+3 | — | Test: rotate by 1, verify decrypted result |
| 5 | Build `DistGaloisKeyStore` — shard generation, GPU storage | Step 4 | — | Confirm memory usage = total/P per GPU |
| 6 | Replace `ensure_key_loaded` hook with distributed path | Step 5 | — | Run existing bootstrap test, check MAE |
| 7 | `dist_bootstrap_bench.cu` — single bootstrap, 2 GPUs | Step 6 | — | Target: <5.4s on 2×H100 |
| 8 | Scaling sweep: 1/2/4/8 GPUs, log bootstrap time | Step 7 | — | Fill Table 4.1 above |
| 9 | BERT layer with DKS — correctness first | Step 8 | — | Run single inference, check output MAE |
| 10 | BERT DKS performance — sweep configs, fill Table 4.2 | Step 9 | — | Target: beat NEXUS 34.9s at N=65536 |
| 11 | Multi-node extension: NCCL cross-node AllReduce | Step 10 | — | MN5, 2-4 nodes, 16-32 GPUs |
| 12 | LLaMA layer with DKS | Step 11 | — | Structural extension of BERT DKS |

---

## 8. Ideas from Jayashankar et al.

> This section records the conceptual insights from Jayashankar et al. that we are
> implementing from scratch (not reusing their code).

| Concept | Description | How we adapt it |
|---|---|---|
| Limb-level parallelism | Key-switching inner product over RNS limbs is parallelizable | Core of our DKS strategy (Section 3) |
| Shard-local GPU execution | Each GPU computes a partial sum without needing other GPUs' keys | Eliminates CPU streaming for the hot path |
| AllReduce for reconstruction | A single collective op after partial KS reconstructs the full CT | We use NCCL; they may use MPI or custom IB primitives |
| Pre-distribution of key shards | Keys distributed once at setup, not at query time | Our `DistGaloisKeyStore` pre-loads shards at startup |
| *(Additional insights)* | *(Fill as we read the paper more carefully)* | — |

---

## 9. What We Are NOT Changing

| Thing | Why left unchanged |
|---|---|
| N = 65536 throughout | No parameter switching; true end-to-end at single N |
| Re-encryption | None needed; DKS works within same parameter set |
| Sparse slots (logn=13) | Keeps bootstrap keys at minimum count; DKS halves their per-GPU size |
| MPI for ciphertext scatter | Works fine for low-frequency ops; no reason to change |
| Head-level parallelism | DKS is orthogonal — can combine DKS-bootstrap with head-parallel MatMul |
| BERT 14-op sequence | All operations remain; only the bootstrap implementation changes |

---

## 10. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Phantom evaluator is not easily modifiable for partial KS | Medium | High | May need to fork eval kernel; localized change |
| NCCL AllReduce overhead exceeds compute gain at small GPU count | Low | Medium | Measure: expect <5ms per rotation at NVLink speeds |
| Numerical precision shifts after AllReduce (floating point order) | Low | Medium | Use uint64 integer arithmetic; CKKS is over integers mod q, not floats |
| MN5 NCCL cross-node slower than expected (Slingshot vs IB) | Medium | Medium | Profile with NCCL bandwidth test first |
| DKS and head-parallel interaction causes incorrect result | Low | High | Test DKS in isolation first (single bootstrap, no head splitting) |
| Memory for key shards + ciphertexts exceeds H100 at some config | Low | Medium | Track live: add memory reporting to benchmark output |

---

## 11. Comparison Anchor: Where We Stand Now vs Where We Aim

| Metric | NEXUS (paper) | Our current | multiNEXUS target |
|---|---|---|---|
| Bootstrap N | 32,768 | 65,536 | 65,536 |
| Bootstrap time (1 op) | 5.6 s (4×A100) | 10.7 s (1×H100, streaming) | <3 s (4×H100, DKS) |
| Key memory per GPU | ~10 GB | ~64 GB (CPU) | ~16 GB (4-GPU DKS) |
| Re-encryption needed | Yes | No | No |
| Single-N throughout | No (3 Ns) | Yes | Yes |
| Multi-node scaling | Not shown | 1–8 nodes demonstrated | 1–8 nodes with DKS bootstrap |
| Full BERT layer | ~34.9 s | ~249 s (12 heads, 4×H100) | ~85 s (DKS + head parallel) |
| LLaMA layer | Not in paper | ~249 s (12 heads, 4×H100) | ~90 s (DKS + head parallel) |

---

*Last updated: 2026-04-12. Update this document as steps complete.*
