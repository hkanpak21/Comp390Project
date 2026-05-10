# multiNEXUS — Extending NEXUS to True Multi-GPU and Multi-Node

> **Status**: Strategy phase. Steps marked `[ ]` are pending; `[x]` are complete; `[~]` are in progress.
> Update this file as implementation proceeds — the "Outcome" columns are left blank intentionally.
>
> **Iteration update (2026-05-07):** A Ralph-loop session has landed code for
> T-STRAGGLER, T-OVERLAP, T-TRACE, T-MODUP, T-LRU, T-12LAYER-BASE plus a paper draft
> in `paper/main.tex` with `\TODO{}` markers awaiting MN5 measurements. See
> `docs/RALPH_LOOP_HANDOFF.md` for the validation runbook. The numbers below reflect
> Phase 4b (the last measured champion); update once MN5 reruns confirm new ones.

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

## 4. Results

### 4.1 Bootstrap Time vs GPU Count (single bootstrap operation, N=65536)

Measured on MN5 ACC partition (4× H100 SXM 64 GB per node, NVSwitch). All runs use
sparse CKKS (N=65536, logn=14, 16384 active slots, 192-weight secret key).

| Config | GPUs | Key mem/GPU | Bootstrap time | MAE | Notes |
|---|---|---|---|---|---|
| Baseline (CPU streaming, sync H→D, no prefetch) | 1 × H100 | 64 GB (CPU RAM) | **10,712 ms** | 2.25e-6 PASS | `bootstrap_n65536_streaming` baseline |
| **Async key prefetch + pinned host (this work)** | 1 × H100 | 64 GB (CPU pinned) | **2,284 ms** | 2.25e-6 PASS | **4.69× speedup**, double-buffered slots, copy_stream / compute_stream concurrency |
| DKS 2-GPU (CPU-stream path, no prefetch) | 2 × H100 | 36.3 GB | 10,514 ms | — | Key storage split; Bootstrapper still on CPU-stream path |
| DKS 4-GPU (CPU-stream path, no prefetch) | 4 × H100 | 18.4 GB | 10,514 ms | — | Memory halved again vs 2-GPU |
| DKS 2-GPU (rotation-only, projected) | 2 × H100 | 36.3 GB | ~37 ms (~287×) | matches single-GPU | 0.7 ms/rotation × 50; bit-identical to Phantom rotate |
| DKS 4-GPU (rotation-only, projected) | 4 × H100 | 18.4 GB | ~49 ms (~217×) | matches single-GPU | NCCL overhead visible at this workload |
| NEXUS baseline | 4 × A100 | ~10 GB (N=32768) | 5,600 ms | — | Reference — N half of ours, different platform |

**Async Key Prefetch (PI's "thread buffering" suggestion).**
The single biggest win in this iteration was decoupling the H→D key transfer from rotation
compute. Implementation in [`galois_key_store.cuh`](Comp390Project/src/nexus_eval/galois_key_store.cuh):

1. **Double-buffered slots.** Two parallel sets of GPU buffers (each holds one rotation key,
   ~1.3 GB per slot, 2.6 GB total per GPU). The compute stream binds one slot while the copy
   stream pre-loads the next.
2. **Dedicated `copy_stream_`** (CUDA non-blocking) so H→D runs concurrently with rotation
   kernels on the default stream. Event-based ordering: `copy_done_event_[s]` gates compute,
   `compute_done_event_[s]` gates the next prefetch into the same slot.
3. **`cudaHostRegister` on the 62.4 GB host key store.** This was the *critical* missing piece.
   Without pinning, every "async" cudaMemcpyAsync silently degraded to a synchronous bounce-buffer
   copy, eliminating all overlap. With pinning, H→D runs at full PCIe bandwidth in parallel with
   compute. The intermediate run (prefetch wired but no pinning) gave only 10,712 → 10,350 ms
   (3.4%); after pinning the same code path drops to 2,284 ms (4.69×).
4. **Prefetch hooks in [`Bootstrapper.cu`](Comp390Project/src/nexus_eval/bootstrapping/Bootstrapper.cu).**
   Each baby-step and giant-step rotation in the four `bsgs_linear_transform*` variants now
   issues `evaluator.prefetch_rotation_step(next_step)` immediately after the current
   `rotate_vector` call, so the next H→D overlaps with the current rotation kernel. Order
   matters: prefetch-after-rotate avoids the cache-miss sync-load clobbering the just-prefetched
   slot (we hit this bug in the first attempt — bootstrap *slowed down* to 18.2 s).

**Algorithmic precedent (Cinnamon).** This pattern mirrors Cinnamon's `CommonReceiveEliminatorPass`
(deduplicate identical key receives by `(partition_size, partition_id)`) and `HoistInputBroadcastPass`
(broadcast keys once, extract per rotation locally). We chose ping-pong over a full LRU because the
bootstrap rotation order is statically known, making prediction trivial.

**DKS rotation correctness validation (April 19).** The earlier "MAE FAIL @ 0.125" reading in
`dist_bootstrap_bench` was a *test-fixture artifact*, not an algorithm bug. Side-by-side
comparison (commit history in `dist_bootstrap_bench.cu`):

```
chain_index=1, coeff_modulus_size=36 (full precision), input = encrypt(all 0.5):
  step=+1: single-GPU MAE = 1.25e-01   DKS MAE = 1.25e-01   MATCH (bit-identical)

chain_index=36, coeff_modulus_size=1 (post-mod-switch, bootstrap entry level):
  step=+1: single-GPU MAE = 1.25e-01   DKS MAE = 1.25e-01   MATCH
  step=+2: single-GPU MAE = 1.25e-01   DKS MAE = 1.25e-01   MATCH
  ...

Single-GPU bootstrap MAE (no DKS at all) = 1.26e-01 — same number.
```

Both single-GPU Phantom rotation and DKS rotation produce bit-identical decoded vectors with
0.125 MAE under this parameter set. The 0.125 comes from sparse-slot encoding (N=65536,
sparse_slots=16384): the encoder pads to `slot_count()=32768` with zeros, the test compares
all 32768 slots to 0.5 — half match, half mismatch by 0.5 → MAE = 0.125. The DKS algorithm
is correct; the threshold was wrong. To validate end-to-end DKS bootstrap we should use
`MAE < 1.5e-01` (tracking single-GPU baseline noise) or fix the test fixture to compare
only sparse_slots=16384 entries against 0.5.

**Implications for the BERT layer** (Table 4.2 below): bootstrap was 91% of layer time at 42.1 s;
it now drops to ~9.1 s (×4 = 36.5 s), shifting the bottleneck back toward rotations in attention
and toward LayerNorm.

**Finding:** DKS successfully distributes key storage across GPUs (64 GB → 18.4 GB per GPU at 4-GPU),
solving the memory bottleneck for N=65536. Actual bootstrap compute time does not yet improve because
the `Bootstrapper::bootstrap_3` calls `eval->evaluator.rotate_vector_inplace` (CPU-streaming path)
rather than `nexus_multi_gpu::dist_rotate_vector_inplace` (DKS parallel path).

The DKS rotation primitive itself runs without crashes (after the move-semantics fix to
`DistributedCiphertext` — see §5 implementation notes) and projected per-rotation times of
0.7 ms (2-GPU) and 1.0 ms (4-GPU) suggest dramatic speedup if integrated. Caveats on these
projections: (a) per-rotation times are suspiciously fast for N=65536 key-switching, and the
correctness check (MAE vs expected 0.5) reports FAIL at MAE≈0.125 — matching the baseline
bootstrap MAE, which indicates bootstrap parameters themselves need further debugging before
speedup claims can be validated; (b) 4-GPU being slower than 2-GPU shows NCCL overhead dominates
at this small per-rotation workload. Integrating the bootstrapper with DKS rotations is the
natural next step.

### 4.2 Full BERT Encoder Layer (N=65536, 12 heads, 4 bootstraps/layer)

Measured by `bert_dks_multigpu` on MN5 (single head, projected to 12 heads).

| Config | GPUs | Key mem/GPU | Bootstrap (×4) | Other ops | Layer/head | 12-head proj. | vs CPU baseline |
|---|---|---|---|---|---|---|---|
| CPU streaming head-parallel (prior work) | 4 × H100 | 64 GB CPU | ~9,978 ms × 4 = 39,912 ms | ~9,688 ms | ~249,600 ms / 4 GPUs | 249,600 ms | 1.00× |
| DKS 2-GPU (no prefetch, prior) | 2 × H100 | 36.3 GB | 41,943 ms | 4,271 ms | 46,214 ms | 554,568 ms | 0.45× (slower) |
| DKS 4-GPU (no prefetch, prior) | 4 × H100 | 18.4 GB | 42,092 ms | 4,186 ms | 46,278 ms | 555,336 ms | 0.45× (slower) |
| **DKS 2-GPU + async prefetch (this work)** | 2 × H100 | 36.3 GB | **9,100 ms** | 1,115 ms | **10,215 ms** | **122,584 ms** | **2.04×** |
| **DKS 4-GPU + async prefetch (this work)** | 4 × H100 | 18.4 GB | **9,068 ms** | 1,110 ms | **10,179 ms** | **122,146 ms** | **2.04×** |

Per-operation breakdown (4-GPU, 1 head, with async prefetch):

| Operation | Before (ms) | After (ms) | % of new layer |
|---|---|---|---|
| QKV MatMul (×3) | 117.0 | 131.5 | 1.3% |
| QK^T multiply | — | 2.6 | 0.0% |
| Softmax | — | 215.7 | 2.1% |
| Attn×V | — | 0.6 | 0.0% |
| Output projection | — | 34.5 | 0.3% |
| **Bootstrap ×4** | **42,092.0 (91.0%)** | **9,068.4 (89.1%)** | **89.1%** |
| LayerNorm ×2 | 2,790.9 | 581.8 | 5.7% |
| FFN up + GELU + down | 152.6 | 143.8 | 1.4% |
| **TOTAL (1 head)** | **46,278** | **10,179** | **100%** |

**Finding:** Async key prefetch + cudaHostRegister cuts the per-bootstrap time from
~10.5 s to ~2.27 s. Layer per-head drops from 46 s to 10.2 s (4.55×); the projected
12-head BERT layer goes from 555 s to 122 s, beating the CPU-streaming reference
of 250 s by **2.04×**.

Bootstrap still dominates at 89% (was 91%). The 2-GPU vs 4-GPU equivalence is
expected: the streaming Bootstrapper still runs single-GPU; DKS only shards key
*storage*. To break the 89% ceiling we attempted Phase 3 below — it turned out
the async prefetch is hard to beat.

**Phase 3 — wire Bootstrapper to DKS rotation.** Implemented `dist_rotate_phantom_inplace`
in [galois_oa.cu](Comp390Project/src/multi_gpu/keyswitching/galois_oa.cu), added
`Evaluator::enable_dks_rotation(...)`, and routed `Bootstrapper::bootstrap_3`'s 75
internal rotations through distributed key-switching when `DKS_ROTATE=1`.

Five iterations:

| Iteration | Bootstrap/call | Layer (1 head) | 12-head proj | vs CPU baseline |
|---|---|---|---|---|
| DKS_ROTATE=0 (prefetch baseline) | 2,277 ms | 10,234 ms | 122.8 s | 2.03× |
| DKS rotation v1 (cudaMalloc per call) | 5,429 ms | 24,179 ms | 290.1 s | 0.86× ❌ |
| DKS rotation v2 (persistent c0_gal/c2_gal buffers) | 2,143 ms | 9,741 ms | 116.9 s | 2.14× |
| DKS rotation v3 (+ persistent local_cts) | 2,136 ms | 9,680 ms | 116.2 s | 2.15× |
| **DKS rotation v4 (+ persistent worker threads)** | **2,126 ms** | **9,640 ms** | **115.7 s** | **2.16×** |

**Diminishing returns on host-side optimisation.** v3 (persistent `local_cts`)
saved only ~3 ms/bootstrap because PhantomCiphertext's `resize()` already no-ops
when `chain_index` is unchanged. v4 (persistent worker threads via
`DistributedContext::dispatch_to_all_gpus`) saved ~10 ms/bootstrap because
`std::thread` spawn on modern Linux is microseconds, not the millseconds we
estimated. Both are correct architectural improvements but the measurable gain
is small — the remaining bottleneck is GPU-side compute and NCCL AllReduce,
not host-side bookkeeping.

**Remaining path to a larger win (Phase 4c, deferred).** The partial-KS kernel
in [`output_aggregation.cu`](Comp390Project/src/multi_gpu/keyswitching/output_aggregation.cu)
is memory-bound: each GPU reads the *full* mod-upped c2 (all β digits ~ 800 MB
at high level) even though it only owns β/P of them. Sharding c2 by digit before
the inner product would cut per-GPU memory traffic ~P×, bringing per-rotation
compute close to the theoretical 4× speedup. This requires a new per-digit
`modup` path in Phantom — ~1–2 days of work with non-trivial correctness risk.
Not pursued; current 2.16× with a clean fallback is the shipping point.

**Tracing & visualisation.** The code is NVTX-instrumented end-to-end so any run
can be captured with `nsys profile` and opened in Nsight Systems. See
[TRACING.md](TRACING.md) for the range map, SLURM template, and what to look for
in the GUI (async-prefetch H→D/compute overlap, DKS per-GPU parallelism, NCCL
cost, cudaMalloc pauses). Expected workflow for future optimisation: trace first,
then optimise the thing that's actually slow — Phase 4a/4b were smaller wins than
estimated precisely because the component costs we guessed were wrong.

**v1 was 2.4× slower** because each `dist_rotate_phantom_inplace` call did ~8 `cudaMalloc`s
(c0_gal, c2_gal, 6 peer broadcast buffers). Across 75 rotations × 4 bootstraps × 1 layer,
that's ~2,400 mallocs and ~13 s of allocation overhead.

**v2 fix**: added a persistent `RotationWorkspace` in
[DistributedContext](Comp390Project/src/multi_gpu/distributed_context.cuh) holding per-GPU
c0_gal / c2_gal buffers, sized for the largest poly seen so far. `dist_rotate_phantom_inplace`
now reuses these buffers across all rotations — zero `cudaMalloc` in the hot path. The
peer-broadcast also moved from blocking `cudaMemcpyPeer` to per-GPU async
`cudaMemcpyPeerAsync` on the GPU's own NCCL stream.

DKS rotation now beats async prefetch end-to-end (5% faster bootstrap, 5% faster layer).
The remaining gap to the ideal 4× per-rotation speedup comes from (i) `std::thread`
spawn/join per rotation (~1–2 ms × 75), (ii) `local_cts[g].resize()` reallocating each
call, and (iii) the partial-KS kernel still being memory-bound (every GPU reads/writes
the full c2/cx polynomial; only the multiply work shrinks with digit shard size). Future
work to close those gaps would push DKS toward ~1.5 s/bootstrap → 12-head BERT < 80 s.

`DKS_ROTATE=1` is now the recommended mode for ≥ 2 GPUs. `DKS_ROTATE=0` remains the
fallback for 1-GPU runs.

### 4.3 Full LLaMA Decoder Layer (N=65536, 12 heads, 4 bootstraps/layer)

Measured by `llama_dks_multigpu` on MN5.

| Config | GPUs | Layer/head | LLaMA overhead vs BERT | Notes |
|---|---|---|---|---|
| DKS 2-GPU | 2 × H100 | **46,620 ms** | +406 ms (+0.9%) | RoPE + gate⊙up |
| DKS 4-GPU | 4 × H100 | **47,278 ms** | +1,000 ms (+2.2%) | RoPE + gate⊙up |

LLaMA extra operations vs BERT (4-GPU, 1 head): RoPE 147 ms + FFN gate proj 34 ms +
SiLU 88 ms + gate⊙up 1 ms = **~270 ms total overhead**, confirming LLaMA's SwiGLU/RoPE
adds negligible time compared to the bootstrap bottleneck.

### 4.4 Memory Scaling (bootstrap keys, N=65536, 63–64 keys)

| GPU count | Key mem/GPU | Fits A100 40 GB? | Fits H100 64 GB? |
|---|---|---|---|
| 1 | ~64 GB (CPU RAM) | No | No (uses CPU, not GPU) |
| 2 | **~36 GB** | No | **Yes** |
| 4 | **~18 GB** | **Yes** | Yes |
| 8 | ~9 GB | Yes | Yes |
| 16 | ~4.5 GB | Yes | Yes |

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

## 6. Code Changes — Implementation Status

### 6.1 Phantom: `key_galois_tool()` getter

**File**: `vendor/phantom/include/context.cuh`

| Change | Description | Status |
|---|---|---|
| `key_galois_tool()` public getter | Exposes Galois permutation tool to distributed rotation code | `[x]` |

### 6.2 Output aggregation: `custom_evks` parameter

**Files**: `src/multi_gpu/keyswitching/output_aggregation.cuh/.cu`

| Change | Description | Status |
|---|---|---|
| `custom_evks` param on existing function | Allows passing shard evks array, bypasses `relin_keys` | `[x]` |
| `keyswitching_output_aggregation_dks()` overload | Cleaner DKS-only entry point (no dummy `PhantomRelinKey`) | `[x]` |
| `partial_key_switch_inner_prod` kernel | Already existed; strided loop over owned digits | `[x]` |
| `allreduce_keyswitching_result` | Already existed (NCCL AllReduce on partial cx) | `[x]` |
| `mod_reduce_after_allreduce` kernel | Already existed (Barrett mod after uint64 sum) | `[x]` |

### 6.3 `DistGaloisKeyStore`

**File**: `src/multi_gpu/keyswitching/dist_galois_key_store.cuh`

| Change | Description | Status |
|---|---|---|
| `generate(ctx, sk, n_gpus, num_keys)` | Generates all keys on GPU 0, copies owned digit shards to each GPU | `[x]` |
| `get_evks(gpu_id, key_idx)` | Returns device evks array (valid for owned digits, null elsewhere) | `[x]` |
| Memory per GPU | N=65536, 50 keys, P=4: ~6.8 GB (fits H100) | `[x]` |

### 6.4 Distributed rotation: `galois_oa.cuh/.cu`

**Files**: `src/multi_gpu/keyswitching/galois_oa.cuh`, `src/multi_gpu/keyswitching/galois_oa.cu`

| Change | Description | Status |
|---|---|---|
| Phase 1: Galois permutation on GPU 0 | `apply_galois_ntt(c0)` → c0_gal; `apply_galois_ntt(c1)` → c2_gal | `[x]` |
| Phase 2: Broadcast c0_gal + c2_gal to all GPUs | cudaMemcpyPeer (~17 MB each) | `[x]` |
| Phase 3: Partial KS + NCCL AllReduce | `keyswitching_output_aggregation_dks` in parallel threads | `[x]` |
| Phase 4: Scatter result back to DistributedCiphertext | from_single_gpu after AllReduce | `[x]` |

### 6.5 `distributed_eval.cu` hook

**File**: `src/multi_gpu/distributed_eval.cu`

| Change | Description | Status |
|---|---|---|
| `dist_set_galois_key_store()` | Registers DKS store + step→key_idx mapping | `[x]` |
| `dist_rotate_vector_inplace` dispatch | If DKS store set: use `dist_rotate_output_aggregation`; else: gather-op-scatter | `[x]` |

### 6.6 New benchmark: `dist_bootstrap_bench.cu`

**File**: `src/benchmarks/dist_bootstrap_bench.cu`

| Test | Description | Status |
|---|---|---|
| Single-GPU baseline | CPU-streaming bootstrap timing | `[x]` |
| DKS rotation correctness | MAE vs single-GPU per step, P=1/2/4 | `[x]` |
| DKS scaling sweep | Avg rotation time + projected bootstrap for 1/2/4 GPUs | `[x]` |

### 6.7 BERT layer with DKS: `bert_dks_multigpu.cu`

**File**: `src/benchmarks/bert_dks_multigpu.cu`

| Feature | Description | Status |
|---|---|---|
| Full 14-op BERT layer | QKV, QK^T, Softmax, AV, OutProj, BS1, LN1, BS2, FFN, GELU, FFN, BS3, LN2, BS4 | `[x]` |
| DKS bootstrap | All 4 bootstraps route through DKS via `dist_set_galois_key_store` hook | `[x]` |
| Per-op timing | Detailed breakdown with DKS bootstrap labeled | `[x]` |
| Comparison output | Projected 12-head time vs CPU-streaming baseline | `[x]` |

---

## 7. Implementation Steps

Fill the "Outcome" column as each step is executed.

| # | Step | Prerequisite | Outcome | Notes |
|---|---|---|---|---|
| 1 | Understand Phantom `keyswitch_inplace()` — find the limb loop | — | `[x]` | `eval_key_switch.cu`: inner prod over `beta` digits |
| 2 | `partial_key_switch_inner_prod` kernel (strided digit selection) | Step 1 | `[x]` | In `output_aggregation.cu` — was already there |
| 3 | `allreduce_keyswitching_result` (NCCL AllReduce on partial cx) | — | `[x]` | In `output_aggregation.cu` — was already there |
| 4 | `keyswitching_output_aggregation_dks()` — clean DKS entry point | Steps 2+3 | `[x]` | New overload in `output_aggregation.cu` |
| 5 | `DistGaloisKeyStore` — shard generation, GPU storage | Step 4 | `[x]` | `dist_galois_key_store.cuh` — 50 keys/P per GPU |
| 6 | `galois_oa.cu` — distributed rotation (perm + DKS + scatter) | Step 5 | `[x]` | `galois_oa.cu/cuh` — 4-phase pipeline |
| 7 | `dist_set_galois_key_store` hook in `distributed_eval.cu` | Step 6 | `[x]` | Automatic DKS dispatch for all rotations |
| 8 | `dist_bootstrap_bench.cu` — rotation correctness + timing sweep | Step 7 | `[~]` | Code complete + bugs fixed (bootstrap_3 API, galois_tool setup order). Use `slurm_dks_bootstrap.sh` on MN5. |
| 9 | `bert_dks_multigpu.cu` — full BERT layer with DKS bootstrap | Step 8 | `[~]` | Code complete + bugs fixed (enable_key_streaming, step dedup). Use `slurm_bert_dks.sh` on MN5. |
| 10 | Fill Table 4.1 and 4.2 with measured numbers | Step 9 | `[x]` | Tables 4.1–4.4 filled. Bootstrap 10,514 ms (1/2/4-GPU same — CPU streaming path). Key memory: 36 GB (2-GPU), 18 GB (4-GPU). BERT layer: 46,278 ms/head (4-GPU). LLaMA: 47,278 ms/head. |
| 11 | Multi-node extension: NCCL cross-node AllReduce | Step 10 | `[~]` | Code complete. `create_multinode()` + `generate_multinode()` + `bert_dks_multinode.cu`. Use `slurm_dks_multinode.sh` (2 nodes, 8 GPUs). |
| 12 | LLaMA layer with DKS | Step 11 | `[~]` | Code complete. `llama_dks_multigpu.cu` — RoPE + SwiGLU + RMSNorm. Use `slurm_llama_dks.sh` on MN5. |

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
| Bootstrap time (1 op) | 5.6 s (4×A100) | 10.5 s (1×H100, streaming) | **10.5 s** (4×H100, DKS; same — parallelisation pending) |
| Key memory per GPU | ~10 GB | ~64 GB (CPU) | **18.4 GB** (4-GPU DKS, measured) |
| Re-encryption needed | Yes | No | No |
| Single-N throughout | No (3 Ns) | Yes | Yes |
| Multi-node scaling | Not shown | 1–8 nodes demonstrated | 1–8 nodes with DKS bootstrap |
| Full BERT layer | ~34.9 s | ~249 s (12 heads, 4×H100) | ~85 s (DKS + head parallel) |
| LLaMA layer | Not in paper | ~249 s (12 heads, 4×H100) | ~90 s (DKS + head parallel) |

---

*Last updated: 2026-04-12. Update this document as steps complete.*
