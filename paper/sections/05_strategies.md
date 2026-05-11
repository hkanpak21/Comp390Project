# Section 5 — Multi-GPU Strategies

> Status: draft v1
> Slice: WRITE-S5
> Depends-on: none (architectural, no measurements)

NEXUS [Zhang et al., NDSS 2025] publishes per-operation CKKS kernels but ships no multi-GPU framework: there is no `cudaSetDevice`, NCCL, MPI, or `std::thread` anywhere in `vendor/nexus/cuda/`. To run a full BERT inference under encryption on multiple GPUs, we built three orthogonal parallelization strategies on top of Phantom's GPU-native CKKS, each targeting a different figure of merit. This section defines the three strategies and the framework that supports them; measurements live in Sections 6 and 7.

## 5.1 Strategy taxonomy

Three strategies, three figures of merit:

| Strategy | Parallelizes | Figure of merit | Scaling regime |
|---|---|---|---|
| **DKS** (Distributed Key-Switching) | One bootstrap across $N$ GPUs | Per-call latency of the bootstrap | Strong |
| **HP** (Head-Parallel BERT) | One BERT inference across $N$ GPUs by attention head | Per-inference latency | Strong |
| **DP** (Data-Parallel per-operation) | $N/G$ independent op calls per GPU | Aggregate throughput | Weak |

The three strategies are not mutually exclusive: a single end-to-end inference can use different strategies for different operations. Section 6 develops a per-operation typology that chooses, for each NEXUS evaluator, which of these three is the best fit at 4-GPU and 16-GPU scales. Section 7 then uses HP at the BERT level for the strong-scaling latency story and DP at the inference level for the weak-scaling throughput story.

## 5.2 DKS — Distributed Key-Switching

**Target.** A single bootstrap call, parallelized across $G$ GPUs.

**Why bootstrap.** Bootstrap is the costliest NEXUS operation (≈250 ms at $\log N = 15$, ≈300 ms at $\log N = 16$ on H100) and dominates end-to-end inference time. Internally it is a chain of rotations and ciphertext-plaintext multiplications; each rotation invokes a key-switch, the second hottest CKKS kernel after NTT.

**Mechanism.** A key-switch decomposes its input ciphertext into $\beta$ digits ($\beta = \lceil \text{size\_Ql} / \alpha \rceil \le \texttt{dnum}$), mod-ups each digit to the extended chain $\mathit{QlP}$, computes an inner product with the evaluation key, and mod-downs back. DKS splits the $\beta$-digit inner product across GPUs: each GPU computes its partial sum and one `ncclAllReduce` over the $\mathit{QlP}$-basis output combines the partials before the local mod-down. Phantom's `key_switch_inner_prod_c2_and_evk` was modified to accept a digit range — the only Phantom-internal modification DKS requires (`src/multi_gpu/keyswitching/output_aggregation.cu:85`).

**Memory sharding.** The same digit ownership pattern is used to shard the bootstrap Galois keys themselves. Without sharding, the 50 rotation keys consumed by a single bootstrap at $\log N = 16$ are approximately 62 GB on the GPU — larger than the 64 GB capacity of a single H100. With DKS sharding across 4 GPUs, each GPU holds only its own digit shard (≈6.8 GB per GPU at $N = 65536$, $\beta = 9$, $G = 4$; see `src/multi_gpu/keyswitching/dist_galois_key_store.cuh:8`). DKS is therefore the *enabling* strategy for $\log N = 16$ on a single node, not merely the *accelerating* one.

**Ownership pattern.** Digit ownership is **STRIDED**, not contiguous: GPU $g$ owns $\{d : d \bmod G = g\}$. An earlier contiguous layout deadlocked when the runtime chain level dropped below $\texttt{dnum}$: the key-switch only iterates the prefix $d \in [0, \beta)$, so trailing GPUs' contiguous shards covered digits never accessed while leading GPUs were asked for null slots. The visible failure mode was a `cudaFreeAsync` invalid-argument cascade and NCCL P2P illegal-memory-access (`src/multi_gpu/keyswitching/dist_galois_key_store.cuh:19`). STRIDED ownership guarantees *any* prefix $[0, \beta)$ is covered. This is captured in non-negotiable lesson #6 in `CLAUDE.md`.

**Role in the paper.** DKS is the strong-scaling story *for the bootstrap operation* in Section 6 and the path that makes $\log N = 16$ runnable at all. It is not the headline framework for end-to-end inference; that is HP-BERT.

## 5.3 HP-BERT — Head-Parallel BERT

**Target.** One BERT-base inference, parallelized across $G$ GPUs for per-inference latency.

**Mechanism.** BERT-base has 12 attention heads × 12 encoder layers. HP-BERT distributes heads across GPUs end-to-end: GPU $g$ owns a subset of the 12 heads and runs *all 12 layers* locally for them. Each GPU holds, pinned to its device, its share of the Galois keys, plaintext encodings of its heads' projection weights, and a private `Bootstrapper`. Only the activations flow between layers across GPUs; the (large) weights and keys never move after initialization.

**Implementation.** On a single node, HP-BERT uses one `std::thread` per GPU, each bound to its device with `cudaSetDevice` at thread startup (`src/benchmarks/bert_hp_multigpu.cu:543`); each thread instantiates its own `PhantomContext`, `Bootstrapper`, and key store. We avoid MPI on single-node because Phantom's CKKS context is thread-safe under our usage (verified by `phantom_threadsafe_smoke.cu` at MAE $= 0$). For multi-node (4 nodes × 4 GPUs = 16 GPUs), we use one MPI rank per node and four worker threads per rank; NCCL communicators span all 16 GPUs (`src/multi_gpu/distributed_context.cuh:83`).

**Thread-safety prerequisites.** Per-thread `PhantomContext` only works because we made Phantom's `default_stream` `thread_local` (Appendix A, Phantom modification #3) and removed a hard-coded `cudaSetDevice(0)` in the stream constructor (modification #4). Both were prerequisite bugs we fixed in `vendor/phantom/`.

**Role in the paper.** HP-BERT is the strong-scaling story *for end-to-end BERT inference* in Section 7 (Goal 2). It is the only strategy in this paper that reduces the wall-clock latency of a single inference by adding GPUs.

## 5.4 DP — Data-Parallel per-operation

**Target.** A batch of independent op calls (different ciphertexts, different output columns, different GELU evaluations), parallelized across $G$ GPUs for aggregate throughput.

**Mechanism.** Each GPU runs a thread that owns a private `PhantomContext` and processes $N/G$ independent calls of the *same* operation with no inter-GPU communication during the call. At 4-GPU we run one process with four threads; at 16-GPU we run $4 \text{ ranks} \times 4 \text{ GPUs}$ with one MPI rank per node and `std::thread`-per-GPU inside each rank.

**Trade-off.** DP does *not* reduce the per-call latency: a single bootstrap, a single GELU, a single softmax still takes the same wall-clock time it took on one GPU. Only the *aggregate* throughput improves, and only when there are at least $G$ independent calls available. For operations whose per-call latency is dominated by framework overhead (per-rank `PhantomContext` setup, Galois key staging), DP at small $G$ may even underperform single-GPU on a per-call basis; this is why Section 6's per-op typology distinguishes "compute-parallel" ops from "data-parallel-throughput" ops.

**Role in the paper.** DP is the per-op benchmark framework used for the NEXUS-on-H100 vs multiNEXUS comparison in Section 6, and it is also the weak-scaling throughput story in Section 7 (running $G$ concurrent independent BERT inferences, one per GPU).

## 5.5 Framework infrastructure

All three strategies share a small body of infrastructure in `src/multi_gpu/`.

**`DistributedContext`** (`src/multi_gpu/distributed_context.cuh:66`) owns one `PhantomContext` per GPU, the NCCL communicators across all GPUs (single- or multi-node), per-GPU CUDA streams, the per-GPU `GpuKeySet` for sharded keys, and the worker-thread pool used by DKS rotations. It also exposes the limb-partitioning helpers used by `DistributedCiphertext` for RNS-limb sharding.

**`RotationWorkspace`** (`src/multi_gpu/distributed_context.cuh:129`) is a per-GPU device-buffer pool allocated once at setup and reused across DKS rotations. It holds `c0_gal` and `c2_gal` broadcast buffers and persistent per-GPU local `PhantomCiphertext` slots reallocated only when `chain_index` changes. This is a direct application of non-negotiable lesson #2 in `CLAUDE.md`: every per-call `cudaMalloc` we removed in Phase 3 was directly observable as ≈2× slowdown in the per-rotation timing.

**Persistent worker threads** (`src/multi_gpu/distributed_context.cuh:151`): one worker per GPU is spawned at `DistributedContext::create()` time, pinned to its device via `cudaSetDevice` on thread startup, and parked on a condition variable. `dispatch_to_all_gpus` submits one lambda per GPU and joins on completion, eliminating the `std::thread` spawn/join overhead that would otherwise be paid per rotation.

**Context destruction discipline.** `DistributedContext::destroy()` calls `release()` on GPU $1..N-1$ contexts and only fully destroys GPU 0's context (`src/multi_gpu/distributed_context.cu:410`). Phantom's `PhantomContext` and `PhantomCiphertext` destructors call `cudaFreeAsync` on a stream captured at construction; for GPU $> 0$ contexts created on non-primary devices, that captured stream is destroyed before the destructor runs, producing an `cudaFreeAsync` on an invalid stream. Releasing instead of destroying skips the destructor entirely on the non-primary contexts. This is non-negotiable lesson #4 in `CLAUDE.md`.

**Async H→D discipline.** Every host buffer used as a source for `cudaMemcpyAsync` is registered with `cudaHostRegister` before the first copy (`src/nexus_eval/galois_key_store.cuh`). Without this, `cudaMemcpyAsync` from pageable host memory is silently synchronous and the entire prefetch / compute overlap collapses to serial execution. This is non-negotiable lesson #1.

**Explicit scale management.** Our Phantom fork has the scale-mismatch validation in `sub_inplace`, `multiply_plain_inplace`, and `add_plain_inplace` commented out (Appendix A, Phantom modification #5) so that lazy rescaling inside bootstrap is permitted. The consequence is that small numerical scale drift can accumulate across chained operations without triggering a runtime error. We re-establish ground truth with an explicit `x.scale() = SCALE` reset before bootstrap on every chained path (most visibly inside QuickMax's argmax loop). This is non-negotiable lesson #7.

## 5.6 Strategy → op matchup (preview)

The per-op typology of Section 6 sorts NEXUS's evaluators into three buckets along the dimension *which of {DKS, HP, DP} delivers the best multi-GPU speedup at $G \in \{4, 16\}$ on H100*:

- **Compute-parallel ops** (MatMul): an output-channel split similar in shape to DKS — but applied to the matrix's output dimension rather than the key-switch's digit dimension — scales near-linearly because each GPU performs a disjoint slice of the arithmetic.
- **Transitional ops** (GELU, LayerNorm): per-call compute is large enough (tens of milliseconds at $\log N = 16$) to absorb the per-rank `PhantomContext` setup overhead, so DP at 4-GPU is the right choice.
- **Data-parallel-throughput ops** (Bootstrap, Softmax, Argmax-at-4-GPU): per-call latency is comparable to or below the per-rank context-setup ceiling, so DP yields throughput rather than latency. Bootstrap separately gets the DKS treatment when single-bootstrap latency matters.

This typology — heterogeneous, not uniform — is the headline contribution of Section 6.

## 5.7 What we do NOT use (and why)

Three strategies that we explicitly do not include in this paper, with the rationale:

- **Layer-pipeline parallelism** (different transformer layers on different GPUs): a fundamentally different parallelization axis from head-parallel. Combining the two is interesting but outside the scope of a single-axis study.
- **Ciphertext pipeline parallelism** (the older `CtPipeline` infrastructure): archived to `src/multi_gpu/archive/pipeline/`. `CtPipeline` predates the per-thread `PhantomContext` pattern and interacts badly with Phantom's `thread_local default_stream`, funnelling all GPU work through a shared stream that serializes what the per-thread model runs in parallel. Dependent benchmarks live in `src/benchmarks/archive/`.
- **Slot-axis SIMD packing for HP-BERT**: NEXUS's published 37.3 s end-to-end on 4× A100 depends on Algorithm 3, which packs all 12 attention heads into the slot axis of one ciphertext, reducing the bootstrap count from $4h$ per layer to 4 per layer. Implementing this is a multi-day refactor of the chained pipeline and is required to beat NEXUS's end-to-end number on a fair workload. It is listed as a follow-up in `docs/PI_REPORT.md` "What is left", explicitly out of scope here.

The boundary is deliberate: this paper isolates the three-strategy axis (DKS / HP / DP) on a chained pipeline that NEXUS does not ship, so the multi-GPU contribution can be measured cleanly against a like-for-like single-GPU H100 baseline.
