# multiNEXUS — Next Phase PRD

**Purpose.** This document drives the ralph loop. Each task below is self-contained
and executable. Dependencies are explicit so unrelated tasks run in parallel.
Results from each task go into the paper at `paper/main.tex`.

---

## What Has Been Done (Context for Every Task)

We started with a problem: running CKKS bootstrapping at ring degree N=65536 on a
single GPU is impossible because the ~62 GB of rotation keys do not fit in GPU memory.
The original approach was to keep all keys on CPU pinned RAM and stream them to the GPU
one at a time before each rotation (10.7 seconds per bootstrap).

We built multiNEXUS on top of the Phantom FHE library. Below is the engineering
history in order. Every agent should read this before touching any code.

| Phase | What Changed | Bootstrap Time | BERT 12-head | Lesson |
|---|---|---|---|---|
| Start — CPU streaming | Keys on CPU, one slot on GPU, sync H→D | 10,712 ms | ~428 s | Baseline |
| DKS storage only | Split 62 GB keys across 4 GPUs (18.4 GB each). Compute path unchanged. | 10,514 ms | ~421 s | Splitting keys does nothing without compute changes |
| Async prefetch + pinned host | Two GPU buffer slots, async H→D on separate stream, `cudaHostRegister` on 62 GB host store | 2,284 ms | ~91 s | Without `cudaHostRegister`, async copies silently run synchronously. This one call was the 4.69× win. |
| DKS rotation (RotationWorkspace) | Route all bootstrap rotations through distributed key-switching. Persistent workspace eliminates per-call `cudaMalloc`. | 2,143 ms | ~86 s | Per-call `cudaMalloc` was 2.4× slower (2,400 mallocs per layer). Persistent buffers are mandatory in hot paths. |
| Persistent worker threads (champion) | Replace per-call `std::thread` spawn with persistent threads | 2,126 ms | **115.7 s** | Thread spawn on Linux is microseconds. Saved only 10 ms. |

**Current champion: Phase 4b — 2.16× over CPU baseline (115.7 s vs 249.6 s projected 12-head BERT).**

The champion was profiled with Nsight Systems. Here is where the 2,126 ms goes:

| Component | Time | % | Status |
|---|---|---|---|
| NTT kernels (all GPUs, redundant) | ~850 ms | 40% | Every GPU runs NTT on all digits; only uses 1/4 of them. Main target for T-MODUP. |
| NCCL AllReduce straggler wait (host jitter) | ~530 ms | 25% | Not real compute. GPUs wait for the slowest to launch. Target for T-STRAGGLER. |
| NCCL AllReduce kernel (real comm) | ~291 ms | 14% | Actual NVLink transfer. Cannot be eliminated; can be hidden. Target for T-OVERLAP. |
| Mod-up / mod-down / rescale / BSGS scalar ops | ~298 ms | 14% | Coarse bucket. Need finer trace to split BSGS from modup/moddown. Target for T-TRACE. |
| Partial key-switch inner product (all 4 GPUs) | ~142 ms | 6.7% | Already parallelised correctly. Not a target. |

**The no-single-kernel-over-3% finding.** Nsight shows no individual CUDA kernel
exceeds 3% of GPU time. This is a breadth-first workload with ~10^5 tiny kernel
launches per bootstrap. Individual kernel optimisation is the wrong approach; the
wins are in scheduling (straggler, overlap) and algorithmic restructuring (per-digit
modup, BSGS parallelism).

---

## Target Results

What we aim to report in the paper after all tasks complete.

| Configuration | Bootstrap | BERT 12-head | vs CPU Baseline | Notes |
|---|---|---|---|---|
| Phase 4b — current champion | 2,126 ms | 115.7 s | 2.16× | Measured |
| + Straggler fix (T-STRAGGLER) | ~1,600 ms | ~87 s | ~2.87× | Removes 530 ms host jitter |
| + AllReduce overlap (T-OVERLAP) | ~1,450 ms | ~79 s | ~3.16× | Hides ~150 ms AllReduce behind modup |
| + Per-digit modup (T-MODUP) | ~960 ms | ~52 s | ~4.80× | Removes redundant NTT from 3/4 of GPUs |
| + BSGS parallelism (T-BSGS, conditional) | ~900 ms | ~49 s | ~5.10× | Only if T-TRACE shows BSGS > 5% |
| **NEXUS** (Zhang et al., NDSS 2025) | 5,600 ms† | 37.3 s† | — | †N=32,768 with re-encryption between parameter sets |
| **Cerium** (Jayashankar et al., 2025) | 7.5 ms‡ | 8.8 s‡ | — | ‡8× B200, sparse polynomial, code not yet public |
| Cinnamon (Jayashankar et al., ASPLOS 2025) | — | 1.67 s | — | Architectural simulation, not real hardware |

**Footnotes for the paper.**
† NEXUS bootstraps at N=32,768 then re-encrypts to N=65,536. Smaller N means
4× fewer key bytes per key and 2× shorter NTT. Their BERT number uses this
multi-N protocol. We use N=65,536 throughout with no re-encryption.
‡ Cerium's 7.5 ms is on B200 GPUs (roughly 2–3× faster memory bandwidth than H100)
with a sparse polynomial representation that reduces effective problem size.
Code not released as of 2026-05; numbers cited from their arXiv preprint.

**What this means.** After T-STRAGGLER + T-MODUP our BERT 12-head projection (~52 s)
is within 40% of NEXUS at a strictly harder setting (larger N, single-N throughout,
no re-encryption). The remaining gap is hardware (H100 vs A100) and protocol.
Cerium is not a direct competitor — it requires a full compiler stack and
purpose-built scheduling passes that took a year to build.

---

## Tasks

Each task lists: what to do, which files to touch, what benchmark to run,
how to measure success, and where results go in the paper. Measurement protocol
throughout: 5 independent runs, report mean ± standard deviation. For benchmarks
where a single run exceeds 60 seconds, 3 runs is acceptable.

---

### T-TRACE — Granular Profiling Trace
**Status:** open  
**Depends on:** nothing  
**Can run in parallel with:** T-STRAGGLER, T-MODUP, T-NEXUS, T-12LAYER-BASE

**What to do.**
Run a fresh Nsight Systems trace of the Phase 4b champion with additional NVTX
instrumentation that breaks the coarse 14% "BSGS scalar ops + modup + moddown"
bucket into its individual components. The goal is to know exactly what fraction
of bootstrap time is BSGS baby steps vs modulus raising vs modulus lowering.

**Files to modify.**
- `src/nexus_eval/bootstrapping/Bootstrapper.cu` — add NVTX push/pop around
  each baby-step multiply loop and each giant-step accumulation loop inside
  `bsgs_linear_transform_*`. Name them `bsgs_baby_step i=N` and `bsgs_giant_step i=N`.
- `src/multi_gpu/keyswitching/output_aggregation.cu` — add NVTX push/pop around
  the `rns_tool.modup(...)` call (label `modup`) and the `moddown_from_NTT` call
  (label `moddown`).

**How to run on MN5.**
```bash
ssh mn5-gpu
cd /gpfs/projects/etur02/hkanpak/Comp390Project
sbatch scripts/mn5/slurm_trace_nsys.sh   # already exists, runs DKS_ROTATE=1
# Fetch trace:
scp mn5-gpu:/gpfs/projects/etur02/hkanpak/Comp390Project/traces/trace_dksrot.nsys-rep ~/nexus-traces/
# Get text summary:
nsys stats --report nvtxsum trace_dksrot.nsys-rep | grep -E "bsgs|modup|moddown"
```

**Success criteria.**
- New NVTX ranges `bsgs_baby_step`, `bsgs_giant_step`, `modup`, `moddown` appear in
  the `nvtxsum` output.
- The four new ranges plus the existing ones account for ≥ 90% of bootstrap wall time.
- Report the ms and % for each new range.

**Downstream gate.**
If `bsgs_baby_step + bsgs_giant_step` total > 5% of bootstrap time → T-BSGS is
worth implementing. If < 5% → mark T-BSGS as "not pursued: profiling shows < 5%"
and document that finding in the paper's Performance Ceiling section.

**Where results go in paper.**
Section 5 (Evaluation), subsection "Where the time goes" — update the profiling
breakdown table with the finer numbers.

---

### T-STRAGGLER — Fix NCCL Launch Jitter
**Status:** open  
**Depends on:** nothing  
**Can run in parallel with:** T-TRACE, T-MODUP, T-NEXUS, T-12LAYER-BASE

**What to do.**
Add CUDA event barriers before each `ncclAllReduce` call so all 4 GPU streams arrive
at the AllReduce at the same time. Currently the AllReduce kernel itself takes 291 ms,
but the wall time is ~820 ms (291 ms kernel + 530 ms straggler wait) because some GPUs
finish their partial key-switch later than others and NCCL cannot start until all
4 GPUs call `ncclAllReduce`. The fix is GPU-side synchronisation, not CPU-side.

**What straggler wait means concretely.**
Each GPU runs `partial_key_switch_inner_prod` on its own CUDA stream. If GPU 0
finishes in 100 ms and GPU 3 finishes in 220 ms, the NCCL AllReduce cannot start on
GPU 0's stream until GPU 3 also calls `ncclAllReduce`. GPU 0 sits idle for 120 ms.
Multiplied across 75 rotations per bootstrap, this adds ~530 ms to bootstrap wall time.

**Implementation.**
1. Add a `ready_event` array to `MultiGpuContext` — one `cudaEvent_t` per GPU.
   Create these events in `DistributedContext::init()` with `cudaEventCreateWithFlags(...,
   cudaEventDisableTiming)` (no-timing events have lower overhead).
2. In `keyswitching_output_aggregation_dks()`, after the `partial_key_switch_inner_prod`
   kernel launch, record this GPU's event:
   ```cpp
   cudaEventRecord(ctx.ready_events[gpu_id], ctx.streams[gpu_id]);
   ```
3. Before `allreduce_keyswitching_result(...)`, make every stream wait for all
   other GPUs' events:
   ```cpp
   for (int g = 0; g < n_gpus; g++)
       cudaStreamWaitEvent(ctx.streams[gpu_id], ctx.ready_events[g], 0);
   ```
4. No change needed to the AllReduce call itself.

**Files to modify.**
- `src/multi_gpu/distributed_context.cuh` — add `std::vector<cudaEvent_t> ready_events`
- `src/multi_gpu/distributed_context.cu` — init/destroy ready_events
- `src/multi_gpu/keyswitching/output_aggregation.cu` — add record + waitEvent calls

**Benchmark and measurement.**
```bash
make -j20 bert_dks_multigpu
sbatch scripts/mn5/slurm_bert_dks.sh   # run 5 times, record bootstrap/call from output
# Also run dist_bootstrap_bench to measure isolated AllReduce timing:
sbatch scripts/mn5/slurm_dks_bootstrap.sh
```
Before and after: capture Nsight trace to verify straggler gap shrinks.

**Success criteria.**
- `dist_bootstrap_bench` AllReduce wall time: mean ± std < 400 ms (was ~820 ms).
- `bert_dks_multigpu` bootstrap/call: mean < 1,700 ms (was 2,126 ms).
- MAE unchanged at 2.25e-6.

**Where results go in paper.**
Section 5 (Evaluation), subsection "Optimization Steps" — add a row to the
bootstrap evolution table. Also Section 6 (Discussion), as an example of how
host-side scheduling wastes compute time.

---

### T-OVERLAP — Overlap AllReduce with Next Rotation's Modup
**Status:** open  
**Depends on:** T-STRAGGLER (event patterns are established there; this extends them)  
**Can run in parallel with:** T-MODUP, T-BSGS

**What to do.**
After T-STRAGGLER, the AllReduce kernel still takes ~291 ms. This is real NVLink
communication and cannot be removed. But it can be hidden: while the AllReduce
for rotation i is running on the NCCL stream, the next rotation's modulus raising
(modup) can start on the compute stream. Currently the code calls
`cudaStreamSynchronize` after AllReduce which stalls the CPU — and therefore stalls
the next modup from launching — until AllReduce completes.

**Implementation.**
1. Remove `cudaStreamSynchronize` from `allreduce_keyswitching_result()`.
2. Instead, record an event after the AllReduce enqueue:
   ```cpp
   cudaEventRecord(ctx.allreduce_done_event[gpu_id], ctx.streams[gpu_id]);
   ```
3. The moddown and add-to-ciphertext steps (Steps 4 and 5 in
   `keyswitching_output_aggregation_dks`) already depend on the AllReduce result.
   They will naturally stall on the stream until AllReduce finishes — no change needed.
4. The CPU thread returns immediately after step 3 and can launch the next rotation's
   modup on the compute stream while AllReduce runs.
5. Add `ctx.allreduce_done_event` to `MultiGpuContext`.

**Files to modify.**
- `src/multi_gpu/keyswitching/output_aggregation.cu` — remove `cudaStreamSynchronize`,
  add event record
- `src/multi_gpu/distributed_context.cuh` — add `allreduce_done_events`
- `src/multi_gpu/keyswitching/galois_oa.cu` — verify that the rotation loop launches
  the next modup without waiting for prior AllReduce on CPU side

**Benchmark and measurement.**
Same benchmarks as T-STRAGGLER. Primary evidence is the Nsight trace: open
`trace_dksrot.nsys-rep` and verify that NCCL collective ops and the next rotation's
`modup` NVTX range overlap in time.

**Success criteria.**
- Nsight trace shows NCCL AllReduce and next rotation modup overlapping by ≥ 100 ms.
- `bert_dks_multigpu` bootstrap/call: mean < 1,550 ms (was ~1,600 ms after T-STRAGGLER).
- MAE unchanged.

**Where results go in paper.**
Section 4 (Design), subsection "Communication-Compute Overlap" — explain the
double-buffering analogy. Section 5, update bootstrap evolution table.

---

### T-MODUP — Per-Digit Modulus Raising in Phantom
**Status:** open  
**Depends on:** nothing  
**Can run in parallel with:** T-TRACE, T-STRAGGLER, T-NEXUS, T-12LAYER-BASE

**What to do.**
In distributed key-switching, each GPU needs to compute the key-switch inner product
for only its assigned digits. But currently every GPU runs the full modulus-raising
step (modup) that expands c2 from ~19 MB to ~780 MB across all digits, even though
each GPU only uses ~195 MB of that output. This wastes 75% of the NTT computation
and 75% of the memory allocation on each GPU. Per-digit modup makes each GPU raise
only the digit range it owns.

**What modulus raising does (for the paper).**
A ciphertext polynomial c2 lives in a ring with L=36 coefficient moduli (about 19 MB
at N=65536). Before the inner-product step, c2 must be decomposed into β≈36 digits
and each digit raised to an extended representation including L + 5 special primes
(about 41 coefficient moduli per digit). The output is β × N × 41 × 8 bytes ≈ 780 MB.
Each GPU owns β/4 of these digits. Currently we allocate and compute all 780 MB per
GPU, then use only 195 MB. The fix: pass a digit range to modup so each GPU computes
and allocates only its 195 MB.

**Implementation.**
1. In `vendor/phantom/src/util/rns.cu`, locate the `modup` function. Add a new
   overload or optional parameters `size_t d_start, size_t d_count` that limit
   the digit loop to `[d_start, d_start + d_count)`. Everything else stays the same.
2. In `src/multi_gpu/keyswitching/output_aggregation.cu`, change the modup call in
   `keyswitching_output_aggregation_dks` to:
   ```cpp
   size_t d_start = gpu_id * (beta / n_gpus);
   size_t d_count = beta / n_gpus;
   rns_tool.modup_partial(t_mod_up.get(), c2, ..., d_start, d_count);
   ```
   Allocate `t_mod_up` for `d_count * size_QlP_n` instead of `beta * size_QlP_n`.
3. Update `partial_key_switch_inner_prod` so it no longer uses the strided pattern
   (d = gpu_id, gpu_id + n_gpus, ...) — instead iterate d = 0..d_count-1 since
   the input array now only contains the owned digits contiguously.

**Files to modify.**
- `vendor/phantom/src/util/rns.cu` — add `modup_partial` overload (Phantom internals,
  handle carefully — this is the highest-risk change in this PRD)
- `vendor/phantom/include/rns.cuh` — declare the new overload
- `src/multi_gpu/keyswitching/output_aggregation.cu` — use `modup_partial`,
  adjust allocation and kernel parameters

**Correctness test before benchmarking.**
Before running the full benchmark, verify correctness with `dist_bootstrap_bench`:
```bash
make -j20 dist_bootstrap_bench
sbatch scripts/mn5/slurm_dks_bootstrap.sh
```
Check that MAE is 2.25e-6 (same as Phase 4b). If MAE > 1e-3, the digit indexing
is wrong — compare digit loop bounds carefully against the original modup.

**Benchmark and measurement.**
After MAE passes: run `bert_dks_multigpu` 5 times.
Also check GPU memory usage per run: `nvidia-smi --query-gpu=memory.used` should
show ~195 MB allocated for `t_mod_up` (not 780 MB).

**Success criteria.**
- `t_mod_up` GPU allocation per DKS rotation: ≤ 200 MB per GPU (was ~800 MB).
- `bert_dks_multigpu` bootstrap/call: mean < 1,300 ms standalone
  (or < 1,100 ms if combined with T-STRAGGLER).
- MAE: 2.25e-6 on every config.

**Where results go in paper.**
Section 4 (Design), subsection "Reducing Redundant NTT Work". Section 5, update
bootstrap evolution table and the profiling breakdown table (NTT % should drop from
40% to ~10%).

---

### T-BSGS — Baby-Step Parallelism in Bootstrap Linear Transforms
**Status:** open — conditional on T-TRACE  
**Depends on:** T-TRACE (proceed only if BSGS > 5% of bootstrap time)  
**Can run in parallel with:** T-OVERLAP, T-MODUP (if T-TRACE condition is met)

**What to do.**
CKKS bootstrapping uses a Baby-Step Giant-Step (BSGS) algorithm to evaluate the
CoeffToSlot and SlotToCoeff linear transforms. The algorithm splits each transform
into a small number of "giant steps" (sums of results) and a larger number of
"baby steps" (independent multiply-then-rotate operations). Because baby steps are
independent of each other, they can run on separate CUDA streams simultaneously
within a single GPU.

**What "baby steps" and "giant steps" mean concretely.**
A BSGS transform with parameters (baby=8, giant=4) does 8 independent baby steps
followed by accumulation into 4 giant steps. Each baby step is: multiply the current
ciphertext by a precomputed plaintext polynomial, then rotate by a fixed offset.
The baby steps for the same giant-step group can all run at the same time.

**Implementation plan (after T-TRACE confirms BSGS > 5%).**
1. In `src/nexus_eval/bootstrapping/Bootstrapper.cu`, identify the baby-step loops
   inside `bsgs_linear_transform_CoeffToSlot` and `bsgs_linear_transform_SlotToCoeff`.
2. For each giant-step group, launch the baby-step operations onto separate CUDA
   streams (create N_BABY_STREAMS = min(baby_count, 4) persistent streams in the
   Bootstrapper constructor).
3. After all baby-step streams finish, synchronise with events before the giant-step
   accumulation. Use `cudaEventRecord` + `cudaStreamWaitEvent` (the same pattern as
   T-STRAGGLER).

**Files to modify.**
- `src/nexus_eval/bootstrapping/Bootstrapper.cu` — restructure baby-step loop to
  use multiple streams; add persistent stream pool to Bootstrapper class
- `src/nexus_eval/bootstrapping/Bootstrapper.cuh` — add stream pool member

**Success criteria.**
- Nsight trace shows multiple baby-step kernels running in parallel on different streams.
- `bert_dks_multigpu` bootstrap/call: mean < 1,000 ms (combined with T-STRAGGLER + T-MODUP),
  or mean < 1,900 ms standalone.
- MAE: 2.25e-6.
- If BSGS was < 5% per T-TRACE: mark this task as "skipped — profiling-justified" and
  write a one-paragraph explanation in the paper's Discussion section.

**Where results go in paper.**
Section 4 (Design), subsection "Intra-Bootstrap Parallelism". Section 5, bootstrap
evolution table. Section 6 (Discussion), either as a measured win or as an example
of profiling preventing premature optimisation.

---

### T-LRU — LRU Key Cache on GPU
**Status:** open  
**Depends on:** nothing  
**Can run in parallel with:** all other tasks

**What to do.**
The current key prefetch uses two GPU buffer slots (ping-pong). Each slot holds one
rotation key (~1.3 GB in the non-DKS streaming path, or ~0.3 GB per GPU in the DKS
path). Bootstrap performs 75 rotations, loading each key from CPU pinned RAM once.
The rotation order inside BSGS is statically determined at compile time, meaning we
know exactly which keys will be needed in what order. An LRU cache with more slots
would reduce H→D transfers whenever the same key is reused within a bootstrap.

Note: with DKS rotation enabled (current champion), keys live on GPU as digit shards
and there is no streaming for DKS rotations. This optimisation targets the baseline
streaming path (DKS_ROTATE=0) and any fallback path on fewer GPUs.

**Implementation.**
1. In `src/nexus_eval/galois_key_store.cuh`, replace the 2-slot ping-pong with an
   N-slot LRU cache (N = how many 0.3 GB-per-GPU key shards fit in remaining GPU
   memory; start with N=10 as a safe default).
2. Track slot occupancy with a map from rotation step → slot index.
3. On `ensure_key_loaded(step)`: if step is in cache, return immediately (cache hit).
   If not, evict the LRU slot and async-copy the key shard into it.

**Files to modify.**
- `src/nexus_eval/galois_key_store.cuh`

**Success criteria.**
- DKS_ROTATE=0 path: H→D transfer count per bootstrap drops from 75 to < 30
  (verify with Nsight `memsum` report: count of `cudaMemcpyAsync` calls tagged
  with `ks_load` NVTX ranges).
- Bootstrap time on DKS_ROTATE=0 path: mean < 1,800 ms (was 2,284 ms for 1-GPU
  async prefetch baseline).
- MAE: 2.25e-6.

**Where results go in paper.**
Section 4 (Design), subsection "Key Prefetch and Caching". Used as an ablation
comparison in Section 5.

---

### T-12LAYER-BASE — Full 12-Layer BERT (Baseline)
**Status:** open  
**Depends on:** nothing  
**Can run in parallel with:** all other tasks

**What to do.**
All current BERT numbers are projections: we measure one encoder layer for one
attention head and multiply by 12. Run the actual full 12 encoder layers end-to-end
and check whether the measured time matches the projection. The block repeats, so
the projection should be accurate — but setup costs, ciphertext chaining overhead,
and GPU warm-up effects may cause deviation.

**Implementation.**
In `src/benchmarks/bert_dks_multigpu.cu`, add a loop that runs the 14-operation
BERT layer 12 times with the output ciphertext of each layer fed as input to the next.
Measure the total wall time and divide by 12 for per-layer numbers.

**Benchmark.**
Run on MN5 with `sbatch scripts/mn5/slurm_bert_dks.sh` (modified for 12 layers), 3 runs.

**Success criteria.**
- Measured 12-layer total time is within ±10% of 12 × single-layer projection.
- If deviation > 10%, report what causes it (likely bootstrap level budget — after 12
  bootstraps the chain index may need re-checking).

**Where results go in paper.**
Section 5 (Evaluation), as a note validating the projection methodology.

---

### T-12LAYER-OPT — Full 12-Layer BERT (After Optimisations)
**Status:** open  
**Depends on:** T-STRAGGLER and T-MODUP must both be complete  
**Can run in parallel with:** T-BSGS, T-OVERLAP

**What to do.**
Repeat T-12LAYER-BASE after T-STRAGGLER and T-MODUP are both merged. This gives
the headline result for the paper.

**Success criteria.**
- 12-layer measured time < 55 s (target: ~49–52 s).
- vs CPU baseline projection of 249.6 s: ≥ 4.5× speedup.
- MAE for all 12 layers: 2.25e-6.

**Where results go in paper.**
Section 5, main results table. Abstract.

---

### T-NEXUS — Parameter-Matched NEXUS Comparison
**Status:** open  
**Depends on:** nothing  
**Can run in parallel with:** all other tasks

**What to do.**
NEXUS reports bootstrap at N=32,768. Run our system at N=32,768 on 4× H100 to create
a direct hardware-matched comparison. At N=32,768 our bootstrap keys are ~15 GB total
and fit on GPU entirely — no CPU streaming needed.

Note: NEXUS uses N=32,768 for bootstrap only, then re-encrypts to N=65,536 for the
rest of the computation. We use N=65,536 throughout. This comparison shows what our
infrastructure achieves at the smaller ring size on newer hardware (H100 vs A100).

**How to run.**
`src/benchmarks/bert_encoder_multigpu.cu` already runs at N=32,768 (the `_n65536`
suffix benchmarks use the larger ring). Check the compile-time constants and run:
```bash
make -j20 bert_encoder_multigpu
sbatch scripts/mn5/slurm_bert_encoder_multigpu.sh  # create if missing; 5 runs
```

**Success criteria.**
- Bootstrap at N=32,768 on 4× H100: mean ± std reported.
- BERT one-layer time at N=32,768: mean ± std reported.
- Compare against NEXUS's reported 5,600 ms bootstrap (4× A100). Note the hardware
  difference (H100 > A100 by ~1.5–2×) and the N=65,536 vs N=32,768 advantage.

**Where results go in paper.**
Section 5, Table "Parameter-Matched Comparison". Also Section 6 (Discussion):
explain why a direct comparison is still approximate due to hardware differences.

---

## Paper Tasks

These tasks write the paper sections in `paper/main.tex`. The paper should be
plain English — explain what was done, what was found, and what it means.
No internal phase names ("Phase 4b"), no jargon without definition.

---

### P-SETUP — Paper Skeleton
**Status:** open  
**Depends on:** nothing

**What to do.**
Create `paper/main.tex` using IEEEtran conference format. Add section stubs with
TODO markers for all sections below. Copy the bibliography infrastructure from
`papers/Scalable.../refs.bib` as a starting point.

Sections:
1. Abstract
2. Introduction
3. Background: Homomorphic Encryption and CKKS Bootstrapping
4. Design: How We Distribute a Single Bootstrap Across GPUs
5. Evaluation
6. Performance Ceiling and Comparison with Cerium
7. Related Work
8. Conclusion

---

### P-BG — Background Section
**Status:** open  
**Depends on:** P-SETUP

**What to write.**
Explain CKKS in plain English: what a ciphertext is, what slots are, why multiplication
consumes a "level", why bootstrapping is needed, and what bootstrapping actually does
(CoeffToSlot → polynomial modular reduction → SlotToCoeff). Then explain key-switching:
what it is, the three steps (decompose into digits, inner product with key, modulus drop),
and why a rotation key at N=65,536 is ~1.3 GB. End with a table showing how ring degree N
affects key size and why N=65,536 forces multi-GPU.

Use Table 6 from `docs/RESULTS_SUMMARY.md` (CKKS micro-benchmarks) and Table 8
(memory footprint) as sources. Cite NEXUS (Zhang et al., NDSS 2025) and the original
CKKS paper (Cheon et al., 2017).

**Target length:** ~1 column (about 400–500 words).

---

### P-DESIGN — Design Section
**Status:** open  
**Depends on:** P-SETUP

**What to write.**
Explain the four design ideas, in order of impact:
1. Async key prefetch with double-buffered GPU slots and `cudaHostRegister` — why
   pinned host memory was the critical enabler.
2. Distributed Key-Switching (DKS) — split the key-switch inner product across GPUs
   so each GPU holds only 1/4 of each key's digits. Explain the AllReduce step.
3. Launch-jitter barrier (T-STRAGGLER result) — what straggler wait is and how
   event-based barriers fix it.
4. Per-digit modulus raising (T-MODUP result) — why every GPU was doing 4× more
   NTT work than necessary and how restricting the digit range fixes it.

Use `docs/RESULTS_SUMMARY.md` Table 7 (the cautionary regression data) as the
motivating example for why naive multi-GPU FHE makes things worse before it makes
them better.

Include a figure showing the key-memory split across GPUs and the AllReduce step.
The figure files already exist as SVGs in `paper/`; pick the most relevant one.

**Target length:** ~1.5 columns.

---

### P-EVAL — Evaluation Section
**Status:** open  
**Depends on:** T-STRAGGLER, T-MODUP, T-12LAYER-OPT, T-NEXUS must all be complete  
**Can start with placeholders using Phase 4b numbers from `docs/RESULTS_SUMMARY.md`**

**What to write.**
Present results in this order:
1. The optimisation story — a table showing bootstrap time evolving from 10.7 s to
   the final optimised number, one row per change, one column per metric. Base this
   on RESULTS_SUMMARY Table 1 and add the new rows from T-STRAGGLER / T-MODUP.
2. Where the time goes — the profiling breakdown (updated after T-TRACE). Show the
   percentage table. Point out the "no single kernel over 3%" finding.
3. Full BERT layer breakdown — one attention head, then 12-head projection, then
   measured 12-layer from T-12LAYER-OPT.
4. Parameter-matched comparison — bootstrap at N=32,768 vs NEXUS, from T-NEXUS.
5. System comparison table — our final numbers vs NEXUS vs Cerium vs Cinnamon,
   with the comparison footnotes from the Target Results section above.

All tables should be self-contained with a caption explaining what the reader
should take away, following the format in `docs/RESULTS_SUMMARY.md`.

**Measurement protocol note in paper.** State: "All multiNEXUS measurements report
the mean of 5 independent runs ± one standard deviation. For benchmarks exceeding
60 seconds per run, 3 runs are used."

---

### P-CEILING — Performance Ceiling Section
**Status:** open  
**Depends on:** T-TRACE, T-STRAGGLER, T-MODUP  
**Note:** This is the most important section of the paper.

**What to write.**
Explain exactly where the remaining time goes after all our optimisations, and why
further progress requires a fundamentally different approach (like Cerium's compiler).

Structure:
1. Show the profiling breakdown after our optimisations. What % remains?
2. Explain that the remaining work is "breadth-first" — ~10^5 tiny kernel launches
   per bootstrap, no single kernel over 3%, so individual kernel tuning cannot help.
3. Explain what Cerium does differently: it uses a compiler that statically schedules
   all kernel launches, eliminates redundant work across the entire bootstrap graph,
   and achieves compute-communication overlap automatically. Show their regression
   data (naïve 8-GPU is 1.2× slower than 1-GPU; with scheduling it reaches 1.93×).
4. Show that our approach and Cerium's approach converge to the same wall: the
   rotation-bound floor. At N=65,536, a single rotation takes ~30 ms (Table 6 in
   `docs/RESULTS_SUMMARY.md`). Bootstrap needs 75 rotations. Minimum possible time
   with perfect GPU utilisation: 75 × 30ms / 4 GPUs = ~563 ms.

This section is the honest answer to "what would it take to reach Cerium's 7.5 ms?"
and explains why it requires not just engineering effort but a different system architecture.

**Target length:** ~1 column.

---

### P-RELATED — Related Work
**Status:** open  
**Depends on:** P-SETUP

**What to write.**
Three paragraphs:
1. Single-GPU FHE inference: NEXUS (Zhang et al., NDSS 2025), cite their 5.6 s
   bootstrap at N=32,768, note the protocol difference (multi-N, re-encryption).
2. Compiler-based multi-GPU FHE: Cinnamon (Jayashankar et al., ASPLOS 2025) as
   the ASIC-targeting predecessor; Cerium (Jayashankar et al., arXiv 2025) as the
   GPU realisation. Note code is not yet public. Cite their bootstrap numbers and
   the naïve-multi-GPU regression they also observe.
3. Other FHE transformer work: EncryptedLLM (De Castro et al., ICML 2025) for
   single-GPU at shallower bootstrap depth; BOLT and Bumblebee for MPC-hybrid
   approaches (note these are not fully homomorphic — they require interaction).

---

### P-ABSTRACT — Abstract and Introduction
**Status:** open  
**Depends on:** P-EVAL (needs the final numbers)

**What to write.**
Abstract: 150–200 words. State the problem (N=65,536 keys do not fit on one GPU),
what we built (DKS + async prefetch + scheduling fixes), and the result (X× over
CPU baseline, Y s projected 12-layer BERT). Cite that we run at strictly larger N
than NEXUS with no re-encryption.

Introduction: ~0.5 column. Motivate privacy-preserving inference briefly. State
the memory problem. Summarise our three contributions:
(1) async prefetch with pinned host memory enabling 4.69× single-GPU win,
(2) distributed key-switching enabling N=65,536 keys to fit across 4× H100,
(3) profiling-driven identification of the remaining performance ceiling.

---

## Dependency Summary

```
T-TRACE ─────────────────────────────────────────► T-BSGS (conditional)
T-STRAGGLER ──────────────────────────────────────► T-OVERLAP
T-STRAGGLER + T-MODUP ────────────────────────────► T-12LAYER-OPT
                                                     P-EVAL (final numbers)
T-TRACE + T-STRAGGLER + T-MODUP ─────────────────► P-CEILING

Independent (start immediately):
  T-TRACE, T-STRAGGLER, T-MODUP, T-LRU,
  T-12LAYER-BASE, T-NEXUS, P-SETUP, P-BG,
  P-DESIGN, P-RELATED

After P-EVAL is complete:
  P-ABSTRACT (needs final numbers)
```

**Recommended execution order for a single agent session.**
Run T-STRAGGLER first (few hours, high impact). While waiting for MN5 results,
write P-BG and P-DESIGN (can be done with existing data). When T-STRAGGLER results
arrive, update P-EVAL with the new row. Then run T-MODUP. After T-MODUP passes
correctness, run T-12LAYER-OPT for the headline number.
