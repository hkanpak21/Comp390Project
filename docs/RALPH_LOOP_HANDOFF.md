# multiNEXUS Ralph Loop — MN5 Sync & Validation Handoff (2026-05-07)

This document is the runbook for syncing three iterations of Ralph-loop changes to MN5
(BSC MareNostrum 5) and validating them. Read top-to-bottom before launching anything.
The MAE sanity check in **Step 1** is non-negotiable: T-MODUP touches Phantom internals.

---

## What changed (3 iterations)

| PRD Task | Status | Files Modified | Key change | Risk |
|---|---|---|---|---|
| T-STRAGGLER | landed | `src/multi_gpu/distributed_context.{cu,cuh}`, `src/multi_gpu/comm/nccl_comm.{cu,cuh}`, `src/multi_gpu/keyswitching/output_aggregation.cu`, `src/multi_gpu/keyswitching/galois_oa.cu` | `ready_events` array on `MultiGpuContext`; record after partial KS, `cudaStreamWaitEvent` barrier across all GPUs before AllReduce | low |
| T-OVERLAP | landed | (above) + new `oa_done_events` vector, removed inner `cudaStreamSynchronize` in `allreduce_keyswitching_result` | replace inner sync with `cudaEventRecord`; cross-stream wait at writeback boundary lets next modup launch while NCCL runs | medium |
| T-TRACE | landed | `src/nexus_eval/bootstrapping/Bootstrapper.cu`, `src/multi_gpu/keyswitching/output_aggregation.cu` | NVTX ranges around 4 BSGS primitives (8 push/pops), plus `modup` and `moddown` | none |
| T-MODUP | landed | `vendor/phantom/include/rns.cuh`, `vendor/phantom/src/rns_bconv.cu`, `src/multi_gpu/keyswitching/output_aggregation.{cu,cuh}`, `src/multi_gpu/keyswitching/dist_galois_key_store.cuh` | new `modup_partial(d_start, d_count)` overload + `modup_copy_partQl_partial_kernel`; CONTIGUOUS digit ownership; `partial_key_switch_inner_prod` signature changed to consume contiguous local digit array | **HIGH** (Phantom internals + DKS ownership change) |
| T-LRU | landed | `src/nexus_eval/galois_key_store.cuh` | N=10 LRU cache (`std::list` + splice) replacing 2-slot ping-pong; targets DKS_ROTATE=0 streaming path | low |
| T-12LAYER-BASE | landed | `src/benchmarks/bert_dks_multigpu.cu`, new `scripts/mn5/slurm_bert_12layer_dks.sh` | `BERT_LAYERS` env loops the 14-op layer; default=1 preserves single-layer baseline | low |
| Paper P-SETUP / P-BG / P-DESIGN / P-RELATED / P-EVAL skeleton / P-CEILING / P-ABSTRACT / P-INTRO / P-CONCL | landed | `paper/main.tex`, `paper/refs.bib` | full IEEEtran draft with `\TODO{}` markers for MN5-derived numbers | none |

**Iteration-2/3 known watch-items** (carried over from `ralph-loop.local.md`):

- BibTeX entries `DeCastro2025`, `Bumblebee2025`, `Jayashankar2025-arxiv` were reconstructed from `study.md` — verify against official proceedings before camera-ready (not blocking).
- Per-key memory table at smaller N is computed scaling, not measured. Honest scaling is fine for the draft.
- Trailing `cudaStreamSynchronize` at OA lines 325/432: investigated and replaced by event-based barrier in T-OVERLAP. If MAE passes but bootstrap time is unchanged, look here.

---

## Sync command

From the local repo root (`/Users/a90/Documents/COURSES/SPRING2026/Comp390/Comp390Project`):

```bash
rsync -avz \
  --include='*/' \
  --include='*.cu' --include='*.cuh' --include='*.h' --include='*.hpp' \
  --include='*.tex' --include='*.bib' \
  --include='*.sh' \
  --exclude='*' \
  src/ scripts/ paper/ vendor/ \
  mn5-gpu:/gpfs/projects/etur02/hkanpak/Comp390Project/
```

Note: `vendor/phantom` is touched in **two** files (`include/rns.cuh`, `src/rns_bconv.cu`) by
T-MODUP, so the sync **must** include `vendor/`. If you previously synced with a stricter
filter, re-run the command above to pick those up.

---

## Build on MN5

```bash
ssh mn5-gpu
module purge && module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1
cd /gpfs/projects/etur02/hkanpak/Comp390Project
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..   # only if cmake hasn't run before

# ⚠ CRITICAL: Phantom is built via ExternalProject_Add WITHOUT BUILD_ALWAYS,
# so editing vendor/phantom/*.cu does NOT auto-trigger a Phantom rebuild.
# Force it explicitly the first time after sync:
cmake --build . --target phantom_ext -- -j20
# (alternative: cd phantom_build && make -j20)

make -j20 dist_bootstrap_bench bert_dks_multigpu bert_encoder_multigpu
```

If make fails: the most likely culprit is the `modup_partial` code in
`vendor/phantom/src/rns_bconv.cu`. See **Build troubleshooting** below.

If `dist_bootstrap_bench` runs but MAE comes back unchanged at 2.25e-6 *with the
exact same wall-clock time as Phase 4b*, the linker probably picked up the old
`libPhantom.so` — the `phantom_ext` rebuild step was skipped. Force it with:
```bash
rm -f phantom_build/lib/libPhantom.so
cmake --build . --target phantom_ext -- -j20
make -j20 dist_bootstrap_bench
```

Phantom rebuild can take 5–10 minutes; a clean rebuild (`make clean && make -j20`) of all
three targets is roughly 15 minutes.

---

## Validation order — DO NOT SKIP

### Step 1: MAE sanity check (~5 min)

T-MODUP changes Phantom internals. Run `dist_bootstrap_bench` **first** and verify
MAE = 2.25e-6 before any timing benchmark. A wrong digit-index makes the bootstrap
silently produce garbage with bad MAE but still finish without crashing.

```bash
sbatch scripts/mn5/slurm_dks_bootstrap.sh
# Wait for the job; then:
ls -lt /gpfs/projects/etur02/hkanpak/logs/dks_bootstrap_*.out | head -1
grep -E "MAE|mae|max_abs_err" /gpfs/projects/etur02/hkanpak/logs/dks_bootstrap_<JOBID>.out
```

Expected: `MAE = 2.25e-6` (or within 1e-7 of that). If MAE > 1e-3, **stop** and go to
**Recovery — if MAE breaks** below. Do not proceed to timing benchmarks.

### Step 2: Bootstrap timing benchmark (~5 min × 5 runs)

```bash
for i in 1 2 3 4 5; do sbatch scripts/mn5/slurm_dks_bootstrap.sh; done
# Once all five complete:
grep -E "AllReduce|bootstrap/call" /gpfs/projects/etur02/hkanpak/logs/dks_bootstrap_*.out
```

PRD success criteria:

- T-STRAGGLER alone: AllReduce wall < 400 ms (was ~820 ms).
- T-STRAGGLER + T-MODUP: bootstrap/call < 1,100 ms (was 2,126 ms).
- MAE: 2.25e-6 on every run.

Compute mean ± std across the 5 runs. These are the numbers that go into the paper's
"Optimization Steps" table.

### Step 3: Single-layer BERT (~10 min)

```bash
for i in 1 2 3 4 5; do sbatch scripts/mn5/slurm_bert_dks.sh; done
grep -E "bootstrap/call|MAE|layer time" /gpfs/projects/etur02/hkanpak/logs/bert_dks_*.out
```

Expected: bootstrap/call < 1,100 ms, MAE = 2.25e-6, single-layer time roughly
4.0–4.5 s (was ~9.6 s in Phase 4b projection).

### Step 4: 12-layer BERT — T-12LAYER-OPT (~60–90 min)

```bash
for i in 1 2 3; do sbatch scripts/mn5/slurm_bert_12layer_dks.sh; done
grep -E "12-layer|total wall|layer-mean" /gpfs/projects/etur02/hkanpak/logs/bert_12layer_dks_*.out
```

Expected: 12-layer total < 55 s (target band 49–52 s), MAE for all 12 layers = 2.25e-6.
3 runs is sufficient per PRD (each run > 60 s).

### Step 5: T-NEXUS comparison at N=32768 (~15 min)

```bash
sbatch scripts/mn5/slurm_bert_encoder_multigpu.sh   # 5-run loop is built into the script
grep -E "bootstrap|N=32768|encoder layer" /gpfs/projects/etur02/hkanpak/logs/bert_encoder_multigpu_n32k_*.out
```

Expected: bootstrap and one-encoder-layer numbers at N=32,768 on 4× H100, for
parameter-matched comparison against NEXUS's 5,600 ms (4× A100, N=32,768).

### Step 6: Granular trace — T-TRACE (~5 min trace + analysis)

```bash
sbatch scripts/mn5/slurm_trace_nsys.sh
# After completion:
scp mn5-gpu:/gpfs/projects/etur02/hkanpak/Comp390Project/traces/trace_dksrot.nsys-rep ~/nexus-traces/
nsys stats --report nvtxsum ~/nexus-traces/trace_dksrot.nsys-rep \
  | grep -E "bsgs_baby_step|bsgs_giant_step|modup|moddown"
```

Use the % of bootstrap wall time for each new range to decide T-BSGS:

- If `bsgs_baby_step + bsgs_giant_step > 5%` → T-BSGS is worth a future iteration.
- If < 5% → mark T-BSGS as profiling-justified skip in the paper's Discussion.

---

## Numbers to capture for the paper

After each step, fill the corresponding `\TODO{}` in `paper/main.tex`:

| `\TODO{}` Marker | Source benchmark |
|---|---|
| Optimization steps table — T-STRAGGLER row | Step 2 (mean AllReduce wall, mean bootstrap/call) |
| Optimization steps table — T-OVERLAP row | Step 2 (compare with-overlap vs without-overlap evidence in trace) |
| Optimization steps table — T-MODUP row | Step 2 (combined T-STRAGGLER+T-MODUP bootstrap/call) |
| "Where the time goes" post-opt column | Step 6 `nvtxsum` |
| Full BERT layer breakdown — measured 12-layer | Step 4 (mean of 3 runs ± std) |
| Parameter-matched comparison — N=32,768 bootstrap | Step 5 (mean ± std of 5 runs) |
| System comparison table — final multiNEXUS row | Steps 2 + 4 |
| Abstract / Conclusion — X× over CPU baseline | computed: 249.6 s ÷ measured 12-layer time from Step 4 |
| Performance Ceiling — remaining-work breakdown | Step 6 `nvtxsum` percentages |

Measurement protocol stated in the paper: mean of 5 independent runs ± one standard
deviation; 3 runs accepted for any benchmark exceeding 60 s per run.

---

## Recovery — if MAE breaks

If Step 1 reports MAE > 1e-3 (or any value materially above 2.25e-6), the most likely
culprit is T-MODUP's CONTIGUOUS-layout indexing:

1. **Suspect file 1**: `vendor/phantom/src/rns_bconv.cu` — `modup_partial` (around lines
   682–815) and the new `modup_copy_partQl_partial_kernel` (around lines 540–558). Verify
   the digit loop bound is `d_count` (not `beta`) and the input pointer offset is
   `c2 + d_start * size_QlP_n` for the source slice.
2. **Fast bisect**: revert T-MODUP only by setting `d_count = beta` and `d_start = 0` at
   each call site in `output_aggregation.cu`. This returns to per-call full modup; if MAE
   passes after this, the indexing inside `modup_partial` is wrong. If MAE still fails
   here, the bug is in T-STRAGGLER/T-OVERLAP and the event ordering.
3. **Suspect file 2**: `src/multi_gpu/keyswitching/output_aggregation.cu` —
   `partial_key_switch_inner_prod` around lines 70–134. The local `c2[d * size_QlP_n]`
   indexing must align with the local layout produced by `modup_partial`. After T-MODUP,
   `d` iterates `0..d_count-1` over a contiguous local buffer; **not** the strided
   `gpu_id, gpu_id + n_gpus, ...` pattern.
4. **Suspect file 3**: `src/multi_gpu/keyswitching/dist_galois_key_store.cuh` — around
   lines 145–153 and 252–261. GPU `g` now owns the contiguous digit range
   `[g * beta / n_gpus, (g+1) * beta / n_gpus)`. If any caller still expects the strided
   pattern, the `evks[d_start + d]` lookup will return null/wrong-key.

If Step 2 or Step 3 pass MAE but bootstrap time is unchanged or slower, T-OVERLAP's
writeback-event wait is the culprit. Inspect `galois_oa.cu` around the
`cudaStreamWaitEvent` before writeback — the event wait may be over-conservative and
forcing the stream to drain.

---

## Build troubleshooting

**T-MODUP compile failures**

- `error: 'modup_partial' is not a member of 'phantom::DRNSTool'` — the header
  (`vendor/phantom/include/rns.cuh`) and source (`vendor/phantom/src/rns_bconv.cu`) are
  out of sync. The rsync above missed one of them. Re-sync `vendor/`.
- `undefined reference to phantom::DRNSTool::modup_partial(...)` — link order or stale
  object file. Run `make clean && make -j20 dist_bootstrap_bench`.
- Phantom rebuild can take 5–10 min on MN5.

**T-LRU compile failures**

- Missing `<list>` or `<unordered_map>` in `galois_key_store.cuh` — iteration 1 added
  these, but verify if compile fails.
- Rule-of-Five conflict: a user-declared destructor with deleted copy ctor and defaulted
  move ctor should be fine. If the compiler complains, `git diff vendor/` should be empty
  — meaning the issue is in our `src/`, not in vendored code.

**NVTX include failures**

- `fatal error: nvtx_tracer.cuh: No such file or directory` — the include is
  `#include "nvtx_tracer.cuh"` and the build expects `src/util/` on the include path.
  Should be set by CMake (`target_include_directories` on the parent `multi_gpu` /
  `nexus_eval` libraries). If not set, change to a relative path
  `#include "../util/nvtx_tracer.cuh"` from the affected file.

---

## Quick smoke command

To confirm the source builds before launching any SLURM job:

```bash
ssh mn5-gpu "cd /gpfs/projects/etur02/hkanpak/Comp390Project/build && \
  make -j20 dist_bootstrap_bench 2>&1 | tail -30"
```

A successful build prints `Linking CXX executable .../dist_bootstrap_bench` (or the CUDA
equivalent) and exits 0. If it fails, the last 30 lines of output usually point to the
file. Re-run with `2>&1 | tail -120` to see more context.

---

## What's NOT in this session (deferred)

- **T-BSGS** (baby-step parallelism) — conditional on Step 6 `nvtxsum` showing
  `baby + giant > 5%`. Implementation plan in PRD §T-BSGS.
- **Single-stream rewrite of `dist_rotate_phantom_inplace`** — would enable removing
  remaining stream0 syncs but is out of PRD scope.
- **Empirical bib-entry verification** — `DeCastro2025`, `Bumblebee2025`,
  `Jayashankar2025-arxiv` reconstructed from `study.md` narrative; need camera-ready
  check.
- **Per-key memory table at smaller N** — currently computed scaling, not measured.

---

## Reference: SLURM scripts in `scripts/mn5/`

| Script | Purpose | Used in step |
|---|---|---|
| `slurm_dks_bootstrap.sh` | Bootstrap-only on `dist_bootstrap_bench` (DKS path) | Steps 1, 2 |
| `slurm_bert_dks.sh` | Single-layer BERT, DKS path | Step 3 |
| `slurm_bert_12layer_dks.sh` | 12-layer BERT, DKS path (BERT_LAYERS=12) | Step 4 |
| `slurm_bert_encoder_multigpu.sh` | BERT encoder at N=32,768 (NEXUS-matched) | Step 5 |
| `slurm_trace_nsys.sh` | Nsight Systems trace for NVTX breakdown | Step 6 |
| `slurm_dist_bootstrap.sh` | Generic bootstrap variants | (reference) |
| `slurm_llama_dks.sh` | LLaMA path (out of scope this iteration) | (reference) |

All output lands in `/gpfs/projects/etur02/hkanpak/logs/<binary>_<JOBID>.out`.

---

## One-paragraph summary

Three iterations landed five code optimizations (T-STRAGGLER, T-OVERLAP, T-TRACE,
T-MODUP, T-LRU), one infrastructure change (T-12LAYER-BASE adds `BERT_LAYERS` env loop +
SLURM script), and a complete IEEEtran paper draft with `\TODO{}` markers for the
MN5-derived numbers. T-MODUP is the highest-risk change because it touches Phantom
internals and changes DKS digit ownership from strided to contiguous; that is why
**Step 1 (MAE sanity)** must pass before any timing benchmark is trusted. The PRD's
success criteria are bootstrap < 1,100 ms, 12-layer < 55 s, and MAE = 2.25e-6 on every
config.
