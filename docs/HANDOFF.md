# multiNEXUS Handoff — Comp390 Spring 2026

**Read this first when picking up the project in a new session.** Self-contained; assumes
no prior conversation context.

---

## 1. Project In One Paragraph

multiNEXUS is a **multi-GPU FHE (CKKS) BERT/LLaMA inference system** built on top of
the Phantom FHE library. Goal: run encrypted BERT-base inference at N=65536 across 4×
H100 GPUs (single MN5 node) faster than a CPU-streaming reference. Hard part: bootstrap
keys total ~64 GB (don't fit on one H100), and bootstrap time dominates everything.
Owner: Halil İbrahim Kanpak, Comp390 student. Compute: BSC MareNostrum 5 (MN5) ACC
partition (4× H100 64GB SXM per node, NVSwitch).

## 2. Current Headline Numbers (4-GPU, N=65536, single BERT layer projected to 12 heads)

| Mode | Bootstrap/call | Layer (1 head) | 12-head BERT | vs CPU streaming baseline |
|---|---|---|---|---|
| Start of project (CPU streaming, head-parallel) | — | — | 249.6 s | 1.00× |
| DKS storage only (no prefetch, no DKS rotation) | 10,514 ms | 46,278 ms | 555.3 s | 0.45× ❌ |
| Phase 1: async key prefetch + cudaHostRegister | 2,277 ms | 10,234 ms | 122.8 s | 2.03× |
| Phase 3 v2: + persistent peer-broadcast buffers | 2,143 ms | 9,741 ms | 116.9 s | 2.14× |
| Phase 4a: + persistent local_cts | 2,136 ms | 9,680 ms | 116.2 s | 2.15× |
| **Phase 4b: + persistent worker threads (current)** | **2,126 ms** | **9,640 ms** | **115.7 s** | **2.16×** |

**MAE on every config**: 2.25e-6 PASS (well below 0.01 threshold). Correctness is solid.

## 3. Where Things Live

**Local Mac** (primary working tree):
```
/Users/a90/Documents/COURSES/SPRING2026/Comp390/Comp390Project/
├── multiNEXUS.md              ← main writeup, all results, tables 4.1/4.2
├── TRACING.md                 ← how to use NVTX + Nsight Systems
├── CLAUDE.md                  ← auto-loaded Claude context (session start)
├── CMakeLists.txt
├── docs/                      ← session documentation
│   ├── HANDOFF.md             ← THIS FILE
│   ├── RESULTS_SUMMARY.md     ← all 10 measurement tables + prior-art comparison
│   ├── PI_BRIEFING.md         ← advisor walkthrough with Mermaid diagrams
│   └── NSIGHT_GUIDE.md        ← how to open and read Nsight traces
├── src/
│   ├── nexus_eval/            ← single-GPU FHE evaluator (CKKS ops, Bootstrapper)
│   │   ├── ckks_evaluator.{cu,cuh}      ← Evaluator class, rotate dispatch
│   │   ├── galois_key_store.cuh         ← CPU→GPU key streaming with prefetch
│   │   └── bootstrapping/Bootstrapper.cu ← bootstrap_3, sfl_*, slottocoeff_*
│   ├── multi_gpu/             ← DKS infrastructure
│   │   ├── distributed_context.{cu,cuh} ← N-GPU context, RotationWorkspace, workers
│   │   └── keyswitching/
│   │       ├── galois_oa.cu             ← dist_rotate_phantom_inplace (Phase 3)
│   │       ├── output_aggregation.cu    ← partial_key_switch_inner_prod kernel
│   │       └── dist_galois_key_store.cuh ← per-GPU sharded Galois keys
│   ├── benchmarks/            ← .cu binaries
│   │   ├── bert_dks_multigpu.cu          ← MAIN champion benchmark
│   │   ├── bootstrap_n65536_streaming.cu ← single-GPU bootstrap test (MAE check)
│   │   └── dist_bootstrap_bench.cu       ← DKS rotation correctness + timing
│   ├── util/nvtx_tracer.cuh   ← NVTX RAII macros (NVTX_SCOPE)
│   └── ...
├── scripts/mn5/               ← SLURM job templates
│   ├── slurm_bert_dks.sh      ← MAIN A/B test (DKS_ROTATE=0 vs =1)
│   ├── slurm_bootstrap_n65536.sh ← single-GPU bootstrap timing test
│   └── ...
├── profiling/                 ← Nsight Systems / Nsight Compute run scripts
├── vendor/
│   ├── phantom/               ← Phantom FHE library (modified, NOT git-cloned vanilla)
│   └── nexus/                 ← reference NEXUS implementation
└── ~/nexus-traces/            ← (NOT in repo) — local Nsight .nsys-rep files (13 traces)
    ├── trace_dksrot.nsys-rep  ← DKS_ROTATE=1 champion timeline (30 MB, Apr 19)
    └── trace_prefetch.nsys-rep ← DKS_ROTATE=0 comparison timeline (25 MB, Apr 19)
```

**MN5** (`ssh mn5-gpu`, user `koc971580`):
```
/gpfs/projects/etur02/hkanpak/Comp390Project/   ← mirror of local tree (rsync target)
├── build/                     ← cmake build dir, contains binaries in build/bin/
├── traces/                    ← .nsys-rep + .sqlite files
└── (everything else mirrors local)
/gpfs/projects/etur02/hkanpak/logs/             ← all SLURM stdout/stderr
/gpfs/projects/etur02/hkanpak/local/            ← NTL/GMP install (Phantom dep)
```

## 4. The Optimization Story (read for context)

The project went through 5 distinct phases. Each one had a hypothesis, a measurement,
and a lesson:

### Phase 0 — DKS storage only (broken)
- Sharded keys 4-way across GPUs (DistGaloisKeyStore) so they fit. Bootstrap stayed at 10.5 s.
- Per-GPU key memory dropped from 64 GB to 18 GB.
- Did NOT improve speed: bootstrap path still ran on GPU 0 with H→D streaming from CPU.
- **Lesson**: storage sharding alone doesn't help; need compute-side parallelism too.

### Phase 1 — Async key prefetch + cudaHostRegister (4.69× win)
- Idea: while rotation N's kernel runs on default stream, H→D for rotation N+1's key
  runs on a separate copy stream — fully overlapped.
- Added double-buffered slots in `GaloisKeyStore` + `cudaStreamWaitEvent` ordering.
- 8 prefetch hooks in `Bootstrapper.cu` baby/giant BSGS loops (always *after* rotate,
  not before — early bug).
- **CRITICAL**: `cudaMemcpyAsync` from `std::vector<uint64_t>` (pageable host) silently
  degrades to sync via a CUDA bounce buffer. Without `cudaHostRegister` on the 62 GB
  key store, async prefetch saved nothing. WITH pinning: bootstrap 10,712 → 2,284 ms.
- **Lesson**: auto-saved as `feedback_pinned_memory_required_for_async.md` in memory.

### Phase 2 — DKS rotation correctness validated
- Earlier reading of MAE = 0.125 was scary. Side-by-side test: ran same input through
  Phantom's single-GPU rotate AND DKS rotate at chain_index=1 (full precision) AND
  chain_index=36 (bootstrap entry).
- Result: **bit-identical output**. Both produce 0.125 MAE.
- The 0.125 is a *test fixture artifact*: encoder pads sparse_slots=16384 to
  slot_count()=32768 with zeros; the bench compares all 32768 slots to 0.5 → half match,
  half mismatch by 0.5 → MAE = 0.125. DKS algorithm is correct.

### Phase 3 — Wire Bootstrapper to DKS rotation (after 2 iterations)
- Added `dist_rotate_phantom_inplace` in `galois_oa.cu` (operates directly on
  PhantomCT on GPU 0, no DCT scatter/gather).
- `Evaluator::enable_dks_rotation(...)` API. Toggled via `DKS_ROTATE=1` env var.
- **v1**: cudaMalloc per call (8 mallocs × 75 rotations × 4 bootstraps = 2400 mallocs/layer).
  Was **2.4× SLOWER** than prefetch (5,429 ms/bootstrap).
- **v2**: persistent `RotationWorkspace` in `DistributedContext` (per-GPU c0_gal/c2_gal
  buffers, sized on-demand). Dropped to 2,143 ms/bootstrap (5% faster than prefetch).
- **Lesson**: cudaMalloc in hot path is murderous; always preallocate.

### Phase 4 — Diminishing returns
- 4a: persistent `local_cts` per GPU (avoid `resize()` per call). Saved 7 ms/bootstrap.
  PhantomCiphertext::resize is no-op when chain_index unchanged; the cost we estimated
  wasn't real.
- 4b: persistent worker threads in `DistributedContext` (avoid `std::thread` spawn/join
  per rotation). Saved 10 ms/bootstrap. Modern Linux thread spawn is microseconds.
- 4c+: deferred. See "What's Left" below.

### Phase 5 — Visualization (NVTX + Nsight Systems)
- Decided to stop guessing about per-component cost. Instrumented end-to-end with NVTX
  ranges (`src/util/nvtx_tracer.cuh`).
- Ran `nsys profile` on both modes. Found the **REAL** bottleneck breakdown:
  - NTT kernels = 40% of DKS bootstrap time (NOT 15% as I estimated)
  - `partial_key_switch_inner_prod` = 6.7% (parallelization is working)
  - `ncclAllReduce` wall = 2.3 s of which 291 ms is kernel — **the rest is straggler wait**
  - One AllReduce per rotation occasionally spikes to 572 ms (variance from GPU sync jitter)
- Trace files: `~/nexus-traces/trace_{prefetch,dksrot}.nsys-rep`.

## 5. What's Left On The Table (Phase 4c, 4d — DEFERRED)

The traces showed the right next steps:

### 4c — Per-digit modup (~1-2 days, biggest theoretical win)
- **What**: each GPU only computes NTT for its β/4 owned digits, not all β.
- **Why**: NTT is 40% of bootstrap time and currently runs redundantly on all GPUs.
  Sharding it 4-way could cut bootstrap to ~1.3 s (40% reduction).
- **Where**: `vendor/phantom/src/util/rns.cu` — modup is monolithic, needs digit-range variant.
- **Risk**: changes Phantom internals; can break unrelated benchmarks. Build flag-gate it.

### 4d — Reduce AllReduce straggler variance (~few hours, modest win)
- **What**: insert explicit `cudaEventRecord` on each GPU before NCCL launch so all
  4 streams start AllReduce at the same wall-clock instant.
- **Why**: the 572 ms straggler tail in NCCL stats is host-side launch jitter, not actual
  comm cost. The kernel itself is 291 ms across all calls.
- **Where**: `src/multi_gpu/keyswitching/output_aggregation.cu` `keyswitching_output_aggregation_dks`
  function, around the `ncclAllReduce` call.

### 4e — DKS rotation for multi-node (deferred — single-node only currently)
- We have `DistGaloisKeyStore::generate_multinode` but Bootstrapper isn't wired to it.
- Multi-node DKS would extend to 8+ GPUs across MN5 nodes via MPI scatter + NCCL across
  nodes. Already have NCCL infrastructure for it.

## 6. How To Continue Working

### Reproduce the current champion result
```bash
ssh mn5-gpu
cd /gpfs/projects/etur02/hkanpak/Comp390Project
sbatch scripts/mn5/slurm_bert_dks.sh   # runs both DKS_ROTATE=0 and =1
# wait ~10 min, then
cat /gpfs/projects/etur02/hkanpak/logs/bert_dks_<JOBID>.out
```

### Edit code locally, push to MN5, rebuild
```bash
# From local Mac:
rsync -avz --include='*/' --include='*.cu' --include='*.cuh' --exclude='*' \
  src/ mn5-gpu:/gpfs/projects/etur02/hkanpak/Comp390Project/src/

# Then on MN5:
ssh mn5-gpu "module purge; module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1; \
             cd /gpfs/projects/etur02/hkanpak/Comp390Project/build && \
             make -j20 bert_dks_multigpu"
# (~30 s build)
```

### Generate a new Nsight Systems trace
```bash
ssh mn5-gpu
cd /gpfs/projects/etur02/hkanpak/Comp390Project
salloc --qos=acc_debug --account=etur02 --gres=gpu:4 --time=00:30:00 bash -lc '
  module purge && module load cuda/12.8 nccl/2.24.3-1 && mkdir -p traces
  for MODE in 0 1; do
    DKS_ROTATE=$MODE nsys profile \
      --trace=cuda,nvtx,osrt,nccl \
      --capture-range=nvtx --nvtx-capture=bootstrap_sparse_3 \
      --output=traces/trace_$([ $MODE = 0 ] && echo prefetch || echo dksrot) \
      --force-overwrite=true ./build/bin/bert_dks_multigpu 4
  done
'
# Pull to local Mac:
rsync -avh mn5-gpu:/gpfs/projects/etur02/hkanpak/Comp390Project/traces/*.nsys-rep ~/nexus-traces/
open -a "NVIDIA Nsight Systems" ~/nexus-traces/trace_dksrot.nsys-rep
```

### Run the standalone bootstrap test (single-GPU, MAE check)
```bash
sbatch /gpfs/projects/etur02/hkanpak/Comp390Project/scripts/mn5/slurm_bootstrap_n65536.sh
# expected: bootstrap MAE = 2.25e-6 PASS, time ~2.3 s (with prefetch)
```

## 7. Key Files Cheat Sheet

| File | Role | Phase added/modified |
|---|---|---|
| `src/util/nvtx_tracer.cuh` | NVTX RAII wrapper | Phase 5 |
| `src/nexus_eval/galois_key_store.cuh` | Double-buffered prefetch + cudaHostRegister | Phase 1 |
| `src/nexus_eval/ckks_evaluator.{cu,cuh}` | rotate dispatch (key streaming + DKS rotation hook) | Phase 1, 3 |
| `src/nexus_eval/bootstrapping/Bootstrapper.cu` | Bootstrap_3 + 8 prefetch hooks in BSGS loops | Phase 1 |
| `src/multi_gpu/distributed_context.{cu,cuh}` | RotationWorkspace + Worker threads + dispatch_to_all_gpus | Phase 3, 4a, 4b |
| `src/multi_gpu/keyswitching/galois_oa.cu` | dist_rotate_phantom_inplace (4-phase pipeline) | Phase 3 |
| `src/multi_gpu/keyswitching/output_aggregation.cu` | partial_key_switch_inner_prod kernel | (existing) |
| `src/multi_gpu/keyswitching/dist_galois_key_store.cuh` | Per-GPU digit sharding | (existing) |
| `src/benchmarks/bert_dks_multigpu.cu` | Main BERT layer benchmark, A/B harness | (existing, modified Phase 3) |
| `multiNEXUS.md` | Tables 4.1/4.2 with all measured numbers + iteration history | continuously |
| `TRACING.md` | NVTX range map + Nsight viewing guide | Phase 5 |

## 8. Important Gotchas & Lessons

1. **`cudaMemcpyAsync` from pageable memory is sync.** Always `cudaHostRegister` source
   for true async H→D. (See `feedback_pinned_memory_required_for_async.md`.)

2. **Rule of Five matters in CUDA-managing classes.** `DistributedCiphertext` had a
   user-declared destructor which suppressed implicit move constructor → assignment
   used shallow copy of GPU pointers → double-free. Fix: explicit move ctor/assignment,
   delete copy.

3. **PhantomContext destructor uses `cudaFreeAsync` on captured stream.** If the stream
   was destroyed first (or thread-local default_stream rotated), it crashes. Pattern:
   in `DistributedContext::destroy()`, `release()` GPU 1..N-1 contexts (intentional leak)
   and only destroy GPU 0's. Same caveat for `PhantomCiphertext` in workspace.

4. **NCCL straggler wait is real.** When 4 GPUs hit `ncclAllReduce` at slightly different
   times (due to launch jitter), AllReduce wall time = max(launch_time + comm_kernel)
   per call. The fast GPUs sit idle waiting.

5. **`acc_debug` QOS on MN5 has 2-hour wall but build steps eat into it.** If a SLURM
   job needs more than ~1.5 hr of compute, separate the build (one job) from the run
   (no-rebuild job).

6. **`bert_dks_multigpu 1` is intentionally a no-op** — DKS has no benefit at 1 GPU
   and key sharding would OOM at N=65536.

7. **`step_to_idx` map MUST be heap-allocated and outlive the bench** when registering
   the DKS rotation function via `Evaluator::enable_dks_rotation` — see line ~417 in
   `bert_dks_multigpu.cu` (`static auto *step_to_idx_fn = new ...`).

8. **NTL/GMP must be installed at `/gpfs/projects/etur02/hkanpak/local/`** for the
   bootstrap polynomial Remez evaluation. NTL not on MN5 by default.

## 9. Memory System (auto-loaded each session)

Located at `~/.claude/projects/-Users-...-Comp390/memory/`. Key files:
- `MEMORY.md` — index
- `feedback_pinned_memory_required_for_async.md` — the cudaHostRegister insight
- `feedback_shallow_copy_keys.md` — FIDESlib pattern for key distribution
- `feedback_validate_gpu_utilization.md` — must prove ALL GPUs run kernels
- `reference_mn5_setup.md` — SLURM, modules, project paths
- `project_state.md` — high-level project status (may be stale)
- `project_aws_quota.md` — AWS blocked, MN5 is the target
- Other `project_*.md` notes

These auto-load. Cite them; trust but verify (especially "9 days old" warnings).

## 10. The Cinnamon/Cerium Reference Note

Three GitHub repos cloned at `/tmp/CinnamonTutorial`, `/tmp/Cinnamon`,
`/tmp/CinnamonCompiler`. Cinnamon is a Python DSL → custom-ASIC-ISA compiler
(ASPLOS 2025, Jayashankar et al.), **not** GPU code. Performance numbers in the paper
come from architectural simulation, not real hardware. Useful only as algorithmic
reference (their `keyswitch_digits.h` digit decomposition table validates our (1,4)
choice; their `CommonReceiveEliminatorPass` and `HoistInputBroadcastPass` inspired
our prefetch and persistent-buffer design). Don't try to "run" Cinnamon on GPUs;
it doesn't compile to CUDA.

## 11. Quick Status Check Commands

```bash
# What jobs are queued/running?
ssh mn5-gpu "squeue -u koc971580 -o '%.10i %.20j %.8T %.10M %.20R'"

# What's the latest log?
ssh mn5-gpu "ls -t /gpfs/projects/etur02/hkanpak/logs/ | head -5"

# Read latest BERT layer result
ssh mn5-gpu "tail -30 \$(ls -t /gpfs/projects/etur02/hkanpak/logs/bert*.out | head -1)"

# Check binary timestamps
ssh mn5-gpu "ls -la /gpfs/projects/etur02/hkanpak/Comp390Project/build/bin/bert_dks_multigpu"
```

## 12. If You Want to Continue Optimization

1. **Open the trace first.** `open -a "Nsight Systems" ~/nexus-traces/trace_dksrot.nsys-rep`.
   Zoom into one `dist_rotate_phantom step=N` range. See where time really goes.
   Don't optimize what `nsys stats` says is < 5% of total.

2. **Quickest win: 4d (NCCL straggler fix).** Add `cudaEventRecord` barriers before
   `ncclAllReduce` in `output_aggregation.cu`. Few hours of work, possibly 200-500 ms
   per bootstrap if straggler tail is real.

3. **Biggest theoretical win: 4c (per-digit modup).** Modify Phantom's `rns_tool.modup`
   to take a digit range. Each GPU only NTTs its owned digits. Cuts NTT time 4×, which
   is 40% of bootstrap. Could push to ~1.3 s/bootstrap. 1-2 days of work, Phantom-internals risk.

4. **Don't bother with**: persistent local_cts further (already done, no win),
   thread pool tweaks (4b already exhausts that), or "hoisted input broadcast"
   (modup is the cost, broadcast is downstream of it).

## 13. Open Questions / Decisions for the User

- **Ship at 2.16× and call it done?** Defensible writeup point. multiNEXUS.md reflects
  this.
- **Or push to 4× via Phase 4c?** Real engineering, real risk, real gain. Worth it
  if the project deliverable demands sub-100s 12-head BERT.
- **Multi-node?** Infrastructure exists (`generate_multinode`, MPI). Would need ~2 days
  of integration. Could push to 8 GPUs (2 nodes) for further scaling demo.

---

**End of handoff.** Drop this entire file as the first message of a new Claude Code
session in `/Users/a90/Documents/COURSES/SPRING2026/Comp390/Comp390Project/` —
CLAUDE.md auto-loads context. Drop this file for full detail.
