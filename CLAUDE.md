# multiNEXUS — Claude session context

## Project in one sentence
Multi-GPU FHE (CKKS) BERT/LLaMA inference on top of Phantom FHE, targeting 4× H100
single-node on BSC MareNostrum 5, at ring degree N=65536 where bootstrap keys (~62 GB)
don't fit on one GPU.

## Current state (as of 2026-04-27)
Phase 4b is the champion: **2.16× over CPU-streaming baseline** (115.7 s vs 249.6 s for
12-head BERT-base projection). MAE 2.25e-6 on every config. Next deferred phases: 4c
(per-digit modup, biggest win, Phantom-internals risk) and 4d (NCCL straggler fix, few
hours). See `docs/RESULTS_SUMMARY.md` for all numbers and prior-art comparison.

## Hardware
- **Compute:** MN5 ACC partition — 4× H100 64GB SXM per node, NVSwitch
- **User:** `koc971580`, SSH alias `mn5-gpu` (key-based auth, no password needed)
- **Project root on MN5:** `/gpfs/projects/etur02/hkanpak/Comp390Project/`
- **Logs on MN5:** `/gpfs/projects/etur02/hkanpak/logs/`
- **AWS:** blocked (no quota), ignore all `scripts/aws/`

## Build on MN5
```bash
ssh mn5-gpu
module purge && module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1
cd /gpfs/projects/etur02/hkanpak/Comp390Project/build
make -j20 bert_dks_multigpu     # ~30 s
```

## Run the champion benchmark
```bash
sbatch /gpfs/projects/etur02/hkanpak/Comp390Project/scripts/mn5/slurm_bert_dks.sh
# output: /gpfs/projects/etur02/hkanpak/logs/bert_dks_<JOBID>.out
```

## Sync local → MN5
```bash
rsync -avz --include='*/' --include='*.cu' --include='*.cuh' --exclude='*' \
  src/ mn5-gpu:/gpfs/projects/etur02/hkanpak/Comp390Project/src/
```

## Key source files
| File | Role |
|---|---|
| `src/benchmarks/bert_dks_multigpu.cu` | Main benchmark (champion) |
| `src/nexus_eval/galois_key_store.cuh` | Async prefetch + cudaHostRegister (Phase 1) |
| `src/nexus_eval/bootstrapping/Bootstrapper.cu` | bootstrap_3 + 8 prefetch hooks |
| `src/multi_gpu/distributed_context.{cu,cuh}` | RotationWorkspace + persistent workers |
| `src/multi_gpu/keyswitching/galois_oa.cu` | dist_rotate_phantom_inplace (Phase 3) |
| `src/multi_gpu/keyswitching/output_aggregation.cu` | partial_key_switch_inner_prod |
| `src/util/nvtx_tracer.cuh` | NVTX RAII macros |

## Non-negotiable lessons (do not rediscover)
1. `cudaMemcpyAsync` from pageable host memory is silently synchronous. Always
   `cudaHostRegister` the source buffer first. This was the Phase 1 insight (10.7 s → 2.3 s).
2. `cudaMalloc` in a hot path kills performance. Phase 3 v1 was 2.4× SLOWER than the
   baseline because of 2,400 mallocs per BERT layer. Use persistent workspaces.
3. C++ Rule of Five is mandatory for GPU-owning classes. Declared destructor suppresses
   implicit move → shallow copy of device pointers → double-free.
4. `PhantomContext`/`PhantomCiphertext` dtors call `cudaFreeAsync` on a captured stream.
   In `DistributedContext::destroy()`, `release()` GPU 1..N-1 contexts — only destroy GPU 0's.
5. NTT kernels are 40% of bootstrap time (not 15% as estimated). Profile before optimizing.
   Traces in `~/nexus-traces/trace_dksrot.nsys-rep` and `trace_prefetch.nsys-rep`.

## Documentation
| File | Purpose |
|---|---|
| `docs/HANDOFF.md` | Full project history and session-start reference |
| `docs/RESULTS_SUMMARY.md` | All 10 measurement tables + NEXUS/Cerium comparison |
| `docs/PI_BRIEFING.md` | Advisor walkthrough with Mermaid system diagrams |
| `docs/NSIGHT_GUIDE.md` | How to open and read the Nsight Systems traces |
| `multiNEXUS.md` | Main technical writeup, Tables 4.1/4.2, all iteration history |
| `TRACING.md` | NVTX range map, nsys run recipe |
| `study.md` | Literature review — NEXUS, Cerium, Cinnamon numbers and citations |

## Profiling
Local traces at `~/nexus-traces/` (13 .nsys-rep + .sqlite pairs, 1.8 GB). Open with
`open -a "NVIDIA Nsight Systems" ~/nexus-traces/trace_dksrot.nsys-rep`.

## Cinnamon / Cerium note
Cinnamon (ASPLOS 2025, Jayashankar et al.) is a Python→ASIC-ISA compiler. Its numbers
are from architectural simulation, not real hardware. Do not try to run it on GPUs.
Cerium (arXiv 2025) is the GPU version but code is not public as of 2026-04. Use
both as algorithmic reference only.
