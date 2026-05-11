# multiNEXUS — Claude session context

## Project in one sentence
Multi-GPU FHE (CKKS) BERT inference on top of Phantom FHE, targeting 4× H100
single-node and 16× H100 multi-node on BSC MareNostrum 5. Two paper
contributions: (1) per-operation multi-GPU typology vs NEXUS-on-H100
single-GPU baseline, (2) end-to-end BERT inference at uniform `logN=15`,
demonstrated via a 1-head × 2-layer unit + saturation check + extrapolation
to full BERT, then shown under both head-parallel (latency, strong scaling)
and data-parallel (throughput, weak scaling).

## Paper plan (operational source of truth: `docs/prd/PRD-multiNEXUS-paper.md`)

**9 sections + Appendix A:**
1. Abstract
2. Introduction
3. Background (CKKS, NEXUS, prior multi-GPU FHE: Cerium, Cinnamon)
4. Identifying NEXUS on H100 (build from source, per-op baseline)
5. Multi-GPU strategies (DKS, head-parallel, data-parallel-per-op)
6. **Goal 1 — Per-op multi-GPU typology** (six op subsections, 6-field template)
7. **Goal 2 — End-to-end at uniform `logN=15`** (unit + saturation + extrapolation + HP-BERT + data-parallel)
8. Discussion
9. Conclusion + future work
10. **Appendix A** — NEXUS/Phantom modifications + bug-fix log

**Per-op 6-field template (Section 6, used for each of Bootstrap, MatMul, GELU, LayerNorm, Softmax, Argmax):**
1. Aim
2. Parallelization strategy
3. Implementation
4. Result (single-GPU + 4-GPU + 16-GPU)
5. Profiling-grounded explanation (nsys/NCU evidence)
6. Profiling-grounded ceiling (why we cannot push further)

Operations are grouped into typology buckets: **compute-parallel** (MatMul),
**transitional** (GELU, LayerNorm), **data-parallel-throughput** (Bootstrap,
Softmax, Argmax-at-4-GPU). Heterogeneity is the story, not a caveat.

**Goal 2 measurement protocol:**
- Unit: `bert_hp_multigpu --n-gpus 1 --heads 1 --layers 2 --N 32768`.
- Saturation check: time(layer 1) ≈ time(layer 2) within 5%.
- Extrapolation: full BERT = 12 × 12 × per-head-per-layer.
- Strong scaling: HP-BERT at 4, 16 GPUs (we already have 172.32 s and 54.27 s).
- Weak scaling: G concurrent independent single-GPU BERT inferences.

## Vertical-slice work convention

Every unit of work is a **vertical slice** that produces exactly one commit.
Slice IDs use the format `<phase>-<NN>` (e.g. `BUG-01`, `PROFILE-02`,
`MEASURE-03`, `WRITE-S6.gelu`, `DOC-01`).

**Phase vocabulary:**
- `BUG` — bug audit and fix on critical-path code
- `PROFILE` — nsys / NCU trace generation
- `MEASURE` — new measurement runs
- `WRITE` — paper section drafts
- `DOC` — repo documentation (CLAUDE, README, MD files)
- `APPENDIX` — Appendix A content

**Commit message format:**
- Subject: `<phase>(<area>): <imperative summary, ≤72 chars>`
- Body: motivation, then technical detail, then JOBID/log refs
- Footer: `Slice: <slice-id>; Depends-on: <upstream-slice-ids or "none">`
- Trailer: `Co-Authored-By:` only if an agent did the work

**One slice per commit. Never bundle.** This makes the dependency graph
reconstructable from `git log`.

The full slice map (with upstream dependencies) is in
`docs/prd/PRD-multiNEXUS-paper.md` "Vertical slice initial map" table.

## Hardware
- **Compute:** MN5 ACC partition — 4× H100 64 GB SXM per node, NVSwitch.
- **User:** `koc971580`, SSH alias `mn5-gpu` (key-based auth).
- **Project root on MN5:** `/gpfs/projects/etur02/hkanpak/Comp390Project/`
- **Logs on MN5:** `/gpfs/projects/etur02/hkanpak/logs/`
- **NTL/GMP install:** `/gpfs/projects/etur02/hkanpak/local/`
- **AWS:** quota blocked; `scripts/archive/aws/` kept for provenance only.

## Build on MN5
```bash
ssh mn5-gpu
module purge && module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1
cd /gpfs/projects/etur02/hkanpak/Comp390Project/build
make -j20 <target>
```

Every benchmark binary links NTL at runtime. SLURM scripts must export:
```bash
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH
```
before launching the binary, otherwise it dies with `libntl.so.44: cannot
open shared object file`.

## Run the headline microbenchmarks
```bash
sbatch scripts/mn5/slurm_bootstrap_align.sh        # single-GPU @ logN=15
sbatch scripts/mn5/slurm_bootstrap_mgpu_align.sh   # 4-GPU data-parallel
sbatch scripts/mn5/slurm_gelu_align.sh             # single-GPU @ logN=16
sbatch scripts/mn5/slurm_gelu_mgpu_align.sh        # 4-GPU data-parallel
# (analogous: layernorm, softmax, matmul, argmax)
```

## Run the chained pipeline references
```bash
sbatch scripts/mn5/slurm_bert_hp_n32768.sh          # HP-BERT @ logN=15 (Goal 2 strong scaling)
sbatch scripts/mn5/slurm_bert_hp_logN15_4node.sh    # HP-BERT 16× H100 @ logN=15
```

## Sync local → MN5
```bash
rsync -avz --include='*/' --include='*.cu' --include='*.cuh' --exclude='*' \
  src/ mn5-gpu:/gpfs/projects/etur02/hkanpak/Comp390Project/src/
```

## Key source files
| File | Role |
|---|---|
| `src/benchmarks/{bootstrap,gelu,layernorm,softmax,matmul,argmax}_align_*.cu` | Per-op microbenchmarks at NEXUS parameter sets |
| `src/benchmarks/{bootstrap,gelu,layernorm,softmax}_mgpu_align.cu` | Data-parallel multi-GPU per-op microbenchmarks |
| `src/benchmarks/bert_hp_multigpu.cu` / `bert_hp_multinode.cu` | Head-parallel BERT (chained); `--layers N --heads N --N {32768,65536}` |
| `src/benchmarks/llama_hp_multigpu.cu` / `llama_hp_multinode.cu` | Head-parallel LLaMA decoder layer (out of scope for this paper) |
| `src/benchmarks/bootstrap_diagnose.cu` | Kernel-identity proof at NEXUS workload |
| `src/nexus_eval/galois_key_store.cuh` | Async prefetch + cudaHostRegister |
| `src/nexus_eval/matrix_mul.cu` | `matrix_mul_range` for output-channel split |
| `src/nexus_eval/bootstrapping/Bootstrapper.cu` | bootstrap_3 + 8 prefetch hooks |
| `src/multi_gpu/distributed_context.{cu,cuh}` | RotationWorkspace + persistent worker threads |
| `src/multi_gpu/keyswitching/galois_oa.cu` | dist_rotate_phantom_inplace |
| `src/multi_gpu/keyswitching/output_aggregation.cu` | partial_key_switch_inner_prod, T-MODUP STRIDED |
| `src/multi_gpu/keyswitching/dist_galois_key_store.cuh` | per-GPU key sharding (STRIDED) |
| `src/util/nvtx_tracer.cuh` | NVTX RAII macros |

## Non-negotiable lessons (do not rediscover)
1. `cudaMemcpyAsync` from pageable host memory is silently synchronous.
   Always `cudaHostRegister` the source buffer first.
2. `cudaMalloc` in a hot path kills performance. Use persistent workspaces
   (RotationWorkspace pattern).
3. C++ Rule of Five is mandatory for GPU-owning classes. A user-declared
   destructor suppresses implicit move → shallow copy of device pointers
   → double-free.
4. `PhantomContext` / `PhantomCiphertext` dtors call `cudaFreeAsync` on a
   captured stream. In `DistributedContext::destroy()`, `release()` GPU
   1..N-1 contexts — only destroy GPU 0's.
5. NTT kernels are 40% of bootstrap time (not 15% as estimated). Profile
   before optimising.
6. T-MODUP digit-shard ownership must be **STRIDED**, not CONTIGUOUS, when
   `chain_beta < dnum` (otherwise NCCL P2P illegal-memory-access cascade).
7. NEXUS's Phantom keeps scale-mismatch checks ENABLED; ours has them
   commented out. Argmax / chained Phantom paths must reset scale
   explicitly before bootstrap (`x.scale() = SCALE`) or drift accumulates
   silently.
8. The `gelu()` evaluator mutates its input in-place via
   `mod_switch_to_inplace`. Per-call benchmarks must re-encrypt a fresh
   ciphertext per loop iteration; warmup will otherwise deplete the base
   modulus.
9. GELU `coeff_modulus` at logN=16 needs **18** forties between the two
   58s (`{58, 18×40, 58}` = 20 limbs total). Earlier code had `i < 17`
   which produced 19 limbs and exhausted the chain mid-`sgn_eval`.
10. Argmax `argmax_align_n32k.cu` only handles single-ciphertext inputs:
    `vocab ≤ sparse_slots = 8192` at logN=15. NEXUS's published vocab=30,522
    requires multi-cipher tournament logic which this binary does not ship.
    The CLI now refuses cleanly with a FATAL message on `vocab > sparse_slots`.

## Strategies (paper terminology)
- **DKS (Distributed Key-Switching).** Shards key-switch digits across GPUs
  for the bootstrap. Used by `bert_dks_multigpu`. Reference path; not the
  Goal 1 headline framework.
- **HP (Head-Parallel BERT).** Each GPU owns a subset of attention heads
  end-to-end through all layers; `std::thread` per GPU; activations flow
  GPU→GPU. **Strong scaling for per-inference latency.** This is Goal 2's
  latency story. Implemented in `bert_hp_multigpu` / `bert_hp_multinode`.
- **DP (Data-Parallel per-operation).** Each GPU thread owns its own
  `PhantomContext` and runs N/G independent op calls; no inter-GPU comm
  during the call. **Throughput; weak scaling.** This is Goal 1's per-op
  framework AND Goal 2's throughput story.

Earlier framings now archived (do not resurrect):
- "Phase 4b 2.16× over CPU streaming" — old single-prong narrative.
- "Pipeline parallelism" (CtPipeline) — abandoned strategy; sources in
  `src/multi_gpu/archive/pipeline/` and dependent benchmarks in
  `src/benchmarks/archive/`.
- "NEXUS uses re-encryption" — fabricated; NEXUS uses three different
  `logN` values per-op, not re-encryption.

## Documentation
| File | Purpose |
|---|---|
| `docs/prd/PRD-multiNEXUS-paper.md` | ★ paper PRD — operational source of truth |
| `docs/PI_REPORT.md` | ★ PI-facing report (markdown source) |
| `docs/report/main.tex` + `main.pdf` | ★ PI-facing report (LaTeX + compiled PDF) |
| `docs/PER_OP_VS_NEXUS.md` | per-op alignment table with full provenance (JOBID + log path) |
| `docs/HPC_PRIMER.md` | CKKS / RNS / NTT / async-copy refresher |
| `docs/MN5_NCCL_CONFIG.md` | MareNostrum 5 NCCL setup |
| `docs/NSIGHT_GUIDE.md` | how to read the Nsight Systems traces |
| `paper/architecture_guide.md` | architectural reference (legacy CtPipeline section noted as archived) |
| `docs/archive/` | superseded planning / measurement docs (do not extend, do not act on) |

## Profiling
Local traces at `~/nexus-traces/`. Open with:
```bash
open -a "NVIDIA Nsight Systems" ~/nexus-traces/<trace>.nsys-rep
```

Multi-GPU per-op nsys traces land under
`experiments/results/<date>_h100x4_<op>-mgpu-nsys/raw/` (see existing
`mgpu-nsys/` and `pipeline-overhead/` dirs for the file convention).

## Cinnamon / Cerium note
Cinnamon (ASPLOS 2025, Jayashankar et al.) is a Python→ASIC-ISA compiler.
Its numbers come from architectural simulation, not real hardware. Do not
attempt to run on GPUs. Cerium (arXiv 2025) is the GPU sibling but code is
not public as of 2026-04. Both are algorithmic reference only.

## Out of scope for this paper (do not start any of these without explicit ask)
- Slot-axis SIMD packing for HP-BERT (multi-day refactor; required to beat
  NEXUS's 37.3 s end-to-end on a fair workload).
- Multi-cipher argmax tournament (required for vocab=30,522 vs NEXUS's
  published 2.48 s).
- Per-rank context pooling (would lift small-op 16-GPU efficiency from
  9–22% to ~30–50%).
- Layer-pipeline parallelism (different layers on different GPUs).
- HP-LLaMA results (BERT-only paper).
- End-to-end at logN=13 or logN=16 (logN=13 lacks chain depth without
  bootstrap; logN=16 was deprioritized in favor of logN=15).
