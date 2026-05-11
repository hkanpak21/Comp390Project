# multiNEXUS — Claude session context

## Project in one sentence
Multi-GPU FHE (CKKS) BERT/LLaMA inference on top of Phantom FHE, targeting
4× H100 single-node and 16× H100 multi-node on BSC MareNostrum 5. Two
prongs: (1) per-operation comparison vs NEXUS at NEXUS's own parameter set
per op, (2) head-parallel chained pipeline at uniform `logN=16`.

## Current strategy (as of 2026-05-11)
**Data-parallel per-op alignment.** Each operation measured at NEXUS's
exact parameter set (logN per op), single-GPU baseline first then
data-parallel across 4 (and optionally 16) H100s. The headline table lives
in `docs/PER_OP_VS_NEXUS.md`. The chained head-parallel pipelines
(`bert_hp_*`, `llama_hp_*`) are kept as the "end-to-end at uniform
`logN=16` that NEXUS's open source can't produce."

Earlier framings now archived (do not resurrect):
- "Phase 4b 2.16× over CPU streaming" — old single-prong narrative.
- "Pipeline parallelism" — abandoned strategy; sources in
  `src/multi_gpu/archive/pipeline/` and dependent benchmarks in
  `src/benchmarks/archive/`.
- "NEXUS uses re-encryption" — fabricated; NEXUS uses three different
  `logN` values per-op, not re-encryption.

## Hardware
- **Compute:** MN5 ACC partition — 4× H100 64GB SXM per node, NVSwitch.
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
sbatch scripts/mn5/slurm_bert_hp_layer.sh           # 4× H100 @ logN=16
sbatch scripts/mn5/slurm_bert_hp_logN15_4node.sh    # 16× H100 @ logN=15
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
| `src/benchmarks/bert_hp_multigpu.cu` / `bert_hp_multinode.cu` | Head-parallel BERT layer (chained) |
| `src/benchmarks/llama_hp_multigpu.cu` / `llama_hp_multinode.cu` | Head-parallel LLaMA decoder layer |
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

## Documentation
| File | Purpose |
|---|---|
| `docs/PER_OP_VS_NEXUS.md` | ★ live alignment table — the headline |
| `docs/HPC_PRIMER.md` | CKKS / RNS / NTT / async-copy refresher |
| `docs/MN5_NCCL_CONFIG.md` | MareNostrum 5 NCCL setup |
| `docs/NSIGHT_GUIDE.md` | how to read the Nsight Systems traces |
| `paper/architecture_guide.md` | architectural reference (note pipeline section is archived) |
| `docs/archive/` | superseded planning / measurement docs (do not extend) |

## Profiling
Local traces at `~/nexus-traces/`. Open with:
```bash
open -a "NVIDIA Nsight Systems" ~/nexus-traces/<trace>.nsys-rep
```

## Cinnamon / Cerium note
Cinnamon (ASPLOS 2025, Jayashankar et al.) is a Python→ASIC-ISA compiler.
Its numbers come from architectural simulation, not real hardware. Do not
attempt to run on GPUs. Cerium (arXiv 2025) is the GPU sibling but code is
not public as of 2026-04. Both are algorithmic reference only.
