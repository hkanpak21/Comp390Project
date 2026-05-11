# multiNEXUS — Multi-GPU FHE Transformer Inference

Extending **NEXUS** (Zhang et al., NDSS 2025) — the first non-interactive
pure-FHE transformer inference protocol — to multi-GPU execution on 4× H100
single-node and 16× H100 multi-node (BSC MareNostrum 5).

**Current focus:** per-operation comparison against NEXUS at NEXUS's own
parameter set per op, plus a head-parallel chained pipeline reference at
uniform `logN=16`. See [`docs/PER_OP_VS_NEXUS.md`](docs/PER_OP_VS_NEXUS.md)
for the live alignment table and [`docs/HPC_PRIMER.md`](docs/HPC_PRIMER.md)
for the CKKS / GPU background.

---

## The setting

NEXUS measures BERT-base end-to-end on 4× A100 in 37.3 s. They use **three
different ring degrees** internally: `logN=13` for MatMul, `logN=15` for
Bootstrap and Argmax, `logN=16` for GELU / LayerNorm / Softmax (key sizes at
the highest N don't fit on a single A100 for the easier ops). Their open
source covers per-op tests at those parameter sets but does not chain them
end-to-end.

Our work has two prongs:

1. **Per-op comparison at NEXUS's own parameter set per op.** Apples-to-apples
   single-GPU vs NEXUS-on-H100, then 4-GPU and 16-GPU data-parallel scaling.
   This is the headline table in [`docs/PER_OP_VS_NEXUS.md`](docs/PER_OP_VS_NEXUS.md).
2. **A head-parallel chained pipeline at uniform `logN=16`** (the strictly
   hardest setting), which produces an end-to-end number NEXUS's open source
   cannot. Implemented in `bert_hp_multigpu` / `bert_hp_multinode` and
   `llama_hp_multigpu` / `llama_hp_multinode`.

---

## Repository structure

```
Comp390Project/
├── src/
│   ├── nexus_eval/                  # Single-GPU FHE evaluator (CKKS ops, Bootstrapper)
│   │   ├── galois_key_store.cuh     # Async prefetch + cudaHostRegister
│   │   ├── ckks_evaluator.{cu,cuh}  # rotate dispatch, DKS rotation hook
│   │   ├── matrix_mul.cu            # matrix_mul_range for output-channel split
│   │   └── bootstrapping/Bootstrapper.cu
│   ├── multi_gpu/                   # Multi-GPU infrastructure
│   │   ├── distributed_context.{cu,cuh}    # RotationWorkspace, persistent workers
│   │   ├── distributed_eval.{cu,cuh}
│   │   ├── keyswitching/                   # DKS rotation, partial KS, T-MODUP
│   │   ├── comm/                           # NCCL primitives
│   │   ├── partition/                      # RNS limb → GPU assignment
│   │   ├── overlap/                        # CUDA stream management
│   │   └── archive/                        # deprecated pipeline-parallel infra
│   ├── benchmarks/                  # Benchmark binaries
│   │   ├── *_align_*.cu             # per-op NEXUS-aligned microbenchmarks
│   │   ├── *_mgpu_align.cu          # per-op data-parallel microbenchmarks
│   │   ├── bert_hp_*.cu             # head-parallel BERT layer (chained)
│   │   ├── llama_hp_*.cu            # head-parallel LLaMA decoder layer
│   │   ├── README.md                # full catalogue
│   │   └── archive/                 # superseded benchmarks
│   └── util/nvtx_tracer.cuh         # NVTX RAII macros
├── scripts/
│   ├── mn5/                         # MareNostrum 5 SLURM scripts (active)
│   ├── regression/                  # correctness + perf regression harness
│   └── archive/aws/                 # AWS scripts (quota blocked, kept for provenance)
├── profiling/                       # Nsight Systems / Nsight Compute run scripts
├── experiments/
│   ├── results/                     # Per-job raw outputs
│   ├── plots/
│   └── archive/                     # Old AWS-era results index
├── docs/
│   ├── PER_OP_VS_NEXUS.md           # ★ live alignment table (the headline)
│   ├── HPC_PRIMER.md                # CKKS / RNS / NTT / async-copy refresher
│   ├── MN5_NCCL_CONFIG.md           # MareNostrum 5 NCCL setup
│   ├── NSIGHT_GUIDE.md              # how to read the Nsight traces
│   └── archive/                     # superseded planning / measurement docs
├── paper/                           # LaTeX manuscript and figures
├── notes/                           # Research log
└── vendor/
    ├── phantom/                     # Phantom FHE (modified, CUDA-native CKKS)
    └── nexus/                       # NEXUS reference implementation
```

---

## Build (on MN5)

**Prerequisites:** CUDA 12.x, NCCL 2.24+, CMake 3.20+, NTL/GMP at
`/gpfs/projects/etur02/hkanpak/local/`.

```bash
ssh mn5-gpu
module purge && module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1
cd /gpfs/projects/etur02/hkanpak/Comp390Project
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90    # H100 = sm_90
make -j20
```

Every benchmark binary links NTL at runtime, so SLURM scripts must export
`LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH`
before launching.

---

## Run a per-op alignment microbenchmark

```bash
# single-GPU baseline at NEXUS logN=15
sbatch scripts/mn5/slurm_bootstrap_align.sh

# data-parallel (4× H100)
sbatch scripts/mn5/slurm_bootstrap_mgpu_align.sh

# output: /gpfs/projects/etur02/hkanpak/logs/bootstrap_*.out
```

The full per-op recipe (one row per operation) is in
[`docs/PER_OP_VS_NEXUS.md`](docs/PER_OP_VS_NEXUS.md) §4.

## Run the chained head-parallel pipeline

```bash
# single-node 4× H100
sbatch scripts/mn5/slurm_bert_hp_layer.sh        # logN=16
sbatch scripts/mn5/slurm_bert_hp_n32768.sh       # logN=15

# multi-node 16× H100
sbatch scripts/mn5/slurm_bert_hp_logN15_4node.sh
sbatch scripts/mn5/slurm_bert_hp_n32768_4node.sh
```

---

## Profiling

Nsight Systems traces from MN5 land in `~/nexus-traces/` locally after
`sync_to_mn5.sh`. Open with:

```bash
open -a "NVIDIA Nsight Systems" ~/nexus-traces/<trace>.nsys-rep
```

NCU recipes for kernel-level analysis are in `scripts/mn5/ncu_profile_*.sh`.

---

## Key papers

- **NEXUS**: Zhang et al., "NEXUS: Secure and Non-Interactive Transformer
  Inference on Encrypted Data", NDSS 2025.
  [Code](https://github.com/zju-abclab/NEXUS)
- **Cerium**: Jayashankar et al., "Cerium: A Compiler and Runtime for FHE on
  GPUs", arXiv:2512.11269, December 2025. (Code not public as of 2026-04.)
- **Cinnamon**: Jayashankar et al., ASPLOS 2025. Simulated ASIC, not real
  GPU hardware — algorithmic reference only.
- **Phantom**: encryptorion-lab/phantom-fhe — GPU-native CKKS.

## License

GPL-3.0 (inherited from NEXUS and Phantom submodules).
