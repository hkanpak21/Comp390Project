# multiNEXUS — Multi-GPU FHE Transformer Inference

Extending **NEXUS** (Zhang et al., NDSS 2025) — the first non-interactive pure-FHE
transformer inference protocol — to multi-GPU execution via distributed key-switching
(DKS) and async key prefetch on 4× H100 GPUs (BSC MareNostrum 5).

**Current result:** 2.16× end-to-end speedup over the CPU-streaming baseline for
BERT-base (N=65536, 12 heads projected: 115.7 s vs 249.6 s), with correctness parity
(MAE 2.25e-6). See `docs/RESULTS_SUMMARY.md` for full comparison vs NEXUS and Cerium.

---

## The core problem

A full set of CKKS bootstrap Galois keys at N=65536 totals ~62 GB. No single H100
(64 GB) has enough headroom to hold them alongside ciphertexts and workspaces. NEXUS
sidesteps this by bootstrapping at N=32768 and re-encrypting; we target N=65536
throughout (no re-encryption). This requires either CPU streaming (our baseline) or
distributing the keys across GPUs (our contribution).

## Approach

Two orthogonal optimisations, combined:

1. **Async key prefetch (Phase 1)** — double-buffer bootstrap Galois keys between
   CPU and GPU. While rotation *N*'s kernel runs on the default stream, H→D for
   rotation *N+1*'s key runs concurrently on a dedicated copy stream.
   Critical implementation detail: the host key store must be pinned via
   `cudaHostRegister`; without it, `cudaMemcpyAsync` silently serialises.
   Win: 10.7 s → 2.3 s per bootstrap (4.69×).

2. **Distributed key-switching (DKS rotation, Phase 3)** — shard the ~62 GB key store
   4-way across GPUs. Each GPU stores β/4 digit blocks of each rotation key.
   Bootstrap rotations are dispatched across all 4 GPUs (partial key-switch inner
   product in parallel, then NCCL AllReduce to aggregate).
   Win over prefetch alone: 2,277 ms → 2,126 ms per bootstrap.

---

## Repository structure

```
Comp390Project/
├── src/
│   ├── nexus_eval/            # Single-GPU FHE evaluator (CKKS ops, Bootstrapper)
│   │   ├── galois_key_store.cuh         # Async prefetch + cudaHostRegister
│   │   ├── ckks_evaluator.{cu,cuh}      # rotate dispatch, DKS rotation hook
│   │   └── bootstrapping/Bootstrapper.cu # bootstrap_3 + 8 prefetch hooks
│   ├── multi_gpu/             # Multi-GPU DKS infrastructure
│   │   ├── distributed_context.{cu,cuh} # RotationWorkspace, persistent workers
│   │   ├── keyswitching/
│   │   │   ├── galois_oa.cu             # dist_rotate_phantom_inplace
│   │   │   ├── output_aggregation.cu    # partial_key_switch_inner_prod kernel
│   │   │   └── dist_galois_key_store.cuh # per-GPU key sharding
│   │   ├── comm/              # NCCL primitives for CKKS limb arrays
│   │   ├── partition/         # RNS limb → GPU assignment
│   │   ├── overlap/           # CUDA stream management
│   │   └── pipeline/          # Multi-node ciphertext pipeline
│   ├── benchmarks/            # All benchmark binaries (.cu)
│   │   └── bert_dks_multigpu.cu         # Main champion benchmark
│   └── util/nvtx_tracer.cuh   # NVTX RAII macros (profiling)
├── scripts/
│   ├── mn5/                   # MareNostrum 5 SLURM scripts (active)
│   └── aws/                   # AWS scripts (inactive — quota blocked)
├── profiling/                 # Nsight Systems / Nsight Compute run scripts
├── experiments/               # Experiment results and plots
│   └── results/2026-03-29_l4_baseline-ops/  # Single-GPU CKKS micro-benchmarks
├── docs/                      # Session documentation
│   ├── HANDOFF.md             # Full project history and session-start reference
│   ├── RESULTS_SUMMARY.md     # All measurement tables + prior-art comparison
│   ├── PI_BRIEFING.md         # Advisor walkthrough with system diagrams
│   └── NSIGHT_GUIDE.md        # How to read the Nsight traces
├── paper/                     # LaTeX manuscript and figures
├── notes/                     # Research log
├── multiNEXUS.md              # Main technical writeup (Tables 4.1/4.2)
├── TRACING.md                 # NVTX range map + Nsight run recipe
├── study.md                   # Literature review (NEXUS, Cerium, Cinnamon)
├── CLAUDE.md                  # Claude Code session context
└── vendor/
    ├── phantom/               # Phantom FHE (modified, CUDA-native CKKS)
    └── nexus/                 # NEXUS reference implementation
```

---

## Build

**Prerequisites:** CUDA 12.x, NCCL 2.24+, CMake 3.20+, NTL/GMP (for Remez bootstrap
polynomials — installed at `/gpfs/projects/etur02/hkanpak/local/` on MN5).

```bash
# On MN5:
module purge && module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90   # H100 = sm_90
make -j20 bert_dks_multigpu
```

Build time: ~30 seconds on MN5 (incremental).

---

## Run the champion benchmark

```bash
sbatch scripts/mn5/slurm_bert_dks.sh
# Output: /gpfs/projects/etur02/hkanpak/logs/bert_dks_<JOBID>.out
# Expected: bootstrap 2,126 ms, 12-head BERT 115.7 s, MAE 2.25e-6 PASS
```

Toggle the two modes via env var:
```bash
DKS_ROTATE=0  # async key prefetch only (2,277 ms/bootstrap)
DKS_ROTATE=1  # + distributed rotation (2,126 ms/bootstrap, current champion)
```

---

## Results

| Phase | Bootstrap/call | 12-head BERT | vs CPU |
|---|---|---|---|
| CPU streaming baseline | — | 249.6 s | 1.00× |
| Phase 0 — DKS storage only | 10,514 ms | 555.3 s | 0.45× ❌ |
| Phase 1 — async prefetch + pinned host | 2,277 ms | 122.8 s | 2.03× |
| Phase 3 v2 — DKS rotation + persistent buffers | 2,143 ms | 116.9 s | 2.14× |
| **Phase 4b — current champion** | **2,126 ms** | **115.7 s** | **2.16×** |

Full breakdown, per-operation BERT layer timing, bootstrap cost composition from
Nsight Systems, and comparison vs NEXUS / Cerium / Cinnamon: see
`docs/RESULTS_SUMMARY.md`.

---

## Profiling

Nsight Systems traces for all configurations are in `~/nexus-traces/` (13 `.nsys-rep`
files). The two champion traces (`trace_prefetch.nsys-rep`, `trace_dksrot.nsys-rep`)
were captured 2026-04-19 on MN5. Opening guide: `docs/NSIGHT_GUIDE.md`.

```bash
open -a "NVIDIA Nsight Systems" ~/nexus-traces/trace_dksrot.nsys-rep
```

---

## Deferred next phases

| Phase | Expected win | Effort | Risk |
|---|---|---|---|
| 4d — NCCL straggler fix | 200–500 ms/bootstrap | Few hours | Low |
| 4c — per-digit modup (NTT sharding) | ~1.3 s/bootstrap (≈4× total) | 1–2 days | High (Phantom internals) |
| 4e — multi-node DKS | 8-GPU 2-node demo | ~2 days | Medium |

See `docs/HANDOFF.md §5` for implementation details on each.

---

## Key papers

- **NEXUS**: Zhang et al., "NEXUS: Secure and Non-Interactive Transformer Inference
  on Encrypted Data", NDSS 2025. [Code](https://github.com/zju-abclab/NEXUS)
- **Cerium**: Jayashankar et al., "Cerium: A Compiler and Runtime for FHE on GPUs",
  arXiv:2512.11269, December 2025.
- **Cinnamon**: Jayashankar et al., ASPLOS 2025 (simulated ASIC, not real GPU hardware).
- **Phantom**: encryptorion-lab/phantom-fhe — GPU-native CKKS.

## License

GPL-3.0 (inherited from NEXUS and Phantom submodules).
