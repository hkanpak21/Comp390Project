# Research Log — NEXUS Multi-GPU Project

**Format**: Append-only. Each entry dated. Record what was tried, what happened, conclusions, and next steps. This file is read by future Claude instances to catch up on project state.

---

## 2026-03-22 — Project Initialization

**What we did**:
- Initialized git repo, connected to `https://github.com/hkanpak21/Comp390Project.git`
- Added NEXUS (`zju-abclab/NEXUS`) and Phantom (`encryptorion-lab/phantom-fhe`) as git submodules under `vendor/`
- Created full project folder structure per plan
- Wrote top-level `CMakeLists.txt` that builds `src/multi_gpu/` as a static library against NCCL + CUDA
- Wrote AWS scripts: `launch_p4d_spot.sh`, `setup_p4d_env.sh`, `parallelcluster_config.yaml`
- Wrote MN5 SLURM scripts: `slurm_1node.sh` (4×H100), `slurm_4node.sh` (16×H100)
- Wrote profiling scripts: `nsys_profile.sh`, `ncu_ntt_profile.sh`

**Current state**:
- Repo is initialized but NOT YET built — no GPU binaries exist
- Submodules cloned but NEXUS/Phantom not yet compiled
- Still need to: check dev box GPU type, build NEXUS on dev box, run single-GPU baseline

**Immediate next steps**:
1. Check dev box GPU: `nvidia-smi`
2. Determine `CMAKE_CUDA_ARCHITECTURES` value for dev box GPU
3. Build Phantom: `cd vendor/phantom && mkdir build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=XX && make -j$(nproc)`
4. Build NEXUS GPU path: `cd vendor/nexus/cuda && mkdir build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=XX && make -j$(nproc)`
5. Run NEXUS BERT-base inference → record latency (expect ~37s on A100, different on dev box GPU)

**Key design decisions made**:
- Use `p4d.24xlarge` spot (~$4.41/hr) for multi-GPU work; dev box handles Wk 1–4 (free)
- MareNostrum 5 (already have access) for final paper-quality multi-node experiments
- Targeting HPC/systems venues (SC, PPoPP, ICS, EuroSys) — novelty is systems/parallelism, not cryptography
- RNS limb-level parallelism following Cerium's approach: GPU c handles limbs where `i % n_gpus == c`
- Two key-switching algorithms to implement and compare: Input Broadcast vs. Output Aggregation

---

## 2026-03-24 — Project Status Assessment

**Purpose**: Full audit of repository state, implementation progress, and readiness for next phase.

### What exists and works

**Infrastructure (100% complete)**:
- Git repo with two submodules (NEXUS, Phantom) — both fully cloned and populated
- `CMakeLists.txt` — properly structured with CUDA/NCCL/MPI detection, Phantom subdirectory integration, benchmark targets guarded by `if(EXISTS ...)`
- `scripts/aws/` — 3 production-ready scripts (spot launcher, env setup, ParallelCluster YAML)
- `scripts/mn5/` — 3 production-ready SLURM scripts (env setup, 1-node job, 4-node job)
- `profiling/` — 2 real scripts (Nsight Systems, Nsight Compute with kernel-targeted profiling)
- `study.md` — 300+ line literature review covering NEXUS, Cerium, and 10+ related systems; explains RNS-CKKS internals, key-switching algorithms, and multi-GPU scaling strategy
- `README.md` — complete project documentation with build instructions and hardware targets
- `.gitignore`, `.gitmodules` — properly configured

**Vendor libraries (100% present, 0% built)**:
- `vendor/phantom/` — Full Phantom FHE GPU-native CKKS library (evaluate.cu=92KB, polymath.cu=34KB, NTT kernels, key-switching, etc.)
- `vendor/nexus/` — Full NEXUS codebase with SEAL 4.1, GPU inference path, data generation scripts

### What does NOT exist yet

**Core implementation (0%)**:
- `src/multi_gpu/comm/` — only `.gitkeep`. Needs: NCCL wrappers for ciphertext RNS limb transfers (AllGather, AllReduce, broadcast)
- `src/multi_gpu/partition/` — only `.gitkeep`. Needs: RNS limb-to-GPU assignment logic (`limb_i → GPU(i % n_gpus)`)
- `src/multi_gpu/keyswitching/` — only `.gitkeep`. Needs: Input Broadcast and Output Aggregation key-switching algorithms
- `src/multi_gpu/overlap/` — only `.gitkeep`. Needs: CUDA stream management, compute-communication overlap, optional CudaGraph capture
- `src/benchmarks/` — only `.gitkeep`. Needs: `bert_inference.cu`, `bootstrapping_bench.cu`, `nccl_bandwidth_test.cu`

**Experimental data (0%)**:
- `experiments/results/` — empty
- `experiments/plots/` — empty

**Paper (0%)**:
- `paper/` — only `.gitkeep`

### Build status

The project has **never been built**. Specifically:
1. Phantom has not been compiled (no `build/` directory under `vendor/phantom/`)
2. NEXUS GPU path has not been compiled
3. The dev box GPU type is unknown (no `nvidia-smi` output recorded)
4. The top-level CMakeLists.txt will produce an empty `nexus_multi_gpu` static library (GLOB finds no .cu files) but won't error — the benchmark `if(EXISTS ...)` guards prevent missing-file failures

### Critical path to first results

1. **Verify GPU hardware**: Run `nvidia-smi`, determine CUDA architecture
2. **Build Phantom**: `cd vendor/phantom && mkdir build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=XX && make -j$(nproc)`
3. **Build NEXUS GPU path**: `cd vendor/nexus/cuda && mkdir build && cd build && cmake .. && make -j$(nproc)`
4. **Run NEXUS baseline**: Single-GPU BERT-base inference, record latency
5. **Implement `nccl_bandwidth_test.cu`**: Simplest benchmark, validates NCCL setup
6. **Implement `src/multi_gpu/comm/`**: NCCL wrappers for RNS limb arrays
7. **Implement `src/multi_gpu/partition/`**: Limb assignment strategy
8. **Implement key-switching**: Input Broadcast first (simpler), then Output Aggregation
9. **Implement `bert_inference.cu`**: End-to-end multi-GPU benchmark
10. **Profile and optimize**: Use existing Nsight scripts

### Risk assessment

- **No GPU available on dev box?** — All multi-GPU work requires AWS p4d or MN5. Dev box may only support build verification
- **NEXUS build complexity** — NEXUS depends on modified SEAL 4.1; build may require manual patching
- **Phantom API stability** — Phantom's internal API (ciphertext memory layout, RNS representation) must be understood deeply for limb-level partitioning. The Phantom library study (completed this session) documents these internals
- **NCCL + MPI interaction** — Multi-node NCCL initialization via MPI needs careful setup; ParallelCluster config already addresses this

### Summary

The project is in a **well-scaffolded but pre-implementation state**. All infrastructure, scripts, documentation, and research groundwork are complete. The vendor libraries are present but unbuilt. Zero lines of core implementation code exist. The immediate priority is building the vendor libraries and running the NEXUS single-GPU baseline to establish ground truth before writing any multi-GPU code.

---

<!-- Add new entries below this line, newest at the bottom -->
