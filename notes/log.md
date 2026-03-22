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

<!-- Add new entries below this line, newest at the bottom -->
