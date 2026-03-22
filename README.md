# NEXUS Multi-GPU/Multi-Node FHE Inference

Extending **NEXUS** (Zhang et al., NDSS 2025) — the first non-interactive pure-FHE transformer inference protocol — from single-node 4×A100 execution to multi-GPU and multi-node distributed inference via NCCL.

**Target venues**: SC, PPoPP, ICS, EuroSys (systems/HPC contribution)

---

## Research Goal

NEXUS achieves BERT-base inference in **37.3 seconds** on 4×A100 GPUs with only **164 MB** of client-server communication — orders of magnitude less than interactive MPC-based alternatives. This project distributes NEXUS's CKKS homomorphic operations across 8+ GPUs and multiple nodes using RNS limb-level parallelism and NCCL collectives, following the approach of Cerium (Jayashankar et al., arXiv 2025) but applied to NEXUS's specific non-interactive protocol.

**Novel contributions**:
1. First multi-node non-interactive secure transformer inference
2. NCCL-based CKKS ciphertext communication primitives (reusable library)
3. Compute-communication overlap for non-interactive FHE
4. Comparative scaling analysis: AWS p4d (NVSwitch+EFA) vs. MareNostrum 5 (InfiniBand NDR200)
5. Bootstrapping distribution across nodes (inter-node bandwidth as bottleneck analysis)

---

## Repository Structure

```
comp390/
├── vendor/
│   ├── nexus/          # git submodule: zju-abclab/NEXUS (NDSS 2025)
│   └── phantom/        # git submodule: encryptorion-lab/phantom-fhe
├── src/
│   ├── multi_gpu/      # Multi-GPU extension (core contribution)
│   │   ├── comm/           # NCCL primitives for CKKS limb arrays
│   │   ├── partition/      # RNS limb → GPU assignment
│   │   ├── keyswitching/   # Input broadcast & output aggregation KS
│   │   └── overlap/        # CUDA stream mgmt + CudaGraph capture
│   └── benchmarks/     # Benchmark harnesses
├── scripts/
│   ├── aws/            # Spot instance launch, ParallelCluster config
│   └── mn5/            # MareNostrum 5 SLURM scripts
├── profiling/          # Nsight Systems / Nsight Compute scripts
├── experiments/        # Results (CSV/JSON) and plots
├── notes/              # Append-only research log
└── paper/              # LaTeX manuscript
```

---

## Getting Started

### Prerequisites
- CUDA 12.x, NCCL 2.29+, CMake 3.20+
- For multi-node: MPI (OpenMPI or Intel MPI), aws-ofi-nccl (on AWS)

### Clone with submodules
```bash
git clone --recurse-submodules https://github.com/hkanpak21/Comp390Project.git
cd Comp390Project
```

### Build (single-node, adjust arch for your GPU)
```bash
mkdir build && cd build
# A100 = 80, H100 = 90, dev box GPU = check with nvidia-smi
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Reproduce NEXUS single-GPU baseline
```bash
cd vendor/nexus
# Follow vendor/nexus/README.md for SEAL build + GPU inference
```

---

## Hardware Targets

| Platform | Config | Purpose |
|----------|--------|---------|
| Dev box (this machine) | 1× GPU | Build, debug, single-GPU baseline, profiling |
| AWS p4d.24xlarge (spot, ~$4.41/hr) | 8× A100 40GB | Multi-GPU development & benchmarks |
| AWS 2×p4d (ParallelCluster) | 16× A100 | Multi-node AWS benchmarks |
| MareNostrum 5 ACC | 4 nodes × 4× H100 64GB | Production paper experiments |

---

## Key Papers

- **NEXUS**: Zhang et al., "NEXUS: Secure and Non-Interactive Transformer Inference on Encrypted Data", NDSS 2025. [Code](https://github.com/zju-abclab/NEXUS)
- **Cerium**: Jayashankar et al., "Cerium: A Compiler and Runtime for FHE on GPUs", arXiv:2512.11269, Dec 2025.
- **Phantom**: encryptorion-lab/phantom-fhe — GPU-native CKKS library used by NEXUS.

---

## License

GPL-3.0 (inherited from NEXUS and Phantom submodules).
