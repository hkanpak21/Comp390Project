# GPU Acquisition Guide — Multi-GPU Key-Switching Experiments

## What You Need

| Phase | GPUs | Duration | Purpose |
|-------|------|----------|---------|
| **Build & Validate** | 2x any GPU (even T4) | 1-2 hours | Compile, run `multi_gpu_keyswitch_test --n-gpus 2` |
| **Correctness + Timing** | 4x H100/A100 | 2-4 hours | Validate both algorithms, measure key-switch latency |
| **Scaling Curves** | 1,2,4,8 GPUs | 4-6 hours | Full scaling analysis for the paper |
| **Multi-Node** (Phase 2) | 2+ nodes x 4 GPUs | 4-8 hours | Inter-node experiments |

---

## Option 1: RunPod (Recommended for Quick Results)

**Best for**: 2-4x H100, pay-per-minute, instant availability.

| Instance | GPUs | VRAM | NVLink | Price | Notes |
|----------|------|------|--------|-------|-------|
| 2x H100 SXM | 2x 80GB | 160GB | Yes (NVSwitch) | ~$5.60/hr | Good for validation |
| 4x H100 SXM | 4x 80GB | 320GB | Yes (NVSwitch) | ~$11.20/hr | Full scaling test |
| 8x H100 SXM | 8x 80GB | 640GB | Yes (NVSwitch) | ~$22.40/hr | Complete Phase 1 |

**Estimated cost for Phase 1**: ~$50-80 total (4-6 hours of 4x H100).

**Setup steps**:
```bash
# 1. Create RunPod pod with PyTorch template (has CUDA 12.x, NCCL pre-installed)
# 2. SSH in and clone repo
git clone --recurse-submodules https://github.com/hkanpak21/Comp390Project.git
cd Comp390Project

# 3. Build
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 4. Validate NCCL
./bin/nccl_bandwidth_test --n-gpus 4

# 5. Run key-switching test
./bin/multi_gpu_keyswitch_test --n-gpus 2 --verbose
./bin/multi_gpu_keyswitch_test --n-gpus 4 --verbose
```

---

## Option 2: Lightning AI

**Best for**: 2-4x A100, simple web UI, credit-based.

| Instance | GPUs | Price |
|----------|------|-------|
| 4x A100 40GB | 4x 40GB | ~$6.40/hr |

Similar setup to RunPod. Use `CMAKE_CUDA_ARCHITECTURES=80` for A100.

---

## Option 3: AWS (Limited by Quota)

**Current quota**: 8 vCPUs for P instances in us-east-1.

| Instance | vCPUs | GPUs | Fits quota? |
|----------|-------|------|-------------|
| p3.2xlarge | 8 | 1x V100 16GB | YES (but 1 GPU only) |
| p3.8xlarge | 32 | 4x V100 16GB | NO (32 > 8) |
| p4d.24xlarge | 96 | 8x A100 40GB | NO (96 > 8) |

**Action items to increase quota**:
1. Re-open support case #177442905100620
2. Include academic justification: "Research project for privacy-preserving ML inference on GPUs, supervised by Prof. [PI name] at [university]. Need 4x A100 (p3.8xlarge = 32 vCPUs) for multi-GPU FHE benchmarks."
3. Start with requesting 32 vCPUs (p3.8xlarge) — more likely to be approved
4. Use the account more (run some t3/t2 instances for a week to build history)

**Alternative**: Use **g-family instances** (no quota issue for G instances):
- g5.12xlarge: 4x A10G 24GB, 48 vCPUs — cheaper but A10G has no NVLink
- g6.xlarge: 1x L4 — you already tested this, single GPU only

---

## Option 4: MareNostrum 5 (Best for Paper-Quality Results)

**Best for**: Reproducible results on known HPC hardware, multi-node, free.

| Config | GPUs | Interconnect |
|--------|------|-------------|
| 1 node | 4x H100 64GB | NVSwitch (intra-node) |
| 2 nodes | 8x H100 64GB | InfiniBand NDR200 (inter-node) |
| 4 nodes | 16x H100 64GB | InfiniBand NDR200 |

**Setup**: Use existing SLURM scripts in `scripts/mn5/`.

```bash
# Single node (4 GPUs)
sbatch scripts/mn5/slurm_1node.sh

# Multi-node (4 nodes, 16 GPUs)
sbatch scripts/mn5/slurm_4node.sh
```

**Pros**: Free, reproducible, H100 64GB (more memory than A100 40GB), InfiniBand for multi-node.
**Cons**: Queue wait times, need to transfer code, SLURM job scheduling.

---

## Recommended Sequence

1. **Today**: Spin up 2x H100 on RunPod (~$6/hr). Build, validate correctness. ~1 hour = ~$6.
2. **This week**: Get 4x H100 on RunPod for 3-4 hours. Full scaling analysis (1,2,4 GPUs). ~$35-45.
3. **Next week**: Submit MN5 jobs for paper-quality results (4x H100 single node).
4. **Phase 2**: MN5 multi-node jobs (8-16x H100) for the multi-node extension.
5. **If AWS approves**: Use p3.8xlarge for V100 comparison data point.

**Total estimated cost for Phase 1**: $40-80 on RunPod/Lightning.

---

## Build Checklist (Run on GPU Machine)

```bash
# 1. Verify GPU + NCCL
nvidia-smi
ls /usr/lib/x86_64-linux-gnu/libnccl* || ls /usr/local/cuda/lib64/libnccl*

# 2. Install missing deps if needed
apt-get update && apt-get install -y cmake libgmp-dev libntl-dev

# 3. Clone and build
git clone --recurse-submodules https://github.com/hkanpak21/Comp390Project.git
cd Comp390Project/build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90  # 90 for H100, 80 for A100
make -j$(nproc)

# 4. Validate
./bin/nccl_bandwidth_test --n-gpus $(nvidia-smi -L | wc -l)
./bin/multi_gpu_keyswitch_test --n-gpus 2 --verbose

# 5. Scaling benchmark
for n in 1 2 4; do
    echo "=== $n GPUs ==="
    ./bin/multi_gpu_keyswitch_test --n-gpus $n
done
```
