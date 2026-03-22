#!/usr/bin/env bash
# setup_mn5_env.sh — Environment setup for MareNostrum 5 ACC partition
# Source this in your shell or add to ~/.bashrc on MN5:
#   source ~/comp390/scripts/mn5/setup_mn5_env.sh

# ── Modules ───────────────────────────────────────────────────────────────────
module purge
module load cuda/12.3
module load nccl/2.20.5-cuda12.3
module load openmpi/4.1.5-cuda12.3
module load cmake/3.27.4
module load python/3.11.5   # For plot scripts

# ── NCCL / InfiniBand environment ────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5            # ConnectX-7 adapters
export NCCL_IB_CUDA_SUPPORT=1      # Enable GPUDirect RDMA over InfiniBand
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=ib0

# ── Build shortcuts ───────────────────────────────────────────────────────────
REPO_DIR="${HOME}/comp390"
export PATH="${REPO_DIR}/build/bin:${PATH}"

echo "MN5 environment loaded."
echo "  CUDA: $(nvcc --version 2>/dev/null | grep release | awk '{print $5}' | tr -d ,)"
echo "  Repo: ${REPO_DIR}"
