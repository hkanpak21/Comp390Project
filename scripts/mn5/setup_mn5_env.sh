#!/usr/bin/env bash
# setup_mn5_env.sh — Environment setup for MareNostrum 5 ACC partition
# Source this in your shell or add to ~/.bashrc on MN5:
#   source ~/comp390/scripts/mn5/setup_mn5_env.sh

# ── Modules ───────────────────────────────────────────────────────────────────
module purge
module load cuda/12.8
module load cmake/3.30.5
module load nccl/2.24.3-1
module load openmpi            # For multi-node MPI

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
