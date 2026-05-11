#!/bin/bash
#SBATCH --job-name=nccl-4node
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/nccl_4node_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/nccl_4node_%j.err

# G1 — NCCL multi-node config on MN5
# 4 nodes × 4 GPUs = 16 GPUs total. 1 MPI rank per GPU.
# Measures all_reduce_perf bandwidth.

echo "════════════════════════════════════════════════════════════"
echo "  NCCL 4-node bandwidth test (G1) — 16× H100 (4 nodes × 4)"
echo "  Job: ${SLURM_JOB_ID}, Nodes: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load gcc/11.4.0 cuda/12.8 nccl/2.24.3-1 openmpi/4.1.5-gcc

NCCL_TESTS=/gpfs/projects/etur02/hkanpak/nccl-tests
BIN=${NCCL_TESTS}/build/all_reduce_perf

# ── NCCL / InfiniBand config for MN5 ─────────────────────────────────────────
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=ib0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

echo ""
echo ">>> Node layout:"
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 hostname
echo ""

echo ">>> GPU info per node:"
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 \
     nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 4-node × 4-GPU all-reduce.
echo "=================================================================="
echo "  multi-node all_reduce_perf : 4 nodes × 4 GPUs = 16 GPUs"
echo "  -b 1M -e 256M -f 2 -g 1 -n 25 -w 5"
echo "=================================================================="
srun --mpi=pmix --ntasks=16 --ntasks-per-node=4 \
     ${BIN} -b 1M -e 256M -f 2 -g 1 -n 25 -w 5

echo ""
echo "End: $(date)"
