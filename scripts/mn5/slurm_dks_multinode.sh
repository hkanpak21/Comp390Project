#!/bin/bash
#SBATCH --job-name=nexus-dks-mnode
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/dks_multinode_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/dks_multinode_%j.err

echo "════════════════════════════════════════════════════════════"
echo "  multiNEXUS DKS Multi-Node — 2 nodes × 4 H100 = 8 GPUs"
echo "  Job: ${SLURM_JOB_ID}, Nodes: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Step 11: NCCL cross-node AllReduce for DKS bootstrap"
echo "  Key sharding: 9 digits / 8 GPUs = 1-2 owned digits/GPU"
echo "  Expected: ~1,500 ms/bootstrap (vs 2,800 ms on 4-GPU)"
echo ""

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1 openmpi/4.1.5

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0

cd ${PROJECT}

echo ">>> Node layout:"
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 hostname
echo ""

echo ">>> GPU info per node:"
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 \
     nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> Running bert_dks_multinode (2 nodes × 4 GPUs = 8 total)..."
srun --ntasks=2 --ntasks-per-node=1 \
     ${PROJECT}/build/bin/bert_dks_multinode --gpus-per-node 4

echo ""
echo "End: $(date)"
