#!/bin/bash
#SBATCH --job-name=nexus-scale-2n
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/scale_2node_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/scale_2node_%j.err

echo "════════════════════════════════════════════════════════════"
echo "  NEXUS Scaling — 2 Nodes (8x H100)"
echo "  Job: ${SLURM_JOB_ID}, Nodes: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1 openmpi/4.1.5-gcc

export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5
export NCCL_IB_CUDA_SUPPORT=1
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
BIN=${PROJECT}/build/bin
cd ${PROJECT}

HEADS=12
HIDDEN=768
INNER=768
SEQ=128
PARAMS="--heads ${HEADS} --hidden ${HIDDEN} --inner ${INNER} --seq-len ${SEQ} --gpus-per-node 4"

echo ""
echo "═══ 2-Node Strong Scaling: ${HEADS} heads across 8 GPUs ═══"
echo ""

srun --mpi=pmix ${BIN}/bert_encoder_multinode ${PARAMS}

echo ""
echo "End: $(date)"
