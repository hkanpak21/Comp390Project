#!/bin/bash
#SBATCH --job-name=align-matmul
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/matmul_align_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/matmul_align_%j.err

# Lane ALIGN-MatMul — NEXUS MatMul alignment (logN=13, poly_degree=8,192).
#
# Mimics NEXUS's MM_test (vendor/nexus/cuda/src/main.cu line 43): plain
# 4096×768 × ciphertext-encoded 768×64 → 64 output columns. Reports
# amortized per-column time (NEXUS published 1.31 s on A100).
#
# Two configurations measured:
#   (a) Single-GPU baseline on GPU 0
#   (b) Multi-GPU output-channel split across 4 GPUs
#
# Median of 3 trials each.

set -e

echo "════════════════════════════════════════════════════════════"
echo "  ALIGN-MatMul — NEXUS MatMul alignment, logN=13 (N=8,192)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> Running matmul_align_n8k --n-gpus 4 --trials 3 ..."
${PROJECT}/build/bin/matmul_align_n8k --n-gpus 4 --trials 3
RC=$?

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
