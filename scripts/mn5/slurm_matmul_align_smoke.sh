#!/bin/bash
#SBATCH --job-name=align-matmul-smoke
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/matmul_align_smoke_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/matmul_align_smoke_%j.err

# Lane MATMUL-SPLIT-FIX — smoke verification of REAL output-channel split.
#
# Validates that MMEvaluator::matrix_mul_range(cols_lo, cols_hi) actually
# partitions per-column work across 4 GPUs, by:
#   (a) measuring single-GPU full 64-col wall-clock (1 trial)
#   (b) measuring 4-GPU split wall-clock (1 trial, each GPU does 16 cols)
#   (c) checking MAE between single-GPU col 0 and 4-GPU GPU-0 col 0
#       (multi-GPU thread 0 produces cols [0,16), so its first column IS
#       col 0 of the overall result)
#
# Acceptance: exit 0, MAE < 1e-5, multi-GPU wall < single-GPU wall.
# Target speedup ≥ 2× (theoretical 4× minus per-thread setup overhead).

set -e

echo "════════════════════════════════════════════════════════════"
echo "  MATMUL-SPLIT-FIX SMOKE — real output-channel split"
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

echo ">>> Running matmul_align_n8k --n-gpus 4 --smoke (one trial each)..."
${PROJECT}/build/bin/matmul_align_n8k --n-gpus 4 --smoke
RC=$?

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
