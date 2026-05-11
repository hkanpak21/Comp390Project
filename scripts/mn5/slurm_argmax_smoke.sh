#!/bin/bash
#SBATCH --job-name=argmax-smoke
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/argmax_smoke_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/argmax_smoke_%j.err

# Lane ARGMAX-FIX smoke test — single GPU, vocab=8, 1 trial.
# Verifies the scale-reset fix lets argmax run all 3 bootstraps without throwing.
# NOT a measurement run; a 30-second pass/fail check.

set -e

echo "════════════════════════════════════════════════════════════"
echo "  ARGMAX-FIX smoke test — single GPU, vocab=8, 1 trial"
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

echo ">>> Running argmax_align_n32k --vocab 8 --n-gpus 1 --batches 1 --trials 1 ..."
${PROJECT}/build/bin/argmax_align_n32k --vocab 8 --n-gpus 1 --batches 1 --trials 1
RC=$?

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
