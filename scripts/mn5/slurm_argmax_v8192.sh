#!/bin/bash
#SBATCH --job-name=argmax-v8192
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/argmax_v8192_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/argmax_v8192_%j.err

# Argmax at vocab=8192 — the maximum vocabulary that fits in a single
# sparse-encoded ciphertext at NEXUS's logN=15 / sparse_slots=8192 setting.
# This is the largest single-cipher comparison we can run cleanly.
# vocab=30,522 (NEXUS published 2.48 s) requires multi-cipher tournament
# logic that argmax_align_n32k does not implement.
# 3 trials each; median reported into PER_OP_VS_NEXUS.md once landed.

set -e

echo "════════════════════════════════════════════════════════════"
echo "  Argmax @ vocab=8192 (logN=15, padded slots=8,192, log_step=13)"
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

echo "════════════════════════════════════════════════════════════"
echo ">>> Phase A: Single-GPU baseline (vocab=8192, 1 batch, 3 trials)"
echo "════════════════════════════════════════════════════════════"
${PROJECT}/build/bin/argmax_align_n32k --vocab 8192 --n-gpus 1 --batches 1 --trials 3
RC_A=$?

echo ""
echo "════════════════════════════════════════════════════════════"
echo ">>> Phase B: 4-GPU data-parallel (vocab=8192, 4 batches, 3 trials)"
echo "════════════════════════════════════════════════════════════"
${PROJECT}/build/bin/argmax_align_n32k --vocab 8192 --n-gpus 4 --batches 4 --trials 3
RC_B=$?

echo ""
echo "Exit codes: single-GPU → ${RC_A}, 4-GPU → ${RC_B}"
echo "End: $(date)"
exit $((RC_A | RC_B))
