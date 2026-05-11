#!/bin/bash
#SBATCH --job-name=argmax-v30k
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/argmax_v30k_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/argmax_v30k_%j.err

# Closes the open cell in docs/PER_OP_VS_NEXUS.md §4.4 — Argmax at NEXUS's
# published vocab=30,522 (BERT vocab; padded to 32,768 for QuickMax).
# NEXUS published 2.48 s on A100; the Lane PEROP-FINAL run measured only
# vocab=8 (their default test fixture). We report:
#   (1) single-GPU  → strict apples-to-apples vs NEXUS A100 published 2.48 s
#   (2) 4-GPU       → multi-GPU per-call latency under 4-batch concurrency
# 3 trials each; median reported into PER_OP_VS_NEXUS.md once landed.

set -e

echo "════════════════════════════════════════════════════════════"
echo "  Argmax @ vocab=30,522 (logN=15, padded slots=32,768)"
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
echo ">>> Phase A: Single-GPU baseline (vocab=30522, 1 batch, 3 trials)"
echo ">>> Direct comparison to NEXUS A100 published 2.48 s"
echo "════════════════════════════════════════════════════════════"
${PROJECT}/build/bin/argmax_align_n32k --vocab 30522 --n-gpus 1 --batches 1 --trials 3
RC_A=$?

echo ""
echo "════════════════════════════════════════════════════════════"
echo ">>> Phase B: 4-GPU data-parallel (vocab=30522, 4 batches, 3 trials)"
echo ">>> Per-call latency under 4-batch concurrency"
echo "════════════════════════════════════════════════════════════"
${PROJECT}/build/bin/argmax_align_n32k --vocab 30522 --n-gpus 4 --batches 4 --trials 3
RC_B=$?

echo ""
echo "Exit codes: single-GPU → ${RC_A}, 4-GPU → ${RC_B}"
echo "End: $(date)"
exit $((RC_A | RC_B))
