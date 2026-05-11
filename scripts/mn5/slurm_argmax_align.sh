#!/bin/bash
#SBATCH --job-name=align-argmax
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/argmax_align_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/argmax_align_%j.err

# Lane ALIGN-Argmax — NEXUS Argmax alignment (logN=15, poly_degree=32,768).
#
# Mimics NEXUS's argmax_test (vendor/nexus/cuda/src/main.cu line 108).
# Two configurations measured:
#   (a) NEXUS default vocab=8 (single-GPU + 4-batch multi-GPU throughput)
#   (b) BERT vocab=30,522 (single-GPU + 4-batch multi-GPU throughput)
#
# NEXUS published 2.48 s for vocab=30,522 on A100 (paper Table III).
# Median of 3 trials each.

set -e

echo "════════════════════════════════════════════════════════════"
echo "  ALIGN-Argmax — NEXUS Argmax alignment, logN=15 (N=32,768)"
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

echo ">>> Configuration A: vocab=8 (NEXUS default test)"
echo ">>> Running argmax_align_n32k --vocab 8 --n-gpus 4 --batches 4 --trials 3 ..."
${PROJECT}/build/bin/argmax_align_n32k --vocab 8 --n-gpus 4 --batches 4 --trials 3
RC_A=$?

echo ""
echo "════════════════════════════════════════════════════════════"
echo ">>> Configuration B: vocab=30522 (BERT vocab)"
echo ">>> Running argmax_align_n32k --vocab 30522 --n-gpus 4 --batches 4 --trials 3 ..."
${PROJECT}/build/bin/argmax_align_n32k --vocab 30522 --n-gpus 4 --batches 4 --trials 3
RC_B=$?

echo ""
echo "Exit codes: vocab=8 → ${RC_A}, vocab=30522 → ${RC_B}"
echo "End: $(date)"
exit $((RC_A | RC_B))
