#!/bin/bash
#SBATCH --job-name=l38b-pk-1g
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/l38b_pack_smoke_1g_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/l38b_pack_smoke_1g_%j.err

# 1 GPU SIMD slot-folded smoke — 8 heads packed into 1 ct, BS once,
# unpacked. Validates pack/unpack math without 4-GPU coordination.
# logN=15 for fastest possible run.

echo "════════════════════════════════════════════════════════════"
echo "  l38b-pack-smoke-1g — single GPU pack→bs→unpack validation"
echo "  Job: ${SLURM_JOB_ID}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "════ TEST: 1 GPU × 8 heads × 1 layer @ logN=15, --pack-bootstrap ════"
${PROJECT}/build/bin/llama_hp_multigpu \
    --N 32768 \
    --n-gpus 1 \
    --heads 8 \
    --layers 1 \
    --model llama-3-8b \
    --pack-bootstrap
RC=$?
echo "exit=${RC}"
echo "End: $(date)"
exit ${RC}
