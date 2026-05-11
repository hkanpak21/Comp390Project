#!/bin/bash
#SBATCH --job-name=l38b-pack-smoke
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/l38b_pack_smoke_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/l38b_pack_smoke_%j.err

# SIMD slot-folded HP-LLaMA smoke. Validates pack→bootstrap→unpack
# at LAYER BOUNDARIES matches the unpacked reference within MAE 1e-5.
# 4 GPUs × 8 heads × 1 layer @ logN=15 (NEXUS-comparable),
# llama-3-8b model. Reference path runs unpacked sequentially (--skip-ref off).

echo "════════════════════════════════════════════════════════════"
echo "  l38b-pack-smoke — verify SIMD slot-folded BS#4 correctness"
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

echo "════ TEST 1: --pack-bootstrap, 4 GPUs × 8 heads × 1 layer @ logN=15 ════"
${PROJECT}/build/bin/llama_hp_multigpu \
    --N 32768 \
    --n-gpus 4 \
    --heads 8 \
    --layers 1 \
    --model llama-3-8b \
    --pack-bootstrap
RC1=$?
echo "TEST 1 (packed) exit=${RC1}"

echo ""
echo "End: $(date)"
exit ${RC1}
