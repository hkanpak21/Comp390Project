#!/bin/bash
#SBATCH --job-name=nexus-matmul-qkv
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/matmul_qkv_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/matmul_qkv_%j.err

# Lane H / F2 — Split QKV matmul across 4 GPUs.
# 192 column-matmuls (3 outputs × 64 cols) split as 48 cols/GPU.
# Acceptance: per-column MAE < 1e-6, per-GPU matmul time ~1/4 of single-GPU ref.

echo "════════════════════════════════════════════════════════════"
echo "  F2 — Split QKV matmul (3 × 64 cols across 4 GPUs)"
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

echo ">>> Running matmul_qkv_split_smoke (4 GPUs, default 3×64 cols × 768 inner)..."
${PROJECT}/build/bin/matmul_qkv_split_smoke --n-gpus 4

EXIT_CODE=$?
echo ""
echo "Exit code: ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}
