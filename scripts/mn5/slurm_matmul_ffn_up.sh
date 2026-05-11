#!/bin/bash
#SBATCH --job-name=nexus-matmul-ffn-up
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/matmul_ffn_up_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/matmul_ffn_up_%j.err

# Lane H / F3 — Split FFN-up matmul (768 → 3072) across 4 GPUs.
# Uses the same matmul_split_smoke binary parameterized with FFN-up
# dimensions: 3072 output cols × 768 inner_dim. Each GPU produces
# 3072/4 = 768 output channels.
# Acceptance: per-column MAE < 1e-6, per-GPU FFN-up time ~12 ms
# (down from ~50 ms single-GPU).

echo "════════════════════════════════════════════════════════════"
echo "  F3 — Split FFN-up matmul (3072 cols × 768 inner across 4 GPUs)"
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

echo ">>> Running matmul_split_smoke (4 GPUs, --cols 3072 --inner 768)..."
${PROJECT}/build/bin/matmul_split_smoke --n-gpus 4 --cols 3072 --inner 768

EXIT_CODE=$?
echo ""
echo "Exit code: ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}
