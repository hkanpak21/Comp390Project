#!/bin/bash
#SBATCH --job-name=nexus-matmul-split
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/matmul_split_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/matmul_split_%j.err

# Lane H / F1 — MatMul output-channel split smoke test.
# Two std::threads each compute half the output columns of a 64×768
# multiply_plain + add_many matmul. Acceptance: per-column MAE < 1e-6
# between the split-path concatenated output and the single-GPU reference.
#
# Slice gating: F1 PASS unblocks F2 (split QKV), F3 (split FFN-up), F4
# (split FFN-down), and F5 (HP-BERT × MatMul split composition).

echo "════════════════════════════════════════════════════════════"
echo "  F1 — MatMul output-channel split smoke test (2 GPUs)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> Running matmul_split_smoke (2 GPUs, default 64 cols × 768 inner)..."
${PROJECT}/build/bin/matmul_split_smoke --n-gpus 2

EXIT_CODE=$?
echo ""
echo "Exit code: ${EXIT_CODE}"
echo "End: $(date)"
exit ${EXIT_CODE}
