#!/bin/bash
#SBATCH --job-name=nexus-hp-smoke
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/bert_hp_smoke_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/bert_hp_smoke_%j.err

# Lane D / S16 — HP-BERT skeleton smoke verification.
# 4 GPUs × 1 head each. Compares each per-GPU head output to a single-GPU
# reference computed on GPU 0. Acceptance: every head MAE < 1e-5.

echo "════════════════════════════════════════════════════════════"
echo "  S16 — HP-BERT skeleton smoke (4 GPUs / 4 heads / 1 trial)"
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

echo ">>> Running bert_hp_multigpu --n-gpus 4 --heads 4 ..."
${PROJECT}/build/bin/bert_hp_multigpu --n-gpus 4 --heads 4
RC=$?

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
