#!/bin/bash
#SBATCH --job-name=nexus-hp-layer
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/bert_hp_layer_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/bert_hp_layer_%j.err

# Lane D / S17 — HP-BERT 12 heads on 4 GPUs (3 heads/GPU sequential).
# Single trial; each per-GPU thread runs 3 heads back-to-back. Verifies
# every head's output against a single-GPU reference (MAE < 1e-5).

echo "════════════════════════════════════════════════════════════"
echo "  S17 — HP-BERT full layer (4 GPUs / 12 heads / 3 per GPU)"
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

echo ">>> Running bert_hp_multigpu --n-gpus 4 --heads 12 ..."
${PROJECT}/build/bin/bert_hp_multigpu --n-gpus 4 --heads 12
RC=$?

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
