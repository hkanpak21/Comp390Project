#!/bin/bash
#SBATCH --job-name=nexus-hp-12layer
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/bert_hp_12layer_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/bert_hp_12layer_%j.err

# Lane H / S18 — Full 12-layer × 12-head HP-BERT measurement.
# Lane D's bert_hp_multigpu binary extended (Lane H S18 patch) with
# `--layers 12` chains layers via output-→input so the full BERT-base
# encoder runs end-to-end on 4 GPUs (3 heads/GPU sequential × 12 layers).
# `--skip-ref` because a 12-layer single-GPU reference would take ~24 min;
# per-layer correctness was already verified in S17.

echo "════════════════════════════════════════════════════════════"
echo "  S18 — HP-BERT 12 layer × 12 head (4 GPUs)"
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

echo ">>> Running bert_hp_multigpu --n-gpus 4 --heads 12 --layers 12 --skip-ref ..."
${PROJECT}/build/bin/bert_hp_multigpu --n-gpus 4 --heads 12 --layers 12 --skip-ref
RC=$?

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
