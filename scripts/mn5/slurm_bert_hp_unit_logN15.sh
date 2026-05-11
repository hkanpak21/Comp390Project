#!/bin/bash
#SBATCH --job-name=hp-unit-n32k
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/bert_hp_unit_logN15_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/bert_hp_unit_logN15_%j.err

# MEASURE-01 — Goal 2 unit run: 1-head × 2-layer @ uniform logN=15 on 1 GPU.
# Produces the timing pair (t_layer_1, t_layer_2) consumed by MEASURE-02's
# saturation check (target: |t_2 - t_1| / t_1 <= 5%). Extrapolation to full
# BERT (Section 7) uses t_per_head_per_layer = mean(t_1, t_2).
#
# Three trials in one job for median-of-3.

set -e

echo "════════════════════════════════════════════════════════════"
echo "  MEASURE-01 — HP-BERT unit run, 1 head × 2 layers @ N=32768 (1× H100)"
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

for TRIAL in 1 2 3; do
    echo ""
    echo "═══════════════ TRIAL ${TRIAL}/3 ═══════════════"
    date
    ${PROJECT}/build/bin/bert_hp_multigpu \
        --N 32768 \
        --n-gpus 1 \
        --heads 1 \
        --layers 2
    RC=$?
    echo "[TRIAL ${TRIAL}] exit=${RC}"
    if [ ${RC} -ne 0 ]; then
        echo "[TRIAL ${TRIAL}] FAILED — aborting remaining trials"
        exit ${RC}
    fi
done

echo ""
echo "All 3 trials complete."
echo "Saturation check: extract t_layer_1 and t_layer_2 per trial from stdout,"
echo "  saturated iff (|t_2 - t_1| / t_1) <= 0.05 for the median trial."
echo "End: $(date)"
exit 0
