#!/bin/bash
#SBATCH --job-name=nexus-hp-n32768
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/bert_hp_n32768_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/bert_hp_n32768_%j.err

# S29 PRIOR-ART — Apples-to-apples vs NEXUS at N=32768.
# Runs the same HP-BERT 12-layer × 12-head champion on 4× H100 with the
# CKKS ring degree dropped from N=65536 to N=32768 (NEXUS's bootstrap
# setting). All other params identical. Three trials in one job for
# median-of-3 measurement.

echo "════════════════════════════════════════════════════════════"
echo "  S29 PRIOR-ART — HP-BERT 12L × 12H @ N=32768 (4× H100)"
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

# Median of 3 trials. We launch sequentially in one job so each trial
# gets a fresh GPU state (the binary exits between trials).
for TRIAL in 1 2 3; do
    echo ""
    echo "═══════════════ TRIAL ${TRIAL}/3 ═══════════════"
    date
    ${PROJECT}/build/bin/bert_hp_multigpu \
        --N 32768 \
        --n-gpus 4 \
        --heads 12 \
        --layers 12 \
        --skip-ref
    RC=$?
    echo "[TRIAL ${TRIAL}] exit=${RC}"
    if [ ${RC} -ne 0 ]; then
        echo "[TRIAL ${TRIAL}] FAILED — aborting remaining trials"
        exit ${RC}
    fi
done

echo ""
echo "All 3 trials complete."
echo "End: $(date)"
exit 0
