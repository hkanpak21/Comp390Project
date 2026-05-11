#!/bin/bash
#SBATCH --job-name=ncu-boot-inner
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/ncu_boot_inner_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/ncu_boot_inner_%j.err

# Lane NCU-PROFILE — Inner-bootstrap follow-up.
#
# The first bootstrap pass (ncu_profile_bootstrap.sh) captured 5 NTT launches
# from the SETUP phase (ciphertext encryption) before it could reach the
# bootstrap-call interior (modraise, key-switch, base-conversion). NCU's
# --launch-count is global across the regex, so a wider regex with low count
# stops on the first matches.
#
# This script narrows the regex to ONLY bootstrap-interior kernels (skipping
# NTTs which we already characterized) and increases --launch-count to 20.

set -e

echo "════════════════════════════════════════════════════════════"
echo "  NCU-PROFILE — Bootstrap interior kernels (modraise, KS, bconv)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
LOGDIR=/gpfs/projects/etur02/hkanpak/logs
REPORT_BASE=${LOGDIR}/ncu_boot_inner_${SLURM_JOB_ID}
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> NCU version:"
ncu --version 2>&1 | head -3
echo ""

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Bootstrap-interior only (NTTs covered in the first bootstrap pass)
KERNEL_REGEX='regex:.*modraise.*|.*key_switch.*|.*bconv.*'

echo ">>> Profiling: bert_hp_multigpu --N 65536 --n-gpus 1 --heads 1 --layers 1 --skip-ref"
echo ">>> Kernel regex (interior): ${KERNEL_REGEX}"
echo ">>> --launch-count 20 (each unique kernel hopefully captured at least once)"
echo ""

ncu \
    --kernel-name "${KERNEL_REGEX}" \
    --launch-count 20 \
    --set full \
    --target-processes all \
    --replay-mode kernel \
    --print-summary per-kernel \
    --csv \
    --log-file ${REPORT_BASE}.csv \
    -o ${REPORT_BASE} \
    --force-overwrite \
    ${PROJECT}/build/bin/bert_hp_multigpu \
        --N 65536 --n-gpus 1 --heads 1 --layers 1 --skip-ref
RC=$?

echo ""
echo "Outputs:"
ls -la ${REPORT_BASE}*

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
