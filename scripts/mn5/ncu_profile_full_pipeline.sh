#!/bin/bash
#SBATCH --job-name=ncu-full
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/ncu_full_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/ncu_full_%j.err

# Lane NCU-PROFILE — Survey-level NCU pass over a short HP-BERT run.
#
# This complements the per-target scripts by profiling EVERY launched kernel
# (no --kernel-name filter) but with the lighter-weight --set basic so the
# total wall is tractable. Output gives a per-kernel time/SM%/memory-throughput
# table for the full single-GPU HP-BERT layer (1 layer × 1 head × N=65536),
# letting us rank kernels by total time without committing to one regex.
#
# NCU multi-GPU support is fragile (and adds replay overhead); profiling
# the single-GPU case with --n-gpus 1 gives kernel-level metrics
# representative of one GPU's work in the 4-GPU configuration. Per-GPU
# multi-process profiling can be added later by wrapping each MPI rank.
#
# Notes on --set basic:
#   * Captures Compute Throughput, Memory Throughput, Issue Slot Utilization,
#     Achieved Occupancy — the headline roofline + occupancy info.
#   * Skips full memory-pipeline / source-counter sections that --set full has.
#   * Per-kernel slowdown ~5-20x vs unprofiled (full is 50-200x).
#
# --launch-count 1 caps each unique kernel to one profiled launch (otherwise
# kernels launched thousands of times in NTT loops would explode the run).

set -e

echo "════════════════════════════════════════════════════════════"
echo "  NCU-PROFILE — Full HP-BERT pipeline (survey, --set basic)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
LOGDIR=/gpfs/projects/etur02/hkanpak/logs
REPORT_BASE=${LOGDIR}/ncu_full_${SLURM_JOB_ID}
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> NCU version:"
ncu --version 2>&1 | head -3
echo ""

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> Profiling full HP-BERT (1 layer, 1 head, N=65536) with --set basic"
echo ">>> --launch-count 1 per unique kernel name (one sample each)"
echo ""

ncu \
    --launch-count 1 \
    --set basic \
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
