#!/bin/bash
#SBATCH --job-name=nexus-h100-baseline-ehpc
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/nexus_baseline_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/nexus_baseline_%j.err

# Phase 0 R3 — Vanilla NEXUS GPU baseline on MN5 H100.
# Mirror of slurm_nexus_baseline.sh but on acc_ehpc QoS to bypass acc_debug
# MaxSubmit=1 throttling. Used as a backup when the debug job is queued behind
# higher-priority work.
#
# What this measures:
#   NEXUS's `main` binary is a per-op micro-benchmark (controlled by TEST_TARGET_IDX).
#   The published 37.34 s "BERT-base" number is reconstructed from per-op runs ×
#   per-layer invocation counts (this is what their paper §VI does).
#
# This script runs all 5 op tests (MatMul, Argmax, SoftMax, LayerNorm, GELU) and
# the standalone `bootstrapping` binary, 3 trials each, on a single H100 GPU
# (NEXUS is NOT multi-GPU — see notes below).

echo "════════════════════════════════════════════════════════════"
echo "  Vanilla NEXUS — Phase 0 R3 baseline on MN5 H100 (ehpc QoS)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1 gcc/11.4.0

NEXUS_BUILD=/gpfs/projects/etur02/hkanpak/Comp390Project/nexus/cuda/build
NEXUS_DATA=/gpfs/projects/etur02/hkanpak/Comp390Project/nexus/data
LOGS=/gpfs/projects/etur02/hkanpak/logs

# NEXUS expects to run from cuda/build/bin/ so that `../../data/...` resolves.
cd ${NEXUS_BUILD}/bin

unset CPATH
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# Capture GPU utilization in background so we can see if NEXUS uses 1 or 4 GPUs.
GPU_LOG=${LOGS}/nexus_baseline_${SLURM_JOB_ID}_gpu_util.log
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -l 5 > ${GPU_LOG} 2>&1 &
NVSMI_PID=$!
echo ">>> GPU utilization logger PID: ${NVSMI_PID}, output: ${GPU_LOG}"
echo ""

# Op test runner: idx=0 MatMul, 1=Argmax, 2=SoftMax, 3=LayerNorm, 4=GELU
OPS=("MatMul" "Argmax" "SoftMax" "LayerNorm" "GELU")
TRIALS=3

for idx in 0 1 2 3 4; do
  OP_NAME="${OPS[$idx]}"
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "  OP: ${OP_NAME} (TEST_TARGET_IDX=${idx})  ×${TRIALS} trials"
  echo "════════════════════════════════════════════════════════════"
  for trial in $(seq 1 ${TRIALS}); do
    echo ""
    echo "--- ${OP_NAME} trial ${trial}/${TRIALS} ---"
    /usr/bin/time -v ./main ${idx} 2>&1 | grep -E "\[${OP_NAME}\]|\[NEXUS-driver\]|Mean Absolute Error|Average Error|Elapsed|Maximum resident|User time|System time" || true
  done
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Bootstrapping standalone (NEXUS bootstrap test)  ×${TRIALS} trials"
echo "════════════════════════════════════════════════════════════"
for trial in $(seq 1 ${TRIALS}); do
  echo ""
  echo "--- Bootstrap trial ${trial}/${TRIALS} ---"
  /usr/bin/time -v ./bootstrapping 2>&1 | grep -E "Bootstrapping|takes:|Mean Absolute Error|Elapsed|Maximum resident" || true
done

# Kill GPU utilization logger.
kill ${NVSMI_PID} 2>/dev/null || true
wait ${NVSMI_PID} 2>/dev/null || true

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  GPU utilization summary (look for nonzero on GPUs 1,2,3 = multi-GPU)"
echo "════════════════════════════════════════════════════════════"
echo ">>> Distinct (index, util.gpu>0) tuples seen:"
awk -F', *' 'NR>1 && $2 ~ /^[0-9]+$/ && $2+0 > 0 {print "  GPU "$1" util "$2" mem "$3}' ${GPU_LOG} | sort -u | head -40

echo ""
echo "End: $(date)"
