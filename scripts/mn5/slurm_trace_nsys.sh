#!/bin/bash
#SBATCH --job-name=nexus-trace-nsys
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/trace_nsys_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/trace_nsys_%j.err

# T-TRACE — granular Nsight Systems profile of the DKS bootstrap.
# Captures NVTX ranges added in iteration 1: bsgs_baby_step, bsgs_giant_step,
# modup, moddown, plus the existing per-rotation / per-phase ranges.
#
# Output: traces/trace_dksrot.nsys-rep — copy back to local Mac with:
#   scp mn5-gpu:/gpfs/projects/etur02/hkanpak/Comp390Project/traces/trace_dksrot.nsys-rep ~/nexus-traces/
# Then summary:
#   nsys stats --report nvtxsum trace_dksrot.nsys-rep | grep -E "bsgs|modup|moddown"

echo "════════════════════════════════════════════════════════════"
echo "  multiNEXUS — Nsight Systems profile (T-TRACE)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export DKS_ROTATE=1   # exercise the DKS rotation path (champion config)

cd ${PROJECT}
mkdir -p traces

NSYS_OUT=${PROJECT}/traces/trace_dksrot
echo ">>> nsys profile → ${NSYS_OUT}.nsys-rep"
echo ""

nsys profile \
    --output=${NSYS_OUT} \
    --force-overwrite=true \
    --trace=cuda,nvtx,cudnn,cublas,nccl \
    --cuda-memory-usage=true \
    --capture-range=nvtx \
    --capture-range-end=stop \
    --nvtx-capture=bootstrap \
    ${PROJECT}/build/bin/dist_bootstrap_bench 4

RC=$?
echo ""
echo "End: $(date) (rc=$RC)"

echo ""
echo "Trace file:"
ls -lh ${NSYS_OUT}.nsys-rep 2>/dev/null || echo "  (missing — nsys may have failed)"

echo ""
echo "Quick on-cluster summary (top NVTX ranges):"
nsys stats --report nvtxsum ${NSYS_OUT}.nsys-rep 2>/dev/null | head -40 || true

echo ""
echo "Next: scp ${NSYS_OUT}.nsys-rep to local for full Nsight GUI inspection."
