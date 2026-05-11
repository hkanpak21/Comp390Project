#!/bin/bash
#SBATCH --job-name=tmodup-memcheck
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH --time=00:55:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/tmodup_memcheck_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/tmodup_memcheck_%j.err

# Lane T-MODUP-FIX-2 — compute-sanitizer (memcheck) on the T-MODUP path.
# Targets the failing 2-GPU DKS_ROTATE=1 run that crashes inside
# partial_key_switch_inner_prod / modup_partial. Slowed ~10-50× by
# instrumentation; we limit BERT_LAYERS=1 and run only the 2-GPU sweep.
# First memcheck error printed will pin the OOB write.

echo "════════════════════════════════════════════════════════════"
echo "  T-MODUP-FIX-2 — cuda-memcheck (2 GPUs, BERT_LAYERS=1)"
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
export DKS_ROTATE=1
export BERT_LAYERS=1

# compute-sanitizer slows execution and may interact with cudaMallocAsync.
# Use plain memcheck mode and disable stack tracing for performance — the
# kernel-name + offset suffices to pin the OOB write site.
SANITIZER=/apps/ACC/CUDA/12.8/bin/compute-sanitizer

cd ${PROJECT}

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> compute-sanitizer version:"
${SANITIZER} --version 2>&1 | head -3
echo ""

echo ">>> Running bert_dks_multigpu (2 GPUs) under memcheck..."
echo "  (this is ~10-50× slower than native; first error will pin the OOB)"
echo ""

# --tool memcheck: out-of-bounds + misaligned + leak
# --target-processes all: instrument fork()/exec() children too
# --launch-timeout 0: don't kill us for slow kernel launches
# --print-limit 5: stop after first 5 errors to keep output small
# --report-api-errors no: skip CUDA API error reports (we already see them in app stderr)
# --track-stream-ordered-races yes: NOT SUPPORTED in 2025.1.0; dropped
${SANITIZER} \
    --tool memcheck \
    --target-processes all \
    --launch-timeout 0 \
    --print-limit 5 \
    --report-api-errors no \
    --error-exitcode 99 \
    ${PROJECT}/build/bin/bert_dks_multigpu 2

RC=$?

echo ""
echo "Exit code: ${RC}  (99 = sanitizer caught error; 0 = clean)"
echo "End: $(date)"
exit ${RC}
