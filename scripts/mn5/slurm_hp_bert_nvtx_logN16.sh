#!/bin/bash
#SBATCH --job-name=align-hp-nvtx-n16
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:35:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/hp_bert_align_nvtx_logN16_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/hp_bert_align_nvtx_logN16_%j.err

# Lane ALIGN — HP-BERT at logN=16 with NVTX scopes for per-op extraction.
#
# Runs bert_hp_multigpu --N 65536 with `nsys profile`. NVTX scopes are now
# wrapped around each TIME_OP block (see src/benchmarks/bert_hp_multigpu.cu
# TIME_OP macro). After the run, `nsys stats --report nvtxsum` gives the
# wall-clock per NVTX scope, which is per-op time at logN=16.
#
# We run 1 layer × 12 heads (canonical S17 config) for cleanest per-op
# attribution; full 12-layer measurements already exist (S18, 376 s).
#
# Output files:
#   hp_bert_align_nvtx_logN16_<JOBID>.nsys-rep    (nsys trace)
#   hp_bert_align_nvtx_logN16_<JOBID>.nvtxsum.txt (per-scope summary)

set -e

echo "════════════════════════════════════════════════════════════"
echo "  ALIGN-HP-BERT-NVTX — NVTX-scoped HP-BERT, logN=16 (N=65,536)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
LOGDIR=/gpfs/projects/etur02/hkanpak/logs
TRACE_BASE=${LOGDIR}/hp_bert_align_nvtx_logN16_${SLURM_JOB_ID}
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> Locating nsys ..."
NSYS=$(which nsys || echo "")
if [ -z "${NSYS}" ]; then
    NSYS=/gpfs/apps/MN5/ACC/CUDA/12.8/nsight-systems-2024.6.2/bin/nsys
    if [ ! -x "${NSYS}" ]; then
        NSYS=/gpfs/apps/MN5/ACC/CUDA/12.8/nsight-systems-2025.1.1/bin/nsys
    fi
fi
echo "    nsys = ${NSYS}"
echo ""

echo ">>> Running nsys profile bert_hp_multigpu --N 65536 --n-gpus 4 --heads 12 --skip-ref ..."
${NSYS} profile \
    -o ${TRACE_BASE} \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --gpu-metrics-devices=none \
    --capture-range=none \
    ${PROJECT}/build/bin/bert_hp_multigpu \
        --N 65536 --n-gpus 4 --heads 12 --layers 1 --skip-ref
RC=$?

echo ""
echo ">>> nsys stats --report nvtxsum ..."
${NSYS} stats --report nvtxsum --format csv,table \
    ${TRACE_BASE}.nsys-rep > ${TRACE_BASE}.nvtxsum.txt 2>&1 || true

echo ""
echo "Outputs:"
ls -la ${TRACE_BASE}*

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
