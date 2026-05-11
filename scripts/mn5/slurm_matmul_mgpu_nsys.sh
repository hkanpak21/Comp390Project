#!/bin/bash
#SBATCH --job-name=mgpu-mm-nsys
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_%j.err

# PROFILE-01 — Per-op multi-GPU NSYS trace, MATMUL output-channel split.
# Mirrors slurm_matmul_align.sh (median of 3 trials).

set -e

echo "════════════════════════════════════════════════════════════"
echo "  MGPU-MM-NSYS — matmul_align_n8k under nsys (4× H100)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
LOGDIR=/gpfs/projects/etur02/hkanpak/logs
TRACE_BASE=${LOGDIR}/matmul_mgpu_nsys_${SLURM_JOB_ID}
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

NSYS=$(which nsys || echo "")
if [ -z "${NSYS}" ]; then
    NSYS=/gpfs/apps/MN5/ACC/CUDA/12.8/nsight-systems-2024.6.2/bin/nsys
fi
echo ">>> nsys = ${NSYS}"

echo ">>> Running nsys profile matmul_align_n8k --n-gpus 4 --trials 3 ..."
# nsys 12.8 on MN5 doesn't accept 'nccl' in --trace.
${NSYS} profile \
    -o ${TRACE_BASE} \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --gpu-metrics-devices=none \
    --capture-range=none \
    ${PROJECT}/build/bin/matmul_align_n8k --n-gpus 4 --trials 3
RC=$?

echo ""
echo ">>> nsys stats reports ..."
${NSYS} stats --report cuda_gpu_sum --format table \
    ${TRACE_BASE}.nsys-rep > ${TRACE_BASE}.cuda_gpu_sum.txt 2>&1 || true
${NSYS} stats --report nvtx_pushpop_sum --format table \
    ${TRACE_BASE}.nsys-rep > ${TRACE_BASE}.nvtxsum.txt 2>&1 || true

echo "Outputs:"
ls -la ${TRACE_BASE}*

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
