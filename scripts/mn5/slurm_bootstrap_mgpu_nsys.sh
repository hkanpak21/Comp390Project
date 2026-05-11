#!/bin/bash
#SBATCH --job-name=mgpu-bs-nsys
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:25:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/bootstrap_mgpu_nsys_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/bootstrap_mgpu_nsys_%j.err

# Lane MGPU-NSYS / TARGET 2 — Per-op multi-GPU NSYS trace, BOOTSTRAP.
#
# Wraps bootstrap_mgpu_align (4× H100, 25 calls per GPU) under nsys to
# inspect per-GPU timeline, identify idle gaps, and confirm the 1.04x
# wall-clock speedup is from genuine 4-way data parallelism.
#
# Workload mirrors slurm_bootstrap_mgpu_align.sh (1 trial, 100 calls total).
# Output:
#   bootstrap_mgpu_nsys_<JOBID>.nsys-rep
#   bootstrap_mgpu_nsys_<JOBID>.cuda_gpu_sum.txt   (per-GPU+kernel)
#   bootstrap_mgpu_nsys_<JOBID>.nvtxsum.txt        (per-NVTX-range)

set -e

echo "════════════════════════════════════════════════════════════"
echo "  MGPU-BS-NSYS — bootstrap_mgpu under nsys (4× H100)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
LOGDIR=/gpfs/projects/etur02/hkanpak/logs
TRACE_BASE=${LOGDIR}/bootstrap_mgpu_nsys_${SLURM_JOB_ID}
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

echo ">>> Running nsys profile bootstrap_mgpu_align --calls 100 --n-gpus 4 ..."
# nsys 12.8 on MN5 doesn't accept 'nccl' in --trace; NCCL is data-parallel
# anyway so no inter-GPU NCCL traffic is expected here.
${NSYS} profile \
    -o ${TRACE_BASE} \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --gpu-metrics-devices=none \
    --capture-range=none \
    ${PROJECT}/build/bin/bootstrap_mgpu_align --calls 100 --n-gpus 4
RC=$?

echo ""
echo ">>> nsys stats --report cuda_gpu_sum (per-GPU+kernel) ..."
${NSYS} stats --report cuda_gpu_sum --format table \
    ${TRACE_BASE}.nsys-rep > ${TRACE_BASE}.cuda_gpu_sum.txt 2>&1 || true

echo ">>> nsys stats --report nvtx_pushpop_sum ..."
${NSYS} stats --report nvtx_pushpop_sum --format table \
    ${TRACE_BASE}.nsys-rep > ${TRACE_BASE}.nvtxsum.txt 2>&1 || true

echo ""
echo "Outputs:"
ls -la ${TRACE_BASE}*

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
