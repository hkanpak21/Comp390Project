#!/bin/bash
#SBATCH --job-name=pipe-overhead
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:35:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/hp_bert_pipeline_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/hp_bert_pipeline_%j.err

# Lane MGPU-NSYS / TARGET 1 — Per-bootstrap pipeline-overhead breakdown.
#
# BOOT-DIAGNOSE proved that of the 1,032 ms per-bootstrap in HP-BERT pipeline
# only ~304 ms is the kernel work (single-GPU standalone bootstrap_3 at the
# HP-BERT chain depth).  The remaining ~728 ms is multi-GPU coordination /
# pipeline overhead (key fetches, syncs, mod_switch chain compaction, etc.).
#
# This SLURM script profiles HP-BERT under nsys with the ENRICHED NVTX scopes
# now wired into bootstrap_sparse_3 (BS_MOD_RAISE / BS_SUBSUM / coefftoslot_3
# / BS_MOD_REDUCTION / BS_SLOTTOCOEFF) and into TIME_OP(bsX) (MOD_SWITCH_DOWN
# / BOOTSTRAP_KERNEL).  The post-run `nsys stats --report nvtx_pushpop_sum`
# attributes the 1,032 ms total to each scope.
#
# Workload: --N 32768 --layers 1 --heads 1.  This matches the BOOT-DIAGNOSE
# config (logN=15, sparse=2^13, main=21, bs=14) so the per-bootstrap numbers
# can be compared directly against the 304 ms / 1,032 ms reference.
#
# Outputs:
#   hp_bert_pipeline_<JOBID>.nsys-rep              (full nsys trace)
#   hp_bert_pipeline_<JOBID>.nvtxsum.txt           (per-scope wall-clock)
#   hp_bert_pipeline_<JOBID>.cuda_gpu_sum.txt      (per-kernel GPU summary)

set -e

echo "════════════════════════════════════════════════════════════"
echo "  PIPE-OVERHEAD — HP-BERT pipeline-overhead breakdown (4× H100)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
LOGDIR=/gpfs/projects/etur02/hkanpak/logs
TRACE_BASE=${LOGDIR}/hp_bert_pipeline_${SLURM_JOB_ID}
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

# We use --trace=cuda,nvtx (HP-BERT has no NCCL — each head runs single-GPU).
# nsys 12.8 on MN5 does NOT accept 'nccl' in --trace; it observes NCCL via
# the cuda runtime trace anyway.
# --capture-range=none captures the entire run; the workload is small
# (1 layer × 1 head × 4 bootstraps = 4 bootstrap_sparse_3 calls).
echo ">>> Running nsys profile bert_hp_multigpu --N 32768 --layers 1 --heads 1 --skip-ref ..."
${NSYS} profile \
    -o ${TRACE_BASE} \
    --force-overwrite=true \
    --trace=cuda,nvtx \
    --gpu-metrics-devices=none \
    --capture-range=none \
    ${PROJECT}/build/bin/bert_hp_multigpu \
        --N 32768 --layers 1 --heads 1 --skip-ref
RC=$?

echo ""
echo ">>> nsys stats --report nvtx_pushpop_sum ..."
${NSYS} stats --report nvtx_pushpop_sum --format table \
    ${TRACE_BASE}.nsys-rep > ${TRACE_BASE}.nvtxsum.txt 2>&1 || true

echo ""
echo ">>> nsys stats --report cuda_gpu_sum ..."
${NSYS} stats --report cuda_gpu_sum --format table \
    ${TRACE_BASE}.nsys-rep > ${TRACE_BASE}.cuda_gpu_sum.txt 2>&1 || true

echo ""
echo "Outputs:"
ls -la ${TRACE_BASE}*

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
