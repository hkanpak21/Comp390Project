#!/bin/bash
#SBATCH --job-name=ncu-nonlin
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/ncu_nonlinear_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/ncu_nonlinear_%j.err

# Lane NCU-PROFILE — Nsight Compute kernel-level profiling of LayerNorm,
# Softmax, GELU.
#
# In our implementation (src/nexus_eval/{layer_norm,softmax,gelu}.cu) these
# non-linear operators have NO custom kernels; they compose from Phantom
# polymath primitives (polynomial multiply, multiply-and-add, scalar multiply,
# scale-and-add). The kernels they exercise are largely a subset of those
# touched by matmul, but with characteristically different access patterns:
#   - LayerNorm: sub_rns_poly + multiply_scalar_rns_poly + many adds
#   - Softmax:   tensor_prod_2x2 (multiply) + multiply_and_scale_add for the
#                Chebyshev exp/sigmoid approximation
#   - GELU:      same poly-multiply primitives for Chebyshev approximation
#
# We profile by running bert_hp_multigpu --layers 1 --heads 1 (which exercises
# all three operators) and matching the multiply/add/scale kernels with
# --launch-count 5. To distinguish operator usage you'd correlate with NVTX
# (nsys) timestamps; this run captures kernel-level metrics (SM%, DRAM%,
# occupancy) which are properties of the kernel, not the surrounding op.

set -e

echo "════════════════════════════════════════════════════════════"
echo "  NCU-PROFILE — Non-linear op kernels (LayerNorm/Softmax/GELU)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
LOGDIR=/gpfs/projects/etur02/hkanpak/logs
REPORT_BASE=${LOGDIR}/ncu_nonlinear_${SLURM_JOB_ID}
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> NCU version:"
ncu --version 2>&1 | head -3
echo ""

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Kernel regex for non-linear-op work (Phantom polymath primitives, no NTTs)
KERNEL_REGEX='regex:.*multiply_scalar.*rns_poly.*|.*multiply_and_scale_add.*|.*multiply_and_add.*rns_poly.*|.*sub_and_scale.*|.*tensor_prod_2x2.*|.*add_many_rns_poly.*|.*negate_rns_poly.*'

echo ">>> Profiling non-linear-op poly kernels with --set full"
echo ">>> Kernel regex: ${KERNEL_REGEX}"
echo ""

ncu \
    --kernel-name "${KERNEL_REGEX}" \
    --launch-count 5 \
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
