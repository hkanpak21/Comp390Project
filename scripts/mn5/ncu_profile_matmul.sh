#!/bin/bash
#SBATCH --job-name=ncu-matmul
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/ncu_matmul_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/ncu_matmul_%j.err

# Lane NCU-PROFILE — Nsight Compute kernel-level profiling of matmul.
#
# Encrypted matmul is the second-largest cost after bootstrap. The matmul-
# specific kernel in our codebase is `kernel_compress_ciphertext`
# (src/nexus_eval/matrix_mul.cu). The heavy lifting (rotate, multiply,
# add) is performed by Phantom polymath kernels:
#   - multiply_rns_poly, multiply_and_add_rns_poly, tensor_prod_*
#   - add_rns_poly, sub_rns_poly
#   - inwt_radix*, fnwt_radix* (NTTs around each multiply)
#   - key_switch_inner_prod_c2_and_evk (rotation key-switch)
#
# Use the standalone matmul_align_n8k binary if present (smallest repro);
# fall back to bert_hp_multigpu with --heads 1 --layers 1 to exercise the
# full attention matmul pipeline.

set -e

echo "════════════════════════════════════════════════════════════"
echo "  NCU-PROFILE — MatMul kernels (single GPU)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
LOGDIR=/gpfs/projects/etur02/hkanpak/logs
REPORT_BASE=${LOGDIR}/ncu_matmul_${SLURM_JOB_ID}
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> NCU version:"
ncu --version 2>&1 | head -3
echo ""

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Pick binary: prefer the standalone matmul micro-bench when available
BIN=${PROJECT}/build/bin/matmul_align_n8k
BIN_ARGS=""
if [ ! -x "${BIN}" ]; then
    echo ">>> matmul_align_n8k not found, falling back to bert_hp_multigpu"
    BIN=${PROJECT}/build/bin/bert_hp_multigpu
    BIN_ARGS="--N 65536 --n-gpus 1 --heads 1 --layers 1 --skip-ref"
fi
echo ">>> Binary: ${BIN} ${BIN_ARGS}"
echo ""

# Kernel regex for matmul-relevant work
#   compress_ciphertext       -> our matmul plaintext compress (matrix_mul.cu)
#   multiply_*_rns_poly       -> Phantom polynomial multiply (polymath.cu)
#   tensor_prod_*             -> Phantom 2x2/mxn ciphertext multiply
#   add_rns_poly, sub_rns_poly -> Phantom add/sub
#   inwt_radix, fnwt_radix    -> NTT around multiplies
#   key_switch                -> rotation/relinearization key-switch
KERNEL_REGEX='regex:.*compress_ciphertext.*|.*multiply.*rns_poly.*|.*tensor_prod.*|.*add_rns_poly.*|.*sub_rns_poly.*|.*key_switch.*'

echo ">>> Profiling matmul kernels with --set full"
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
    ${BIN} ${BIN_ARGS}
RC=$?

echo ""
echo "Outputs:"
ls -la ${REPORT_BASE}*

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
