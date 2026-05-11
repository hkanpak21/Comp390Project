#!/bin/bash
#SBATCH --job-name=ncu-bootstrap
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/ncu_bootstrap_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/ncu_bootstrap_%j.err

# Lane NCU-PROFILE — Nsight Compute kernel-level profiling of bootstrap.
#
# Bootstrap is the dominant op (~40% of HP-BERT wall-clock per nsys nvtxsum).
# Inside one bootstrap call live: NTT/INTT (radix-8 phase1/phase2 variants),
# key-switch inner-product, base-conversion (modup/moddown), modraise.
#
# Strategy: run bert_hp_multigpu with the smallest config that triggers exactly
# one full bootstrap call (--n-gpus 1 --heads 1 --layers 1), and use NCU's
# --kernel-name regex to capture the kernels of interest. NCU SERIALIZES
# kernels — full --set full is ~50-200x slowdown per profiled kernel — so we
# limit to --launch-count 5 per matched name and use --set full only on the
# bootstrap-specific kernels.
#
# Output: ncu_bootstrap_<JOBID>.ncu-rep (binary, open in NCU UI or
# `ncu --import ... --print-summary`).

set -e

echo "════════════════════════════════════════════════════════════"
echo "  NCU-PROFILE — Bootstrap kernels (single GPU)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
LOGDIR=/gpfs/projects/etur02/hkanpak/logs
REPORT_BASE=${LOGDIR}/ncu_bootstrap_${SLURM_JOB_ID}
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> NCU version:"
ncu --version 2>&1 | head -3
echo ""

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Kernel regex — bootstrap-specific work
#   modraise           -> kernel_modraise_inplace (Bootstrapper.cu)
#   inwt_radix         -> Phantom inverse-NTT (intt_2d.cu)
#   fnwt_radix         -> Phantom forward-NTT (fntt_2d.cu)
#   key_switch         -> Phantom key-switch inner product (eval_key_switch.cu)
#   bconv              -> Phantom base conversion (rns_bconv.cu) — used in modup/down
#
# --launch-count 5 caps profiling per matched name (NCU profiles serially;
# bootstrap launches hundreds of NTT kernels, profiling all is impractical).
# --replay-mode kernel: each profiled kernel is replayed to gather all metric
# sets (default; needed for --set full).
KERNEL_REGEX='regex:.*modraise.*|.*inwt_radix.*|.*fnwt_radix.*|.*key_switch.*|.*bconv.*'

echo ">>> Profiling: bert_hp_multigpu --N 65536 --n-gpus 1 --heads 1 --layers 1 --skip-ref"
echo ">>> Kernel regex: ${KERNEL_REGEX}"
echo ">>> Set: full (comprehensive metrics, ~100x slower per profiled kernel)"
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
