#!/bin/bash
#SBATCH --job-name=tmodup-smoke
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/tmodup_smoke_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/tmodup_smoke_%j.err

# Lane T-MODUP-FIX-2 — post-fix verification.
# Single-layer DKS_ROTATE=1 sweep across 2 and 4 GPUs to confirm the
# T-MODUP fix runs cleanly. Acceptance: bootstrap < 1,800 ms (ideal target
# ~1,500 ms), no NCCL P2P errors, no cudaFreeAsync illegal-memory-access.

echo "════════════════════════════════════════════════════════════"
echo "  T-MODUP-FIX-2 smoke — DKS_ROTATE=1 1L sweep (2 then 4 GPUs)"
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

cd ${PROJECT}

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> Running bert_dks_multigpu (2 GPUs, 1 layer)..."
${PROJECT}/build/bin/bert_dks_multigpu 2
RC2=$?
echo ""
echo "[2-GPU exit: ${RC2}]"
echo ""

if [ ${RC2} -eq 0 ]; then
    echo ">>> Running bert_dks_multigpu (4 GPUs, 1 layer)..."
    ${PROJECT}/build/bin/bert_dks_multigpu 4
    RC4=$?
    echo ""
    echo "[4-GPU exit: ${RC4}]"
fi

echo ""
echo "End: $(date)"
exit ${RC2}
