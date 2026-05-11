#!/bin/bash
#SBATCH --job-name=hp-nccl-smoke
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:2
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/hp_nccl_smoke_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/hp_nccl_smoke_%j.err

# Smoke: verify the NCCL multi-node bootstrap works end-to-end.
# 2 nodes × 2 ranks = 4 ranks. layers=1 (~5 s compute per rank). Single trial.
# If wireup, SK broadcast, and end-of-trial reduce all complete cleanly,
# the NCCL migration is structurally correct and the longer 4-node × 12-layer
# job (40368260) will produce a meaningful timing measurement.

echo "════════════════════════════════════════════════════════════"
echo "  HP-BERT NCCL SMOKE — 2 nodes × 2 ranks, 1 layer, N=32768"
echo "  Job: ${SLURM_JOB_ID}, Nodes: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load gcc/11.4.0 cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

BOOTSTRAP_DIR=/gpfs/projects/etur02/hkanpak/scratch
mkdir -p "${BOOTSTRAP_DIR}"

# NCCL/IB env (matches docs/MN5_NCCL_CONFIG.md). DEBUG=INFO so we can see
# the rail topology in the log on first run.
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=ib0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

cd ${PROJECT}

echo ">>> Node layout:"
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 hostname

# 4 ranks total, 1 layer = quick verification. heads=4 so all 4 ranks active.
srun --mpi=none --ntasks=4 --ntasks-per-node=2 \
    ${PROJECT}/build/bin/bert_hp_multinode \
    --N 32768 \
    --heads 4 \
    --layers 1 \
    --skip-ref \
    --bootstrap-dir ${BOOTSTRAP_DIR}
RC=$?

echo ""
echo "Exit: ${RC}"
echo "End: $(date)"
exit ${RC}
