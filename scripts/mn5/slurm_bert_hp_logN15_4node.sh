#!/bin/bash
#SBATCH --job-name=hp-bert-logN15-16gpu
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/bert_hp_logN15_16gpu_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/bert_hp_logN15_16gpu_%j.err

# HP-BERT 16× H100 multi-node @ logN=15 (poly_degree 32,768)
#
# Fills missing matrix cell. Re-measurement of the configuration that
# was previously labeled both "logN=15" (line 164 of NEXUS_ALIGNMENT_PLAN.md)
# and "logN=16" (line 84) — we want a definitive, clean re-run with a fresh
# JOBID and 3-trial median to give an unambiguous data point.
#
# 4 nodes × 4 H100 = 16 GPUs total. 1 process per GPU (NCCL native).
# Heads 0..11 are active (1 head per rank); ranks 12..15 are idle.
# Median-of-3 trials. NCCL bootstrap via shared GPFS file (no MPI).

echo "════════════════════════════════════════════════════════════"
echo "  HP-BERT MULTINODE (NCCL) — logN=15 (N=32,768) — 12L × 12H"
echo "  4 nodes × 4 H100 = 16 GPUs total"
echo "  Job: ${SLURM_JOB_ID}, Nodes: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load gcc/11.4.0 cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

# Shared-FS scratch for the ncclUniqueId bootstrap file.
BOOTSTRAP_DIR=/gpfs/projects/etur02/hkanpak/scratch
mkdir -p "${BOOTSTRAP_DIR}"

# ── NCCL / InfiniBand config for MN5 (per docs/MN5_NCCL_CONFIG.md) ──
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=ib0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

cd ${PROJECT}

echo ""
echo ">>> Node layout:"
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 hostname
echo ""

echo ">>> GPU info per node:"
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 \
     nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Median of 3 trials. Each trial gets a fresh CUDA state (binary exits between).
for TRIAL in 1 2 3; do
    echo ""
    echo "═══════════════ TRIAL ${TRIAL}/3 ═══════════════"
    date
    srun --mpi=none --ntasks=16 --ntasks-per-node=4 \
         ${PROJECT}/build/bin/bert_hp_multinode \
         --N 32768 \
         --heads 12 \
         --layers 12 \
         --skip-ref \
         --bootstrap-dir ${BOOTSTRAP_DIR}
    RC=$?
    echo "[TRIAL ${TRIAL}] exit=${RC}"
    if [ ${RC} -ne 0 ]; then
        echo "[TRIAL ${TRIAL}] FAILED — aborting remaining trials"
        exit ${RC}
    fi
done

echo ""
echo "All 3 trials complete."
echo "End: $(date)"
exit 0
