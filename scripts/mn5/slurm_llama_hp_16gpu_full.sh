#!/bin/bash
#SBATCH --job-name=nexus-llama-hp-16gpu-full
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/llama_hp_16gpu_full_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/llama_hp_16gpu_full_%j.err

# Lane LLAMA-FULL — Full 32-layer HP-LLaMA multi-node 16× H100, N=65536.
# Analog of slurm_bert_hp_n32768_4node.sh but for LLaMA (32 heads, 32 layers).
# 16 ranks distributed round-robin: 32 heads / 16 ranks = 2 heads per rank.
#
# Expected wall-clock: bounded below by 2 × single-head-32-layer time
# (each rank runs 2 heads sequentially × 32 layers each). With the 105 s/layer
# from H2 (32 heads × 4 GPUs), per-head time at 2-heads-per-rank should be
# ~2 × 105 / 2 (due to less per-GPU contention) ≈ ~50-60 s/layer × 32 layers
# = ~1,600-1,900 s. Budget 2 h = 7,200 s gives margin for 3 trials if scaling
# holds, or 1 trial in worst case.

echo "════════════════════════════════════════════════════════════"
echo "  LLAMA-FULL MULTINODE (NCCL) — HP-LLaMA 32L × 32H @ N=65536"
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

# 3 trials sequential. Each trial: fresh CUDA state.
for TRIAL in 1 2 3; do
    echo ""
    echo "═══════════════ TRIAL ${TRIAL}/3 ═══════════════"
    date
    srun --mpi=none --ntasks=16 --ntasks-per-node=4 \
         ${PROJECT}/build/bin/llama_hp_multinode \
         --N 65536 \
         --heads 32 \
         --layers 32 \
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
echo "All trials complete."
echo "End: $(date)"
exit 0
