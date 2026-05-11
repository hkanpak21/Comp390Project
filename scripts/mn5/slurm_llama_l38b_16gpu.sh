#!/bin/bash
#SBATCH --job-name=nexus-llama-l38b-16gpu
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/llama_l38b_16gpu_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/llama_l38b_16gpu_%j.err

# Lane LLAMA-NEXUS-MATCH — UNIQUE multi-node demonstration of LLaMA-3-8B
# at NEXUS Table IV parameters (ffn_dim=14336, seq_len=8) on 4 nodes ×
# 4 H100 = 16 GPUs total. NEXUS has no public multi-node LLaMA result
# at any parameter set — this is therefore the unique 16-GPU number.
# 1 trial.

echo "════════════════════════════════════════════════════════════"
echo "  LLAMA-NEXUS-MATCH MULTINODE — HP-LLaMA-3-8B 16× H100 (4 nodes)"
echo "  N=65536 / seq_len=8 / ffn_dim=14336"
echo "  Job: ${SLURM_JOB_ID}, Nodes: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load gcc/11.4.0 cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

BOOTSTRAP_DIR=/gpfs/projects/etur02/hkanpak/scratch
mkdir -p "${BOOTSTRAP_DIR}"

# NCCL / InfiniBand config for MN5 (per docs/MN5_NCCL_CONFIG.md)
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

for TRIAL in 1; do
    echo ""
    echo "═══════════════ TRIAL ${TRIAL} ═══════════════"
    date
    srun --mpi=none --ntasks=16 --ntasks-per-node=4 \
         ${PROJECT}/build/bin/llama_hp_multinode \
         --N 65536 \
         --heads 32 \
         --layers 32 \
         --model llama-3-8b \
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
