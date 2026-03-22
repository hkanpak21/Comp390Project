#!/bin/bash
#SBATCH --job-name=nexus-1node
#SBATCH --account=your_project_account          # REPLACE with MN5 project account
#SBATCH --qos=acc_bsccase                       # GPU QoS on MN5
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4                     # 1 task per GPU
#SBATCH --cpus-per-task=20                      # MN5: 80 cores / 4 GPUs
#SBATCH --gres=gpu:4                            # MN5 has 4× H100 per node
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=experiments/results/nexus_1node_%j.out
#SBATCH --error=experiments/results/nexus_1node_%j.err

echo "=== NEXUS 1-Node Benchmark (4× H100) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"

# ── Modules ───────────────────────────────────────────────────────────────────
module purge
module load cuda/12.3
module load nccl/2.20.5-cuda12.3
module load openmpi/4.1.5-cuda12.3
module load cmake/3.27.4

# ── NCCL / InfiniBand environment ────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5                        # ConnectX-7 adapters on MN5
export NCCL_IB_CUDA_SUPPORT=1                  # GPUDirect RDMA
export NCCL_SOCKET_IFNAME=ib0
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500

# ── Run single-node BERT-base benchmark ──────────────────────────────────────
cd "${SLURM_SUBMIT_DIR}"

echo "--- Single-GPU baseline ---"
srun --ntasks=1 --gres=gpu:1 \
    ./build/bin/bert_inference \
    --n-gpus 1 \
    --output experiments/results/bert_1gpu_${SLURM_JOB_ID}.csv

echo "--- 2-GPU scaling ---"
srun --ntasks=2 --gres=gpu:2 \
    ./build/bin/bert_inference \
    --n-gpus 2 \
    --output experiments/results/bert_2gpu_${SLURM_JOB_ID}.csv

echo "--- 4-GPU scaling ---"
srun --ntasks=4 --gres=gpu:4 \
    ./build/bin/bert_inference \
    --n-gpus 4 \
    --output experiments/results/bert_4gpu_${SLURM_JOB_ID}.csv

echo ""
echo "--- Bootstrapping benchmark (isolated, 4 GPUs) ---"
srun --ntasks=4 --gres=gpu:4 \
    ./build/bin/bootstrapping_bench \
    --n-gpus 4 \
    --output experiments/results/bootstrapping_4gpu_${SLURM_JOB_ID}.csv

echo "End: $(date)"
echo "Results in experiments/results/"
