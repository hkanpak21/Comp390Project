#!/bin/bash
#SBATCH --job-name=nexus-4node
#SBATCH --account=your_project_account          # REPLACE with MN5 project account
#SBATCH --qos=acc_bsccase
#SBATCH --partition=acc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4                     # 1 task per GPU
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=experiments/results/nexus_4node_%j.out
#SBATCH --error=experiments/results/nexus_4node_%j.err

echo "=== NEXUS 4-Node Benchmark (16× H100) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NODELIST}"
echo "Start: $(date)"

module purge
module load cuda/12.3
module load nccl/2.20.5-cuda12.3
module load openmpi/4.1.5-cuda12.3
module load cmake/3.27.4

# ── NCCL / InfiniBand ────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=ib0
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500

cd "${SLURM_SUBMIT_DIR}"

# ── Strong scaling: BERT-base, increasing GPU count ──────────────────────────
# (1 GPU baseline already done in slurm_1node.sh)

echo "--- 8-GPU strong scaling (2 nodes × 4 GPUs) ---"
srun --nodes=2 --ntasks=8 --ntasks-per-node=4 \
    ./build/bin/bert_inference \
    --n-gpus 8 \
    --output experiments/results/bert_8gpu_${SLURM_JOB_ID}.csv

echo "--- 16-GPU strong scaling (4 nodes × 4 GPUs) ---"
srun --nodes=4 --ntasks=16 --ntasks-per-node=4 \
    ./build/bin/bert_inference \
    --n-gpus 16 \
    --output experiments/results/bert_16gpu_${SLURM_JOB_ID}.csv

echo ""
echo "--- Bootstrapping: 8 GPUs ---"
srun --nodes=2 --ntasks=8 --ntasks-per-node=4 \
    ./build/bin/bootstrapping_bench \
    --n-gpus 8 \
    --output experiments/results/bootstrapping_8gpu_${SLURM_JOB_ID}.csv

echo "--- Bootstrapping: 16 GPUs ---"
srun --nodes=4 --ntasks=16 --ntasks-per-node=4 \
    ./build/bin/bootstrapping_bench \
    --n-gpus 16 \
    --output experiments/results/bootstrapping_16gpu_${SLURM_JOB_ID}.csv

echo ""
echo "--- NCCL bandwidth test (baseline) ---"
srun --nodes=4 --ntasks=16 --ntasks-per-node=4 \
    ./build/bin/nccl_bandwidth_test \
    --output experiments/results/nccl_bandwidth_4node_${SLURM_JOB_ID}.csv

echo "End: $(date)"
