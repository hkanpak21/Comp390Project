#!/bin/bash
#SBATCH --job-name=bert-matmul-real
#SBATCH --account=etur02
#SBATCH --qos=acc_bsccase
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=bert_matmul_real_%j.out
#SBATCH --error=bert_matmul_real_%j.err

echo "=== Real BERT MatMul Multi-GPU Benchmark ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"

module purge
module load cuda/12.3
module load nccl/2.20.5-cuda12.3
module load cmake/3.27.4

cd "${SLURM_SUBMIT_DIR}"

# Test with reduced inner_dim first (faster)
echo ""
echo "--- 1 GPU baseline (64 cols × 32 inner) ---"
./build/bin/bert_matmul_real --n-gpus 1 --cols 64 --inner 32

echo ""
echo "--- 2 GPUs (64 cols × 32 inner) ---"
./build/bin/bert_matmul_real --n-gpus 2 --cols 64 --inner 32

echo ""
echo "--- 4 GPUs (64 cols × 32 inner) ---"
./build/bin/bert_matmul_real --n-gpus 4 --cols 64 --inner 32

# Full BERT inner dimension if time permits
echo ""
echo "--- 4 GPUs (64 cols × 128 inner) ---"
./build/bin/bert_matmul_real --n-gpus 4 --cols 64 --inner 128

echo ""
echo "End: $(date)"
