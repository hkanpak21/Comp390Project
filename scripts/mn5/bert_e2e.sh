#!/bin/bash
#SBATCH --job-name=bert-e2e
#SBATCH --account=etur02
#SBATCH --qos=acc_bsccase
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=bert_e2e_%j.out
#SBATCH --error=bert_e2e_%j.err

echo "=== End-to-End BERT Layer Inference ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"

module purge
module load cuda/12.3
module load nccl/2.20.5-cuda12.3
module load cmake/3.27.4

cd "${SLURM_SUBMIT_DIR}"

echo ""
echo "--- 1 GPU (16 cols × 32 inner) ---"
./build/bin/bert_e2e_inference --n-gpus 1 --cols 16 --inner 32

echo ""
echo "--- 4 GPUs (16 cols × 32 inner) ---"
./build/bin/bert_e2e_inference --n-gpus 4 --cols 16 --inner 32

echo ""
echo "--- 4 GPUs (64 cols × 32 inner) ---"
./build/bin/bert_e2e_inference --n-gpus 4 --cols 64 --inner 32

echo ""
echo "End: $(date)"
