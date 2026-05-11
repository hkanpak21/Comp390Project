#!/bin/bash
#SBATCH --job-name=bert-ops-test
#SBATCH --account=etur02
#SBATCH --qos=acc_bsccase
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=bert_ops_test_%j.out
#SBATCH --error=bert_ops_test_%j.err

echo "=== BERT Operations Correctness Test ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start: $(date)"

module purge
module load cuda/12.3
module load nccl/2.20.5-cuda12.3
module load cmake/3.27.4

# NTL is at /gpfs/projects/etur02/hkanpak/local — needed at runtime by Phantom→NEXUS chain.
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd "${SLURM_SUBMIT_DIR}"

echo ""
echo "--- GELU Test ---"
./build/bin/bert_ops_test --op gelu

echo ""
echo "--- Softmax Test ---"
./build/bin/bert_ops_test --op softmax

echo ""
echo "--- LayerNorm Test ---"
./build/bin/bert_ops_test --op layernorm

echo ""
echo "End: $(date)"
