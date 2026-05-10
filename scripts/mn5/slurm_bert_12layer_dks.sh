#!/bin/bash
#SBATCH --job-name=nexus-bert-12layer-dks
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/bert_12layer_dks_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/bert_12layer_dks_%j.err

echo "════════════════════════════════════════════════════════════"
echo "  multiNEXUS BERT DKS — Full 12-Layer Encoder, N=65536 (T-12LAYER-BASE)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Architecture: 12 encoder layers × 1 head, 4 bootstraps/layer, DKS on 4×H100"
echo "  Goal: measure actual 12-layer wall time vs 12 × single-layer projection"
echo ""

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

# Phase 4b champion path — distributed key-switching for rotations.
export DKS_ROTATE=1

# Run all 12 BERT encoder layers end-to-end (T-12LAYER-BASE).
export BERT_LAYERS=12

cd ${PROJECT}

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> BERT_LAYERS=${BERT_LAYERS}"
echo ""

echo ">>> Running bert_dks_multigpu (4 GPUs, 12 layers)..."
${PROJECT}/build/bin/bert_dks_multigpu 4

echo ""
echo "End: $(date)"
