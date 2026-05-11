#!/bin/bash
#SBATCH --job-name=nexus-llama-baseline
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/llama_baseline_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/llama_baseline_%j.err

echo "════════════════════════════════════════════════════════════"
echo "  multiNEXUS LLaMA-7B Layer Baseline (PRD slice H1)"
echo "  1 GPU, N=65536, hidden=4096, heads=32, head_dim=128, seq_len=128"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Architecture: 1 LLaMA-7B decoder layer end-to-end (1 head's compute)"
echo "  CPU-streaming Galois keys (~62 GB key store, doesn't fit on one H100)"
echo "  Reports per-component breakdown + sequential-baseline projection."
echo ""

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> Running llama_layer_baseline (default seq_len=128)..."
${PROJECT}/build/bin/llama_layer_baseline --seq-len 128

echo ""
echo "End: $(date)"
