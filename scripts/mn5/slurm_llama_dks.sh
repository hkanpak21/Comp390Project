#!/bin/bash
#SBATCH --job-name=nexus-llama-dks
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/llama_dks_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/llama_dks_%j.err

echo "════════════════════════════════════════════════════════════"
echo "  multiNEXUS LLaMA DKS — Full Decoder Layer, N=65536"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Architecture: 12 heads, SwiGLU FFN, RoPE, 4 bootstraps/layer"
echo "  DKS on 4×H100. LLaMA overhead vs BERT: RoPE + gate + gate⊙up"
echo ""

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

cd ${PROJECT}

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> Running llama_dks_multigpu (1 GPU baseline)..."
${PROJECT}/build/bin/llama_dks_multigpu 1
echo ""

echo ">>> Running llama_dks_multigpu (2 GPUs)..."
${PROJECT}/build/bin/llama_dks_multigpu 2
echo ""

echo ">>> Running llama_dks_multigpu (4 GPUs)..."
${PROJECT}/build/bin/llama_dks_multigpu 4

echo ""
echo "End: $(date)"
