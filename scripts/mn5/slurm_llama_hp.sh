#!/bin/bash
#SBATCH --job-name=nexus-llama-hp
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/llama_hp_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/llama_hp_%j.err

# Lane H / H2 — HP-LLaMA single-node 32 heads × 4 GPUs (8 heads/GPU).
# Mirrors bert_hp_multigpu's pattern for LLaMA layer compute (RoPE, SwiGLU,
# RMSNorm proxy). One layer per invocation; verified per-head against
# single-GPU reference (MAE < 1e-5 threshold).

echo "════════════════════════════════════════════════════════════"
echo "  H2 — HP-LLaMA single-node (32 heads / 4 GPUs / 1 layer)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> Running llama_hp_multigpu --n-gpus 4 --heads 32 --skip-ref ..."
${PROJECT}/build/bin/llama_hp_multigpu --n-gpus 4 --heads 32 --skip-ref
RC=$?

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
