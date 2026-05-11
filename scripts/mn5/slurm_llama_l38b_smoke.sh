#!/bin/bash
#SBATCH --job-name=l38b-smoke
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/l38b_smoke_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/l38b_smoke_%j.err

# Lane LLAMA-NEXUS-MATCH smoke — validate --model llama-3-8b path works
# correctly. 1 head × 1 layer, fast turnaround on debug QoS.

echo "════════════════════════════════════════════════════════════"
echo "  l38b-smoke — validate --model llama-3-8b CLI flag"
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
echo "════ TEST 1: --model llama-3-8b, 32 heads (proper LLaMA-3-8B), 1 layer ════"
# Use 32 heads so ffn_inner = 14336/32 = 448 (the production value).
# 1 layer to fit in 15-min debug budget.
${PROJECT}/build/bin/llama_hp_multigpu \
    --N 65536 \
    --n-gpus 1 \
    --heads 32 \
    --layers 1 \
    --model llama-3-8b \
    --skip-ref
RC1=$?
echo "TEST 1 exit=${RC1}"

echo ""
echo "End: $(date)"
exit ${RC1}
