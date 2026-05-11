#!/bin/bash
#SBATCH --job-name=nexus-llama-l38b-1gpu
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/llama_l38b_1gpu_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/llama_l38b_1gpu_%j.err

# Lane LLAMA-NEXUS-MATCH — single-GPU baseline at NEXUS Table IV LLaMA-3-8B
# parameters (ffn_dim=14336, seq_len=8). All 32 layers serialized on 1 GPU,
# the apples-to-apples baseline that anchors the 4-GPU and 16-GPU rows
# below. Worst-case wall-clock guess: 32 × ~120 s/layer single-GPU sequential
# (8 heads-equivalent batched serially per GPU at 4-GPU baseline scaled up
# 4× = ~480 s/layer, capped by 4 h budget). 1 trial.

echo "════════════════════════════════════════════════════════════"
echo "  LLAMA-NEXUS-MATCH — HP-LLaMA-3-8B 1× H100 / N=65536 / seq_len=8"
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

for TRIAL in 1; do
    echo ""
    echo "═══════════════ TRIAL ${TRIAL} ═══════════════"
    date
    ${PROJECT}/build/bin/llama_hp_multigpu \
        --N 65536 \
        --n-gpus 1 \
        --heads 32 \
        --layers 32 \
        --model llama-3-8b \
        --skip-ref
    RC=$?
    echo "[TRIAL ${TRIAL}] exit=${RC}"
    if [ ${RC} -ne 0 ]; then
        echo "[TRIAL ${TRIAL}] FAILED — aborting remaining trials"
        exit ${RC}
    fi
done

echo ""
echo "All trials complete."
echo "End: $(date)"
exit 0
