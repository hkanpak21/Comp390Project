#!/bin/bash
#SBATCH --job-name=nexus-llama-l38b-4gpu
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/llama_l38b_4gpu_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/llama_l38b_4gpu_%j.err

# Lane LLAMA-NEXUS-MATCH — DIRECT NEXUS comparison: 4× H100 at the EXACT
# NEXUS Table IV LLaMA-3-8B parameter set (ffn_dim=14336, seq_len=8).
# Same multi-GPU count (4), same model, same algorithm — only hardware
# generation differs (H100 vs A100). 32 layers × 32 heads, 1 trial.

echo "════════════════════════════════════════════════════════════"
echo "  LLAMA-NEXUS-MATCH — HP-LLaMA-3-8B 4× H100 / N=65536 / seq_len=8"
echo "  Direct compare to NEXUS 4× A100 = 51.84 s"
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
        --n-gpus 4 \
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
