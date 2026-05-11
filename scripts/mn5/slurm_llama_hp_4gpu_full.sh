#!/bin/bash
#SBATCH --job-name=nexus-llama-hp-4gpu-full
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/llama_hp_4gpu_full_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/llama_hp_4gpu_full_%j.err

# Lane LLAMA-FULL — Full 32-layer HP-LLaMA single-node 4× H100, N=65536.
# Headline number for direct comparison vs NEXUS Table IV LLaMA-3-8B
# (51.84 s on 4× A100). Mirrors slurm_bert_hp_n32768.sh trial loop pattern.
#
# Per-trial expected wall-clock: ~32× single-layer HP-LLaMA (105.2 s) =
# ~3,360 s = ~56 min worst case (head parallelism keeps wall-clock per
# layer flat at ~105 s, 32 layers in sequence). 1.5 h slot fits exactly
# 1 trial; we run only 1 trial to fit budget. (For 3-trial median, raise
# --time to 03:30:00 and re-submit.)

echo "════════════════════════════════════════════════════════════"
echo "  LLAMA-FULL — HP-LLaMA full 32-layer / 4× H100 / N=65536"
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

# 1 trial — sized to fit the 1.5 h SLURM budget for 32 layers × 32 heads.
# (Per-trial est. 56 min; 3-trial median requires --time=03:30:00 in a
# follow-up submission.)
for TRIAL in 1; do
    echo ""
    echo "═══════════════ TRIAL ${TRIAL} ═══════════════"
    date
    ${PROJECT}/build/bin/llama_hp_multigpu \
        --N 65536 \
        --n-gpus 4 \
        --heads 32 \
        --layers 32 \
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
