#!/bin/bash
#SBATCH --job-name=l38b-pack-full
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/l38b_pack_full_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/l38b_pack_full_%j.err

# SIMD slot-folded HP-LLaMA full measurement: LLaMA-3-8B (32 layers × 32 heads)
# at logN=15 (NEXUS-comparable), 4× H100, 1 trial. Compares against:
#   NEXUS LLaMA-3-8B (4× A100): 51.84 s
#   Our previous logN=16 unpacked baseline: 2,799 s
#   Our logN=15 unpacked (separate, in flight)

echo "════════════════════════════════════════════════════════════"
echo "  LLAMA-PACK-FULL — HP-LLaMA-3-8B 4× H100 / N=32768 / packed BS#4"
echo "  Compare to NEXUS 4× A100 = 51.84 s"
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

echo "═══════════════ TRIAL 1 (packed) ═══════════════"
date
${PROJECT}/build/bin/llama_hp_multigpu \
    --N 32768 \
    --n-gpus 4 \
    --heads 32 \
    --layers 32 \
    --model llama-3-8b \
    --pack-bootstrap \
    --skip-ref
RC=$?
echo "[TRIAL 1] exit=${RC}"

echo ""
echo "End: $(date)"
exit ${RC}
