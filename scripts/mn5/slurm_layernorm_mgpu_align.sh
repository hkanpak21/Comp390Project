#!/bin/bash
#SBATCH --job-name=mgpu-ln
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/layernorm_mgpu_align_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/layernorm_mgpu_align_%j.err

# Lane MGPU-NEXUS-MICRO — multi-GPU NEXUS LayerNorm microbench, 4× H100.
#
# Mimics NEXUS's layernorm_test (vendor/nexus/cuda/src/main.cu, COEFF_MODULI[3])
# at logN=16, len=1024, data-parallel across 4 GPUs (each GPU runs 25 of
# the 100 calls).
#
# Compare: per-call median should match single-GPU (~45 ms); effective
# per-call (wall/100) should be ~11 ms (4× speedup).

set -e

echo "════════════════════════════════════════════════════════════"
echo "  MGPU-MICRO LayerNorm (4× H100) — NEXUS LayerNorm, logN=16"
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

echo ">>> TRIAL 1 — layernorm_mgpu_align --calls 100 --n-gpus 4"
${PROJECT}/build/bin/layernorm_mgpu_align --calls 100 --n-gpus 4
RC1=$?
echo ""

echo ">>> TRIAL 2 — layernorm_mgpu_align --calls 100 --n-gpus 4"
${PROJECT}/build/bin/layernorm_mgpu_align --calls 100 --n-gpus 4
RC2=$?
echo ""

echo ">>> TRIAL 3 — layernorm_mgpu_align --calls 100 --n-gpus 4"
${PROJECT}/build/bin/layernorm_mgpu_align --calls 100 --n-gpus 4
RC3=$?

echo ""
echo "Trial exit codes: ${RC1} ${RC2} ${RC3}"
echo "End: $(date)"
exit $((RC1 + RC2 + RC3))
