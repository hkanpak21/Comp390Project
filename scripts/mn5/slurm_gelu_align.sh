#!/bin/bash
#SBATCH --job-name=align-gelu-1gpu
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/gelu_align_n65k_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/gelu_align_n65k_%j.err

# Lane ALIGN-SINGLE — NEXUS GELU alignment microbench, single-GPU H100.
#
# Mimics NEXUS's gelu_test (vendor/nexus/cuda/src/main.cu lines 287-316,
# COEFF_MODULI[4]) at NEXUS's parameter set (logN=16, slot_count=32768).
#
# 100 isolated GELU calls; reports median + σ.
# Expected: ~69 ms ± 5%, matching NEXUS-on-H100's 69 ms (JOBID 40367787).

set -e

echo "════════════════════════════════════════════════════════════"
echo "  ALIGN-GELU (single-GPU H100) — NEXUS GELU, logN=16"
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

echo ">>> Running gelu_align_n65k --calls 100 ..."
${PROJECT}/build/bin/gelu_align_n65k --calls 100
RC=$?

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
