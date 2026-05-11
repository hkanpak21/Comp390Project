#!/bin/bash
#SBATCH --job-name=align-bs-1gpu
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/bootstrap_align_n32k_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/bootstrap_align_n32k_%j.err

# Lane ALIGN-SINGLE — NEXUS Bootstrap alignment microbench, single-GPU H100.
#
# Mimics NEXUS's standalone bootstrap test (vendor/nexus/cuda/src/bootstrapping.cu)
# at NEXUS's parameter set (logN=15, sparse_slots=2^13=8192, hamming=192).
#
# 100 isolated bootstrap calls; reports median + σ.
# Expected: ~250 ms ± 5%, matching NEXUS-on-H100's 252.8 ms (JOBID 40367787).

set -e

echo "════════════════════════════════════════════════════════════"
echo "  ALIGN-Bootstrap (single-GPU H100) — NEXUS bootstrap, logN=15"
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

echo ">>> Running bootstrap_align_n32k --calls 100 ..."
${PROJECT}/build/bin/bootstrap_align_n32k --calls 100
RC=$?

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
