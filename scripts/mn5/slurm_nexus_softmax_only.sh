#!/bin/bash
#SBATCH --job-name=nexus-softmax-only
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/nexus_softmax_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/nexus_softmax_%j.err

# Supplementary job: re-run Softmax (TEST_TARGET_IDX=2) with corrected grep so the
# `[Softmax] 128 x 128 takes:` line is captured (the parent baseline script's grep
# used "[SoftMax]" which is case-mismatched against the binary's "[Softmax]").

echo "════════════════════════════════════════════════════════════"
echo "  NEXUS Softmax-only re-run on MN5 H100"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1 gcc/11.4.0

NEXUS_BUILD=/gpfs/projects/etur02/hkanpak/Comp390Project/nexus/cuda/build
cd ${NEXUS_BUILD}/bin

unset CPATH
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

TRIALS=3
for trial in $(seq 1 ${TRIALS}); do
  echo ""
  echo "--- Softmax trial ${trial}/${TRIALS} ---"
  /usr/bin/time -v ./main 2 2>&1 | grep -E "\[Softmax\]|\[NEXUS-driver\]|Mean Absolute Error|Average Error|Elapsed|Maximum resident|User time|System time" || true
done

echo ""
echo "End: $(date)"
