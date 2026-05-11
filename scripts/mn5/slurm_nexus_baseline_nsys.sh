#!/bin/bash
#SBATCH --job-name=nexus-h100-nsys
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/nexus_nsys_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/nexus_nsys_%j.err

# Phase 0 R3 follow-up: nsys profile of vanilla NEXUS runs to capture
# per-kernel breakdown and confirm single-GPU usage.

echo "════════════════════════════════════════════════════════════"
echo "  Vanilla NEXUS — nsys profiling on MN5 H100"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1 gcc/11.4.0

NEXUS_BUILD=/gpfs/projects/etur02/hkanpak/Comp390Project/nexus/cuda/build
LOGS=/gpfs/projects/etur02/hkanpak/logs

cd ${NEXUS_BUILD}/bin

unset CPATH
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

# Profile the GELU op (idx=4) and bootstrapping — the most expensive ops we care about.
NSYS=$(which nsys 2>/dev/null || echo "/apps/ACC/CUDA/12.8/bin/nsys")
echo ">>> nsys: $NSYS"
$NSYS --version || true
echo ""

for op_idx in 0 4; do
  echo ""
  echo "--- nsys profile main ${op_idx} ---"
  ${NSYS} profile \
    --output ${LOGS}/nexus_op${op_idx}_${SLURM_JOB_ID} \
    --trace cuda,nvtx,osrt \
    --force-overwrite=true \
    ./main ${op_idx} 2>&1 | tail -30 || true
done

echo ""
echo "--- nsys profile bootstrapping ---"
${NSYS} profile \
  --output ${LOGS}/nexus_boot_${SLURM_JOB_ID} \
  --trace cuda,nvtx,osrt \
  --force-overwrite=true \
  ./bootstrapping 2>&1 | tail -30 || true

echo ""
echo "End: $(date)"
ls -la ${LOGS}/nexus_*_${SLURM_JOB_ID}.* 2>/dev/null
