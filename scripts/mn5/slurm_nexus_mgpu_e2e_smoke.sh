#!/bin/bash
#SBATCH --job-name=nexus-mgpu-smoke
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/nexus_mgpu_smoke_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/nexus_mgpu_smoke_%j.err

# THE HEADLINE EXPERIMENT — smoke test for nexus_mgpu_e2e.
# 1 layer × 12 heads on 4× H100. Verifies the chain runs end-to-end without
# crashing. Goal: confirm we can drop the full 12-layer × 12-head measurement.

echo "════════════════════════════════════════════════════════════"
echo "  HEADLINE-SMOKE — nexus_mgpu_e2e 1L × 12H (4× H100)"
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

echo ">>> Running nexus_mgpu_e2e --n-gpus 4 --heads 12 --layers 1 --skip-ref ..."
${PROJECT}/build/bin/nexus_mgpu_e2e \
    --n-gpus 4 \
    --heads 12 \
    --layers 1 \
    --skip-ref
RC=$?

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
