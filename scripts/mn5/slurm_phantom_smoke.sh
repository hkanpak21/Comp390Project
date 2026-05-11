#!/bin/bash
#SBATCH --job-name=nexus-phantom-smoke
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/phantom_smoke_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/phantom_smoke_%j.err

# Lane D / S15 — Phantom thread-safety smoke test.
# Two std::threads each call bootstrap_3 on a separate GPU with their
# own PhantomContext + GaloisKeyStore. Acceptance: both threads finish
# without exception and decrypted MAE < 1e-5 vs all-1.0 reference.

echo "════════════════════════════════════════════════════════════"
echo "  S15 — Phantom thread-safety smoke test (2 GPUs)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

echo ">>> Running phantom_threadsafe_smoke (2 GPUs)..."
${PROJECT}/build/bin/phantom_threadsafe_smoke 2
RC=$?

echo ""
echo "Exit code: ${RC}"
echo "End: $(date)"
exit ${RC}
