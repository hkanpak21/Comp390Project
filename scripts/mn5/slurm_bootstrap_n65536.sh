#!/bin/bash
#SBATCH --job-name=nexus-boot65k
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/boot_n65536_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/boot_n65536_%j.err

echo "════════════════════════════════════════════════════════════"
echo "  Bootstrap Test — N=65536 (1 GPU)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ""
echo ">>> Bootstrap accuracy test at N=65536"
echo ">>> This tests if bootstrap works at full NEXUS parameter set"
echo ">>> Expected: MAE < 0.01, memory usage ~25-35 GB on H100 64GB"
echo ""

${PROJECT}/build/bin/bootstrap_test_n65536

echo ""
echo "End: $(date)"
