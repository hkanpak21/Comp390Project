#!/bin/bash
#SBATCH --job-name=nexus-dks-boot
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/dks_bootstrap_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/dks_bootstrap_%j.err

echo "════════════════════════════════════════════════════════════"
echo "  multiNEXUS DKS Bootstrap Benchmark — N=65536"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Measures: DKS rotation latency + projected bootstrap speedup"
echo "  Parameters: N=65536, sparse_slots=16384, 50 bootstrap keys"
echo "  Scaling: 1 / 2 / 4 H100s"
echo ""

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

# NCCL tuning for NVLink (intra-node)
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

cd ${PROJECT}

echo ">>> GPU topology:"
nvidia-smi topo -m
echo ""

echo ">>> Running dist_bootstrap_bench (1/2/4 GPUs)..."
${PROJECT}/build/bin/dist_bootstrap_bench 4

echo ""
echo "End: $(date)"
