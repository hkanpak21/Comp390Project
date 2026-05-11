#!/bin/bash
#SBATCH --job-name=hp-tput-16g
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/bert_hp_throughput_16gpu_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/bert_hp_throughput_16gpu_%j.err

# MEASURE-04 — Goal 2 data-parallel throughput at 16-GPU (weak scaling).
# Launches G=16 concurrent single-GPU bert_hp_multigpu instances across
# 4 nodes (4 instances per node, one per GPU). Each runs full
# 12-layer × 12-head BERT at logN=15 independently. Aggregate throughput
# = 16 inferences / max(wall_i for i in 1..16).
#
# Uses srun --multi-prog to dispatch one task per GPU. Each task pins
# itself to its local-rank GPU via SLURM_LOCALID.

set -e

echo "════════════════════════════════════════════════════════════"
echo "  MEASURE-04 — HP-BERT throughput, 16 concurrent × 1 GPU each"
echo "  Job: ${SLURM_JOB_ID}, Nodes: ${SLURM_JOB_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
LOGDIR=/gpfs/projects/etur02/hkanpak/logs
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> GPU info (head node):"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Wrapper script — pins each rank to its local GPU and runs one
# independent bert_hp_multigpu instance.
WRAPPER=$(mktemp /tmp/hp_throughput_wrap_${SLURM_JOB_ID}_XXXX.sh)
cat > ${WRAPPER} <<'WRAP'
#!/bin/bash
set -e
GLOBAL_RANK=${SLURM_PROCID}
LOCAL_RANK=${SLURM_LOCALID}
INSTANCE_LOG=/gpfs/projects/etur02/hkanpak/logs/bert_hp_throughput_16gpu_${SLURM_JOB_ID}_r${GLOBAL_RANK}.log
{
    echo "[rank ${GLOBAL_RANK} local ${LOCAL_RANK} host $(hostname)] start $(date)"
    export CUDA_VISIBLE_DEVICES=${LOCAL_RANK}
    export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH
    /gpfs/projects/etur02/hkanpak/Comp390Project/build/bin/bert_hp_multigpu \
        --N 32768 \
        --n-gpus 1 \
        --heads 12 \
        --layers 12 \
        --skip-ref
    echo "[rank ${GLOBAL_RANK}] end $(date)"
} > ${INSTANCE_LOG} 2>&1
WRAP
chmod +x ${WRAPPER}

START_NS=$(date +%s%N)

# 16 tasks total, 4 per node; --mpi=none because instances are independent.
srun --mpi=none --kill-on-bad-exit=0 ${WRAPPER}
RC=$?

END_NS=$(date +%s%N)
WALL_S=$(awk "BEGIN { print (${END_NS} - ${START_NS}) / 1e9 }")
THROUGHPUT_IPS=$(awk "BEGIN { print 16 / ${WALL_S} }")

rm -f ${WRAPPER}

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Aggregate wall time: ${WALL_S} s"
echo "  Aggregate throughput (16 inferences / wall): ${THROUGHPUT_IPS} inferences/s"
echo "  Per-instance logs: ${LOGDIR}/bert_hp_throughput_16gpu_${SLURM_JOB_ID}_r*.log"
echo "════════════════════════════════════════════════════════════"

echo "Exit: ${RC}"
echo "End: $(date)"
exit ${RC}
