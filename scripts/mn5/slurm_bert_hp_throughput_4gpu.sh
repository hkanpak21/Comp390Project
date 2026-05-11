#!/bin/bash
#SBATCH --job-name=hp-tput-4g
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/bert_hp_throughput_4gpu_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/bert_hp_throughput_4gpu_%j.err

# MEASURE-03 — Goal 2 data-parallel throughput at 4-GPU (weak scaling).
# Launches G=4 concurrent single-GPU bert_hp_multigpu instances, one per
# GPU, each running full 12-layer × 12-head BERT at logN=15. Aggregate
# throughput = 4 inferences / max(wall_i for i in 1..4).
#
# Each instance is pinned to its GPU via CUDA_VISIBLE_DEVICES. No
# inter-instance communication.

set -e

echo "════════════════════════════════════════════════════════════"
echo "  MEASURE-03 — HP-BERT throughput, 4 concurrent × 1 GPU each"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
LOGDIR=/gpfs/projects/etur02/hkanpak/logs
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

cd ${PROJECT}

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

START_NS=$(date +%s%N)

PIDS=()
for GPU in 0 1 2 3; do
    INSTANCE_LOG=${LOGDIR}/bert_hp_throughput_4gpu_${SLURM_JOB_ID}_gpu${GPU}.log
    (
        export CUDA_VISIBLE_DEVICES=${GPU}
        echo "[gpu${GPU}] start $(date)"
        ${PROJECT}/build/bin/bert_hp_multigpu \
            --N 32768 \
            --n-gpus 1 \
            --heads 12 \
            --layers 12 \
            --skip-ref
        echo "[gpu${GPU}] end $(date)"
    ) > ${INSTANCE_LOG} 2>&1 &
    PIDS+=($!)
    echo ">>> launched instance for gpu${GPU} (pid $!)"
done

# Wait for all instances; collect exit codes.
EXIT=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo ">>> instance gpu${i} OK"
    else
        rc=$?
        echo ">>> instance gpu${i} FAILED (exit=${rc})"
        EXIT=${rc}
    fi
done

END_NS=$(date +%s%N)
WALL_S=$(awk "BEGIN { print (${END_NS} - ${START_NS}) / 1e9 }")
THROUGHPUT_IPS=$(awk "BEGIN { print 4 / ${WALL_S} }")

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Aggregate wall time: ${WALL_S} s"
echo "  Aggregate throughput (4 inferences / wall): ${THROUGHPUT_IPS} inferences/s"
echo "  Per-instance logs: ${LOGDIR}/bert_hp_throughput_4gpu_${SLURM_JOB_ID}_gpu*.log"
echo "════════════════════════════════════════════════════════════"

echo "End: $(date)"
exit ${EXIT}
