#!/bin/bash
#SBATCH --job-name=nexus-bert-enc-n32k
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/bert_encoder_multigpu_n32k_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/bert_encoder_multigpu_n32k_%j.err

echo "════════════════════════════════════════════════════════════"
echo "  multiNEXUS BERT Encoder — N=32768, 4×H100 (NEXUS-matched)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Architecture: 4 heads, 4 bootstraps/layer, 4×H100"
echo "  Comparison target: NEXUS reports 5,600 ms bootstrap on 4×A100 at N=32768"
echo "  Note: H100 is ~1.5–2× faster than A100 in raw throughput"
echo "  Note: NEXUS uses N=32768 only for bootstrap; we use the same N here"
echo "        to make a hardware-matched comparison at equal ring degree."
echo ""

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

cd ${PROJECT}

echo ">>> GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 5 independent runs so the user can compute mean ± std for both
# bootstrap latency and one-layer BERT encoder time.
NRUNS=5
for i in $(seq 1 ${NRUNS}); do
    echo "════════════════════════════════════════════════════════════"
    echo ">>> Run ${i}/${NRUNS}: bert_encoder_multigpu --n-gpus 4 --heads 4"
    echo "════════════════════════════════════════════════════════════"
    ${PROJECT}/build/bin/bert_encoder_multigpu --n-gpus 4 --heads 4 --inner 16
    echo ""
done

echo ""
echo "End: $(date)"
