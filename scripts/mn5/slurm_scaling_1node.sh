#!/bin/bash
#SBATCH --job-name=nexus-scale-1n
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/scale_1node_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/scale_1node_%j.err

echo "════════════════════════════════════════════════════════════"
echo "  NEXUS Scaling Experiment — 1 Node (4x H100)"
echo "  Job: ${SLURM_JOB_ID}, Node: ${SLURM_NODELIST}"
echo "  Start: $(date)"
echo "════════════════════════════════════════════════════════════"

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

export NCCL_DEBUG=WARN
export NCCL_IB_HCA=mlx5
export NCCL_IB_CUDA_SUPPORT=1
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
BIN=${PROJECT}/build/bin
cd ${PROJECT}

# NEXUS-matching params: 12 heads, hidden=768, inner=768, seq=128
HEADS=12
HIDDEN=768
INNER=768
SEQ=128
PARAMS="--heads ${HEADS} --hidden ${HIDDEN} --inner ${INNER} --seq-len ${SEQ}"

echo ""
echo "═══ N=65536 (NEXUS full param set) ═══"
echo "═══ Params: heads=${HEADS} hidden=${HIDDEN} inner=${INNER} seq=${SEQ} ═══"
echo ""

# Note: At N=65536, single GPU may OOM. Start with multi-GPU.
# If 1 GPU works, great — it shows H100 64GB can fit what A100 40GB couldn't.

# ── 1 GPU (may OOM — that's the point) ──
echo ">>> 1 GPU (testing if N=65536 fits on single H100 64GB)"
${BIN}/bert_encoder_multigpu --n-gpus 1 ${PARAMS} || echo ">>> 1 GPU: OOM or error (expected at N=65536)"
echo ""

# ── 2 GPU ──
echo ">>> 2 GPU scaling"
${BIN}/bert_encoder_multigpu --n-gpus 2 ${PARAMS} || echo ">>> 2 GPU: OOM or error"
echo ""

# ── 4 GPU (full node — this MUST work) ──
echo ">>> 4 GPU scaling (full node)"
${BIN}/bert_encoder_multigpu --n-gpus 4 ${PARAMS}
echo ""

echo "End: $(date)"
