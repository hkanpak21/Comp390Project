#!/bin/bash
#SBATCH --job-name=boot-diag
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_ehpc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/bootstrap_diagnose_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/bootstrap_diagnose_%j.err

# Lane BOOT-DIAGNOSE — workload decomposition for the 4× bootstrap gap.
#
# Runs OUR Bootstrapper at TWO configs back-to-back on the same H100:
#   1. NEXUS-standalone workload  (logN=15, sparse=2^13, main=16, bs=14)
#       → expect ~252 ms (already confirmed by JOBID 40368887: 250.6 ms)
#   2. HP-BERT workload           (logN=15, sparse=2^13, main=21, bs=14)
#       → tells us how much of the 1,032 ms in-pipeline cost is workload
#         vs pipeline overhead.
#
# 100 isolated calls per config; reports median + σ.

set -e

echo "════════════════════════════════════════════════════════════"
echo "  BOOT-DIAGNOSE — bootstrap workload decomposition (single H100)"
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

BIN=${PROJECT}/build/bin/bootstrap_diagnose
if [ ! -x "${BIN}" ]; then
    echo "[FATAL] binary not found at ${BIN}"
    echo "        Build with: cd build && make -j20 bootstrap_diagnose"
    exit 1
fi

#===========================================================================
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  PHASE A — NEXUS standalone workload (control)"
echo "  logN=15, logn=13 (sparse=8192), main_mod=16, bs_mod=14"
echo "  Expected: ~250 ms (confirms binary matches bootstrap_align_n32k)"
echo "════════════════════════════════════════════════════════════"
${BIN} --logN 15 --logn 13 --main-mod 16 --bs-mod 14 --calls 100
RC1=$?
echo "Exit code Phase A: ${RC1}"

#===========================================================================
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  PHASE B — HP-BERT workload (diagnostic)"
echo "  logN=15, logn=13 (sparse=8192), main_mod=21, bs_mod=14"
echo "  In-pipeline reference: 1,032 ms"
echo "  Expected: tells us workload vs pipeline overhead split"
echo "════════════════════════════════════════════════════════════"
${BIN} --logN 15 --logn 13 --main-mod 21 --bs-mod 14 --calls 100
RC2=$?
echo "Exit code Phase B: ${RC2}"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  BOOT-DIAGNOSE COMPLETE"
echo "  Phase A exit: ${RC1} (NEXUS workload)"
echo "  Phase B exit: ${RC2} (HP-BERT workload)"
echo "  End: $(date)"
echo "════════════════════════════════════════════════════════════"
exit $((RC1 + RC2))
