#!/bin/bash
# Submit all N=65536 scaling experiments
# Usage: bash scripts/mn5/submit_all_scaling.sh
#
# Order:
#   1. Bootstrap test at N=65536 (verify correctness + memory)
#   2. 1-node scaling (1, 2, 4 GPUs)
#   3. 2-node (8 GPUs)
#   4. 4-node (16 GPUs)
#   5. 8-node (32 GPUs)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p /gpfs/projects/etur02/hkanpak/logs

echo "Submitting N=65536 scaling experiments..."
echo ""

# Step 0: Bootstrap test (must pass before anything else)
J0=$(sbatch --parsable ${SCRIPT_DIR}/slurm_bootstrap_n65536.sh)
echo "  [0] Bootstrap test N=65536: job ${J0}"

# Step 1: 1-node (1, 2, 4 GPUs)
J1=$(sbatch --parsable --dependency=afterok:${J0} ${SCRIPT_DIR}/slurm_scaling_1node.sh)
echo "  [1] 1-node scaling:         job ${J1} (after bootstrap passes)"

# Step 2: 2-node (8 GPUs)
J2=$(sbatch --parsable --dependency=afterany:${J1} ${SCRIPT_DIR}/slurm_scaling_2node.sh)
echo "  [2] 2-node scaling:         job ${J2}"

# Step 3: 4-node (16 GPUs)
J4=$(sbatch --parsable --dependency=afterany:${J2} ${SCRIPT_DIR}/slurm_scaling_4node.sh)
echo "  [3] 4-node scaling:         job ${J4}"

# Step 4: 8-node (32 GPUs)
J8=$(sbatch --parsable --dependency=afterany:${J4} ${SCRIPT_DIR}/slurm_scaling_8node.sh)
echo "  [4] 8-node scaling:         job ${J8}"

echo ""
echo "All jobs submitted with dependencies."
echo "Monitor: squeue -u \$USER"
echo "Results: /gpfs/projects/etur02/hkanpak/logs/"
echo ""
echo "NOTE: Bootstrap test uses afterok dependency — if it fails,"
echo "      scaling jobs will NOT run. Fix bootstrap first."
