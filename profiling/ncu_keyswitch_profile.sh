#!/usr/bin/env bash
# ncu_keyswitch_profile.sh — Targeted Nsight Compute profile of key-switching ONLY
#
# Usage:
#   ./profiling/ncu_keyswitch_profile.sh <binary> <output_prefix> [binary_args...]
#
# Why separate from ncu_ntt_profile.sh?
# Key-switching and NTT have OPPOSITE bottleneck profiles:
#   NTT           -> memory-bound  -> optimize for HBM bandwidth, coalescing
#   Key-switching -> compute-bound -> optimize for int64 throughput, occupancy
# Profiling them together in --set full is slow; splitting them allows focused fixes.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <binary> <output_prefix> [binary_args...]"
    exit 1
fi

BINARY="$1"; OUTPUT_PREFIX="$2"; shift 2
OUTPUT_DIR="experiments/results/profiles"
mkdir -p "${OUTPUT_DIR}"
OUTFILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}"

echo "=== Nsight Compute — Key-Switching Kernel Profile ==="
echo "  Binary : ${BINARY}  Args: $*"
echo "  Output : ${OUTFILE}.ncu-rep"
echo ""

ncu \
    --set full \
    --replay-mode kernel \
    --kernel-id "::1:5" \
    -k "regex:(key_switch_inner_prod|relin|galois_apply)" \
    --section SpeedOfLight \
    --section ComputeWorkloadAnalysis \
    --section InstructionStats \
    --section WarpStateStatistics \
    --output "${OUTFILE}" \
    --force-overwrite \
    "${BINARY}" "$@"

echo ""
echo "Profile saved: ${OUTFILE}.ncu-rep"
echo ""
echo "Key-switch target metrics:"
echo "  sm__throughput > 70%          (want compute-bound)"
echo "  smsp__inst_executed_pipe_xu_pred_on.avg  (int64 execution throughput)"
echo "  Low DRAM stalls               (compute, not memory bottleneck)"
