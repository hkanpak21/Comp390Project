#!/usr/bin/env bash
# ncu_ntt_profile.sh — Deep Nsight Compute profile of NTT and key-switching kernels
# Usage: ./profiling/ncu_ntt_profile.sh [binary] [output_prefix]
#
# Run AFTER nsys_profile.sh to identify which kernels to dig into.
# ncu is expensive (10-100× slowdown per kernel) — target specific kernels only.

set -euo pipefail

BINARY="${1:-./build/bin/bert_inference}"
OUTPUT="${2:-ntt_profile_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="experiments/results/profiles"

mkdir -p "${OUTPUT_DIR}"

echo "=== Nsight Compute — NTT Kernel Deep Dive ==="
echo "  Binary : ${BINARY}"
echo "  Output : ${OUTPUT_DIR}/${OUTPUT}.ncu-rep"
echo ""

# ── Profile NTT kernels ───────────────────────────────────────────────────────
# -k regex: targets kernels matching pattern; adjust based on actual kernel names from nsys
# --set full: collect all metrics (slow but comprehensive)
# --replay-mode: application (replays full app once per metric set)
ncu \
    --set full \
    --replay-mode application \
    -k "regex:(ntt|NTT|poly_mul|key_switch|bootstrap)" \
    --output "${OUTPUT_DIR}/${OUTPUT}" \
    --force-overwrite \
    "${BINARY}" "$@"

echo ""
echo "Profile saved: ${OUTPUT_DIR}/${OUTPUT}.ncu-rep"
echo ""
echo "Key metrics to examine:"
echo "  - sm__throughput.avg.pct_of_peak_sustained_elapsed  (compute utilization)"
echo "  - l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum      (global memory loads)"
echo "  - dram__bytes.sum / duration                         (memory bandwidth)"
echo "  - sm__warps_active.avg.pct_of_peak_sustained_active (occupancy)"
echo ""
echo "Roofline: look for memory-bound (NTT typically is) vs compute-bound kernels"
echo ""

# ── Quick summary ─────────────────────────────────────────────────────────────
echo "=== Top kernels by duration ==="
ncu \
    --import "${OUTPUT_DIR}/${OUTPUT}.ncu-rep" \
    --print-summary per-kernel 2>/dev/null | head -30
