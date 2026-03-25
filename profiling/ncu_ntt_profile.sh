#!/usr/bin/env bash
# ncu_ntt_profile.sh — Deep Nsight Compute profile of NTT and key-switching kernels
#
# Usage:
#   ./profiling/ncu_ntt_profile.sh <binary> <output_prefix> [binary_args...]
#
# Run AFTER nsys_profile.sh — use nsys results to confirm kernel names first.
# ncu slows execution 10-100x; target specific kernels only.
#
# Requires: Nsight Compute >= 2022.3 (pre-installed on Deep Learning AMI)

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <binary> <output_prefix> [binary_args...]"
    exit 1
fi

BINARY="$1"
OUTPUT_PREFIX="$2"
shift 2                  # $@ now contains only the binary's own arguments

OUTPUT_DIR="experiments/results/profiles"
mkdir -p "${OUTPUT_DIR}"
OUTFILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}"

echo "=== Nsight Compute — FHE Kernel Deep Dive ==="
echo "  Binary : ${BINARY}"
echo "  Args   : $*"
echo "  Output : ${OUTFILE}.ncu-rep"
echo ""

# ---------------------------------------------------------------------------
# Kernel name patterns (from actual Phantom source code)
#
# NTT kernels (phantom/include/ntt.cuh):
#   nwt_2d_radix8_forward_inplace
#   nwt_2d_radix8_backward_inplace
#   nwt_2d_radix8_forward_inplace_fuse_moddown
#   nwt_2d_radix8_forward_modup_fuse
#   nwt_2d_radix8_backward_inplace_include_special_mod
#
# Key-switching (phantom/include/evaluate.cuh):
#   key_switch_inner_prod_c2_and_evk
#
# RNS arithmetic:
#   multiply_rns_poly, add_rns_poly, sub_rns_poly
#
# If the regex matches too many kernels, narrow with --kernel-id ::N:M
# (profile only Nth through Mth invocation of each matched kernel)
# ---------------------------------------------------------------------------
KERNEL_REGEX="nwt_2d_radix8|key_switch_inner_prod|multiply_rns_poly|add_rns_poly"

# ---------------------------------------------------------------------------
# Profile
#
# --replay-mode kernel : profiles each kernel invocation independently by
#   replaying it N times to collect different metric sets. SAFER than
#   'application' for programs using NCCL (application replay can deadlock
#   because NCCL requires all ranks to make progress simultaneously).
#
# --kernel-id ::1:3 : profile only invocations 1-3 of each matched kernel
#   (remove to profile all invocations, but that's very slow)
#
# Key sections collected:
#   SpeedOfLight            - roofline position (are we compute or memory bound?)
#   MemoryWorkloadAnalysis  - L1/L2/DRAM utilization and bandwidth
#   WarpStateStatistics     - where warps are stalling (memory latency vs compute)
#   Occupancy               - active warps / theoretical max
# ---------------------------------------------------------------------------
ncu \
    --set full \
    --replay-mode kernel \
    --kernel-id "::1:3" \
    -k "regex:(${KERNEL_REGEX})" \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section WarpStateStatistics \
    --section Occupancy \
    --output "${OUTFILE}" \
    --force-overwrite \
    "${BINARY}" "$@"

echo ""
echo "Profile saved: ${OUTFILE}.ncu-rep"

# ---------------------------------------------------------------------------
# Quick text summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Per-Kernel Summary ==="
ncu --import "${OUTFILE}.ncu-rep" --print-summary per-kernel 2>/dev/null | head -40 || true

echo ""
echo "=== Key Metrics Reference ==="
cat << 'EOF'
METRIC                                                     INTERPRETATION
------                                                     --------------
sm__throughput.avg.pct_of_peak_sustained_elapsed           Compute util %.
                                                            >80% -> compute-bound (good for key-switch)
                                                            <40% -> memory-bound (expected for NTT)

dram__bytes_read.sum / gpc__cycles_elapsed.avg * clock     HBM bandwidth.
                                                            A100 peak: ~2039 GB/s
                                                            NTT target: >1600 GB/s (80% BW util)

sm__warps_active.avg.pct_of_peak_sustained_active          Occupancy %.
                                                            >50% good; <25% = launch config issue

smsp__warp_issue_stalled_long_scoreboard_per_warp_active   L2/DRAM latency stalls.
                                                            High -> memory-bound, consider prefetching

smsp__warp_issue_stalled_mio_throttle_per_warp_active      Math issue throttle.
                                                            High -> compute-bound

ROOFLINE FOR NTT (N=65536, L=20 limbs):
  Arithmetic intensity = (NTT FLOPs) / (bytes moved)
                       = (N * L * 16 * log2(N)) / (N * L * 8 bytes)  ~= 2 FLOP/byte
  A100 roofline at intensity 2: limited by HBM bandwidth (~2TB/s * 2 = ~4 TFLOP/s)
  If measured < roofline -> NTT is sub-optimal (register pressure or bank conflicts)
EOF

echo ""
echo "Export metrics to CSV for plotting:"
echo "  ncu --import ${OUTFILE}.ncu-rep \\"
echo "      --csv --print-details all \\"
echo "      --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\\"
echo "                l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \\"
echo "      | tee ${OUTPUT_DIR}/${OUTPUT_PREFIX}_metrics.csv"
echo ""
echo "To view in GUI: scp ${OUTFILE}.ncu-rep <local>: && ncu-ui <file>"
