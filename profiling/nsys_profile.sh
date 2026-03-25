#!/usr/bin/env bash
# nsys_profile.sh — Profile NEXUS FHE inference with Nsight Systems
#
# Usage:
#   ./profiling/nsys_profile.sh <binary> <output_prefix> [binary_args...]
#
# Examples:
#   ./profiling/nsys_profile.sh ./build/benchmarks/bert_inference bert_1gpu --n-gpus 1
#   ./profiling/nsys_profile.sh ./build/benchmarks/bert_inference bert_8gpu --n-gpus 8
#
# Requires: Nsight Systems >= 2022.1 (pre-installed on Deep Learning AMI)

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

echo "=== Nsight Systems Profile ==="
echo "  Binary   : ${BINARY}"
echo "  Args     : $*"
echo "  Output   : ${OUTFILE}.nsys-rep"
command -v nvidia-smi &>/dev/null && \
    echo "  GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# ---------------------------------------------------------------------------
# Profile
# Trace flags:
#   cuda  - CUDA kernel launches, memcpy, memset, sync
#   nvtx  - NVTX region annotations in our benchmarks (FHE ops, layers, comm)
#   nccl  - NCCL collective calls (AllGather, AllReduce, Broadcast)
#   osrt  - OS runtime (thread/mutex scheduling) to correlate CPU stalls
#
# --gpu-metrics-device=all : hardware PM counters from ALL GPUs simultaneously
# --cudabacktrace=all      : capture kernel launch call stack (for attribution)
# --cpuctxsw=none          : skip CPU context switch events (too noisy on 96-core)
# ---------------------------------------------------------------------------
nsys profile \
    --trace=cuda,nvtx,nccl,osrt \
    --gpu-metrics-device=all \
    --cudabacktrace=all \
    --cpuctxsw=none \
    --output="${OUTFILE}" \
    --force-overwrite=true \
    "${BINARY}" "$@"

echo ""
echo "Profile saved: ${OUTFILE}.nsys-rep"

# ---------------------------------------------------------------------------
# Text summaries (useful on headless EC2 without GUI)
# ---------------------------------------------------------------------------
echo ""
echo "=== Generating text summaries ==="

nsys stats --report gputrace --format csv \
    --output "${OUTFILE}_gputrace" "${OUTFILE}.nsys-rep" 2>/dev/null || true

nsys stats --report nvtxsum --format csv \
    --output "${OUTFILE}_nvtxsum" "${OUTFILE}.nsys-rep" 2>/dev/null || true

nsys stats --report nccl --format csv \
    --output "${OUTFILE}_nccl"    "${OUTFILE}.nsys-rep" 2>/dev/null || true

echo ""
echo "=== Top CUDA Kernels by Total Time ==="
[[ -f "${OUTFILE}_gputrace.csv" ]] && \
    sort -t',' -k3 -rn "${OUTFILE}_gputrace.csv" | head -20 | column -t -s',' 2>/dev/null || true

echo ""
echo "What to look for in the timeline:"
echo "  nwt_2d_radix8_*           NTT kernels    (memory-bound, should be 80%+ HBM BW)"
echo "  key_switch_inner_prod_*   Key-switching  (compute-bound)"
echo "  ncclAllGather/AllReduce   NCCL comms     (should OVERLAP with kernel rows)"
echo "  NVTX rows [KeySwitch] [Bootstrap] [MatMul]  our annotations"
echo ""
echo "To view in GUI: scp ${OUTFILE}.nsys-rep <local>: && nsys-ui <file>"
