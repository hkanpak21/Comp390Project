#!/usr/bin/env bash
# nsys_multigpu_profile.sh — Nsight Systems profile for multi-GPU NCCL runs
#
# Usage:
#   ./profiling/nsys_multigpu_profile.sh <n_gpus> <output_prefix> [extra_binary_args...]
#
# Example:
#   ./profiling/nsys_multigpu_profile.sh 8 bert_8gpu
#   ./profiling/nsys_multigpu_profile.sh 4 bert_4gpu_oa --algo 1
#
# Multi-GPU profiling notes:
#   - nsys captures ALL GPUs when --gpu-metrics-device=all is set.
#   - NCCL timeline shows collectives as green bands; they should OVERLAP
#     with GPU compute rows (blue). No overlap = communication is serialized.
#   - Set NCCL_DEBUG=INFO to get NCCL initialization log in stdout.
#   - Set NCCL_ALGO=Ring or NCCL_ALGO=Tree to force algorithm selection.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <n_gpus> <output_prefix> [extra_binary_args...]"
    exit 1
fi

N_GPUS="$1"
OUTPUT_PREFIX="$2"
shift 2

BINARY="./build/benchmarks/bert_inference"
OUTPUT_DIR="experiments/results/profiles"
mkdir -p "${OUTPUT_DIR}"
OUTFILE="${OUTPUT_DIR}/${OUTPUT_PREFIX}"

echo "=== Multi-GPU Nsight Systems Profile ==="
echo "  GPUs     : ${N_GPUS}"
echo "  Binary   : ${BINARY}"
echo "  Extra    : $*"
echo "  Output   : ${OUTFILE}.nsys-rep"
echo ""

# NCCL environment for profiling
export NCCL_DEBUG=INFO              # log initialization + topology detection
export NCCL_DEBUG_SUBSYS=ALL        # include all NCCL subsystems in log
export NCCL_ALGO=Ring               # force Ring algorithm for determinism
# export NCCL_PROTO=Simple          # uncomment to disable pipelining (simpler traces)

# nsys flags specific to multi-GPU:
#   --gpu-metrics-device=all  : must capture all GPUs for NVSwitch visibility
#   --trace=nccl              : shows NCCL collectives as separate timeline rows
#   --mpi-impl=openmpi        : if using mpirun for multi-process launch
#
# For single-process multi-GPU (our default), no mpi flags needed.

nsys profile \
    --trace=cuda,nvtx,nccl,osrt \
    --gpu-metrics-device=all \
    --cudabacktrace=all \
    --cpuctxsw=none \
    --output="${OUTFILE}" \
    --force-overwrite=true \
    "${BINARY}" --n-gpus "${N_GPUS}" "$@"

echo ""
echo "Profile saved: ${OUTFILE}.nsys-rep"

# Text summaries
nsys stats --report gputrace --format csv \
    --output "${OUTFILE}_gputrace" "${OUTFILE}.nsys-rep" 2>/dev/null || true
nsys stats --report nvtxsum  --format csv \
    --output "${OUTFILE}_nvtxsum" "${OUTFILE}.nsys-rep" 2>/dev/null || true
nsys stats --report nccl     --format csv \
    --output "${OUTFILE}_nccl"    "${OUTFILE}.nsys-rep" 2>/dev/null || true

echo ""
echo "=== NCCL Collective Summary ==="
[[ -f "${OUTFILE}_nccl.csv" ]] && column -t -s',' "${OUTFILE}_nccl.csv" | head -20 || true

echo ""
echo "What to verify in the timeline:"
echo "  1. NCCL AllGather/AllReduce rows appear OVERLAPPING with GPU compute rows."
echo "     If they appear AFTER the compute rows (serialized), compute-comm overlap"
echo "     is not working -> check stream_manager.cuh event synchronization."
echo "  2. All 8 GPU rows show similar activity (no idle GPUs = good load balance)."
echo "  3. NVTX [NCCL-AllGather] range duration < [KeySwitch-InnerProduct] range."
echo "     If comm > compute, NCCL is the bottleneck (unexpected for NVSwitch)."
