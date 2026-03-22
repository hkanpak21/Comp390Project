#!/usr/bin/env bash
# nsys_profile.sh — Profile NEXUS with Nsight Systems
# Usage: ./profiling/nsys_profile.sh [binary] [output_prefix]
# Example: ./profiling/nsys_profile.sh ./build/bin/bert_inference bert_1gpu

set -euo pipefail

BINARY="${1:-./build/bin/bert_inference}"
OUTPUT="${2:-fhe_profile_$(hostname)_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="experiments/results/profiles"

mkdir -p "${OUTPUT_DIR}"

echo "=== Nsight Systems Profile ==="
echo "  Binary   : ${BINARY}"
echo "  Output   : ${OUTPUT_DIR}/${OUTPUT}.nsys-rep"
echo "  GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# ── Profile command ───────────────────────────────────────────────────────────
# --trace: cuda (kernels/memcpy), nvtx (our annotations), nccl (collectives)
# --gpu-metrics-device: collect hardware metrics
# --cudabacktrace: resolve kernel launch stack traces
nsys profile \
    --trace=cuda,nvtx,nccl,osrt \
    --gpu-metrics-device=all \
    --cudabacktrace=all \
    --output="${OUTPUT_DIR}/${OUTPUT}" \
    --force-overwrite=true \
    "${BINARY}" "$@"

echo ""
echo "Profile saved: ${OUTPUT_DIR}/${OUTPUT}.nsys-rep"
echo ""
echo "To view: nsys-ui ${OUTPUT_DIR}/${OUTPUT}.nsys-rep"
echo "Or export timeline stats:"
echo "  nsys stats --report gputrace ${OUTPUT_DIR}/${OUTPUT}.nsys-rep"

# ── Quick text summary ────────────────────────────────────────────────────────
echo ""
echo "=== Top CUDA kernels by time ==="
nsys stats \
    --report gputrace \
    --format csv \
    "${OUTPUT_DIR}/${OUTPUT}.nsys-rep" 2>/dev/null \
    | head -20 \
    | column -t -s','
