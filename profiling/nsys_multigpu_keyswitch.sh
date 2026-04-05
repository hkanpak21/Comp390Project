#!/bin/bash
# Profile multi-GPU key-switching with Nsight Systems
#
# Produces a timeline showing:
#   - CUDA kernel execution per GPU
#   - NCCL communication (AllGather, AllReduce)
#   - Compute-communication overlap
#   - Memory allocation overhead
#
# Usage:
#   ./profiling/nsys_multigpu_keyswitch.sh [n_gpus] [binary]
#
# Example:
#   ./profiling/nsys_multigpu_keyswitch.sh 2 ./build/bin/multi_gpu_keyswitch_test
#
# Output: profiling/reports/multigpu_ks_<date>.nsys-rep

N_GPUS=${1:-2}
BINARY=${2:-"./build/bin/multi_gpu_keyswitch_test"}
REPORT_DIR="profiling/reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_NAME="multigpu_ks_${N_GPUS}gpu_${TIMESTAMP}"

mkdir -p "$REPORT_DIR"

echo "=== Nsight Systems Multi-GPU Key-Switching Profile ==="
echo "GPUs: $N_GPUS"
echo "Binary: $BINARY"
echo "Output: ${REPORT_DIR}/${REPORT_NAME}.nsys-rep"
echo ""

# Profile with CUDA, NCCL, NVTX, and OS runtime tracing
nsys profile \
    --output "${REPORT_DIR}/${REPORT_NAME}" \
    --trace cuda,nvtx,nccl,osrt \
    --cuda-memory-usage true \
    --gpu-metrics-device all \
    --force-overwrite true \
    --stats true \
    "$BINARY" --n-gpus "$N_GPUS" --verbose

echo ""
echo "=== Profile saved to ${REPORT_DIR}/${REPORT_NAME}.nsys-rep ==="
echo "Open with: nsys-ui ${REPORT_DIR}/${REPORT_NAME}.nsys-rep"

# Print summary statistics
echo ""
echo "=== Quick Stats ==="
nsys stats "${REPORT_DIR}/${REPORT_NAME}.nsys-rep" --report cuda_gpu_kern_sum 2>/dev/null | head -30
echo ""
nsys stats "${REPORT_DIR}/${REPORT_NAME}.nsys-rep" --report nccl_gpu_kern_sum 2>/dev/null | head -20
