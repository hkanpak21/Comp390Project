#!/bin/bash
# Build script for MN5
# Usage: bash scripts/mn5/build_mn5.sh [--with-mpi]
#
# Default: builds single-node and multi-GPU targets WITHOUT MPI.
# This avoids the Intel-compiled libmpi.so linker error on MN5 ACC partition.
# Pass --with-mpi to also build multi-node targets (requires correct OpenMPI module).

set -e

WITH_MPI=0
for arg in "$@"; do
    [[ "$arg" == "--with-mpi" ]] && WITH_MPI=1
done

# Load modules — do NOT load openmpi for the default build (causes Intel linker error)
module purge
if [[ $WITH_MPI -eq 1 ]]; then
    module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1 openmpi
    MPI_FLAG="-DNEXUS_USE_MPI=ON"
    echo "Building WITH MPI support (multi-node targets enabled)"
else
    module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1
    MPI_FLAG="-DNEXUS_USE_MPI=OFF"
    echo "Building WITHOUT MPI (single-node + multi-GPU only; avoids Intel linker error)"
fi

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
cd ${PROJECT}

# Ensure NTL is available (needed by bootstrapping)
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH
export CPATH=/gpfs/projects/etur02/hkanpak/local/include:$CPATH

mkdir -p build && cd build

cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    ${MPI_FLAG} \
    -DCMAKE_BUILD_TYPE=Release \
    -DNTL_DIR=/gpfs/projects/etur02/hkanpak/local

# Single-node and multi-GPU targets (always built)
SINGLE_NODE_TARGETS="bootstrap_test_n65536 dist_bootstrap_bench bert_dks_multigpu llama_dks_multigpu bert_encoder_multigpu_n65536 llama_layer_multigpu_n65536 bert_hp_multigpu llama_hp_multigpu bootstrap_align_n32k bootstrap_mgpu_align gelu_align_n65k gelu_mgpu_align layernorm_align_n65k layernorm_mgpu_align softmax_align_n65k softmax_mgpu_align matmul_align_n8k argmax_align_n32k bootstrap_diagnose"

if [[ $WITH_MPI -eq 1 ]]; then
    MULTINODE_TARGETS="bert_dks_multinode bert_encoder_multinode_n65536 llama_layer_multinode_n65536 bert_hp_multinode llama_hp_multinode"
    ALL_TARGETS="${SINGLE_NODE_TARGETS} ${MULTINODE_TARGETS}"
else
    ALL_TARGETS="${SINGLE_NODE_TARGETS}"
fi

make -j$(nproc) ${ALL_TARGETS} 2>&1 | tail -50

echo ""
echo "Build complete. Binaries:"
for t in ${ALL_TARGETS}; do
    ls -la bin/${t} 2>/dev/null || echo "  MISSING: bin/${t}"
done
