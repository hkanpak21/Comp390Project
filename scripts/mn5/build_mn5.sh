#!/bin/bash
# Build script for MN5
# Usage: bash scripts/mn5/build_mn5.sh

set -e

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1 openmpi

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
cd ${PROJECT}

# Ensure NTL is available (needed by bootstrapping)
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH
export CPATH=/gpfs/projects/etur02/hkanpak/local/include:$CPATH

mkdir -p build && cd build

cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DUSE_MPI=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DNTL_DIR=/gpfs/projects/etur02/hkanpak/local

make -j$(nproc) bootstrap_test_n65536 bert_encoder_multigpu bert_encoder_multinode 2>&1 | tail -20

echo ""
echo "Build complete. Binaries:"
ls -la bin/bootstrap_test_n65536 bin/bert_encoder_multigpu bin/bert_encoder_multinode 2>/dev/null || echo "Check build errors above"
