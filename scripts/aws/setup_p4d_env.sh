#!/usr/bin/env bash
# setup_p4d_env.sh — Verify and configure p4d.24xlarge for NEXUS multi-GPU
# Run this on the p4d instance after launch (Deep Learning Base GPU AMI assumed)
# Usage: bash setup_p4d_env.sh

set -euo pipefail

echo "=== NEXUS Multi-GPU Environment Setup ==="
echo "Host: $(hostname) | $(date)"
echo ""

# ── 1. GPU inventory ──────────────────────────────────────────────────────────
echo "--- GPU Inventory ---"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
echo ""

# ── 2. CUDA version ───────────────────────────────────────────────────────────
echo "--- CUDA ---"
nvcc --version
echo "CUDA_HOME: ${CUDA_HOME:-not set}"
echo ""

# ── 3. NCCL version ───────────────────────────────────────────────────────────
echo "--- NCCL ---"
if [ -f /usr/include/nccl.h ]; then
    grep "NCCL_VERSION\b" /usr/include/nccl.h | head -3
else
    find /usr -name "nccl.h" 2>/dev/null | head -3
fi
# Check aws-ofi-nccl plugin
if ldconfig -p | grep -q "libnccl-net"; then
    echo "aws-ofi-nccl: FOUND"
else
    echo "aws-ofi-nccl: NOT FOUND — install from https://github.com/aws/aws-ofi-nccl"
fi
echo ""

# ── 4. EFA / libfabric ────────────────────────────────────────────────────────
echo "--- EFA / libfabric ---"
fi_info -p efa 2>/dev/null || echo "fi_info: EFA provider not found (OK for single-node)"
echo ""

# ── 5. CMake version ──────────────────────────────────────────────────────────
echo "--- CMake ---"
cmake --version | head -1
echo ""

# ── 6. Clone / update repo ────────────────────────────────────────────────────
REPO_DIR="${HOME}/comp390"
if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "--- Cloning project repo ---"
    git clone --recurse-submodules https://github.com/hkanpak21/Comp390Project.git "${REPO_DIR}"
else
    echo "--- Updating repo ---"
    cd "${REPO_DIR}" && git pull && git submodule update --init --recursive
fi
echo ""

# ── 7. Build Phantom ─────────────────────────────────────────────────────────
echo "--- Building Phantom FHE ---"
cd "${REPO_DIR}/vendor/phantom"
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo "Phantom build complete."
echo ""

# ── 8. Build NEXUS GPU (cuda/) ────────────────────────────────────────────────
echo "--- Building NEXUS cuda/ ---"
cd "${REPO_DIR}/vendor/nexus/cuda"
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo "NEXUS GPU build complete."
echo ""

# ── 9. Set environment variables for NCCL ────────────────────────────────────
cat >> "${HOME}/.bashrc" << 'EOF'

# NCCL / EFA environment for NEXUS multi-GPU
export NCCL_DEBUG=WARN
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_PROTO=simple
export NCCL_SOCKET_IFNAME=^lo,docker
EOF
echo "NCCL env vars appended to ~/.bashrc"
echo ""

echo "=== Setup complete ==="
echo "Next: run a single-GPU baseline to verify NEXUS is working"
echo "  cd ${REPO_DIR}/vendor/nexus/cuda/build && ./bert_inference"
