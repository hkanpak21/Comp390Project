#!/bin/bash
# Sync local code to MN5. Excludes build artifacts, only ships source.
# Usage: ./scripts/sync_to_mn5.sh
set -e

LOCAL_ROOT="/Users/a90/Documents/COURSES/SPRING2026/Comp390/Comp390Project"
MN5_ROOT="/gpfs/projects/etur02/hkanpak/Comp390Project"
MN5_HOST="mn5-gpu"

echo "[sync] Syncing src/ → MN5..."
rsync -avz --include='*/' --include='*.cu' --include='*.cuh' --include='*.h' --include='*.cpp' --include='*.txt' --exclude='*' \
    "${LOCAL_ROOT}/src/" "${MN5_HOST}:${MN5_ROOT}/src/"

echo "[sync] Syncing scripts/mn5/ → MN5..."
rsync -avz "${LOCAL_ROOT}/scripts/mn5/" "${MN5_HOST}:${MN5_ROOT}/scripts/mn5/"

echo "[sync] Syncing CMakeLists.txt → MN5..."
rsync -avz "${LOCAL_ROOT}/CMakeLists.txt" "${MN5_HOST}:${MN5_ROOT}/CMakeLists.txt"

echo "[sync] Done."
