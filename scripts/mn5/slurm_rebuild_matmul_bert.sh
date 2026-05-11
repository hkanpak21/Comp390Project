#!/bin/bash
#SBATCH --job-name=rebuild-mb
#SBATCH --account=etur02
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/projects/etur02/hkanpak/logs/rebuild_mb_%j.out
#SBATCH --error=/gpfs/projects/etur02/hkanpak/logs/rebuild_mb_%j.err

# FIX-BUILD-01: clean rebuild of matmul_align_n8k + bert_hp_multigpu
# after fixing the duplicate-ct_pipeline-symbol link error.
#
# Background: prior jobs 40422689 / 40422772 failed with multiple-definition
# linker errors on nexus_multi_gpu::CtPipeline::* because GLOB_RECURSE picked
# up two copies of ct_pipeline.cu (archive/pipeline/ + stale pipeline/).
# Fix applied prior to this job: removed stale src/multi_gpu/pipeline/ on MN5,
# synced CMakeLists.txt with `EXCLUDE REGEX "src/multi_gpu/archive/"`.

module purge
module load cuda/12.8 cmake/3.30.5 nccl/2.24.3-1

PROJECT=/gpfs/projects/etur02/hkanpak/Comp390Project
export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH
export CPATH=/gpfs/projects/etur02/hkanpak/local/include:$CPATH

cd ${PROJECT}

echo "=== FIX-BUILD-01 rebuild start: $(date) ==="
echo "Sanity: no stale src/multi_gpu/pipeline/"
ls -d src/multi_gpu/pipeline 2>/dev/null && { echo "FATAL: stale dir still present"; exit 1; } || echo "OK"
echo "Sanity: CMakeLists.txt has EXCLUDE filter"
grep -n 'FILTER MULTI_GPU_SRCS EXCLUDE REGEX' CMakeLists.txt || { echo "FATAL: EXCLUDE filter missing"; exit 1; }

# Clean stale build artifacts for the multi_gpu lib + the two target binaries.
# Force re-cmake by removing the cache; keep other dirs (Phantom, NEXUS) intact
# for fast incremental rebuild.
cd build
rm -rf CMakeFiles/nexus_multi_gpu.dir \
       CMakeFiles/matmul_align_n8k.dir \
       CMakeFiles/bert_hp_multigpu.dir \
       libnexus_multi_gpu.a \
       bin/matmul_align_n8k \
       bin/bert_hp_multigpu \
       CMakeCache.txt

cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    -DNEXUS_USE_MPI=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DNTL_DIR=/gpfs/projects/etur02/hkanpak/local

echo "=== make start: $(date) ==="
make -j80 matmul_align_n8k bert_hp_multigpu
RC=$?
echo "=== make end: $(date) (rc=$RC) ==="

echo ""
echo "=== Binary check ==="
for t in matmul_align_n8k bert_hp_multigpu; do
    if [ -x bin/$t ]; then
        echo "OK   bin/$t $(stat -c '%y %s' bin/$t)"
    else
        echo "MISS bin/$t"
        RC=1
    fi
done

exit $RC
