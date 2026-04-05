cmake_minimum_required(VERSION 3.20)
project(NexusMultiGPU LANGUAGES CXX CUDA)

# ── C++ / CUDA standards ──────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── GPU architecture ──────────────────────────────────────────────────────────
# A100 = 80, H100 = 90. Override from command line if needed:
#   cmake -DCMAKE_CUDA_ARCHITECTURES="80;90" ..
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

# ── Build type ────────────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ── Find packages ─────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# NCCL — looks for libnccl in standard paths; override with -DNCCL_ROOT=...
find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${NCCL_ROOT}/include $ENV{NCCL_ROOT}/include)
find_library(NCCL_LIBRARY nccl HINTS ${NCCL_ROOT}/lib $ENV{NCCL_ROOT}/lib)
if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(WARNING "NCCL not found — multi-GPU targets will be skipped")
    set(NCCL_FOUND FALSE)
else()
    set(NCCL_FOUND TRUE)
    message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
endif()

# MPI (needed for multi-node NCCL init)
find_package(MPI COMPONENTS CXX)

# ── Vendor submodules ─────────────────────────────────────────────────────────
# NEXUS (builds its own SEAL 4.1 + GPU inference)
# We include it as an external project or use its cmake targets directly.
# For now expose its include paths; actual build is via vendor/nexus/cuda/
set(NEXUS_ROOT ${CMAKE_SOURCE_DIR}/vendor/nexus)
set(NEXUS_SEAL_INCLUDE ${NEXUS_ROOT}/thirdparty/SEAL-4.1-bs/native/src)

# Phantom FHE (GPU-native CKKS)
set(PHANTOM_ROOT ${CMAKE_SOURCE_DIR}/vendor/phantom)
if(EXISTS ${PHANTOM_ROOT}/CMakeLists.txt)
    add_subdirectory(${PHANTOM_ROOT} phantom_build EXCLUDE_FROM_ALL)
endif()

# ── src/multi_gpu library ─────────────────────────────────────────────────────
if(NCCL_FOUND)
    file(GLOB_RECURSE MULTI_GPU_SRCS
        src/multi_gpu/comm/*.cu
        src/multi_gpu/partition/*.cu
        src/multi_gpu/keyswitching/*.cu
        src/multi_gpu/overlap/*.cu
    )

    add_library(nexus_multi_gpu STATIC ${MULTI_GPU_SRCS})
    target_include_directories(nexus_multi_gpu PUBLIC
        src/multi_gpu
        ${NCCL_INCLUDE_DIR}
        ${NEXUS_SEAL_INCLUDE}
        ${CUDAToolkit_INCLUDE_DIRS}
        ${PHANTOM_ROOT}/include
    )
    target_link_libraries(nexus_multi_gpu PUBLIC
        ${NCCL_LIBRARY}
        CUDA::cudart
        CUDA::cuda_driver
    )
    if(TARGET phantom)
        target_link_libraries(nexus_multi_gpu PUBLIC phantom)
    endif()
    if(MPI_FOUND)
        target_link_libraries(nexus_multi_gpu PUBLIC MPI::MPI_CXX)
        target_compile_definitions(nexus_multi_gpu PUBLIC USE_MPI)
    endif()
    set_target_properties(nexus_multi_gpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# ── Benchmarks ────────────────────────────────────────────────────────────────
if(NCCL_FOUND)
    foreach(bench nccl_bandwidth_test multi_gpu_keyswitch_test)
        if(EXISTS ${CMAKE_SOURCE_DIR}/src/benchmarks/${bench}.cu)
            add_executable(${bench} src/benchmarks/${bench}.cu)
            target_link_libraries(${bench} PRIVATE nexus_multi_gpu)
            # Link Phantom if the target was built
            if(TARGET phantom)
                target_link_libraries(${bench} PRIVATE phantom)
            endif()
            target_include_directories(${bench} PRIVATE
                ${PHANTOM_ROOT}/include
            )
            set_target_properties(${bench} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                CUDA_SEPARABLE_COMPILATION ON
            )
        endif()
    endforeach()
endif()

# ── Status summary ────────────────────────────────────────────────────────────
message(STATUS "=== NexusMultiGPU Build Config ===")
message(STATUS "  CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Build type         : ${CMAKE_BUILD_TYPE}")
message(STATUS "  NCCL found         : ${NCCL_FOUND}")
message(STATUS "  MPI found          : ${MPI_FOUND}")
if(EXISTS ${PHANTOM_ROOT}/CMakeLists.txt)
    message(STATUS "  Phantom submodule  : YES")
else()
    message(STATUS "  Phantom submodule  : NO")
endif()
