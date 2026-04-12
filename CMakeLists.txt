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
# Phantom's CMakeLists uses CMAKE_SOURCE_DIR for include paths, which breaks
# when built as a subdirectory. We build it via ExternalProject instead.
set(PHANTOM_ROOT ${CMAKE_SOURCE_DIR}/vendor/phantom)

include(ExternalProject)
ExternalProject_Add(phantom_ext
    SOURCE_DIR ${PHANTOM_ROOT}
    BINARY_DIR ${CMAKE_BINARY_DIR}/phantom_build
    CMAKE_ARGS
        -DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DPHANTOM_ENABLE_EXAMPLE=OFF
        -DPHANTOM_ENABLE_BENCH=OFF
        -DPHANTOM_ENABLE_TEST=OFF
    INSTALL_COMMAND ""
    BUILD_BYPRODUCTS ${CMAKE_BINARY_DIR}/phantom_build/lib/libPhantom.so
)

# Create an imported target for Phantom
add_library(Phantom SHARED IMPORTED)
set_target_properties(Phantom PROPERTIES
    IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/phantom_build/lib/libPhantom.so
    INTERFACE_INCLUDE_DIRECTORIES ${PHANTOM_ROOT}/include
)
add_dependencies(Phantom phantom_ext)

# ── NEXUS evaluator library (ported BERT ops + bootstrapping) ────────────────
file(GLOB NEXUS_EVAL_SRCS src/nexus_eval/*.cu)
file(GLOB NEXUS_BOOT_SRCS src/nexus_eval/bootstrapping/*.cu
                           src/nexus_eval/bootstrapping/common/*.cu)
add_library(nexus_evaluators STATIC ${NEXUS_EVAL_SRCS} ${NEXUS_BOOT_SRCS})
target_include_directories(nexus_evaluators PUBLIC
    src/nexus_eval
    src/nexus_eval/bootstrapping
    src/nexus_eval/bootstrapping/common
    ${PHANTOM_ROOT}/include
    /gpfs/projects/etur02/hkanpak/local/include
    /usr/local/include
)
target_link_libraries(nexus_evaluators PUBLIC Phantom CUDA::cudart)

# NTL/GMP for bootstrapping polynomial generation (Remez approximation)
set(NTL_INCLUDE_DIR "/gpfs/projects/etur02/hkanpak/local/include" CACHE PATH "NTL include dir")
find_library(NTL_LIBRARY ntl HINTS /gpfs/projects/etur02/hkanpak/local/lib /usr/local/lib)
find_library(GMP_LIBRARY gmp HINTS /usr/lib64 /usr/local/lib)
if(NTL_LIBRARY)
    target_link_libraries(nexus_evaluators PUBLIC ${NTL_LIBRARY} m pthread)
    target_include_directories(nexus_evaluators SYSTEM PUBLIC ${NTL_INCLUDE_DIR})
    target_compile_options(nexus_evaluators PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-I${NTL_INCLUDE_DIR}>)
    message(STATUS "NTL found: ${NTL_LIBRARY}, headers: ${NTL_INCLUDE_DIR}")
    if(GMP_LIBRARY)
        target_link_libraries(nexus_evaluators PUBLIC ${GMP_LIBRARY})
        message(STATUS "GMP found: ${GMP_LIBRARY}")
    endif()
else()
    message(WARNING "NTL not found — bootstrapping will not link")
endif()

set_target_properties(nexus_evaluators PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(nexus_evaluators phantom_ext)

# ── src/multi_gpu library ─────────────────────────────────────────────────────
if(NCCL_FOUND)
    file(GLOB_RECURSE MULTI_GPU_SRCS
        src/multi_gpu/*.cu
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
    if(TARGET Phantom)
        target_link_libraries(nexus_multi_gpu PUBLIC Phantom)
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
    foreach(bench nccl_bandwidth_test multi_gpu_keyswitch_test dist_bert_layer_bench spmd_keyswitch_bench spmd_oa_bench ks_breakdown_bench bert_layer_scaling ct_pipeline_bench multi_node_bench bert_matmul_pipeline bert_ops_test bert_matmul_real bert_e2e_inference bert_e2e_multigpu bert_multinode bert_multinode_e2e bert_connected_pipeline bert_connected_multigpu bert_connected_multinode bootstrap_test bootstrap_test_n65536 bootstrap_distkey_test bootstrap_n65536_streaming rotate_stream_test bert_encoder_layer bert_encoder_multigpu bert_encoder_multinode)
        if(EXISTS ${CMAKE_SOURCE_DIR}/src/benchmarks/${bench}.cu)
            add_executable(${bench} src/benchmarks/${bench}.cu)
            target_link_libraries(${bench} PRIVATE nexus_multi_gpu)
            # Link Phantom if the target was built
            if(TARGET Phantom)
                target_link_libraries(${bench} PRIVATE Phantom)
            endif()
            # Link NEXUS evaluators for BERT benchmarks
            if(TARGET nexus_evaluators)
                target_link_libraries(${bench} PRIVATE nexus_evaluators)
            endif()
            target_include_directories(${bench} PRIVATE
                ${PHANTOM_ROOT}/include
                src/nexus_eval
                src/nexus_eval/bootstrapping
                src/nexus_eval/bootstrapping/common
                /gpfs/projects/etur02/hkanpak/local/include
            )
            set_target_properties(${bench} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
                CUDA_SEPARABLE_COMPILATION ON
            )
            # NTL include for CUDA compilation (needed by bootstrapping headers)
            if(NTL_LIBRARY)
                target_compile_options(${bench} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-I/gpfs/projects/etur02/hkanpak/local/include>)
            endif()
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
