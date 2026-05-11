cmake_minimum_required(VERSION 3.20)

# MN5 patch: do NOT hardcode CUDA 11 — use whatever nvcc is in PATH (cuda/12.8 module).
# Original line was:
#   if (UNIX) set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11/bin/nvcc") endif()

project(NEXUS-CUDA LANGUAGES CXX CUDA)

# CMAKE config
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

# Debugging config
option(SHOW_WARNINGS "Enable warnings" OFF)
if (SHOW_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-comment")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra,-Wpedantic,-Wno-comment")
endif()

# Language config
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA config — MN5 patch: default to H100 (sm_90) instead of "native" auto-detect,
# because the CMake configure step runs on a login node without GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 --default-stream per-thread")

# Print CUDA info
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")

# PhantomFHE config
option(PHANTOM_USE_CUDA_PTX "Use CUDA PTX Assembly" ON)
message(STATUS "Use CUDA PTX Assembly: ${PHANTOM_USE_CUDA_PTX}")
if (PHANTOM_USE_CUDA_PTX)
    add_compile_definitions(PHANTOM_USE_CUDA_PTX)
endif()

# PhantomFHE
include_directories(thirdparty/phantom-fhe/include)

# MN5 patch: NTL/GMP — NTL is in /gpfs/projects/etur02/hkanpak/local, GMP is system /usr/lib64.
# Allow override with -DNTL_ROOT=...
if(NOT DEFINED NTL_ROOT)
  set(NTL_ROOT "/gpfs/projects/etur02/hkanpak/local")
endif()
# IMPORTANT: must use add_compile_options, not include_directories, because CPATH
# already contains ${NTL_ROOT}/include (set by build script) and CMake silently
# filters that path as an "implicit include", breaking #include <NTL/mat_RR.h>.
add_compile_options(-I${NTL_ROOT}/include)
link_directories(${NTL_ROOT}/lib)

# NexusCUDA
file(GLOB_RECURSE NEXUS_CUDA_SOURCE_FILES CONFIGURE_DEPENDS
  "src/main.cu"
  "src/gelu.cu"
  "src/softmax.cu"
  "src/layer_norm.cu"
  "src/matrix_mul.cu"
  "src/matrix_mul_p.cu"
  "src/ckks_evaluator.cu"
  "src/argmax.cu"
  "src/bootstrapping/*.cu"
  "thirdparty/phantom-fhe/src/*.cu"
)
add_executable(main ${NEXUS_CUDA_SOURCE_FILES})
target_link_libraries(main PRIVATE ntl gmp m pthread)
target_include_directories(main PRIVATE
  ${NTL_ROOT}/include
  ${NTL_ROOT}/include/NTL
  src/bootstrapping/commmon
  src/bootstrapping
)

# NexusCUDA Boostrapping
file(GLOB_RECURSE NEXUS_CUDA_BOOTSTRAPING_SOURCE_FILES CONFIGURE_DEPENDS
  "src/bootstrapping.cu"
  "src/bootstrapping/*.cu"
  "src/ckks_evaluator.cu"
  "thirdparty/phantom-fhe/src/*.cu"
)

add_executable(bootstrapping ${NEXUS_CUDA_BOOTSTRAPING_SOURCE_FILES})
target_link_libraries(bootstrapping PRIVATE ntl gmp m pthread)
target_include_directories(bootstrapping PRIVATE
  ${NTL_ROOT}/include
  ${NTL_ROOT}/include/NTL
  src/bootstrapping/commmon
  src/bootstrapping
)
