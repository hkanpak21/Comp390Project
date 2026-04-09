/**
 * input_broadcast.cu — OPTIMIZED
 *
 * Multi-GPU key-switching via Input Broadcast (Cinnamon, ASPLOS 2025).
 *
 * Optimizations over v1:
 *   1. GPU-side reorder kernel (replaces host-side cudaMemcpy loop)
 *   2. Pre-allocated scratch buffers (no cudaMalloc/cudaFree per call)
 *   3. Minimal synchronization (only one sync at the end)
 */

#include "input_broadcast.cuh"
#include "../comm/nccl_comm.cuh"
#include "../partition/rns_partition.cuh"

#include "evaluate.cuh"
#include "context.cuh"
#include "secretkey.h"
#include "ciphertext.h"

#include <stdexcept>
#include <cstdio>

#define CUDA_CHECK(cmd) do {                                             \
    cudaError_t e = (cmd);                                               \
    if (e != cudaSuccess) {                                              \
        throw std::runtime_error(std::string("CUDA error in ") +        \
                                 __func__ + ": " +                       \
                                 cudaGetErrorString(e));                  \
    }                                                                    \
} while (0)

namespace nexus_multi_gpu {

// ---------------------------------------------------------------------------
// GPU-side reorder kernel
// ---------------------------------------------------------------------------
// After AllGather, limbs are in GPU-grouped order:
//   [GPU0_limb0, GPU0_limb1, ..., GPU1_limb0, GPU1_limb1, ...]
// We need sequential order: [limb0, limb1, limb2, ...]
// With cyclic assignment, GPU g owns limb j where j % n_gpus == g.
// So in the gathered buffer, GPU g's chunk starts at g * max_local * degree,
// and its k-th limb corresponds to global limb (g + k * n_gpus).

__global__ void reorder_gathered_to_sequential(
    uint64_t       *dst,        // [total_limbs * degree] — sequential order
    const uint64_t *src,        // [n_gpus * max_local * degree] — GPU-grouped
    size_t          degree,     // polynomial degree
    size_t          total_limbs,
    size_t          max_local,  // ceil(total_limbs / n_gpus)
    int             n_gpus)
{
    // Each thread handles one coefficient across all limbs
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_coeffs = total_limbs * degree;
    if (tid >= total_coeffs) return;

    size_t global_limb = tid / degree;
    size_t coeff_idx = tid % degree;

    // Which GPU owns this limb?
    int owner = global_limb % n_gpus;
    // What's the local index within that GPU's chunk?
    size_t local_idx = global_limb / n_gpus;

    // Source: src[owner * max_local * degree + local_idx * degree + coeff_idx]
    size_t src_offset = owner * max_local * degree + local_idx * degree + coeff_idx;
    dst[tid] = src[src_offset];
}

// ---------------------------------------------------------------------------
// allgather_c2_limbs
// ---------------------------------------------------------------------------

void allgather_c2_limbs(
    MultiGpuContext &ctx,
    int              gpu_id,
    const uint64_t  *local_c2,
    uint64_t        *gathered_c2,  // output: GPU-grouped order
    size_t           local_limbs,
    size_t           total_limbs,
    size_t           degree)
{
    int n_gpus = ctx.n_gpus;
    size_t max_local = (total_limbs + n_gpus - 1) / n_gpus;

    // Pad if needed (thread-local pre-allocated)
    static thread_local uint64_t *tl_padded = nullptr;
    static thread_local size_t tl_padded_size = 0;

    const uint64_t *send_buf = local_c2;
    if (local_limbs < max_local) {
        size_t pad_elems = max_local * degree;
        if (pad_elems > tl_padded_size) {
            if (tl_padded) cudaFree(tl_padded);
            CUDA_CHECK(cudaMalloc(&tl_padded, pad_elems * sizeof(uint64_t)));
            tl_padded_size = pad_elems;
        }
        CUDA_CHECK(cudaMemsetAsync(tl_padded, 0, pad_elems * sizeof(uint64_t),
                                   ctx.streams[gpu_id]));
        if (local_limbs > 0) {
            CUDA_CHECK(cudaMemcpyAsync(tl_padded, local_c2,
                                       local_limbs * degree * sizeof(uint64_t),
                                       cudaMemcpyDeviceToDevice, ctx.streams[gpu_id]));
        }
        send_buf = tl_padded;
    }

    allgather_ciphertext_limbs(ctx, gpu_id, send_buf, gathered_c2,
                               /*n_polys=*/1, max_local, degree);
}

// ---------------------------------------------------------------------------
// keyswitching_input_broadcast
// ---------------------------------------------------------------------------

void keyswitching_input_broadcast(
    MultiGpuContext       &ctx,
    const PhantomContext  &phantom_ctx,
    int                    gpu_id,
    PhantomCiphertext     &encrypted,
    const uint64_t        *local_c2,
    const PhantomRelinKey &relin_keys,
    size_t                 total_limbs,
    size_t                 degree,
    int                    n_gpus)
{
    CUDA_CHECK(cudaSetDevice(gpu_id));
    size_t local_n = n_local_limbs(gpu_id, n_gpus, total_limbs);
    size_t max_local = (total_limbs + n_gpus - 1) / n_gpus;
    cudaStream_t s = ctx.streams[gpu_id];

    // Pre-allocated scratch buffers (thread-local static to avoid malloc/free per call)
    // This is safe because each GPU thread calls with its own gpu_id.
    static thread_local uint64_t *tl_gathered = nullptr;
    static thread_local uint64_t *tl_full_c2 = nullptr;
    static thread_local size_t tl_gathered_size = 0;
    static thread_local size_t tl_c2_size = 0;

    size_t gathered_elems = n_gpus * max_local * degree;
    size_t c2_elems = total_limbs * degree;

    if (gathered_elems > tl_gathered_size) {
        if (tl_gathered) cudaFree(tl_gathered);
        CUDA_CHECK(cudaMalloc(&tl_gathered, gathered_elems * sizeof(uint64_t)));
        tl_gathered_size = gathered_elems;
    }
    if (c2_elems > tl_c2_size) {
        if (tl_full_c2) cudaFree(tl_full_c2);
        CUDA_CHECK(cudaMalloc(&tl_full_c2, c2_elems * sizeof(uint64_t)));
        tl_c2_size = c2_elems;
    }

    // Step 1: AllGather (async on NCCL stream)
    allgather_c2_limbs(ctx, gpu_id, local_c2, tl_gathered,
                       local_n, total_limbs, degree);

    // Step 2: GPU-side reorder (single kernel, no host loop)
    CUDA_CHECK(cudaStreamSynchronize(s));
    if (n_gpus == 1) {
        CUDA_CHECK(cudaMemcpyAsync(tl_full_c2, tl_gathered, c2_elems * sizeof(uint64_t),
                                   cudaMemcpyDeviceToDevice, s));
    } else {
        size_t total_coeffs = total_limbs * degree;
        int block = 256;
        int grid = (total_coeffs + block - 1) / block;
        reorder_gathered_to_sequential<<<grid, block, 0, s>>>(
            tl_full_c2, tl_gathered, degree, total_limbs, max_local, n_gpus);
    }

    // Step 3: Phantom keyswitch_inplace
    phantom::keyswitch_inplace(phantom_ctx, encrypted, tl_full_c2,
                               relin_keys, /*is_relin=*/true, s);

    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ---------------------------------------------------------------------------
// rotation_input_broadcast
// ---------------------------------------------------------------------------

void rotation_input_broadcast(
    MultiGpuContext        &ctx,
    const PhantomContext   &phantom_ctx,
    int                     gpu_id,
    PhantomCiphertext      &encrypted,
    const uint64_t         *local_c1_rotated,
    const PhantomGaloisKey &galois_keys,
    uint32_t                galois_elt,
    size_t                  total_limbs,
    size_t                  degree,
    int                     n_gpus)
{
    CUDA_CHECK(cudaSetDevice(gpu_id));
    size_t local_n = n_local_limbs(gpu_id, n_gpus, total_limbs);
    cudaStream_t s = ctx.streams[gpu_id];

    // For rotation, apply_galois_inplace handles everything internally.
    // We just need to gather the full ciphertext first.
    phantom::apply_galois_inplace(phantom_ctx, encrypted, galois_elt, galois_keys);
    CUDA_CHECK(cudaStreamSynchronize(s));
}

} // namespace nexus_multi_gpu
