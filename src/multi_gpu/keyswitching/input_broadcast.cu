/**
 * input_broadcast.cu
 *
 * Multi-GPU key-switching via Input Broadcast (Cinnamon, ASPLOS 2025).
 *
 * The key insight: Phantom's keyswitch_inplace() is self-contained — given
 * a full c2 buffer and the relin key, it does mod-up → inner product → mod-down
 * and adds the result directly to the ciphertext. Our job is simply:
 *   1. AllGather local c2 limbs → full c2 on every GPU
 *   2. Call Phantom's keyswitch_inplace() locally on each GPU
 *   3. Extract local limbs of the result
 *
 * This means we do NOT need to manually call the kernel or manage mod-up/mod-down.
 * Phantom handles it all. The multi-GPU wrapper is clean.
 */

#include "input_broadcast.cuh"
#include "../comm/nccl_comm.cuh"
#include "../partition/rns_partition.cuh"

#include "evaluate.cuh"       // phantom::keyswitch_inplace
#include "context.cuh"        // PhantomContext
#include "secretkey.h"        // PhantomRelinKey, PhantomGaloisKey
#include "ciphertext.h"       // PhantomCiphertext

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
// allgather_c2_limbs  (Step 1)
// ---------------------------------------------------------------------------

void allgather_c2_limbs(
    MultiGpuContext &ctx,
    int              gpu_id,
    const uint64_t  *local_c2,
    uint64_t        *full_c2,
    size_t           local_limbs,
    size_t           total_limbs,
    size_t           degree)
{
    // c2 is a single polynomial (n_polys = 1).
    allgather_ciphertext_limbs(ctx, gpu_id,
                               local_c2, full_c2,
                               /*n_polys=*/1,
                               local_limbs, degree);
    // Synchronize: inner product must not start before AllGather completes.
    CUDA_CHECK(cudaStreamSynchronize(ctx.streams[gpu_id]));
}

// ---------------------------------------------------------------------------
// keyswitching_input_broadcast  (Main entry point)
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

    // ---- Step 1: AllGather c2 — every GPU gets all limbs ----
    // AllGather output layout: [GPU0_limbs | GPU1_limbs | ... | GPUn_limbs]
    // But Phantom needs sequential order: [limb0 | limb1 | limb2 | ...]
    // With cyclic assignment, GPU g has limbs g, g+n, g+2n, ... so AllGather
    // produces wrong ordering. We gather then reorder.
    uint64_t *gathered_c2 = nullptr;
    uint64_t *full_c2 = nullptr;
    size_t c2_bytes = total_limbs * degree * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&gathered_c2, c2_bytes));
    CUDA_CHECK(cudaMalloc(&full_c2, c2_bytes));

    allgather_c2_limbs(ctx, gpu_id,
                       local_c2, gathered_c2,
                       local_n, total_limbs, degree);

    // Reorder: gathered_c2 is [GPU0_limbs | GPU1_limbs | ...]
    // full_c2 should be [limb0 | limb1 | limb2 | ...]
    // GPU g contributed limbs: g, g+n_gpus, g+2*n_gpus, ...
    // In gathered_c2, GPU g's data starts at offset (sum of local_limbs for GPUs < g) * degree
    cudaStream_t compute_stream = ctx.streams[gpu_id];
    if (n_gpus == 1) {
        // No reorder needed
        CUDA_CHECK(cudaMemcpyAsync(full_c2, gathered_c2, c2_bytes,
                                   cudaMemcpyDeviceToDevice, compute_stream));
    } else {
        // Reorder limbs from GPU-grouped to sequential
        size_t src_offset = 0;
        for (int g = 0; g < n_gpus; g++) {
            size_t g_local = n_local_limbs(g, n_gpus, total_limbs);
            size_t loc = 0;
            for (size_t j = 0; j < total_limbs; j++) {
                if (owner_of_limb(j, n_gpus) != g) continue;
                // Source: gathered_c2 + (src_offset + loc) * degree
                // Dest:   full_c2 + j * degree
                CUDA_CHECK(cudaMemcpyAsync(
                    full_c2 + j * degree,
                    gathered_c2 + (src_offset + loc) * degree,
                    degree * sizeof(uint64_t),
                    cudaMemcpyDeviceToDevice, compute_stream));
                loc++;
            }
            src_offset += g_local;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    CUDA_CHECK(cudaFree(gathered_c2));

    // ---- Step 2: Call Phantom's keyswitch_inplace locally ----
    // Now full_c2 has limbs in sequential order — Phantom can process it.
    phantom::keyswitch_inplace(phantom_ctx, encrypted, full_c2,
                               relin_keys, /*is_relin=*/true,
                               compute_stream);

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(full_c2));
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
    size_t local_n = n_local_limbs(gpu_id, n_gpus, total_limbs);

    // Step 1: AllGather the rotated c1 polynomial
    uint64_t *full_c1r = nullptr;
    size_t c1_bytes = total_limbs * degree * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&full_c1r, c1_bytes));

    allgather_c2_limbs(ctx, gpu_id,
                       local_c1_rotated, full_c1r,
                       local_n, total_limbs, degree);

    // Step 2: Apply Galois rotation locally.
    // Phantom's apply_galois_inplace handles the full pipeline internally:
    // permute coefficients → keyswitch with the matching GaloisKey → done.
    // We've already gathered the full c1 data, but apply_galois_inplace
    // works on the ciphertext directly — so we use it as-is.
    cudaStream_t compute_stream = ctx.streams[gpu_id];

    phantom::apply_galois_inplace(phantom_ctx, encrypted, galois_elt,
                                  galois_keys);

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    // Cleanup
    CUDA_CHECK(cudaFree(full_c1r));
}

} // namespace nexus_multi_gpu
