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
    size_t local_n = n_local_limbs(gpu_id, n_gpus, total_limbs);

    // ---- Step 1: AllGather c2 — every GPU gets all limbs ----
    uint64_t *full_c2 = nullptr;
    size_t c2_bytes = total_limbs * degree * sizeof(uint64_t);
    CUDA_CHECK(cudaMalloc(&full_c2, c2_bytes));

    allgather_c2_limbs(ctx, gpu_id,
                       local_c2, full_c2,
                       local_n, total_limbs, degree);

    // ---- Step 2: Call Phantom's keyswitch_inplace locally ----
    // This does the full pipeline: mod-up → inner product → mod-down → add to ct.
    // Each GPU has the full c2 and the full relin key, so this is a standard
    // single-GPU operation — no further communication needed.
    cudaStream_t compute_stream = ctx.streams[gpu_id];

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

    // Step 2: Galois key-switch locally
    // GaloisKey stores a vector of RelinKeys, indexed by galois_elt mapping.
    // Phantom's apply_galois_inplace handles the full pipeline.
    cudaStream_t compute_stream = ctx.streams[gpu_id];

    // For rotation, keyswitch_inplace is called with the galois relin key.
    // The galois key for a given element maps to a specific relin key index.
    // Phantom's rotate_internal uses: galois_keys.get_relin_keys(galois_elt_idx)
    // Here we call keyswitch_inplace directly with the rotated c1.
    const auto &rk = galois_keys.get_relin_keys(
        phantom_ctx.get_context_data(0).parms().galois_tool().get_elt_from_step(galois_elt));

    phantom::keyswitch_inplace(phantom_ctx, encrypted, full_c1r,
                               rk, /*is_relin=*/false,
                               compute_stream);

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    // Cleanup
    CUDA_CHECK(cudaFree(full_c1r));
}

} // namespace nexus_multi_gpu
