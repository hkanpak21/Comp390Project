#pragma once
/**
 * output_aggregation.cuh
 *
 * Multi-GPU key-switching: Output Aggregation algorithm (Cinnamon, ASPLOS 2025).
 *
 * Algorithm overview
 * ------------------
 * Single-GPU key-switching inner product:
 *   result = sum_{d=0}^{beta-1} modup(c2, d) * evk[d]
 *
 * Output Aggregation distributes this across GPUs:
 *   1. Each GPU g processes digits d where d % n_gpus == g:
 *        partial_g = sum_{d: d%n==g} modup(c2_local, d) * evk[d]
 *   2. ReduceScatter: partial sums across GPUs → each GPU gets its local slice.
 *      (Or AllReduce if we want every GPU to have the full result.)
 *
 * Communication: one AllReduce of the inner product output (2 * size_QlP * N bytes).
 * c2 is NEVER fully replicated — lower peak memory than Input Broadcast.
 *
 * Implementation note
 * -------------------
 * This requires a CUSTOM partial inner product kernel because Phantom's
 * key_switch_inner_prod_c2_and_evk processes all beta digits. We modify the
 * kernel to iterate only over the local digit range [d_start, d_start + d_count).
 * After AllReduce, mod-down can proceed locally on each GPU.
 */

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"

namespace nexus_multi_gpu {

struct MultiGpuContext;

// ---------------------------------------------------------------------------
// Core API
// ---------------------------------------------------------------------------

/**
 * keyswitching_output_aggregation
 *
 * Multi-GPU relinearization using Output Aggregation.
 *
 * Pipeline:
 *   1. Each GPU does mod-up on its local digits of c2
 *   2. Each GPU computes partial inner product (local digits only)
 *   3. AllReduce partial inner products across GPUs
 *   4. Each GPU does mod-down on the combined result
 *   5. Add correction to ciphertext
 *
 * @param ctx           Multi-GPU NCCL context
 * @param phantom_ctx   Phantom FHE context
 * @param gpu_id        Calling GPU (0-based)
 * @param encrypted     Ciphertext to relinearize (modified in-place)
 * @param c2            Full c2 polynomial (all limbs, on this GPU)
 *                      NOTE: unlike Input Broadcast, Output Aggregation
 *                      requires c2 to be available for mod-up. In practice,
 *                      c2 comes from the ciphertext multiplication and is
 *                      already present on each GPU before key-switching.
 * @param relin_keys    Relinearization key (replicated on all GPUs)
 * @param n_gpus        Number of GPUs
 */
void keyswitching_output_aggregation(
    MultiGpuContext       &ctx,
    const PhantomContext  &phantom_ctx,
    int                    gpu_id,
    PhantomCiphertext     &encrypted,
    uint64_t              *c2,
    const PhantomRelinKey &relin_keys,
    int                    n_gpus
);

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * allreduce_keyswitching_result — AllReduce partial inner products across GPUs.
 * In-place sum of uint64_t values. Caller must apply modular reduction afterward.
 */
void allreduce_keyswitching_result(
    MultiGpuContext &ctx,
    int              gpu_id,
    uint64_t        *partial_cx,
    size_t           count
);

} // namespace nexus_multi_gpu
