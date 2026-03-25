#pragma once
/**
 * input_broadcast.cuh
 *
 * Multi-GPU key-switching: Input Broadcast algorithm.
 *
 * Algorithm overview
 * ------------------
 * In single-GPU CKKS, key-switching with c2 proceeds as:
 *   1. Decompose c2 into beta "digits" (each a sub-polynomial in a smaller basis).
 *   2. Inner product: sum_{d=0}^{beta-1} c2_digit_d * evk_d
 *   3. Basis extension: lift from P basis to Q basis.
 *
 * In multi-GPU Input Broadcast:
 *   - Initially, each GPU g holds only the local limbs of c2
 *     (those j where j % n_gpus == g).
 *   - Step 1: AllGather c2 — each GPU broadcasts its local limbs so that every
 *     GPU ends up with ALL limbs of c2.
 *   - Step 2: Each GPU performs the full inner product computation LOCALLY
 *     (no further communication needed — each GPU has the full c2 and full evk).
 *   - Step 3: Each GPU does basis extension locally.
 *   - Result: every GPU holds the complete key-switched output in its local memory.
 *     The caller can then extract the limbs it owns.
 *
 * Communication cost: one AllGather of size (n_polys_c2 * total_limbs * degree * 8) bytes.
 * On NVSwitch with 600 GB/s per GPU, a 20 MB ciphertext transfer takes ~33 us.
 *
 * Trade-off vs Output Aggregation
 * --------------------------------
 * Input Broadcast replicates c2 on all GPUs -> higher memory use but simpler computation.
 * Output Aggregation keeps c2 distributed -> lower memory but requires AllReduce on output.
 * Input Broadcast is preferred when GPU memory is not the limiting factor.
 */

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Forward declaration: Phantom types referenced by pointer only.
class PhantomContext;
class PhantomCiphertext;
struct PhantomRelinKey;
struct PhantomGaloisKey;

namespace nexus_multi_gpu {

struct MultiGpuContext;  // forward decl from nccl_comm.cuh

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/**
 * keyswitching_input_broadcast
 *
 * Performs multi-GPU relinearization (key-switching c2 -> correction for c0, c1)
 * using the Input Broadcast strategy.
 *
 * The function is called on ALL GPUs concurrently (SPMD style).
 * Each GPU passes its local ciphertext buffer and the shared key material.
 *
 * Preconditions:
 *   - `local_c2`  holds the limbs of c2 belonging to `gpu_id` (cyclic assignment).
 *   - `evk`       is the full relinearization key, replicated on all GPUs.
 *   - `ctx`       holds NCCL communicators + streams for all GPUs.
 *
 * Postconditions:
 *   - `local_out_c0_correction` and `local_out_c1_correction` hold the key-switched
 *     corrections for c0 and c1 (the limbs belonging to `gpu_id`).
 *   - Add these to the local c0/c1 limbs to complete relinearization.
 *
 * @param ctx                      Multi-GPU NCCL context
 * @param phantom_ctx              Phantom FHE context (for RNS tools, NTT tables)
 * @param gpu_id                   Calling GPU index (0-based)
 * @param chain_index              Current modulus chain level of the ciphertext
 * @param local_c2                 GPU buffer: local limbs of c2 [local_limbs * degree]
 * @param evk                      Full relinearization key (replicated on all GPUs)
 * @param local_out_c0_correction  Output: correction for c0 (local limbs)
 * @param local_out_c1_correction  Output: correction for c1 (local limbs)
 * @param n_polys_c2               Always 1 (c2 is a single polynomial)
 * @param total_limbs              Total RNS limbs at current chain level
 * @param degree                   poly_modulus_degree (N)
 * @param n_gpus                   Total number of GPUs
 */
void keyswitching_input_broadcast(
    MultiGpuContext        &ctx,
    const PhantomContext   &phantom_ctx,
    int                     gpu_id,
    size_t                  chain_index,
    const uint64_t         *local_c2,
    const PhantomRelinKey  &evk,
    uint64_t               *local_out_c0_correction,
    uint64_t               *local_out_c1_correction,
    size_t                  n_polys_c2,
    size_t                  total_limbs,
    size_t                  degree,
    int                     n_gpus
);

/**
 * rotation_input_broadcast
 *
 * Variant for Galois-key-based rotation (used in rotate_vector operations).
 * Identical algorithm to keyswitching_input_broadcast but uses a GaloisKey
 * instead of a RelinKey.
 *
 * @param galois_elt  The Galois element for the desired rotation step
 */
void rotation_input_broadcast(
    MultiGpuContext        &ctx,
    const PhantomContext   &phantom_ctx,
    int                     gpu_id,
    size_t                  chain_index,
    const uint64_t         *local_c1_rotated,
    const PhantomGaloisKey &gk,
    uint32_t                galois_elt,
    uint64_t               *local_out_c0_correction,
    uint64_t               *local_out_c1_new,
    size_t                  total_limbs,
    size_t                  degree,
    int                     n_gpus
);

// ---------------------------------------------------------------------------
// Internal helpers (exposed for testing)
// ---------------------------------------------------------------------------

/**
 * allgather_c2_limbs
 *
 * Step 1 of Input Broadcast: AllGather local c2 limbs so every GPU gets all limbs.
 *
 * @param ctx         NCCL context
 * @param gpu_id      Calling GPU
 * @param local_c2    Local limbs of c2: [local_n_limbs * degree]
 * @param full_c2     Output full c2 buffer: [total_limbs * degree]
 * @param local_limbs Number of limbs owned by this GPU
 * @param total_limbs Total limbs
 * @param degree      poly_modulus_degree
 */
void allgather_c2_limbs(
    MultiGpuContext &ctx,
    int              gpu_id,
    const uint64_t  *local_c2,
    uint64_t        *full_c2,
    size_t           local_limbs,
    size_t           total_limbs,
    size_t           degree
);

/**
 * local_inner_product
 *
 * Step 2 of Input Broadcast: given full c2 (all limbs) and the full evk, compute the
 * inner product locally. This is single-GPU Phantom's key_switch_inner_prod kernel.
 *
 * We call into Phantom's existing CUDA kernel directly. This function is a thin wrapper
 * that sets up the kernel launch parameters and calls the Phantom kernel.
 *
 * @param full_c2       Full c2 buffer [total_limbs * degree] in NTT form
 * @param evk_data      Raw evk key data pointer
 * @param out_c0        Output correction for c0 [total_limbs * degree]
 * @param out_c1        Output correction for c1 [total_limbs * degree]
 * @param modulus_ptr   Pointer to DModulus array for current chain level
 * @param beta          Key decomposition count
 * @param total_limbs   Total RNS limbs
 * @param degree        poly_modulus_degree
 * @param stream        CUDA stream
 */
void local_inner_product(
    const uint64_t        *full_c2,
    const uint64_t *const *evk_data,
    uint64_t              *out_c0,
    uint64_t              *out_c1,
    const void            *modulus_ptr,
    size_t                 beta,
    size_t                 size_QP,
    size_t                 total_limbs,
    size_t                 degree,
    cudaStream_t           stream
);

} // namespace nexus_multi_gpu
