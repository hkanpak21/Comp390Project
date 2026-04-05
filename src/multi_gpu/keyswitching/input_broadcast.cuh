#pragma once
/**
 * input_broadcast.cuh
 *
 * Multi-GPU key-switching: Input Broadcast algorithm.
 *
 * Algorithm overview (Cinnamon, ASPLOS 2025)
 * ------------------------------------------
 * In single-GPU CKKS, key-switching for c2 proceeds as:
 *   1. mod-up: decompose c2 into beta digits, extend each to basis QlP
 *   2. inner-product: sum_{d=0}^{beta-1} digit_d * evk_d
 *   3. mod-down: convert the result back from QlP basis to Ql basis
 *
 * In multi-GPU Input Broadcast:
 *   - Initially, each GPU g holds only local limbs of c2 (j where j % n == g).
 *   - Step 1: AllGather c2 — every GPU gets ALL limbs.
 *   - Step 2: Each GPU runs the FULL Phantom keyswitch_inplace locally
 *     (mod-up → inner product → mod-down). No further communication.
 *   - Step 3: Extract the local-limb slice of the result.
 *
 * Communication cost: one AllGather of |c2| bytes per key-switch.
 *
 * Trade-off vs Output Aggregation
 * --------------------------------
 * Input Broadcast replicates c2 on all GPUs -> higher memory, simpler code.
 * Output Aggregation keeps c2 distributed -> needs AllReduce on output.
 */

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

// Phantom types
#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"

namespace nexus_multi_gpu {

struct MultiGpuContext;  // forward decl from nccl_comm.cuh

// ---------------------------------------------------------------------------
// Core API
// ---------------------------------------------------------------------------

/**
 * keyswitching_input_broadcast
 *
 * Multi-GPU relinearization using Input Broadcast.
 * Called on ALL GPUs concurrently (SPMD style).
 *
 * This function wraps Phantom's full keyswitch_inplace pipeline:
 *   1. AllGather local c2 → full c2 on every GPU
 *   2. Phantom keyswitch_inplace(context, encrypted, full_c2, relin_key)
 *      which internally does: mod-up → inner product → mod-down → add to ct
 *
 * @param ctx            Multi-GPU NCCL context
 * @param phantom_ctx    Phantom FHE context (RNS tools, NTT tables, modulus chain)
 * @param gpu_id         Calling GPU index (0-based)
 * @param encrypted      Ciphertext to relinearize (modified in-place on return)
 *                       Must contain local limbs only on entry; full result on exit.
 * @param local_c2       GPU buffer: local limbs of c2 [local_limbs * degree]
 * @param relin_keys     Relinearization key (replicated on all GPUs)
 * @param total_limbs    Total RNS limbs at current level (size_Ql for CKKS)
 * @param degree         poly_modulus_degree (N = 65536 for NEXUS)
 * @param n_gpus         Number of GPUs
 */
void keyswitching_input_broadcast(
    MultiGpuContext       &ctx,
    const PhantomContext  &phantom_ctx,
    int                    gpu_id,
    PhantomCiphertext     &encrypted,
    const uint64_t        *local_c2,
    const PhantomRelinKey &relin_keys,
    size_t                 total_limbs,
    size_t                 degree,
    int                    n_gpus
);

/**
 * rotation_input_broadcast
 *
 * Multi-GPU Galois rotation using Input Broadcast.
 * Identical pattern: AllGather rotated c1, then local keyswitch with GaloisKey.
 */
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
    int                     n_gpus
);

// ---------------------------------------------------------------------------
// Internal helpers (exposed for unit testing)
// ---------------------------------------------------------------------------

/**
 * allgather_c2_limbs — Step 1: collect all limbs of c2 on every GPU.
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

} // namespace nexus_multi_gpu
