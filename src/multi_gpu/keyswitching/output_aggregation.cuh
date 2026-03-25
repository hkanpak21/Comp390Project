#pragma once
/**
 * output_aggregation.cuh
 *
 * Multi-GPU key-switching: Output Aggregation algorithm.
 *
 * Algorithm overview
 * ------------------
 * In single-GPU CKKS, the key-switching inner product over c2 digits and evk is:
 *   result = sum_{d=0}^{beta-1} c2_digit_d * evk_d
 *
 * In multi-GPU Output Aggregation:
 *   - c2 remains DISTRIBUTED: GPU g holds only its local limbs of c2.
 *   - Each GPU computes a PARTIAL inner product from its local limbs:
 *       partial_g = sum_{d where owner(d)==g} c2_digit_d * evk_d
 *   - AllReduce (sum): all GPUs contribute partial results so that every GPU
 *     gets the full sum:
 *       result = sum_g partial_g
 *   - Each GPU extracts its local portion of the result.
 *
 * Communication cost: one AllReduce of size (n_polys_out * total_limbs * degree * 8) bytes.
 * AllReduce on NVSwitch has ~2x lower bandwidth cost than AllGather for the same message size
 * because it's a reduce-scatter + allgather internally; for intra-node NVSwitch topologies
 * they are approximately equivalent, but AllReduce is preferred when GPU HBM is the bottleneck
 * (c2 never needs to be replicated fully on each GPU).
 *
 * Trade-off vs Input Broadcast
 * ----------------------------
 * Output Aggregation: lower peak memory (c2 stays distributed), one AllReduce.
 * Input Broadcast:    higher peak memory (full c2 replicated), one AllGather.
 * For NVSwitch-connected A100s, both are latency-dominated; benchmark both.
 */

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

class PhantomContext;
class PhantomCiphertext;
struct PhantomRelinKey;
struct PhantomGaloisKey;

namespace nexus_multi_gpu {

struct MultiGpuContext;

// ---------------------------------------------------------------------------
// Core algorithm
// ---------------------------------------------------------------------------

/**
 * keyswitching_output_aggregation
 *
 * Performs multi-GPU relinearization using the Output Aggregation strategy.
 * Called concurrently on all GPUs (SPMD).
 *
 * Postconditions:
 *   After the AllReduce, `local_out_c0_correction` and `local_out_c1_correction`
 *   on GPU g hold the key-switching corrections for all limbs owned by g.
 *   Add to local c0/c1 limbs to complete relinearization.
 *
 * @param ctx                      Multi-GPU NCCL context
 * @param phantom_ctx              Phantom FHE context
 * @param gpu_id                   Calling GPU index
 * @param chain_index              Current chain level
 * @param local_c2                 Local limbs of c2 [local_n_limbs * degree]
 * @param evk                      Full relinearization key (replicated on all GPUs)
 * @param local_out_c0_correction  Output: correction for c0 (all limbs after AllReduce)
 * @param local_out_c1_correction  Output: correction for c1 (all limbs after AllReduce)
 * @param total_limbs              Total RNS limbs at this chain level
 * @param degree                   poly_modulus_degree
 * @param n_gpus                   Total number of GPUs
 */
void keyswitching_output_aggregation(
    MultiGpuContext        &ctx,
    const PhantomContext   &phantom_ctx,
    int                     gpu_id,
    size_t                  chain_index,
    const uint64_t         *local_c2,
    const PhantomRelinKey  &evk,
    uint64_t               *local_out_c0_correction,
    uint64_t               *local_out_c1_correction,
    size_t                  total_limbs,
    size_t                  degree,
    int                     n_gpus
);

/**
 * rotation_output_aggregation
 *
 * Variant for Galois-key-based rotation.
 */
void rotation_output_aggregation(
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
 * partial_inner_product_local_limbs
 *
 * Computes the partial inner product using ONLY the local limbs owned by gpu_id.
 * This is a modified version of Phantom's key_switch_inner_prod that processes
 * only a subset of the decomposition digits (those whose source limbs are local).
 *
 * @param local_c2         Local limbs of c2 [local_n_limbs * degree]
 * @param evk_data         Full evk key data (replicated on each GPU)
 * @param partial_out_c0   Partial contribution to c0 correction [total_limbs * degree]
 * @param partial_out_c1   Partial contribution to c1 correction [total_limbs * degree]
 * @param modulus_ptr      DModulus array
 * @param beta             Total key decomposition count
 * @param local_digit_start First digit index owned by this GPU
 * @param local_digit_count Number of digits owned by this GPU
 * @param total_limbs      Total RNS limbs
 * @param degree           poly_modulus_degree
 * @param stream           CUDA stream
 */
void partial_inner_product_local_limbs(
    const uint64_t        *local_c2,
    const uint64_t *const *evk_data,
    uint64_t              *partial_out_c0,
    uint64_t              *partial_out_c1,
    const void            *modulus_ptr,
    size_t                 beta,
    size_t                 local_digit_start,
    size_t                 local_digit_count,
    size_t                 total_limbs,
    size_t                 degree,
    cudaStream_t           stream
);

/**
 * allreduce_keyswitching_result
 *
 * AllReduce (sum) the partial inner products across all GPUs.
 * After this call, every GPU has the complete key-switching result.
 *
 * @param ctx              NCCL context
 * @param gpu_id           Calling GPU
 * @param partial_out_c0   In/out: partial -> full result for c0
 * @param partial_out_c1   In/out: partial -> full result for c1
 * @param total_limbs      Total RNS limbs
 * @param degree           poly_modulus_degree
 */
void allreduce_keyswitching_result(
    MultiGpuContext &ctx,
    int              gpu_id,
    uint64_t        *partial_out_c0,
    uint64_t        *partial_out_c1,
    size_t           total_limbs,
    size_t           degree
);

} // namespace nexus_multi_gpu
