/**
 * galois_oa.cuh
 *
 * Distributed rotation (Galois automorphism) via Output Aggregation.
 *
 * Single-GPU rotation = apply_galois_permutation + keyswitch_inplace.
 * Distributed rotation splits the key-switching across GPUs:
 *   - GPU 0 applies the permutation (c1 → c2_gal, c0 → c0_gal)
 *   - All GPUs broadcast c2_gal and do partial KS on their digit shard
 *   - NCCL AllReduce sums partial corrections
 *   - Each GPU mod-downs and adds correction to c0_gal
 *
 * Memory benefit: each GPU holds 1/P of key digits →
 *   50 bootstrap keys at N=65536 fit on a single H100 with P=4.
 *
 * Compute benefit: KS inner product parallelized P×.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "context.cuh"
#include "ciphertext.h"
#include "dist_galois_key_store.cuh"
#include "../distributed_context.cuh"
#include "../comm/nccl_comm.cuh"

namespace nexus_multi_gpu {

/**
 * dist_rotate_output_aggregation
 *
 * Rotate a distributed ciphertext by `steps` slots using Output Aggregation DKS.
 *
 * @param ctx         DistributedContext (NCCL comms, per-GPU PhantomContexts)
 * @param dct         In/out: distributed ciphertext (scattered by RNS limbs)
 * @param steps       Number of slots to rotate (positive = left, negative = right)
 * @param key_store   Pre-loaded sharded Galois keys
 * @param key_idx     Index into key_store (which rotation key to use)
 */
void dist_rotate_output_aggregation(
    DistributedContext      &ctx,
    DistributedCiphertext   &dct,
    int                      steps,
    const DistGaloisKeyStore &key_store,
    size_t                   key_idx
);

/**
 * dist_relinearize_output_aggregation
 *
 * Relinearize a distributed ciphertext via Output Aggregation DKS.
 * Uses sharded relin key (pre-distributed across GPUs).
 *
 * @param ctx         DistributedContext
 * @param dct         In/out: ciphertext to relinearize (must have size() == 3)
 * @param relin_evks  Per-GPU sharded relin key evks pointer arrays [n_gpus]
 * @param beta        Number of KS digits
 */
void dist_relinearize_output_aggregation(
    DistributedContext    &ctx,
    DistributedCiphertext &dct,
    uint64_t             **relin_evks[],   // relin_evks[gpu_id] = device evks array for that GPU
    size_t                 beta
);

} // namespace nexus_multi_gpu
