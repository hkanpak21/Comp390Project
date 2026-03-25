#pragma once
/**
 * nccl_comm.cuh
 *
 * NCCL collective wrappers for distributing RNS-CKKS ciphertext limbs across GPUs.
 *
 * Design notes
 * ------------
 * A PhantomCiphertext stores `size` polynomials, each split into `coeff_modulus_size`
 * RNS limbs. Each limb is a flat array of `poly_modulus_degree` uint64_t values.
 *
 * For multi-GPU execution we assign limb j to GPU (j % n_gpus). The functions below
 * orchestrate the NCCL collectives required by the two key-switching algorithms:
 *
 *   Input Broadcast  : AllGather -- each GPU sends its limbs, receives all limbs.
 *   Output Aggregation: AllReduce -- each GPU computes partial inner product,
 *                        then all partial results are summed across GPUs.
 *
 * All functions are non-blocking (use the provided NCCL communicator + CUDA stream).
 * The caller is responsible for stream synchronization.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

namespace nexus_multi_gpu {

// ---------------------------------------------------------------------------
// Communicator context
// ---------------------------------------------------------------------------

/**
 * MultiGpuContext
 *
 * Holds one ncclComm_t per GPU plus the associated CUDA streams and device IDs.
 * Construct once at program start, pass by pointer to all multi-GPU calls.
 */
struct MultiGpuContext {
    int          n_gpus;                   ///< total number of GPUs in use
    std::vector<int>          device_ids;  ///< CUDA device IDs (e.g. {0,1,2,3})
    std::vector<ncclComm_t>   comms;       ///< one NCCL communicator per GPU
    std::vector<cudaStream_t> streams;     ///< one compute+NCCL stream per GPU

    /// Initialize NCCL communicators for the given device list.
    /// Must be called from a single host thread.
    static MultiGpuContext create(const std::vector<int> &dev_ids);

    /// Cleanly destroy all NCCL communicators and CUDA streams.
    void destroy();
};

// ---------------------------------------------------------------------------
// Limb layout helpers
// ---------------------------------------------------------------------------

/**
 * LimbRange
 *
 * Describes the contiguous slice of a ciphertext buffer that belongs to one GPU.
 * The limbs owned by GPU g are {g, g+n, g+2n, ...} (cyclic assignment).
 *
 * This helper is consumed by scatter / gather functions below.
 */
struct LimbRange {
    size_t start_limb;   ///< index of first limb owned by this GPU
    size_t n_local;      ///< number of limbs owned by this GPU
};

/// Returns the LimbRange for the given gpu_id.
LimbRange get_limb_range(int gpu_id, int n_gpus, size_t total_limbs);

// ---------------------------------------------------------------------------
// Point-to-point limb transfers (used by scatter/gather)
// ---------------------------------------------------------------------------

/**
 * scatter_limbs_to_gpu
 *
 * Copies the limbs belonging to `dst_gpu` from a full ciphertext buffer
 * (on `src_gpu`) into `dst_buf` on `dst_gpu`.
 *
 * @param src_gpu      Source GPU device ID (holds the full ciphertext)
 * @param dst_gpu      Destination GPU device ID
 * @param src_data     GPU pointer on src_gpu: [n_polys][total_limbs][degree]
 * @param dst_data     GPU pointer on dst_gpu: [n_polys][local_limbs][degree]
 * @param n_polys      Number of polynomials in the ciphertext (typically 2)
 * @param total_limbs  Total number of RNS limbs
 * @param degree       poly_modulus_degree (N)
 * @param n_gpus       Total number of GPUs
 * @param stream       CUDA stream on dst_gpu
 */
void scatter_limbs_to_gpu(int src_gpu, int dst_gpu,
                          const uint64_t *src_data,
                          uint64_t       *dst_data,
                          size_t n_polys,
                          size_t total_limbs,
                          size_t degree,
                          int n_gpus,
                          cudaStream_t stream);

/**
 * gather_limbs_from_gpu
 *
 * Copies local limbs from `src_gpu` back into the correct positions of the
 * full ciphertext buffer on `dst_gpu`.
 */
void gather_limbs_from_gpu(int src_gpu, int dst_gpu,
                           const uint64_t *src_data,
                           uint64_t       *dst_data,
                           size_t n_polys,
                           size_t total_limbs,
                           size_t degree,
                           int n_gpus,
                           cudaStream_t stream);

// ---------------------------------------------------------------------------
// Collective operations
// ---------------------------------------------------------------------------

/**
 * allgather_ciphertext_limbs
 *
 * Each GPU holds local_limbs_per_gpu limbs of a ciphertext.
 * After the call, every GPU has ALL limbs of the ciphertext.
 *
 * This is the first communication step for the Input Broadcast key-switching
 * algorithm: once every GPU has all limbs of c2, each GPU can independently
 * compute the full key-switching inner product.
 *
 * Internally uses ncclAllGather on ncclUint64.
 *
 * @param ctx          MultiGpuContext (NCCL comms + streams)
 * @param gpu_id       Index into ctx (which GPU is calling)
 * @param local_buf    Send buffer on this GPU: [n_polys][local_limbs][degree]
 * @param recv_buf     Receive buffer on this GPU: [n_polys][total_limbs][degree]
 * @param n_polys      Number of ciphertext polynomials
 * @param local_limbs  Number of limbs owned by this GPU
 * @param degree       poly_modulus_degree (N)
 */
void allgather_ciphertext_limbs(MultiGpuContext &ctx,
                                int gpu_id,
                                const uint64_t *local_buf,
                                uint64_t       *recv_buf,
                                size_t n_polys,
                                size_t local_limbs,
                                size_t degree);

/**
 * allreduce_partial_keyswitching
 *
 * Each GPU has computed a partial key-switching result from its local limbs.
 * AllReduce (sum) merges all partial results so every GPU has the final result.
 *
 * This is the communication step for the Output Aggregation key-switching
 * algorithm.
 *
 * @param ctx          MultiGpuContext
 * @param gpu_id       Index into ctx
 * @param partial_buf  In/out buffer: [n_polys][local_limbs][degree] (partial -> full result)
 * @param n_polys      Number of ciphertext polynomials
 * @param local_limbs  Number of limbs managed by this GPU
 * @param degree       poly_modulus_degree (N)
 */
void allreduce_partial_keyswitching(MultiGpuContext &ctx,
                                    int gpu_id,
                                    uint64_t *partial_buf,
                                    size_t n_polys,
                                    size_t local_limbs,
                                    size_t degree);

/**
 * broadcast_ciphertext
 *
 * Broadcast a full ciphertext from `root_gpu` to all other GPUs.
 * Used to distribute shared inputs (e.g. relin/galois keys) at setup time.
 *
 * @param ctx        MultiGpuContext
 * @param gpu_id     Index into ctx for the calling GPU
 * @param buf        Buffer on this GPU: [n_polys][total_limbs][degree]
 * @param n_polys    Number of polynomials
 * @param n_limbs    Total limbs in the ciphertext
 * @param degree     poly_modulus_degree
 * @param root_gpu   Source GPU index (0-based)
 */
void broadcast_ciphertext(MultiGpuContext &ctx,
                          int gpu_id,
                          uint64_t *buf,
                          size_t n_polys,
                          size_t n_limbs,
                          size_t degree,
                          int root_gpu);

} // namespace nexus_multi_gpu
