#pragma once
/**
 * rns_partition.cuh
 *
 * RNS limb-to-GPU assignment and ciphertext scatter/gather utilities.
 *
 * The partitioning strategy is cyclic (identical to Cerium):
 *   GPU g owns limb j  iff  (j % n_gpus) == g
 *
 * This gives perfect load balance regardless of the actual prime sizes, and
 * maps well to the interleaved memory layout already used in PhantomCiphertext.
 */

#include <cstddef>
#include <cstdint>
#include <vector>
#include <cassert>

namespace nexus_multi_gpu {

// ---------------------------------------------------------------------------
// Limb ownership queries
// ---------------------------------------------------------------------------

/**
 * owner_of_limb
 * Returns which GPU (0-based) owns limb j under cyclic partitioning.
 */
inline int owner_of_limb(size_t limb_j, int n_gpus) {
    return static_cast<int>(limb_j % static_cast<size_t>(n_gpus));
}

/**
 * local_index_of_limb
 * Returns the local index of global limb j on its owning GPU.
 * Example: n_gpus=4, limb j=9 -> global owner GPU 1, local index 2.
 */
inline size_t local_index_of_limb(size_t limb_j, int n_gpus) {
    return limb_j / static_cast<size_t>(n_gpus);
}

/**
 * global_limb_index
 * Inverse: given GPU g and local limb index loc, returns the global limb index.
 */
inline size_t global_limb_index(int gpu_id, size_t local_idx, int n_gpus) {
    return local_idx * static_cast<size_t>(n_gpus) + static_cast<size_t>(gpu_id);
}

/**
 * n_local_limbs
 * Returns how many limbs GPU g owns given total_limbs limbs and n_gpus GPUs.
 * With cyclic assignment: ceil(total_limbs / n_gpus) or floor(...).
 */
inline size_t n_local_limbs(int gpu_id, int n_gpus, size_t total_limbs) {
    // cyclic: GPU g owns limbs g, g+n, g+2n, ...
    // Count = floor((total_limbs - gpu_id - 1) / n_gpus) + 1  for gpu_id < total_limbs
    if (static_cast<size_t>(gpu_id) >= total_limbs) return 0;
    return (total_limbs - static_cast<size_t>(gpu_id) - 1) / static_cast<size_t>(n_gpus) + 1;
}

/**
 * limbs_for_gpu
 * Returns a vector of global limb indices owned by GPU g.
 * (Useful for iteration in loops.)
 */
inline std::vector<size_t> limbs_for_gpu(int gpu_id, int n_gpus, size_t total_limbs) {
    std::vector<size_t> result;
    for (size_t j = static_cast<size_t>(gpu_id); j < total_limbs; j += static_cast<size_t>(n_gpus)) {
        result.push_back(j);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Buffer layout
// ---------------------------------------------------------------------------

/**
 * local_buf_offset
 *
 * Given a flat ciphertext buffer laid out as:
 *   [poly_i][limb_j][coeff_k]  with stride = degree per limb
 *
 * and a local (per-GPU) buffer laid out as:
 *   [poly_i][local_limb_loc][coeff_k]
 *
 * returns the byte offset into the local buffer for (poly_i, local_limb_loc, coeff=0).
 */
inline size_t local_buf_offset(size_t poly_i,
                               size_t local_limb_loc,
                               size_t local_n_limbs,
                               size_t degree) {
    return (poly_i * local_n_limbs + local_limb_loc) * degree;
}

/**
 * global_buf_offset
 * Returns the offset in the full (global) ciphertext buffer for (poly_i, global_limb_j, coeff=0).
 * Buffer layout: [poly_i][limb_j][coeff_k] with total_limbs limbs.
 */
inline size_t global_buf_offset(size_t poly_i,
                                size_t limb_j,
                                size_t total_limbs,
                                size_t degree) {
    return (poly_i * total_limbs + limb_j) * degree;
}

// ---------------------------------------------------------------------------
// GPU-side scatter / gather kernels
// ---------------------------------------------------------------------------

/**
 * kernel_scatter_limbs
 *
 * CUDA kernel: copies limbs from a full ciphertext buffer (global layout) into
 * a compacted local buffer (local layout) for a specific GPU.
 *
 * Called on the device that owns the destination buffer.
 *
 * Grid: (n_polys * local_n_limbs, 1, 1), each block handles one (poly, local_limb).
 * Block: (min(degree, 256), 1, 1).
 */
__global__ void kernel_scatter_limbs(
    const uint64_t *global_buf,   ///< Source: full ciphertext [n_polys][total_limbs][degree]
    uint64_t       *local_buf,    ///< Dest: local ciphertext  [n_polys][local_limbs][degree]
    size_t          n_polys,
    size_t          total_limbs,
    size_t          local_n_limbs,
    size_t          degree,
    int             gpu_id,
    int             n_gpus
);

/**
 * kernel_gather_limbs
 *
 * CUDA kernel: writes a local buffer back into the correct positions of a
 * full ciphertext buffer.  Inverse of kernel_scatter_limbs.
 */
__global__ void kernel_gather_limbs(
    const uint64_t *local_buf,    ///< Source: local [n_polys][local_limbs][degree]
    uint64_t       *global_buf,   ///< Dest: full  [n_polys][total_limbs][degree]
    size_t          n_polys,
    size_t          total_limbs,
    size_t          local_n_limbs,
    size_t          degree,
    int             gpu_id,
    int             n_gpus
);

// ---------------------------------------------------------------------------
// Host-callable wrappers
// ---------------------------------------------------------------------------

/**
 * scatter_ciphertext_to_gpu
 *
 * From a full ciphertext buffer `global_buf` (on any device), scatter the limbs
 * that belong to `gpu_id` into `local_buf` on that GPU.
 *
 * Both pointers must be accessible from the GPU that will run the kernel
 * (use cudaMemcpyPeer or peer access if they are on different devices).
 */
void scatter_ciphertext_to_gpu(
    const uint64_t *global_buf,
    uint64_t       *local_buf,
    size_t          n_polys,
    size_t          total_limbs,
    size_t          degree,
    int             gpu_id,
    int             n_gpus,
    cudaStream_t    stream
);

/**
 * gather_ciphertext_from_gpu
 *
 * Inverse of scatter_ciphertext_to_gpu: collects local limbs from `gpu_id`
 * back into the full ciphertext buffer.
 */
void gather_ciphertext_from_gpu(
    const uint64_t *local_buf,
    uint64_t       *global_buf,
    size_t          n_polys,
    size_t          total_limbs,
    size_t          degree,
    int             gpu_id,
    int             n_gpus,
    cudaStream_t    stream
);

} // namespace nexus_multi_gpu
