/**
 * rns_partition.cu
 *
 * CUDA kernel implementations for RNS limb scatter/gather.
 */

#include "rns_partition.cuh"

#define CUDA_CHECK(cmd) do {                                              \
    cudaError_t e = (cmd);                                                \
    if (e != cudaSuccess) {                                               \
        throw std::runtime_error(std::string("CUDA error: ") +           \
                                 cudaGetErrorString(e));                  \
    }                                                                     \
} while (0)

namespace nexus_multi_gpu {

// ---------------------------------------------------------------------------
// kernel_scatter_limbs
// ---------------------------------------------------------------------------
// Grid: gridDim.x = n_polys * local_n_limbs
// Block: blockDim.x = min(degree, 256)
// Each thread copies one contiguous chunk of uint64s for one (poly, local_limb).

__global__ void kernel_scatter_limbs(
    const uint64_t *global_buf,
    uint64_t       *local_buf,
    size_t          n_polys,
    size_t          total_limbs,
    size_t          local_n_limbs,
    size_t          degree,
    int             gpu_id,
    int             n_gpus)
{
    // Determine which (poly, local_limb) this block handles.
    size_t block_id    = blockIdx.x;
    size_t poly        = block_id / local_n_limbs;
    size_t local_loc   = block_id % local_n_limbs;
    size_t global_limb = local_loc * static_cast<size_t>(n_gpus) + static_cast<size_t>(gpu_id);

    if (poly >= n_polys || global_limb >= total_limbs) return;

    const uint64_t *src = global_buf + (poly * total_limbs + global_limb) * degree;
    uint64_t       *dst = local_buf  + (poly * local_n_limbs + local_loc) * degree;

    // Each thread copies multiple elements (stride = blockDim.x).
    for (size_t k = threadIdx.x; k < degree; k += blockDim.x) {
        dst[k] = src[k];
    }
}

// ---------------------------------------------------------------------------
// kernel_gather_limbs
// ---------------------------------------------------------------------------

__global__ void kernel_gather_limbs(
    const uint64_t *local_buf,
    uint64_t       *global_buf,
    size_t          n_polys,
    size_t          total_limbs,
    size_t          local_n_limbs,
    size_t          degree,
    int             gpu_id,
    int             n_gpus)
{
    size_t block_id    = blockIdx.x;
    size_t poly        = block_id / local_n_limbs;
    size_t local_loc   = block_id % local_n_limbs;
    size_t global_limb = local_loc * static_cast<size_t>(n_gpus) + static_cast<size_t>(gpu_id);

    if (poly >= n_polys || global_limb >= total_limbs) return;

    const uint64_t *src = local_buf  + (poly * local_n_limbs + local_loc) * degree;
    uint64_t       *dst = global_buf + (poly * total_limbs + global_limb) * degree;

    for (size_t k = threadIdx.x; k < degree; k += blockDim.x) {
        dst[k] = src[k];
    }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

void scatter_ciphertext_to_gpu(
    const uint64_t *global_buf,
    uint64_t       *local_buf,
    size_t          n_polys,
    size_t          total_limbs,
    size_t          degree,
    int             gpu_id,
    int             n_gpus,
    cudaStream_t    stream)
{
    size_t local_n = n_local_limbs(gpu_id, n_gpus, total_limbs);
    if (local_n == 0) return;

    dim3 grid(static_cast<unsigned>(n_polys * local_n));
    dim3 block(static_cast<unsigned>(std::min(degree, size_t{256})));

    kernel_scatter_limbs<<<grid, block, 0, stream>>>(
        global_buf, local_buf,
        n_polys, total_limbs, local_n, degree,
        gpu_id, n_gpus);
}

void gather_ciphertext_from_gpu(
    const uint64_t *local_buf,
    uint64_t       *global_buf,
    size_t          n_polys,
    size_t          total_limbs,
    size_t          degree,
    int             gpu_id,
    int             n_gpus,
    cudaStream_t    stream)
{
    size_t local_n = n_local_limbs(gpu_id, n_gpus, total_limbs);
    if (local_n == 0) return;

    dim3 grid(static_cast<unsigned>(n_polys * local_n));
    dim3 block(static_cast<unsigned>(std::min(degree, size_t{256})));

    kernel_gather_limbs<<<grid, block, 0, stream>>>(
        local_buf, global_buf,
        n_polys, total_limbs, local_n, degree,
        gpu_id, n_gpus);
}

} // namespace nexus_multi_gpu
