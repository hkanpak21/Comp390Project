/**
 * output_aggregation.cu
 *
 * Implementation of Output Aggregation key-switching for multi-GPU FHE.
 *
 * Algorithm recap:
 *   1. Each GPU computes a PARTIAL inner product from its local c2 limbs.
 *   2. AllReduce (sum) across all GPUs -> every GPU holds the full result.
 *   3. Each GPU keeps the result for its local limbs.
 *
 * Advantage over Input Broadcast:
 *   - c2 is never replicated in full on any GPU.
 *   - Lower peak HBM usage: important for larger parameter sets or many
 *     simultaneous ciphertexts (e.g., during bootstrapping).
 *
 * Disadvantage:
 *   - The partial inner product is harder to implement correctly: we need
 *     to handle the digit decomposition of c2 using only local limbs.
 *   - Requires an AllReduce at the OUTPUT (not input), which can be harder
 *     to overlap with computation.
 *
 * TODO on EC2: replace partial_inner_product_local_limbs stub with
 * the actual Phantom key-switch kernel call using only local decomposition digits.
 */

#include "output_aggregation.cuh"
#include "../comm/nccl_comm.cuh"
#include "../partition/rns_partition.cuh"

#include <stdexcept>
#include <cstring>

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
// partial_inner_product_local_limbs
// ---------------------------------------------------------------------------
// STUB: Computes partial key-switch inner product from local c2 limbs only.
// On EC2: replace with modified Phantom kernel that processes only digits
// corresponding to local limbs (digit_start to digit_start + digit_count).

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
    cudaStream_t           stream)
{
    // Stub: zero output (no-op partial product).
    CUDA_CHECK(cudaMemsetAsync(partial_out_c0, 0,
               total_limbs * degree * sizeof(uint64_t), stream));
    CUDA_CHECK(cudaMemsetAsync(partial_out_c1, 0,
               total_limbs * degree * sizeof(uint64_t), stream));

    (void)local_c2; (void)evk_data; (void)modulus_ptr;
    (void)beta; (void)local_digit_start; (void)local_digit_count;

    // TODO on EC2:
    // For digits d in [local_digit_start, local_digit_start + local_digit_count):
    //   Decompose local_c2 into digit d using hybrid/HPS decomposition.
    //   Multiply by evk_data[d] and accumulate into partial_out_c0/c1.
    // This is a partial sum; the AllReduce below will sum across GPUs.
}

// ---------------------------------------------------------------------------
// allreduce_keyswitching_result
// ---------------------------------------------------------------------------

void allreduce_keyswitching_result(
    MultiGpuContext &ctx,
    int              gpu_id,
    uint64_t        *partial_out_c0,
    uint64_t        *partial_out_c1,
    size_t           total_limbs,
    size_t           degree)
{
    size_t count = total_limbs * degree;
    // Group start/end ensures both AllReduces are issued atomically.
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclAllReduce(partial_out_c0, partial_out_c0,
                             count, ncclUint64, ncclSum,
                             ctx.comms[gpu_id], ctx.streams[gpu_id]));
    NCCL_CHECK(ncclAllReduce(partial_out_c1, partial_out_c1,
                             count, ncclUint64, ncclSum,
                             ctx.comms[gpu_id], ctx.streams[gpu_id]));
    NCCL_CHECK(ncclGroupEnd());
    CUDA_CHECK(cudaStreamSynchronize(ctx.streams[gpu_id]));
    // Note: the AllReduce sums uint64_t values without modular reduction.
    // Caller must apply modular reduction (mod q_j for each limb j) afterward.
}

// ---------------------------------------------------------------------------
// keyswitching_output_aggregation
// ---------------------------------------------------------------------------

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
    int                     n_gpus)
{
    size_t local_n = n_local_limbs(gpu_id, n_gpus, total_limbs);

    // Determine which digits belong to this GPU.
    // With cyclic limb assignment, GPU g owns digits d where d % n_gpus == g.
    // (In the HPS decomposition, digit d corresponds to limb group d.)
    size_t local_digit_start = static_cast<size_t>(gpu_id);
    size_t local_digit_count = local_n;  // one digit per local limb (approximation)

    // Allocate full-size partial output buffers.
    uint64_t *partial_c0 = nullptr, *partial_c1 = nullptr;
    CUDA_CHECK(cudaMalloc(&partial_c0, total_limbs * degree * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&partial_c1, total_limbs * degree * sizeof(uint64_t)));

    // Step 1: Compute partial inner product from local limbs.
    cudaStream_t compute_stream = ctx.streams[gpu_id];
    partial_inner_product_local_limbs(
        local_c2,
        /*evk_data=*/nullptr,
        partial_c0, partial_c1,
        /*modulus_ptr=*/nullptr,
        /*beta=*/total_limbs,
        local_digit_start, local_digit_count,
        total_limbs, degree,
        compute_stream);

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    // Step 2: AllReduce to sum partial contributions from all GPUs.
    allreduce_keyswitching_result(ctx, gpu_id,
                                  partial_c0, partial_c1,
                                  total_limbs, degree);

    // Step 3: Extract local limbs of the combined result.
    gather_ciphertext_from_gpu(partial_c0, local_out_c0_correction,
                               1, total_limbs, degree, gpu_id, n_gpus,
                               compute_stream);
    gather_ciphertext_from_gpu(partial_c1, local_out_c1_correction,
                               1, total_limbs, degree, gpu_id, n_gpus,
                               compute_stream);

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    CUDA_CHECK(cudaFree(partial_c0));
    CUDA_CHECK(cudaFree(partial_c1));

    (void)phantom_ctx; (void)chain_index; (void)evk;
}

// ---------------------------------------------------------------------------
// rotation_output_aggregation
// ---------------------------------------------------------------------------

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
    int                     n_gpus)
{
    size_t local_n = n_local_limbs(gpu_id, n_gpus, total_limbs);

    uint64_t *partial_c0 = nullptr, *partial_c1 = nullptr;
    CUDA_CHECK(cudaMalloc(&partial_c0, total_limbs * degree * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&partial_c1, total_limbs * degree * sizeof(uint64_t)));

    cudaStream_t s = ctx.streams[gpu_id];
    partial_inner_product_local_limbs(
        local_c1_rotated, nullptr,
        partial_c0, partial_c1,
        nullptr, total_limbs,
        static_cast<size_t>(gpu_id), local_n,
        total_limbs, degree, s);

    CUDA_CHECK(cudaStreamSynchronize(s));
    allreduce_keyswitching_result(ctx, gpu_id, partial_c0, partial_c1,
                                  total_limbs, degree);

    gather_ciphertext_from_gpu(partial_c0, local_out_c0_correction,
                               1, total_limbs, degree, gpu_id, n_gpus, s);
    gather_ciphertext_from_gpu(partial_c1, local_out_c1_new,
                               1, total_limbs, degree, gpu_id, n_gpus, s);

    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(partial_c0));
    CUDA_CHECK(cudaFree(partial_c1));

    (void)phantom_ctx; (void)chain_index; (void)gk; (void)galois_elt;
}

} // namespace nexus_multi_gpu
