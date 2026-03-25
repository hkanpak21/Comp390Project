/**
 * input_broadcast.cu
 *
 * Implementation of Input Broadcast key-switching for multi-GPU FHE.
 *
 * Algorithm recap:
 *   1. AllGather c2 limbs -> every GPU holds full c2.
 *   2. Each GPU runs the standard Phantom key-switch inner product locally.
 *   3. Extract local-limb slice of the result.
 *
 * Integration point with Phantom:
 *   The single-GPU key-switch inner product is implemented in Phantom as the
 *   kernel `key_switch_inner_prod_c2_and_evk` (in evaluate.cuh).
 *   We call the same kernel here — the only difference is that we first
 *   populate the full c2 buffer via AllGather before calling the kernel.
 *
 * Dependency boundary:
 *   - allgather_c2_limbs: calls NCCL (nccl_stream)
 *   - local_inner_product: calls CUDA kernels (compute_stream)
 *   These are serialized: inner product starts after AllGather completes.
 *
 * TODO on EC2:
 *   Replace the placeholder call in local_inner_product() with the actual
 *   Phantom kernel invocation once we have confirmed Phantom's internal API
 *   for the key_switch_inner_prod function.
 */

#include "input_broadcast.cuh"
#include "../comm/nccl_comm.cuh"
#include "../partition/rns_partition.cuh"

#include <stdexcept>
#include <cstring>
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
// allgather_c2_limbs
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
    // Wait for AllGather to complete before returning.
    CUDA_CHECK(cudaStreamSynchronize(ctx.streams[gpu_id]));
}

// ---------------------------------------------------------------------------
// local_inner_product
// ---------------------------------------------------------------------------
// This is a STUB that will be replaced with the actual Phantom kernel call on EC2.
// The function signature matches what we know about Phantom's key-switch kernel.
//
// On EC2, this should call:
//   key_switch_inner_prod_c2_and_evk<<<grid, block, 0, stream>>>(
//       out_c0, out_c1, full_c2, evk_data, modulus, n, size_QP, size_QP_n,
//       size_QlP, size_QlP_n, size_Q, size_Ql, beta, reduction_threshold);
//
// For local testing and CI, we write zeros (no-op).

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
    cudaStream_t           stream)
{
    // Stub: zero-fill outputs. Replace with Phantom kernel call on EC2.
    size_t buf_size = total_limbs * degree * sizeof(uint64_t);
    CUDA_CHECK(cudaMemsetAsync(out_c0, 0, buf_size, stream));
    CUDA_CHECK(cudaMemsetAsync(out_c1, 0, buf_size, stream));

    // TODO on EC2:
    // extern void key_switch_inner_prod_c2_and_evk(
    //     uint64_t*, uint64_t*, const uint64_t*, const uint64_t* const*,
    //     const DModulus*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);
    //
    // const DModulus* mod = reinterpret_cast<const DModulus*>(modulus_ptr);
    // dim3 grid(...); dim3 block(...);
    // key_switch_inner_prod_c2_and_evk<<<grid, block, 0, stream>>>(
    //     out_c0, out_c1 (packed), full_c2, evk_data, mod, degree,
    //     size_QP, size_QP*degree, ..., total_limbs, beta, ...);
    (void)full_c2; (void)evk_data; (void)modulus_ptr;
    (void)beta; (void)size_QP;
}

// ---------------------------------------------------------------------------
// keyswitching_input_broadcast
// ---------------------------------------------------------------------------

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
    int                     n_gpus)
{
    size_t local_n = n_local_limbs(gpu_id, n_gpus, total_limbs);

    // ---- Step 1: Allocate full c2 buffer on this GPU ----
    uint64_t *full_c2 = nullptr;
    CUDA_CHECK(cudaMalloc(&full_c2, total_limbs * degree * sizeof(uint64_t)));

    // ---- Step 2: AllGather — collect all limbs of c2 on this GPU ----
    allgather_c2_limbs(ctx, gpu_id,
                       local_c2, full_c2,
                       local_n, total_limbs, degree);

    // ---- Step 3: Allocate output buffers (full size for the inner product) ----
    uint64_t *full_c0_corr = nullptr, *full_c1_corr = nullptr;
    CUDA_CHECK(cudaMalloc(&full_c0_corr, total_limbs * degree * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&full_c1_corr, total_limbs * degree * sizeof(uint64_t)));

    // ---- Step 4: Run the inner product locally (each GPU has full c2 + full evk) ----
    cudaStream_t compute_stream = ctx.streams[gpu_id];
    local_inner_product(
        full_c2,
        /*evk_data=*/nullptr,    // TODO: extract raw pointer from PhantomRelinKey
        full_c0_corr, full_c1_corr,
        /*modulus_ptr=*/nullptr, // TODO: extract from PhantomContext
        /*beta=*/0,              // TODO: compute from context
        /*size_QP=*/total_limbs, total_limbs, degree,
        compute_stream);

    // ---- Step 5: Extract local limbs of the result ----
    // The inner product produces full c0/c1 corrections (all limbs).
    // We only store the local-limb slice for this GPU.
    gather_ciphertext_from_gpu(full_c0_corr, local_out_c0_correction,
                               /*n_polys=*/1, total_limbs, degree,
                               gpu_id, n_gpus, compute_stream);
    gather_ciphertext_from_gpu(full_c1_corr, local_out_c1_correction,
                               /*n_polys=*/1, total_limbs, degree,
                               gpu_id, n_gpus, compute_stream);

    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(full_c2));
    CUDA_CHECK(cudaFree(full_c0_corr));
    CUDA_CHECK(cudaFree(full_c1_corr));
}

// ---------------------------------------------------------------------------
// rotation_input_broadcast
// ---------------------------------------------------------------------------

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
    int                     n_gpus)
{
    // Rotation is structurally identical to relinearization:
    // key-switch the rotated c1 polynomial using the GaloisKey for galois_elt.
    // We delegate to the same AllGather + inner product pattern.

    size_t local_n = n_local_limbs(gpu_id, n_gpus, total_limbs);

    uint64_t *full_c1r = nullptr;
    CUDA_CHECK(cudaMalloc(&full_c1r, total_limbs * degree * sizeof(uint64_t)));

    allgather_c2_limbs(ctx, gpu_id,
                       local_c1_rotated, full_c1r,
                       local_n, total_limbs, degree);

    uint64_t *full_c0c = nullptr, *full_c1n = nullptr;
    CUDA_CHECK(cudaMalloc(&full_c0c, total_limbs * degree * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&full_c1n, total_limbs * degree * sizeof(uint64_t)));

    cudaStream_t s = ctx.streams[gpu_id];
    local_inner_product(
        full_c1r, nullptr, full_c0c, full_c1n,
        nullptr, 0, total_limbs, total_limbs, degree, s);

    gather_ciphertext_from_gpu(full_c0c, local_out_c0_correction,
                               1, total_limbs, degree, gpu_id, n_gpus, s);
    gather_ciphertext_from_gpu(full_c1n, local_out_c1_new,
                               1, total_limbs, degree, gpu_id, n_gpus, s);

    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(full_c1r));
    CUDA_CHECK(cudaFree(full_c0c));
    CUDA_CHECK(cudaFree(full_c1n));

    (void)phantom_ctx; (void)chain_index; (void)gk; (void)galois_elt;
}

} // namespace nexus_multi_gpu
