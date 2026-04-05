/**
 * output_aggregation.cu
 *
 * Multi-GPU key-switching via Output Aggregation (Cinnamon, ASPLOS 2025).
 *
 * The key difference from Input Broadcast: c2 is NOT gathered to every GPU.
 * Instead, each GPU processes only its assigned digits of the decomposition
 * and contributes a partial inner product. AllReduce sums these partials.
 *
 * Pipeline per GPU:
 *   1. mod-up: decompose c2 into beta digits. Each GPU does mod-up for ALL
 *      digits (c2 is available locally after ciphertext multiplication).
 *   2. partial inner product: GPU g accumulates only digits d where d%n == g.
 *   3. AllReduce: sum partial inner products across GPUs.
 *   4. mod-down: convert from QlP basis to Ql basis (local, no communication).
 *   5. Add correction to ciphertext.
 *
 * Custom kernel: We modify Phantom's key_switch_inner_prod_c2_and_evk to
 * accept a digit range [d_start, d_count) instead of iterating 0..beta-1.
 * This is the only Phantom-internal modification required.
 */

#include "output_aggregation.cuh"
#include "../comm/nccl_comm.cuh"
#include "../partition/rns_partition.cuh"

#include "evaluate.cuh"
#include "context.cuh"
#include "secretkey.h"
#include "ciphertext.h"
#include "rns.cuh"
#include "polymath.cuh"
#include "uintmodmath.cuh"

#include <stdexcept>
#include <cstdio>

#define CUDA_CHECK(cmd) do {                                             \
    cudaError_t e = (cmd);                                               \
    if (e != cudaSuccess) {                                              \
        throw std::runtime_error(std::string("CUDA error in ") +        \
                                 __func__ + ": " +                       \
                                 cudaGetErrorString(e));                  \
    }                                                                    \
} while (0)

#define NCCL_CHECK(cmd) do {                                             \
    ncclResult_t r = (cmd);                                              \
    if (r != ncclSuccess) {                                              \
        throw std::runtime_error(std::string("NCCL error in ") +        \
                                 __func__ + ": " +                       \
                                 ncclGetErrorString(r));                  \
    }                                                                    \
} while (0)

using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

namespace nexus_multi_gpu {

// ---------------------------------------------------------------------------
// Partial inner product kernel
// ---------------------------------------------------------------------------
// This is a modified version of Phantom's key_switch_inner_prod_c2_and_evk
// that processes only digits in range [d_start, d_start + d_count).
// The output is a PARTIAL sum — AllReduce across GPUs yields the full result.

__global__ void partial_key_switch_inner_prod(
    uint64_t       *dst,
    const uint64_t *c2,            // mod-up'd c2 [beta * size_QlP_n]
    const uint64_t *const *evks,   // evk pointers [beta]
    const DModulus *modulus,
    size_t          n,             // poly_modulus_degree
    size_t          size_QP,
    size_t          size_QP_n,
    size_t          size_QlP,
    size_t          size_QlP_n,
    size_t          size_Q,
    size_t          size_Ql,
    size_t          d_start,       // first digit for this GPU
    size_t          d_count,       // number of digits for this GPU
    size_t          reduction_threshold)
{
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < size_QlP_n;
         tid += blockDim.x * gridDim.x)
    {
        size_t nid = tid / n;
        size_t twr = (nid >= size_Ql ? size_Q + (nid - size_Ql) : nid);
        DModulus mod = modulus[twr];
        uint64_t evk_id = (tid % n) + twr * n;
        uint64_t c2_id = (tid % n) + nid * n;

        uint128_t prod0, prod1;
        uint128_t acc0 = {0, 0};
        uint128_t acc1 = {0, 0};

        // Accumulate only digits assigned to this GPU: [d_start, d_start + d_count)
        for (uint64_t di = 0; di < d_count; di++) {
            uint64_t d = d_start + di;

            if (di > 0 && reduction_threshold == 0) {
                acc0.lo = barrett_reduce_uint128_uint64(acc0, mod.value(), mod.const_ratio());
                acc0.hi = 0;
                acc1.lo = barrett_reduce_uint128_uint64(acc1, mod.value(), mod.const_ratio());
                acc1.hi = 0;
            }

            prod0 = multiply_uint64_uint64(c2[c2_id + d * size_QlP_n], evks[d][evk_id]);
            add_uint128_uint128(acc0, prod0, acc0);

            prod1 = multiply_uint64_uint64(c2[c2_id + d * size_QlP_n], evks[d][evk_id + size_QP_n]);
            add_uint128_uint128(acc1, prod1, acc1);
        }

        // Barrett reduce and store partial result
        dst[tid]              = barrett_reduce_uint128_uint64(acc0, mod.value(), mod.const_ratio());
        dst[tid + size_QlP_n] = barrett_reduce_uint128_uint64(acc1, mod.value(), mod.const_ratio());
    }
}

// ---------------------------------------------------------------------------
// allreduce_keyswitching_result
// ---------------------------------------------------------------------------

void allreduce_keyswitching_result(
    MultiGpuContext &ctx,
    int              gpu_id,
    uint64_t        *partial_cx,
    size_t           count)
{
    NCCL_CHECK(ncclAllReduce(partial_cx, partial_cx,
                             count, ncclUint64, ncclSum,
                             ctx.comms[gpu_id], ctx.streams[gpu_id]));
    CUDA_CHECK(cudaStreamSynchronize(ctx.streams[gpu_id]));
}

// ---------------------------------------------------------------------------
// Modular reduction kernel (post-AllReduce)
// ---------------------------------------------------------------------------
// AllReduce sums uint64 values without modular reduction.
// After AllReduce, each element must be reduced mod q_j for its limb j.

__global__ void mod_reduce_after_allreduce(
    uint64_t       *data,
    const DModulus *modulus,
    size_t          n,       // poly_modulus_degree
    size_t          n_limbs, // size_QlP
    size_t          size_Q,
    size_t          size_Ql)
{
    size_t total = n_limbs * n;
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += blockDim.x * gridDim.x)
    {
        size_t nid = tid / n;
        size_t twr = (nid >= size_Ql ? size_Q + (nid - size_Ql) : nid);
        DModulus mod = modulus[twr];
        data[tid] = barrett_reduce_uint64_uint64(data[tid], mod.value(), mod.const_ratio());
    }
}

// ---------------------------------------------------------------------------
// keyswitching_output_aggregation
// ---------------------------------------------------------------------------

void keyswitching_output_aggregation(
    MultiGpuContext       &ctx,
    const PhantomContext  &phantom_ctx,
    int                    gpu_id,
    PhantomCiphertext     &encrypted,
    uint64_t              *c2,
    const PhantomRelinKey &relin_keys,
    int                    n_gpus)
{
    const cudaStream_t &s = ctx.streams[gpu_id];

    // ---- Extract parameters (same logic as Phantom's keyswitch_inplace) ----
    auto &key_context_data = phantom_ctx.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto n = key_parms.poly_modulus_degree();
    auto &key_modulus = key_parms.coeff_modulus();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();

    // For CKKS: levelsDropped = chain_index - 1
    uint32_t levelsDropped = encrypted.chain_index() - 1;
    auto &rns_tool = phantom_ctx.get_context_data(1 + levelsDropped).gpu_rns_tool();
    auto modulus_QP = phantom_ctx.gpu_rns_tables().modulus();

    size_t size_Ql = rns_tool.base_Ql().size();
    size_t size_Q = size_QP - size_P;
    size_t size_QlP = size_Ql + size_P;
    auto size_Ql_n = size_Ql * n;
    auto size_QlP_n = size_QlP * n;

    // Handle HPS leveled scaling if needed
    auto mul_tech = key_parms.mul_tech();
    if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
        auto t_cks = make_cuda_auto_ptr<uint64_t>(size_Q * n, s);
        cudaMemcpyAsync(t_cks.get(), c2, size_Q * n * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, s);
        rns_tool.scaleAndRound_HPS_Q_Ql(c2, t_cks.get(), s);
    }

    // ---- Step 1: mod-up (each GPU does this for ALL digits) ----
    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();
    auto t_mod_up = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);
    rns_tool.modup(t_mod_up.get(), c2, phantom_ctx.gpu_rns_tables(), key_parms.scheme(), s);

    // ---- Step 2: partial inner product (only local digits) ----
    // Assign digits to GPUs round-robin: GPU g processes digits d where d % n_gpus == g
    size_t d_start = static_cast<size_t>(gpu_id);
    size_t d_count = 0;
    for (size_t d = d_start; d < beta; d += n_gpus)
        d_count++;

    // However, the kernel needs contiguous digit iteration. Since digits may not
    // be contiguous for this GPU, we need to handle the strided pattern.
    // For simplicity with the existing kernel structure, if beta <= n_gpus,
    // each GPU gets at most 1 digit. Otherwise we call the kernel multiple times.

    auto cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);
    CUDA_CHECK(cudaMemsetAsync(cx.get(), 0, 2 * size_QlP_n * sizeof(uint64_t), s));

    auto reduction_threshold =
        (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;

    // Process each digit assigned to this GPU
    for (size_t d = gpu_id; d < beta; d += n_gpus) {
        partial_key_switch_inner_prod<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
            cx.get(), t_mod_up.get(), relin_keys.public_keys_ptr(),
            modulus_QP, n, size_QP, size_QP * n,
            size_QlP, size_QlP_n, size_Q, size_Ql,
            d, 1,  // one digit at a time
            reduction_threshold);
    }

    // ---- Step 3: AllReduce partial inner products ----
    allreduce_keyswitching_result(ctx, gpu_id, cx.get(), 2 * size_QlP_n);

    // ---- Step 3.5: Modular reduction after AllReduce ----
    // AllReduce summed uint64 values; need mod reduction for correctness.
    for (size_t i = 0; i < 2; i++) {
        mod_reduce_after_allreduce<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
            cx.get() + i * size_QlP_n, modulus_QP, n, size_QlP, size_Q, size_Ql);
    }

    // ---- Step 4: mod-down (local, no communication) ----
    for (size_t i = 0; i < 2; i++) {
        auto cx_i = cx.get() + i * size_QlP_n;
        rns_tool.moddown_from_NTT(cx_i, cx_i, phantom_ctx.gpu_rns_tables(),
                                  key_parms.scheme(), s);
    }

    // ---- Step 5: Add correction to ciphertext ----
    for (size_t i = 0; i < 2; i++) {
        auto cx_i = cx.get() + i * size_QlP_n;

        if (mul_tech == mul_tech_type::hps_overq_leveled && levelsDropped) {
            auto ct_i = encrypted.data() + i * size_Q * n;
            auto t_cx = make_cuda_auto_ptr<uint64_t>(size_Q * n, s);
            rns_tool.ExpandCRTBasis_Ql_Q(t_cx.get(), cx_i, s);
            add_to_ct_kernel<<<(size_Q * n) / blockDimGlb.x, blockDimGlb, 0, s>>>(
                ct_i, t_cx.get(), rns_tool.base_Q().base(), n, size_Q);
        } else {
            auto ct_i = encrypted.data() + i * size_Ql_n;
            add_to_ct_kernel<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
                ct_i, cx_i, rns_tool.base_Ql().base(), n, size_Ql);
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(s));
}

} // namespace nexus_multi_gpu
