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
 *   1. mod-up: each GPU runs modup_partial over its OWN CONTIGUOUS digit
 *      range [d_start, d_start + d_count); no work for unowned digits.
 *   2. partial inner product: accumulate over the same owned range only.
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
#include "nvtx_tracer.cuh"

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

// Partial inner product kernel for Output Aggregation.
//
// T-MODUP layout: each GPU has a CONTIGUOUS chunk of digits in `c2`.
//   c2 holds d_count digits at slots [0 .. d_count) (size d_count*size_QlP_n)
//   evks holds beta global pointers; only [d_start .. d_start+d_count) are valid
// The kernel walks local digit i in [0, d_count), reading c2 at local slot i and
// evks at global slot d_start + i.
__global__ void partial_key_switch_inner_prod(
    uint64_t       *dst,
    const uint64_t *c2,            // mod-up'd c2 [d_count * size_QlP_n], CONTIGUOUS
    const uint64_t *const *evks,   // evk pointers [beta], indexed by GLOBAL digit
    const DModulus *modulus,
    size_t          n,             // poly_modulus_degree
    size_t          size_QP,
    size_t          size_QP_n,
    size_t          size_QlP,
    size_t          size_QlP_n,
    size_t          size_Q,
    size_t          size_Ql,
    size_t          d_start,       // global index of first owned digit
    size_t          d_count,       // number of owned digits (length of c2 in slots)
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

        // T-MODUP: walk owned digits CONTIGUOUSLY.
        //   c2 indexed locally at slot d in [0, d_count).
        //   evks indexed globally at d_start + d.
        bool first = true;
        for (size_t d = 0; d < d_count; d++) {
            if (!first && reduction_threshold == 0) {
                acc0.lo = barrett_reduce_uint128_uint64(acc0, mod.value(), mod.const_ratio());
                acc0.hi = 0;
                acc1.lo = barrett_reduce_uint128_uint64(acc1, mod.value(), mod.const_ratio());
                acc1.hi = 0;
            }
            first = false;

            const size_t global_d = d_start + d;
            const uint64_t c2_val = c2[c2_id + d * size_QlP_n];

            prod0 = multiply_uint64_uint64(c2_val, evks[global_d][evk_id]);
            add_uint128_uint128(acc0, prod0, acc0);

            prod1 = multiply_uint64_uint64(c2_val, evks[global_d][evk_id + size_QP_n]);
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
    // T-OVERLAP: replace blocking cudaStreamSynchronize with a recorded event.
    // Downstream mod-reduce / moddown / add-to-ct kernels run on the SAME stream
    // and therefore correctly serialize after the AllReduce — no host-side wait
    // needed. The CPU thread now returns immediately and can launch the next
    // rotation's modup, overlapping it with this rotation's ~291 ms AllReduce.
    if (gpu_id < (int)ctx.allreduce_done_events.size() &&
        ctx.allreduce_done_events[gpu_id]) {
        CUDA_CHECK(cudaEventRecord(ctx.allreduce_done_events[gpu_id],
                                   ctx.streams[gpu_id]));
    }
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
        data[tid] = data[tid] % mod.value();
    }
}

// ---------------------------------------------------------------------------
// add correction to ciphertext (replaces Phantom's add_to_ct_kernel)
// ---------------------------------------------------------------------------
__global__ void oa_add_to_ct(
    uint64_t       *ct,     // destination: ct polynomial
    const uint64_t *cx,     // source: key-switch correction
    const DModulus *modulus,
    size_t          n,      // poly_modulus_degree
    size_t          n_limbs)
{
    size_t total = n_limbs * n;
    for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < total;
         tid += blockDim.x * gridDim.x)
    {
        size_t limb_idx = tid / n;
        DModulus mod = modulus[limb_idx];
        uint64_t sum = ct[tid] + cx[tid];
        if (sum >= mod.value()) sum -= mod.value();
        ct[tid] = sum;
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
    int                    n_gpus,
    uint64_t             **custom_evks)
{
    CUDA_CHECK(cudaSetDevice(gpu_id));
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

    // ---- Step 1: T-MODUP (per-digit mod-up; each GPU computes only its owned digits) ----
    // CONTIGUOUS ownership: GPU g owns digits [g * d_count, (g+1) * d_count).
    // The DKS evks shard (in the DKS variant) and the relin_keys path (single-GPU
    // baseline-style) both treat the evks pointer array as global-indexed.
    // Uneven contiguous sharding: distribute remainder across the first GPUs so
    // each GPU owns either floor(beta/n) or floor(beta/n)+1 contiguous digits.
    // Some keys at smaller chain levels have beta < n_gpus (e.g., beta=3 with 4 GPUs):
    // the trailing GPUs receive d_count=0 and become no-ops for those keys.
    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();
    const size_t d_count_min = beta / static_cast<size_t>(n_gpus);
    const size_t remainder   = beta % static_cast<size_t>(n_gpus);
    const size_t d_start = static_cast<size_t>(gpu_id) * d_count_min +
                           std::min(static_cast<size_t>(gpu_id), remainder);
    const size_t d_count = d_count_min + (static_cast<size_t>(gpu_id) < remainder ? 1 : 0);

    // Layout 1 (CONTIGUOUS): allocate only d_count digits worth of t_mod_up.
    // At N=65536, beta=36, n_gpus=4 this is ~195 MB instead of ~780 MB per GPU.
    // Allocate at least 1 element to avoid cudaMallocAsync(0) returning a
    // sentinel pointer that some downstream sanity checks reject. The kernel
    // does not read this buffer when d_count == 0, so a 1-element scratch
    // is sufficient.
    auto t_mod_up = make_cuda_auto_ptr<uint64_t>(
        std::max<size_t>(1, d_count * size_QlP_n), s);
    {
        NVTX_SCOPE("modup");
        rns_tool.modup_partial(t_mod_up.get(), c2, phantom_ctx.gpu_rns_tables(),
                               key_parms.scheme(), d_start, d_count, s);
    }

    // ---- Step 2: partial inner product (CONTIGUOUS owned digits) ----
    auto cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);

    // Defensive zero-init of cx: even though the kernel writes every output
    // position when d_count > 0, an explicit memset guards against the
    // d_count == 0 path (small chain levels where beta < n_gpus) where the
    // kernel launch would still be made but contribute nothing meaningful;
    // also makes ncclAllReduce's contract on the input buffer trivially valid.
    CUDA_CHECK(cudaMemsetAsync(cx.get(), 0,
                               2 * size_QlP_n * sizeof(uint64_t), s));

    auto reduction_threshold =
        (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;

    // Use custom evks shard if provided (DKS), otherwise use relin_keys
    const uint64_t *const *evks_ptr = custom_evks
        ? const_cast<const uint64_t *const *>(custom_evks)
        : relin_keys.public_keys_ptr();

    // Kernel walks d in [0, d_count) over CONTIGUOUS local t_mod_up,
    // and indexes evks at the global digit index d_start + d.
    // Skip kernel launch entirely when d_count == 0 (this GPU contributes 0).
    if (d_count > 0)
    partial_key_switch_inner_prod<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
        cx.get(), t_mod_up.get(), evks_ptr,
        modulus_QP, n, size_QP, size_QP * n,
        size_QlP, size_QlP_n, size_Q, size_Ql,
        d_start, d_count,
        reduction_threshold);

    // ---- T-STRAGGLER: GPU-side barrier before AllReduce (see DKS variant) ----
    if (gpu_id < (int)ctx.ready_events.size() && ctx.ready_events[gpu_id]) {
        CUDA_CHECK(cudaEventRecord(ctx.ready_events[gpu_id], s));
        for (int g = 0; g < n_gpus; ++g) {
            if (g < (int)ctx.ready_events.size() && ctx.ready_events[g]) {
                CUDA_CHECK(cudaStreamWaitEvent(s, ctx.ready_events[g], 0));
            }
        }
    }

    // ---- Step 3: AllReduce partial inner products ----
    allreduce_keyswitching_result(ctx, gpu_id, cx.get(), 2 * size_QlP_n);

    // ---- Step 3.5: Modular reduction after AllReduce ----
    // AllReduce summed uint64 values; need mod reduction for correctness.
    for (size_t i = 0; i < 2; i++) {
        mod_reduce_after_allreduce<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
            cx.get() + i * size_QlP_n, modulus_QP, n, size_QlP, size_Q, size_Ql);
    }

    // ---- Step 4: mod-down ----
    {
        NVTX_SCOPE("moddown");
        for (size_t i = 0; i < 2; i++) {
            auto cx_i = cx.get() + i * size_QlP_n;
            rns_tool.moddown_from_NTT(cx_i, cx_i, phantom_ctx.gpu_rns_tables(),
                                      key_parms.scheme(), s);
        }
    }

    // ---- Step 5: Add correction to ciphertext ----
    // ct_i += cx_i (mod q_j) for each limb j in [0, size_Ql)
    for (size_t i = 0; i < 2; i++) {
        auto cx_i = cx.get() + i * size_QlP_n;
        auto ct_i = encrypted.data() + i * size_Ql_n;
        oa_add_to_ct<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
            ct_i, cx_i, modulus_QP, n, size_Ql);
    }

    // T-OVERLAP (reverted): host sync. The cross-stream wait at writeback was
    // buggy under some chain levels — restore the safe sync until investigated.
    if (gpu_id < (int)ctx.oa_done_events.size() && ctx.oa_done_events[gpu_id]) {
        CUDA_CHECK(cudaEventRecord(ctx.oa_done_events[gpu_id], s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
}

// ---------------------------------------------------------------------------
// keyswitching_output_aggregation_dks — DKS variant (no PhantomRelinKey needed)
// ---------------------------------------------------------------------------
// Identical to keyswitching_output_aggregation but takes custom_evks directly.
// Used by galois_oa.cu for distributed rotation where each GPU only holds
// its shard of the Galois key digits.

void keyswitching_output_aggregation_dks(
    MultiGpuContext       &ctx,
    const PhantomContext  &phantom_ctx,
    int                    gpu_id,
    PhantomCiphertext     &encrypted,
    uint64_t              *c2,
    uint64_t             **custom_evks,
    int                    n_gpus)
{
    CUDA_CHECK(cudaSetDevice(gpu_id));
    const cudaStream_t &s = ctx.streams[gpu_id];

    // ---- Extract parameters ----
    auto &key_context_data = phantom_ctx.get_context_data(0);
    auto &key_parms        = key_context_data.parms();
    auto n                 = key_parms.poly_modulus_degree();
    auto &key_modulus      = key_parms.coeff_modulus();
    size_t size_P          = key_parms.special_modulus_size();
    size_t size_QP         = key_modulus.size();

    uint32_t levelsDropped = encrypted.chain_index() - 1;
    auto &rns_tool         = phantom_ctx.get_context_data(1 + levelsDropped).gpu_rns_tool();
    auto modulus_QP        = phantom_ctx.gpu_rns_tables().modulus();

    size_t size_Ql   = rns_tool.base_Ql().size();
    size_t size_Q    = size_QP - size_P;
    size_t size_QlP  = size_Ql + size_P;
    auto size_Ql_n   = size_Ql * n;
    auto size_QlP_n  = size_QlP * n;

    // ---- Step 1: T-MODUP — per-digit mod-up (only owned digits) ----
    // CONTIGUOUS ownership: GPU g owns digits [g*d_count .. (g+1)*d_count).
    // This matches the contiguous layout used by DistGaloisKeyStore so that
    // evks[d_start + d] is always non-null for d in [0, d_count).
    // Uneven contiguous sharding (see non-DKS variant for rationale).
    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();
    const size_t d_count_min = beta / static_cast<size_t>(n_gpus);
    const size_t remainder   = beta % static_cast<size_t>(n_gpus);
    const size_t d_start = static_cast<size_t>(gpu_id) * d_count_min +
                           std::min(static_cast<size_t>(gpu_id), remainder);
    const size_t d_count = d_count_min + (static_cast<size_t>(gpu_id) < remainder ? 1 : 0);

    // Layout 1 (CONTIGUOUS): only d_count digits' worth of t_mod_up.
    // Allocate at least 1 element to avoid cudaMallocAsync(0) returning a
    // sentinel pointer that some downstream sanity checks reject. The kernel
    // does not read this buffer when d_count == 0, so a 1-element scratch
    // is sufficient.
    auto t_mod_up = make_cuda_auto_ptr<uint64_t>(
        std::max<size_t>(1, d_count * size_QlP_n), s);
    {
        NVTX_SCOPE("modup");
        rns_tool.modup_partial(t_mod_up.get(), c2, phantom_ctx.gpu_rns_tables(),
                               key_parms.scheme(), d_start, d_count, s);
    }

    // ---- Step 2: partial inner product (CONTIGUOUS owned digits) ----
    auto cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);
    // Defensive zero-init: covers the d_count == 0 case (small chain levels)
    // and gives ncclAllReduce a trivially valid input even if the kernel
    // is skipped on this GPU.
    CUDA_CHECK(cudaMemsetAsync(cx.get(), 0,
                               2 * size_QlP_n * sizeof(uint64_t), s));

    auto reduction_threshold =
        (1 << (bits_per_uint64 -
               static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;

    if (d_count > 0)
    partial_key_switch_inner_prod<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
        cx.get(), t_mod_up.get(),
        const_cast<const uint64_t *const *>(custom_evks),
        modulus_QP, n, size_QP, size_QP * n,
        size_QlP, size_QlP_n, size_Q, size_Ql,
        d_start, d_count,
        reduction_threshold);

    // ---- T-STRAGGLER: GPU-side barrier before AllReduce ----
    // Record this GPU's "ready for AllReduce" event, then make this stream wait
    // on ALL GPUs' ready events. After this point every stream has reached the
    // same logical point, so ncclAllReduce starts in lockstep across the 4 GPUs
    // and the previous ~530 ms straggler wait is eliminated.
    if (gpu_id < (int)ctx.ready_events.size() && ctx.ready_events[gpu_id]) {
        CUDA_CHECK(cudaEventRecord(ctx.ready_events[gpu_id], s));
        for (int g = 0; g < n_gpus; ++g) {
            if (g < (int)ctx.ready_events.size() && ctx.ready_events[g]) {
                CUDA_CHECK(cudaStreamWaitEvent(s, ctx.ready_events[g], 0));
            }
        }
    }

    // ---- Step 3: AllReduce partial inner products ----
    allreduce_keyswitching_result(ctx, gpu_id, cx.get(), 2 * size_QlP_n);

    // ---- Step 3.5: Modular reduction after AllReduce ----
    for (size_t i = 0; i < 2; i++) {
        mod_reduce_after_allreduce<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
            cx.get() + i * size_QlP_n, modulus_QP, n, size_QlP, size_Q, size_Ql);
    }

    // ---- Step 4: mod-down ----
    {
        NVTX_SCOPE("moddown");
        for (size_t i = 0; i < 2; i++) {
            auto cx_i = cx.get() + i * size_QlP_n;
            rns_tool.moddown_from_NTT(cx_i, cx_i, phantom_ctx.gpu_rns_tables(),
                                      key_parms.scheme(), s);
        }
    }

    // ---- Step 5: Add correction to ciphertext ----
    for (size_t i = 0; i < 2; i++) {
        auto cx_i = cx.get() + i * size_QlP_n;
        auto ct_i = encrypted.data() + i * size_Ql_n;
        oa_add_to_ct<<<size_Ql_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
            ct_i, cx_i, modulus_QP, n, size_Ql);
    }

    // T-OVERLAP: replace the trailing host sync with a recorded event. The
    // rotation caller's writeback memcpy (on stream0) will cudaStreamWaitEvent
    // on this event before issuing its memcpy, preserving GPU-side ordering
    // without blocking the worker CPU thread.
    if (gpu_id < (int)ctx.oa_done_events.size() && ctx.oa_done_events[gpu_id]) {
        CUDA_CHECK(cudaEventRecord(ctx.oa_done_events[gpu_id], s));
    } else {
        CUDA_CHECK(cudaStreamSynchronize(s));
    }
}

} // namespace nexus_multi_gpu
