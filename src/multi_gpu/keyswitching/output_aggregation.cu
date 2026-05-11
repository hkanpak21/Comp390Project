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

// Partial inner product kernel for Output Aggregation — STRIDED ownership.
//
// (T-MODUP-FIX-2 2026-05-10): Reverted from a CONTIGUOUS layout that broke
// at chain levels where chain_beta < dnum (DKS shards are pre-sized at dnum
// but chain_beta varies → contiguous slices on trailing GPUs were never
// accessed and trailing GPUs' shards covered the wrong digits → kernel
// dereferenced nullptr evks slots → cudaFreeAsync invalid argument cascade).
//
// Layout: c2 holds the FULL mod-up output [beta * size_QlP_n] on every GPU
// (modup is replicated). evks holds beta global pointers; GPU g owns the
// strided subset {evks[g], evks[g+n_gpus], evks[g+2*n_gpus], ...} ∩ [0, beta)
// (rest are nullptr on this GPU). The kernel walks d = gpu_id, gpu_id+n_gpus,
// ... < beta — every accessed slot is owned. Trailing GPUs with no owned
// digits in [0, beta) (when beta < n_gpus) are guarded out at the kernel
// launch site (`if (d_count > 0)` in the caller).
__global__ void partial_key_switch_inner_prod(
    uint64_t       *dst,
    const uint64_t *c2,            // mod-up'd c2 [beta * size_QlP_n], FULL
    const uint64_t *const *evks,   // evk pointers [beta], strided ownership
    const DModulus *modulus,
    size_t          n,             // poly_modulus_degree
    size_t          size_QP,
    size_t          size_QP_n,
    size_t          size_QlP,
    size_t          size_QlP_n,
    size_t          size_Q,
    size_t          size_Ql,
    size_t          gpu_id,        // this GPU's index (start of strided walk)
    size_t          n_gpus,        // total GPUs (stride)
    size_t          beta,          // total digits (loop end)
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

        // STRIDED ownership: walk d = gpu_id, gpu_id+n_gpus, ..., < beta.
        // Both c2 and evks are indexed at the GLOBAL digit d.
        bool first = true;
        for (size_t d = gpu_id; d < beta; d += n_gpus) {
            if (!first && reduction_threshold == 0) {
                acc0.lo = barrett_reduce_uint128_uint64(acc0, mod.value(), mod.const_ratio());
                acc0.hi = 0;
                acc1.lo = barrett_reduce_uint128_uint64(acc1, mod.value(), mod.const_ratio());
                acc1.hi = 0;
            }
            first = false;

            const uint64_t c2_val = c2[c2_id + d * size_QlP_n];

            prod0 = multiply_uint64_uint64(c2_val, evks[d][evk_id]);
            add_uint128_uint128(acc0, prod0, acc0);

            prod1 = multiply_uint64_uint64(c2_val, evks[d][evk_id + size_QP_n]);
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
    // T-OVERLAP: inert — see docs/HPC_PRIMER.md §6.3 for null-result analysis.
    // The event-record below is harmless plumbing kept for future rotation-pipelining
    // experiments; deployed Phase 4b code does not wait on this event because
    // rotation N+1 reads rotation N's output, leaving no slack to overlap.
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

    // ---- Step 1: mod-up (FULL — replicated on every GPU) ----
    // (T-MODUP-FIX-2 2026-05-10): Reverted modup_partial → full rns_tool.modup
    // because the partial code path indexed DKS shards by chain-level beta but
    // the shards were sized at dnum (= max-chain beta). At lower chain levels
    // chain_beta < dnum, so trailing GPUs' contiguous shards covered the wrong
    // global digits and the partial KS kernel dereferenced nullptr evks slots.
    // STRIDED ownership + FULL modup keeps each GPU's evks coverage of any
    // prefix [0, chain_beta) intact, at the cost of replicating the modup work
    // (which was the T-MODUP win). Net effect on Phase 4b: bootstrap is the
    // same as the May-7 working state (~2,098 ms).
    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    // Number of strided digits owned by THIS GPU at this chain level.
    // (beta - gpu_id + n_gpus - 1) / n_gpus, clamped to ≥ 0.
    size_t d_count = (beta > static_cast<size_t>(gpu_id))
        ? (beta - static_cast<size_t>(gpu_id) + static_cast<size_t>(n_gpus) - 1)
              / static_cast<size_t>(n_gpus)
        : 0;

    // Allocate FULL t_mod_up (sized for all beta digits). At N=65,536, beta=20:
    // ~220 MB per GPU. Fits comfortably on 64 GB H100s.
    auto t_mod_up = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);
    {
        NVTX_SCOPE("modup");
        rns_tool.modup(t_mod_up.get(), c2, phantom_ctx.gpu_rns_tables(),
                       key_parms.scheme(), s);
    }

    // ---- Step 2: partial inner product (STRIDED owned digits) ----
    auto cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);

    // Defensive zero-init of cx: even though the kernel writes every output
    // position when d_count > 0, an explicit memset guards against the
    // d_count == 0 path (small chain levels where beta < n_gpus) where the
    // kernel launch is skipped and this GPU contributes a zero partial sum.
    CUDA_CHECK(cudaMemsetAsync(cx.get(), 0,
                               2 * size_QlP_n * sizeof(uint64_t), s));

    auto reduction_threshold =
        (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;

    // Use custom evks shard if provided (DKS), otherwise use relin_keys.
    // Both are global beta-sized arrays; the kernel only accesses owned (non-null)
    // entries via the strided pattern.
    const uint64_t *const *evks_ptr = custom_evks
        ? const_cast<const uint64_t *const *>(custom_evks)
        : relin_keys.public_keys_ptr();

    // Kernel walks d = gpu_id, gpu_id+n_gpus, ... < beta over the FULL t_mod_up
    // (replicated on every GPU) and indexes evks at GLOBAL d (always owned).
    // Skip kernel launch entirely when d_count == 0 (β < n_gpus and this GPU
    // owns nothing — happens at the deepest chain levels of bootstrap).
    if (d_count > 0)
    partial_key_switch_inner_prod<<<size_QlP_n / blockDimGlb.x, blockDimGlb, 0, s>>>(
        cx.get(), t_mod_up.get(), evks_ptr,
        modulus_QP, n, size_QP, size_QP * n,
        size_QlP, size_QlP_n, size_Q, size_Ql,
        static_cast<size_t>(gpu_id), static_cast<size_t>(n_gpus), beta,
        reduction_threshold);

    // ---- T-STRAGGLER: GPU-side barrier before AllReduce (see DKS variant) ----
    // T-OVERLAP: inert — see docs/HPC_PRIMER.md §6.3 for null-result analysis.
    // The ready_events barrier was meant to eliminate the ~530 ms straggler-wait
    // bucket inside ncclAllReduce. Finer NVTX measurement reclassified that bucket
    // as AllReduce kernel time (not host-side jitter), so this barrier currently
    // serializes a non-existent skew. Plumbing left in place for future tracing.
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

    // T-OVERLAP: inert — see docs/HPC_PRIMER.md §6.3 for null-result analysis.
    // The oa_done_events handoff was reverted (cross-stream wait at writeback was
    // buggy at some chain levels); the deployed path falls back to the safe host
    // sync below. Event record kept for symmetry with the DKS variant.
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

    // ---- Step 1: mod-up (FULL — replicated on every GPU) ----
    // (T-MODUP-FIX-2 2026-05-10): see non-DKS variant comment for the full
    // rationale. STRIDED ownership at full key beta means modup must run over
    // all beta digits because the partial KS kernel walks strided indices that
    // cover the entire [0, beta) range across all GPUs.
    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    size_t d_count = (beta > static_cast<size_t>(gpu_id))
        ? (beta - static_cast<size_t>(gpu_id) + static_cast<size_t>(n_gpus) - 1)
              / static_cast<size_t>(n_gpus)
        : 0;

    auto t_mod_up = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s);
    {
        NVTX_SCOPE("modup");
        rns_tool.modup(t_mod_up.get(), c2, phantom_ctx.gpu_rns_tables(),
                       key_parms.scheme(), s);
    }

    // ---- Step 2: partial inner product (STRIDED owned digits) ----
    auto cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, s);
    // Defensive zero-init: covers the d_count == 0 case (small chain levels
    // where beta < n_gpus and this GPU contributes nothing) and gives
    // ncclAllReduce a trivially valid input even if the kernel is skipped.
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
        static_cast<size_t>(gpu_id), static_cast<size_t>(n_gpus), beta,
        reduction_threshold);

    // ---- T-STRAGGLER: GPU-side barrier before AllReduce ----
    // T-OVERLAP: inert — see docs/HPC_PRIMER.md §6.3 for null-result analysis.
    // The "~530 ms straggler wait" this barrier targeted turned out to be
    // ncclAllReduce kernel time itself (NVTX re-measurement on 2026-04-19), not
    // host-side launch jitter. The barrier therefore eliminates a non-existent
    // skew. Plumbing kept correct and harmless; the published null result is
    // documented in the paper rather than masked by removing the code.
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

    // T-OVERLAP: inert — see docs/HPC_PRIMER.md §6.3 for null-result analysis.
    // The oa_done_events handoff was meant to let stream0 (rotation writeback)
    // gate on this event without blocking the worker CPU thread. Deployed Phase
    // 4b does not exploit it: the worker thread joins after each rotation, so
    // the alleged CPU-thread savings have no downstream consumer. The trailing
    // host sync (else branch) remains the actual ordering primitive.
    if (gpu_id < (int)ctx.oa_done_events.size() && ctx.oa_done_events[gpu_id]) {
        CUDA_CHECK(cudaEventRecord(ctx.oa_done_events[gpu_id], s));
    } else {
        CUDA_CHECK(cudaStreamSynchronize(s));
    }
}

} // namespace nexus_multi_gpu
