/**
 * nccl_comm.cu
 *
 * Implementation of NCCL collective wrappers for RNS-CKKS ciphertext distribution.
 *
 * All NCCL calls use ncclUint64 since ciphertext limb data is uint64_t.
 * All functions are non-blocking with respect to the host — they enqueue work
 * into CUDA streams and return immediately.
 */

#include "nccl_comm.cuh"

#include <stdexcept>
#include <string>
#include <cstring>

#define NCCL_CHECK(cmd) do {                                              \
    ncclResult_t r = (cmd);                                               \
    if (r != ncclSuccess) {                                               \
        throw std::runtime_error(std::string("NCCL error: ") +           \
                                 ncclGetErrorString(r) +                  \
                                 " at " __FILE__ ":" + std::to_string(__LINE__)); \
    }                                                                     \
} while (0)

#define CUDA_CHECK(cmd) do {                                              \
    cudaError_t e = (cmd);                                                \
    if (e != cudaSuccess) {                                               \
        throw std::runtime_error(std::string("CUDA error: ") +           \
                                 cudaGetErrorString(e) +                  \
                                 " at " __FILE__ ":" + std::to_string(__LINE__)); \
    }                                                                     \
} while (0)

namespace nexus_multi_gpu {

// ---------------------------------------------------------------------------
// MultiGpuContext
// ---------------------------------------------------------------------------

MultiGpuContext MultiGpuContext::create(const std::vector<int> &dev_ids) {
    MultiGpuContext ctx;
    ctx.n_gpus      = static_cast<int>(dev_ids.size());
    ctx.device_ids  = dev_ids;
    ctx.comms.resize(ctx.n_gpus);
    ctx.streams.resize(ctx.n_gpus);
    ctx.ready_events.resize(ctx.n_gpus);
    ctx.allreduce_done_events.resize(ctx.n_gpus);
    ctx.oa_done_events.resize(ctx.n_gpus);

    // ncclCommInitAll initializes one communicator per device in a single call.
    NCCL_CHECK(ncclCommInitAll(ctx.comms.data(), ctx.n_gpus, dev_ids.data()));

    // Create one CUDA stream per device for NCCL operations.
    for (int g = 0; g < ctx.n_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(dev_ids[g]));
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.streams[g], cudaStreamNonBlocking));
        // T-STRAGGLER / T-OVERLAP: per-GPU events used as GPU-side barriers around
        // ncclAllReduce. cudaEventDisableTiming reduces overhead (we never query elapsed time).
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.ready_events[g], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.allreduce_done_events[g], cudaEventDisableTiming));
        // T-OVERLAP: end-of-OA event used as cross-stream signal to the writeback
        // memcpy on stream0 in galois_oa.cu (replaces trailing cudaStreamSynchronize).
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.oa_done_events[g], cudaEventDisableTiming));
    }

    return ctx;
}

void MultiGpuContext::destroy() {
    // FIX-BUG-03-01: drain pending work on every GPU *before* tearing down
    // streams or comms. Without this, kernels enqueued on `streams[g]`
    // (including the AllReduce NCCL operates on) may still be in flight
    // when cudaStreamDestroy / ncclCommDestroy run — undefined behaviour
    // ranging from silent leaks to segfaults during shutdown.
    // BUG-03 audit HIGH-6.
    for (int g = 0; g < n_gpus; ++g) {
        cudaSetDevice(device_ids[g]);
        cudaDeviceSynchronize();
    }

    for (int g = 0; g < n_gpus; ++g) {
        cudaSetDevice(device_ids[g]);
        if (g < (int)ready_events.size() && ready_events[g]) {
            cudaEventDestroy(ready_events[g]);
            ready_events[g] = nullptr;
        }
        if (g < (int)allreduce_done_events.size() && allreduce_done_events[g]) {
            cudaEventDestroy(allreduce_done_events[g]);
            allreduce_done_events[g] = nullptr;
        }
        if (g < (int)oa_done_events.size() && oa_done_events[g]) {
            cudaEventDestroy(oa_done_events[g]);
            oa_done_events[g] = nullptr;
        }
        // FIX-BUG-03-01: NCCL holds references to the stream it was
        // created/used on. Destroy the comm BEFORE the stream so the comm
        // releases its handle while the stream is still valid; reversing
        // the order leaves NCCL with a dangling stream reference at
        // ncclCommDestroy time.
        ncclCommDestroy(comms[g]);
        cudaStreamDestroy(streams[g]);
    }
    ready_events.clear();
    allreduce_done_events.clear();
    oa_done_events.clear();
    comms.clear();
    streams.clear();
}

// ---------------------------------------------------------------------------
// LimbRange helpers
// ---------------------------------------------------------------------------

LimbRange get_limb_range(int gpu_id, int n_gpus, size_t total_limbs) {
    LimbRange r;
    r.start_limb = static_cast<size_t>(gpu_id);
    // Count limbs: those with index g, g+n, g+2n, ... up to total_limbs-1
    if (static_cast<size_t>(gpu_id) >= total_limbs) {
        r.n_local = 0;
    } else {
        r.n_local = (total_limbs - static_cast<size_t>(gpu_id) - 1)
                    / static_cast<size_t>(n_gpus) + 1;
    }
    return r;
}

// ---------------------------------------------------------------------------
// scatter_limbs_to_gpu / gather_limbs_from_gpu
// ---------------------------------------------------------------------------
// These copy non-contiguous limb slices between devices using peer access.
// For each polynomial and each local limb owned by dst_gpu, we do one
// cudaMemcpyPeerAsync call covering `degree` uint64 elements.

void scatter_limbs_to_gpu(int src_gpu, int dst_gpu,
                          const uint64_t *src_data,
                          uint64_t       *dst_data,
                          size_t n_polys,
                          size_t total_limbs,
                          size_t degree,
                          int n_gpus,
                          cudaStream_t stream) {
    size_t dst_local_limbs = (static_cast<size_t>(dst_gpu) < total_limbs)
        ? (total_limbs - static_cast<size_t>(dst_gpu) - 1) / static_cast<size_t>(n_gpus) + 1
        : 0;

    size_t local_loc = 0;
    for (size_t j = static_cast<size_t>(dst_gpu); j < total_limbs;
         j += static_cast<size_t>(n_gpus), ++local_loc) {
        for (size_t poly = 0; poly < n_polys; ++poly) {
            const uint64_t *src_ptr =
                src_data + (poly * total_limbs + j) * degree;
            uint64_t *dst_ptr =
                dst_data + (poly * dst_local_limbs + local_loc) * degree;
            CUDA_CHECK(cudaMemcpyPeerAsync(dst_ptr, dst_gpu,
                                           src_ptr, src_gpu,
                                           degree * sizeof(uint64_t), stream));
        }
    }
}

void gather_limbs_from_gpu(int src_gpu, int dst_gpu,
                           const uint64_t *src_data,
                           uint64_t       *dst_data,
                           size_t n_polys,
                           size_t total_limbs,
                           size_t degree,
                           int n_gpus,
                           cudaStream_t stream) {
    size_t src_local_limbs = (static_cast<size_t>(src_gpu) < total_limbs)
        ? (total_limbs - static_cast<size_t>(src_gpu) - 1) / static_cast<size_t>(n_gpus) + 1
        : 0;

    size_t local_loc = 0;
    for (size_t j = static_cast<size_t>(src_gpu); j < total_limbs;
         j += static_cast<size_t>(n_gpus), ++local_loc) {
        for (size_t poly = 0; poly < n_polys; ++poly) {
            const uint64_t *src_ptr =
                src_data + (poly * src_local_limbs + local_loc) * degree;
            uint64_t *dst_ptr =
                dst_data + (poly * total_limbs + j) * degree;
            CUDA_CHECK(cudaMemcpyPeerAsync(dst_ptr, dst_gpu,
                                           src_ptr, src_gpu,
                                           degree * sizeof(uint64_t), stream));
        }
    }
}

// ---------------------------------------------------------------------------
// allgather_ciphertext_limbs
// ---------------------------------------------------------------------------
// Uses ncclAllGather: each GPU sends its local_limbs * degree * n_polys words;
// every GPU receives all n_gpus * local_limbs words in order.
//
// NCCL AllGather requires the send buffer to be the "send_count" chunk of the
// receive buffer (in-place variant) OR separate src/dst buffers.  We use the
// separate-buffer form so the caller controls memory layout.

void allgather_ciphertext_limbs(MultiGpuContext &ctx,
                                int gpu_id,
                                const uint64_t *local_buf,
                                uint64_t       *recv_buf,
                                size_t n_polys,
                                size_t local_limbs,
                                size_t degree) {
    // NCCL AllGather requires ALL ranks to send the SAME count.
    // With cyclic limb assignment, some GPUs may have one more limb than others
    // when total_limbs % n_gpus != 0. We pad to max_local_limbs.
    size_t max_local = (local_limbs > 0) ? local_limbs : 0;
    // Compute max across all GPUs: max_local = ceil(total_limbs / n_gpus)
    // Since we don't have total_limbs here, we receive it from the caller.
    // For now, use local_limbs directly — the caller must ensure all GPUs
    // pass the SAME value for local_limbs (= max_local_limbs across all GPUs).
    size_t send_count = local_limbs * degree;

    NCCL_CHECK(ncclGroupStart());
    for (size_t poly = 0; poly < n_polys; ++poly) {
        const uint64_t *send = local_buf + poly * send_count;
        uint64_t       *recv = recv_buf  + poly * (static_cast<size_t>(ctx.n_gpus) * send_count);
        NCCL_CHECK(ncclAllGather(send, recv, send_count,
                                 ncclUint64,
                                 ctx.comms[gpu_id],
                                 ctx.streams[gpu_id]));
    }
    NCCL_CHECK(ncclGroupEnd());
}

// ---------------------------------------------------------------------------
// allreduce_partial_keyswitching
// ---------------------------------------------------------------------------

void allreduce_partial_keyswitching(MultiGpuContext &ctx,
                                    int gpu_id,
                                    uint64_t *partial_buf,
                                    size_t n_polys,
                                    size_t local_limbs,
                                    size_t degree) {
    size_t count = n_polys * local_limbs * degree;
    // In-place AllReduce: partial_buf serves as both send and receive buffer.
    NCCL_CHECK(ncclAllReduce(partial_buf, partial_buf, count,
                             ncclUint64, ncclSum,
                             ctx.comms[gpu_id],
                             ctx.streams[gpu_id]));
    // Note: ncclSum on uint64_t performs modular-free integer addition.
    // The caller must apply modular reduction after the AllReduce.
}

// ---------------------------------------------------------------------------
// broadcast_ciphertext
// ---------------------------------------------------------------------------

void broadcast_ciphertext(MultiGpuContext &ctx,
                          int gpu_id,
                          uint64_t *buf,
                          size_t n_polys,
                          size_t n_limbs,
                          size_t degree,
                          int root_gpu) {
    size_t count = n_polys * n_limbs * degree;
    NCCL_CHECK(ncclBroadcast(buf, buf, count,
                             ncclUint64, root_gpu,
                             ctx.comms[gpu_id],
                             ctx.streams[gpu_id]));
}

} // namespace nexus_multi_gpu
