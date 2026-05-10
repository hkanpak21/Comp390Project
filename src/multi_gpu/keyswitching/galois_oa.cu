/**
 * galois_oa.cu
 *
 * Distributed Galois rotation via Output Aggregation.
 *
 * Pipeline for dist_rotate_output_aggregation:
 *
 *  Phase 1 (GPU 0):
 *    - Gather distributed CT to a full PhantomCiphertext on GPU 0
 *    - Apply Galois permutation to c0 and c1:
 *        c0_gal ← apply_galois_ntt(c0, elt_idx)
 *        c2_gal ← apply_galois_ntt(c1, elt_idx)
 *
 *  Phase 2 (all GPUs):
 *    - Broadcast c2_gal from GPU 0 to all other GPUs (full polynomial, ~17 MB)
 *    - Broadcast c0_gal similarly
 *
 *  Phase 3 (all GPUs in parallel threads, NCCL-synchronized):
 *    For each GPU g:
 *      a. Create a local PhantomCiphertext with c0_gal in slot[0], zeros in slot[1]
 *      b. Call keyswitching_output_aggregation(c2_gal, custom_evks=key_store.get_evks(g,key_idx))
 *         This does: modup(c2_gal) → partial KS (local digits) → AllReduce → mod-down → add to ct
 *
 *  Phase 4:
 *    - Scatter the resulting full CT back to DistributedCiphertext
 *
 * Communication:
 *   - 2 × peer-copies of ~17 MB each (broadcast c2_gal, c0_gal): ~0.3 ms on NVLink
 *   - 1 × ncclAllReduce of 2 × size_QlP × N uint64_t ≈ 32 MB: ~0.07 ms on NVLink
 *
 * This replaces the "gather-operate-scatter" placeholder in distributed_eval.cu.
 */

#include "galois_oa.cuh"
#include "output_aggregation.cuh"
#include "../partition/rns_partition.cuh"

#include "evaluate.cuh"
#include "context.cuh"
#include "ciphertext.h"
#include "galois.cuh"
#include "util/globals.h"
#include "nvtx_tracer.cuh"

#include <thread>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstring>

#define GAL_CUDA_CHECK(cmd) do {                                           \
    cudaError_t _e = (cmd);                                               \
    if (_e != cudaSuccess) {                                              \
        fprintf(stderr, "[galois_oa] CUDA error %s:%d: %s\n",           \
                __FILE__, __LINE__, cudaGetErrorString(_e));              \
        throw std::runtime_error(cudaGetErrorString(_e));                 \
    }                                                                     \
} while(0)

namespace nexus_multi_gpu {

// ---------------------------------------------------------------------------
// Broadcast helper: copy a device buffer from src_gpu to all other GPUs
// Returns per-GPU device pointers (index 0 = src_gpu's original pointer).
// Caller is responsible for freeing all returned pointers (except index src_gpu).
// ---------------------------------------------------------------------------
static std::vector<uint64_t*> broadcast_buffer(
    int src_gpu, int n_gpus,
    const uint64_t *src_ptr, size_t n_bytes)
{
    std::vector<uint64_t*> ptrs(n_gpus, nullptr);
    ptrs[src_gpu] = const_cast<uint64_t*>(src_ptr);  // no copy for src

    for (int g = 0; g < n_gpus; g++) {
        if (g == src_gpu) continue;
        GAL_CUDA_CHECK(cudaSetDevice(g));
        GAL_CUDA_CHECK(cudaMalloc(&ptrs[g], n_bytes));
        GAL_CUDA_CHECK(cudaMemcpyPeer(ptrs[g], g, src_ptr, src_gpu, n_bytes));
    }
    GAL_CUDA_CHECK(cudaSetDevice(0));
    return ptrs;
}

// ---------------------------------------------------------------------------
// dist_rotate_output_aggregation
// ---------------------------------------------------------------------------

void dist_rotate_output_aggregation(
    DistributedContext      &ctx,
    DistributedCiphertext   &dct,
    int                      steps,
    const DistGaloisKeyStore &key_store,
    size_t                   key_idx)
{
    const int n_gpus = ctx.n_gpus();

    // =========================================================
    // Phase 1: Gather and apply Galois permutation on GPU 0
    // =========================================================
    GAL_CUDA_CHECK(cudaSetDevice(0));
    const auto &stream0 = phantom::util::global_variables::default_stream->get_stream();

    // Gather distributed CT to GPU 0
    PhantomCiphertext ct_full = dct.to_single_gpu(ctx, 0);

    auto &pctx0 = ctx.context(0);
    auto *gtool  = pctx0.key_galois_tool();
    if (!gtool)
        throw std::runtime_error("[dist_rotate_oa] Galois tool not initialized on GPU 0");

    // Map step → galois element → index
    uint32_t galois_elt = gtool->get_elt_from_step(steps);
    const auto &gelts   = gtool->galois_elts();
    auto it = std::find(gelts.begin(), gelts.end(), galois_elt);
    if (it == gelts.end())
        throw std::runtime_error("[dist_rotate_oa] Galois element not found — key not registered");
    size_t gelt_idx = static_cast<size_t>(std::distance(gelts.begin(), it));

    size_t coeff_mod_size = ct_full.coeff_modulus_size();
    size_t N              = ct_full.poly_modulus_degree();
    size_t poly_bytes     = coeff_mod_size * N * sizeof(uint64_t);

    // Allocate temp buffer for permutation output (used for both c0 and c2)
    uint64_t *c0_gal_dev = nullptr;
    uint64_t *c2_gal_dev = nullptr;
    GAL_CUDA_CHECK(cudaMalloc(&c0_gal_dev, poly_bytes));
    GAL_CUDA_CHECK(cudaMalloc(&c2_gal_dev, poly_bytes));

    // Apply Galois permutation to c0 → c0_gal
    gtool->apply_galois_ntt(ct_full.data(), coeff_mod_size,
                             gelt_idx, c0_gal_dev, stream0);

    // Apply Galois permutation to c1 → c2_gal (will be key-switched)
    gtool->apply_galois_ntt(ct_full.data() + coeff_mod_size * N, coeff_mod_size,
                             gelt_idx, c2_gal_dev, stream0);

    GAL_CUDA_CHECK(cudaStreamSynchronize(stream0));

    // =========================================================
    // Phase 2: Broadcast c0_gal and c2_gal to all GPUs
    // =========================================================
    // Each GPU will need:
    //   - c2_gal (full) for modup in partial KS
    //   - c0_gal (full) to add correction onto

    auto c0_gal_all = broadcast_buffer(0, n_gpus, c0_gal_dev, poly_bytes);
    auto c2_gal_all = broadcast_buffer(0, n_gpus, c2_gal_dev, poly_bytes);

    // Build temporary PhantomCiphertext containers on each GPU
    // Each has [c0_gal, zeros] — KS will add correction in-place
    std::vector<PhantomCiphertext> local_cts(n_gpus);
    for (int g = 0; g < n_gpus; g++) {
        GAL_CUDA_CHECK(cudaSetDevice(g));
        // Resize to (coeff_mod_size, N, 2 polys) matching the full CT
        local_cts[g].resize(ctx.context(g), ct_full.chain_index(), 2,
                            g == 0 ? stream0 : cudaStreamPerThread);
        local_cts[g].set_scale(ct_full.scale());
        local_cts[g].set_ntt_form(true);

        size_t lbytes = 2 * poly_bytes;
        // Zero out slot [1] (c1), copy c0_gal into slot [0]
        GAL_CUDA_CHECK(cudaMemset(local_cts[g].data(), 0, lbytes));
        GAL_CUDA_CHECK(cudaMemcpy(local_cts[g].data(), c0_gal_all[g],
                                  poly_bytes, cudaMemcpyDeviceToDevice));
    }

    // Sync all GPUs before NCCL
    for (int g = 0; g < n_gpus; g++) {
        GAL_CUDA_CHECK(cudaSetDevice(g));
        GAL_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // =========================================================
    // Phase 3: Parallel partial KS + NCCL AllReduce + mod-down
    // =========================================================
    // Build MultiGpuContext from DistributedContext for output_aggregation
    MultiGpuContext mgctx;
    mgctx.n_gpus = n_gpus;
    mgctx.device_ids.resize(n_gpus);
    mgctx.comms.resize(n_gpus);
    mgctx.streams.resize(n_gpus);
    // T-STRAGGLER / T-OVERLAP: shallow-copy persistent events (see inplace variant).
    mgctx.ready_events          = ctx.ready_events();
    mgctx.allreduce_done_events = ctx.allreduce_done_events();
    mgctx.oa_done_events        = ctx.oa_done_events();
    for (int g = 0; g < n_gpus; g++) {
        mgctx.device_ids[g] = g;
        mgctx.comms[g]      = ctx.comm(g);
        mgctx.streams[g]    = ctx.stream(g);
    }

    // All GPU threads must call NCCL together
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errs(n_gpus, nullptr);

    for (int g = 0; g < n_gpus; g++) {
        threads.emplace_back([&, g]() {
            try {
                GAL_CUDA_CHECK(cudaSetDevice(g));

                // keyswitching_output_aggregation_dks adds correction directly
                // into local_cts[g] using the per-GPU key shard evks array.
                keyswitching_output_aggregation_dks(
                    mgctx,
                    ctx.context(g),
                    g,
                    local_cts[g],
                    c2_gal_all[g],
                    key_store.get_evks(g, key_idx),
                    n_gpus
                );
            } catch (...) {
                errs[g] = std::current_exception();
            }
        });
    }

    for (auto &t : threads) t.join();

    // Re-throw any thread exceptions
    for (int g = 0; g < n_gpus; g++) {
        if (errs[g]) std::rethrow_exception(errs[g]);
    }

    // =========================================================
    // Phase 4: Scatter result back to distributed ciphertext
    // After keyswitching_output_aggregation, all local_cts are identical
    // (AllReduce guarantees each GPU has the same result).
    // Scatter GPU 0's result to the DistributedCiphertext.
    // =========================================================
    // T-OVERLAP: keyswitching_output_aggregation_dks no longer host-syncs.
    // from_single_gpu() issues blocking cudaMemcpyPeer on the default stream
    // (NOT ctx.stream(0)), so we must wait on GPU 0's OA event before reading
    // local_cts[0].data(). Host-side wait is acceptable here — this is the
    // slow scatter path, not the inplace fast path.
    {
        const auto &oa_evts = ctx.oa_done_events();
        if (!oa_evts.empty() && oa_evts[0]) {
            GAL_CUDA_CHECK(cudaSetDevice(0));
            GAL_CUDA_CHECK(cudaEventSynchronize(oa_evts[0]));
        }
    }
    dct.free_all();
    dct = DistributedCiphertext::from_single_gpu(ctx, local_cts[0], 0);

    // =========================================================
    // Cleanup
    // =========================================================
    for (int g = 0; g < n_gpus; g++) {
        if (g != 0 && c0_gal_all[g]) {
            GAL_CUDA_CHECK(cudaSetDevice(g));
            GAL_CUDA_CHECK(cudaFree(c0_gal_all[g]));
        }
        if (g != 0 && c2_gal_all[g]) {
            GAL_CUDA_CHECK(cudaSetDevice(g));
            GAL_CUDA_CHECK(cudaFree(c2_gal_all[g]));
        }
    }
    GAL_CUDA_CHECK(cudaSetDevice(0));
    GAL_CUDA_CHECK(cudaFree(c0_gal_dev));
    GAL_CUDA_CHECK(cudaFree(c2_gal_dev));
}

// ---------------------------------------------------------------------------
// dist_rotate_phantom_inplace
// ---------------------------------------------------------------------------
// Same OA algorithm as dist_rotate_output_aggregation but skips the DCT
// scatter/gather on input/output. Used by the Bootstrapper (which holds a
// PhantomCiphertext on GPU 0). Saves ~10–30 ms per rotation versus wrapping
// in a temporary DistributedCiphertext.

void dist_rotate_phantom_inplace(
    DistributedContext       &ctx,
    PhantomCiphertext        &ct,
    int                       steps,
    const DistGaloisKeyStore &key_store,
    size_t                    key_idx)
{
    NVTX_SCOPE_FMT("dist_rotate_phantom step=%d", steps);
    const int n_gpus = ctx.n_gpus();

    GAL_CUDA_CHECK(cudaSetDevice(0));
    const auto &stream0 = phantom::util::global_variables::default_stream->get_stream();

    auto &pctx0 = ctx.context(0);
    auto *gtool = pctx0.key_galois_tool();
    if (!gtool)
        throw std::runtime_error("[dist_rotate_phantom] Galois tool not initialized on GPU 0");

    uint32_t galois_elt = gtool->get_elt_from_step(steps);
    const auto &gelts = gtool->galois_elts();
    auto it = std::find(gelts.begin(), gelts.end(), galois_elt);
    if (it == gelts.end())
        throw std::runtime_error("[dist_rotate_phantom] Galois element not found");
    size_t gelt_idx = static_cast<size_t>(std::distance(gelts.begin(), it));

    size_t coeff_mod_size = ct.coeff_modulus_size();
    size_t N              = ct.poly_modulus_degree();
    size_t poly_bytes     = coeff_mod_size * N * sizeof(uint64_t);

    // Phase 3 fast path: persistent per-GPU c0_gal/c2_gal buffers, no per-call cudaMalloc.
    ctx.ensure_rotation_workspace(poly_bytes);
    auto &ws = ctx.rotation_workspace();

    // Apply Galois permutation directly into GPU 0's persistent slot.
    {
        NVTX_SCOPE("P1_galois_ntt");
        uint64_t *c0_gal_dev = ws.c0_gal[0];
        uint64_t *c2_gal_dev = ws.c2_gal[0];
        gtool->apply_galois_ntt(ct.data(), coeff_mod_size, gelt_idx, c0_gal_dev, stream0);
        gtool->apply_galois_ntt(ct.data() + coeff_mod_size * N, coeff_mod_size,
                                 gelt_idx, c2_gal_dev, stream0);
        GAL_CUDA_CHECK(cudaStreamSynchronize(stream0));
    }

    // Peer-broadcast into the persistent workspace on each remote GPU
    {
        NVTX_SCOPE("P2_peer_broadcast");
        uint64_t *c0_gal_dev = ws.c0_gal[0];
        uint64_t *c2_gal_dev = ws.c2_gal[0];
        for (int g = 1; g < n_gpus; g++) {
            GAL_CUDA_CHECK(cudaSetDevice(g));
            GAL_CUDA_CHECK(cudaMemcpyPeerAsync(ws.c0_gal[g], g, c0_gal_dev, 0, poly_bytes, ctx.stream(g)));
            GAL_CUDA_CHECK(cudaMemcpyPeerAsync(ws.c2_gal[g], g, c2_gal_dev, 0, poly_bytes, ctx.stream(g)));
        }
        for (int g = 1; g < n_gpus; g++) {
            GAL_CUDA_CHECK(cudaSetDevice(g));
            GAL_CUDA_CHECK(cudaStreamSynchronize(ctx.stream(g)));
        }
    }

    // Phase 4a: persistent per-GPU local_cts. Reuse across rotations; resize
    // only when chain_index changes (rare during a single bootstrap).
    if ((int)ws.local_cts.size() < n_gpus) {
        ws.local_cts.resize(n_gpus);
        ws.local_chain_index.resize(n_gpus, 0);
    }
    auto &local_cts = ws.local_cts;  // reference into the workspace
    const size_t target_chain = ct.chain_index();
    for (int g = 0; g < n_gpus; g++) {
        GAL_CUDA_CHECK(cudaSetDevice(g));
        if (ws.local_chain_index[g] != target_chain || local_cts[g].coeff_modulus_size() == 0) {
            local_cts[g].resize(ctx.context(g), target_chain, 2,
                                g == 0 ? stream0 : cudaStreamPerThread);
            ws.local_chain_index[g] = target_chain;
        }
        local_cts[g].set_scale(ct.scale());
        local_cts[g].set_ntt_form(true);

        size_t lbytes = 2 * poly_bytes;
        GAL_CUDA_CHECK(cudaMemset(local_cts[g].data(), 0, lbytes));
        GAL_CUDA_CHECK(cudaMemcpy(local_cts[g].data(), ws.c0_gal[g],
                                  poly_bytes, cudaMemcpyDeviceToDevice));
    }
    for (int g = 0; g < n_gpus; g++) {
        GAL_CUDA_CHECK(cudaSetDevice(g));
        GAL_CUDA_CHECK(cudaDeviceSynchronize());
    }

    MultiGpuContext mgctx;
    mgctx.n_gpus = n_gpus;
    mgctx.device_ids.resize(n_gpus);
    mgctx.comms.resize(n_gpus);
    mgctx.streams.resize(n_gpus);
    // T-STRAGGLER / T-OVERLAP: shallow-copy the persistent per-GPU events from
    // DistributedContext so keyswitching_output_aggregation_dks can use them
    // as GPU-side barriers around ncclAllReduce, and oa_done_events lets the
    // writeback memcpy below wait on OA completion without a host sync.
    mgctx.ready_events          = ctx.ready_events();
    mgctx.allreduce_done_events = ctx.allreduce_done_events();
    mgctx.oa_done_events        = ctx.oa_done_events();
    for (int g = 0; g < n_gpus; g++) {
        mgctx.device_ids[g] = g;
        mgctx.comms[g]      = ctx.comm(g);
        mgctx.streams[g]    = ctx.stream(g);
    }

    // Phase 4b: dispatch to persistent worker threads in DistributedContext —
    // no per-rotation std::thread spawn/join.
    {
        NVTX_SCOPE("P3_dispatch_partialKS");
        std::vector<std::function<void()>> work(n_gpus);
        for (int g = 0; g < n_gpus; g++) {
            work[g] = [&, g]() {
                NVTX_SCOPE_FMT("partialKS gpu=%d", g);
                keyswitching_output_aggregation_dks(
                    mgctx, ctx.context(g), g,
                    local_cts[g], ws.c2_gal[g],
                    key_store.get_evks(g, key_idx),
                    n_gpus);
            };
        }
        ctx.dispatch_to_all_gpus(work);
    }

    // Phase 4 — write the result back into ct (GPU 0 D2D, no DCT scatter)
    {
        NVTX_SCOPE("P4_writeback");
        GAL_CUDA_CHECK(cudaSetDevice(0));
        // T-OVERLAP: keyswitching_output_aggregation_dks no longer host-syncs
        // before returning. Its end is signalled via ctx.oa_done_events()[g]
        // recorded on ctx.stream(g). The writeback runs on stream0 (a different
        // stream from ctx.stream(0)), so we must gate it with a GPU-side wait
        // on GPU 0's OA event before queuing the memcpy. Order MUST be:
        //   cudaStreamWaitEvent(stream0, oa_done_events[0])  THEN  cudaMemcpyAsync(stream0).
        const auto &oa_evts = ctx.oa_done_events();
        if (!oa_evts.empty() && oa_evts[0]) {
            GAL_CUDA_CHECK(cudaStreamWaitEvent(stream0, oa_evts[0], 0));
        }
        GAL_CUDA_CHECK(cudaMemcpyAsync(ct.data(), local_cts[0].data(),
                                       2 * poly_bytes, cudaMemcpyDeviceToDevice, stream0));
        GAL_CUDA_CHECK(cudaStreamSynchronize(stream0));
    }
}

// ---------------------------------------------------------------------------
// dist_relinearize_output_aggregation (placeholder — relin DKS)
// ---------------------------------------------------------------------------

void dist_relinearize_output_aggregation(
    DistributedContext    &ctx,
    DistributedCiphertext &dct,
    uint64_t             **relin_evks[],
    size_t                 beta)
{
    // TODO: implement when relin key sharding is added
    // For now, fall back to gather-operate-scatter
    (void)relin_evks;
    (void)beta;

    // Gather to GPU 0, relinearize there, scatter back
    GAL_CUDA_CHECK(cudaSetDevice(0));
    PhantomCiphertext ct = dct.to_single_gpu(ctx, 0);
    // Note: caller must provide relin keys separately for GPU 0
    // This is a stub — actual relin DKS left as future work
    dct.free_all();
    dct = DistributedCiphertext::from_single_gpu(ctx, ct, 0);
}

} // namespace nexus_multi_gpu
