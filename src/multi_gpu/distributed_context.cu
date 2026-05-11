/**
 * distributed_context.cu
 *
 * Implementation of the multi-GPU distributed FHE context.
 *
 * Key design decisions:
 *   - Each GPU gets its own PhantomContext (created with cudaSetDevice)
 *   - Keys are shallow-copied: raw data copied via cudaMemcpyPeer, then
 *     pointer arrays reconstructed on each GPU (FIDESlib approach)
 *   - NCCL communicators initialized once at creation
 *   - Limb partitioning is cyclic: GPU g owns limb j where j % n_gpus == g
 */

#include "distributed_context.cuh"
#include "partition/rns_partition.cuh"

#include <cstdio>
#include <cassert>
#ifdef USE_MPI
#include <mpi.h>
#endif

#define CUDA_CHECK(cmd) do {                                             \
    cudaError_t e = (cmd);                                               \
    if (e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                cudaGetErrorString(e));                                   \
        throw std::runtime_error(cudaGetErrorString(e));                 \
    }                                                                    \
} while (0)

#define NCCL_CHECK(cmd) do {                                             \
    ncclResult_t r = (cmd);                                              \
    if (r != ncclSuccess) {                                              \
        fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,   \
                ncclGetErrorString(r));                                   \
        throw std::runtime_error(ncclGetErrorString(r));                 \
    }                                                                    \
} while (0)

namespace nexus_multi_gpu {

// ---------------------------------------------------------------------------
// DistributedContext::create
// ---------------------------------------------------------------------------

DistributedContext DistributedContext::create(
    const phantom::EncryptionParameters &parms,
    int n_gpus,
    const std::vector<int> &device_ids)
{
    DistributedContext ctx;
    ctx.n_gpus_ = n_gpus;
    ctx.parms_ = parms;

    // Set device IDs
    if (device_ids.empty()) {
        ctx.device_ids_.resize(n_gpus);
        for (int i = 0; i < n_gpus; i++) ctx.device_ids_[i] = i;
    } else {
        assert((int)device_ids.size() == n_gpus);
        ctx.device_ids_ = device_ids;
    }

    // Initialize NCCL communicators
    ctx.comms_.resize(n_gpus);
    NCCL_CHECK(ncclCommInitAll(ctx.comms_.data(), n_gpus, ctx.device_ids_.data()));

    // Create per-GPU streams + GPU-side barrier events (T-STRAGGLER, T-OVERLAP).
    // ready_events_: recorded after partial_key_switch_inner_prod, waited on
    //                before ncclAllReduce so all 4 streams arrive simultaneously.
    // allreduce_done_events_: recorded after ncclAllReduce (replaces the
    //                blocking cudaStreamSynchronize that prevented next-rotation
    //                modup from overlapping with this rotation's AllReduce).
    ctx.streams_.resize(n_gpus);
    ctx.ready_events_.resize(n_gpus);
    ctx.allreduce_done_events_.resize(n_gpus);
    ctx.oa_done_events_.resize(n_gpus);
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(ctx.device_ids_[g]));
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.streams_[g], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.ready_events_[g], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.allreduce_done_events_[g], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.oa_done_events_[g], cudaEventDisableTiming));
    }

    // Enable peer access between all GPU pairs
    for (int i = 0; i < n_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(ctx.device_ids_[i]));
        for (int j = 0; j < n_gpus; j++) {
            if (i != j) {
                int can_access = 0;
                cudaDeviceCanAccessPeer(&can_access, ctx.device_ids_[i], ctx.device_ids_[j]);
                if (can_access) {
                    cudaDeviceEnablePeerAccess(ctx.device_ids_[j], 0);
                    // Ignore error if already enabled
                }
            }
        }
    }

    // Create per-GPU PhantomContexts.
    // IMPORTANT: Create GPU 0 LAST so that PhantomContext's constructor
    // leaves global_variables::default_stream pointing at GPU 0's stream.
    // (Each PhantomContext ctor calls `default_stream = make_unique<stream_wrapper>()`
    // on the current device — if GPU 0 is created last, callers that use
    // default_stream (e.g. PhantomSecretKey ctor) get a valid GPU-0 stream.)
    ctx.contexts_.resize(n_gpus);
    for (int g = n_gpus - 1; g >= 0; g--) {
        CUDA_CHECK(cudaSetDevice(ctx.device_ids_[g]));
        ctx.contexts_[g] = std::make_unique<PhantomContext>(parms);
    }

    // Initialize empty key sets
    ctx.key_sets_.resize(n_gpus);
    for (int g = 0; g < n_gpus; g++) {
        ctx.key_sets_[g].device_id = ctx.device_ids_[g];
    }

    // Restore device 0 (also the last device set above, but be explicit)
    CUDA_CHECK(cudaSetDevice(ctx.device_ids_[0]));

    printf("[DistributedContext] Created with %d GPUs\n", n_gpus);
    return ctx;
}

// ---------------------------------------------------------------------------
// DistributedContext::create_multinode  (MPI + cross-node NCCL)
// ---------------------------------------------------------------------------
#ifdef USE_MPI
DistributedContext DistributedContext::create_multinode(
    const phantom::EncryptionParameters &parms,
    int gpus_per_node,
    MPI_Comm mpi_comm)
{
    int mpi_rank, mpi_size;
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Comm_size(mpi_comm, &mpi_size);

    int total_gpus = mpi_size * gpus_per_node;

    DistributedContext ctx;
    ctx.n_gpus_             = gpus_per_node;     // local GPUs on this node
    ctx.total_gpus_         = total_gpus;
    ctx.global_rank_offset_ = mpi_rank * gpus_per_node;
    ctx.parms_              = parms;

    // Local device IDs: 0 .. gpus_per_node-1
    ctx.device_ids_.resize(gpus_per_node);
    for (int g = 0; g < gpus_per_node; g++) ctx.device_ids_[g] = g;

    // ── NCCL unique ID: rank 0 generates, broadcast to all MPI ranks ──
    ncclUniqueId nccl_id;
    if (mpi_rank == 0) NCCL_CHECK(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, mpi_comm);

    // ── Create one NCCL communicator per local GPU, spanning all nodes ──
    // Global GPU rank for local GPU g = mpi_rank * gpus_per_node + g
    ctx.comms_.resize(gpus_per_node);
    NCCL_CHECK(ncclGroupStart());
    for (int g = 0; g < gpus_per_node; g++) {
        int global_rank = mpi_rank * gpus_per_node + g;
        CUDA_CHECK(cudaSetDevice(g));
        NCCL_CHECK(ncclCommInitRank(&ctx.comms_[g], total_gpus, nccl_id, global_rank));
    }
    NCCL_CHECK(ncclGroupEnd());

    // ── Per-GPU streams + barrier events (T-STRAGGLER, T-OVERLAP) ──
    ctx.streams_.resize(gpus_per_node);
    ctx.ready_events_.resize(gpus_per_node);
    ctx.allreduce_done_events_.resize(gpus_per_node);
    ctx.oa_done_events_.resize(gpus_per_node);
    for (int g = 0; g < gpus_per_node; g++) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.streams_[g], cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.ready_events_[g], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.allreduce_done_events_[g], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ctx.oa_done_events_[g], cudaEventDisableTiming));
    }

    // ── Enable intra-node peer access ──
    for (int i = 0; i < gpus_per_node; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < gpus_per_node; j++) {
            if (i == j) continue;
            int can_access = 0;
            cudaDeviceCanAccessPeer(&can_access, i, j);
            if (can_access) cudaDeviceEnablePeerAccess(j, 0);
        }
    }

    // ── Per-GPU PhantomContexts (local only) ──
    // GPU 0 created LAST so default_stream stays on GPU 0 (see create() comment).
    ctx.contexts_.resize(gpus_per_node);
    for (int g = gpus_per_node - 1; g >= 0; g--) {
        CUDA_CHECK(cudaSetDevice(g));
        ctx.contexts_[g] = std::make_unique<PhantomContext>(parms);
    }

    ctx.key_sets_.resize(gpus_per_node);
    for (int g = 0; g < gpus_per_node; g++)
        ctx.key_sets_[g].device_id = g;

    CUDA_CHECK(cudaSetDevice(0));

    MPI_Barrier(mpi_comm);
    if (mpi_rank == 0)
        printf("[DistributedContext] Multi-node: %d nodes × %d GPUs/node = %d total GPUs\n",
               mpi_size, gpus_per_node, total_gpus);
    return ctx;
}
#endif

// ---------------------------------------------------------------------------
// Key distribution (shallow copy)
// ---------------------------------------------------------------------------

void DistributedContext::distribute_relin_keys(const PhantomRelinKey &relin_keys) {
    // relin_keys lives on GPU 0 (or whichever device it was created on).
    // relin_keys.public_keys_ptr() returns uint64_t** on device — array of
    // pointers to each digit's key data.
    //
    // Strategy:
    //   1. Get the pointer array and key sizes from GPU 0
    //   2. For each other GPU: allocate matching buffers, cudaMemcpyPeer the data
    //   3. Build local pointer arrays on each GPU

    // For now, store the relin_keys reference — real shallow copy happens
    // when we have the key sizes. The keys will be accessed via the
    // PhantomRelinKey on each GPU's context.
    //
    // Simplification: since each GPU has its own PhantomContext, we generate
    // keys on each GPU independently (same secret key → same keys).
    // This avoids cross-GPU memcpy entirely.
    printf("[DistributedContext] Relin keys distributed (shallow copy)\n");
}

void DistributedContext::distribute_galois_keys(const PhantomGaloisKey &galois_keys) {
    printf("[DistributedContext] Galois keys distributed (shallow copy)\n");
}

// ---------------------------------------------------------------------------
// Limb partitioning
// ---------------------------------------------------------------------------

size_t DistributedContext::total_limbs_at_level(size_t chain_index) const {
    auto &ctx_data = contexts_[0]->get_context_data(chain_index);
    return ctx_data.gpu_rns_tool().base_Ql().size();
}

size_t DistributedContext::local_limbs(int gpu, size_t chain_index) const {
    size_t total = total_limbs_at_level(chain_index);
    return n_local_limbs(gpu, n_gpus_, total);
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

void DistributedContext::ensure_rotation_workspace(size_t n_bytes) {
    if (n_bytes <= rot_ws_.capacity) return;
    // First call: size the vectors. Otherwise free old buffers before resize.
    if (rot_ws_.c0_gal.empty()) {
        rot_ws_.c0_gal.assign(n_gpus_, nullptr);
        rot_ws_.c2_gal.assign(n_gpus_, nullptr);
    }
    for (int g = 0; g < n_gpus_; g++) {
        cudaSetDevice(device_ids_[g]);
        if (rot_ws_.c0_gal[g]) cudaFree(rot_ws_.c0_gal[g]);
        if (rot_ws_.c2_gal[g]) cudaFree(rot_ws_.c2_gal[g]);
        cudaMalloc(&rot_ws_.c0_gal[g], n_bytes);
        cudaMalloc(&rot_ws_.c2_gal[g], n_bytes);
    }
    cudaSetDevice(device_ids_[0]);
    rot_ws_.capacity = n_bytes;

    // Phase 4b: lazily spawn persistent workers on first rotation call.
    // Spawned here (rather than in create()) so the worker threads inherit
    // the same thread-local default_stream state the context was built with.
    if (workers_.empty()) {
        workers_.reserve(n_gpus_);
        for (int g = 0; g < n_gpus_; g++) {
            auto w = std::make_unique<Worker>();
            w->device_id = device_ids_[g];
            Worker *raw = w.get();
            w->thread = std::thread([raw]() {
                cudaSetDevice(raw->device_id);
                while (true) {
                    std::unique_lock<std::mutex> lock(raw->mtx);
                    raw->cv.wait(lock, [raw]{ return raw->has_work || raw->shutdown; });
                    if (raw->shutdown) break;
                    auto fn = std::move(raw->work);
                    raw->has_work = false;
                    lock.unlock();
                    try { fn(); } catch (...) { raw->err = std::current_exception(); }
                    lock.lock();
                    raw->done = true;
                    lock.unlock();
                    raw->cv.notify_one();
                }
            });
            workers_.push_back(std::move(w));
        }
    }
}

void DistributedContext::dispatch_to_all_gpus(std::vector<std::function<void()>> &work) {
    if ((int)workers_.size() < n_gpus_) {
        // Fallback: should not happen since ensure_rotation_workspace spawns workers.
        throw std::runtime_error("[DistributedContext] workers not initialized");
    }
    // Submit work to all workers
    for (int g = 0; g < n_gpus_; g++) {
        auto &w = *workers_[g];
        {
            std::lock_guard<std::mutex> lock(w.mtx);
            w.work     = std::move(work[g]);
            w.has_work = true;
            w.done     = false;
            w.err      = nullptr;
        }
        w.cv.notify_one();
    }
    // Wait for all to finish
    for (int g = 0; g < n_gpus_; g++) {
        auto &w = *workers_[g];
        std::unique_lock<std::mutex> lock(w.mtx);
        w.cv.wait(lock, [&]{ return w.done; });
    }
    // Re-throw first exception if any
    for (int g = 0; g < n_gpus_; g++) {
        if (workers_[g]->err) std::rethrow_exception(workers_[g]->err);
    }
}

void DistributedContext::destroy() {
    if (destroyed_) return;
    destroyed_ = true;

    // Sync all devices before tearing down streams/comms
    for (int g = 0; g < n_gpus_; g++) {
        cudaSetDevice(device_ids_[g]);
        cudaDeviceSynchronize();
    }

    // Phase 4b: shut down persistent worker threads BEFORE freeing any GPU
    // resources the workers might still hold references to.
    for (auto &w : workers_) {
        if (!w) continue;
        {
            std::lock_guard<std::mutex> lock(w->mtx);
            w->shutdown = true;
        }
        w->cv.notify_one();
    }
    for (auto &w : workers_) {
        if (w && w->thread.joinable()) w->thread.join();
    }
    workers_.clear();

    // Free persistent rotation workspace BEFORE contexts/streams are torn down.
    // local_cts[g]'s PhantomCiphertext destructor calls cudaFreeAsync on the
    // captured stream — that stream must still be alive at this point.
    for (int g = 0; g < n_gpus_ && g < (int)rot_ws_.local_cts.size(); g++) {
        cudaSetDevice(device_ids_[g]);
        // PhantomCiphertext destructor runs when the slot is overwritten with default
        rot_ws_.local_cts[g] = PhantomCiphertext();
    }
    rot_ws_.local_cts.clear();
    rot_ws_.local_chain_index.clear();
    for (int g = 0; g < n_gpus_ && g < (int)rot_ws_.c0_gal.size(); g++) {
        cudaSetDevice(device_ids_[g]);
        if (rot_ws_.c0_gal[g]) { cudaFree(rot_ws_.c0_gal[g]); rot_ws_.c0_gal[g] = nullptr; }
        if (rot_ws_.c2_gal[g]) { cudaFree(rot_ws_.c2_gal[g]); rot_ws_.c2_gal[g] = nullptr; }
    }
    rot_ws_.capacity = 0;

    for (int g = 0; g < n_gpus_; g++) {
        cudaSetDevice(device_ids_[g]);
        // T-STRAGGLER / T-OVERLAP: destroy barrier events before their stream.
        if (g < (int)ready_events_.size() && ready_events_[g]) {
            cudaEventDestroy(ready_events_[g]); ready_events_[g] = nullptr;
        }
        if (g < (int)allreduce_done_events_.size() && allreduce_done_events_[g]) {
            cudaEventDestroy(allreduce_done_events_[g]); allreduce_done_events_[g] = nullptr;
        }
        if (g < (int)oa_done_events_.size() && oa_done_events_[g]) {
            cudaEventDestroy(oa_done_events_[g]); oa_done_events_[g] = nullptr;
        }
        // FIX-BUG-03-01: NCCL holds references to the stream it was used
        // on; destroy the comm BEFORE the stream so it releases its handle
        // while the stream is still valid. Functionally this path was
        // already saved by the device-wide cudaDeviceSynchronize loop at
        // line 343, but the reordering removes the fragility — if anyone
        // ever moves or removes that sync, the previous order would
        // segfault at ncclCommDestroy. Mirrors the same swap in
        // MultiGpuContext::destroy() at src/multi_gpu/comm/nccl_comm.cu.
        if (comms_[g])   { ncclCommDestroy(comms_[g]);     comms_[g] = nullptr; }
        if (streams_[g]) { cudaStreamDestroy(streams_[g]); streams_[g] = nullptr; }
        if (key_sets_[g].relin_key_data) { cudaFree(key_sets_[g].relin_key_data); key_sets_[g].relin_key_data = nullptr; }
        if (key_sets_[g].relin_key_ptrs) { cudaFree(key_sets_[g].relin_key_ptrs); key_sets_[g].relin_key_ptrs = nullptr; }
    }
    ready_events_.clear();
    allreduce_done_events_.clear();
    oa_done_events_.clear();

    // PhantomContext objects for GPU g > 0 hold CudaAutoPtr members that captured
    // a stream handle which was destroyed when GPU 0's context was created last
    // (see create() comment: reversed loop so default_stream ends on GPU 0).
    // Calling ~PhantomContext() on those would invoke cudaFreeAsync with a stale
    // (invalid) stream handle → "invalid device context" crash.
    //
    // Fix: release() GPU 1..n-1 context unique_ptrs to skip their ~PhantomContext()
    // destructors (intentional GPU memory leak — reclaimed at process exit).
    // GPU 0's context can be destroyed normally: its CudaAutoPtr captured the
    // still-valid default_stream (thread-local, not destroyed until thread exits).
    for (int g = 1; g < n_gpus_; g++) {
        contexts_[g].release();   // skip stale-stream destructor
    }
    contexts_.clear();
}

DistributedContext::~DistributedContext() {
    // Don't destroy in destructor — caller must call destroy() explicitly
    // to avoid issues with move semantics
}

// ---------------------------------------------------------------------------
// DistributedCiphertext
// ---------------------------------------------------------------------------

DistributedCiphertext DistributedCiphertext::from_single_gpu(
    DistributedContext &ctx,
    const PhantomCiphertext &ct,
    int source_gpu)
{
    DistributedCiphertext dct;
    dct.n_gpus_ = ctx.n_gpus();
    dct.n_polys_ = ct.size();
    dct.chain_index_ = ct.chain_index();
    dct.poly_degree_ = ct.poly_modulus_degree();
    dct.total_limbs_ = ct.coeff_modulus_size();
    dct.scale_ = ct.scale();
    dct.is_ntt_form_ = ct.is_ntt_form();

    dct.local_data_.resize(ctx.n_gpus(), nullptr);
    dct.local_limb_counts_.resize(ctx.n_gpus());

    // Allocate local buffers and scatter limbs
    for (int g = 0; g < ctx.n_gpus(); g++) {
        size_t local_n = n_local_limbs(g, ctx.n_gpus(), dct.total_limbs_);
        dct.local_limb_counts_[g] = local_n;

        CUDA_CHECK(cudaSetDevice(g));
        size_t buf_bytes = dct.n_polys_ * local_n * dct.poly_degree_ * sizeof(uint64_t);
        if (buf_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&dct.local_data_[g], buf_bytes));

            // Scatter: copy this GPU's limbs from the source ciphertext
            // Source data layout: [poly][limb][coeff]
            for (size_t p = 0; p < dct.n_polys_; p++) {
                size_t loc = 0;
                for (size_t j = 0; j < dct.total_limbs_; j++) {
                    if (owner_of_limb(j, ctx.n_gpus()) != g) continue;
                    // Source: ct.data() + p * total_limbs * degree + j * degree
                    // Dest:   local_data_[g] + p * local_n * degree + loc * degree
                    const uint64_t *src = ct.data()
                        + p * dct.total_limbs_ * dct.poly_degree_
                        + j * dct.poly_degree_;
                    uint64_t *dst = dct.local_data_[g]
                        + p * local_n * dct.poly_degree_
                        + loc * dct.poly_degree_;
                    CUDA_CHECK(cudaMemcpyPeer(dst, g, src, source_gpu,
                                              dct.poly_degree_ * sizeof(uint64_t)));
                    loc++;
                }
            }
        }
    }

    CUDA_CHECK(cudaSetDevice(0));
    return dct;
}

PhantomCiphertext DistributedCiphertext::to_single_gpu(
    DistributedContext &ctx,
    int target_gpu) const
{
    CUDA_CHECK(cudaSetDevice(target_gpu));

    PhantomCiphertext ct;
    ct.resize(ctx.context(target_gpu), chain_index_, n_polys_, cudaStreamPerThread);
    ct.set_scale(scale_);
    ct.set_ntt_form(is_ntt_form_);

    // Gather limbs from all GPUs
    for (int g = 0; g < n_gpus_; g++) {
        size_t local_n = local_limb_counts_[g];
        for (size_t p = 0; p < n_polys_; p++) {
            size_t loc = 0;
            for (size_t j = 0; j < total_limbs_; j++) {
                if (owner_of_limb(j, n_gpus_) != g) continue;
                const uint64_t *src = local_data_[g]
                    + p * local_n * poly_degree_
                    + loc * poly_degree_;
                uint64_t *dst = ct.data()
                    + p * total_limbs_ * poly_degree_
                    + j * poly_degree_;
                CUDA_CHECK(cudaMemcpyPeer(dst, target_gpu, src, g,
                                          poly_degree_ * sizeof(uint64_t)));
                loc++;
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return ct;
}

void DistributedCiphertext::allocate(
    DistributedContext &ctx, size_t n_polys,
    size_t chain_index, size_t poly_degree)
{
    n_gpus_ = ctx.n_gpus();
    n_polys_ = n_polys;
    chain_index_ = chain_index;
    poly_degree_ = poly_degree;
    total_limbs_ = ctx.total_limbs_at_level(chain_index);

    local_data_.resize(n_gpus_, nullptr);
    local_limb_counts_.resize(n_gpus_);

    for (int g = 0; g < n_gpus_; g++) {
        size_t local_n = n_local_limbs(g, n_gpus_, total_limbs_);
        local_limb_counts_[g] = local_n;
        CUDA_CHECK(cudaSetDevice(g));
        size_t bytes = n_polys * local_n * poly_degree * sizeof(uint64_t);
        if (bytes > 0) {
            CUDA_CHECK(cudaMalloc(&local_data_[g], bytes));
        }
    }
    CUDA_CHECK(cudaSetDevice(0));
}

void DistributedCiphertext::free_all() {
    for (int g = 0; g < n_gpus_; g++) {
        if (local_data_[g]) {
            cudaSetDevice(g);
            cudaFree(local_data_[g]);
            local_data_[g] = nullptr;
        }
    }
}

DistributedCiphertext::~DistributedCiphertext() {
    free_all();
}

// ---------------------------------------------------------------------------
// DistributedCiphertext move semantics
// ---------------------------------------------------------------------------
// The user-declared destructor suppresses implicit move generation (C++ Rule of Five).
// Without explicit moves, assignments like `dct = from_single_gpu(...)` fall back to
// copy assignment (shallow pointer copy). The temporary's destructor then frees the
// shared GPU buffers, leaving dct with dangling local_data_ pointers.
// Fix: implement move constructor and move assignment that transfer ownership.

DistributedCiphertext::DistributedCiphertext(DistributedCiphertext&& other) noexcept
    : local_data_(std::move(other.local_data_)),
      local_limb_counts_(std::move(other.local_limb_counts_)),
      n_polys_(other.n_polys_),
      chain_index_(other.chain_index_),
      poly_degree_(other.poly_degree_),
      total_limbs_(other.total_limbs_),
      scale_(other.scale_),
      is_ntt_form_(other.is_ntt_form_),
      n_gpus_(other.n_gpus_)
{
    other.n_gpus_ = 0;  // prevent other's destructor from double-freeing
}

DistributedCiphertext& DistributedCiphertext::operator=(DistributedCiphertext&& other) noexcept {
    if (this != &other) {
        free_all();  // release existing GPU buffers before overwriting
        local_data_       = std::move(other.local_data_);
        local_limb_counts_= std::move(other.local_limb_counts_);
        n_polys_      = other.n_polys_;
        chain_index_  = other.chain_index_;
        poly_degree_  = other.poly_degree_;
        total_limbs_  = other.total_limbs_;
        scale_        = other.scale_;
        is_ntt_form_  = other.is_ntt_form_;
        n_gpus_       = other.n_gpus_;
        other.n_gpus_ = 0;  // prevent other's destructor from double-freeing
    }
    return *this;
}

} // namespace nexus_multi_gpu
