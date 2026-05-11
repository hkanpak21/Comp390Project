#pragma once
/**
 * distributed_context.cuh
 *
 * Core multi-GPU FHE infrastructure for NEXUS.
 *
 * DistributedContext manages:
 *   - One PhantomContext per GPU (created with cudaSetDevice)
 *   - Shallow-copy evaluation keys on each GPU (FIDESlib pattern)
 *   - NCCL communicators for inter-GPU collectives
 *   - RNS limb partitioning across GPUs
 *
 * Design: SPMD (Single Program Multiple Data) — each GPU runs the same
 * code on its local subset of RNS limbs. Communication happens only at
 * key-switching points (relinearize, rotate).
 *
 * Key insight: Most FHE operations (add, multiply_plain, rescale, NTT)
 * are per-limb and need ZERO communication. Only key-switching and
 * rotation require inter-GPU data movement.
 */

#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "context.cuh"
#include "secretkey.h"
#include "ciphertext.h"

namespace nexus_multi_gpu {

// ---------------------------------------------------------------------------
// GPU-local key storage (shallow copy approach from FIDESlib)
// ---------------------------------------------------------------------------
// Keys are read-only during inference. We copy key DATA to each GPU once
// at initialization, then create lightweight wrappers pointing to local copies.

struct GpuKeySet {
    int device_id;
    // Raw GPU pointers to key data on this device
    // These are shallow copies — just the uint64_t* arrays, not PhantomRelinKey objects
    uint64_t *relin_key_data = nullptr;    // device memory on this GPU
    size_t    relin_key_bytes = 0;
    uint64_t **relin_key_ptrs = nullptr;   // device array of pointers (for kernel access)
    size_t    relin_n_keys = 0;

    // Galois keys: map from galois_elt -> key data
    // For NEXUS BERT: we need specific rotation steps
    std::vector<uint64_t*> galois_key_data;   // one per galois element
    std::vector<uint64_t**> galois_key_ptrs;  // device pointer arrays
};

// ---------------------------------------------------------------------------
// DistributedContext
// ---------------------------------------------------------------------------

class DistributedContext {
public:
    // Create distributed context across n_gpus GPUs (single node).
    // Initializes NCCL, creates per-GPU PhantomContexts, distributes keys.
    static DistributedContext create(
        const phantom::EncryptionParameters &parms,
        int n_gpus,
        const std::vector<int> &device_ids = {}  // default: 0..n_gpus-1
    );

#ifdef USE_MPI
    // Create multi-node context. Each MPI rank manages gpus_per_node local GPUs.
    // NCCL communicators span ALL nodes (total = mpi_size * gpus_per_node).
    // After creation:
    //   n_gpus()           == gpus_per_node  (local GPUs only)
    //   total_gpus()       == mpi_size * gpus_per_node
    //   global_rank(g)     == mpi_rank * gpus_per_node + g
    static DistributedContext create_multinode(
        const phantom::EncryptionParameters &parms,
        int gpus_per_node,
        MPI_Comm mpi_comm = MPI_COMM_WORLD
    );
#endif

    // Multi-node accessors (safe to call on single-node; returns n_gpus / 0 / 0)
    int total_gpus()              const { return total_gpus_ > 0 ? total_gpus_ : n_gpus_; }
    int global_rank(int local_gpu)const { return global_rank_offset_ + local_gpu; }
    int global_rank_offset()      const { return global_rank_offset_; }

    // Distribute keys from GPU 0 to all GPUs (shallow copy)
    void distribute_relin_keys(const PhantomRelinKey &relin_keys);
    void distribute_galois_keys(const PhantomGaloisKey &galois_keys);

    // Access per-GPU resources
    int n_gpus() const { return n_gpus_; }
    PhantomContext &context(int gpu) { return *contexts_[gpu]; }
    const PhantomContext &context(int gpu) const { return *contexts_[gpu]; }
    ncclComm_t comm(int gpu) const { return comms_[gpu]; }
    cudaStream_t stream(int gpu) const { return streams_[gpu]; }
    const GpuKeySet &keys(int gpu) const { return key_sets_[gpu]; }

    // Limb partitioning
    size_t total_limbs_at_level(size_t chain_index) const;
    size_t local_limbs(int gpu, size_t chain_index) const;

    // Move-only (contains unique_ptr vector)
    DistributedContext() = default;
    DistributedContext(const DistributedContext&) = delete;
    DistributedContext& operator=(const DistributedContext&) = delete;
    DistributedContext(DistributedContext&&) = default;
    DistributedContext& operator=(DistributedContext&&) = default;

    // Cleanup
    void destroy();
    ~DistributedContext();

    // ── Persistent rotation workspace (Phase 3 optimisation) ──
    // Per-GPU device buffers reused across DKS rotations to avoid per-call
    // cudaMalloc. Sized for the largest poly_bytes seen so far; grows on
    // demand. Each GPU holds its own c0_gal and c2_gal slot — for the
    // source GPU these point into the input poly directly (no copy needed
    // when broadcast is the same buffer); for peer GPUs they hold the
    // received broadcast.
    struct RotationWorkspace {
        std::vector<uint64_t*> c0_gal;       // [n_gpus] device buffers (or null on src)
        std::vector<uint64_t*> c2_gal;       // [n_gpus] device buffers (or null on src)
        size_t                 capacity = 0; // bytes per buffer
        // Phase 4a: persistent per-GPU local PhantomCiphertext, sized for the
        // last-seen chain_index. Reused across rotations; only reallocated
        // when chain_index changes (which is rare during a single bootstrap).
        std::vector<PhantomCiphertext> local_cts;
        std::vector<size_t>            local_chain_index;  // tracks current size, ~0 = unallocated
    };
    RotationWorkspace &rotation_workspace() { return rot_ws_; }

    // Ensure each GPU's c0_gal/c2_gal buffers are at least n_bytes.
    // Idempotent and reentrant-safe (single-threaded use during rotation).
    void ensure_rotation_workspace(size_t n_bytes);

    // ── Phase 4b: persistent worker threads for DKS rotation ──
    // One worker thread per GPU is spawned at DistributedContext::create() time
    // and remains alive until destroy(). Each worker is pinned to its device
    // (cudaSetDevice on thread startup) and loops waiting for work.
    // dispatch_to_all_gpus submits one lambda per GPU and blocks until all
    // workers report done. Eliminates std::thread spawn/join per rotation.
    struct Worker {
        std::thread             thread;
        std::mutex              mtx;
        std::condition_variable cv;
        std::function<void()>   work;
        std::exception_ptr      err;
        bool                    has_work = false;
        bool                    done     = false;
        bool                    shutdown = false;
        int                     device_id = -1;
    };
    // dispatch work[g] to worker[g] on GPU g, wait for all to finish.
    // Re-throws first worker exception on the caller's thread.
    void dispatch_to_all_gpus(std::vector<std::function<void()>> &work);
    bool has_workers() const { return !workers_.empty(); }

    // T-STRAGGLER / T-OVERLAP: persistent per-GPU events used as GPU-side
    // barriers around ncclAllReduce in keyswitching_output_aggregation_dks.
    // Created in DistributedContext::create() with cudaEventDisableTiming and
    // shallow-copied into the per-call MultiGpuContext built by galois_oa.cu.
    //
    // T-OVERLAP: inert — see docs/HPC_PRIMER.md §6.3 for null-result analysis.
    // (Also: commit 44c849f / docs/PRD §S28.) T-STRAGGLER barrier was measured
    // null on H100×4 NVSwitch (NCCL ring already issues launches in lockstep
    // within ~5 µs); T-OVERLAP cross-stream gate exposed a chain-level
    // dependent race and was reverted to a host sync. Events are still
    // allocated and recorded so the dispatch path is unchanged, but they no
    // longer eliminate the straggler wait. Kept as scaffolding for future
    // re-enable behind a runtime flag.
    const std::vector<cudaEvent_t>& ready_events()         const { return ready_events_; }
    const std::vector<cudaEvent_t>& allreduce_done_events() const { return allreduce_done_events_; }
    // T-OVERLAP: end-of-OA events recorded after oa_add_to_ct in
    // keyswitching_output_aggregation_dks. Waited on by the rotation caller's
    // writeback memcpy (on stream0) so we can drop the trailing host sync.
    //
    // T-OVERLAP: inert — see docs/HPC_PRIMER.md §6.3 for null-result analysis.
    // (Also: commit 44c849f / docs/PRD §S28.) The cross-stream wait was buggy
    // under chain levels where size_Ql changed between rotations and was
    // reverted in favor of the safe cudaStreamSynchronize at line ~358.
    // Event recording is still emitted but the writeback never waits on it.
    const std::vector<cudaEvent_t>& oa_done_events()        const { return oa_done_events_; }

private:
    int n_gpus_ = 0;
    std::vector<int> device_ids_;
    std::vector<std::unique_ptr<PhantomContext>> contexts_;
    std::vector<ncclComm_t> comms_;
    std::vector<cudaStream_t> streams_;
    std::vector<GpuKeySet> key_sets_;
    // T-STRAGGLER / T-OVERLAP: per-GPU CUDA events (cudaEventDisableTiming).
    // ready_events_ is recorded after partial_key_switch_inner_prod and waited
    // on before ncclAllReduce. allreduce_done_events_ is recorded after the
    // ncclAllReduce enqueue (replaces the cudaStreamSynchronize that previously
    // blocked the CPU and prevented overlap with the next rotation's modup).
    //
    // T-OVERLAP: inert — see docs/HPC_PRIMER.md §6.3 for null-result analysis.
    // (Also: commit 44c849f / docs/PRD §S28.) Both barrier paths produced null
    // wins on H100×4 NVSwitch and the OA-done cross-stream gate was unsafe at
    // chain transitions. Events are still allocated/recorded but the kernel
    // order on the same stream already enforces the required dependency, and
    // the host-side sync at line 358 of output_aggregation.cu is what actually
    // serializes the writeback.
    std::vector<cudaEvent_t> ready_events_;
    std::vector<cudaEvent_t> allreduce_done_events_;
    // T-OVERLAP: per-GPU events recorded at the END of
    // keyswitching_output_aggregation_dks (after oa_add_to_ct), replacing the
    // trailing cudaStreamSynchronize. Used as a cross-stream gate for the
    // rotation caller's writeback memcpy on stream0.
    //
    // T-OVERLAP: inert — see docs/HPC_PRIMER.md §6.3 for null-result analysis.
    // (Also: commit 44c849f / docs/PRD §S28.) Reverted to host sync; events
    // are still recorded for trace continuity.
    std::vector<cudaEvent_t> oa_done_events_;
    phantom::EncryptionParameters parms_;
    bool destroyed_ = false;
    // Multi-node fields (0 if single-node)
    int total_gpus_          = 0;
    int global_rank_offset_  = 0;
    // Phase 3: persistent rotation workspace
    RotationWorkspace rot_ws_;
    // Phase 4b: persistent worker threads (one per GPU)
    std::vector<std::unique_ptr<Worker>> workers_;
};

// ---------------------------------------------------------------------------
// DistributedCiphertext
// ---------------------------------------------------------------------------
// A ciphertext distributed across GPUs by RNS limbs.
// Each GPU holds its local limbs as a raw uint64_t* buffer.

class DistributedCiphertext {
public:
    // Create from a single-GPU PhantomCiphertext (scatter limbs to GPUs)
    static DistributedCiphertext from_single_gpu(
        DistributedContext &ctx,
        const PhantomCiphertext &ct,
        int source_gpu = 0
    );

    // Gather back to a single-GPU PhantomCiphertext
    PhantomCiphertext to_single_gpu(
        DistributedContext &ctx,
        int target_gpu = 0
    ) const;

    // Per-GPU data access
    uint64_t *data(int gpu) { return local_data_[gpu]; }
    const uint64_t *data(int gpu) const { return local_data_[gpu]; }

    // Metadata (same across all GPUs)
    size_t size() const { return n_polys_; }
    size_t chain_index() const { return chain_index_; }
    size_t poly_degree() const { return poly_degree_; }
    size_t total_limbs() const { return total_limbs_; }
    double scale() const { return scale_; }
    bool is_ntt_form() const { return is_ntt_form_; }
    size_t local_limb_count(int gpu) const { return local_limb_counts_[gpu]; }

    void set_chain_index(size_t idx) { chain_index_ = idx; }
    void set_scale(double s) { scale_ = s; }

    // Allocate local buffers on each GPU
    void allocate(DistributedContext &ctx, size_t n_polys,
                  size_t chain_index, size_t poly_degree);

    // Free GPU memory
    void free_all();
    ~DistributedCiphertext();

    // Move-only: raw GPU pointers cannot be safely shallow-copied.
    // The user-declared destructor suppresses implicit move generation (C++ Rule of Five),
    // so we must define them explicitly. Copy is deleted to prevent accidental double-free.
    DistributedCiphertext() = default;
    DistributedCiphertext(const DistributedCiphertext&) = delete;
    DistributedCiphertext& operator=(const DistributedCiphertext&) = delete;
    DistributedCiphertext(DistributedCiphertext&& other) noexcept;
    DistributedCiphertext& operator=(DistributedCiphertext&& other) noexcept;

private:
    std::vector<uint64_t*> local_data_;  // one buffer per GPU
    std::vector<size_t> local_limb_counts_;
    size_t n_polys_ = 0;
    size_t chain_index_ = 0;
    size_t poly_degree_ = 0;
    size_t total_limbs_ = 0;
    double scale_ = 1.0;
    bool is_ntt_form_ = true;
    int n_gpus_ = 0;
};

} // namespace nexus_multi_gpu
