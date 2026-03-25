#pragma once
/**
 * stream_manager.cuh
 *
 * CUDA stream management and compute--communication overlap for multi-GPU FHE.
 *
 * Overlap strategy
 * ----------------
 * FHE operations on different parts of the ciphertext pipeline can often proceed
 * concurrently. The key insight is:
 *
 *   1. While GPU g is performing a NCCL AllGather for key-switching c2 limbs,
 *      the same GPU can simultaneously run NTT on limbs that do NOT participate in
 *      the current communication phase.
 *
 *   2. After the AllGather completes, the key-switch inner product can begin while
 *      NTT on subsequent ciphertext components is already in flight.
 *
 * We use two CUDA streams per GPU:
 *   - compute_stream: NTT, polynomial arithmetic, modular reduction
 *   - nccl_stream:    NCCL AllGather / AllReduce / Broadcast collectives
 *
 * A CUDA event chain synchronizes the two streams at dependency boundaries.
 *
 * CudaGraph capture
 * -----------------
 * For the bootstrapping pipeline (a long fixed sequence of ~50 FHE operations),
 * we optionally capture the entire sequence into a CudaGraph. This eliminates
 * kernel launch overhead (~5us per launch * 50 kernels = 250us saved) and allows
 * the CUDA runtime to optimize the execution schedule. The capture phase runs once;
 * subsequent bootstrapping calls replay the graph in microseconds.
 *
 * Enable with: StreamManager::enable_graph_capture(true) before bootstrapping.
 */

#include <cstddef>
#include <vector>
#include <functional>
#include <cuda_runtime.h>

namespace nexus_multi_gpu {

// ---------------------------------------------------------------------------
// PerGpuStreams
// ---------------------------------------------------------------------------

/**
 * PerGpuStreams
 *
 * Owns the two streams and event synchronization objects for one GPU.
 */
struct PerGpuStreams {
    int          device_id;
    cudaStream_t compute;          ///< For NTT, poly arithmetic
    cudaStream_t nccl;             ///< For NCCL collectives
    cudaEvent_t  compute_done;     ///< Signals when compute work is complete
    cudaEvent_t  nccl_done;        ///< Signals when NCCL collective is complete

    /// Create streams and events on the given device.
    static PerGpuStreams create(int dev_id);

    /// Synchronize both streams (wait for all pending work).
    void sync_all() const;

    /// Signal that compute work up to this point is done (record event).
    void signal_compute_done();

    /// Make the NCCL stream wait for the compute event before starting the collective.
    void nccl_wait_for_compute();

    /// Make the compute stream wait for the NCCL event before continuing.
    void compute_wait_for_nccl();

    void destroy();
};

// ---------------------------------------------------------------------------
// StreamManager
// ---------------------------------------------------------------------------

/**
 * StreamManager
 *
 * Central manager for all per-GPU streams. Provides:
 *   - Access to per-GPU stream pairs
 *   - Barrier synchronization across all GPUs
 *   - Optional CudaGraph capture for bootstrapping
 */
class StreamManager {
public:
    explicit StreamManager(const std::vector<int> &device_ids);
    ~StreamManager();

    /// Access the stream pair for a specific GPU.
    PerGpuStreams &gpu(int gpu_id);
    const PerGpuStreams &gpu(int gpu_id) const;

    /// Block the host until ALL GPUs have finished ALL pending work.
    void barrier_all();

    /// Block the host until the specified GPU has finished all pending work.
    void barrier_gpu(int gpu_id);

    // -----------------------------------------------------------------------
    // CudaGraph capture
    // -----------------------------------------------------------------------

    /// Enable/disable CudaGraph capture mode.
    /// When enabled, the next call to begin_capture() starts recording.
    void enable_graph_capture(bool enable);
    bool graph_capture_enabled() const { return graph_capture_enabled_; }

    /**
     * begin_capture
     *
     * Start capturing all CUDA operations on the compute stream of `gpu_id`
     * into a CudaGraph. All CUDA calls between begin_capture() and end_capture()
     * are recorded, NOT executed.
     */
    void begin_capture(int gpu_id);

    /**
     * end_capture
     *
     * Finalize the CudaGraph recording and store the executable graph instance.
     * Returns the graph instance handle for later replay.
     */
    cudaGraphExec_t end_capture(int gpu_id);

    /**
     * replay_graph
     *
     * Execute a previously captured CudaGraph on the compute stream of `gpu_id`.
     * This is a single asynchronous launch that replays all captured work.
     */
    void replay_graph(int gpu_id, cudaGraphExec_t graph_instance);

    /**
     * destroy_graph
     *
     * Free a CudaGraph instance when it is no longer needed.
     */
    static void destroy_graph(cudaGraphExec_t graph_instance);

private:
    std::vector<PerGpuStreams> gpu_streams_;
    bool                       graph_capture_enabled_ = false;
};

// ---------------------------------------------------------------------------
// OverlapScheduler
// ---------------------------------------------------------------------------

/**
 * OverlapScheduler
 *
 * High-level scheduler for overlapping NCCL communication with computation.
 *
 * Usage pattern for Input Broadcast key-switching:
 *
 *   scheduler.begin_compute_phase(gpu_id);
 *     // ... perform NTT on non-communicated limbs ...
 *   scheduler.begin_comm_phase(gpu_id);
 *     // ... launch NCCL AllGather on nccl_stream ...
 *   scheduler.sync_comm_then_compute(gpu_id);
 *     // ... inner product on full c2 (after AllGather) ...
 *   scheduler.end_phase(gpu_id);
 */
class OverlapScheduler {
public:
    explicit OverlapScheduler(StreamManager &mgr);

    /// Get the compute stream for a GPU (for NTT / poly ops).
    cudaStream_t compute_stream(int gpu_id) const;

    /// Get the NCCL stream for a GPU (for NCCL ops).
    cudaStream_t nccl_stream(int gpu_id) const;

    /**
     * schedule_compute_comm_overlap
     *
     * Submit work so that `compute_fn` and `comm_fn` run concurrently on
     * separate streams for the given GPU.
     *
     * `compute_fn(stream)` should launch CUDA kernels on the provided stream.
     * `comm_fn(stream)`    should call NCCL collectives on the provided stream.
     *
     * After both complete, `post_fn(compute_stream)` is called for any work
     * that depends on both having finished.
     */
    void schedule_compute_comm_overlap(
        int gpu_id,
        std::function<void(cudaStream_t)> compute_fn,
        std::function<void(cudaStream_t)> comm_fn,
        std::function<void(cudaStream_t)> post_fn
    );

private:
    StreamManager &mgr_;
};

} // namespace nexus_multi_gpu
