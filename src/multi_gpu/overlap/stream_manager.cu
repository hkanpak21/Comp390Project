/**
 * stream_manager.cu
 *
 * Implementation of CUDA stream management and overlap scheduling.
 */

#include "stream_manager.cuh"
#include <stdexcept>
#include <string>

#define CUDA_CHECK(cmd) do {                                              \
    cudaError_t e = (cmd);                                                \
    if (e != cudaSuccess) {                                               \
        throw std::runtime_error(std::string("CUDA error: ") +           \
                                 cudaGetErrorString(e));                  \
    }                                                                     \
} while (0)

namespace nexus_multi_gpu {

// ---------------------------------------------------------------------------
// PerGpuStreams
// ---------------------------------------------------------------------------

PerGpuStreams PerGpuStreams::create(int dev_id) {
    PerGpuStreams s;
    s.device_id = dev_id;
    CUDA_CHECK(cudaSetDevice(dev_id));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s.compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s.nccl,    cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&s.compute_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&s.nccl_done,    cudaEventDisableTiming));
    return s;
}

void PerGpuStreams::sync_all() const {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamSynchronize(compute));
    CUDA_CHECK(cudaStreamSynchronize(nccl));
}

void PerGpuStreams::signal_compute_done() {
    CUDA_CHECK(cudaEventRecord(compute_done, compute));
}

void PerGpuStreams::nccl_wait_for_compute() {
    CUDA_CHECK(cudaStreamWaitEvent(nccl, compute_done, 0));
}

void PerGpuStreams::compute_wait_for_nccl() {
    CUDA_CHECK(cudaEventRecord(nccl_done, nccl));
    CUDA_CHECK(cudaStreamWaitEvent(compute, nccl_done, 0));
}

void PerGpuStreams::destroy() {
    CUDA_CHECK(cudaSetDevice(device_id));
    cudaStreamDestroy(compute);
    cudaStreamDestroy(nccl);
    cudaEventDestroy(compute_done);
    cudaEventDestroy(nccl_done);
}

// ---------------------------------------------------------------------------
// StreamManager
// ---------------------------------------------------------------------------

StreamManager::StreamManager(const std::vector<int> &device_ids) {
    gpu_streams_.reserve(device_ids.size());
    for (int dev : device_ids) {
        gpu_streams_.push_back(PerGpuStreams::create(dev));
    }
}

StreamManager::~StreamManager() {
    for (auto &s : gpu_streams_) s.destroy();
}

PerGpuStreams &StreamManager::gpu(int gpu_id) {
    return gpu_streams_.at(static_cast<size_t>(gpu_id));
}

const PerGpuStreams &StreamManager::gpu(int gpu_id) const {
    return gpu_streams_.at(static_cast<size_t>(gpu_id));
}

void StreamManager::barrier_all() {
    for (auto &s : gpu_streams_) s.sync_all();
}

void StreamManager::barrier_gpu(int gpu_id) {
    gpu(gpu_id).sync_all();
}

void StreamManager::enable_graph_capture(bool enable) {
    graph_capture_enabled_ = enable;
}

void StreamManager::begin_capture(int gpu_id) {
    CUDA_CHECK(cudaSetDevice(gpu(gpu_id).device_id));
    CUDA_CHECK(cudaStreamBeginCapture(gpu(gpu_id).compute,
                                      cudaStreamCaptureModeGlobal));
}

cudaGraphExec_t StreamManager::end_capture(int gpu_id) {
    cudaGraph_t     graph;
    cudaGraphExec_t instance;

    CUDA_CHECK(cudaStreamEndCapture(gpu(gpu_id).compute, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
    CUDA_CHECK(cudaGraphDestroy(graph));

    return instance;
}

void StreamManager::replay_graph(int gpu_id, cudaGraphExec_t instance) {
    CUDA_CHECK(cudaGraphLaunch(instance, gpu(gpu_id).compute));
}

void StreamManager::destroy_graph(cudaGraphExec_t instance) {
    cudaGraphExecDestroy(instance);
}

// ---------------------------------------------------------------------------
// OverlapScheduler
// ---------------------------------------------------------------------------

OverlapScheduler::OverlapScheduler(StreamManager &mgr) : mgr_(mgr) {}

cudaStream_t OverlapScheduler::compute_stream(int gpu_id) const {
    return mgr_.gpu(gpu_id).compute;
}

cudaStream_t OverlapScheduler::nccl_stream(int gpu_id) const {
    return mgr_.gpu(gpu_id).nccl;
}

void OverlapScheduler::schedule_compute_comm_overlap(
    int gpu_id,
    std::function<void(cudaStream_t)> compute_fn,
    std::function<void(cudaStream_t)> comm_fn,
    std::function<void(cudaStream_t)> post_fn)
{
    auto &s = mgr_.gpu(gpu_id);

    // Launch compute work on the compute stream.
    compute_fn(s.compute);

    // Launch comm work on the nccl stream (runs concurrently with compute).
    comm_fn(s.nccl);

    // After both finish, run post work on the compute stream.
    // Record nccl_done after comm completes, then make compute wait for it.
    s.compute_wait_for_nccl();
    // Also record compute_done and make nccl wait (for correct ordering on replay).
    s.signal_compute_done();
    s.nccl_wait_for_compute();

    post_fn(s.compute);
}

} // namespace nexus_multi_gpu
