/**
 * nccl_bandwidth_test.cu
 *
 * Validates NCCL setup and measures NVSwitch bandwidth for ciphertext-sized messages.
 *
 * This is the simplest benchmark — it does NOT require Phantom or CUDA kernels.
 * Run this first on EC2 to confirm that:
 *   1. NCCL initializes correctly on all 8 GPUs
 *   2. Peer access is enabled between all GPU pairs
 *   3. NVSwitch bandwidth meets the expected ~600 GB/s per GPU
 *
 * Usage:
 *   ./nccl_bandwidth_test [--n-gpus N] [--msg-size-mb M] [--iters I]
 *
 * Expected output on p4d.24xlarge (8x A100, NVSwitch):
 *   AllGather   bandwidth: ~4800 GB/s aggregate (~600 GB/s per GPU)
 *   AllReduce   bandwidth: ~2400 GB/s aggregate
 *   Broadcast   bandwidth: ~4800 GB/s aggregate
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

// ---------------------------------------------------------------------------
// Minimal argument parsing
// ---------------------------------------------------------------------------

struct Config {
    int    n_gpus       = 8;
    size_t msg_size_mb  = 20;   // default: ~1 ciphertext (20 MB for N=65536, L=20)
    int    iters        = 20;
    bool   verbose      = false;
};

static Config parse_args(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--n-gpus"       && i + 1 < argc) c.n_gpus      = std::atoi(argv[++i]);
        if (a == "--msg-size-mb"  && i + 1 < argc) c.msg_size_mb = std::atol(argv[++i]);
        if (a == "--iters"        && i + 1 < argc) c.iters       = std::atoi(argv[++i]);
        if (a == "--verbose")                       c.verbose     = true;
    }
    return c;
}

// ---------------------------------------------------------------------------
// Error macros
// ---------------------------------------------------------------------------

#define CUDA_CHECK(cmd) do {                                          \
    cudaError_t e = (cmd);                                            \
    if (e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s at %s:%d\n",                  \
                cudaGetErrorString(e), __FILE__, __LINE__);           \
        std::exit(1);                                                 \
    }                                                                 \
} while (0)

#define NCCL_CHECK(cmd) do {                                          \
    ncclResult_t r = (cmd);                                           \
    if (r != ncclSuccess) {                                           \
        fprintf(stderr, "NCCL error %s at %s:%d\n",                  \
                ncclGetErrorString(r), __FILE__, __LINE__);           \
        std::exit(1);                                                 \
    }                                                                 \
} while (0)

// ---------------------------------------------------------------------------
// Per-GPU worker state
// ---------------------------------------------------------------------------

struct GpuState {
    int            dev;
    ncclComm_t     comm;
    cudaStream_t   stream;
    uint64_t      *send_buf;
    uint64_t      *recv_buf;
    size_t         n_elements;   // elements per GPU
    cudaEvent_t    start, stop;
};

// ---------------------------------------------------------------------------
// Benchmark runners
// ---------------------------------------------------------------------------

// Warm up + timed AllGather.
// Returns observed bandwidth in GB/s (aggregate across all GPUs).
static double bench_allgather(std::vector<GpuState> &gpus, int iters) {
    int n_gpus = static_cast<int>(gpus.size());
    size_t N   = gpus[0].n_elements;

    // Warm-up pass
    for (int w = 0; w < 3; ++w) {
        ncclGroupStart();
        for (auto &g : gpus)
            NCCL_CHECK(ncclAllGather(g.send_buf, g.recv_buf, N,
                                     ncclUint64, g.comm, g.stream));
        ncclGroupEnd();
    }
    for (auto &g : gpus) CUDA_CHECK(cudaStreamSynchronize(g.stream));

    // Timed passes
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        ncclGroupStart();
        for (auto &g : gpus)
            NCCL_CHECK(ncclAllGather(g.send_buf, g.recv_buf, N,
                                     ncclUint64, g.comm, g.stream));
        ncclGroupEnd();
    }
    for (auto &g : gpus) CUDA_CHECK(cudaStreamSynchronize(g.stream));
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count() / iters;
    // AllGather sends N*n_gpus total elements of 8 bytes per iter across the fabric.
    double bytes = static_cast<double>(N) * static_cast<double>(n_gpus)
                 * sizeof(uint64_t) * static_cast<double>(n_gpus);
    return bytes / elapsed_s / 1e9;
}

static double bench_allreduce(std::vector<GpuState> &gpus, int iters) {
    int n_gpus = static_cast<int>(gpus.size());
    size_t N   = gpus[0].n_elements * static_cast<size_t>(n_gpus);

    for (int w = 0; w < 3; ++w) {
        ncclGroupStart();
        for (auto &g : gpus)
            NCCL_CHECK(ncclAllReduce(g.recv_buf, g.recv_buf, N,
                                     ncclUint64, ncclSum, g.comm, g.stream));
        ncclGroupEnd();
    }
    for (auto &g : gpus) CUDA_CHECK(cudaStreamSynchronize(g.stream));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        ncclGroupStart();
        for (auto &g : gpus)
            NCCL_CHECK(ncclAllReduce(g.recv_buf, g.recv_buf, N,
                                     ncclUint64, ncclSum, g.comm, g.stream));
        ncclGroupEnd();
    }
    for (auto &g : gpus) CUDA_CHECK(cudaStreamSynchronize(g.stream));
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count() / iters;
    // AllReduce: 2*(n-1)/n * n * N bytes (ring algorithm)
    double bytes = 2.0 * (n_gpus - 1.0) / n_gpus
                 * static_cast<double>(n_gpus)
                 * static_cast<double>(N) * sizeof(uint64_t);
    return bytes / elapsed_s / 1e9;
}

static double bench_broadcast(std::vector<GpuState> &gpus, int iters) {
    size_t N = gpus[0].n_elements * static_cast<size_t>(gpus.size());

    for (int w = 0; w < 3; ++w) {
        ncclGroupStart();
        for (auto &g : gpus)
            NCCL_CHECK(ncclBroadcast(g.recv_buf, g.recv_buf, N,
                                     ncclUint64, 0, g.comm, g.stream));
        ncclGroupEnd();
    }
    for (auto &g : gpus) CUDA_CHECK(cudaStreamSynchronize(g.stream));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        ncclGroupStart();
        for (auto &g : gpus)
            NCCL_CHECK(ncclBroadcast(g.recv_buf, g.recv_buf, N,
                                     ncclUint64, 0, g.comm, g.stream));
        ncclGroupEnd();
    }
    for (auto &g : gpus) CUDA_CHECK(cudaStreamSynchronize(g.stream));
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count() / iters;
    double bytes = static_cast<double>(N) * sizeof(uint64_t)
                 * static_cast<double>(gpus.size());
    return bytes / elapsed_s / 1e9;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    printf("=== NCCL Bandwidth Test ===\n");
    printf("GPUs:        %d\n", cfg.n_gpus);
    printf("Msg size:    %zu MB\n", cfg.msg_size_mb);
    printf("Iterations:  %d\n\n", cfg.iters);

    // Check GPU count
    int n_devs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_devs));
    if (n_devs < cfg.n_gpus) {
        fprintf(stderr, "ERROR: requested %d GPUs but only %d available\n",
                cfg.n_gpus, n_devs);
        return 1;
    }

    // Setup
    std::vector<int> dev_ids(cfg.n_gpus);
    std::iota(dev_ids.begin(), dev_ids.end(), 0);

    std::vector<ncclComm_t> comms(cfg.n_gpus);
    NCCL_CHECK(ncclCommInitAll(comms.data(), cfg.n_gpus, dev_ids.data()));

    size_t total_bytes   = cfg.msg_size_mb * 1024 * 1024;
    size_t total_elems   = total_bytes / sizeof(uint64_t);
    size_t per_gpu_elems = total_elems / cfg.n_gpus;

    std::vector<GpuState> gpus(cfg.n_gpus);
    for (int g = 0; g < cfg.n_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        gpus[g].dev  = g;
        gpus[g].comm = comms[g];
        CUDA_CHECK(cudaStreamCreateWithFlags(&gpus[g].stream, cudaStreamNonBlocking));

        // send_buf: per-GPU slice, recv_buf: full gather buffer
        CUDA_CHECK(cudaMalloc(&gpus[g].send_buf, per_gpu_elems * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&gpus[g].recv_buf, total_elems   * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(gpus[g].send_buf, 1, per_gpu_elems * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(gpus[g].recv_buf, 0, total_elems   * sizeof(uint64_t)));
        gpus[g].n_elements = per_gpu_elems;
    }

    // Enable peer access
    for (int i = 0; i < cfg.n_gpus; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        for (int j = 0; j < cfg.n_gpus; ++j) {
            if (i == j) continue;
            int can = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can, i, j));
            if (can) cudaDeviceEnablePeerAccess(j, 0);
        }
    }

    // Run benchmarks
    double bw_ag = bench_allgather(gpus, cfg.iters);
    printf("AllGather  bandwidth: %.1f GB/s aggregate\n", bw_ag);

    double bw_ar = bench_allreduce(gpus, cfg.iters);
    printf("AllReduce  bandwidth: %.1f GB/s aggregate\n", bw_ar);

    double bw_bc = bench_broadcast(gpus, cfg.iters);
    printf("Broadcast  bandwidth: %.1f GB/s aggregate\n", bw_bc);

    // Ciphertext-specific numbers
    printf("\n--- Ciphertext Transfer Times (N=65536, L=20, 2 polys = 20.97 MB) ---\n");
    double ct_bytes = 2.0 * 20.0 * 65536.0 * 8.0;  // bytes per ciphertext
    if (bw_ag > 0)
        printf("AllGather  one ciphertext: %.0f us\n", ct_bytes / (bw_ag * 1e9 / 1e6));
    if (bw_ar > 0)
        printf("AllReduce  one ciphertext: %.0f us\n", ct_bytes / (bw_ar * 1e9 / 1e6));

    // Cleanup
    for (int g = 0; g < cfg.n_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(gpus[g].send_buf));
        CUDA_CHECK(cudaFree(gpus[g].recv_buf));
        CUDA_CHECK(cudaStreamDestroy(gpus[g].stream));
        ncclCommDestroy(comms[g]);
    }

    printf("\nDone.\n");
    return 0;
}
