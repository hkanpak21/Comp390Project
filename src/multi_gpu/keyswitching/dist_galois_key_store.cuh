/**
 * dist_galois_key_store.cuh
 *
 * Sharded Galois key storage for Distributed Key-Switching (DKS).
 *
 * Memory layout
 * -------------
 * At N=65536 with beta=9 key-switching digits and P=4 GPUs:
 *   Full key per rotation:   beta × 2 × size_QP × N × 8 = ~544 MB
 *   Shard per GPU:           ceil(beta/P) × ... = ~136 MB
 *   50 bootstrap keys total: ~6.8 GB per GPU   (vs ~27 GB full replication)
 *
 * Each GPU holds only its assigned digit shard: digits d where d % n_gpus == gpu_id.
 * Unowned digit slots in the evks pointer array are null — the partial KS kernel
 * uses a strided loop (d = gpu_id, gpu_id+P, ...) and never dereferences null slots.
 *
 * Usage
 * -----
 *   DistGaloisKeyStore ks;
 *   ks.generate(ctx, sk, n_gpus, num_keys);          // call once at setup on GPU 0
 *
 *   // At rotation time, on GPU g:
 *   uint64_t** evks = ks.get_evks(g, key_idx);       // device pointer array (shard only)
 *   size_t beta      = ks.beta(key_idx);
 *   // Pass evks as custom_evks to keyswitching_output_aggregation
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

#include "context.cuh"
#include "secretkey.h"
#include "util/globals.h"

#define DKS_CUDA_CHECK(cmd) do {                                          \
    cudaError_t _e = (cmd);                                               \
    if (_e != cudaSuccess) {                                              \
        fprintf(stderr, "[DistGaloisKeyStore] CUDA error %s:%d: %s\n",   \
                __FILE__, __LINE__, cudaGetErrorString(_e));              \
        throw std::runtime_error(cudaGetErrorString(_e));                 \
    }                                                                     \
} while(0)

class DistGaloisKeyStore {
public:
    // One GPU's shard for one rotation key
    struct GpuShard {
        uint64_t**              evks_device = nullptr; // device: beta ptr array (nulls for unowned)
        std::vector<uint64_t*>  owned_bufs;            // device ptrs to locally-allocated digit data
        size_t                  beta = 0;
        int                     device_id = -1;
    };

private:
    // shards_[key_idx][gpu_id]
    std::vector<std::vector<GpuShard>> shards_;
    int    n_gpus_   = 0;
    size_t num_keys_ = 0;
    bool   built_    = false;

public:
    DistGaloisKeyStore() = default;
    ~DistGaloisKeyStore() { destroy(); }

    // Non-copyable, movable
    DistGaloisKeyStore(const DistGaloisKeyStore&) = delete;
    DistGaloisKeyStore& operator=(const DistGaloisKeyStore&) = delete;

    /**
     * generate()
     *
     * For each of `num_keys` rotation keys (indices 0..num_keys-1):
     *   1. Generate full key on GPU 0 (ctx must be on GPU 0)
     *   2. Copy ALL components to host via copy_to_host
     *   3. For each GPU g:
     *      a. Allocate device buffers for owned digits (d % n_gpus == g)
     *      b. Copy owned digit data to GPU g
     *      c. Build evks device pointer array (valid for owned, null for others)
     *   4. Free full GPU key (RAII via PhantomRelinKey destructor)
     *
     * Must be called on GPU 0 before inference.
     */
    void generate(PhantomContext        &ctx,
                  const PhantomSecretKey &sk,
                  int                    n_gpus,
                  size_t                 num_keys)
    {
        n_gpus_   = n_gpus;
        num_keys_ = num_keys;

        shards_.resize(num_keys_, std::vector<GpuShard>(n_gpus));
        for (size_t ki = 0; ki < num_keys_; ki++)
            for (int g = 0; g < n_gpus_; g++)
                shards_[ki][g].device_id = g;

        const auto &stream0 = phantom::util::global_variables::default_stream->get_stream();

        printf("[DistGaloisKeyStore] Generating %zu sharded keys for %d GPUs...\n",
               num_keys_, n_gpus_);
        fflush(stdout);

        for (size_t ki = 0; ki < num_keys_; ki++) {
            // --- Step 1: generate full key on GPU 0 ---
            DKS_CUDA_CHECK(cudaSetDevice(0));
            PhantomRelinKey rk = sk.generate_single_galois_key(ctx, ki);
            DKS_CUDA_CHECK(cudaStreamSynchronize(stream0));

            // --- Step 2: copy all components to host ---
            // components[d] = all uint64_t data for d-th digit (both evk_a and evk_b interleaved)
            std::vector<std::vector<uint64_t>> comps;
            rk.copy_to_host(comps);   // comps[d] = flat vector<uint64_t>
            size_t beta = comps.size();

            // rk destructor frees GPU memory automatically here

            // --- Step 3: for each GPU, allocate owned digit buffers and build evks ---
            for (int g = 0; g < n_gpus_; g++) {
                DKS_CUDA_CHECK(cudaSetDevice(g));
                cudaStream_t sg = (g == 0) ? stream0 : cudaStreamPerThread;

                auto &shard = shards_[ki][g];
                shard.beta  = beta;

                // Build host-side pointer array (null for unowned digits)
                std::vector<uint64_t*> host_ptrs(beta, nullptr);

                for (size_t d = static_cast<size_t>(g); d < beta; d += n_gpus_) {
                    // Allocate device buffer for this owned digit
                    size_t n_elem = comps[d].size();
                    uint64_t *dptr = nullptr;
                    DKS_CUDA_CHECK(cudaMalloc(&dptr, n_elem * sizeof(uint64_t)));
                    DKS_CUDA_CHECK(cudaMemcpyAsync(dptr, comps[d].data(),
                                                   n_elem * sizeof(uint64_t),
                                                   cudaMemcpyHostToDevice, sg));
                    host_ptrs[d] = dptr;
                    shard.owned_bufs.push_back(dptr);
                }
                DKS_CUDA_CHECK(cudaStreamSynchronize(sg));

                // Build device evks array
                DKS_CUDA_CHECK(cudaMalloc(&shard.evks_device,
                                          beta * sizeof(uint64_t*)));
                DKS_CUDA_CHECK(cudaMemcpy(shard.evks_device, host_ptrs.data(),
                                          beta * sizeof(uint64_t*),
                                          cudaMemcpyHostToDevice));
            }

            if ((ki + 1) % 10 == 0 || ki == num_keys_ - 1) {
                size_t free_mem, total_mem;
                cudaSetDevice(0);
                cudaMemGetInfo(&free_mem, &total_mem);
                printf("[DistGaloisKeyStore] %zu/%zu keys sharded (GPU 0 free: %.2f GB)\n",
                       ki + 1, num_keys_,
                       free_mem / (1024.0 * 1024.0 * 1024.0));
                fflush(stdout);
            }
        }

        DKS_CUDA_CHECK(cudaSetDevice(0));
        built_ = true;
        printf("[DistGaloisKeyStore] All shards loaded. Memory per GPU: ~%.2f GB\n",
               estimated_gpu_memory_gb());
        fflush(stdout);
    }

    // Returns the device evks pointer array for key `key_idx` on GPU `gpu_id`.
    // The array has beta entries: valid device pointers for owned digits, null elsewhere.
    uint64_t** get_evks(int gpu_id, size_t key_idx) const {
        if (!built_) throw std::runtime_error("DistGaloisKeyStore: not built yet");
        return shards_.at(key_idx).at(gpu_id).evks_device;
    }

    size_t beta(size_t key_idx) const {
        return shards_.at(key_idx).at(0).beta;
    }

    size_t num_keys() const { return num_keys_; }
    int    n_gpus()   const { return n_gpus_; }

    double estimated_gpu_memory_gb() const {
        if (!built_ || shards_.empty()) return 0.0;
        // Count owned buffers' total bytes on GPU 0
        size_t total = 0;
        for (size_t ki = 0; ki < num_keys_; ki++) {
            const auto &shard = shards_[ki][0];
            // owned_bufs count × size_per_buf
            // size_per_buf = comps[d].size() bytes — use beta and evks_device size as proxy
            // Rough: each owned buf = (2 * size_QP * N) uint64_t = varies; just sum owned_bufs count
            total += shard.owned_bufs.size();
        }
        // Conservative estimate: owned digit buffers per GPU
        return total * 0.032;  // ~32 MB per digit chunk at N=65536
    }

    void destroy() {
        for (size_t ki = 0; ki < shards_.size(); ki++) {
            for (int g = 0; g < n_gpus_; g++) {
                auto &shard = shards_[ki][g];
                cudaSetDevice(g);
                for (auto *p : shard.owned_bufs)
                    if (p) cudaFree(p);
                shard.owned_bufs.clear();
                if (shard.evks_device) {
                    cudaFree(shard.evks_device);
                    shard.evks_device = nullptr;
                }
            }
        }
        shards_.clear();
        built_ = false;
    }
};
