/**
 * galois_key_store.cuh
 *
 * CPU-side Galois key storage for N=65536 bootstrap where all keys
 * don't fit in GPU memory simultaneously.
 *
 * Strategy: generate keys one at a time, save to CPU pinned memory,
 * load on demand during rotation. GPU holds at most 1 key at a time.
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "context.cuh"
#include "secretkey.h"
#include "util/globals.h"

class GaloisKeyStore {
public:
    struct HostKey {
        std::vector<std::vector<uint64_t>> components; // dnum vectors of uint64 data
        bool valid = false;
    };

private:
    std::vector<HostKey> keys_;
    size_t num_keys_ = 0;

    // Preallocated GPU buffers (reused for all keys — avoids alloc/free churn)
    std::vector<void *> gpu_buffers_;      // dnum buffers, each sized to fit one component
    std::vector<size_t> buffer_sizes_;     // size in bytes of each buffer
    void *gpu_ptr_array_ = nullptr;        // GPU array of uint64_t* (dnum entries)
    int last_loaded_idx_ = -1;
    size_t dnum_ = 0;

public:
    GaloisKeyStore() = default;

    ~GaloisKeyStore() {
        // Use same stream-ordered pool as Phantom
        const auto &stream = phantom::util::global_variables::default_stream->get_stream();
        for (auto p : gpu_buffers_) if (p) cudaFreeAsync(p, stream);
        if (gpu_ptr_array_) cudaFreeAsync(gpu_ptr_array_, stream);
    }

    // Generate all keys one at a time, saving each to CPU and freeing GPU
    void generate_all_keys(PhantomContext &ctx, const PhantomSecretKey &sk, size_t num_keys) {
        num_keys_ = num_keys;
        keys_.resize(num_keys);

        printf("[KeyStore] Generating %zu keys one at a time (CPU streaming)...\n", num_keys);
        fflush(stdout);

        const auto &stream = phantom::util::global_variables::default_stream->get_stream();
        for (size_t i = 0; i < num_keys; i++) {
            PhantomRelinKey rk = sk.generate_single_galois_key(ctx, i);
            cudaStreamSynchronize(stream);
            rk.copy_to_host(keys_[i].components);
            keys_[i].valid = true;
            // rk destructor frees GPU memory

            if ((i + 1) % 10 == 0 || i == num_keys - 1) {
                size_t free_mem, total_mem;
                cudaMemGetInfo(&free_mem, &total_mem);
                printf("[KeyStore] Generated %zu/%zu keys (GPU free: %.2f GB)\n",
                       i + 1, num_keys, free_mem / (1024.0*1024.0*1024.0));
                fflush(stdout);
            }
        }

        // Pre-allocate GPU buffers using SAME stream-ordered pool as Phantom
        // (reuse 'stream' from above in generate_all_keys)
        dnum_ = keys_[0].components.size();
        gpu_buffers_.resize(dnum_);
        buffer_sizes_.resize(dnum_);
        std::vector<uint64_t *> ptrs(dnum_);
        for (size_t c = 0; c < dnum_; c++) {
            buffer_sizes_[c] = keys_[0].components[c].size() * sizeof(uint64_t);
            cudaMallocAsync(&gpu_buffers_[c], buffer_sizes_[c], stream);
            ptrs[c] = (uint64_t *)gpu_buffers_[c];
        }
        cudaMallocAsync(&gpu_ptr_array_, dnum_ * sizeof(uint64_t *), stream);
        cudaMemcpyAsync(gpu_ptr_array_, ptrs.data(), dnum_ * sizeof(uint64_t *),
                        cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        printf("[KeyStore] Pre-allocated %zu GPU key buffers (%.2f GB reusable, stream-ordered pool)\n",
               dnum_, dnum_ * buffer_sizes_[0] / (1024.0*1024.0*1024.0));
    }

    void load_key_to_gpu(size_t idx, PhantomGaloisKey &galois_keys) {
        static int loads = 0; loads++;
        fprintf(stderr, "[KS] load #%d idx=%zu prev=%d dnum=%zu bufsz=%zu\n",
                loads, idx, last_loaded_idx_, dnum_, buffer_sizes_.empty()?0:buffer_sizes_[0]);
        fflush(stderr);

        if ((int)idx == last_loaded_idx_) { fprintf(stderr, "[KS] already loaded\n"); return; }

        const auto &stream = phantom::util::global_variables::default_stream->get_stream();
        cudaStreamSynchronize(stream);

        for (size_t c = 0; c < dnum_; c++) {
            cudaError_t e = cudaMemcpyAsync(gpu_buffers_[c], keys_[idx].components[c].data(),
                            buffer_sizes_[c], cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) fprintf(stderr, "[KS] memcpy c=%zu fail: %s\n", c, cudaGetErrorString(e));
        }
        cudaStreamSynchronize(stream);

        std::vector<size_t> elem_counts(dnum_);
        for (size_t c = 0; c < dnum_; c++) elem_counts[c] = buffer_sizes_[c] / sizeof(uint64_t);

        PhantomRelinKey &slot = galois_keys.get_mutable_relin_key(idx);
        slot.set_external_buffers(gpu_buffers_, elem_counts, (uint64_t **)gpu_ptr_array_);
        fprintf(stderr, "[KS] load done\n"); fflush(stderr);

        last_loaded_idx_ = (int)idx;
    }

    size_t num_keys() const { return num_keys_; }
};
