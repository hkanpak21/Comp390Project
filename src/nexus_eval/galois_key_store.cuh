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
        for (auto p : gpu_buffers_) if (p) cudaFree(p);
        if (gpu_ptr_array_) cudaFree(gpu_ptr_array_);
    }

    // Generate all keys one at a time, saving each to CPU and freeing GPU
    void generate_all_keys(PhantomContext &ctx, const PhantomSecretKey &sk, size_t num_keys) {
        num_keys_ = num_keys;
        keys_.resize(num_keys);

        printf("[KeyStore] Generating %zu keys one at a time (CPU streaming)...\n", num_keys);
        fflush(stdout);

        for (size_t i = 0; i < num_keys; i++) {
            PhantomRelinKey rk = sk.generate_single_galois_key(ctx, i);
            cudaDeviceSynchronize();
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

        // Pre-allocate GPU buffers (reused for all keys)
        dnum_ = keys_[0].components.size();
        gpu_buffers_.resize(dnum_);
        buffer_sizes_.resize(dnum_);
        std::vector<uint64_t *> ptrs(dnum_);
        for (size_t c = 0; c < dnum_; c++) {
            buffer_sizes_[c] = keys_[0].components[c].size() * sizeof(uint64_t);
            cudaMalloc(&gpu_buffers_[c], buffer_sizes_[c]);
            ptrs[c] = (uint64_t *)gpu_buffers_[c];
        }
        cudaMalloc(&gpu_ptr_array_, dnum_ * sizeof(uint64_t *));
        cudaMemcpy(gpu_ptr_array_, ptrs.data(), dnum_ * sizeof(uint64_t *),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        printf("[KeyStore] Pre-allocated %zu GPU key buffers (%.2f GB reusable)\n",
               dnum_, dnum_ * buffer_sizes_[0] / (1024.0*1024.0*1024.0));
    }

    // Load key at index `idx` — copy CPU data to the pre-allocated GPU buffers,
    // and rebuild the slot's cuda_auto_ptr wrappers to point at them (non-owning)
    void load_key_to_gpu(size_t idx, PhantomGaloisKey &galois_keys) {
        static int call_count = 0;
        call_count++;
        if (call_count <= 3 || call_count % 20 == 0) {
            printf("[KeyStore] load_key_to_gpu call #%d, idx=%zu\n", call_count, idx);
            fflush(stdout);
        }

        if ((int)idx == last_loaded_idx_) return;

        cudaDeviceSynchronize(); // wait for any in-flight kernels reading old key

        // Copy key data from CPU to pre-allocated GPU buffers (synchronous)
        for (size_t c = 0; c < dnum_; c++) {
            cudaError_t err = cudaMemcpy(gpu_buffers_[c], keys_[idx].components[c].data(),
                                          buffer_sizes_[c], cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "[KeyStore] cudaMemcpy failed idx=%zu c=%zu: %s\n",
                        idx, c, cudaGetErrorString(err));
                return;
            }
        }
        cudaDeviceSynchronize();

        PhantomRelinKey &slot = galois_keys.get_mutable_relin_key(idx);
        slot.set_external_buffers(gpu_buffers_, (uint64_t **)gpu_ptr_array_);

        last_loaded_idx_ = (int)idx;
    }

    size_t num_keys() const { return num_keys_; }
};
