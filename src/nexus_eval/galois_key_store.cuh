/**
 * galois_key_store.cuh
 *
 * CPU-side Galois key storage for N=65536 bootstrap where all keys
 * don't fit in GPU memory simultaneously.
 *
 * Strategy: generate keys one at a time, save to CPU pinned memory,
 * load on demand during rotation. GPU holds at most 2 keys at a time
 * (double-buffered: one being used by compute, one being prefetched).
 *
 * The prefetch path uses a dedicated copy stream so H→D for the next
 * rotation overlaps with the current rotation kernel on default_stream.
 * Synchronization is event-based:
 *   copy_done_event_[s]    fires when H→D for slot s finishes
 *   compute_done_event_[s] fires when the rotation that read slot s finishes
 */

#pragma once

#include <vector>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#include "context.cuh"
#include "secretkey.h"
#include "util/globals.h"
#include "nvtx_tracer.cuh"

class GaloisKeyStore {
public:
    struct HostKey {
        std::vector<std::vector<uint64_t>> components; // dnum vectors of uint64 data
        bool valid = false;
    };

private:
    static constexpr int N_SLOTS = 2;

    std::vector<HostKey> keys_;
    size_t num_keys_ = 0;

    // Two parallel sets of GPU buffers (one per slot)
    std::vector<void *> gpu_buffers_[N_SLOTS];   // [slot][component]
    std::vector<size_t> buffer_sizes_;           // shared: same component sizes for both slots
    void *gpu_ptr_array_[N_SLOTS] = {nullptr, nullptr}; // device array of dnum uint64_t* per slot

    int  slot_resident_idx_[N_SLOTS] = {-1, -1};         // which key idx is in each slot
    bool compute_event_recorded_[N_SLOTS] = {false, false};

    cudaEvent_t copy_done_event_[N_SLOTS]    = {nullptr, nullptr};
    cudaEvent_t compute_done_event_[N_SLOTS] = {nullptr, nullptr};
    cudaStream_t copy_stream_ = nullptr;

    int    current_slot_    = -1;   // slot bound by most recent load_key_to_gpu
    int    last_loaded_idx_ = -1;
    size_t dnum_            = 0;
    // Stream captured at generate_all_keys() time — NOT read at destruction time.
    // Reading default_stream in the destructor is unsafe: the stream may have been
    // reset (and the old stream destroyed) by a subsequent PhantomContext constructor
    // between generate_all_keys() and destruction.
    cudaStream_t alloc_stream_ = nullptr;

public:
    GaloisKeyStore() = default;

    ~GaloisKeyStore() {
        if (alloc_stream_ == nullptr) return;
        // Unregister pinned host memory so future allocators can reuse the pages.
        for (auto &k : keys_) {
            if (!k.valid) continue;
            for (auto &comp : k.components) {
                if (!comp.empty()) cudaHostUnregister(comp.data());
            }
        }
        for (int s = 0; s < N_SLOTS; s++) {
            for (auto p : gpu_buffers_[s]) if (p) cudaFreeAsync(p, alloc_stream_);
            if (gpu_ptr_array_[s])    cudaFreeAsync(gpu_ptr_array_[s], alloc_stream_);
            if (copy_done_event_[s])    cudaEventDestroy(copy_done_event_[s]);
            if (compute_done_event_[s]) cudaEventDestroy(compute_done_event_[s]);
        }
        if (copy_stream_) cudaStreamDestroy(copy_stream_);
    }

    // Generate all keys one at a time, saving each to CPU and freeing GPU
    void generate_all_keys(PhantomContext &ctx, const PhantomSecretKey &sk, size_t num_keys) {
        num_keys_ = num_keys;
        keys_.resize(num_keys);

        printf("[KeyStore] Generating %zu keys one at a time (CPU streaming, double-buffered)...\n", num_keys);
        fflush(stdout);

        // Capture stream ONCE here; used for all allocs and for the destructor free.
        alloc_stream_ = phantom::util::global_variables::default_stream->get_stream();
        const cudaStream_t &stream = alloc_stream_;
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

        // Pre-allocate TWO sets of GPU buffers (double buffer for prefetch overlap)
        dnum_ = keys_[0].components.size();
        buffer_sizes_.resize(dnum_);
        for (size_t c = 0; c < dnum_; c++) {
            buffer_sizes_[c] = keys_[0].components[c].size() * sizeof(uint64_t);
        }
        for (int s = 0; s < N_SLOTS; s++) {
            gpu_buffers_[s].resize(dnum_);
            std::vector<uint64_t *> ptrs(dnum_);
            for (size_t c = 0; c < dnum_; c++) {
                cudaMallocAsync(&gpu_buffers_[s][c], buffer_sizes_[c], stream);
                ptrs[c] = (uint64_t *)gpu_buffers_[s][c];
            }
            cudaMallocAsync(&gpu_ptr_array_[s], dnum_ * sizeof(uint64_t *), stream);
            cudaMemcpyAsync(gpu_ptr_array_[s], ptrs.data(), dnum_ * sizeof(uint64_t *),
                            cudaMemcpyHostToDevice, stream);
        }
        cudaStreamSynchronize(stream);

        // Dedicated non-blocking copy stream so H→D can run concurrently with
        // rotation kernels on the default stream.
        cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking);
        for (int s = 0; s < N_SLOTS; s++) {
            cudaEventCreateWithFlags(&copy_done_event_[s],    cudaEventDisableTiming);
            cudaEventCreateWithFlags(&compute_done_event_[s], cudaEventDisableTiming);
        }

        printf("[KeyStore] Pre-allocated %d×%zu GPU key buffers (%.2f GB total, double-buffered)\n",
               N_SLOTS, dnum_,
               N_SLOTS * dnum_ * buffer_sizes_[0] / (1024.0*1024.0*1024.0));
        fflush(stdout);

        // Pin all host key components so cudaMemcpyAsync on copy_stream_ is truly
        // async. Without this, each "async" H→D stages through a CUDA-internal
        // pinned bounce buffer and effectively blocks the copy stream.
        size_t pinned_bytes = 0;
        size_t pinned_failures = 0;
        for (size_t i = 0; i < num_keys_; i++) {
            for (size_t c = 0; c < dnum_; c++) {
                void  *ptr = keys_[i].components[c].data();
                size_t sz  = keys_[i].components[c].size() * sizeof(uint64_t);
                cudaError_t e = cudaHostRegister(ptr, sz, cudaHostRegisterDefault);
                if (e == cudaSuccess) pinned_bytes += sz;
                else                  pinned_failures++;
            }
        }
        printf("[KeyStore] Pinned %.2f GB of host key memory (%zu failures)\n",
               pinned_bytes / (1024.0*1024.0*1024.0), pinned_failures);
        fflush(stdout);
    }

    // Kick an async H→D for key idx into the spare slot.
    // Idempotent: if idx is already resident in either slot, returns immediately.
    void prefetch(size_t idx) {
        NVTX_SCOPE_FMT("ks_prefetch idx=%zu", idx);
        if (idx >= num_keys_ || !keys_[idx].valid) return;
        if (copy_stream_ == nullptr) return;  // not initialized yet
        // Skip if already resident in either slot
        for (int s = 0; s < N_SLOTS; s++) if (slot_resident_idx_[s] == (int)idx) return;

        // Pick the slot NOT currently in use by compute
        int s = (current_slot_ < 0) ? 0 : (1 - current_slot_);

        // Wait for the previous rotation that read slot s to complete before overwrite
        if (compute_event_recorded_[s]) {
            cudaStreamWaitEvent(copy_stream_, compute_done_event_[s], 0);
        }
        for (size_t c = 0; c < dnum_; c++) {
            cudaMemcpyAsync(gpu_buffers_[s][c], keys_[idx].components[c].data(),
                            buffer_sizes_[c], cudaMemcpyHostToDevice, copy_stream_);
        }
        cudaEventRecord(copy_done_event_[s], copy_stream_);
        slot_resident_idx_[s] = (int)idx;
    }

    void load_key_to_gpu(size_t idx, PhantomGaloisKey &galois_keys) {
        NVTX_SCOPE_FMT("ks_load idx=%zu", idx);
        if ((int)idx == last_loaded_idx_) return;

        const cudaStream_t &stream = alloc_stream_;

        // Find slot already holding this key (set by a prior prefetch)
        int s = -1;
        for (int i = 0; i < N_SLOTS; i++) if (slot_resident_idx_[i] == (int)idx) { s = i; break; }

        if (s < 0) {
            // Cache miss → synchronous prefetch into spare slot
            prefetch(idx);
            for (int i = 0; i < N_SLOTS; i++) if (slot_resident_idx_[i] == (int)idx) { s = i; break; }
            if (s < 0) {
                fprintf(stderr, "[KS] prefetch failed for idx=%zu\n", idx);
                return;
            }
        }

        // Compute stream waits for H→D into slot s to finish (no host sync)
        cudaStreamWaitEvent(stream, copy_done_event_[s], 0);

        // If we previously bound a different slot, record an event on the compute
        // stream marking that the prior rotation has been queued — the next prefetch
        // into that slot must wait on this event before overwriting.
        if (current_slot_ >= 0 && current_slot_ != s) {
            cudaEventRecord(compute_done_event_[current_slot_], stream);
            compute_event_recorded_[current_slot_] = true;
        }

        std::vector<size_t> elem_counts(dnum_);
        for (size_t c = 0; c < dnum_; c++) elem_counts[c] = buffer_sizes_[c] / sizeof(uint64_t);

        PhantomRelinKey &slot = galois_keys.get_mutable_relin_key(idx);
        slot.set_external_buffers(gpu_buffers_[s], elem_counts, (uint64_t **)gpu_ptr_array_[s]);

        current_slot_    = s;
        last_loaded_idx_ = (int)idx;
    }

    size_t num_keys() const { return num_keys_; }
};
