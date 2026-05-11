/**
 * galois_key_store.cuh
 *
 * CPU-side Galois key storage for N=65536 bootstrap where all keys
 * don't fit in GPU memory simultaneously.
 *
 * Strategy: generate keys one at a time, save to CPU pinned memory,
 * load on demand during rotation. GPU holds at most cache_size_ keys at a
 * time, managed by an N-slot LRU cache (N is a constructor parameter,
 * default kDefaultCacheSize = 10).
 *
 * Each cache slot has its own H→D copy event so that the rotation kernel
 * (on default_stream) can wait only for the slot it actually consumes,
 * while other slots may still be receiving prefetched data on copy_stream_.
 * A per-slot compute_done_event marks the rotation that last consumed the
 * slot — eviction blocks copy_stream_ on it before overwriting.
 *
 * T-LRU note: the previous implementation used a fixed compile-time
 * `kCacheSize = 10`. That constant is preserved as the default value of the
 * constructor parameter so existing call sites (all of which default-
 * construct GaloisKeyStore) keep their current 10-slot behaviour.
 */

#pragma once

#include <vector>
#include <list>
#include <unordered_map>
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

    // Default LRU slot count. Kept as a public constant so call sites that
    // want the historical N=10 behaviour can still reference it by name.
    // The active slot count is `cache_size_` (constructor argument).
    static constexpr size_t kDefaultCacheSize = 10;

private:
    // N-slot LRU cache. Set once via the constructor; never resized.
    size_t cache_size_ = kDefaultCacheSize;

    std::vector<HostKey> keys_;
    size_t num_keys_ = 0;

    // cache_size_ parallel sets of GPU buffers (one per LRU slot)
    std::vector<std::vector<void *>> gpu_buffers_;   // [slot][component]
    std::vector<size_t> buffer_sizes_;               // shared across slots
    std::vector<void *>  gpu_ptr_array_;             // [slot] device array of dnum uint64_t*

    std::vector<int>  slot_resident_idx_;            // [slot] which key idx (or -1)
    std::vector<bool> compute_event_recorded_;       // [slot]

    std::vector<cudaEvent_t> copy_done_event_;       // [slot]
    std::vector<cudaEvent_t> compute_done_event_;    // [slot]
    cudaStream_t copy_stream_ = nullptr;

    // LRU bookkeeping. lru_order_: front = MRU, back = LRU. step_to_slot_ maps
    // resident key idx → slot index. slot_to_iter_ allows O(1) splice on hit.
    std::list<int> lru_order_;
    std::unordered_map<int, size_t> step_to_slot_;
    std::vector<std::list<int>::iterator> slot_to_iter_;

    int    current_slot_    = -1;   // slot bound by most recent load_key_to_gpu
    int    last_loaded_idx_ = -1;
    size_t dnum_            = 0;
    // Stream captured at generate_all_keys() time — NOT read at destruction time.
    // Reading default_stream in the destructor is unsafe: the stream may have been
    // reset (and the old stream destroyed) by a subsequent PhantomContext constructor
    // between generate_all_keys() and destruction.
    cudaStream_t alloc_stream_ = nullptr;

    void touch_mru(size_t slot) {
        lru_order_.splice(lru_order_.begin(), lru_order_, slot_to_iter_[slot]);
    }

public:
    // Construct a GaloisKeyStore with `cache_size` GPU LRU slots.
    // Defaults to kDefaultCacheSize (10) which preserves the historical
    // ping-pong-replacement behaviour from the pre-T-LRU code.
    explicit GaloisKeyStore(size_t cache_size = kDefaultCacheSize)
        : cache_size_(cache_size == 0 ? kDefaultCacheSize : cache_size) {}

    // Rule of Five: this class owns GPU buffers, events, a stream, and pinned
    // host registrations. Copying would shallow-copy device pointers and
    // double-free; the user-declared dtor also suppresses implicit moves so we
    // re-enable them explicitly.
    GaloisKeyStore(const GaloisKeyStore &) = delete;
    GaloisKeyStore &operator=(const GaloisKeyStore &) = delete;
    GaloisKeyStore(GaloisKeyStore &&) noexcept = default;
    GaloisKeyStore &operator=(GaloisKeyStore &&) noexcept = default;

    size_t cache_size() const { return cache_size_; }

    ~GaloisKeyStore() {
        if (alloc_stream_ == nullptr) return;
        // Unregister pinned host memory so future allocators can reuse the pages.
        for (auto &k : keys_) {
            if (!k.valid) continue;
            for (auto &comp : k.components) {
                if (!comp.empty()) cudaHostUnregister(comp.data());
            }
        }
        for (size_t s = 0; s < gpu_buffers_.size(); s++) {
            for (auto p : gpu_buffers_[s]) if (p) cudaFreeAsync(p, alloc_stream_);
            if (s < gpu_ptr_array_.size() && gpu_ptr_array_[s])
                cudaFreeAsync(gpu_ptr_array_[s], alloc_stream_);
            if (s < copy_done_event_.size()    && copy_done_event_[s])
                cudaEventDestroy(copy_done_event_[s]);
            if (s < compute_done_event_.size() && compute_done_event_[s])
                cudaEventDestroy(compute_done_event_[s]);
        }
        if (copy_stream_) cudaStreamDestroy(copy_stream_);
    }

    // Generate all keys one at a time, saving each to CPU and freeing GPU
    void generate_all_keys(PhantomContext &ctx, const PhantomSecretKey &sk, size_t num_keys) {
        num_keys_ = num_keys;
        keys_.resize(num_keys);

        printf("[KeyStore] Generating %zu keys one at a time (CPU streaming, %zu-slot LRU)...\n",
               num_keys, cache_size_);
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

        // Pre-allocate cache_size_ sets of GPU buffers up front (no cudaMalloc in hot path).
        // T-LRU constraint: workspace memory must be preallocated at construction-time
        // (here = generate_all_keys, the earliest moment we know the per-key shard size).
        dnum_ = keys_[0].components.size();
        buffer_sizes_.resize(dnum_);
        for (size_t c = 0; c < dnum_; c++) {
            buffer_sizes_[c] = keys_[0].components[c].size() * sizeof(uint64_t);
        }

        gpu_buffers_.assign(cache_size_, std::vector<void *>(dnum_, nullptr));
        gpu_ptr_array_.assign(cache_size_, nullptr);
        slot_resident_idx_.assign(cache_size_, -1);
        compute_event_recorded_.assign(cache_size_, false);
        copy_done_event_.assign(cache_size_, nullptr);
        compute_done_event_.assign(cache_size_, nullptr);
        slot_to_iter_.assign(cache_size_, lru_order_.end());

        // Seed lru_order_ with all slots in MRU-to-LRU order so that
        // `lru_order_.back()` on the first prefetch returns a valid slot
        // (slot cache_size_-1) instead of dereferencing end() — that was
        // a regression introduced when the 2-slot fixed buffer was
        // replaced with the N-slot LRU. The slot ordering here is
        // arbitrary; every slot starts with `slot_resident_idx_[s] = -1`,
        // so the first miss simply picks the last entry and overwrites it.
        for (size_t s = 0; s < cache_size_; s++) {
            lru_order_.push_back(static_cast<int>(s));
            auto it = std::prev(lru_order_.end());
            slot_to_iter_[s] = it;
        }

        for (size_t s = 0; s < cache_size_; s++) {
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
        for (size_t s = 0; s < cache_size_; s++) {
            cudaEventCreateWithFlags(&copy_done_event_[s],    cudaEventDisableTiming);
            cudaEventCreateWithFlags(&compute_done_event_[s], cudaEventDisableTiming);
        }

        printf("[KeyStore] Pre-allocated %zu×%zu GPU key buffers (%.2f GB total, %zu-slot LRU)\n",
               cache_size_, dnum_,
               cache_size_ * dnum_ * buffer_sizes_[0] / (1024.0*1024.0*1024.0),
               cache_size_);
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

    // Kick an async H→D for key idx. Idempotent: if idx is already resident,
    // touches it MRU and returns. On miss, evicts LRU slot and async-copies.
    void prefetch(size_t idx) {
        if (idx >= num_keys_ || !keys_[idx].valid) return;
        if (copy_stream_ == nullptr) return;  // not initialized yet

        auto it = step_to_slot_.find((int)idx);
        if (it != step_to_slot_.end()) {
            NVTX_SCOPE_FMT("ks_load_hit idx=%zu", idx);
            touch_mru(it->second);
            return;
        }

        NVTX_SCOPE_FMT("ks_load_miss idx=%zu", idx);
        // Evict LRU (back of list) — guaranteed to be currently unbound by compute
        // because we only place a slot at front when we use it. If somehow the LRU
        // happens to equal current_slot_, fall back to the next-oldest unbound slot.
        size_t s = (size_t)lru_order_.back();
        if ((int)s == current_slot_ && lru_order_.size() > 1) {
            auto rit = std::next(lru_order_.rbegin());
            s = (size_t)*rit;
        }

        // Drop the old resident's mapping (if any) before overwriting the slot.
        if (slot_resident_idx_[s] >= 0) {
            step_to_slot_.erase(slot_resident_idx_[s]);
        }

        // Wait for the previous rotation that read slot s to finish before overwrite
        if (compute_event_recorded_[s]) {
            cudaStreamWaitEvent(copy_stream_, compute_done_event_[s], 0);
        }
        for (size_t c = 0; c < dnum_; c++) {
            cudaMemcpyAsync(gpu_buffers_[s][c], keys_[idx].components[c].data(),
                            buffer_sizes_[c], cudaMemcpyHostToDevice, copy_stream_);
        }
        cudaEventRecord(copy_done_event_[s], copy_stream_);
        slot_resident_idx_[s] = (int)idx;

        // Promote slot s to MRU and update the step→slot map.
        if (slot_to_iter_[s] != lru_order_.end()) {
            lru_order_.erase(slot_to_iter_[s]);
        }
        lru_order_.push_front((int)s);
        slot_to_iter_[s] = lru_order_.begin();
        step_to_slot_[(int)idx] = s;
    }

    void load_key_to_gpu(size_t idx, PhantomGaloisKey &galois_keys) {
        NVTX_SCOPE_FMT("ks_load idx=%zu", idx);
        if ((int)idx == last_loaded_idx_) return;

        const cudaStream_t &stream = alloc_stream_;

        auto it = step_to_slot_.find((int)idx);
        size_t s;
        if (it == step_to_slot_.end()) {
            // Cache miss → synchronous prefetch (will allocate slot & start H→D)
            prefetch(idx);
            it = step_to_slot_.find((int)idx);
            if (it == step_to_slot_.end()) {
                fprintf(stderr, "[KS] prefetch failed for idx=%zu\n", idx);
                return;
            }
            s = it->second;
        } else {
            s = it->second;
            touch_mru(s);
        }

        // Compute stream waits for H→D into slot s to finish (no host sync)
        cudaStreamWaitEvent(stream, copy_done_event_[s], 0);

        // If we previously bound a different slot, record an event on the compute
        // stream marking that the prior rotation has been queued — the next prefetch
        // into that slot must wait on this event before overwriting.
        if (current_slot_ >= 0 && (size_t)current_slot_ != s) {
            cudaEventRecord(compute_done_event_[current_slot_], stream);
            compute_event_recorded_[current_slot_] = true;
        }

        std::vector<size_t> elem_counts(dnum_);
        for (size_t c = 0; c < dnum_; c++) elem_counts[c] = buffer_sizes_[c] / sizeof(uint64_t);

        PhantomRelinKey &slot = galois_keys.get_mutable_relin_key(idx);
        slot.set_external_buffers(gpu_buffers_[s], elem_counts, (uint64_t **)gpu_ptr_array_[s]);

        current_slot_    = (int)s;
        last_loaded_idx_ = (int)idx;
    }

    size_t num_keys() const { return num_keys_; }
};
