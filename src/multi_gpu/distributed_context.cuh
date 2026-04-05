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
    // Create distributed context across n_gpus GPUs.
    // Initializes NCCL, creates per-GPU PhantomContexts, distributes keys.
    static DistributedContext create(
        const phantom::EncryptionParameters &parms,
        int n_gpus,
        const std::vector<int> &device_ids = {}  // default: 0..n_gpus-1
    );

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

    // Cleanup
    void destroy();
    ~DistributedContext();

private:
    int n_gpus_ = 0;
    std::vector<int> device_ids_;
    std::vector<std::unique_ptr<PhantomContext>> contexts_;
    std::vector<ncclComm_t> comms_;
    std::vector<cudaStream_t> streams_;
    std::vector<GpuKeySet> key_sets_;
    phantom::EncryptionParameters parms_;
    bool destroyed_ = false;
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

    void set_chain_index(size_t idx) { chain_index_ = idx; }
    void set_scale(double s) { scale_ = s; }

    // Allocate local buffers on each GPU
    void allocate(DistributedContext &ctx, size_t n_polys,
                  size_t chain_index, size_t poly_degree);

    // Free GPU memory
    void free_all();
    ~DistributedCiphertext();

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
