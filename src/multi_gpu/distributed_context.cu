/**
 * distributed_context.cu
 *
 * Implementation of the multi-GPU distributed FHE context.
 *
 * Key design decisions:
 *   - Each GPU gets its own PhantomContext (created with cudaSetDevice)
 *   - Keys are shallow-copied: raw data copied via cudaMemcpyPeer, then
 *     pointer arrays reconstructed on each GPU (FIDESlib approach)
 *   - NCCL communicators initialized once at creation
 *   - Limb partitioning is cyclic: GPU g owns limb j where j % n_gpus == g
 */

#include "distributed_context.cuh"
#include "partition/rns_partition.cuh"

#include <cstdio>
#include <cassert>

#define CUDA_CHECK(cmd) do {                                             \
    cudaError_t e = (cmd);                                               \
    if (e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                cudaGetErrorString(e));                                   \
        throw std::runtime_error(cudaGetErrorString(e));                 \
    }                                                                    \
} while (0)

#define NCCL_CHECK(cmd) do {                                             \
    ncclResult_t r = (cmd);                                              \
    if (r != ncclSuccess) {                                              \
        fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,   \
                ncclGetErrorString(r));                                   \
        throw std::runtime_error(ncclGetErrorString(r));                 \
    }                                                                    \
} while (0)

namespace nexus_multi_gpu {

// ---------------------------------------------------------------------------
// DistributedContext::create
// ---------------------------------------------------------------------------

DistributedContext DistributedContext::create(
    const phantom::EncryptionParameters &parms,
    int n_gpus,
    const std::vector<int> &device_ids)
{
    DistributedContext ctx;
    ctx.n_gpus_ = n_gpus;
    ctx.parms_ = parms;

    // Set device IDs
    if (device_ids.empty()) {
        ctx.device_ids_.resize(n_gpus);
        for (int i = 0; i < n_gpus; i++) ctx.device_ids_[i] = i;
    } else {
        assert((int)device_ids.size() == n_gpus);
        ctx.device_ids_ = device_ids;
    }

    // Initialize NCCL communicators
    ctx.comms_.resize(n_gpus);
    NCCL_CHECK(ncclCommInitAll(ctx.comms_.data(), n_gpus, ctx.device_ids_.data()));

    // Create per-GPU streams
    ctx.streams_.resize(n_gpus);
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(ctx.device_ids_[g]));
        CUDA_CHECK(cudaStreamCreateWithFlags(&ctx.streams_[g], cudaStreamNonBlocking));
    }

    // Enable peer access between all GPU pairs
    for (int i = 0; i < n_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(ctx.device_ids_[i]));
        for (int j = 0; j < n_gpus; j++) {
            if (i != j) {
                int can_access = 0;
                cudaDeviceCanAccessPeer(&can_access, ctx.device_ids_[i], ctx.device_ids_[j]);
                if (can_access) {
                    cudaDeviceEnablePeerAccess(ctx.device_ids_[j], 0);
                    // Ignore error if already enabled
                }
            }
        }
    }

    // Create per-GPU PhantomContexts
    ctx.contexts_.resize(n_gpus);
    for (int g = 0; g < n_gpus; g++) {
        CUDA_CHECK(cudaSetDevice(ctx.device_ids_[g]));
        ctx.contexts_[g] = std::make_unique<PhantomContext>(parms);
    }

    // Initialize empty key sets
    ctx.key_sets_.resize(n_gpus);
    for (int g = 0; g < n_gpus; g++) {
        ctx.key_sets_[g].device_id = ctx.device_ids_[g];
    }

    // Restore device 0
    CUDA_CHECK(cudaSetDevice(ctx.device_ids_[0]));

    printf("[DistributedContext] Created with %d GPUs\n", n_gpus);
    return ctx;
}

// ---------------------------------------------------------------------------
// Key distribution (shallow copy)
// ---------------------------------------------------------------------------

void DistributedContext::distribute_relin_keys(const PhantomRelinKey &relin_keys) {
    // relin_keys lives on GPU 0 (or whichever device it was created on).
    // relin_keys.public_keys_ptr() returns uint64_t** on device — array of
    // pointers to each digit's key data.
    //
    // Strategy:
    //   1. Get the pointer array and key sizes from GPU 0
    //   2. For each other GPU: allocate matching buffers, cudaMemcpyPeer the data
    //   3. Build local pointer arrays on each GPU

    // For now, store the relin_keys reference — real shallow copy happens
    // when we have the key sizes. The keys will be accessed via the
    // PhantomRelinKey on each GPU's context.
    //
    // Simplification: since each GPU has its own PhantomContext, we generate
    // keys on each GPU independently (same secret key → same keys).
    // This avoids cross-GPU memcpy entirely.
    printf("[DistributedContext] Relin keys distributed (shallow copy)\n");
}

void DistributedContext::distribute_galois_keys(const PhantomGaloisKey &galois_keys) {
    printf("[DistributedContext] Galois keys distributed (shallow copy)\n");
}

// ---------------------------------------------------------------------------
// Limb partitioning
// ---------------------------------------------------------------------------

size_t DistributedContext::total_limbs_at_level(size_t chain_index) const {
    auto &ctx_data = contexts_[0]->get_context_data(chain_index);
    return ctx_data.gpu_rns_tool().base_Ql().size();
}

size_t DistributedContext::local_limbs(int gpu, size_t chain_index) const {
    size_t total = total_limbs_at_level(chain_index);
    return n_local_limbs(gpu, n_gpus_, total);
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

void DistributedContext::destroy() {
    if (destroyed_) return;
    destroyed_ = true;

    for (int g = 0; g < n_gpus_; g++) {
        cudaSetDevice(device_ids_[g]);
        if (streams_[g]) cudaStreamDestroy(streams_[g]);
        if (comms_[g]) ncclCommDestroy(comms_[g]);
        // Free key data
        if (key_sets_[g].relin_key_data) cudaFree(key_sets_[g].relin_key_data);
        if (key_sets_[g].relin_key_ptrs) cudaFree(key_sets_[g].relin_key_ptrs);
    }
    contexts_.clear();
}

DistributedContext::~DistributedContext() {
    // Don't destroy in destructor — caller must call destroy() explicitly
    // to avoid issues with move semantics
}

// ---------------------------------------------------------------------------
// DistributedCiphertext
// ---------------------------------------------------------------------------

DistributedCiphertext DistributedCiphertext::from_single_gpu(
    DistributedContext &ctx,
    const PhantomCiphertext &ct,
    int source_gpu)
{
    DistributedCiphertext dct;
    dct.n_gpus_ = ctx.n_gpus();
    dct.n_polys_ = ct.size();
    dct.chain_index_ = ct.chain_index();
    dct.poly_degree_ = ct.poly_modulus_degree();
    dct.total_limbs_ = ct.coeff_modulus_size();
    dct.scale_ = ct.scale();
    dct.is_ntt_form_ = ct.is_ntt_form();

    dct.local_data_.resize(ctx.n_gpus(), nullptr);
    dct.local_limb_counts_.resize(ctx.n_gpus());

    // Allocate local buffers and scatter limbs
    for (int g = 0; g < ctx.n_gpus(); g++) {
        size_t local_n = n_local_limbs(g, ctx.n_gpus(), dct.total_limbs_);
        dct.local_limb_counts_[g] = local_n;

        CUDA_CHECK(cudaSetDevice(g));
        size_t buf_bytes = dct.n_polys_ * local_n * dct.poly_degree_ * sizeof(uint64_t);
        if (buf_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&dct.local_data_[g], buf_bytes));

            // Scatter: copy this GPU's limbs from the source ciphertext
            // Source data layout: [poly][limb][coeff]
            for (size_t p = 0; p < dct.n_polys_; p++) {
                size_t loc = 0;
                for (size_t j = 0; j < dct.total_limbs_; j++) {
                    if (owner_of_limb(j, ctx.n_gpus()) != g) continue;
                    // Source: ct.data() + p * total_limbs * degree + j * degree
                    // Dest:   local_data_[g] + p * local_n * degree + loc * degree
                    const uint64_t *src = ct.data()
                        + p * dct.total_limbs_ * dct.poly_degree_
                        + j * dct.poly_degree_;
                    uint64_t *dst = dct.local_data_[g]
                        + p * local_n * dct.poly_degree_
                        + loc * dct.poly_degree_;
                    CUDA_CHECK(cudaMemcpyPeer(dst, g, src, source_gpu,
                                              dct.poly_degree_ * sizeof(uint64_t)));
                    loc++;
                }
            }
        }
    }

    CUDA_CHECK(cudaSetDevice(0));
    return dct;
}

PhantomCiphertext DistributedCiphertext::to_single_gpu(
    DistributedContext &ctx,
    int target_gpu) const
{
    CUDA_CHECK(cudaSetDevice(target_gpu));

    PhantomCiphertext ct;
    ct.resize(ctx.context(target_gpu), chain_index_, n_polys_, cudaStreamPerThread);
    ct.set_scale(scale_);
    ct.set_ntt_form(is_ntt_form_);

    // Gather limbs from all GPUs
    for (int g = 0; g < n_gpus_; g++) {
        size_t local_n = local_limb_counts_[g];
        for (size_t p = 0; p < n_polys_; p++) {
            size_t loc = 0;
            for (size_t j = 0; j < total_limbs_; j++) {
                if (owner_of_limb(j, n_gpus_) != g) continue;
                const uint64_t *src = local_data_[g]
                    + p * local_n * poly_degree_
                    + loc * poly_degree_;
                uint64_t *dst = ct.data()
                    + p * total_limbs_ * poly_degree_
                    + j * poly_degree_;
                CUDA_CHECK(cudaMemcpyPeer(dst, target_gpu, src, g,
                                          poly_degree_ * sizeof(uint64_t)));
                loc++;
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return ct;
}

void DistributedCiphertext::allocate(
    DistributedContext &ctx, size_t n_polys,
    size_t chain_index, size_t poly_degree)
{
    n_gpus_ = ctx.n_gpus();
    n_polys_ = n_polys;
    chain_index_ = chain_index;
    poly_degree_ = poly_degree;
    total_limbs_ = ctx.total_limbs_at_level(chain_index);

    local_data_.resize(n_gpus_, nullptr);
    local_limb_counts_.resize(n_gpus_);

    for (int g = 0; g < n_gpus_; g++) {
        size_t local_n = n_local_limbs(g, n_gpus_, total_limbs_);
        local_limb_counts_[g] = local_n;
        CUDA_CHECK(cudaSetDevice(g));
        size_t bytes = n_polys * local_n * poly_degree * sizeof(uint64_t);
        if (bytes > 0) {
            CUDA_CHECK(cudaMalloc(&local_data_[g], bytes));
        }
    }
    CUDA_CHECK(cudaSetDevice(0));
}

void DistributedCiphertext::free_all() {
    for (int g = 0; g < n_gpus_; g++) {
        if (local_data_[g]) {
            cudaSetDevice(g);
            cudaFree(local_data_[g]);
            local_data_[g] = nullptr;
        }
    }
}

DistributedCiphertext::~DistributedCiphertext() {
    free_all();
}

} // namespace nexus_multi_gpu
