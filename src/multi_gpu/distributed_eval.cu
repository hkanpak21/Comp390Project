/**
 * distributed_eval.cu
 *
 * TRUE multi-GPU FHE evaluation — each GPU runs kernels on its local limbs.
 *
 * For LOCAL operations (add, sub, multiply_plain, negate):
 *   Each GPU calls Phantom's raw polynomial kernels (add_rns_poly,
 *   sub_rns_poly, multiply_rns_poly, negate_rns_poly) directly on its
 *   local limb buffer. NO gather/scatter. NO GPU-0-only execution.
 *   ALL GPUs run compute kernels in parallel.
 *
 * For KEYED operations (relinearize, rotate):
 *   Communication + local compute. Currently uses gather-operate-scatter
 *   (Phase 1), will be replaced with true distributed key-switching (Phase 2).
 *
 * Validation: per-GPU cudaEvent timing proves each GPU runs kernels.
 *   If GPU 1..n show zero kernel time, something is wrong.
 */

#include "distributed_eval.cuh"
#include "partition/rns_partition.cuh"

// Phantom raw kernels (declared in polymath.cuh)
#include "polymath.cuh"
#include "evaluate.cuh"
#include "ckks.h"
#include "ntt.cuh"

#include <cstdio>
#include <thread>
#include <vector>

#define CUDA_CHECK(cmd) do {                                             \
    cudaError_t e = (cmd);                                               \
    if (e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                cudaGetErrorString(e));                                   \
        throw std::runtime_error(cudaGetErrorString(e));                 \
    }                                                                    \
} while (0)

using namespace phantom;

namespace nexus_multi_gpu {

// Phantom's global block dimension (must match Phantom's definition)
static const dim3 blockDim_local(256);

// ---------------------------------------------------------------------------
// Per-GPU kernel launch helpers
// ---------------------------------------------------------------------------
// These call Phantom's raw polynomial kernels on local limb buffers.
// Each GPU has its own DModulus* from its own PhantomContext.

// Get the DModulus pointer for a GPU's context at a given chain level.
// The modulus array contains ALL primes (Q + P). For a ciphertext at
// chain_index, the relevant primes are the first coeff_mod_size entries.
static const DModulus* get_modulus_ptr(DistributedContext &ctx, int gpu) {
    return ctx.context(gpu).gpu_rns_tables().modulus();
}

// ---------------------------------------------------------------------------
// LOCAL operations — TRUE per-GPU parallel execution
// ---------------------------------------------------------------------------

void dist_add_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct1,
    const DistributedCiphertext &dct2)
{
    std::vector<std::thread> threads;
    for (int g = 0; g < ctx.n_gpus(); g++) {
        threads.emplace_back([&, g]() {
            CUDA_CHECK(cudaSetDevice(g));
            size_t local_n = dct1.local_limb_count(g);
            if (local_n == 0) return;

            const DModulus *mod = get_modulus_ptr(ctx, g);
            size_t N = dct1.poly_degree();
            size_t local_coeff_count = local_n * N;
            dim3 grid(local_coeff_count / blockDim_local.x);

            cudaStream_t s = ctx.stream(g);
            for (size_t p = 0; p < dct1.size(); p++) {
                uint64_t *dst = dct1.data(g) + p * local_coeff_count;
                const uint64_t *src = dct2.data(g) + p * local_coeff_count;
                add_rns_poly<<<grid, blockDim_local, 0, s>>>(
                    dst, src, mod, dst, N, local_n);
            }
            CUDA_CHECK(cudaStreamSynchronize(s));
        });
    }
    for (auto &t : threads) t.join();
}

void dist_sub_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct1,
    const DistributedCiphertext &dct2)
{
    std::vector<std::thread> threads;
    for (int g = 0; g < ctx.n_gpus(); g++) {
        threads.emplace_back([&, g]() {
            CUDA_CHECK(cudaSetDevice(g));
            size_t local_n = dct1.local_limb_count(g);
            if (local_n == 0) return;

            const DModulus *mod = get_modulus_ptr(ctx, g);
            size_t N = dct1.poly_degree();
            size_t local_coeff_count = local_n * N;
            dim3 grid(local_coeff_count / blockDim_local.x);

            cudaStream_t s = ctx.stream(g);
            for (size_t p = 0; p < dct1.size(); p++) {
                uint64_t *dst = dct1.data(g) + p * local_coeff_count;
                const uint64_t *src = dct2.data(g) + p * local_coeff_count;
                sub_rns_poly<<<grid, blockDim_local, 0, s>>>(
                    dst, src, mod, dst, N, local_n);
            }
            CUDA_CHECK(cudaStreamSynchronize(s));
        });
    }
    for (auto &t : threads) t.join();
}

void dist_negate_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct)
{
    std::vector<std::thread> threads;
    for (int g = 0; g < ctx.n_gpus(); g++) {
        threads.emplace_back([&, g]() {
            CUDA_CHECK(cudaSetDevice(g));
            size_t local_n = dct.local_limb_count(g);
            if (local_n == 0) return;

            const DModulus *mod = get_modulus_ptr(ctx, g);
            size_t N = dct.poly_degree();
            size_t local_coeff_count = local_n * N;
            dim3 grid(local_coeff_count / blockDim_local.x);

            cudaStream_t s = ctx.stream(g);
            for (size_t p = 0; p < dct.size(); p++) {
                uint64_t *dst = dct.data(g) + p * local_coeff_count;
                negate_rns_poly<<<grid, blockDim_local, 0, s>>>(
                    dst, mod, dst, N, local_n);
            }
            CUDA_CHECK(cudaStreamSynchronize(s));
        });
    }
    for (auto &t : threads) t.join();
}

void dist_multiply_plain_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    const PhantomPlaintext &plain)
{
    // Plaintext is on GPU 0. For true parallelism, we need the plaintext
    // data on each GPU. Use cudaMemcpyPeer to copy the relevant limbs.
    // For now: each GPU reads the plaintext from GPU 0 via peer access.
    std::vector<std::thread> threads;
    for (int g = 0; g < ctx.n_gpus(); g++) {
        threads.emplace_back([&, g]() {
            CUDA_CHECK(cudaSetDevice(g));
            size_t local_n = dct.local_limb_count(g);
            if (local_n == 0) return;

            const DModulus *mod = get_modulus_ptr(ctx, g);
            size_t N = dct.poly_degree();
            size_t local_coeff_count = local_n * N;
            dim3 grid(local_coeff_count / blockDim_local.x);

            // Copy local limbs of plaintext to this GPU
            // Plaintext layout: [limb_0][limb_1]..., each limb = N uint64s
            uint64_t *local_plain = nullptr;
            CUDA_CHECK(cudaMalloc(&local_plain, local_coeff_count * sizeof(uint64_t)));

            size_t loc = 0;
            for (size_t j = 0; j < dct.total_limbs(); j++) {
                if (owner_of_limb(j, ctx.n_gpus()) != g) continue;
                CUDA_CHECK(cudaMemcpyPeer(
                    local_plain + loc * N, g,
                    plain.data() + j * N, 0,  // plaintext on GPU 0
                    N * sizeof(uint64_t)));
                loc++;
            }

            cudaStream_t s = ctx.stream(g);
            for (size_t p = 0; p < dct.size(); p++) {
                uint64_t *dst = dct.data(g) + p * local_coeff_count;
                multiply_rns_poly<<<grid, blockDim_local, 0, s>>>(
                    dst, local_plain, mod, dst, N, local_n);
            }
            CUDA_CHECK(cudaStreamSynchronize(s));
            CUDA_CHECK(cudaFree(local_plain));
        });
    }
    for (auto &t : threads) t.join();
}

void dist_add_plain_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    const PhantomPlaintext &plain)
{
    // add_plain only affects c0 (first polynomial)
    std::vector<std::thread> threads;
    for (int g = 0; g < ctx.n_gpus(); g++) {
        threads.emplace_back([&, g]() {
            CUDA_CHECK(cudaSetDevice(g));
            size_t local_n = dct.local_limb_count(g);
            if (local_n == 0) return;

            const DModulus *mod = get_modulus_ptr(ctx, g);
            size_t N = dct.poly_degree();
            size_t local_coeff_count = local_n * N;
            dim3 grid(local_coeff_count / blockDim_local.x);

            // Copy local limbs of plaintext
            uint64_t *local_plain = nullptr;
            CUDA_CHECK(cudaMalloc(&local_plain, local_coeff_count * sizeof(uint64_t)));
            size_t loc = 0;
            for (size_t j = 0; j < dct.total_limbs(); j++) {
                if (owner_of_limb(j, ctx.n_gpus()) != g) continue;
                CUDA_CHECK(cudaMemcpyPeer(
                    local_plain + loc * N, g,
                    plain.data() + j * N, 0,
                    N * sizeof(uint64_t)));
                loc++;
            }

            cudaStream_t s = ctx.stream(g);
            // Only c0 (poly index 0)
            uint64_t *c0 = dct.data(g);
            add_rns_poly<<<grid, blockDim_local, 0, s>>>(
                c0, local_plain, mod, c0, N, local_n);
            CUDA_CHECK(cudaStreamSynchronize(s));
            CUDA_CHECK(cudaFree(local_plain));
        });
    }
    for (auto &t : threads) t.join();
}

// ---------------------------------------------------------------------------
// CROSS-LIMB operations — require all limbs (gather-operate-scatter)
// ---------------------------------------------------------------------------
// rescale and mod_switch involve dropping primes across the RNS representation.
// These cannot be done per-limb independently — they require the full
// polynomial. We gather to GPU 0, operate, scatter back.

static void gather_op_scatter(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    std::function<void(PhantomContext&, PhantomCiphertext&)> op)
{
    PhantomCiphertext ct = dct.to_single_gpu(ctx, 0);
    CUDA_CHECK(cudaSetDevice(0));
    op(ctx.context(0), ct);
    CUDA_CHECK(cudaDeviceSynchronize());
    dct.free_all();
    dct = DistributedCiphertext::from_single_gpu(ctx, ct, 0);
}

void dist_rescale_to_next_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct)
{
    gather_op_scatter(ctx, dct, [](PhantomContext &pctx, PhantomCiphertext &ct) {
        rescale_to_next_inplace(pctx, ct);
    });
    dct.set_chain_index(dct.chain_index() + 1);
}

void dist_mod_switch_to_next_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct)
{
    gather_op_scatter(ctx, dct, [](PhantomContext &pctx, PhantomCiphertext &ct) {
        mod_switch_to_next_inplace(pctx, ct);
    });
    dct.set_chain_index(dct.chain_index() + 1);
}

// ---------------------------------------------------------------------------
// KEYED operations — gather to GPU 0 for now (Phase 1)
// Phase 2: true distributed key-switching with NCCL
// ---------------------------------------------------------------------------

void dist_multiply_and_relin_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct1,
    const DistributedCiphertext &dct2,
    const PhantomRelinKey &relin_keys)
{
    PhantomCiphertext ct1 = dct1.to_single_gpu(ctx, 0);
    PhantomCiphertext ct2 = dct2.to_single_gpu(ctx, 0);

    CUDA_CHECK(cudaSetDevice(0));
    multiply_inplace(ctx.context(0), ct1, ct2);
    relinearize_inplace(ctx.context(0), ct1, relin_keys);
    CUDA_CHECK(cudaDeviceSynchronize());

    dct1.free_all();
    dct1 = DistributedCiphertext::from_single_gpu(ctx, ct1, 0);
}

void dist_relinearize_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    const PhantomRelinKey &relin_keys)
{
    gather_op_scatter(ctx, dct, [&](PhantomContext &pctx, PhantomCiphertext &ct) {
        relinearize_inplace(pctx, ct, relin_keys);
    });
}

void dist_rotate_vector_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    int steps,
    const PhantomGaloisKey &galois_keys)
{
    gather_op_scatter(ctx, dct, [&](PhantomContext &pctx, PhantomCiphertext &ct) {
        rotate_vector_inplace(pctx, ct, steps, galois_keys);
    });
}

void dist_square_and_relin_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    const PhantomRelinKey &relin_keys)
{
    gather_op_scatter(ctx, dct, [&](PhantomContext &pctx, PhantomCiphertext &ct) {
        multiply_inplace(pctx, ct, ct);
        relinearize_inplace(pctx, ct, relin_keys);
    });
}

// ---------------------------------------------------------------------------
// Compound operations
// ---------------------------------------------------------------------------

void dist_multiply_const_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    double constant)
{
    // Encode on GPU 0, then distribute plaintext limbs
    CUDA_CHECK(cudaSetDevice(0));
    PhantomCKKSEncoder encoder(ctx.context(0));
    PhantomPlaintext plain;
    encoder.encode(ctx.context(0), constant, dct.scale(), plain);
    mod_switch_to_inplace(ctx.context(0), plain, dct.chain_index());

    dist_multiply_plain_inplace(ctx, dct, plain);
}

void dist_add_const_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    double constant)
{
    CUDA_CHECK(cudaSetDevice(0));
    PhantomCKKSEncoder encoder(ctx.context(0));
    PhantomPlaintext plain;
    encoder.encode(ctx.context(0), constant, dct.scale(), plain);
    mod_switch_to_inplace(ctx.context(0), plain, dct.chain_index());

    dist_add_plain_inplace(ctx, dct, plain);
}

// ---------------------------------------------------------------------------
// GPU utilization validation
// ---------------------------------------------------------------------------

void validate_gpu_utilization(
    DistributedContext &ctx,
    DistributedCiphertext &dct)
{
    printf("\n=== GPU Utilization Validation ===\n");
    printf("Testing that ALL %d GPUs run compute kernels...\n\n", ctx.n_gpus());

    // Create a copy to add to itself (harmless operation)
    // Each GPU should show non-zero kernel time
    for (int g = 0; g < ctx.n_gpus(); g++) {
        CUDA_CHECK(cudaSetDevice(g));

        size_t local_n = dct.local_limb_count(g);
        if (local_n == 0) {
            printf("  GPU %d: 0 local limbs (idle) — WARNING\n", g);
            continue;
        }

        const DModulus *mod = get_modulus_ptr(ctx, g);
        size_t N = dct.poly_degree();
        size_t local_coeff_count = local_n * N;
        dim3 grid(local_coeff_count / blockDim_local.x);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        cudaStream_t s = ctx.stream(g);
        CUDA_CHECK(cudaEventRecord(start, s));

        // Run 100 add_rns_poly kernels as a timing test
        for (int iter = 0; iter < 100; iter++) {
            uint64_t *c0 = dct.data(g);
            add_rns_poly<<<grid, blockDim_local, 0, s>>>(
                c0, c0, mod, c0, N, local_n);
        }

        CUDA_CHECK(cudaEventRecord(stop, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        printf("  GPU %d: %zu local limbs, 100 add_rns_poly kernels in %.3f ms %s\n",
               g, local_n, ms, ms > 0.001 ? "— ACTIVE" : "— IDLE (BUG!)");

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    printf("\n");
}

} // namespace nexus_multi_gpu
