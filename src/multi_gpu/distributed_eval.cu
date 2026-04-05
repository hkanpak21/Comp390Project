/**
 * distributed_eval.cu
 *
 * Multi-GPU wrappers for FHE operations.
 *
 * Architecture:
 *   LOCAL ops: gather limbs → local PhantomCiphertext → Phantom op → scatter back
 *   KEYED ops: gather to single GPU → Phantom keyswitch → scatter results
 *
 * For Phase 1, we use a "gather-operate-scatter" pattern:
 *   1. Gather distributed limbs to GPU 0 as a full PhantomCiphertext
 *   2. Run the Phantom operation on GPU 0
 *   3. Scatter results back to distributed limbs
 *
 * This is NOT the final architecture (no actual parallelism for compute),
 * but it validates correctness and lets us profile where communication
 * vs computation time goes. Phase 2 will implement true per-GPU local ops.
 *
 * Why this approach first:
 *   - Phantom's evaluate functions expect PhantomCiphertext objects with
 *     specific metadata (chain_index, coeff_mod_size, etc.)
 *   - Creating "partial" PhantomCiphertexts with only local limbs requires
 *     modifying Phantom internals (which we don't want to do yet)
 *   - gather-operate-scatter lets us validate the full BERT pipeline
 *     and identify bottlenecks with Nsight before optimizing
 */

#include "distributed_eval.cuh"
#include "evaluate.cuh"
#include "ckks.h"

#include <cstdio>

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

// ---------------------------------------------------------------------------
// Helper: gather → operate → scatter pattern
// ---------------------------------------------------------------------------
// Gathers distributed ciphertext to GPU 0, runs a Phantom operation,
// scatters the result back. Used for all operations in Phase 1.

static void gather_op_scatter(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    std::function<void(PhantomContext&, PhantomCiphertext&)> op)
{
    // 1. Gather to GPU 0
    PhantomCiphertext ct = dct.to_single_gpu(ctx, 0);

    // 2. Operate on GPU 0
    CUDA_CHECK(cudaSetDevice(0));
    op(ctx.context(0), ct);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. Free old distributed data, scatter new result
    dct.free_all();
    dct = DistributedCiphertext::from_single_gpu(ctx, ct, 0);
}

// Same but for binary operations (two input ciphertexts)
static void gather_binary_op_scatter(
    DistributedContext &ctx,
    DistributedCiphertext &dct1,
    const DistributedCiphertext &dct2,
    std::function<void(PhantomContext&, PhantomCiphertext&, const PhantomCiphertext&)> op)
{
    PhantomCiphertext ct1 = dct1.to_single_gpu(ctx, 0);
    PhantomCiphertext ct2 = dct2.to_single_gpu(ctx, 0);

    CUDA_CHECK(cudaSetDevice(0));
    op(ctx.context(0), ct1, ct2);
    CUDA_CHECK(cudaDeviceSynchronize());

    dct1.free_all();
    dct1 = DistributedCiphertext::from_single_gpu(ctx, ct1, 0);
}

// ---------------------------------------------------------------------------
// LOCAL operations
// ---------------------------------------------------------------------------

void dist_add_plain_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    const PhantomPlaintext &plain)
{
    gather_op_scatter(ctx, dct, [&](PhantomContext &pctx, PhantomCiphertext &ct) {
        add_plain_inplace(pctx, ct, plain);
    });
}

void dist_multiply_plain_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    const PhantomPlaintext &plain)
{
    gather_op_scatter(ctx, dct, [&](PhantomContext &pctx, PhantomCiphertext &ct) {
        multiply_plain_inplace(pctx, ct, plain);
    });
}

void dist_rescale_to_next_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct)
{
    gather_op_scatter(ctx, dct, [&](PhantomContext &pctx, PhantomCiphertext &ct) {
        rescale_to_next_inplace(pctx, ct);
    });
    // Update chain index after rescale
    dct.set_chain_index(dct.chain_index() + 1);
}

void dist_add_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct1,
    const DistributedCiphertext &dct2)
{
    gather_binary_op_scatter(ctx, dct1, dct2,
        [](PhantomContext &pctx, PhantomCiphertext &ct1, const PhantomCiphertext &ct2) {
            add_inplace(pctx, ct1, ct2);
        });
}

void dist_sub_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct1,
    const DistributedCiphertext &dct2)
{
    gather_binary_op_scatter(ctx, dct1, dct2,
        [](PhantomContext &pctx, PhantomCiphertext &ct1, const PhantomCiphertext &ct2) {
            sub_inplace(pctx, ct1, ct2);
        });
}

void dist_negate_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct)
{
    gather_op_scatter(ctx, dct, [](PhantomContext &pctx, PhantomCiphertext &ct) {
        negate_inplace(pctx, ct);
    });
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
// KEYED operations (require communication)
// ---------------------------------------------------------------------------

void dist_multiply_and_relin_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct1,
    const DistributedCiphertext &dct2,
    const PhantomRelinKey &relin_keys)
{
    // Phase 1: gather both, multiply+relin on GPU 0, scatter
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
        // square = multiply with self
        PhantomCiphertext ct_copy;
        ct_copy.resize(pctx, ct.chain_index(), ct.size(), cudaStreamPerThread);
        cudaMemcpyAsync(ct_copy.data(), ct.data(),
                        ct.size() * ct.coeff_modulus_size() * ct.poly_modulus_degree() * sizeof(uint64_t),
                        cudaMemcpyDeviceToDevice, cudaStreamPerThread);
        ct_copy.set_scale(ct.scale());
        ct_copy.set_ntt_form(ct.is_ntt_form());
        ct_copy.set_chain_index(ct.chain_index());
        ct_copy.set_coeff_modulus_size(ct.coeff_modulus_size());
        ct_copy.set_poly_modulus_degree(ct.poly_modulus_degree());

        multiply_inplace(pctx, ct, ct_copy);
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
    gather_op_scatter(ctx, dct, [&](PhantomContext &pctx, PhantomCiphertext &ct) {
        PhantomCKKSEncoder encoder(pctx);
        PhantomPlaintext plain;
        encoder.encode(pctx, constant, ct.scale(), plain);
        mod_switch_to_inplace(pctx, plain, ct.chain_index());
        multiply_plain_inplace(pctx, ct, plain);
    });
}

void dist_add_const_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    double constant)
{
    gather_op_scatter(ctx, dct, [&](PhantomContext &pctx, PhantomCiphertext &ct) {
        PhantomCKKSEncoder encoder(pctx);
        PhantomPlaintext plain;
        encoder.encode(pctx, constant, ct.scale(), plain);
        mod_switch_to_inplace(pctx, plain, ct.chain_index());
        add_plain_inplace(pctx, ct, plain);
    });
}

} // namespace nexus_multi_gpu
