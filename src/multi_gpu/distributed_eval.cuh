#pragma once
/**
 * distributed_eval.cuh
 *
 * Multi-GPU wrappers for all FHE operations used by NEXUS.
 *
 * Operation classification:
 *   LOCAL  — per-limb, zero communication: add, sub, multiply_plain, rescale,
 *            mod_switch, NTT, negate
 *   KEYED  — requires key-switching = communication: relinearize, rotate_vector,
 *            apply_galois, multiply (ct*ct then relin)
 *   BOOTSTRAP — multiple key-switches internally
 *
 * For LOCAL ops: each GPU calls Phantom on its local PhantomCiphertext
 *   independently. No synchronization needed.
 *
 * For KEYED ops: we use Input Broadcast (AllGather c2 → local keyswitch)
 *   or Output Aggregation (partial keyswitch → AllReduce).
 *
 * For BOOTSTRAP: each GPU runs bootstrapping on its local ciphertext.
 *   Bootstrapping internally uses rotations (KEYED), so communication
 *   happens at each rotation within the bootstrap.
 *
 * Usage pattern:
 *   DistributedContext dctx = DistributedContext::create(parms, n_gpus);
 *   // ... generate keys on GPU 0, distribute ...
 *   DistributedCiphertext dct = DistributedCiphertext::from_single_gpu(dctx, ct);
 *
 *   // All ops below run on all GPUs in parallel:
 *   dist_add_plain_inplace(dctx, dct, plain);
 *   dist_multiply_plain_inplace(dctx, dct, plain);
 *   dist_rescale_to_next_inplace(dctx, dct);
 *   dist_relinearize_inplace(dctx, dct, relin_keys);  // ← communication here
 *   dist_rotate_vector_inplace(dctx, dct, steps, galois_keys);  // ← communication
 */

#include "distributed_context.cuh"
#include "evaluate.cuh"
#include "ckks.h"

#include <thread>
#include <functional>

namespace nexus_multi_gpu {

// ---------------------------------------------------------------------------
// Parallel execution helper
// ---------------------------------------------------------------------------
// Launches a function on each GPU in a separate thread, waits for all.

inline void parallel_for_gpus(DistributedContext &ctx,
                              std::function<void(int gpu_id)> fn) {
    if (ctx.n_gpus() == 1) {
        cudaSetDevice(0);
        fn(0);
        return;
    }
    std::vector<std::thread> threads;
    for (int g = 0; g < ctx.n_gpus(); g++) {
        threads.emplace_back([&ctx, &fn, g]() {
            cudaSetDevice(g);
            fn(g);
        });
    }
    for (auto &t : threads) t.join();
}

// ---------------------------------------------------------------------------
// LOCAL operations (zero communication)
// ---------------------------------------------------------------------------
// Each GPU operates on its local PhantomCiphertext independently.
// We need a local PhantomCiphertext per GPU — these are reconstructed from
// the DistributedCiphertext's raw limb data on each GPU.

// Helper: create a local PhantomCiphertext view from distributed data on one GPU.
// This wraps the local limb buffer as a PhantomCiphertext that Phantom can operate on.
// NOTE: This is a shallow view — the data pointer is NOT owned by the PhantomCiphertext.
PhantomCiphertext make_local_ct(
    DistributedContext &ctx, int gpu_id,
    DistributedCiphertext &dct);

// Add plaintext in-place (LOCAL: per-limb, no communication)
void dist_add_plain_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    const PhantomPlaintext &plain);

// Multiply by plaintext in-place (LOCAL)
void dist_multiply_plain_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    const PhantomPlaintext &plain);

// Rescale to next level (LOCAL: drops one limb from each GPU)
void dist_rescale_to_next_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct);

// Add two distributed ciphertexts (LOCAL)
void dist_add_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct1,
    const DistributedCiphertext &dct2);

// Subtract (LOCAL)
void dist_sub_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct1,
    const DistributedCiphertext &dct2);

// Negate (LOCAL)
void dist_negate_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct);

// Modulus switch (LOCAL)
void dist_mod_switch_to_next_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct);

// ---------------------------------------------------------------------------
// KEYED operations (require communication)
// ---------------------------------------------------------------------------

// Ciphertext * ciphertext multiplication + relinearization
// Communication: key-switching for relinearization
void dist_multiply_and_relin_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct1,
    const DistributedCiphertext &dct2,
    const PhantomRelinKey &relin_keys);

// Relinearize only (for when multiply was done separately)
// Communication: AllGather or AllReduce depending on algorithm
void dist_relinearize_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    const PhantomRelinKey &relin_keys);

// Rotate vector by `steps` positions
// Communication: key-switching with galois key
void dist_rotate_vector_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    int steps,
    const PhantomGaloisKey &galois_keys);

// Square + relinearize (common pattern in polynomial evaluation)
void dist_square_and_relin_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    const PhantomRelinKey &relin_keys);

// ---------------------------------------------------------------------------
// Compound operations (used by NEXUS neural network layers)
// ---------------------------------------------------------------------------

// Multiply by constant scalar (LOCAL: encode + multiply_plain)
void dist_multiply_const_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    double constant);

// Add constant (LOCAL: encode + add_plain)
void dist_add_const_inplace(
    DistributedContext &ctx,
    DistributedCiphertext &dct,
    double constant);

} // namespace nexus_multi_gpu
