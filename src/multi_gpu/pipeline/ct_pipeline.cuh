#pragma once
/**
 * ct_pipeline.cuh
 *
 * Ciphertext-level pipeline parallelism for multi-GPU FHE.
 *
 * Key insight: In a BERT layer, operations like MatMul produce many
 * independent ciphertexts. Instead of distributing ONE key-switch across
 * GPUs (limited to 1.3x by Amdahl's Law), we distribute DIFFERENT
 * ciphertexts to different GPUs. Each GPU does full single-GPU operations
 * on its batch — embarrassingly parallel, zero communication.
 *
 * Usage:
 *   CtPipeline pipe = CtPipeline::create(parms, n_gpus, secret_key);
 *
 *   // Scatter 64 ciphertexts across 4 GPUs (16 each)
 *   pipe.scatter(ciphertexts);
 *
 *   // Each GPU independently: multiply_plain + relinearize + rescale
 *   pipe.execute([](int gpu, PhantomContext &ctx, PhantomRelinKey &rk,
 *                   PhantomCKKSEncoder &enc, vector<PhantomCiphertext> &cts) {
 *       for (auto &ct : cts) {
 *           multiply_plain_inplace(ctx, ct, plain);
 *           relinearize_inplace(ctx, ct, rk);
 *           rescale_to_next_inplace(ctx, ct);
 *       }
 *   });
 *
 *   // Gather results back to GPU 0
 *   auto results = pipe.gather();
 */

#include <cuda_runtime.h>
#include <vector>
#include <functional>
#include <thread>
#include <memory>
#include <sstream>

// Phantom headers — order matters
#include "ciphertext.h"
#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"

namespace nexus_multi_gpu {

class CtPipeline {
public:
    // Create pipeline with per-GPU PhantomContexts and shared keys.
    // secret_key is used to derive relin/galois keys on each GPU.
    static CtPipeline create(
        const phantom::EncryptionParameters &parms,
        int n_gpus,
        PhantomSecretKey &secret_key);

    // Scatter ciphertexts round-robin across GPUs.
    // ct[i] → GPU (i % n_gpus). Uses cudaMemcpyPeer.
    void scatter(const std::vector<PhantomCiphertext> &cts);

    // Execute a function on each GPU's local ciphertext batch.
    // The function receives: gpu_id, context, relin_key, encoder, local_cts.
    // All GPUs run in parallel via std::thread.
    using GpuFn = std::function<void(
        int gpu_id,
        PhantomContext &ctx,
        PhantomRelinKey &rk,
        PhantomCKKSEncoder &enc,
        std::vector<PhantomCiphertext> &local_cts)>;

    void execute(GpuFn fn);

    // Gather results back to GPU 0 in original order.
    std::vector<PhantomCiphertext> gather();

    int n_gpus() const { return n_gpus_; }
    size_t total_cts() const { return total_cts_; }

    void destroy();

private:
    int n_gpus_ = 0;
    size_t total_cts_ = 0;
    phantom::EncryptionParameters parms_;

    // Per-GPU state
    struct GpuState {
        PhantomContext *ctx = nullptr;
        PhantomSecretKey *sk = nullptr;
        PhantomRelinKey *rk = nullptr;
        PhantomCKKSEncoder *enc = nullptr;
        std::vector<PhantomCiphertext> local_cts;
        std::vector<size_t> original_indices;  // for gather reordering
    };
    std::vector<GpuState> states_;
};

} // namespace nexus_multi_gpu
