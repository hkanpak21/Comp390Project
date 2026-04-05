/**
 * dist_bert_layer_bench.cu
 *
 * Benchmark a single BERT transformer layer using distributed FHE.
 *
 * BERT-base layer structure (from NEXUS paper Table IV):
 *   1. MatMul: Q,K,V projection    (R^{128x768} x R^{768x768})  x3
 *   2. MatMul: Q*K^T attention      (R^{128x64} x R^{64x128})   x12 heads
 *   3. Softmax                       (R^{128x128})               x12
 *   4. MatMul: attn*V               (R^{128x128} x R^{128x64})  x12
 *   5. MatMul: output projection    (R^{128x768} x R^{768x768})
 *   6. LayerNorm
 *   7. MatMul: FFN1                  (R^{128x768} x R^{768x3072})
 *   8. GELU
 *   9. MatMul: FFN2                  (R^{128x3072} x R^{3072x768})
 *  10. LayerNorm
 *  + Bootstrapping at strategic points (5 per layer)
 *
 * This benchmark measures per-operation timing to identify bottlenecks
 * for multi-GPU optimization.
 *
 * Usage:
 *   ./dist_bert_layer_bench --n-gpus 2 [--verbose]
 */

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>

// Phantom FHE
#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

// Multi-GPU
#include "../multi_gpu/distributed_context.cuh"
#include "../multi_gpu/distributed_eval.cuh"
#include "../multi_gpu/nvtx_ranges.cuh"

using namespace std;
using namespace phantom;
using namespace nexus_multi_gpu;

struct Timer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        return chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
    }
};

struct Config {
    int n_gpus = 1;
    bool verbose = false;
    size_t poly_degree = 8192;   // Start small; 65536 for NEXUS
    size_t n_moduli = 10;        // Start small; 35 for NEXUS
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc) cfg.n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose")) cfg.verbose = true;
        else if (!strcmp(argv[i], "--N") && i+1 < argc) cfg.poly_degree = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--L") && i+1 < argc) cfg.n_moduli = atoi(argv[++i]);
    }
    return cfg;
}

// Simulates a sequence of BERT operations on distributed ciphertexts
// and measures per-operation timing.
struct BertLayerTiming {
    double matmul_ms = 0;       // all MatMul operations
    double softmax_ms = 0;      // softmax (rotations + exp)
    double gelu_ms = 0;         // GELU (sgn_eval + polynomial)
    double layernorm_ms = 0;    // LayerNorm (rotations + sqrt)
    double bootstrap_ms = 0;    // bootstrapping
    double local_ops_ms = 0;    // add, mul_plain, rescale
    double comm_ms = 0;         // gather/scatter + NCCL
    double total_ms = 0;
};

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count < cfg.n_gpus) {
        fprintf(stderr, "ERROR: need %d GPUs, have %d\n", cfg.n_gpus, device_count);
        return 1;
    }

    printf("=== Distributed BERT Layer Benchmark ===\n");
    printf("GPUs: %d, N=%zu, L=%zu\n\n", cfg.n_gpus, cfg.poly_degree, cfg.n_moduli);

    // ---- Setup ----
    Timer timer;

    printf("[1] Creating distributed context...\n");
    timer.start();
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(cfg.poly_degree);
    vector<int> bit_sizes = {60};
    for (size_t i = 0; i < cfg.n_moduli - 2; i++) bit_sizes.push_back(40);
    bit_sizes.push_back(60);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(cfg.poly_degree, bit_sizes));

    DistributedContext dctx = DistributedContext::create(parms, cfg.n_gpus);
    printf("   Created in %.1f ms\n", timer.elapsed_ms());

    printf("[2] Generating keys on GPU 0...\n");
    timer.start();
    cudaSetDevice(0);
    PhantomSecretKey sk(dctx.context(0));
    PhantomRelinKey rk = sk.gen_relinkey(dctx.context(0));
    printf("   Keys in %.1f ms\n", timer.elapsed_ms());

    printf("[3] Creating test ciphertext...\n");
    timer.start();
    PhantomCKKSEncoder encoder(dctx.context(0));
    size_t slots = cfg.poly_degree / 2;
    vector<double> data(slots);
    for (size_t i = 0; i < slots; i++) data[i] = (rand() % 1000) / 100.0 - 5.0;

    PhantomPlaintext plain;
    const double SCALE = (double)(1ULL << 40);
    encoder.encode(dctx.context(0), data, SCALE, plain);

    PhantomCiphertext ct;
    sk.encrypt_symmetric(dctx.context(0), plain, ct);
    printf("   Encrypted in %.1f ms\n", timer.elapsed_ms());

    // ---- Distribute ciphertext ----
    printf("[4] Distributing ciphertext across %d GPUs...\n", cfg.n_gpus);
    timer.start();
    DistributedCiphertext dct = DistributedCiphertext::from_single_gpu(dctx, ct, 0);
    printf("   Distributed in %.1f ms\n", timer.elapsed_ms());

    // ---- Validate GPU utilization ----
    printf("[5] Validating GPU utilization...\n");
    validate_gpu_utilization(dctx, dct);

    // ---- Benchmark individual operations ----
    printf("[6] Benchmarking operations (10 iterations each)...\n\n");
    constexpr int ITERS = 10;
    BertLayerTiming timing;

    // --- LOCAL: add_plain ---
    {
        timer.start();
        for (int i = 0; i < ITERS; i++) {
            dist_add_plain_inplace(dctx, dct, plain);
        }
        double t = timer.elapsed_ms() / ITERS;
        timing.local_ops_ms += t;
        printf("   add_plain_inplace:      %.3f ms avg (LOCAL — all GPUs parallel)\n", t);
    }

    // --- LOCAL: multiply_plain ---
    {
        timer.start();
        for (int i = 0; i < ITERS; i++) {
            dist_multiply_plain_inplace(dctx, dct, plain);
        }
        double t = timer.elapsed_ms() / ITERS;
        timing.local_ops_ms += t;
        printf("   multiply_plain_inplace: %.3f ms avg (LOCAL — all GPUs parallel)\n", t);
    }

    // --- LOCAL: add (ct + ct) ---
    {
        // Create a second distributed ciphertext
        PhantomCiphertext ct2;
        sk.encrypt_symmetric(dctx.context(0), plain, ct2);
        DistributedCiphertext dct2 = DistributedCiphertext::from_single_gpu(dctx, ct2, 0);

        timer.start();
        for (int i = 0; i < ITERS; i++) {
            dist_add_inplace(dctx, dct, dct2);
        }
        double t = timer.elapsed_ms() / ITERS;
        timing.local_ops_ms += t;
        printf("   add_inplace (ct+ct):    %.3f ms avg (LOCAL — all GPUs parallel)\n", t);
    }

    // --- LOCAL: negate ---
    {
        timer.start();
        for (int i = 0; i < ITERS; i++) {
            dist_negate_inplace(dctx, dct);
        }
        double t = timer.elapsed_ms() / ITERS;
        timing.local_ops_ms += t;
        printf("   negate_inplace:         %.3f ms avg (LOCAL — all GPUs parallel)\n", t);
    }

    // --- CROSS-LIMB: rescale ---
    {
        // Need to re-create ciphertext at fresh level for rescale
        PhantomCiphertext ct_fresh;
        sk.encrypt_symmetric(dctx.context(0), plain, ct_fresh);

        // multiply_plain to create a rescalable ciphertext
        multiply_plain_inplace(dctx.context(0), ct_fresh, plain);

        DistributedCiphertext dct_fresh = DistributedCiphertext::from_single_gpu(dctx, ct_fresh, 0);

        timer.start();
        dist_rescale_to_next_inplace(dctx, dct_fresh);
        double t = timer.elapsed_ms();
        timing.local_ops_ms += t;
        printf("   rescale_to_next:        %.3f ms (CROSS-LIMB — gather/scatter)\n", t);
    }

    // --- KEYED: relinearize ---
    {
        // Create a size-3 ciphertext via multiplication
        PhantomCiphertext ct_a, ct_b;
        sk.encrypt_symmetric(dctx.context(0), plain, ct_a);
        sk.encrypt_symmetric(dctx.context(0), plain, ct_b);
        multiply_inplace(dctx.context(0), ct_a, ct_b);

        DistributedCiphertext dct_relin = DistributedCiphertext::from_single_gpu(dctx, ct_a, 0);

        timer.start();
        dist_relinearize_inplace(dctx, dct_relin, rk);
        double t = timer.elapsed_ms();
        timing.matmul_ms += t;
        printf("   relinearize_inplace:    %.3f ms (KEYED — communication)\n", t);
    }

    // ---- Summary ----
    printf("\n=== Operation Classification Summary ===\n");
    printf("LOCAL ops (true multi-GPU parallel):  %.3f ms\n", timing.local_ops_ms);
    printf("KEYED ops (communication required):   %.3f ms\n", timing.matmul_ms);
    printf("\nLOCAL ops are the target for multi-GPU speedup.\n");
    printf("KEYED ops require distributed key-switching (Phase 2).\n");

    // ---- Cleanup ----
    dctx.destroy();
    printf("\nDone.\n");
    return 0;
}
