/**
 * multi_gpu_keyswitch_test.cu
 *
 * End-to-end validation of multi-GPU key-switching algorithms.
 *
 * Test plan:
 *   1. Initialize Phantom CKKS context (same params as NEXUS: N=65536, L=20)
 *   2. Encrypt a random plaintext, multiply to get a 3-element ciphertext (c0, c1, c2)
 *   3. Run single-GPU relinearize_inplace as ground truth
 *   4. Run Input Broadcast key-switching on n_gpus
 *   5. Run Output Aggregation key-switching on n_gpus
 *   6. Compare results against ground truth (MAE < epsilon)
 *
 * Usage:
 *   ./multi_gpu_keyswitch_test --n-gpus 2 [--verbose]
 *
 * Requires: 2+ NVIDIA GPUs with peer access (NVLink or PCIe P2P).
 */

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <chrono>

// Phantom FHE
#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

// Our multi-GPU code
#include "../multi_gpu/comm/nccl_comm.cuh"
#include "../multi_gpu/partition/rns_partition.cuh"
#include "../multi_gpu/keyswitching/input_broadcast.cuh"
#include "../multi_gpu/keyswitching/output_aggregation.cuh"

using namespace std;
using namespace phantom;
using namespace nexus_multi_gpu;

// ---- Timing utility ----
struct Timer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto t1 = chrono::high_resolution_clock::now();
        return chrono::duration<double, milli>(t1 - t0).count();
    }
};

// ---- Config ----
struct Config {
    int n_gpus = 2;
    bool verbose = false;
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i + 1 < argc)
            cfg.n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose"))
            cfg.verbose = true;
    }
    return cfg;
}

// ---- Main ----
int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    // Check GPU count
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count < cfg.n_gpus) {
        fprintf(stderr, "ERROR: requested %d GPUs but only %d available\n",
                cfg.n_gpus, device_count);
        return 1;
    }

    printf("=== Multi-GPU Key-Switching Validation Test ===\n");
    printf("GPUs: %d\n\n", cfg.n_gpus);

    // ---- Step 1: Initialize Phantom CKKS context ----
    printf("[1/6] Initializing Phantom CKKS context...\n");
    Timer timer;
    timer.start();

    // NEXUS-compatible parameters: N=65536, L=20, scale=2^40
    constexpr size_t POLY_DEGREE = 65536;
    const double SCALE = (double)(1ULL << 40);
    constexpr size_t N_MODULI = 20;  // Use L=20 for a manageable test

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(POLY_DEGREE);

    // Build modulus chain: 60-bit anchor + (N_MODULI-2) x 40-bit + 60-bit tail
    vector<int> bit_sizes;
    bit_sizes.push_back(60);  // anchor
    for (size_t i = 0; i < N_MODULI - 2; i++)
        bit_sizes.push_back(40);
    bit_sizes.push_back(60);  // tail (special prime for key-switching)
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(POLY_DEGREE, bit_sizes));

    PhantomContext context(parms);

    printf("   Context created in %.1f ms\n", timer.elapsed_ms());

    // ---- Step 2: Generate keys ----
    printf("[2/6] Generating keys...\n");
    timer.start();

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    printf("   Keys generated in %.1f ms\n", timer.elapsed_ms());

    // ---- Step 3: Encrypt, multiply, get c2 for key-switching ----
    printf("[3/6] Encrypting and multiplying...\n");
    timer.start();

    PhantomCKKSEncoder encoder(context);

    // Create two random plaintexts
    size_t slot_count = POLY_DEGREE / 2;
    vector<double> input_a(slot_count), input_b(slot_count);
    srand(42);
    for (size_t i = 0; i < slot_count; i++) {
        input_a[i] = (rand() % 1000) / 100.0 - 5.0;
        input_b[i] = (rand() % 1000) / 100.0 - 5.0;
    }

    PhantomPlaintext plain_a, plain_b;
    encoder.encode(context, input_a, SCALE, plain_a);
    encoder.encode(context, input_b, SCALE, plain_b);

    PhantomCiphertext ct_a, ct_b;
    secret_key.encrypt_symmetric(context, plain_a, ct_a);
    secret_key.encrypt_symmetric(context, plain_b, ct_b);

    // Multiply in-place (creates a 3-element ciphertext with c2)
    multiply_inplace(context, ct_a, ct_b);
    // ct_a now has size=3 (c0, c1, c2)

    printf("   Encrypt + multiply in %.1f ms\n", timer.elapsed_ms());

    // Initialize NCCL early
    vector<int> dev_ids(cfg.n_gpus);
    for (int i = 0; i < cfg.n_gpus; i++) dev_ids[i] = i;
    MultiGpuContext mgpu_ctx = MultiGpuContext::create(dev_ids);

    // ---- Step 4: Ground truth — single-GPU relinearize ----
    printf("[4/6] Computing ground truth (single-GPU relinearize)...\n");
    timer.start();

    // We need separate ciphertexts for each test. Re-encrypt and multiply.
    PhantomCiphertext ct_gt;
    {
        PhantomCiphertext tmp_a, tmp_b;
        secret_key.encrypt_symmetric(context, plain_a, tmp_a);
        secret_key.encrypt_symmetric(context, plain_b, tmp_b);
        multiply_inplace(context, tmp_a, tmp_b);
        ct_gt = std::move(tmp_a);
    }
    relinearize_inplace(context, ct_gt, relin_keys);

    printf("   Single-GPU relinearize in %.1f ms\n", timer.elapsed_ms());

    // Decrypt ground truth for comparison
    PhantomPlaintext plain_gt;
    secret_key.decrypt(context, ct_gt, plain_gt);
    vector<double> result_gt;
    encoder.decode(context, plain_gt, result_gt);

    // ---- Step 5: Input Broadcast key-switching ----
    printf("[5/6] Testing Input Broadcast key-switching (%d GPUs)...\n", cfg.n_gpus);
    timer.start();

    // Fresh multiply for Input Broadcast test
    PhantomCiphertext ct_ib;
    {
        PhantomCiphertext tmp_a, tmp_b;
        secret_key.encrypt_symmetric(context, plain_a, tmp_a);
        secret_key.encrypt_symmetric(context, plain_b, tmp_b);
        multiply_inplace(context, tmp_a, tmp_b);
        ct_ib = std::move(tmp_a);
    }

    // Extract c2 pointer (3rd polynomial of the ciphertext)
    auto chain_idx = ct_ib.chain_index();
    auto &ctx_data = context.get_context_data(chain_idx);
    size_t size_Ql = ctx_data.gpu_rns_tool().base_Ql().size();
    uint64_t *c2_ptr = ct_ib.data() + 2 * size_Ql * POLY_DEGREE;

    // Call Input Broadcast (on GPU 0 as the "local" GPU)
    keyswitching_input_broadcast(mgpu_ctx, context, /*gpu_id=*/0,
                                 ct_ib, c2_ptr, relin_keys,
                                 size_Ql, POLY_DEGREE, cfg.n_gpus);

    // Resize ciphertext back to 2 elements (c2 has been consumed)
    ct_ib.resize(context, chain_idx, 2, cudaStreamPerThread);

    double ib_time = timer.elapsed_ms();

    // Decrypt and compare
    PhantomPlaintext plain_ib;
    secret_key.decrypt(context, ct_ib, plain_ib);
    vector<double> result_ib;
    encoder.decode(context, plain_ib, result_ib);

    double ib_mae = 0.0;
    for (size_t i = 0; i < slot_count; i++)
        ib_mae += fabs(result_ib[i] - result_gt[i]);
    ib_mae /= slot_count;

    printf("   Input Broadcast: %.1f ms, MAE vs ground truth: %.2e\n",
           ib_time, ib_mae);
    printf("   %s\n", ib_mae < 1e-3 ? "PASS" : "FAIL");

    // ---- Step 6: Output Aggregation key-switching ----
    printf("[6/6] Testing Output Aggregation key-switching (%d GPUs)...\n", cfg.n_gpus);
    timer.start();

    // Fresh multiply for Output Aggregation test
    PhantomCiphertext ct_oa;
    {
        PhantomCiphertext tmp_a, tmp_b;
        secret_key.encrypt_symmetric(context, plain_a, tmp_a);
        secret_key.encrypt_symmetric(context, plain_b, tmp_b);
        multiply_inplace(context, tmp_a, tmp_b);
        ct_oa = std::move(tmp_a);
    }
    uint64_t *c2_ptr_oa = ct_oa.data() + 2 * size_Ql * POLY_DEGREE;

    keyswitching_output_aggregation(mgpu_ctx, context, /*gpu_id=*/0,
                                    ct_oa, c2_ptr_oa, relin_keys,
                                    cfg.n_gpus);

    ct_oa.resize(context, chain_idx, 2, cudaStreamPerThread);

    double oa_time = timer.elapsed_ms();

    PhantomPlaintext plain_oa;
    secret_key.decrypt(context, ct_oa, plain_oa);
    vector<double> result_oa;
    encoder.decode(context, plain_oa, result_oa);

    double oa_mae = 0.0;
    for (size_t i = 0; i < slot_count; i++)
        oa_mae += fabs(result_oa[i] - result_gt[i]);
    oa_mae /= slot_count;

    printf("   Output Aggregation: %.1f ms, MAE vs ground truth: %.2e\n",
           oa_time, oa_mae);
    printf("   %s\n", oa_mae < 1e-3 ? "PASS" : "FAIL");

    // ---- Summary ----
    printf("\n=== Summary ===\n");
    printf("Single-GPU relinearize:     ground truth\n");
    printf("Input Broadcast (%d GPUs):  MAE=%.2e  %s\n",
           cfg.n_gpus, ib_mae, ib_mae < 1e-3 ? "PASS" : "FAIL");
    printf("Output Aggregation (%d GPUs): MAE=%.2e  %s\n",
           cfg.n_gpus, oa_mae, oa_mae < 1e-3 ? "PASS" : "FAIL");

    if (cfg.verbose) {
        printf("\nFirst 5 slots (ground truth vs Input Broadcast vs Output Aggregation):\n");
        for (int i = 0; i < 5; i++)
            printf("  [%d] GT=%.6f  IB=%.6f  OA=%.6f\n",
                   i, result_gt[i], result_ib[i], result_oa[i]);
    }

    // Cleanup
    mgpu_ctx.destroy();

    bool passed = (ib_mae < 1e-3) && (oa_mae < 1e-3);
    printf("\n%s\n", passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return passed ? 0 : 1;
}
