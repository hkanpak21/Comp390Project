/**
 * ct_pipeline_bench.cu
 *
 * Demonstrates REAL multi-GPU speedup via ciphertext-level parallelism.
 *
 * Workload: 64 independent ciphertexts, each processed with:
 *   multiply_plain → relinearize → rescale
 * This simulates BERT MatMul's independent output column operations.
 *
 * Ground truth: 1 GPU processes all 64 ciphertexts serially.
 * Pipeline: N GPUs process 64/N ciphertexts each in parallel.
 * Expected speedup: ~N (embarrassingly parallel, zero communication).
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <sstream>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

#include "../multi_gpu/pipeline/ct_pipeline.cuh"

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
    int n_gpus = 2;
    size_t poly_degree = 8192;
    size_t n_moduli = 5;
    int n_cts = 64;
    int warmup = 2;
    int iters = 5;
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc) cfg.n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i+1 < argc) cfg.poly_degree = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--L") && i+1 < argc) cfg.n_moduli = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-cts") && i+1 < argc) cfg.n_cts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc) cfg.warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) cfg.iters = atoi(argv[++i]);
    }
    return cfg;
}

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < cfg.n_gpus) {
        fprintf(stderr, "Need %d GPUs, have %d\n", cfg.n_gpus, dev_count);
        return 1;
    }

    printf("=== Ciphertext Pipeline Benchmark (REAL SPEEDUP) ===\n");
    printf("GPUs: %d, N=%zu, L=%zu, ciphertexts=%d, warmup=%d, iters=%d\n\n",
           cfg.n_gpus, cfg.poly_degree, cfg.n_moduli, cfg.n_cts, cfg.warmup, cfg.iters);

    // Setup
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(cfg.poly_degree);
    vector<int> bits = {60};
    for (size_t i = 0; i < cfg.n_moduli - 2; i++) bits.push_back(40);
    bits.push_back(60);
    parms.set_coeff_modulus(arith::CoeffModulus::Create(cfg.poly_degree, bits));
    const double SCALE = (double)(1ULL << 40);

    // Create context and keys on GPU 0
    printf("[1] Setup (GPU 0)...\n");
    Timer timer;
    timer.start();

    cudaSetDevice(0);
    PhantomContext ctx0(parms);
    PhantomSecretKey sk(ctx0);
    PhantomRelinKey rk = sk.gen_relinkey(ctx0);
    PhantomCKKSEncoder enc(ctx0);

    // Create plaintext for multiply_plain
    size_t slots = cfg.poly_degree / 2;
    vector<double> plain_data(slots, 0.5);
    PhantomPlaintext plain;
    enc.encode(ctx0, plain_data, SCALE, plain);

    printf("   Setup in %.1f ms\n", timer.elapsed_ms());

    // Create batch of ciphertexts (all on GPU 0 initially)
    printf("[2] Creating %d ciphertexts...\n", cfg.n_cts);
    timer.start();

    vector<double> input(slots);
    srand(42);
    for (size_t i = 0; i < slots; i++) input[i] = (rand() % 1000) / 100.0 - 5.0;

    PhantomPlaintext pt_input;
    enc.encode(ctx0, input, SCALE, pt_input);

    vector<PhantomCiphertext> cts(cfg.n_cts);
    for (int i = 0; i < cfg.n_cts; i++) {
        sk.encrypt_symmetric(ctx0, pt_input, cts[i]);
    }
    printf("   Created %d ciphertexts in %.1f ms\n", cfg.n_cts, timer.elapsed_ms());

    // Workload per ciphertext: multiply_plain + rescale (1 level consumed).
    // Then 20x add_plain to simulate heavy accumulation (no level cost).
    // Total: ~22 kernel launches per ct — enough to saturate H100.
    auto process_ct = [&](PhantomContext &c, PhantomCiphertext &ct,
                          PhantomRelinKey &r, PhantomCKKSEncoder &e) {
        PhantomPlaintext lp;
        e.encode(c, plain_data, ct.scale(), lp);
        multiply_plain_inplace(c, ct, lp);
        rescale_to_next_inplace(c, ct);
        // Re-encode at new scale
        e.encode(c, plain_data, ct.scale(), lp);
        for (int rep = 0; rep < 20; rep++) {
            add_plain_inplace(c, ct, lp);
        }
    };

    // === GROUND TRUTH: single GPU, serial ===
    printf("[3] Ground truth (1 GPU, %d cts serial)...\n", cfg.n_cts);

    // Warmup
    for (int w = 0; w < cfg.warmup; w++) {
        vector<PhantomCiphertext> batch(cfg.n_cts);
        for (int i = 0; i < cfg.n_cts; i++)
            sk.encrypt_symmetric(ctx0, pt_input, batch[i]);
        for (auto &ct : batch) process_ct(ctx0, ct, rk, enc);
        cudaDeviceSynchronize();
    }

    // Timed
    timer.start();
    for (int it = 0; it < cfg.iters; it++) {
        vector<PhantomCiphertext> batch(cfg.n_cts);
        for (int i = 0; i < cfg.n_cts; i++)
            sk.encrypt_symmetric(ctx0, pt_input, batch[i]);
        for (auto &ct : batch) process_ct(ctx0, ct, rk, enc);
        cudaDeviceSynchronize();
    }
    double gt_avg = timer.elapsed_ms() / cfg.iters;
    printf("   Ground truth: %.2f ms per batch\n", gt_avg);

    // === PIPELINE: N GPUs, parallel ===
    printf("[4] Pipeline (%d GPUs, %d cts / %d = %d per GPU)...\n",
           cfg.n_gpus, cfg.n_cts, cfg.n_gpus, cfg.n_cts / cfg.n_gpus);

    timer.start();
    CtPipeline pipe = CtPipeline::create(parms, cfg.n_gpus, sk);
    printf("   Pipeline created in %.1f ms\n", timer.elapsed_ms());

    // Warmup
    for (int w = 0; w < cfg.warmup; w++) {
        vector<PhantomCiphertext> batch(cfg.n_cts);
        cudaSetDevice(0);
        for (int i = 0; i < cfg.n_cts; i++)
            sk.encrypt_symmetric(ctx0, pt_input, batch[i]);

        pipe.scatter(batch);
        pipe.execute([&](int gpu, PhantomContext &c, PhantomRelinKey &r,
                         PhantomCKKSEncoder &e, vector<PhantomCiphertext> &local) {
            PhantomPlaintext lp;
            e.encode(c, plain_data, SCALE, lp);
            for (auto &ct : local) {
                multiply_plain_inplace(c, ct, lp);
                rescale_to_next_inplace(c, ct);
            }
        });
        pipe.gather();
    }

    // Timed
    timer.start();
    for (int it = 0; it < cfg.iters; it++) {
        vector<PhantomCiphertext> batch(cfg.n_cts);
        cudaSetDevice(0);
        for (int i = 0; i < cfg.n_cts; i++)
            sk.encrypt_symmetric(ctx0, pt_input, batch[i]);

        pipe.scatter(batch);
        pipe.execute([&](int gpu, PhantomContext &c, PhantomRelinKey &r,
                         PhantomCKKSEncoder &e, vector<PhantomCiphertext> &local) {
            PhantomPlaintext lp;
            e.encode(c, plain_data, SCALE, lp);
            for (auto &ct : local) {
                multiply_plain_inplace(c, ct, lp);
                rescale_to_next_inplace(c, ct);
            }
        });
        pipe.gather();
    }
    double pipe_avg = timer.elapsed_ms() / cfg.iters;

    printf("   Pipeline: %.2f ms per batch\n", pipe_avg);

    // Validate: decrypt first ciphertext from pipeline vs ground truth
    {
        cudaSetDevice(0);
        // Re-run once for validation
        vector<PhantomCiphertext> gt_batch(1);
        sk.encrypt_symmetric(ctx0, pt_input, gt_batch[0]);
        process_ct(ctx0, gt_batch[0], rk, enc);
        PhantomPlaintext gt_pt;
        sk.decrypt(ctx0, gt_batch[0], gt_pt);
        vector<double> gt_result;
        enc.decode(ctx0, gt_pt, gt_result);

        vector<PhantomCiphertext> pipe_batch(1);
        sk.encrypt_symmetric(ctx0, pt_input, pipe_batch[0]);
        pipe.scatter(pipe_batch);
        pipe.execute([&](int gpu, PhantomContext &c, PhantomRelinKey &r,
                         PhantomCKKSEncoder &e, vector<PhantomCiphertext> &local) {
            PhantomPlaintext lp;
            e.encode(c, plain_data, SCALE, lp);
            for (auto &ct : local)
                { multiply_plain_inplace(c, ct, lp); rescale_to_next_inplace(c, ct); }
        });
        auto pipe_result_cts = pipe.gather();

        PhantomPlaintext pipe_pt;
        sk.decrypt(ctx0, pipe_result_cts[0], pipe_pt);
        vector<double> pipe_result;
        enc.decode(ctx0, pipe_pt, pipe_result);

        double mae = 0;
        for (size_t i = 0; i < slots; i++) mae += fabs(pipe_result[i] - gt_result[i]);
        mae /= slots;
        printf("   MAE: %.2e  %s\n", mae, mae < 1e-3 ? "PASS" : "FAIL");
    }

    // === COMPUTE-ONLY TIMING (pre-scatter, no gather) ===
    printf("\n[5] Compute-only timing (pre-scattered, no gather)...\n");

    // Pre-scatter once
    {
        vector<PhantomCiphertext> batch(cfg.n_cts);
        cudaSetDevice(0);
        for (int i = 0; i < cfg.n_cts; i++)
            sk.encrypt_symmetric(ctx0, pt_input, batch[i]);
        pipe.scatter(batch);
    }

    // Ground truth compute-only: process pre-encrypted cts
    double gt_compute = 0;
    {
        cudaSetDevice(0);
        vector<PhantomCiphertext> batch(cfg.n_cts);
        for (int i = 0; i < cfg.n_cts; i++)
            sk.encrypt_symmetric(ctx0, pt_input, batch[i]);

        // Warmup
        for (int w = 0; w < cfg.warmup; w++) {
            for (auto &ct : batch) process_ct(ctx0, ct, rk, enc);
            cudaDeviceSynchronize();
            // Re-encrypt for next warmup
            for (int i = 0; i < cfg.n_cts; i++)
                sk.encrypt_symmetric(ctx0, pt_input, batch[i]);
        }

        timer.start();
        for (auto &ct : batch) process_ct(ctx0, ct, rk, enc);
        cudaDeviceSynchronize();
        gt_compute = timer.elapsed_ms();
        printf("   GT compute-only (%d cts, 1 GPU): %.2f ms\n", cfg.n_cts, gt_compute);
    }

    // Pipeline compute-only: execute on pre-scattered data
    double pipe_compute = 0;
    {
        // Re-scatter fresh data
        vector<PhantomCiphertext> batch(cfg.n_cts);
        cudaSetDevice(0);
        for (int i = 0; i < cfg.n_cts; i++)
            sk.encrypt_symmetric(ctx0, pt_input, batch[i]);
        pipe.scatter(batch);

        // Warmup
        for (int w = 0; w < cfg.warmup; w++) {
            pipe.execute([&](int gpu, PhantomContext &c, PhantomRelinKey &r,
                             PhantomCKKSEncoder &e, vector<PhantomCiphertext> &local) {
                for (auto &ct : local) process_ct(c, ct, r, e);
            });
            // Re-scatter for next warmup
            pipe.scatter(batch);
        }

        timer.start();
        pipe.execute([&](int gpu, PhantomContext &c, PhantomRelinKey &r,
                         PhantomCKKSEncoder &e, vector<PhantomCiphertext> &local) {
            for (auto &ct : local) process_ct(c, ct, r, e);
        });
        cudaSetDevice(0);
        cudaDeviceSynchronize();
        pipe_compute = timer.elapsed_ms();
        printf("   Pipeline compute-only (%d cts, %d GPUs): %.2f ms\n",
               cfg.n_cts, cfg.n_gpus, pipe_compute);
    }

    double compute_speedup = gt_compute / pipe_compute;
    printf("   COMPUTE-ONLY SPEEDUP: %.2fx\n", compute_speedup);

    // Full summary
    double speedup = gt_avg / pipe_avg;
    printf("\n=== Results ===\n");
    printf("Ciphertexts:                    %d\n", cfg.n_cts);
    printf("Ground truth (1 GPU, full):     %.2f ms\n", gt_avg);
    printf("Pipeline (%d GPUs, full):       %.2f ms\n", cfg.n_gpus, pipe_avg);
    printf("Full speedup:                   %.2fx\n", speedup);
    printf("GT compute-only:                %.2f ms\n", gt_compute);
    printf("Pipeline compute-only:          %.2f ms\n", pipe_compute);
    printf("COMPUTE-ONLY SPEEDUP:           %.2fx\n", compute_speedup);
    printf("Compute efficiency:             %.1f%%\n", compute_speedup / cfg.n_gpus * 100.0);

    pipe.destroy();
    printf("\nDone.\n");
    return 0;
}
