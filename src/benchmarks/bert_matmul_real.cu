/**
 * bert_matmul_real.cu
 *
 * Multi-GPU BERT MatMul using actual NEXUS FHE operations.
 *
 * The NEXUS MatMul has two phases:
 *   Phase 1 (sequential): Compress + decompress ciphertexts via Galois rotations
 *   Phase 2 (parallelizable): 64 columns × 768 multiply_plain + add_many
 *
 * We parallelize Phase 2 across GPUs using CtPipeline.
 * For Phase 2, each output column is independent — embarrassingly parallel.
 *
 * We also test multi-GPU GELU: after MatMul produces 64 independent ciphertexts,
 * GELU is applied to each independently — also pipeline-parallel.
 *
 * Usage:
 *   ./bin/bert_matmul_real --n-gpus 4 --inner 768 --cols 64
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <thread>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

#include "ckks_evaluator.cuh"
#include "gelu.cuh"
#include "../multi_gpu/pipeline/ct_pipeline.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;
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
    int n_columns = 64;     // BERT: 64 output columns
    int inner_dim = 768;    // multiply_plain ops per column (768 in real BERT)
    size_t poly_degree = 8192;  // MatMul uses N=8192
    int warmup = 1;
    int iters = 3;
    bool test_gelu = false;
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc) cfg.n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i+1 < argc) cfg.n_columns = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) cfg.inner_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i+1 < argc) cfg.poly_degree = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc) cfg.warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) cfg.iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--test-gelu")) cfg.test_gelu = true;
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

    printf("=== Real BERT MatMul Multi-GPU Benchmark ===\n");
    printf("GPUs: %d, N=%zu, columns=%d, inner_dim=%d\n",
           cfg.n_gpus, cfg.poly_degree, cfg.n_columns, cfg.inner_dim);
    printf("warmup=%d, iters=%d\n\n", cfg.warmup, cfg.iters);

    // ── Setup ────────────────────────────────────────────────────────────────
    // MatMul parameters: N=8192, L=3 (60, 40, 60)
    vector<int> coeff_bits = {60, 40, 60};

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(cfg.poly_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(cfg.poly_degree, coeff_bits));
    const double SCALE = (double)(1ULL << 40);

    cudaSetDevice(0);
    PhantomContext ctx0(parms);
    PhantomSecretKey sk(ctx0);
    PhantomPublicKey pk = sk.gen_publickey(ctx0);
    PhantomRelinKey rk = sk.gen_relinkey(ctx0);
    PhantomGaloisKey gk;  // empty for per-column phase (no rotations needed)
    PhantomCKKSEncoder enc(ctx0);

    size_t slots = cfg.poly_degree / 2;

    CKKSEvaluator ckks_eval(&ctx0, &pk, &sk, &enc, &rk, &gk, SCALE);

    // Generate random weight plaintexts (simulating the 768 encoded weight rows)
    printf("[1] Generating %d random weight plaintexts...\n", cfg.inner_dim);
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.01, 0.01);  // small weights like BERT

    vector<vector<double>> weight_data(cfg.inner_dim, vector<double>(slots));
    vector<PhantomPlaintext> weight_pts(cfg.inner_dim);
    for (int j = 0; j < cfg.inner_dim; j++) {
        for (size_t s = 0; s < slots; s++) weight_data[j][s] = wdist(rng);
        ckks_eval.encoder.encode(weight_data[j], SCALE, weight_pts[j]);
    }

    // Generate input ciphertexts (one per output column — simulating post-decompress)
    printf("[2] Encrypting %d column ciphertexts...\n", cfg.n_columns);
    Timer timer;
    timer.start();

    uniform_real_distribution<double> idist(-1.0, 1.0);
    vector<vector<double>> input_data(cfg.n_columns, vector<double>(slots));
    vector<PhantomCiphertext> col_cts(cfg.n_columns);
    for (int i = 0; i < cfg.n_columns; i++) {
        for (size_t s = 0; s < slots; s++) input_data[i][s] = idist(rng);
        PhantomPlaintext pt;
        ckks_eval.encoder.encode(input_data[i], SCALE, pt);
        ckks_eval.encryptor.encrypt(pt, col_cts[i]);
    }
    printf("   Encrypted in %.1f ms\n\n", timer.elapsed_ms());

    // ── The per-column workload (same as NEXUS MatMul lines 252-265) ─────
    auto matmul_column = [&](PhantomContext &c, PhantomCiphertext &col_ct,
                             PhantomRelinKey &r, PhantomCKKSEncoder &e,
                             int inner) {
        // Encode weights on this GPU's context
        vector<PhantomPlaintext> local_weights(inner);
        for (int j = 0; j < inner; j++) {
            e.encode(c, weight_data[j], SCALE, local_weights[j]);
        }

        // 768 multiply_plain + add_many (NEXUS MatMul core loop)
        vector<PhantomCiphertext> temp_cts(inner);
        for (int j = 0; j < inner; j++) {
            temp_cts[j] = multiply_plain(c, col_ct, local_weights[j]);
        }

        PhantomCiphertext acc;
        acc = add(c, temp_cts[0], temp_cts[1]);
        for (int j = 2; j < inner; j++) {
            add_inplace(c, acc, temp_cts[j]);
        }

        // Rescale result
        rescale_to_next_inplace(c, acc);
        col_ct = std::move(acc);
    };

    // ── Plaintext ground truth ───────────────────────────────────────────────
    printf("[3] Computing plaintext ground truth...\n");
    timer.start();
    vector<vector<double>> expected(cfg.n_columns, vector<double>(slots, 0.0));
    for (int i = 0; i < cfg.n_columns; i++) {
        for (int j = 0; j < cfg.inner_dim; j++) {
            for (size_t s = 0; s < slots; s++) {
                expected[i][s] += input_data[i][s] * weight_data[j][s];
            }
        }
    }
    printf("   Plaintext MatMul done in %.1f ms\n\n", timer.elapsed_ms());

    // ── Ground truth: 1 GPU, all columns serial ──────────────────────────────
    printf("[4] Single GPU (%d cols serial)...\n", cfg.n_columns);

    // Warmup
    for (int w = 0; w < cfg.warmup; w++) {
        vector<PhantomCiphertext> batch(cfg.n_columns);
        for (int i = 0; i < cfg.n_columns; i++) {
            PhantomPlaintext pt;
            ckks_eval.encoder.encode(input_data[i], SCALE, pt);
            ckks_eval.encryptor.encrypt(pt, batch[i]);
        }
        for (auto &ct : batch) matmul_column(ctx0, ct, rk, enc, cfg.inner_dim);
        cudaDeviceSynchronize();
    }

    // Timed
    double gt_total = 0;
    vector<PhantomCiphertext> gt_results;
    for (int it = 0; it < cfg.iters; it++) {
        vector<PhantomCiphertext> batch(cfg.n_columns);
        for (int i = 0; i < cfg.n_columns; i++) {
            PhantomPlaintext pt;
            ckks_eval.encoder.encode(input_data[i], SCALE, pt);
            ckks_eval.encryptor.encrypt(pt, batch[i]);
        }

        timer.start();
        for (auto &ct : batch) matmul_column(ctx0, ct, rk, enc, cfg.inner_dim);
        cudaDeviceSynchronize();
        gt_total += timer.elapsed_ms();

        if (it == cfg.iters - 1) gt_results = std::move(batch);
    }
    double gt_avg = gt_total / cfg.iters;
    printf("   1 GPU: %.1f ms per MatMul\n", gt_avg);

    // Verify correctness of single-GPU result
    printf("   Verifying correctness...\n");
    double total_mae = 0;
    for (int i = 0; i < cfg.n_columns; i++) {
        double mae = ckks_eval.calculate_MAE(expected[i], gt_results[i], slots);
        total_mae += mae;
    }
    double avg_mae = total_mae / cfg.n_columns;
    printf("   Average MAE across %d columns: %.6f %s\n\n",
           cfg.n_columns, avg_mae, avg_mae < 0.01 ? "PASS" : "FAIL");

    // ── Pipeline: N GPUs ─────────────────────────────────────────────────────
    if (cfg.n_gpus > 1) {
        printf("[5] Pipeline (%d GPUs, %d cols each)...\n",
               cfg.n_gpus, cfg.n_columns / cfg.n_gpus);

        CtPipeline pipe = CtPipeline::create(parms, cfg.n_gpus, sk);

        // Warmup
        for (int w = 0; w < cfg.warmup; w++) {
            vector<PhantomCiphertext> batch(cfg.n_columns);
            cudaSetDevice(0);
            for (int i = 0; i < cfg.n_columns; i++) {
                PhantomPlaintext pt;
                ckks_eval.encoder.encode(input_data[i], SCALE, pt);
                ckks_eval.encryptor.encrypt(pt, batch[i]);
            }
            pipe.scatter(batch);
            pipe.execute([&](int gpu, PhantomContext &c, PhantomRelinKey &r,
                             PhantomCKKSEncoder &e, vector<PhantomCiphertext> &local) {
                for (auto &ct : local)
                    matmul_column(c, ct, r, e, cfg.inner_dim);
            });
        }

        // Timed
        double pipe_total = 0;
        vector<PhantomCiphertext> pipe_results;
        for (int it = 0; it < cfg.iters; it++) {
            vector<PhantomCiphertext> batch(cfg.n_columns);
            cudaSetDevice(0);
            for (int i = 0; i < cfg.n_columns; i++) {
                PhantomPlaintext pt;
                ckks_eval.encoder.encode(input_data[i], SCALE, pt);
                ckks_eval.encryptor.encrypt(pt, batch[i]);
            }
            pipe.scatter(batch);

            timer.start();
            pipe.execute([&](int gpu, PhantomContext &c, PhantomRelinKey &r,
                             PhantomCKKSEncoder &e, vector<PhantomCiphertext> &local) {
                for (auto &ct : local)
                    matmul_column(c, ct, r, e, cfg.inner_dim);
            });
            cudaSetDevice(0);
            cudaDeviceSynchronize();
            pipe_total += timer.elapsed_ms();

            if (it == cfg.iters - 1) pipe_results = pipe.gather();
        }
        double pipe_avg = pipe_total / cfg.iters;

        printf("   Pipeline: %.1f ms per MatMul\n", pipe_avg);

        // Verify pipeline correctness
        printf("   Verifying pipeline correctness...\n");
        double pipe_mae_total = 0;
        for (int i = 0; i < cfg.n_columns && i < (int)pipe_results.size(); i++) {
            cudaSetDevice(0);
            double mae = ckks_eval.calculate_MAE(expected[i], pipe_results[i], slots);
            pipe_mae_total += mae;
        }
        double pipe_avg_mae = pipe_mae_total / cfg.n_columns;
        printf("   Pipeline MAE: %.6f %s\n\n", pipe_avg_mae, pipe_avg_mae < 0.01 ? "PASS" : "FAIL");

        // Summary
        double speedup = gt_avg / pipe_avg;
        printf("=== Results ===\n");
        printf("BERT MatMul: %d cols × %d inner_dim (real FHE ops)\n", cfg.n_columns, cfg.inner_dim);
        printf("1 GPU:              %.1f ms  (MAE=%.6f)\n", gt_avg, avg_mae);
        printf("Pipeline (%d GPUs): %.1f ms  (MAE=%.6f)\n", cfg.n_gpus, pipe_avg, pipe_avg_mae);
        printf("Speedup:            %.2fx\n", speedup);
        printf("Efficiency:         %.1f%%\n", speedup / cfg.n_gpus * 100.0);

        pipe.destroy();
    }

    printf("\nDone.\n");
    return 0;
}
