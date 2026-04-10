/**
 * bert_matmul_pipeline.cu
 *
 * Simulates NEXUS BERT MatMul using actual Phantom FHE operations
 * distributed via ciphertext-level pipeline parallelism.
 *
 * NEXUS MatMul (from matrix_mul.cu lines 252-275):
 *   for each of 64 output columns:
 *     for j in 0..767:
 *       temp[j] = multiply_plain(ct[j], weight[j])
 *     result[col] = add_many(temp)
 *     rescale(result[col])
 *
 * The 64 output columns are INDEPENDENT — perfect for pipeline parallelism.
 * Each column does 768 multiply_plain + add_many + rescale = heavy workload.
 *
 * We simulate with configurable inner_dim (number of multiply_plain per column)
 * to control workload intensity.
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
    int n_gpus = 1;
    size_t poly_degree = 8192;
    size_t n_moduli = 10;  // L=10 gives enough levels for multiply+rescale
    int n_columns = 64;    // BERT: 64 output columns
    int inner_dim = 32;    // number of multiply_plain per column (768 in real BERT)
    int warmup = 1;
    int iters = 3;
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc) cfg.n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i+1 < argc) cfg.poly_degree = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--L") && i+1 < argc) cfg.n_moduli = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i+1 < argc) cfg.n_columns = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) cfg.inner_dim = atoi(argv[++i]);
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

    printf("=== BERT MatMul Pipeline Benchmark ===\n");
    printf("GPUs: %d, N=%zu, L=%zu\n", cfg.n_gpus, cfg.poly_degree, cfg.n_moduli);
    printf("Columns: %d (independent ciphertexts)\n", cfg.n_columns);
    printf("Inner dim: %d (multiply_plain ops per column)\n", cfg.inner_dim);
    printf("warmup=%d, iters=%d\n\n", cfg.warmup, cfg.iters);

    // Setup
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(cfg.poly_degree);
    vector<int> bits = {60};
    for (size_t i = 0; i < cfg.n_moduli - 2; i++) bits.push_back(40);
    bits.push_back(60);
    parms.set_coeff_modulus(arith::CoeffModulus::Create(cfg.poly_degree, bits));
    const double SCALE = (double)(1ULL << 40);

    cudaSetDevice(0);
    PhantomContext ctx0(parms);
    PhantomSecretKey sk(ctx0);
    PhantomRelinKey rk = sk.gen_relinkey(ctx0);
    PhantomCKKSEncoder enc(ctx0);

    size_t slots = cfg.poly_degree / 2;
    vector<double> weight_data(slots, 0.01);  // small weights

    // Pre-encode weight plaintexts
    PhantomPlaintext weight_plain;
    enc.encode(ctx0, weight_data, SCALE, weight_plain);

    Timer timer;

    // Create "input" ciphertexts (one per output column, as in NEXUS MatMul)
    printf("[1] Creating %d column ciphertexts...\n", cfg.n_columns);
    timer.start();

    vector<double> input_data(slots, 1.0);
    PhantomPlaintext input_pt;
    enc.encode(ctx0, input_data, SCALE, input_pt);

    vector<PhantomCiphertext> col_cts(cfg.n_columns);
    for (int i = 0; i < cfg.n_columns; i++) {
        sk.encrypt_symmetric(ctx0, input_pt, col_cts[i]);
    }
    printf("   Created in %.1f ms\n", timer.elapsed_ms());

    // The MatMul column workload:
    // For each column: inner_dim multiply_plain + add accumulation + rescale
    // This mirrors NEXUS's: temp[j] = multiply_plain(ct, weight[j]); add_many(temp)
    auto matmul_column = [&](PhantomContext &c, PhantomCiphertext &col_ct,
                             PhantomRelinKey &r, PhantomCKKSEncoder &e) {
        PhantomPlaintext wp;
        e.encode(c, weight_data, col_ct.scale(), wp);

        // Accumulate inner_dim multiply_plain + add results
        // First multiply
        PhantomCiphertext acc;
        acc.resize(c, col_ct.chain_index(), col_ct.size(), cudaStreamPerThread);
        size_t ct_bytes = col_ct.size() * col_ct.coeff_modulus_size() * col_ct.poly_modulus_degree() * sizeof(uint64_t);
        cudaMemcpy(acc.data(), col_ct.data(), ct_bytes, cudaMemcpyDeviceToDevice);
        acc.set_scale(col_ct.scale());
        acc.set_ntt_form(col_ct.is_ntt_form());
        multiply_plain_inplace(c, acc, wp);

        // Accumulate remaining
        for (int j = 1; j < cfg.inner_dim; j++) {
            PhantomCiphertext temp;
            temp.resize(c, col_ct.chain_index(), col_ct.size(), cudaStreamPerThread);
            cudaMemcpy(temp.data(), col_ct.data(), ct_bytes, cudaMemcpyDeviceToDevice);
            temp.set_scale(col_ct.scale());
            temp.set_ntt_form(col_ct.is_ntt_form());
            multiply_plain_inplace(c, temp, wp);
            add_inplace(c, acc, temp);
        }

        // Rescale result
        rescale_to_next_inplace(c, acc);

        // Write back
        col_ct = std::move(acc);
    };

    // === GROUND TRUTH: 1 GPU, all columns serial ===
    printf("[2] Ground truth (1 GPU, %d columns serial)...\n", cfg.n_columns);

    // Warmup
    for (int w = 0; w < cfg.warmup; w++) {
        vector<PhantomCiphertext> batch(cfg.n_columns);
        for (int i = 0; i < cfg.n_columns; i++)
            sk.encrypt_symmetric(ctx0, input_pt, batch[i]);
        for (auto &ct : batch) matmul_column(ctx0, ct, rk, enc);
        cudaDeviceSynchronize();
    }

    // Timed
    double gt_total = 0;
    for (int it = 0; it < cfg.iters; it++) {
        vector<PhantomCiphertext> batch(cfg.n_columns);
        for (int i = 0; i < cfg.n_columns; i++)
            sk.encrypt_symmetric(ctx0, input_pt, batch[i]);

        timer.start();
        for (auto &ct : batch) matmul_column(ctx0, ct, rk, enc);
        cudaDeviceSynchronize();
        gt_total += timer.elapsed_ms();
    }
    double gt_avg = gt_total / cfg.iters;
    printf("   Ground truth: %.1f ms per MatMul (%d cols × %d inner)\n",
           gt_avg, cfg.n_columns, cfg.inner_dim);

    // === PIPELINE: N GPUs ===
    printf("[3] Pipeline (%d GPUs, %d cols / %d = %d per GPU)...\n",
           cfg.n_gpus, cfg.n_columns, cfg.n_gpus, cfg.n_columns / cfg.n_gpus);

    CtPipeline pipe = CtPipeline::create(parms, cfg.n_gpus, sk);

    // Warmup
    for (int w = 0; w < cfg.warmup; w++) {
        vector<PhantomCiphertext> batch(cfg.n_columns);
        cudaSetDevice(0);
        for (int i = 0; i < cfg.n_columns; i++)
            sk.encrypt_symmetric(ctx0, input_pt, batch[i]);
        pipe.scatter(batch);
        pipe.execute([&](int gpu, PhantomContext &c, PhantomRelinKey &r,
                         PhantomCKKSEncoder &e, vector<PhantomCiphertext> &local) {
            for (auto &ct : local) matmul_column(c, ct, r, e);
        });
    }

    // Timed
    double pipe_total = 0;
    for (int it = 0; it < cfg.iters; it++) {
        vector<PhantomCiphertext> batch(cfg.n_columns);
        cudaSetDevice(0);
        for (int i = 0; i < cfg.n_columns; i++)
            sk.encrypt_symmetric(ctx0, input_pt, batch[i]);
        pipe.scatter(batch);

        timer.start();
        pipe.execute([&](int gpu, PhantomContext &c, PhantomRelinKey &r,
                         PhantomCKKSEncoder &e, vector<PhantomCiphertext> &local) {
            for (auto &ct : local) matmul_column(c, ct, r, e);
        });
        cudaSetDevice(0);
        cudaDeviceSynchronize();
        pipe_total += timer.elapsed_ms();
    }
    double pipe_avg = pipe_total / cfg.iters;

    printf("   Pipeline: %.1f ms per MatMul\n", pipe_avg);

    // Summary
    double speedup = gt_avg / pipe_avg;
    printf("\n=== Results ===\n");
    printf("BERT MatMul: %d columns × %d inner_dim\n", cfg.n_columns, cfg.inner_dim);
    printf("Ground truth (1 GPU):     %.1f ms\n", gt_avg);
    printf("Pipeline (%d GPUs):       %.1f ms\n", cfg.n_gpus, pipe_avg);
    printf("Speedup:                  %.2fx\n", speedup);
    printf("Efficiency:               %.1f%%\n", speedup / cfg.n_gpus * 100.0);
    printf("Per-column (1 GPU):       %.3f ms\n", gt_avg / cfg.n_columns);
    printf("Per-column (pipeline):    %.3f ms\n", pipe_avg / (cfg.n_columns / cfg.n_gpus));

    pipe.destroy();
    printf("\nDone.\n");
    return 0;
}
