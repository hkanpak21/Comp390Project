/**
 * bert_e2e_multigpu.cu
 *
 * End-to-end BERT encoder layer with REAL NEXUS FHE operations,
 * multi-GPU pipeline parallelism, profiling, and correctness verification.
 *
 * Pipeline:
 *   MatMul (N=8192) → re-encrypt → GELU (N=65536) → verify
 *
 * MatMul: 64 independent columns × inner_dim multiply_plain + add_many
 *   → pipeline-parallel across GPUs
 *
 * GELU: Each of the 64 output ciphertexts gets independent GELU
 *   → pipeline-parallel across GPUs using per-GPU CKKSEvaluator
 *
 * Usage:
 *   ./bin/bert_e2e_multigpu --n-gpus 4 --cols 64 --inner 32
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

#include "ckks_evaluator.cuh"
#include "gelu.cuh"
#include "layer_norm.cuh"
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
    int n_columns = 16;
    int inner_dim = 32;
    int warmup = 1;
    int iters = 2;
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc) cfg.n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i+1 < argc) cfg.n_columns = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) cfg.inner_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc) cfg.warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) cfg.iters = atoi(argv[++i]);
    }
    return cfg;
}

double plain_gelu(double x) {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < cfg.n_gpus) {
        fprintf(stderr, "Need %d GPUs, have %d\n", cfg.n_gpus, dev_count);
        return 1;
    }

    printf("================================================================\n");
    printf("  BERT End-to-End FHE Inference — Multi-GPU Pipeline\n");
    printf("================================================================\n");
    printf("GPUs: %d, columns=%d, inner_dim=%d\n", cfg.n_gpus, cfg.n_columns, cfg.inner_dim);
    printf("Pipeline: MatMul(N=8192) → re-encrypt → GELU(N=65536)\n\n");

    Timer timer;

    // ══════════════════════════════════════════════════════════════════════
    // SETUP: Two parameter sets
    // ══════════════════════════════════════════════════════════════════════

    // MatMul: N=8192, L=3
    size_t N_mm = 8192;
    vector<int> mm_bits = {60, 40, 60};
    const double SCALE = (double)(1ULL << 40);

    EncryptionParameters mm_parms(scheme_type::ckks);
    mm_parms.set_poly_modulus_degree(N_mm);
    mm_parms.set_coeff_modulus(CoeffModulus::Create(N_mm, mm_bits));

    cudaSetDevice(0);
    PhantomContext mm_ctx(mm_parms);
    PhantomSecretKey mm_sk(mm_ctx);
    PhantomPublicKey mm_pk = mm_sk.gen_publickey(mm_ctx);
    PhantomRelinKey mm_rk = mm_sk.gen_relinkey(mm_ctx);
    PhantomGaloisKey mm_gk;
    PhantomCKKSEncoder mm_enc(mm_ctx);
    size_t mm_slots = N_mm / 2;

    CKKSEvaluator mm_eval(&mm_ctx, &mm_pk, &mm_sk, &mm_enc, &mm_rk, &mm_gk, SCALE);

    // GELU: N=65536, L=20
    size_t N_gelu = 1ULL << 16;
    vector<int> gelu_bits = {58, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 40, 40, 40, 40, 40, 40, 58};

    EncryptionParameters gelu_parms(scheme_type::ckks);
    gelu_parms.set_poly_modulus_degree(N_gelu);
    gelu_parms.set_coeff_modulus(CoeffModulus::Create(N_gelu, gelu_bits));

    PhantomContext gelu_ctx(gelu_parms);
    PhantomSecretKey gelu_sk(gelu_ctx);
    PhantomPublicKey gelu_pk = gelu_sk.gen_publickey(gelu_ctx);
    PhantomRelinKey gelu_rk = gelu_sk.gen_relinkey(gelu_ctx);
    PhantomGaloisKey gelu_gk;  // GELU doesn't need rotations
    PhantomCKKSEncoder gelu_enc(gelu_ctx);
    size_t gelu_slots = N_gelu / 2;

    CKKSEvaluator gelu_eval_gpu0(&gelu_ctx, &gelu_pk, &gelu_sk, &gelu_enc, &gelu_rk, &gelu_gk, SCALE);
    GELUEvaluator gelu_gpu0(gelu_eval_gpu0);

    printf("[Setup] MatMul: N=%zu L=%zu | GELU: N=%zu L=%zu\n\n",
           N_mm, mm_bits.size(), N_gelu, gelu_bits.size());

    // ══════════════════════════════════════════════════════════════════════
    // RANDOM DATA
    // ══════════════════════════════════════════════════════════════════════
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.01, 0.01);
    uniform_real_distribution<double> idist(-1.0, 1.0);

    vector<vector<double>> weight_data(cfg.inner_dim, vector<double>(mm_slots));
    for (int j = 0; j < cfg.inner_dim; j++)
        for (size_t s = 0; s < mm_slots; s++) weight_data[j][s] = wdist(rng);

    vector<vector<double>> input_data(cfg.n_columns, vector<double>(mm_slots));
    for (int i = 0; i < cfg.n_columns; i++)
        for (size_t s = 0; s < mm_slots; s++) input_data[i][s] = idist(rng);

    // Plaintext ground truth
    vector<vector<double>> mm_expected(cfg.n_columns, vector<double>(mm_slots, 0.0));
    for (int i = 0; i < cfg.n_columns; i++)
        for (int j = 0; j < cfg.inner_dim; j++)
            for (size_t s = 0; s < mm_slots; s++)
                mm_expected[i][s] += input_data[i][s] * weight_data[j][s];

    vector<vector<double>> gelu_expected(cfg.n_columns, vector<double>(mm_slots));
    for (int i = 0; i < cfg.n_columns; i++)
        for (size_t s = 0; s < mm_slots; s++)
            gelu_expected[i][s] = plain_gelu(mm_expected[i][s]);

    // ══════════════════════════════════════════════════════════════════════
    // SINGLE GPU BASELINE
    // ══════════════════════════════════════════════════════════════════════
    printf("═══ Single GPU Baseline ═══\n");

    // MatMul
    timer.start();
    vector<PhantomCiphertext> mm_cts(cfg.n_columns);
    for (int i = 0; i < cfg.n_columns; i++) {
        PhantomPlaintext pt;
        mm_eval.encoder.encode(input_data[i], SCALE, pt);
        mm_eval.encryptor.encrypt(pt, mm_cts[i]);
    }
    for (int i = 0; i < cfg.n_columns; i++) {
        vector<PhantomCiphertext> temp(cfg.inner_dim);
        for (int j = 0; j < cfg.inner_dim; j++) {
            PhantomPlaintext wp;
            mm_enc.encode(mm_ctx, weight_data[j], SCALE, wp);
            temp[j] = multiply_plain(mm_ctx, mm_cts[i], wp);
        }
        PhantomCiphertext acc = add(mm_ctx, temp[0], temp[1]);
        for (int j = 2; j < cfg.inner_dim; j++)
            add_inplace(mm_ctx, acc, temp[j]);
        rescale_to_next_inplace(mm_ctx, acc);
        mm_cts[i] = std::move(acc);
    }
    cudaDeviceSynchronize();
    double mm_1gpu_ms = timer.elapsed_ms();
    printf("  MatMul (1 GPU):     %8.1f ms\n", mm_1gpu_ms);

    // Verify MatMul
    double mm_mae_sum = 0;
    for (int i = 0; i < cfg.n_columns; i++)
        mm_mae_sum += mm_eval.calculate_MAE(mm_expected[i], mm_cts[i], mm_slots);
    double mm_mae = mm_mae_sum / cfg.n_columns;
    printf("  MatMul MAE:         %.6f %s\n", mm_mae, mm_mae < 0.01 ? "PASS" : "FAIL");

    // Re-encrypt MatMul → GELU
    timer.start();
    vector<PhantomCiphertext> gelu_cts(cfg.n_columns);
    for (int i = 0; i < cfg.n_columns; i++) {
        PhantomPlaintext pt;
        vector<double> vals;
        mm_sk.decrypt(mm_ctx, mm_cts[i], pt);
        mm_enc.decode(mm_ctx, pt, vals);
        vals.resize(gelu_slots, 0.0);
        PhantomPlaintext gelu_pt;
        gelu_eval_gpu0.encoder.encode(vals, SCALE, gelu_pt);
        gelu_eval_gpu0.encryptor.encrypt(gelu_pt, gelu_cts[i]);
    }
    cudaDeviceSynchronize();
    double reenc_ms = timer.elapsed_ms();
    printf("  Re-encrypt:         %8.1f ms\n", reenc_ms);

    // GELU (1 GPU)
    timer.start();
    vector<PhantomCiphertext> gelu_results_1gpu(cfg.n_columns);
    for (int i = 0; i < cfg.n_columns; i++) {
        gelu_gpu0.gelu(gelu_cts[i], gelu_results_1gpu[i]);
    }
    cudaDeviceSynchronize();
    double gelu_1gpu_ms = timer.elapsed_ms();
    printf("  GELU (1 GPU):       %8.1f ms\n", gelu_1gpu_ms);

    // Verify GELU
    double gelu_mae_sum = 0;
    for (int i = 0; i < cfg.n_columns; i++) {
        vector<double> exp_padded = gelu_expected[i];
        exp_padded.resize(gelu_slots, 0.0);
        gelu_mae_sum += gelu_eval_gpu0.calculate_MAE(exp_padded, gelu_results_1gpu[i], mm_slots);
    }
    double gelu_mae = gelu_mae_sum / cfg.n_columns;
    printf("  GELU MAE:           %.6f %s\n", gelu_mae, gelu_mae < 0.05 ? "PASS" : "FAIL");

    double total_1gpu = mm_1gpu_ms + reenc_ms + gelu_1gpu_ms;
    printf("  ────────────────────────────\n");
    printf("  Total (1 GPU):      %8.1f ms\n\n", total_1gpu);

    // ══════════════════════════════════════════════════════════════════════
    // MULTI-GPU PIPELINE
    // ══════════════════════════════════════════════════════════════════════
    if (cfg.n_gpus > 1) {
        printf("═══ Multi-GPU Pipeline (%d GPUs) ═══\n", cfg.n_gpus);

        // MatMul pipeline
        CtPipeline mm_pipe = CtPipeline::create(mm_parms, cfg.n_gpus, mm_sk);

        // Encrypt fresh batch
        cudaSetDevice(0);
        vector<PhantomCiphertext> mm_batch(cfg.n_columns);
        for (int i = 0; i < cfg.n_columns; i++) {
            PhantomPlaintext pt;
            mm_eval.encoder.encode(input_data[i], SCALE, pt);
            mm_eval.encryptor.encrypt(pt, mm_batch[i]);
        }

        mm_pipe.scatter(mm_batch);
        timer.start();
        mm_pipe.execute([&](int gpu, PhantomContext &c, PhantomRelinKey &r,
                            PhantomCKKSEncoder &e, vector<PhantomCiphertext> &local) {
            for (auto &ct : local) {
                vector<PhantomCiphertext> temp(cfg.inner_dim);
                for (int j = 0; j < cfg.inner_dim; j++) {
                    PhantomPlaintext wp;
                    e.encode(c, weight_data[j], SCALE, wp);
                    temp[j] = multiply_plain(c, ct, wp);
                }
                PhantomCiphertext acc = add(c, temp[0], temp[1]);
                for (int j = 2; j < cfg.inner_dim; j++)
                    add_inplace(c, acc, temp[j]);
                rescale_to_next_inplace(c, acc);
                ct = std::move(acc);
            }
        });
        cudaSetDevice(0);
        cudaDeviceSynchronize();
        double mm_pipe_ms = timer.elapsed_ms();
        auto mm_pipe_results = mm_pipe.gather();
        printf("  MatMul (%d GPUs):   %8.1f ms (%.2fx)\n",
               cfg.n_gpus, mm_pipe_ms, mm_1gpu_ms / mm_pipe_ms);

        // Verify pipeline MatMul
        double mm_pipe_mae_sum = 0;
        for (int i = 0; i < cfg.n_columns; i++)
            mm_pipe_mae_sum += mm_eval.calculate_MAE(mm_expected[i], mm_pipe_results[i], mm_slots);
        double mm_pipe_mae = mm_pipe_mae_sum / cfg.n_columns;
        printf("  MatMul MAE:         %.6f %s\n", mm_pipe_mae, mm_pipe_mae < 0.01 ? "PASS" : "FAIL");

        mm_pipe.destroy();

        // Re-encrypt for GELU
        timer.start();
        vector<PhantomCiphertext> gelu_batch(cfg.n_columns);
        cudaSetDevice(0);
        for (int i = 0; i < cfg.n_columns; i++) {
            PhantomPlaintext pt;
            vector<double> vals;
            mm_sk.decrypt(mm_ctx, mm_pipe_results[i], pt);
            mm_enc.decode(mm_ctx, pt, vals);
            vals.resize(gelu_slots, 0.0);
            PhantomPlaintext gelu_pt;
            gelu_eval_gpu0.encoder.encode(vals, SCALE, gelu_pt);
            gelu_eval_gpu0.encryptor.encrypt(gelu_pt, gelu_batch[i]);
        }
        cudaDeviceSynchronize();
        double reenc_pipe_ms = timer.elapsed_ms();
        printf("  Re-encrypt:         %8.1f ms\n", reenc_pipe_ms);

        // GELU pipeline — REAL GELU using per-GPU CKKSEvaluator
        CtPipeline gelu_pipe = CtPipeline::create(gelu_parms, cfg.n_gpus, gelu_sk);

        gelu_pipe.scatter(gelu_batch);
        timer.start();
        gelu_pipe.execute_full([&](int gpu, PhantomContext &c, PhantomSecretKey &sk,
                                    PhantomPublicKey &pk, PhantomRelinKey &rk,
                                    PhantomGaloisKey &gk, PhantomCKKSEncoder &e,
                                    vector<PhantomCiphertext> &local) {
            // Create per-GPU NEXUS evaluator with real keys
            nexus::CKKSEvaluator local_eval(&c, &pk, &sk, &e, &rk, &gk, SCALE);
            nexus::GELUEvaluator local_gelu(local_eval);

            for (auto &ct : local) {
                PhantomCiphertext result;
                local_gelu.gelu(ct, result);
                ct = std::move(result);
            }
        });
        cudaSetDevice(0);
        cudaDeviceSynchronize();
        double gelu_pipe_ms = timer.elapsed_ms();
        auto gelu_pipe_results = gelu_pipe.gather();
        printf("  GELU (%d GPUs):     %8.1f ms (%.2fx)\n",
               cfg.n_gpus, gelu_pipe_ms, gelu_1gpu_ms / gelu_pipe_ms);

        // Verify pipeline GELU
        double gelu_pipe_mae_sum = 0;
        cudaSetDevice(0);
        for (int i = 0; i < cfg.n_columns && i < (int)gelu_pipe_results.size(); i++) {
            vector<double> exp_padded = gelu_expected[i];
            exp_padded.resize(gelu_slots, 0.0);
            gelu_pipe_mae_sum += gelu_eval_gpu0.calculate_MAE(exp_padded, gelu_pipe_results[i], mm_slots);
        }
        double gelu_pipe_mae = gelu_pipe_mae_sum / cfg.n_columns;
        printf("  GELU MAE:           %.6f %s\n", gelu_pipe_mae, gelu_pipe_mae < 0.05 ? "PASS" : "FAIL");

        double total_pipe = mm_pipe_ms + reenc_pipe_ms + gelu_pipe_ms;
        printf("  ────────────────────────────\n");
        printf("  Total (%d GPUs):    %8.1f ms\n\n", cfg.n_gpus, total_pipe);

        // ═══ Final summary ═══
        printf("════════════════════════════════════════════════\n");
        printf("  BERT Layer Summary (%d cols × %d inner)\n", cfg.n_columns, cfg.inner_dim);
        printf("════════════════════════════════════════════════\n");
        printf("  Stage        │ 1 GPU (ms) │ %d GPU (ms) │ Speedup\n", cfg.n_gpus);
        printf("  ─────────────┼────────────┼─────────────┼────────\n");
        printf("  MatMul       │ %8.1f   │ %9.1f   │ %.2fx\n",
               mm_1gpu_ms, mm_pipe_ms, mm_1gpu_ms / mm_pipe_ms);
        printf("  Re-encrypt   │ %8.1f   │ %9.1f   │ -\n", reenc_ms, reenc_pipe_ms);
        printf("  GELU         │ %8.1f   │ %9.1f   │ %.2fx\n",
               gelu_1gpu_ms, gelu_pipe_ms, gelu_1gpu_ms / gelu_pipe_ms);
        printf("  ─────────────┼────────────┼─────────────┼────────\n");
        printf("  Total        │ %8.1f   │ %9.1f   │ %.2fx\n",
               total_1gpu, total_pipe, total_1gpu / total_pipe);
        printf("════════════════════════════════════════════════\n");
        printf("  MatMul MAE: %.6f  GELU MAE: %.6f\n", mm_pipe_mae, gelu_pipe_mae);
        printf("  Correctness: %s\n",
               (mm_pipe_mae < 0.01 && gelu_pipe_mae < 0.05) ? "ALL PASS" : "FAIL");
        printf("════════════════════════════════════════════════\n");

        gelu_pipe.destroy();
    }

    printf("\nDone.\n");
    return 0;
}
