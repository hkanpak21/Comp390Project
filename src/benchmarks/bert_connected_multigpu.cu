/**
 * bert_connected_multigpu.cu
 *
 * Multi-GPU connected BERT pipeline with bootstrapping.
 * MatMul produces independent ciphertexts → GELU + Bootstrap pipelined across GPUs.
 *
 * Usage:
 *   ./bin/bert_connected_multigpu --n-gpus 4 --cols 16 --inner 16
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
#include "matrix_mul.cuh"
#include "bootstrapping/Bootstrapper.cuh"
#include "../multi_gpu/pipeline/ct_pipeline.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;
using namespace nexus_multi_gpu;

struct PerfTimer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        return chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
    }
};

double plain_gelu(double x) {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

int main(int argc, char **argv) {
    int n_gpus = 1, n_columns = 8, inner_dim = 16;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc) n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i+1 < argc) n_columns = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) inner_dim = atoi(argv[++i]);
    }

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < n_gpus) {
        fprintf(stderr, "Need %d GPUs, have %d\n", n_gpus, dev_count);
        return 1;
    }

    printf("================================================================\n");
    printf("  Connected BERT Pipeline — Multi-GPU (%d GPUs)\n", n_gpus);
    printf("================================================================\n");
    printf("N=65536, columns=%d, inner=%d\n", n_columns, inner_dim);
    printf("Pipeline: MatMul → GELU → Bootstrap (all at N=65536)\n\n");

    PerfTimer timer;

    // ═══ Setup ═══
    size_t N = 1ULL << 16;
    int main_mod_count = 17, bs_mod_count = 14;
    int total_levels = main_mod_count + bs_mod_count;
    int logp = 46, logq = 51, log_special = 51;

    vector<int> coeff_bits;
    coeff_bits.push_back(logq);
    for (int i = 0; i < main_mod_count; i++) coeff_bits.push_back(logp);
    for (int i = 0; i < bs_mod_count; i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(log_special);
    double SCALE = pow(2.0, logp);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));

    cudaSetDevice(0);
    PhantomContext ctx(parms);
    PhantomCKKSEncoder enc(ctx);
    PhantomSecretKey sk(ctx);
    PhantomPublicKey pk = sk.gen_publickey(ctx);
    PhantomRelinKey rk = sk.gen_relinkey(ctx);
    PhantomGaloisKey gk = sk.create_galois_keys(ctx);
    size_t slots = enc.slot_count();

    CKKSEvaluator ckks_eval(&ctx, &pk, &sk, &enc, &rk, &gk, SCALE);
    GELUEvaluator gelu_eval(ckks_eval);
    MMEvaluator mm_eval(ckks_eval);

    // Bootstrapper on GPU 0
    long logn = 15, logNh = 15;
    printf("[Setup] Initializing bootstrapper...\n");
    timer.start();
    Bootstrapper bootstrapper(10, logn, logNh, total_levels, SCALE, 25, 59, 2, 1, &ckks_eval);
    bootstrapper.slot_vec.push_back(logn);
    bootstrapper.prepare_mod_polynomial();
    bootstrapper.generate_LT_coefficient_3();
    printf("[Setup] Bootstrapper: %.1f ms\n", timer.elapsed_ms());

    // Random data
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.01, 0.01), idist(-1.0, 1.0);

    vector<vector<double>> weight_data(inner_dim, vector<double>(slots, 0.0));
    for (int j = 0; j < inner_dim; j++)
        for (size_t s = 0; s < min((size_t)768, slots); s++) weight_data[j][s] = wdist(rng);

    vector<vector<double>> input_data(n_columns, vector<double>(slots, 0.0));
    for (int i = 0; i < n_columns; i++)
        for (size_t s = 0; s < min((size_t)768, slots); s++) input_data[i][s] = idist(rng);

    // Ground truth
    vector<vector<double>> gelu_expected(n_columns, vector<double>(slots, 0.0));
    for (int i = 0; i < n_columns; i++) {
        for (int j = 0; j < inner_dim; j++)
            for (size_t s = 0; s < slots; s++)
                gelu_expected[i][s] += input_data[i][s] * weight_data[j][s];
        for (size_t s = 0; s < slots; s++)
            gelu_expected[i][s] = plain_gelu(gelu_expected[i][s]);
    }

    // ═══ Single GPU baseline ═══
    printf("\n═══ Single GPU Baseline ═══\n");

    // Encrypt
    vector<PhantomCiphertext> cts(n_columns);
    for (int i = 0; i < n_columns; i++) {
        PhantomPlaintext pt;
        ckks_eval.encoder.encode(input_data[i], SCALE, pt);
        ckks_eval.encryptor.encrypt(pt, cts[i]);
        for (int l = 0; l < bs_mod_count; l++)
            ckks_eval.evaluator.mod_switch_to_next_inplace(cts[i]);
    }

    // MatMul
    timer.start();
    vector<PhantomCiphertext> mm_res;
    mm_eval.matrix_mul_unified(cts, weight_data, n_columns, mm_res);
    cudaDeviceSynchronize();
    double mm_1gpu = timer.elapsed_ms();
    printf("  MatMul: %.1f ms\n", mm_1gpu);

    // GELU
    timer.start();
    vector<PhantomCiphertext> gelu_res(n_columns);
    for (int i = 0; i < n_columns; i++)
        gelu_eval.gelu(mm_res[i], gelu_res[i]);
    cudaDeviceSynchronize();
    double gelu_1gpu = timer.elapsed_ms();
    printf("  GELU: %.1f ms\n", gelu_1gpu);

    // Bootstrap
    timer.start();
    vector<PhantomCiphertext> boot_res(n_columns);
    for (int i = 0; i < n_columns; i++) {
        while (gelu_res[i].coeff_modulus_size() > 1)
            ckks_eval.evaluator.mod_switch_to_next_inplace(gelu_res[i]);
        bootstrapper.bootstrap_3(boot_res[i], gelu_res[i]);
    }
    cudaDeviceSynchronize();
    double boot_1gpu = timer.elapsed_ms();
    printf("  Bootstrap: %.1f ms\n", boot_1gpu);

    double total_1gpu = mm_1gpu + gelu_1gpu + boot_1gpu;
    printf("  Total: %.1f ms\n", total_1gpu);

    // ═══ Multi-GPU ═══
    if (n_gpus > 1) {
        printf("\n═══ Multi-GPU Pipeline (%d GPUs) ═══\n", n_gpus);

        CtPipeline pipe = CtPipeline::create(parms, n_gpus, sk);
        pipe.enable_galois_keys();

        // Re-encrypt fresh batch
        cudaSetDevice(0);
        vector<PhantomCiphertext> fresh_cts(n_columns);
        for (int i = 0; i < n_columns; i++) {
            PhantomPlaintext pt;
            ckks_eval.encoder.encode(input_data[i], SCALE, pt);
            ckks_eval.encryptor.encrypt(pt, fresh_cts[i]);
            for (int l = 0; l < bs_mod_count; l++)
                ckks_eval.evaluator.mod_switch_to_next_inplace(fresh_cts[i]);
        }

        // MatMul (pipeline)
        pipe.scatter(fresh_cts);
        timer.start();
        pipe.execute([&](int gpu, PhantomContext &c, PhantomRelinKey &r,
                         PhantomCKKSEncoder &e, vector<PhantomCiphertext> &local) {
            for (auto &ct : local) {
                vector<PhantomCiphertext> temp(inner_dim);
                for (int j = 0; j < inner_dim; j++) {
                    PhantomPlaintext wp;
                    e.encode(c, weight_data[j], SCALE, wp);
                    temp[j] = multiply_plain(c, ct, wp);
                }
                PhantomCiphertext acc = add(c, temp[0], temp[1]);
                for (int j = 2; j < inner_dim; j++)
                    add_inplace(c, acc, temp[j]);
                rescale_to_next_inplace(c, acc);
                ct = std::move(acc);
            }
        });
        cudaSetDevice(0); cudaDeviceSynchronize();
        double mm_pipe = timer.elapsed_ms();
        auto mm_pipe_res = pipe.gather();
        printf("  MatMul (%d GPUs): %.1f ms (%.2fx)\n", n_gpus, mm_pipe, mm_1gpu / mm_pipe);

        // GELU + Bootstrap (pipeline with full key access)
        pipe.scatter(mm_pipe_res);
        timer.start();
        pipe.execute_full([&](int gpu, PhantomContext &c, PhantomSecretKey &lsk,
                              PhantomPublicKey &lpk, PhantomRelinKey &lrk,
                              PhantomGaloisKey &lgk, PhantomCKKSEncoder &e,
                              vector<PhantomCiphertext> &local) {
            // Per-GPU CKKSEvaluator + GELU + Bootstrapper
            CKKSEvaluator local_eval(&c, &lpk, &lsk, &e, &lrk, &lgk, SCALE);
            GELUEvaluator local_gelu(local_eval);
            Bootstrapper local_bs(10, logn, logNh, total_levels, SCALE, 25, 59, 2, 1, &local_eval);
            local_bs.slot_vec.push_back(logn);
            local_bs.prepare_mod_polynomial();
            local_bs.generate_LT_coefficient_3();

            for (auto &ct : local) {
                // GELU
                PhantomCiphertext gelu_out;
                local_gelu.gelu(ct, gelu_out);
                // Bootstrap
                while (gelu_out.coeff_modulus_size() > 1)
                    local_eval.evaluator.mod_switch_to_next_inplace(gelu_out);
                PhantomCiphertext refreshed;
                local_bs.bootstrap_3(refreshed, gelu_out);
                ct = std::move(refreshed);
            }
        });
        cudaSetDevice(0); cudaDeviceSynchronize();
        double gelu_boot_pipe = timer.elapsed_ms();
        printf("  GELU+Bootstrap (%d GPUs): %.1f ms (%.2fx)\n",
               n_gpus, gelu_boot_pipe, (gelu_1gpu + boot_1gpu) / gelu_boot_pipe);

        double total_pipe = mm_pipe + gelu_boot_pipe;

        printf("\n════════════════════════════════════════════════\n");
        printf("  Stage          │ 1 GPU (ms)  │ %d GPU (ms) │ Speedup\n", n_gpus);
        printf("  ───────────────┼─────────────┼─────────────┼────────\n");
        printf("  MatMul         │ %9.1f   │ %9.1f   │ %.2fx\n",
               mm_1gpu, mm_pipe, mm_1gpu / mm_pipe);
        printf("  GELU+Bootstrap │ %9.1f   │ %9.1f   │ %.2fx\n",
               gelu_1gpu + boot_1gpu, gelu_boot_pipe, (gelu_1gpu + boot_1gpu) / gelu_boot_pipe);
        printf("  ───────────────┼─────────────┼─────────────┼────────\n");
        printf("  Total          │ %9.1f   │ %9.1f   │ %.2fx\n",
               total_1gpu, total_pipe, total_1gpu / total_pipe);
        printf("════════════════════════════════════════════════\n");
        printf("  Privacy: No re-encryption — true FHE pipeline\n");
        printf("════════════════════════════════════════════════\n");

        pipe.destroy();
    }

    printf("\nDone.\n");
    return 0;
}
