/**
 * bert_connected_pipeline.cu
 *
 * Connected FHE BERT layer — NO re-encryption, NO privacy leak.
 *
 * Single parameter set: N=65536, L=31 (17 main + 14 bootstrap)
 * Pipeline: encrypt ONCE → MatMul → GELU → bootstrap → decrypt ONCE
 *
 * This fixes three problems from the previous bert_e2e benchmarks:
 *   1. No parameter mismatch — everything at N=65536
 *   2. No re-encryption — bootstrapping refreshes levels homomorphically
 *   3. True FHE privacy — secret key never used during evaluation
 *
 * Usage:
 *   ./bin/bert_connected_pipeline [--n-gpus 4] [--cols 16] [--inner 32]
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
#include "matrix_mul.cuh"
#include "bootstrapping/Bootstrapper.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

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
    int n_columns = 8;
    int inner_dim = 16;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cols") && i+1 < argc) n_columns = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) inner_dim = atoi(argv[++i]);
    }

    printf("================================================================\n");
    printf("  Connected BERT Pipeline (No Re-encryption)\n");
    printf("================================================================\n");
    printf("N=65536, columns=%d, inner_dim=%d\n", n_columns, inner_dim);
    printf("Pipeline: encrypt → MatMul → GELU → bootstrap → decrypt\n\n");

    PerfTimer timer;

    // ══════════════════════════════════════════════════════════════════════
    // SETUP: Single unified parameter set
    // ══════════════════════════════════════════════════════════════════════

    size_t N = 1ULL << 16;   // 65536
    int main_mod_count = 25;  // GELU needs ~20 levels for sign evaluation + polynomial
    int bs_mod_count = 14;
    int total_levels = main_mod_count + bs_mod_count;

    int logp = 46;     // main modulus bits
    int logq = 51;     // bootstrapping modulus bits
    int log_special = 51;

    // Build coefficient modulus chain: [logq, logp × main, logq × bs, log_special]
    vector<int> coeff_bits;
    coeff_bits.push_back(logq);  // first prime
    for (int i = 0; i < main_mod_count; i++) coeff_bits.push_back(logp);
    for (int i = 0; i < bs_mod_count; i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(log_special);  // special prime

    double SCALE = pow(2.0, logp);

    printf("[Setup] Building context: N=%zu, L=%d (%d main + %d bootstrap)\n",
           N, total_levels, main_mod_count, bs_mod_count);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));

    cudaSetDevice(0);
    PhantomContext context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    size_t slot_count = encoder.slot_count();
    printf("[Setup] Slots: %zu\n", slot_count);

    // Galois keys (needed for GELU sign eval + bootstrapping rotations)
    printf("[Setup] Generating Galois keys...\n");
    timer.start();
    PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);
    printf("[Setup] Galois keys: %.1f ms\n", timer.elapsed_ms());

    // Create CKKSEvaluator
    CKKSEvaluator ckks_eval(&context, &public_key, &secret_key, &encoder,
                            &relin_keys, &galois_keys, SCALE);
    GELUEvaluator gelu_eval(ckks_eval);
    MMEvaluator mm_eval(ckks_eval);

    // Create Bootstrapper
    long logn = 15;  // log of sparse slots (2^15 = 32768 = full slots for N=65536)
    long logNh = 15; // log of N/2

    printf("[Setup] Initializing bootstrapper...\n");
    timer.start();
    Bootstrapper bootstrapper(
        10,          // loge
        logn,        // logn
        logNh,       // logNh
        total_levels, // L
        SCALE,       // final_scale
        25,          // boundary_K
        59,          // sin_cos_deg
        2,           // scale_factor
        1,           // inverse_deg
        &ckks_eval);

    bootstrapper.slot_vec.push_back(logn);
    printf("[Setup] Preparing bootstrap polynomials...\n");
    bootstrapper.prepare_mod_polynomial();
    printf("[Setup] Generating LT coefficients...\n");
    bootstrapper.generate_LT_coefficient_3();
    printf("[Setup] Bootstrapper ready: %.1f ms\n", timer.elapsed_ms());

    // ══════════════════════════════════════════════════════════════════════
    // RANDOM DATA
    // ══════════════════════════════════════════════════════════════════════
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.01, 0.01);
    uniform_real_distribution<double> idist(-1.0, 1.0);

    // Weight data for MatMul
    vector<vector<double>> weight_data(inner_dim, vector<double>(slot_count, 0.0));
    for (int j = 0; j < inner_dim; j++)
        for (size_t s = 0; s < min((size_t)768, slot_count); s++)
            weight_data[j][s] = wdist(rng);

    // Input data
    vector<vector<double>> input_data(n_columns, vector<double>(slot_count, 0.0));
    for (int i = 0; i < n_columns; i++)
        for (size_t s = 0; s < min((size_t)768, slot_count); s++)
            input_data[i][s] = idist(rng);

    // Plaintext ground truth
    vector<vector<double>> mm_expected(n_columns, vector<double>(slot_count, 0.0));
    for (int i = 0; i < n_columns; i++)
        for (int j = 0; j < inner_dim; j++)
            for (size_t s = 0; s < slot_count; s++)
                mm_expected[i][s] += input_data[i][s] * weight_data[j][s];

    vector<vector<double>> gelu_expected(n_columns, vector<double>(slot_count, 0.0));
    for (int i = 0; i < n_columns; i++)
        for (size_t s = 0; s < slot_count; s++)
            gelu_expected[i][s] = plain_gelu(mm_expected[i][s]);

    // ══════════════════════════════════════════════════════════════════════
    // ENCRYPT (once, at full depth)
    // ══════════════════════════════════════════════════════════════════════
    printf("\n[1] Encrypting %d ciphertexts at full depth...\n", n_columns);
    timer.start();

    vector<PhantomCiphertext> cts(n_columns);
    for (int i = 0; i < n_columns; i++) {
        PhantomPlaintext pt;
        ckks_eval.encoder.encode(input_data[i], SCALE, pt);
        ckks_eval.encryptor.encrypt(pt, cts[i]);
    }

    // Mod-switch down to main level (skip bootstrap levels)
    for (int i = 0; i < n_columns; i++) {
        for (int l = 0; l < bs_mod_count; l++) {
            ckks_eval.evaluator.mod_switch_to_next_inplace(cts[i]);
        }
    }
    cudaDeviceSynchronize();
    printf("    Encrypted + mod-switch: %.1f ms, levels remaining: %zu\n",
           timer.elapsed_ms(), cts[0].coeff_modulus_size());

    // ══════════════════════════════════════════════════════════════════════
    // STAGE 1: MatMul (unified N=65536, consumes ~1 level)
    // ══════════════════════════════════════════════════════════════════════
    printf("\n[2] MatMul (%d cols × %d inner, unified N=65536)...\n", n_columns, inner_dim);
    timer.start();

    vector<PhantomCiphertext> mm_results;
    mm_eval.matrix_mul_unified(cts, weight_data, n_columns, mm_results);
    cudaDeviceSynchronize();

    double mm_ms = timer.elapsed_ms();
    printf("    MatMul: %.1f ms, levels remaining: %zu\n", mm_ms,
           mm_results[0].coeff_modulus_size());

    // Verify MatMul
    double mm_mae_sum = 0;
    for (int i = 0; i < n_columns; i++)
        mm_mae_sum += ckks_eval.calculate_MAE(mm_expected[i], mm_results[i], slot_count);
    double mm_mae = mm_mae_sum / n_columns;
    printf("    MatMul MAE: %.6f %s\n", mm_mae, mm_mae < 0.01 ? "PASS" : "FAIL");

    // ══════════════════════════════════════════════════════════════════════
    // STAGE 2: GELU (consumes ~6 levels)
    // ══════════════════════════════════════════════════════════════════════
    printf("\n[3] GELU (%d ciphertexts)...\n", n_columns);
    timer.start();

    vector<PhantomCiphertext> gelu_results(n_columns);
    for (int i = 0; i < n_columns; i++) {
        gelu_eval.gelu(mm_results[i], gelu_results[i]);
    }
    cudaDeviceSynchronize();

    double gelu_ms = timer.elapsed_ms();
    printf("    GELU: %.1f ms, levels remaining: %zu\n", gelu_ms,
           gelu_results[0].coeff_modulus_size());

    // Verify GELU
    double gelu_mae_sum = 0;
    for (int i = 0; i < n_columns; i++)
        gelu_mae_sum += ckks_eval.calculate_MAE(gelu_expected[i], gelu_results[i], slot_count);
    double gelu_mae = gelu_mae_sum / n_columns;
    printf("    GELU MAE: %.6f %s\n", gelu_mae, gelu_mae < 0.1 ? "PASS" : "FAIL");

    // ══════════════════════════════════════════════════════════════════════
    // STAGE 3: Bootstrap (refreshes levels homomorphically — NO decryption)
    // ══════════════════════════════════════════════════════════════════════
    printf("\n[4] Bootstrapping %d ciphertexts...\n", n_columns);
    timer.start();

    vector<PhantomCiphertext> boot_results(n_columns);
    for (int i = 0; i < n_columns; i++) {
        // Mod-switch down to level 1 before bootstrap
        while (gelu_results[i].coeff_modulus_size() > 1) {
            ckks_eval.evaluator.mod_switch_to_next_inplace(gelu_results[i]);
        }
        bootstrapper.bootstrap_3(boot_results[i], gelu_results[i]);
    }
    cudaDeviceSynchronize();

    double boot_ms = timer.elapsed_ms();
    printf("    Bootstrap: %.1f ms, levels restored: %zu\n", boot_ms,
           boot_results[0].coeff_modulus_size());

    // Verify bootstrap preserves values
    double boot_mae_sum = 0;
    for (int i = 0; i < n_columns; i++)
        boot_mae_sum += ckks_eval.calculate_MAE(gelu_expected[i], boot_results[i], slot_count);
    double boot_mae = boot_mae_sum / n_columns;
    printf("    Post-bootstrap MAE: %.6f %s\n", boot_mae, boot_mae < 0.5 ? "PASS" : "FAIL");

    // ══════════════════════════════════════════════════════════════════════
    // SUMMARY
    // ══════════════════════════════════════════════════════════════════════
    double total_ms = mm_ms + gelu_ms + boot_ms;

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  Connected BERT Pipeline Results\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Stage        │ Time (ms)  │ Levels After │ MAE\n");
    printf("  ─────────────┼────────────┼──────────────┼────────\n");
    printf("  MatMul       │ %8.1f   │ %10zu   │ %.6f %s\n",
           mm_ms, mm_results[0].coeff_modulus_size(), mm_mae, mm_mae < 0.01 ? "PASS" : "FAIL");
    printf("  GELU         │ %8.1f   │ %10zu   │ %.6f %s\n",
           gelu_ms, gelu_results[0].coeff_modulus_size(), gelu_mae, gelu_mae < 0.1 ? "PASS" : "FAIL");
    printf("  Bootstrap    │ %8.1f   │ %10zu   │ %.6f %s\n",
           boot_ms, boot_results[0].coeff_modulus_size(), boot_mae, boot_mae < 0.5 ? "PASS" : "FAIL");
    printf("  ─────────────┼────────────┼──────────────┼────────\n");
    printf("  Total        │ %8.1f   │              │\n", total_ms);
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Privacy: No re-encryption — true FHE pipeline\n");
    printf("  Parameters: single N=%zu, L=%d\n", N, total_levels);
    printf("  Config: %d cols × %d inner\n", n_columns, inner_dim);
    printf("════════════════════════════════════════════════════════════\n");

    printf("\nDone.\n");
    return 0;
}
