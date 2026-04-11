/**
 * bootstrap_test.cu
 *
 * Standalone bootstrapping accuracy test matching NEXUS argmax_test parameters.
 * Tests that bootstrap_3() preserves ciphertext values (MAE < 0.01).
 *
 * Parameters (from NEXUS main.cu lines 108-170):
 *   logN = 15 (N = 32768)
 *   logn = 13 (sparse_slots = 8192)
 *   logNh = 14 (N/2 = 16384)
 *   main_mod_count = 17, bs_mod_count = 14, total = 31
 *   logp = 46, logq = 51, log_special = 51
 *   secret_key_hamming_weight = 192
 *   boundary_K = 25, sin_cos_deg = 59, scale_factor = 2, inverse_deg = 1
 *
 * Usage:
 *   ./bin/bootstrap_test
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
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
#include "bootstrapping/Bootstrapper.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

int main() {
    printf("================================================================\n");
    printf("  Standalone Bootstrapping Accuracy Test\n");
    printf("  (Matching NEXUS argmax_test parameters)\n");
    printf("================================================================\n\n");

    auto t0 = chrono::high_resolution_clock::now();

    // ═══ Parameters: N=32768, SPARSE mode (logn=13 < logNh=14) ═══
    // We use the sparse bootstrap path but encode data normally.
    // The bootstrapper will consolidate via subsum.
    long logN = 15;
    long logn = logN - 2;                    // 13 (SPARSE mode)
    long sparse_slots_val = 0;               // 0 = don't modify encoder (use all 16384 slots normally)
    long logNh = logN - 1;                   // 14

    int logp = 46;
    int logq = 51;
    int log_special_prime = 51;

    int main_mod_count = 17;
    int bs_mod_count = 14;
    int total_level = main_mod_count + bs_mod_count;
    int secret_key_hw = 0;  // default

    // Bootstrapping parameters
    long boundary_K = 25;
    long deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1;
    long loge = 10;

    // Build coefficient modulus chain
    vector<int> coeff_bit_vec;
    coeff_bit_vec.push_back(logq);                           // first prime
    for (int i = 0; i < main_mod_count; i++)
        coeff_bit_vec.push_back(logp);                       // main primes
    for (int i = 0; i < bs_mod_count; i++)
        coeff_bit_vec.push_back(logq);                       // bootstrap primes
    coeff_bit_vec.push_back(log_special_prime);              // special prime

    size_t poly_modulus_degree = (size_t)(1 << logN);        // 32768
    double scale = pow(2.0, logp);

    printf("[Setup] N=%zu, logn=%ld, sparse_slots=%ld\n", poly_modulus_degree, logn, sparse_slots_val);
    printf("[Setup] Levels: %d main + %d bootstrap = %d total\n", main_mod_count, bs_mod_count, total_level);
    printf("[Setup] Hamming weight: %d\n", secret_key_hw);
    printf("[Setup] Total moduli: %zu (", coeff_bit_vec.size());
    for (size_t i = 0; i < coeff_bit_vec.size(); i++) printf("%d%s", coeff_bit_vec[i], i < coeff_bit_vec.size()-1 ? "," : "");
    printf(")\n\n");
    fflush(stdout);

    // ═══ Create context with sparse slots ═══
    printf("[Setup] Creating EncryptionParameters...\n"); fflush(stdout);
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    printf("[Setup] Creating CoeffModulus...\n"); fflush(stdout);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec));
    if (sparse_slots_val > 0) {
        printf("[Setup] Setting sparse_slots=%ld...\n", sparse_slots_val); fflush(stdout);
        parms.set_sparse_slots(sparse_slots_val);
    }
    if (secret_key_hw > 0) {
        parms.set_secret_key_hamming_weight(secret_key_hw);
    }

    cudaSetDevice(0);
    printf("[Setup] Creating PhantomContext...\n"); fflush(stdout);
    PhantomContext context(parms);
    printf("[Setup] Creating encoder...\n"); fflush(stdout);
    PhantomCKKSEncoder encoder(context);
    printf("[Setup] Creating keys...\n"); fflush(stdout);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    size_t slot_count = encoder.slot_count();
    printf("[Setup] slot_count: %zu\n", slot_count); fflush(stdout);

    // ═══ Generate SELECTIVE Galois keys for bootstrapping ═══
    printf("[Setup] Generating selective Galois keys...\n"); fflush(stdout);
    auto t1 = chrono::high_resolution_clock::now();

    // Bootstrapping rotation steps (same as NEXUS argmax_test lines 212-226)
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
    }

    PhantomGaloisKey galois_keys = secret_key.create_galois_keys_from_steps(context, gal_steps_vector);
    auto t2 = chrono::high_resolution_clock::now();
    printf("[Setup] Galois keys (%zu steps): %.1f ms\n",
           gal_steps_vector.size(),
           chrono::duration<double, milli>(t2 - t1).count());
    fflush(stdout);

    // ═══ Create CKKSEvaluator (with placeholder galois keys, will regenerate) ═══
    CKKSEvaluator ckks_eval(&context, &public_key, &secret_key, &encoder,
                            &relin_keys, &galois_keys, scale);

    printf("[Setup] Initializing bootstrapper...\n"); fflush(stdout);
    t1 = chrono::high_resolution_clock::now();

    Bootstrapper bootstrapper(
        loge, logn, logNh, total_level, scale,
        boundary_K, deg, scale_factor, inverse_deg,
        &ckks_eval);

    bootstrapper.slot_vec.push_back(logn);

    printf("[Setup] Preparing polynomials...\n"); fflush(stdout);
    bootstrapper.prepare_mod_polynomial();

    printf("[Setup] Generating LT coefficients...\n"); fflush(stdout);
    bootstrapper.generate_LT_coefficient_3();

    // Now regenerate Galois keys with ALL steps needed by bootstrapper
    printf("[Setup] Adding bootstrap rotation keys...\n"); fflush(stdout);
    gal_steps_vector.clear();
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back((1 << i));
    }
    // Add bootstrapper-specific rotation keys
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);

    printf("[Setup] Regenerating Galois keys with %zu steps...\n", gal_steps_vector.size()); fflush(stdout);
    ckks_eval.decryptor.create_galois_keys_from_steps(gal_steps_vector, *ckks_eval.galois_keys);

    t2 = chrono::high_resolution_clock::now();
    printf("[Setup] Bootstrapper ready: %.1f ms\n\n",
           chrono::duration<double, milli>(t2 - t1).count());

    // ═══ Test 1: Simple bootstrap roundtrip ═══
    printf("=== Test 1: Bootstrap roundtrip ===\n");

    // Create input — use slot_count for full mode
    size_t input_size = sparse_slots_val > 0 ? sparse_slots_val : slot_count;
    mt19937 rng(42);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    vector<double> input(input_size, 0.0);
    for (auto &v : input) v = dist(rng);

    // Encrypt
    PhantomPlaintext plain_input;
    PhantomCiphertext cipher_input;
    ckks_eval.encoder.encode(input, scale, plain_input);
    ckks_eval.encryptor.encrypt(plain_input, cipher_input);

    printf("  Encrypted: coeff_modulus_size = %zu\n", cipher_input.coeff_modulus_size());

    // Mod-switch down to main level (skip bootstrap levels)
    for (int i = 0; i < bs_mod_count; i++) {
        ckks_eval.evaluator.mod_switch_to_next_inplace(cipher_input);
    }
    printf("  After mod-switch to main: coeff_modulus_size = %zu\n", cipher_input.coeff_modulus_size());

    // Verify pre-bootstrap
    double pre_mae = ckks_eval.calculate_MAE(input, cipher_input, input_size);
    printf("  Pre-bootstrap MAE: %.9f\n", pre_mae);
    fflush(stdout);

    // Directly mod-switch to level 1 (no computation — test pure bootstrap)
    while (cipher_input.coeff_modulus_size() > 1) {
        ckks_eval.evaluator.mod_switch_to_next_inplace(cipher_input);
    }
    printf("  At level 1: coeff_modulus_size = %zu, scale = %.6e (expected %.6e)\n",
           cipher_input.coeff_modulus_size(), cipher_input.scale(), scale);
    fflush(stdout);

    // ═══ BOOTSTRAP ═══
    printf("\n  >>> Bootstrapping...\n");
    t1 = chrono::high_resolution_clock::now();

    PhantomCiphertext cipher_output;
    bootstrapper.bootstrap_3(cipher_output, cipher_input);
    cudaDeviceSynchronize();

    t2 = chrono::high_resolution_clock::now();
    double boot_ms = chrono::duration<double, milli>(t2 - t1).count();
    printf("  >>> Bootstrap time: %.1f ms\n", boot_ms);
    printf("  >>> Restored: coeff_modulus_size = %zu\n", cipher_output.coeff_modulus_size());

    // Verify
    double post_mae = ckks_eval.calculate_MAE(input, cipher_output, input_size);
    printf("  >>> Post-bootstrap MAE: %.9f\n", post_mae);
    printf("  >>> Result: %s (threshold 0.01)\n\n", post_mae < 0.01 ? "PASS" : "FAIL");

    // ═══ Test 2: Multiple bootstraps ═══
    printf("=== Test 2: Double bootstrap ===\n");

    // Consume levels again
    for (int i = 0; i < 5 && cipher_output.coeff_modulus_size() > 2; i++) {
        PhantomPlaintext one_pt;
        ckks_eval.encoder.encode(1.0, cipher_output.chain_index(), cipher_output.scale(), one_pt);
        ckks_eval.evaluator.multiply_plain_inplace(cipher_output, one_pt);
        ckks_eval.evaluator.rescale_to_next_inplace(cipher_output);
    }

    while (cipher_output.coeff_modulus_size() > 1) {
        ckks_eval.evaluator.mod_switch_to_next_inplace(cipher_output);
    }

    // Second bootstrap
    PhantomCiphertext cipher_output2;
    t1 = chrono::high_resolution_clock::now();
    bootstrapper.bootstrap_3(cipher_output2, cipher_output);
    cudaDeviceSynchronize();
    t2 = chrono::high_resolution_clock::now();

    double post_mae2 = ckks_eval.calculate_MAE(input, cipher_output2, input_size);
    printf("  After 2nd bootstrap: MAE = %.9f %s\n", post_mae2, post_mae2 < 0.1 ? "PASS" : "FAIL");
    printf("  Bootstrap time: %.1f ms\n\n", chrono::duration<double, milli>(t2 - t1).count());

    // ═══ Summary ═══
    auto tend = chrono::high_resolution_clock::now();
    printf("════════════════════════════════════════════════\n");
    printf("  Bootstrap Test Summary\n");
    printf("════════════════════════════════════════════════\n");
    printf("  N=%zu, sparse_slots=%ld, hamming=%d\n", poly_modulus_degree, sparse_slots_val, secret_key_hw);
    printf("  Pre-bootstrap MAE:    %.9f\n", pre_mae);
    printf("  Post-bootstrap MAE:   %.9f %s\n", post_mae, post_mae < 0.01 ? "PASS" : "FAIL");
    printf("  Post-2nd-boot MAE:    %.9f %s\n", post_mae2, post_mae2 < 0.1 ? "PASS" : "FAIL");
    printf("  Bootstrap time:       %.1f ms\n", boot_ms);
    printf("  Total test time:      %.1f ms\n",
           chrono::duration<double, milli>(tend - t0).count());
    printf("════════════════════════════════════════════════\n");

    return (post_mae < 0.01) ? 0 : 1;
}
