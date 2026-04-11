/**
 * bert_ops_test.cu
 *
 * Tests each NEXUS BERT operation (GELU, Softmax, LayerNorm) with random data.
 * Verifies FHE results match plaintext computation via MAE.
 *
 * Usage:
 *   ./bin/bert_ops_test [--op gelu|softmax|layernorm|all]
 */

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

#include "ckks_evaluator.cuh"
#include "gelu.cuh"
#include "softmax.cuh"
#include "layer_norm.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

// ── Plaintext reference implementations ──────────────────────────────────────

double plain_gelu(double x) {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

vector<double> plain_softmax(const vector<double> &input, int len) {
    vector<double> result(input.size(), 0.0);
    for (int start = 0; start < (int)input.size(); start += len) {
        double max_val = *max_element(input.begin() + start, input.begin() + start + len);
        double sum = 0.0;
        for (int i = 0; i < len && start + i < (int)input.size(); i++) {
            result[start + i] = exp(input[start + i] - max_val);
            sum += result[start + i];
        }
        for (int i = 0; i < len && start + i < (int)input.size(); i++) {
            result[start + i] /= sum;
        }
    }
    return result;
}

vector<double> plain_layernorm(const vector<double> &input, int len) {
    // Simplified LayerNorm matching NEXUS implementation:
    // 1. fold: x += rotate(x, -len)
    // 2. compute variance: sum(x^2) / 768
    // 3. multiply by 1/sqrt(variance) * x
    vector<double> result(input.size(), 0.0);
    for (int start = 0; start < (int)input.size(); start += len) {
        double sum_sq = 0.0;
        for (int i = 0; i < len && start + i < (int)input.size(); i++) {
            // Fold step: x[i] += x[i + len] (but we compute on the original for simplicity)
            double val = input[start + i];
            sum_sq += val * val;
        }
        double variance = sum_sq / 768.0;
        double inv_sqrt_var = 1.0 / sqrt(variance);
        for (int i = 0; i < len && start + i < (int)input.size(); i++) {
            result[start + i] = input[start + i] * inv_sqrt_var;
        }
    }
    return result;
}

// ── Test functions ───────────────────────────────────────────────────────────

bool test_gelu() {
    printf("\n=== GELU Test ===\n");

    // Parameters: N=65536, L=20 (matching NEXUS)
    size_t N = 1ULL << 16;
    double SCALE = pow(2.0, 40);
    vector<int> coeff_bits = {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58};

    printf("Setting up: N=%zu, L=%zu...\n", N, coeff_bits.size());

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));

    PhantomContext context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;  // empty — GELU doesn't need rotations

    size_t slot_count = encoder.slot_count();
    printf("Slots: %zu\n", slot_count);

    CKKSEvaluator ckks_eval(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, SCALE);
    GELUEvaluator gelu_eval(ckks_eval);

    // Generate random input in [-3.5, 3.5]
    mt19937 rng(42);
    uniform_real_distribution<double> dist(-3.5, 3.5);
    vector<double> input(slot_count);
    for (auto &v : input) v = dist(rng);

    // Plaintext ground truth
    vector<double> expected(slot_count);
    for (size_t i = 0; i < slot_count; i++) {
        expected[i] = plain_gelu(input[i]);
    }

    // FHE computation
    printf("Encrypting...\n");
    PhantomPlaintext plain_input;
    PhantomCiphertext cipher_input, cipher_output;
    ckks_eval.encoder.encode(input, SCALE, plain_input);
    ckks_eval.encryptor.encrypt(plain_input, cipher_input);

    printf("Running GELU (FHE)...\n");
    auto t0 = chrono::high_resolution_clock::now();
    gelu_eval.gelu(cipher_input, cipher_output);
    cudaDeviceSynchronize();
    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();
    printf("GELU time: %.1f ms\n", ms);
    printf("Moduli remaining: %zu\n", cipher_output.coeff_modulus_size());

    // Decrypt and compare
    double mae = ckks_eval.calculate_MAE(expected, cipher_output, slot_count);
    printf("MAE: %.6f\n", mae);

    bool pass = mae < 0.05;
    printf("GELU: %s (MAE %.6f, threshold 0.05)\n", pass ? "PASS" : "FAIL", mae);
    return pass;
}

bool test_softmax() {
    printf("\n=== Softmax Test ===\n");

    // Parameters: N=65536, L=18
    size_t N = 1ULL << 16;
    double SCALE = pow(2.0, 40);
    vector<int> coeff_bits = {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58};

    printf("Setting up: N=%zu, L=%zu...\n", N, coeff_bits.size());

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));

    PhantomContext context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    printf("Generating Galois keys (all)...\n");
    PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

    size_t slot_count = encoder.slot_count();
    printf("Slots: %zu\n", slot_count);

    CKKSEvaluator ckks_eval(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, SCALE);
    SoftmaxEvaluator softmax_eval(ckks_eval);

    int softmax_len = 128;  // softmax over 128 positions

    // Generate random input in [0, 1]
    mt19937 rng(42);
    uniform_real_distribution<double> dist(0.0, 1.0);
    vector<double> input(slot_count, 0.0);
    for (int i = 0; i < softmax_len; i++) input[i] = dist(rng);

    // Plaintext ground truth
    vector<double> expected = plain_softmax(input, softmax_len);

    // FHE computation
    printf("Encrypting...\n");
    PhantomPlaintext plain_input;
    PhantomCiphertext cipher_input, cipher_output;
    ckks_eval.encoder.encode(input, SCALE, plain_input);
    ckks_eval.encryptor.encrypt(plain_input, cipher_input);

    printf("Running Softmax (FHE, len=%d)...\n", softmax_len);
    auto t0 = chrono::high_resolution_clock::now();
    softmax_eval.softmax(cipher_input, cipher_output, softmax_len);
    cudaDeviceSynchronize();
    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();
    printf("Softmax time: %.1f ms\n", ms);
    printf("Moduli remaining: %zu\n", cipher_output.coeff_modulus_size());

    // Decrypt and compare
    double mae = ckks_eval.calculate_MAE(expected, cipher_output, softmax_len);
    printf("MAE (first %d slots): %.6f\n", softmax_len, mae);

    bool pass = mae < 0.1;
    printf("Softmax: %s (MAE %.6f, threshold 0.1)\n", pass ? "PASS" : "FAIL", mae);
    return pass;
}

bool test_layernorm() {
    printf("\n=== LayerNorm Test ===\n");

    // Parameters: N=65536, L=20
    size_t N = 1ULL << 16;
    double SCALE = pow(2.0, 40);
    vector<int> coeff_bits = {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58};

    printf("Setting up: N=%zu, L=%zu...\n", N, coeff_bits.size());

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));

    PhantomContext context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);

    printf("Generating Galois keys (all)...\n");
    PhantomGaloisKey galois_keys = secret_key.create_galois_keys(context);

    size_t slot_count = encoder.slot_count();
    printf("Slots: %zu\n", slot_count);

    CKKSEvaluator ckks_eval(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, SCALE);
    LNEvaluator ln_eval(ckks_eval);

    int ln_len = 1024;  // LayerNorm dimension

    // Generate random input
    mt19937 rng(42);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    vector<double> input(slot_count, 0.0);
    for (int i = 0; i < ln_len; i++) input[i] = dist(rng);

    // Plaintext ground truth
    vector<double> expected = plain_layernorm(input, ln_len);

    // FHE computation
    printf("Encrypting...\n");
    PhantomPlaintext plain_input;
    PhantomCiphertext cipher_input, cipher_output;
    ckks_eval.encoder.encode(input, SCALE, plain_input);
    ckks_eval.encryptor.encrypt(plain_input, cipher_input);

    printf("Running LayerNorm (FHE, len=%d)...\n", ln_len);
    auto t0 = chrono::high_resolution_clock::now();
    ln_eval.layer_norm(cipher_input, cipher_output, ln_len);
    cudaDeviceSynchronize();
    auto t1 = chrono::high_resolution_clock::now();
    double ms = chrono::duration<double, milli>(t1 - t0).count();
    printf("LayerNorm time: %.1f ms\n", ms);
    printf("Moduli remaining: %zu\n", cipher_output.coeff_modulus_size());

    // Decrypt and compare
    double mae = ckks_eval.calculate_MAE(expected, cipher_output, ln_len);
    printf("MAE (first %d slots): %.6f\n", ln_len, mae);

    bool pass = mae < 0.05;
    printf("LayerNorm: %s (MAE %.6f, threshold 0.05)\n", pass ? "PASS" : "FAIL", mae);
    return pass;
}

int main(int argc, char **argv) {
    string op = "all";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--op") && i + 1 < argc) op = argv[++i];
    }

    printf("=== NEXUS BERT Operations Correctness Test ===\n");
    printf("Operation: %s\n", op.c_str());

    int pass = 0, fail = 0;

    if (op == "gelu" || op == "all") {
        if (test_gelu()) pass++; else fail++;
    }
    if (op == "softmax" || op == "all") {
        if (test_softmax()) pass++; else fail++;
    }
    if (op == "layernorm" || op == "all") {
        if (test_layernorm()) pass++; else fail++;
    }

    printf("\n=== Summary ===\n");
    printf("Passed: %d, Failed: %d\n", pass, fail);

    return fail > 0 ? 1 : 0;
}
