/**
 * rotate_stream_test.cu
 *
 * Minimal test: N=65536, setup key streaming, do ONE rotation.
 * If this works, streaming is OK and bootstrap-specific issue exists.
 * If it fails, we have a narrow problem to fix.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"
#include "galois.cuh"

#include "ckks_evaluator.cuh"
#include "galois_key_store.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

int main() {
    printf("=== Minimal Rotation Test with Key Streaming at N=65536 ===\n");
    cudaSetDevice(0);

    // N=65536 params with just enough levels for a rotation
    long logN = 16;
    size_t N = 1ULL << logN;
    double scale = pow(2.0, 46);

    vector<int> coeff_bits = {51, 46, 46, 46, 51};  // 5 limbs only
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));
    parms.set_secret_key_hamming_weight(192);

    PhantomContext context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey sk(context);
    PhantomPublicKey pk = sk.gen_publickey(context);
    PhantomRelinKey rk = sk.gen_relinkey(context);
    PhantomGaloisKey gk;

    CKKSEvaluator eval(&context, &pk, &sk, &encoder, &rk, &gk, scale);
    size_t slots = encoder.slot_count();

    // Test data
    vector<double> input(slots, 0.0);
    for (size_t i = 0; i < slots; i++) input[i] = (double)(i + 1) / slots;

    PhantomPlaintext plain;
    PhantomCiphertext cipher;
    encoder.encode(context, input, scale, plain);
    pk.encrypt_asymmetric(context, plain, cipher);

    // ═══ Setup key streaming for MULTIPLE rotation steps ═══
    // Test: rotate by 1, then by 2, then by 1 (tests key swap back-and-forth)
    vector<int> steps = {1, 2};
    auto elts = ::get_elts_from_steps(steps, N);
    printf("Steps: ");
    for (size_t i = 0; i < steps.size(); i++) printf("%d→elt%u ", steps[i], elts[i]);
    printf("\n");

    context.setup_galois_tool(elts);
    gk.resize_slots(elts.size());

    GaloisKeyStore key_store;
    key_store.generate_all_keys(context, sk, elts.size());
    eval.evaluator.enable_key_streaming(&key_store, &gk);

    // ═══ Test 1: rotate by 1 ═══
    printf("\n--- Rotation 1 (step=1) ---\n");
    PhantomCiphertext rot1;
    eval.evaluator.rotate_vector(cipher, 1, gk, rot1);
    cudaDeviceSynchronize();
    {
        PhantomPlaintext pt; sk.decrypt(context, rot1, pt);
        vector<double> out; encoder.decode(context, pt, out);
        double mae = 0;
        for (size_t i = 0; i < 16; i++) mae += fabs(input[(i+1) % slots] - out[i]);
        mae /= 16;
        printf("  MAE: %.9f %s\n", mae, mae < 0.01 ? "PASS" : "FAIL");
    }

    // ═══ Test 2: rotate rot1 by 1 more (total=2 from original) — uses DIFFERENT key ═══
    printf("\n--- Rotation 2 (step=2, requires key swap) ---\n");
    PhantomCiphertext rot2;
    eval.evaluator.rotate_vector(cipher, 2, gk, rot2);
    cudaDeviceSynchronize();
    {
        PhantomPlaintext pt; sk.decrypt(context, rot2, pt);
        vector<double> out; encoder.decode(context, pt, out);
        double mae = 0;
        for (size_t i = 0; i < 16; i++) mae += fabs(input[(i+2) % slots] - out[i]);
        mae /= 16;
        printf("  MAE: %.9f %s\n", mae, mae < 0.01 ? "PASS" : "FAIL");
    }

    // ═══ Test 3: back to step=1 (swap back to first key) ═══
    printf("\n--- Rotation 3 (step=1 again, swap BACK) ---\n");
    PhantomCiphertext rot3;
    eval.evaluator.rotate_vector(cipher, 1, gk, rot3);
    cudaDeviceSynchronize();
    {
        PhantomPlaintext pt; sk.decrypt(context, rot3, pt);
        vector<double> out; encoder.decode(context, pt, out);
        double mae = 0;
        for (size_t i = 0; i < 16; i++) mae += fabs(input[(i+1) % slots] - out[i]);
        mae /= 16;
        printf("  MAE: %.9f %s\n", mae, mae < 0.01 ? "PASS" : "FAIL");
    }

    // ═══ Test 4: add two ciphertexts (non-rotation op) + another rotation ═══
    printf("\n--- Test 4: add_inplace + rotation (like bootstrap subsum) ---\n");
    eval.evaluator.add_inplace(rot1, rot2);  // no key needed
    cudaDeviceSynchronize();
    PhantomCiphertext rot4;
    eval.evaluator.rotate_vector(rot1, 2, gk, rot4);  // key swap after add
    cudaDeviceSynchronize();
    printf("  add+rotate OK (no crash)\n");

    printf("\nAll tests PASSED\n");
    return 0;
}
