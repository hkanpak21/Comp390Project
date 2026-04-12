/**
 * bootstrap_distkey_test.cu
 *
 * Tests multi-GPU key distribution at N=32768 where bootstrap is known to work.
 * If this PASSES: the problem is N=65536 bootstrap params, not key distribution.
 * If this FAILS: the problem is in our serialize/deserialize rotation path.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <set>
#include <sstream>
#include <thread>

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
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);

    printf("================================================================\n");
    printf("  Distributed-Key Bootstrap Test at N=32768\n");
    printf("  Purpose: isolate whether key distribution breaks accuracy\n");
    printf("  GPUs available: %d\n", dev_count);
    printf("================================================================\n\n");

    if (dev_count < 2) { fprintf(stderr, "Need 2 GPUs\n"); return 1; }

    // ═══ Known-good parameters (N=32768) ═══
    long logN = 15;
    long logn = logN - 2;              // 13
    long sparse_slots_val = 1L << logn; // 8192
    long logNh = logN - 1;             // 14

    int logp = 46, logq = 51, log_special = 51;
    int main_mod = 17, bs_mod = 14;
    int total_level = main_mod + bs_mod;
    double SCALE = pow(2.0, logp);

    vector<int> coeff_bits;
    coeff_bits.push_back(logq);
    for (int i = 0; i < main_mod; i++) coeff_bits.push_back(logp);
    for (int i = 0; i < bs_mod; i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(log_special);

    size_t N = 1ULL << logN;

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));
    parms.set_sparse_slots(sparse_slots_val);
    parms.set_secret_key_hamming_weight(192);

    long boundary_K = 25, deg = 59, scale_factor = 2, inverse_deg = 1, loge = 10;

    // ═══ CONTROL: Single-GPU bootstrap (no distribution) ═══
    printf("=== CONTROL: Single GPU bootstrap ===\n");
    cudaSetDevice(0);
    PhantomContext ctx0(parms);
    PhantomCKKSEncoder enc0(ctx0);
    PhantomSecretKey sk0(ctx0);
    PhantomPublicKey pk0 = sk0.gen_publickey(ctx0);
    PhantomRelinKey rk0 = sk0.gen_relinkey(ctx0);
    PhantomGaloisKey gk0;
    CKKSEvaluator eval_ctrl(&ctx0, &pk0, &sk0, &enc0, &rk0, &gk0, SCALE);

    Bootstrapper bs_ctrl(loge, logn, logNh, total_level, SCALE,
                         boundary_K, deg, scale_factor, inverse_deg, &eval_ctrl);
    bs_ctrl.slot_vec.push_back(logn);
    bs_ctrl.prepare_mod_polynomial();
    bs_ctrl.generate_LT_coefficient_3();

    vector<int> all_steps;
    all_steps.push_back(0);
    for (int i = 0; i < logN - 1; i++) all_steps.push_back(1 << i);
    bs_ctrl.addLeftRotKeys_Linear_to_vector_3(all_steps);
    eval_ctrl.decryptor.create_galois_keys_from_steps(all_steps, *eval_ctrl.galois_keys);

    // Encrypt
    stringstream sk_buf; sk0.save(sk_buf);
    size_t slots = enc0.slot_count();
    size_t input_size = sparse_slots_val;
    mt19937 rng(42);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    vector<double> sparse_input(input_size);
    for (auto &v : sparse_input) v = dist(rng);
    vector<double> input(slots, 0.0);
    for (size_t i = 0; i < slots; i++) input[i] = sparse_input[i % input_size];

    PhantomPlaintext pt; PhantomCiphertext ct;
    enc0.encode(ctx0, input, SCALE, pt);
    pk0.encrypt_asymmetric(ctx0, pt, ct);
    for (int i = 0; i < bs_mod; i++) eval_ctrl.evaluator.mod_switch_to_next_inplace(ct);
    while (ct.coeff_modulus_size() > 1) eval_ctrl.evaluator.mod_switch_to_next_inplace(ct);

    PhantomCiphertext ctrl_out;
    auto t1 = chrono::high_resolution_clock::now();
    bs_ctrl.bootstrap_3(ctrl_out, ct);
    cudaDeviceSynchronize();
    auto t2 = chrono::high_resolution_clock::now();
    double ctrl_mae = eval_ctrl.calculate_MAE(input, ctrl_out, input_size);
    printf("  Control MAE: %.9f (%s) in %.1f ms\n\n",
           ctrl_mae, ctrl_mae < 0.01 ? "PASS" : "FAIL",
           chrono::duration<double, milli>(t2 - t1).count());

    // ═══ TEST: Distributed-key bootstrap ═══
    printf("=== TEST: Distributed-key bootstrap (2 GPUs) ===\n");

    // New evaluator for distributed test
    PhantomGaloisKey gk_dist;
    CKKSEvaluator eval_dist(&ctx0, &pk0, &sk0, &enc0, &rk0, &gk_dist, SCALE);

    Bootstrapper bs_dist(loge, logn, logNh, total_level, SCALE,
                         boundary_K, deg, scale_factor, inverse_deg, &eval_dist);
    bs_dist.slot_vec.push_back(logn);
    bs_dist.prepare_mod_polynomial();
    bs_dist.generate_LT_coefficient_3();

    // Recompute all steps
    vector<int> all_steps2;
    all_steps2.push_back(0);
    for (int i = 0; i < logN - 1; i++) all_steps2.push_back(1 << i);
    bs_dist.addLeftRotKeys_Linear_to_vector_3(all_steps2);

    std::set<int> step_set(all_steps2.begin(), all_steps2.end());
    all_steps2.assign(step_set.begin(), step_set.end());
    printf("  Total steps: %zu\n", all_steps2.size());

    // Split: first half on GPU 0, second half on GPU 1
    size_t half = all_steps2.size() / 2;
    vector<int> gpu0_steps(all_steps2.begin(), all_steps2.begin() + half);
    vector<int> gpu1_steps(all_steps2.begin() + half, all_steps2.end());
    printf("  GPU 0: %zu keys, GPU 1: %zu keys\n", gpu0_steps.size(), gpu1_steps.size());

    // GPU 0 keys
    cudaSetDevice(0);
    eval_dist.decryptor.create_galois_keys_from_steps(gpu0_steps, *eval_dist.galois_keys);

    // GPU 1 setup via persistent worker
    eval_dist.evaluator.setup_remote_gpu_full(1, parms, sk_buf.str(), gpu1_steps);
    cudaSetDevice(0);

    printf("  Distributed keys ready\n");

    // Re-encrypt same input
    PhantomCiphertext ct2;
    enc0.encode(ctx0, input, SCALE, pt);
    pk0.encrypt_asymmetric(ctx0, pt, ct2);
    for (int i = 0; i < bs_mod; i++) eval_dist.evaluator.mod_switch_to_next_inplace(ct2);
    while (ct2.coeff_modulus_size() > 1) eval_dist.evaluator.mod_switch_to_next_inplace(ct2);

    PhantomCiphertext dist_out;
    t1 = chrono::high_resolution_clock::now();
    bs_dist.bootstrap_3(dist_out, ct2);
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    t2 = chrono::high_resolution_clock::now();
    double dist_mae = eval_dist.calculate_MAE(input, dist_out, input_size);
    printf("  Distributed MAE: %.9f (%s) in %.1f ms\n\n",
           dist_mae, dist_mae < 0.01 ? "PASS" : "FAIL",
           chrono::duration<double, milli>(t2 - t1).count());

    // ═══ Verdict ═══
    printf("════════════════════════════════════════════════\n");
    printf("  Verdict\n");
    printf("════════════════════════════════════════════════\n");
    printf("  Control (1 GPU):     MAE=%.9f %s\n", ctrl_mae, ctrl_mae < 0.01 ? "PASS" : "FAIL");
    printf("  Distributed (2 GPU): MAE=%.9f %s\n", dist_mae, dist_mae < 0.01 ? "PASS" : "FAIL");
    if (ctrl_mae < 0.01 && dist_mae > 0.01)
        printf("  >> KEY DISTRIBUTION BREAKS ACCURACY\n");
    else if (ctrl_mae < 0.01 && dist_mae < 0.01)
        printf("  >> KEY DISTRIBUTION WORKS — N=65536 params are the issue\n");
    else
        printf("  >> BOTH FAIL — deeper issue\n");
    printf("════════════════════════════════════════════════\n");

    eval_dist.evaluator.shutdown_remote_gpu();
    return (dist_mae < 0.01) ? 0 : 1;
}
