/**
 * bootstrap_test_n65536.cu
 *
 * Bootstrap test at N=65536 — single GPU with memory optimization.
 * Strategy: free relin key + public key before bootstrap (not needed),
 * then create all 48 Galois keys on 1 GPU.
 *
 * Memory budget:
 *   Context + SK: ~2.2 GB
 *   48 Galois keys: ~62.4 GB (1.3 GB each)
 *   Total: ~64.6 GB — exceeds H100 64 GB
 *   After freeing PK+RK (-1.6 GB): ~63.0 GB — fits!
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

static void print_gpu_memory(const char* label) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[Memory] %s: %.2f GB used / %.2f GB total (%.2f GB free)\n",
           label,
           (total_mem - free_mem) / (1024.0*1024.0*1024.0),
           total_mem / (1024.0*1024.0*1024.0),
           free_mem / (1024.0*1024.0*1024.0));
    fflush(stdout);
}

int main() {
    printf("================================================================\n");
    printf("  Bootstrap Test — N=65536, Single GPU, Memory Optimized\n");
    printf("================================================================\n\n");

    auto t0 = chrono::high_resolution_clock::now();
    cudaSetDevice(0);
    print_gpu_memory("Initial");

    // ═══ Parameters ═══
    long logN = 16;
    long logn = logN - 2;
    long sparse_slots_val = 1L << logn;
    long logNh = logN - 1;

    int logp = 46, logq = 51, log_special = 51;
    int main_mod = 21, bs_mod = 14;
    int total_level = main_mod + bs_mod;

    long boundary_K = 25, deg = 59, scale_factor = 2, inverse_deg = 1, loge = 10;

    vector<int> coeff_bits;
    coeff_bits.push_back(logq);
    for (int i = 0; i < main_mod; i++) coeff_bits.push_back(logp);
    for (int i = 0; i < bs_mod; i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(log_special);

    size_t N = 1ULL << logN;
    double scale = pow(2.0, logp);

    printf("[Setup] N=%zu, logn=%ld, sparse_slots=%ld\n", N, logn, sparse_slots_val);
    printf("[Setup] Levels: %d main + %d bootstrap = %d total\n", main_mod, bs_mod, total_level);
    fflush(stdout);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));
    parms.set_sparse_slots(sparse_slots_val);
    parms.set_secret_key_hamming_weight(192);

    // ═══ Create context and keys ═══
    PhantomContext context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;  // empty for now

    size_t slots = encoder.slot_count();
    print_gpu_memory("After context + PK + RK (no Galois)");

    CKKSEvaluator ckks_eval(&context, &public_key, &secret_key, &encoder,
                            &relin_keys, &galois_keys, scale);

    // ═══ Encrypt test data BEFORE freeing PK ═══
    size_t input_size = sparse_slots_val;
    mt19937 rng(42);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    vector<double> sparse_input(input_size);
    for (auto &v : sparse_input) v = dist(rng);
    vector<double> input(slots, 0.0);
    for (size_t i = 0; i < slots; i++) input[i] = sparse_input[i % input_size];

    PhantomPlaintext pt;
    PhantomCiphertext ct;
    encoder.encode(context, input, scale, pt);
    public_key.encrypt_asymmetric(context, pt, ct);
    for (int i = 0; i < bs_mod; i++)
        ckks_eval.evaluator.mod_switch_to_next_inplace(ct);

    printf("[Test] Encrypted: coeff_modulus_size=%zu\n", ct.coeff_modulus_size());

    // Verify pre-bootstrap
    {
        PhantomPlaintext tmp;
        secret_key.decrypt(context, ct, tmp);
        vector<double> dec;
        encoder.decode(context, tmp, dec);
        double mae = 0;
        size_t cmp = std::min(dec.size(), input_size);
        for (size_t i = 0; i < cmp; i++) mae += fabs(sparse_input[i] - dec[i]);
        mae /= cmp;
        printf("[Test] Pre-bootstrap MAE: %.9f\n", mae);
    }
    print_gpu_memory("After encryption");

    // ═══ FREE PK and RK to reclaim ~1.6 GB ═══
    printf("\n[Memory] Freeing PK and RK (not needed for bootstrap)...\n");
    { PhantomPublicKey empty_pk; public_key = std::move(empty_pk); }
    { PhantomRelinKey empty_rk; relin_keys = std::move(empty_rk); }
    print_gpu_memory("After freeing PK + RK");

    // ═══ Setup bootstrapper ═══
    printf("\n[Setup] Initializing bootstrapper...\n"); fflush(stdout);
    Bootstrapper bootstrapper(loge, logn, logNh, total_level, scale,
                              boundary_K, deg, scale_factor, inverse_deg, &ckks_eval);
    bootstrapper.slot_vec.push_back(logn);
    bootstrapper.prepare_mod_polynomial();
    bootstrapper.generate_LT_coefficient_3();
    print_gpu_memory("After LT coefficients");

    // Compute all needed rotation steps
    vector<int> all_steps;
    all_steps.push_back(0);
    for (int i = 0; i < logN - 1; i++) all_steps.push_back(1 << i);
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(all_steps);

    std::set<int> step_set(all_steps.begin(), all_steps.end());
    all_steps.assign(step_set.begin(), step_set.end());
    printf("[Setup] Total unique Galois keys needed: %zu\n", all_steps.size());
    printf("[Setup] Estimated key memory: %.1f GB\n", all_steps.size() * 1.3);
    fflush(stdout);

    // ═══ Create ALL Galois keys on single GPU ═══
    printf("[Setup] Creating all %zu Galois keys...\n", all_steps.size());
    fflush(stdout);
    auto t1 = chrono::high_resolution_clock::now();
    ckks_eval.decryptor.create_galois_keys_from_steps(all_steps, *ckks_eval.galois_keys);
    auto t2 = chrono::high_resolution_clock::now();
    printf("[Setup] Galois keys created in %.1f ms\n",
           chrono::duration<double, milli>(t2 - t1).count());
    print_gpu_memory("After ALL Galois keys");

    // ═══ Bootstrap ═══
    while (ct.coeff_modulus_size() > 1)
        ckks_eval.evaluator.mod_switch_to_next_inplace(ct);
    printf("\n[Test] At level 1, starting bootstrap...\n");
    fflush(stdout);

    t1 = chrono::high_resolution_clock::now();
    PhantomCiphertext ct_out;
    bootstrapper.bootstrap_3(ct_out, ct);
    cudaDeviceSynchronize();
    t2 = chrono::high_resolution_clock::now();
    double boot_ms = chrono::duration<double, milli>(t2 - t1).count();
    printf("[Test] Bootstrap time: %.1f ms\n", boot_ms);
    printf("[Test] Restored: coeff_modulus_size=%zu\n", ct_out.coeff_modulus_size());

    double post_mae = ckks_eval.calculate_MAE(input, ct_out, input_size);
    printf("[Test] Post-bootstrap MAE: %.9f %s\n", post_mae, post_mae < 0.01 ? "PASS" : "FAIL");
    print_gpu_memory("After bootstrap");

    // ═══ Summary ═══
    auto tend = chrono::high_resolution_clock::now();
    printf("\n════════════════════════════════════════════════\n");
    printf("  N=65536 Bootstrap — Single GPU Summary\n");
    printf("════════════════════════════════════════════════\n");
    printf("  N=%zu, sparse_slots=%ld\n", N, sparse_slots_val);
    printf("  Galois keys: %zu (freed PK+RK to fit)\n", all_steps.size());
    printf("  MAE: %.9f %s\n", post_mae, post_mae < 0.01 ? "PASS" : "FAIL");
    printf("  Bootstrap time: %.1f ms\n", boot_ms);
    printf("  Total: %.1f s\n", chrono::duration<double>(tend - t0).count());
    printf("════════════════════════════════════════════════\n");

    return (post_mae < 0.01) ? 0 : 1;
}
