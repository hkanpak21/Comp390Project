/**
 * bootstrap_n65536_streaming.cu
 *
 * Bootstrap at N=65536 using CPU-side Galois key streaming.
 * All 48 Galois keys (~62 GB) stored in CPU pinned memory;
 * loaded one at a time to GPU during rotation (each ~1.3 GB).
 *
 * GPU memory: context (~2 GB) + 1 key (~1.3 GB) + ciphertexts (~2 GB) = ~5 GB
 * (vs 62.4 GB if all keys were on GPU — does not fit on H100 64 GB)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <set>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

#include "galois.cuh"
#include "ckks_evaluator.cuh"
#include "galois_key_store.cuh"
#include "bootstrapping/Bootstrapper.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

static void print_mem(const char *label) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[Mem] %s: %.2f GB used / %.2f GB total (%.2f GB free)\n",
           label,
           (total_mem - free_mem) / (1024.0*1024.0*1024.0),
           total_mem / (1024.0*1024.0*1024.0),
           free_mem / (1024.0*1024.0*1024.0));
    fflush(stdout);
}

// Use Phantom's own get_elts_from_steps for correctness

int main() {
    printf("================================================================\n");
    printf("  N=65536 Bootstrap with CPU-Side Key Streaming\n");
    printf("================================================================\n\n");

    auto t0 = chrono::high_resolution_clock::now();
    cudaSetDevice(0);
    print_mem("Initial");

    // ═══ Parameters (N=65536) ═══
    long logN = 16;
    long logn = logN - 2;
    long sparse_slots_val = 1L << logn;
    long logNh = logN - 1;
    int logp = 46, logq = 51, log_special = 51;
    int main_mod = 21, bs_mod = 14;
    int total_level = main_mod + bs_mod;
    long boundary_K = 25, deg = 59, scale_factor = 2, inverse_deg = 1, loge = 10;
    double scale = pow(2.0, logp);
    size_t N = 1ULL << logN;

    vector<int> coeff_bits;
    coeff_bits.push_back(logq);
    for (int i = 0; i < main_mod; i++) coeff_bits.push_back(logp);
    for (int i = 0; i < bs_mod; i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(log_special);

    printf("[Setup] N=%zu, logn=%ld, sparse_slots=%ld\n", N, logn, sparse_slots_val);
    printf("[Setup] Levels: %d main + %d bootstrap = %d total\n", main_mod, bs_mod, total_level);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));
    parms.set_sparse_slots(sparse_slots_val);
    parms.set_secret_key_hamming_weight(192);

    PhantomContext context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;  // empty slots, populated on demand
    size_t slots = encoder.slot_count();
    print_mem("After context + PK + RK");

    CKKSEvaluator ckks_eval(&context, &public_key, &secret_key, &encoder,
                            &relin_keys, &galois_keys, scale);

    // ═══ Encrypt test input BEFORE freeing PK ═══
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

    // ═══ Free PK only (RK is needed by mod_reducer for polynomial evaluation) ═══
    { PhantomPublicKey empty; public_key = std::move(empty); }
    print_mem("After freeing PK (keeping RK for bootstrap polynomial eval)");

    // ═══ Initialize bootstrapper (computes LT coeffs on CPU) ═══
    printf("[Setup] Initializing bootstrapper...\n"); fflush(stdout);
    auto t1 = chrono::high_resolution_clock::now();

    Bootstrapper bootstrapper(loge, logn, logNh, total_level, scale,
                              boundary_K, deg, scale_factor, inverse_deg, &ckks_eval);
    bootstrapper.slot_vec.push_back(logn);
    bootstrapper.prepare_mod_polynomial();
    bootstrapper.generate_LT_coefficient_3();

    // Collect all needed rotation steps
    vector<int> all_steps;
    all_steps.push_back(0);
    for (int i = 0; i < logN - 1; i++) all_steps.push_back(1 << i);
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(all_steps);

    std::set<int> step_set(all_steps.begin(), all_steps.end());
    all_steps.assign(step_set.begin(), step_set.end());
    printf("[Setup] Total unique rotation steps: %zu\n", all_steps.size());

    // Convert steps to Galois elements using Phantom's own function
    // (ensures consistency with what the rotation kernel expects)
    auto all_elts = ::get_elts_from_steps(all_steps, N);

    // Set up the Galois tool with all elements (no keys yet)
    context.setup_galois_tool(all_elts);

    // Allocate 48 empty slots in galois_keys
    galois_keys.resize_slots(all_elts.size());

    auto t2 = chrono::high_resolution_clock::now();
    printf("[Setup] Bootstrapper ready: %.1f ms\n",
           chrono::duration<double, milli>(t2 - t1).count());
    print_mem("After LT coefficients + tool setup");

    // ═══ Generate all keys to CPU (one at a time) ═══
    printf("\n[KeyStore] Generating all %zu keys to CPU pinned memory...\n", all_elts.size());
    fflush(stdout);
    t1 = chrono::high_resolution_clock::now();

    GaloisKeyStore key_store;
    key_store.generate_all_keys(context, secret_key, all_elts.size());

    t2 = chrono::high_resolution_clock::now();
    printf("[KeyStore] Key generation: %.1f s\n",
           chrono::duration<double>(t2 - t1).count());
    print_mem("After all keys on CPU (GPU should be mostly free)");

    // ═══ Enable key streaming in evaluator ═══
    ckks_eval.evaluator.enable_key_streaming(&key_store, &galois_keys);

    // ═══ Bootstrap ═══
    while (ct.coeff_modulus_size() > 1)
        ckks_eval.evaluator.mod_switch_to_next_inplace(ct);
    printf("\n[Test] At level 1, starting bootstrap with key streaming...\n");
    print_mem("Before bootstrap");
    fflush(stdout);

    t1 = chrono::high_resolution_clock::now();
    PhantomCiphertext ct_out;
    bootstrapper.bootstrap_3(ct_out, ct);
    cudaDeviceSynchronize();
    t2 = chrono::high_resolution_clock::now();
    double boot_ms = chrono::duration<double, milli>(t2 - t1).count();

    printf("[Test] Bootstrap time: %.1f ms\n", boot_ms);
    printf("[Test] Restored level: %zu\n", ct_out.coeff_modulus_size());
    print_mem("After bootstrap");

    // Need to recreate secret key decryption (sk is still valid)
    PhantomPlaintext out_pt;
    secret_key.decrypt(context, ct_out, out_pt);
    vector<double> out_vals;
    encoder.decode(context, out_pt, out_vals);
    double mae = 0;
    size_t cmp = std::min(out_vals.size(), input_size);
    for (size_t i = 0; i < cmp; i++) mae += fabs(input[i] - out_vals[i]);
    mae /= cmp;

    printf("[Test] Post-bootstrap MAE: %.9f %s\n", mae, mae < 0.01 ? "PASS" : "FAIL");

    auto tend = chrono::high_resolution_clock::now();
    printf("\n════════════════════════════════════════════════\n");
    printf("  N=65536 Bootstrap — CPU Key Streaming Summary\n");
    printf("════════════════════════════════════════════════\n");
    printf("  N=%zu, sparse_slots=%ld\n", N, sparse_slots_val);
    printf("  Keys: %zu on CPU, 1 on GPU at a time\n", all_elts.size());
    printf("  MAE: %.9f %s\n", mae, mae < 0.01 ? "PASS" : "FAIL");
    printf("  Bootstrap time: %.1f ms\n", boot_ms);
    printf("  Total: %.1f s\n", chrono::duration<double>(tend - t0).count());
    printf("════════════════════════════════════════════════\n");

    return (mae < 0.01) ? 0 : 1;
}
