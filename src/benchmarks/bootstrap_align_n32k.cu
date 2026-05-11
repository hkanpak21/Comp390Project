/**
 * bootstrap_align_n32k.cu
 *
 * Lane ALIGN-SINGLE — NEXUS Bootstrap alignment microbench (single-GPU).
 *
 * Mimics NEXUS's standalone bootstrap test
 * (vendor/nexus/cuda/src/bootstrapping.cu) exactly:
 *
 *   logN = 15        → poly_degree = 32,768
 *   logn = 13        → sparse_slots = 8,192
 *   logp = 46, logq = 51, log_special_prime = 51
 *   secret_key_hamming_weight = 192
 *   remaining_level = 16, boot_level = 14, total = 30
 *   boundary_K = 25, deg = 59, scale_factor = 2, inverse_deg = 1, loge = 10
 *
 * Calls bootstrapper.bootstrap_3() in a loop, isolating per-call wall-clock
 * with cudaDeviceSynchronize on either side. Reports median of 100 calls
 * (after a warmup call).
 *
 * Single-GPU only (the goal is "OUR code on single H100", to compare apples-
 * to-apples against NEXUS-on-H100 = 252.8 ms from JOBID 40367787).
 *
 * Expected: ~250 ms ± 5%, since OUR Bootstrapper.cu is byte-identical to
 * NEXUS's vendored Bootstrapper.cu (only `params_id() → chain_index()`
 * Phantom API differences).
 *
 * CLI:
 *   bootstrap_align_n32k                  (default --calls 100)
 *   bootstrap_align_n32k --calls 50
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

#include "ckks_evaluator.cuh"
#include "bootstrapping/Bootstrapper.cuh"
#include "util/nvtx_tracer.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

namespace {

struct Config {
    int calls = 100;
};

Config parse_cli(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--calls" && i + 1 < argc) c.calls = atoi(argv[++i]);
        else if (a == "--help" || a == "-h") {
            printf("Usage: bootstrap_align_n32k [--calls N]\n");
            exit(0);
        }
    }
    if (c.calls < 1) c.calls = 1;
    return c;
}

double median(vector<double> v) {
    if (v.empty()) return 0.0;
    sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1) ? v[n/2] : 0.5 * (v[n/2-1] + v[n/2]);
}

double stdev(const vector<double> &v) {
    if (v.size() < 2) return 0.0;
    double m = 0.0;
    for (double x : v) m += x;
    m /= v.size();
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return sqrt(s / (v.size() - 1));
}

double percentile(vector<double> v, double pct) {
    if (v.empty()) return 0.0;
    sort(v.begin(), v.end());
    size_t idx = (size_t)((pct / 100.0) * (v.size() - 1));
    return v[idx];
}

}  // namespace

int main(int argc, char **argv) {
    Config cfg = parse_cli(argc, argv);

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < 1) {
        fprintf(stderr, "[FATAL] no CUDA devices visible — abort\n");
        return 1;
    }
    cudaSetDevice(0);

    printf("════════════════════════════════════════════════════════════\n");
    printf("  ALIGN-Bootstrap (single-GPU) — NEXUS bootstrap at logN=15\n");
    printf("  poly_degree=32,768, sparse_slots=8,192, hamming=192\n");
    printf("  Coeff moduli: {51, 16×46, 14×51, 51} (NEXUS bootstrapping.cu)\n");
    printf("  Bootstrap params: K=25, deg=59, scale_factor=2, loge=10\n");
    printf("  Calls: %d (median + σ reported)\n", cfg.calls);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    // ═══ NEXUS bootstrap params (matches vendor/nexus/cuda/src/bootstrapping.cu) ═══
    long boundary_K   = 25;
    long deg          = 59;
    long scale_factor = 2;
    long inverse_deg  = 1;
    long logN         = 15;
    long loge         = 10;
    long logn         = 13;
    long sparse_slots = (1L << logn);    // 8,192

    int logp = 46, logq = 51, log_special_prime = 51;
    int secret_key_hw = 192;

    int remaining_level = 16;
    int boot_level      = 14;
    int total_level     = remaining_level + boot_level;

    // Build coeff modulus chain matching NEXUS exactly
    vector<int> coeff_bits;
    coeff_bits.push_back(logq);
    for (int i = 0; i < remaining_level; i++) coeff_bits.push_back(logp);
    for (int i = 0; i < boot_level;      i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(log_special_prime);

    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bits));
    parms.set_secret_key_hamming_weight(secret_key_hw);
    parms.set_sparse_slots(sparse_slots);

    printf("[Setup] N=%zu, sparse_slots=%ld, total_level=%d (%d main + %d boot)\n",
           poly_modulus_degree, sparse_slots, total_level, remaining_level, boot_level);
    fflush(stdout);

    // ═══ Build CKKS context + keys ═══
    auto t_setup_begin = chrono::high_resolution_clock::now();
    PhantomContext     context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey   secret_key(context);
    PhantomPublicKey   public_key = secret_key.gen_publickey(context);
    PhantomRelinKey    relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey   galois_keys;

    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder,
                                 &relin_keys, &galois_keys, scale);

    Bootstrapper bootstrapper(
        loge, logn, logN - 1, total_level, scale,
        boundary_K, deg, scale_factor, inverse_deg, &ckks_evaluator);

    printf("[Setup] Generating mod polynomials...\n"); fflush(stdout);
    bootstrapper.prepare_mod_polynomial();

    printf("[Setup] Adding bootstrapping galois keys...\n"); fflush(stdout);
    vector<int> gal_steps;
    gal_steps.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps.push_back(1 << i);
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps);

    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps,
                                                           *(ckks_evaluator.galois_keys));
    bootstrapper.slot_vec.push_back(logn);

    printf("[Setup] Generating LT coefficients...\n"); fflush(stdout);
    bootstrapper.generate_LT_coefficient_3();

    auto t_setup_end = chrono::high_resolution_clock::now();
    double setup_ms = chrono::duration<double, milli>(t_setup_end - t_setup_begin).count();
    printf("[Setup] context+keys+bootstrapper ready in %.0f ms\n", setup_ms);
    fflush(stdout);

    // ═══ Encode + encrypt one input ciphertext (reused across calls) ═══
    size_t slot_count = encoder.slot_count();
    printf("[Setup] encoder slot_count = %zu\n", slot_count);

    mt19937 rng(424242);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    vector<double> sparse(sparse_slots, 0.0);
    for (auto &v : sparse) v = dist(rng);
    vector<double> input(slot_count, 0.0);
    for (size_t i = 0; i < slot_count; i++) input[i] = sparse[i % sparse_slots];

    PhantomPlaintext plain;
    PhantomCiphertext base_cipher;
    encoder.encode(context, input, scale, plain);
    public_key.encrypt_asymmetric(context, plain, base_cipher);

    // Mod-switch all the way down to lowest level (NEXUS does this before bootstrap)
    for (int i = 0; i < total_level; i++) {
        ckks_evaluator.evaluator.mod_switch_to_next_inplace(base_cipher);
    }
    printf("[Setup] base ciphertext at chain bottom: coeff_modulus_size=%zu\n",
           base_cipher.coeff_modulus_size());

    // Sanity: pre-bootstrap MAE
    {
        PhantomPlaintext tmp;
        secret_key.decrypt(context, base_cipher, tmp);
        vector<double> dec;
        encoder.decode(context, tmp, dec);
        double mae = 0.0;
        size_t cmp = std::min(dec.size(), (size_t)sparse_slots);
        for (size_t i = 0; i < cmp; i++) mae += fabs(sparse[i] - dec[i]);
        mae /= cmp;
        printf("[Setup] pre-bootstrap MAE = %.6e (expected near machine precision)\n", mae);
        fflush(stdout);
    }

    // ═══ Warmup call ═══
    printf("\n[Warmup] running 1 bootstrap to warm GPU caches/kernels...\n");
    fflush(stdout);
    {
        PhantomCiphertext warmup_in = base_cipher;
        PhantomCiphertext rtn;
        cudaDeviceSynchronize();
        auto t0 = chrono::high_resolution_clock::now();
        bootstrapper.bootstrap_3(rtn, warmup_in);
        cudaDeviceSynchronize();
        auto t1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(t1 - t0).count();

        // Sanity: post-bootstrap MAE on the warmup result
        PhantomPlaintext tmp;
        secret_key.decrypt(context, rtn, tmp);
        vector<double> dec;
        encoder.decode(context, tmp, dec);
        double mae = 0.0;
        size_t cmp = std::min(dec.size(), (size_t)sparse_slots);
        for (size_t i = 0; i < cmp; i++) mae += fabs(sparse[i] - dec[i]);
        mae /= cmp;
        printf("[Warmup] bootstrap=%.1f ms, MAE=%.6e (threshold 0.01: %s)\n",
               ms, mae, mae < 0.01 ? "PASS" : "FAIL");
        fflush(stdout);
    }

    // ═══ Measurement loop: N calls of isolated bootstrap_3 ═══
    printf("\n[Measure] running %d isolated bootstrap calls...\n", cfg.calls);
    fflush(stdout);

    vector<double> per_call_ms;
    per_call_ms.reserve(cfg.calls);

    auto wall_t0 = chrono::high_resolution_clock::now();
    for (int k = 0; k < cfg.calls; k++) {
        // Fresh input each call: copy base_cipher (already at chain bottom)
        // bootstrap_3 takes the input by reference; we want a fresh ciphertext
        // each call so that we measure the bootstrap itself, not the decay of
        // a re-bootstrapped ciphertext.
        PhantomCiphertext input_ct = base_cipher;
        PhantomCiphertext rtn;

        cudaDeviceSynchronize();
        auto t0 = chrono::high_resolution_clock::now();
        {
            NVTX_SCOPE("bootstrap_3_isolated");
            bootstrapper.bootstrap_3(rtn, input_ct);
        }
        cudaDeviceSynchronize();
        auto t1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(t1 - t0).count();
        per_call_ms.push_back(ms);

        if (k < 5 || (k % 10 == 0) || k == cfg.calls - 1) {
            printf("  call %3d/%d: %.2f ms\n", k + 1, cfg.calls, ms);
            fflush(stdout);
        }
    }
    auto wall_t1 = chrono::high_resolution_clock::now();
    double wall_total_s = chrono::duration<double>(wall_t1 - wall_t0).count();

    // ═══ Stats ═══
    double med_ms   = median(per_call_ms);
    double sigma_ms = stdev(per_call_ms);
    double min_ms   = *min_element(per_call_ms.begin(), per_call_ms.end());
    double max_ms   = *max_element(per_call_ms.begin(), per_call_ms.end());
    double p10_ms   = percentile(per_call_ms, 10);
    double p90_ms   = percentile(per_call_ms, 90);

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  ALIGN-Bootstrap HEADLINE (single-GPU H100, logN=15)\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Calls measured:            %d\n", cfg.calls);
    printf("  Wall-clock (all calls):    %.1f s\n", wall_total_s);
    printf("  Per-call median:           %.2f ms\n", med_ms);
    printf("  Per-call σ:                %.2f ms (%.2f%% of median)\n",
           sigma_ms, 100.0 * sigma_ms / med_ms);
    printf("  Per-call min / max:        %.2f / %.2f ms\n", min_ms, max_ms);
    printf("  Per-call p10 / p90:        %.2f / %.2f ms\n", p10_ms, p90_ms);
    printf("  ----\n");
    printf("  NEXUS published (A100):    5,630 ms\n");
    printf("  NEXUS measured H100:       252.8 ms (JOBID 40367787, sparse 2^13)\n");
    printf("  multiNEXUS (THIS RUN):     %.2f ms (median)\n", med_ms);
    printf("  ratio THIS / NEXUS-H100:   %.3fx (≈1.00x means same kernel/workload)\n",
           med_ms / 252.8);
    printf("  ratio THIS / NEXUS-A100:   %.3fx (lower = faster than NEXUS A100)\n",
           med_ms / 5630.0);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    return 0;
}
