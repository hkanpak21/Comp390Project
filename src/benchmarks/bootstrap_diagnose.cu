/**
 * bootstrap_diagnose.cu
 *
 * Lane BOOT-DIAGNOSE — diagnostic microbench that decomposes the
 * "in-pipeline 1,032 ms vs NEXUS-standalone 252.8 ms" 4× gap on H100.
 *
 * Hypotheses being tested:
 *   H1: Workload mismatch  (different sparse_slots / full slots)
 *   H2: Chain depth diff   (HP-BERT main_mod=21 vs NEXUS remaining_level=16)
 *   H3: Pipeline overhead  (key fetch / NCCL interleave; not testable here)
 *   H4: Implementation diff (rejected by bootstrap_align_n32k @ 250.64 ms,
 *                            JOBID 40368887, identical to NEXUS 252.8 ms)
 *
 * What this binary does:
 *   - Builds the NEXUS Bootstrapper (src/nexus_eval/bootstrapping/Bootstrapper.cu,
 *     byte-identical to vendor/nexus/cuda/src/bootstrapping/Bootstrapper.cu).
 *   - Configurable via CLI: --logN, --logn, --main-mod, --bs-mod, --calls.
 *   - Runs N isolated bootstrap_3 calls (warmup + measure), median + σ.
 *   - Reports vs the two reference points (NEXUS-standalone, HP-BERT in-pipeline).
 *
 * Default config = NEXUS standalone (matches bootstrap_align_n32k):
 *   logN=15, logn=13 (sparse_slots=8192), main_mod=16, bs_mod=14, total=30.
 *
 * HP-BERT config (use --main-mod 21):
 *   logN=15, logn=13, main_mod=21, bs_mod=14, total=35.
 *
 * Single-GPU only (we want the bootstrap kernel cost in isolation).
 *
 * CLI:
 *   bootstrap_diagnose                                              (NEXUS workload)
 *   bootstrap_diagnose --main-mod 21                                (HP-BERT workload)
 *   bootstrap_diagnose --logN 15 --logn 13 --main-mod 21 --bs-mod 14 --calls 100
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
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
    long logN     = 15;     // ring degree exponent
    long logn     = 13;     // sparse-slots exponent (0 = full slots = logN-1)
    int  main_mod = 16;     // remaining-level moduli (NEXUS=16, HP-BERT=21)
    int  bs_mod   = 14;     // bootstrap moduli (both NEXUS and HP-BERT use 14)
    int  calls    = 100;    // measurement count
    bool full_slots = false; // override: encode at full slots (logn = logN-1)
};

void print_usage() {
    printf("Usage: bootstrap_diagnose [OPTIONS]\n");
    printf("Options:\n");
    printf("  --logN N         Ring degree exponent (default 15, N=32768)\n");
    printf("  --logn N         Sparse-slots exponent (default 13, slots=8192)\n");
    printf("                   Pass --logn -1 (or --full-slots) for full slots = logN-1\n");
    printf("  --full-slots     Use full slots (logn = logN-1)\n");
    printf("  --main-mod N     Remaining-level moduli (default 16; HP-BERT uses 21)\n");
    printf("  --bs-mod N       Bootstrap moduli (default 14)\n");
    printf("  --calls N        Number of isolated bootstrap calls (default 100)\n");
    printf("  --help, -h       Show this help\n");
    printf("\nReference points:\n");
    printf("  NEXUS standalone   logN=15 logn=13 main=16 bs=14 → 252.8 ms (H100)\n");
    printf("  multiNEXUS HP-BERT logN=15 logn=13 main=21 bs=14 → 1,032 ms in-pipeline\n");
}

Config parse_cli(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if      (a == "--logN"        && i + 1 < argc) c.logN     = atol(argv[++i]);
        else if (a == "--logn"        && i + 1 < argc) c.logn     = atol(argv[++i]);
        else if (a == "--main-mod"    && i + 1 < argc) c.main_mod = atoi(argv[++i]);
        else if (a == "--bs-mod"      && i + 1 < argc) c.bs_mod   = atoi(argv[++i]);
        else if (a == "--calls"       && i + 1 < argc) c.calls    = atoi(argv[++i]);
        else if (a == "--full-slots") c.full_slots = true;
        else if (a == "--help" || a == "-h") {
            print_usage();
            exit(0);
        } else {
            fprintf(stderr, "[FATAL] Unknown arg: %s\n", a.c_str());
            print_usage();
            exit(2);
        }
    }
    if (c.full_slots || c.logn < 0) c.logn = c.logN - 1;
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

    long sparse_slots = 1L << cfg.logn;
    long logNh        = cfg.logN - 1;
    int  total_level  = cfg.main_mod + cfg.bs_mod;
    int  logp         = 46;
    int  logq         = 51;
    int  log_special  = 51;
    int  hamming      = 192;
    long boundary_K   = 25;
    long deg          = 59;
    long scale_factor = 2;
    long inverse_deg  = 1;
    long loge         = 10;

    printf("════════════════════════════════════════════════════════════\n");
    printf("  BOOT-DIAGNOSE — single-GPU bootstrap workload sweep\n");
    printf("  CONFIG: logN=%ld, logn=%ld (sparse_slots=%ld), main_mod=%d, bs_mod=%d, total=%d\n",
           cfg.logN, cfg.logn, sparse_slots, cfg.main_mod, cfg.bs_mod, total_level);
    printf("  Coeff moduli: {%d, %d×%d, %d×%d, %d}\n",
           logq, cfg.main_mod, logp, cfg.bs_mod, logq, log_special);
    printf("  Bootstrap params: K=%ld, deg=%ld, scale_factor=%ld, loge=%ld, hamming=%d\n",
           boundary_K, deg, scale_factor, loge, hamming);
    printf("  Calls: %d (median + σ reported)\n", cfg.calls);
    // Tag the run for easy identification
    if (cfg.logN == 15 && cfg.logn == 13 && cfg.main_mod == 16 && cfg.bs_mod == 14)
        printf("  TAG: NEXUS-STANDALONE-WORKLOAD (target 252.8 ms)\n");
    else if (cfg.logN == 15 && cfg.logn == 13 && cfg.main_mod == 21 && cfg.bs_mod == 14)
        printf("  TAG: HP-BERT-WORKLOAD (in-pipeline target 1,032 ms)\n");
    else
        printf("  TAG: CUSTOM\n");
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    // Build coeff modulus chain
    vector<int> coeff_bits;
    coeff_bits.push_back(logq);
    for (int i = 0; i < cfg.main_mod; i++) coeff_bits.push_back(logp);
    for (int i = 0; i < cfg.bs_mod;   i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(log_special);

    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << cfg.logN);
    double scale = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bits));
    parms.set_secret_key_hamming_weight(hamming);
    parms.set_sparse_slots(sparse_slots);

    printf("[Setup] N=%zu, sparse_slots=%ld, total_level=%d (%d main + %d boot)\n",
           poly_modulus_degree, sparse_slots, total_level, cfg.main_mod, cfg.bs_mod);
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
        loge, cfg.logn, logNh, total_level, scale,
        boundary_K, deg, scale_factor, inverse_deg, &ckks_evaluator);

    printf("[Setup] Generating mod polynomials...\n"); fflush(stdout);
    bootstrapper.prepare_mod_polynomial();

    printf("[Setup] Adding bootstrapping galois keys...\n"); fflush(stdout);
    vector<int> gal_steps;
    gal_steps.push_back(0);
    for (int i = 0; i < cfg.logN - 1; i++) {
        gal_steps.push_back(1 << i);
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps);

    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps,
                                                           *(ckks_evaluator.galois_keys));
    bootstrapper.slot_vec.push_back(cfg.logn);

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

    // ═══ Measurement loop ═══
    printf("\n[Measure] running %d isolated bootstrap calls...\n", cfg.calls);
    fflush(stdout);

    vector<double> per_call_ms;
    per_call_ms.reserve(cfg.calls);

    auto wall_t0 = chrono::high_resolution_clock::now();
    for (int k = 0; k < cfg.calls; k++) {
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
    printf("  BOOT-DIAGNOSE HEADLINE (single-GPU H100)\n");
    printf("  Config: logN=%ld, logn=%ld, main_mod=%d, bs_mod=%d, total=%d\n",
           cfg.logN, cfg.logn, cfg.main_mod, cfg.bs_mod, total_level);
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Calls measured:            %d\n", cfg.calls);
    printf("  Wall-clock (all calls):    %.1f s\n", wall_total_s);
    printf("  Per-call median:           %.2f ms\n", med_ms);
    printf("  Per-call σ:                %.2f ms (%.2f%% of median)\n",
           sigma_ms, 100.0 * sigma_ms / med_ms);
    printf("  Per-call min / max:        %.2f / %.2f ms\n", min_ms, max_ms);
    printf("  Per-call p10 / p90:        %.2f / %.2f ms\n", p10_ms, p90_ms);
    printf("  ----\n");
    printf("  REFERENCE POINTS:\n");
    printf("    NEXUS published (A100):    5,630 ms (logN=15, sparse=2^13, main=16, bs=14)\n");
    printf("    NEXUS measured H100:         252.8 ms (JOBID 40367787, sparse=2^13)\n");
    printf("    multiNEXUS standalone H100:  250.6 ms (JOBID 40368887, NEXUS workload)\n");
    printf("    multiNEXUS HP-BERT in-pipe:1,032 ms (logN=15, main=21, bs=14, in-pipeline)\n");
    printf("  ----\n");
    printf("  THIS RUN (median):         %.2f ms\n", med_ms);
    printf("  Ratio THIS / NEXUS-H100:   %.3fx\n", med_ms / 252.8);
    printf("  Ratio THIS / HP-BERT-pipe: %.3fx\n", med_ms / 1032.0);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    return 0;
}
