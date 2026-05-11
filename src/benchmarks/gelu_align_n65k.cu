/**
 * gelu_align_n65k.cu
 *
 * Lane ALIGN-SINGLE — NEXUS GELU alignment microbench (single-GPU).
 *
 * Mimics NEXUS's gelu_test (vendor/nexus/cuda/src/main.cu lines 287-316,
 * COEFF_MODULI[4]) at NEXUS's chosen GELU parameter set:
 *
 *   logN = 16          → poly_degree = 65,536, slot_count = 32,768
 *   COEFF_MODULI[4]    = {58, 40×17, 58} (19 moduli total)
 *   scale              = 2^40
 *   input length       = 32,768 slots (full slot range, NEXUS gelu_input_32768.txt)
 *
 * Calls GELUEvaluator::gelu() in a loop, isolating per-call wall-clock with
 * cudaDeviceSynchronize on either side. Reports median of 100 calls (after
 * a warmup call).
 *
 * Single-GPU only (the goal is "OUR code on single H100", to compare apples-
 * to-apples against NEXUS-on-H100 = 69 ms from JOBID 40367787).
 *
 * Expected: ~69 ms ± 5%, since OUR gelu.cu is byte-identical to NEXUS's
 * vendored gelu.cu (only `params_id() → chain_index()` API differences).
 *
 * CLI:
 *   gelu_align_n65k                  (default --calls 100)
 *   gelu_align_n65k --calls 50
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
#include "gelu.cuh"
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
            printf("Usage: gelu_align_n65k [--calls N]\n");
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
    printf("  ALIGN-GELU (single-GPU) — NEXUS GELU at logN=16\n");
    printf("  poly_degree=65,536, slot_count=32,768\n");
    printf("  Coeff moduli: {58, 18×40, 58} (NEXUS COEFF_MODULI[4], 20 moduli)\n");
    printf("  Calls: %d (median + σ reported)\n", cfg.calls);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    // ═══ NEXUS GELU CKKS params (matches main.cu line 290 + COEFF_MODULI[4]) ═══
    size_t N = 1ULL << 16;
    double SCALE = pow(2.0, 40);

    // From main.cu line 37 (COEFF_MODULI[4], the GELU row, 20 entries):
    //   {58, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 58}
    // (NEXUS uses 18 forties between two 58s — counted by `tr ',' '\n' | wc -l`)
    vector<int> coeff_bits;
    coeff_bits.push_back(58);
    for (int i = 0; i < 18; i++) coeff_bits.push_back(40);
    coeff_bits.push_back(58);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));

    printf("[Setup] N=%zu, %zu moduli (%d bits total)\n",
           N, coeff_bits.size(),
           [&]() { int s = 0; for (int b : coeff_bits) s += b; return s; }());
    fflush(stdout);

    // ═══ Build context + keys ═══
    auto t_setup_begin = chrono::high_resolution_clock::now();
    PhantomContext     context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey   secret_key(context);
    PhantomPublicKey   public_key = secret_key.gen_publickey(context);
    PhantomRelinKey    relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey   galois_keys = secret_key.create_galois_keys(context);

    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder,
                                 &relin_keys, &galois_keys, SCALE);
    GELUEvaluator gelu_evaluator(ckks_evaluator);

    auto t_setup_end = chrono::high_resolution_clock::now();
    double setup_ms = chrono::duration<double, milli>(t_setup_end - t_setup_begin).count();
    printf("[Setup] context+keys+evaluator ready in %.0f ms\n", setup_ms);
    fflush(stdout);

    // ═══ Encode + encrypt one input ciphertext (reused across calls) ═══
    size_t slot_count = encoder.slot_count();
    printf("[Setup] encoder slot_count = %zu\n", slot_count);

    // NEXUS reads gelu_input_32768.txt (32768 floats in [-4, 4]). We synthesize
    // matching-shape random data with a fixed seed; GELU compute cost is
    // independent of input contents (it's a fixed-degree polynomial eval).
    mt19937 rng(13371337);
    uniform_real_distribution<double> dist(-4.0, 4.0);
    vector<double> input(slot_count, 0.0);
    for (size_t i = 0; i < slot_count; i++) input[i] = dist(rng);

    PhantomPlaintext plain_input;
    PhantomCiphertext base_cipher;
    ckks_evaluator.encoder.encode(input, SCALE, plain_input);
    ckks_evaluator.encryptor.encrypt(plain_input, base_cipher);

    printf("[Setup] base ciphertext: coeff_modulus_size=%zu, scale=%.2e\n",
           base_cipher.coeff_modulus_size(), base_cipher.scale());
    fflush(stdout);

    // ═══ Warmup call ═══
    printf("\n[Warmup] running 1 GELU to warm GPU caches/kernels...\n");
    fflush(stdout);
    {
        PhantomCiphertext warmup_in = base_cipher;
        PhantomCiphertext rtn;
        cudaDeviceSynchronize();
        auto t0 = chrono::high_resolution_clock::now();
        gelu_evaluator.gelu(warmup_in, rtn);
        cudaDeviceSynchronize();
        auto t1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(t1 - t0).count();
        printf("[Warmup] gelu=%.1f ms\n", ms);
        fflush(stdout);
    }

    // ═══ Measurement loop: N calls of isolated gelu ═══
    printf("\n[Measure] running %d isolated GELU calls...\n", cfg.calls);
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
            NVTX_SCOPE("gelu_isolated");
            gelu_evaluator.gelu(input_ct, rtn);
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
    printf("  ALIGN-GELU HEADLINE (single-GPU H100, logN=16, slots=32,768)\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Calls measured:            %d\n", cfg.calls);
    printf("  Wall-clock (all calls):    %.1f s\n", wall_total_s);
    printf("  Per-call median:           %.2f ms\n", med_ms);
    printf("  Per-call σ:                %.2f ms (%.2f%% of median)\n",
           sigma_ms, 100.0 * sigma_ms / med_ms);
    printf("  Per-call min / max:        %.2f / %.2f ms\n", min_ms, max_ms);
    printf("  Per-call p10 / p90:        %.2f / %.2f ms\n", p10_ms, p90_ms);
    printf("  ----\n");
    printf("  NEXUS published (A100):    3,350 ms\n");
    printf("  NEXUS measured H100:       69 ms (JOBID 40367787)\n");
    printf("  multiNEXUS (THIS RUN):     %.2f ms (median)\n", med_ms);
    printf("  ratio THIS / NEXUS-H100:   %.3fx (≈1.00x means same kernel/workload)\n",
           med_ms / 69.0);
    printf("  ratio THIS / NEXUS-A100:   %.3fx (lower = faster than NEXUS A100)\n",
           med_ms / 3350.0);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    return 0;
}
