/**
 * layernorm_align_n65k.cu
 *
 * Lane ALIGN-SINGLE — NEXUS LayerNorm alignment microbench (single-GPU).
 *
 * Mimics NEXUS's layernorm_test (vendor/nexus/cuda/src/main.cu lines 318-347,
 * COEFF_MODULI[3]) at NEXUS's chosen LayerNorm parameter set:
 *
 *   logN = 16          → poly_degree = 65,536, slot_count = 32,768
 *   COEFF_MODULI[3]    = {58, 40×18, 58} (20 moduli total)
 *   scale              = 2^40
 *   layer_norm len     = 1024 (NEXUS arg in main.cu line 342)
 *   input              = layernorm_input_16_768.txt (16×768 = 12,288 values)
 *
 * Calls LNEvaluator::layer_norm(x, res, 1024) in a loop, isolating per-call
 * wall-clock with cudaDeviceSynchronize on either side. Reports median of
 * 100 calls (after a warmup call).
 *
 * Single-GPU only. Expected: ~45 ms ± 5%, since OUR layer_norm.cu is
 * byte-identical to NEXUS's vendored layer_norm.cu (only `params_id() →
 * chain_index()` API differences).
 *
 * CLI:
 *   layernorm_align_n65k                  (default --calls 100)
 *   layernorm_align_n65k --calls 50
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
#include "layer_norm.cuh"
#include "util/nvtx_tracer.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

namespace {

struct Config {
    int calls = 100;
    int len = 1024;   // NEXUS uses 1024 in main.cu line 342
};

Config parse_cli(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--calls" && i + 1 < argc) c.calls = atoi(argv[++i]);
        else if (a == "--len"   && i + 1 < argc) c.len   = atoi(argv[++i]);
        else if (a == "--help"  || a == "-h") {
            printf("Usage: layernorm_align_n65k [--calls N] [--len L]\n");
            exit(0);
        }
    }
    if (c.calls < 1) c.calls = 1;
    if (c.len   < 1) c.len   = 1024;
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
    printf("  ALIGN-LayerNorm (single-GPU) — NEXUS LayerNorm at logN=16\n");
    printf("  poly_degree=65,536, slot_count=32,768, len=%d\n", cfg.len);
    printf("  Coeff moduli: {58, 18×40, 58} (NEXUS COEFF_MODULI[3])\n");
    printf("  Calls: %d (median + σ reported)\n", cfg.calls);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    // ═══ NEXUS LayerNorm CKKS params (matches main.cu COEFF_MODULI[3]) ═══
    size_t N = 1ULL << 16;
    double SCALE = pow(2.0, 40);

    // From main.cu line 36 (COEFF_MODULI[3], 20 entries):
    //   {58, 40×18, 58}
    vector<int> coeff_bits;
    coeff_bits.push_back(58);
    for (int i = 0; i < 18; i++) coeff_bits.push_back(40);
    coeff_bits.push_back(58);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));

    printf("[Setup] N=%zu, %zu moduli\n", N, coeff_bits.size());
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
    LNEvaluator ln_evaluator(ckks_evaluator);

    auto t_setup_end = chrono::high_resolution_clock::now();
    double setup_ms = chrono::duration<double, milli>(t_setup_end - t_setup_begin).count();
    printf("[Setup] context+keys+evaluator ready in %.0f ms\n", setup_ms);
    fflush(stdout);

    // ═══ Encode + encrypt one input ciphertext (reused across calls) ═══
    size_t slot_count = encoder.slot_count();
    printf("[Setup] encoder slot_count = %zu\n", slot_count);

    // NEXUS reads layernorm_input_16_768.txt = 16×768 = 12,288 doubles in [-1, 1].
    // We synthesize matching-shape random data (16 rows × 768 cols of [-1, 1])
    // padded to slot_count with zeros. layer_norm compute cost is independent of
    // input contents (it's a fixed-degree polynomial eval + Newton iter).
    mt19937 rng(98798765);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    vector<double> input(slot_count, 0.0);
    for (int i = 0; i < 16 * 768 && (size_t)i < slot_count; i++) input[i] = dist(rng);

    PhantomPlaintext plain_input;
    PhantomCiphertext base_cipher;
    ckks_evaluator.encoder.encode(input, SCALE, plain_input);
    ckks_evaluator.encryptor.encrypt(plain_input, base_cipher);

    printf("[Setup] base ciphertext: coeff_modulus_size=%zu, scale=%.2e\n",
           base_cipher.coeff_modulus_size(), base_cipher.scale());
    fflush(stdout);

    // ═══ Warmup call ═══
    printf("\n[Warmup] running 1 LayerNorm to warm GPU caches/kernels...\n");
    fflush(stdout);
    {
        PhantomCiphertext warmup_in = base_cipher;
        PhantomCiphertext rtn;
        cudaDeviceSynchronize();
        auto t0 = chrono::high_resolution_clock::now();
        ln_evaluator.layer_norm(warmup_in, rtn, cfg.len);
        cudaDeviceSynchronize();
        auto t1 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, milli>(t1 - t0).count();
        printf("[Warmup] layer_norm=%.1f ms\n", ms);
        fflush(stdout);
        // FIX-BUG-01-01 (DECODE-VALIDITY GATE): decode the warmup output and
        // refuse to enter the measurement loop on NaN/Inf or out-of-range
        // values. LayerNorm output should be approximately mean-0, unit-variance
        // per row; |value| ≤ ~5 covers the 5-sigma tail. SANITY_BOUND = 10.0
        // catches catastrophic failures (modulus exhaustion in the Newton sqrt
        // iteration, scale drift) without flagging legitimate output noise.
        {
            PhantomCiphertext check_in = base_cipher;
            PhantomCiphertext check_out;
            ln_evaluator.layer_norm(check_in, check_out, cfg.len);
            PhantomPlaintext pt;
            secret_key.decrypt(context, check_out, pt);
            vector<double> dec;
            encoder.decode(context, pt, dec);
            const double SANITY_BOUND = 10.0;
            size_t cmp = std::min(dec.size(), (size_t)(16 * 768));
            size_t bad = 0;
            double dec_abs_max = 0.0;
            for (size_t i = 0; i < cmp; i++) {
                if (!std::isfinite(dec[i])) { bad++; continue; }
                double a = std::fabs(dec[i]);
                if (a > dec_abs_max) dec_abs_max = a;
            }
            printf("[Warmup] decode-validity: |max|=%.3f over %zu slots, "
                   "non-finite=%zu (bound |x|<%.0f)\n",
                   dec_abs_max, cmp, bad, SANITY_BOUND);
            fflush(stdout);
            if (bad > 0 || dec_abs_max > SANITY_BOUND) {
                fprintf(stderr,
                        "[FATAL] FIX-BUG-01-01: warmup LayerNorm output failed "
                        "decode-validity gate (non-finite=%zu, |max|=%.3e)\n",
                        bad, dec_abs_max);
                fflush(stderr);
                return 2;
            }
        }
    }

    // ═══ Measurement loop: N calls of isolated layer_norm ═══
    printf("\n[Measure] running %d isolated LayerNorm calls...\n", cfg.calls);
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
            NVTX_SCOPE("layer_norm_isolated");
            ln_evaluator.layer_norm(input_ct, rtn, cfg.len);
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
    printf("  ALIGN-LayerNorm HEADLINE (single-GPU H100, logN=16, len=%d)\n", cfg.len);
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Calls measured:            %d\n", cfg.calls);
    printf("  Wall-clock (all calls):    %.1f s\n", wall_total_s);
    printf("  Per-call median:           %.2f ms\n", med_ms);
    printf("  Per-call σ:                %.2f ms (%.2f%% of median)\n",
           sigma_ms, 100.0 * sigma_ms / med_ms);
    printf("  Per-call min / max:        %.2f / %.2f ms\n", min_ms, max_ms);
    printf("  Per-call p10 / p90:        %.2f / %.2f ms\n", p10_ms, p90_ms);
    printf("  ----\n");
    printf("  NEXUS published (A100):    1,010 ms\n");
    printf("  NEXUS measured H100:       45 ms (JOBID 40367787)\n");
    printf("  multiNEXUS (THIS RUN):     %.2f ms (median)\n", med_ms);
    printf("  ratio THIS / NEXUS-H100:   %.3fx (≈1.00x means same kernel/workload)\n",
           med_ms / 45.0);
    printf("  ratio THIS / NEXUS-A100:   %.3fx (lower = faster than NEXUS A100)\n",
           med_ms / 1010.0);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    return 0;
}
