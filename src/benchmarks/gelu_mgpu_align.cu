/**
 * gelu_mgpu_align.cu
 *
 * Lane MGPU-NEXUS-MICRO — multi-GPU NEXUS GELU micro-benchmark.
 *
 * Mimics NEXUS's gelu_test (vendor/nexus/cuda/src/main.cu, COEFF_MODULI[4])
 * at NEXUS's chosen GELU parameter set:
 *
 *   logN = 16          → poly_degree = 65,536, slot_count = 32,768
 *   COEFF_MODULI[4]    = {58, 40×17, 58} (19 moduli total)
 *   scale              = 2^40
 *   input length       = 32,768 slots (full slot range)
 *
 * Pattern (mirrors phantom_threadsafe_smoke.cu):
 *   - Each GPU gets its own std::thread.
 *   - Each thread sets cudaSetDevice(g), constructs its own PhantomContext,
 *     builds its own GELUEvaluator.
 *   - Each thread runs its share of the calls (calls / n_gpus, rounded up).
 *
 * CLI:
 *   gelu_mgpu_align                         (default --calls 100 --n-gpus 4)
 *   gelu_mgpu_align --calls 100 --n-gpus 4
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
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
    int n_gpus = 4;
};

Config parse_cli(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--calls"   && i + 1 < argc) c.calls  = atoi(argv[++i]);
        else if (a == "--n-gpus" && i + 1 < argc) c.n_gpus = atoi(argv[++i]);
        else if (a == "--help"   || a == "-h") {
            printf("Usage: gelu_mgpu_align [--calls N] [--n-gpus G]\n");
            exit(0);
        }
    }
    if (c.calls < 1) c.calls = 1;
    if (c.n_gpus < 1) c.n_gpus = 1;
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

struct ThreadResult {
    int gpu = -1;
    bool finished = false;
    bool exception = false;
    string err;
    int calls_done = 0;
    vector<double> per_call_ms;
};

}  // namespace

int main(int argc, char **argv) {
    Config cfg = parse_cli(argc, argv);

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < cfg.n_gpus) {
        fprintf(stderr, "[FATAL] requested %d GPUs but only %d visible — abort\n",
                cfg.n_gpus, dev_count);
        return 1;
    }

    printf("════════════════════════════════════════════════════════════\n");
    printf("  MGPU-MICRO GELU — NEXUS GELU, logN=16, %d GPUs\n", cfg.n_gpus);
    printf("  poly_degree=65,536, slot_count=32,768\n");
    printf("  Coeff moduli: {58, 18×40, 58} (NEXUS COEFF_MODULI[4], 20 moduli)\n");
    printf("  Total calls: %d (≈%d per GPU)\n", cfg.calls,
           (cfg.calls + cfg.n_gpus - 1) / cfg.n_gpus);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    // ═══ NEXUS GELU CKKS params (matches main.cu COEFF_MODULI[4]) ═══
    size_t N = 1ULL << 16;
    double SCALE = pow(2.0, 40);

    // NEXUS GELU COEFF_MODULI[4] = {58, 40×18, 58} (20 moduli, not 19 — counted
    // from vendor/nexus/cuda/src/main.cu line 37 with `tr ',' '\n' | wc -l`).
    vector<int> coeff_bits;
    coeff_bits.push_back(58);
    for (int i = 0; i < 18; i++) coeff_bits.push_back(40);
    coeff_bits.push_back(58);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));

    vector<ThreadResult> results(cfg.n_gpus);
    int chunk = (cfg.calls + cfg.n_gpus - 1) / cfg.n_gpus;

    atomic<int> ready{0};

    auto wall_t0 = chrono::high_resolution_clock::now();

    vector<thread> threads;
    for (int g = 0; g < cfg.n_gpus; g++) {
        int start_call = g * chunk;
        int end_call   = std::min(start_call + chunk, cfg.calls);
        if (start_call >= cfg.calls) {
            results[g].finished = true;
            continue;
        }
        int my_calls = end_call - start_call;
        threads.emplace_back([&, g, my_calls]() {
            ThreadResult &r = results[g];
            r.gpu = g;
            r.per_call_ms.reserve(my_calls);
            try {
                cudaSetDevice(g);

                PhantomContext     context(parms);
                PhantomCKKSEncoder encoder(context);
                PhantomSecretKey   secret_key(context);
                PhantomPublicKey   public_key = secret_key.gen_publickey(context);
                PhantomRelinKey    relin_keys = secret_key.gen_relinkey(context);
                PhantomGaloisKey   galois_keys = secret_key.create_galois_keys(context);

                CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder,
                                             &relin_keys, &galois_keys, SCALE);
                GELUEvaluator gelu_evaluator(ckks_evaluator);

                size_t slot_count = encoder.slot_count();

                mt19937 rng(13371337 + g);
                uniform_real_distribution<double> dist(-4.0, 4.0);
                vector<double> input(slot_count, 0.0);
                for (size_t i = 0; i < slot_count; i++) input[i] = dist(rng);

                PhantomPlaintext plain_input;
                ckks_evaluator.encoder.encode(input, SCALE, plain_input);

                // Warmup — use a throwaway ciphertext so we don't mutate
                // anything reused below. NEXUS's gelu mutates its input in
                // place via mod_switch_to_inplace; reusing the same ct twice
                // depletes the modulus chain and triggers "end of modulus
                // switching chain reached" on the second call.
                {
                    PhantomCiphertext warm_in;
                    ckks_evaluator.encryptor.encrypt(plain_input, warm_in);
                    PhantomCiphertext rtn;
                    gelu_evaluator.gelu(warm_in, rtn);
                    cudaDeviceSynchronize();
                }

                printf("[T%d / GPU %d] setup done, %d calls scheduled\n",
                       g, g, my_calls);
                fflush(stdout);

                ready.fetch_add(1);
                while (ready.load() < cfg.n_gpus) { /* busy wait */ }

                for (int k = 0; k < my_calls; k++) {
                    // Fresh ciphertext per iteration — gelu() mutates its
                    // input in place (nexus_eval/gelu.cu calls
                    // mod_switch_to_inplace on x), so a single base_cipher
                    // copy would be depleted after one call.
                    PhantomCiphertext input_ct;
                    ckks_evaluator.encryptor.encrypt(plain_input, input_ct);
                    PhantomCiphertext rtn;

                    cudaDeviceSynchronize();
                    auto t0 = chrono::high_resolution_clock::now();
                    {
                        NVTX_SCOPE("gelu_mgpu");
                        gelu_evaluator.gelu(input_ct, rtn);
                    }
                    cudaDeviceSynchronize();
                    auto t1 = chrono::high_resolution_clock::now();
                    double ms = chrono::duration<double, milli>(t1 - t0).count();
                    r.per_call_ms.push_back(ms);

                    if (k < 3 || k == my_calls - 1) {
                        printf("[T%d / GPU %d] call %d/%d: %.2f ms\n",
                               g, g, k + 1, my_calls, ms);
                        fflush(stdout);
                    }
                }

                r.calls_done = my_calls;
                r.finished = true;
            } catch (std::exception &e) {
                r.exception = true;
                r.err = e.what();
                fprintf(stderr, "[T%d / GPU %d] EXCEPTION: %s\n", g, g, e.what());
            } catch (const char *s) {
                r.exception = true;
                r.err = s ? s : "(null char*)";
                fprintf(stderr, "[T%d / GPU %d] EXCEPTION (char*): %s\n", g, g, r.err.c_str());
            } catch (...) {
                r.exception = true;
                r.err = "(unknown)";
                fprintf(stderr, "[T%d / GPU %d] EXCEPTION (unknown)\n", g, g);
            }
        });
    }

    for (auto &t : threads) t.join();
    auto wall_t1 = chrono::high_resolution_clock::now();
    double wall_total_s = chrono::duration<double>(wall_t1 - wall_t0).count();

    bool any_exception = false;
    int  total_calls_done = 0;
    vector<double> all_per_call_ms;

    printf("\n────────────── Per-thread results ──────────────\n");
    for (auto &r : results) {
        if (r.exception) {
            printf("  GPU %d: EXCEPTION (%s)\n", r.gpu, r.err.c_str());
            any_exception = true;
        } else if (r.calls_done == 0) {
            printf("  GPU %d: idle (no calls assigned)\n", r.gpu);
        } else {
            double med = median(r.per_call_ms);
            printf("  GPU %d: %d calls, median=%.2f ms\n", r.gpu, r.calls_done, med);
            total_calls_done += r.calls_done;
            for (double m : r.per_call_ms) all_per_call_ms.push_back(m);
        }
    }

    double med_ms   = median(all_per_call_ms);
    double sigma_ms = stdev(all_per_call_ms);
    double min_ms   = all_per_call_ms.empty() ? 0.0 : *min_element(all_per_call_ms.begin(), all_per_call_ms.end());
    double max_ms   = all_per_call_ms.empty() ? 0.0 : *max_element(all_per_call_ms.begin(), all_per_call_ms.end());
    double p10_ms   = percentile(all_per_call_ms, 10);
    double p90_ms   = percentile(all_per_call_ms, 90);
    double effective_per_call_ms = (total_calls_done > 0)
        ? (wall_total_s * 1000.0) / total_calls_done : 0.0;

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  MGPU-MICRO GELU HEADLINE (logN=16, %d GPUs)\n", cfg.n_gpus);
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Total calls done:          %d / %d (target)\n", total_calls_done, cfg.calls);
    printf("  Wall-clock (all calls):    %.2f s\n", wall_total_s);
    printf("  Per-call median (per-GPU): %.2f ms\n", med_ms);
    printf("  Per-call σ:                %.2f ms (%.2f%% of median)\n",
           sigma_ms, med_ms > 0 ? 100.0 * sigma_ms / med_ms : 0.0);
    printf("  Per-call min / max:        %.2f / %.2f ms\n", min_ms, max_ms);
    printf("  Per-call p10 / p90:        %.2f / %.2f ms\n", p10_ms, p90_ms);
    printf("  Effective per-call (wall/N): %.2f ms\n", effective_per_call_ms);
    printf("  ----\n");
    printf("  NEXUS published (A100):    3,350 ms\n");
    printf("  NEXUS measured H100:       69 ms (single-GPU)\n");
    printf("  multiNEXUS single-GPU:     ~69 ms (Lane ALIGN-SINGLE)\n");
    printf("  multiNEXUS %d-GPU effective: %.2f ms\n", cfg.n_gpus, effective_per_call_ms);
    printf("  speedup vs single-GPU:     %.2fx\n",
           69.0 / std::max(1.0, effective_per_call_ms));
    printf("  speedup vs NEXUS A100:     %.2fx\n",
           3350.0 / std::max(1.0, effective_per_call_ms));
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    return any_exception ? 1 : 0;
}
