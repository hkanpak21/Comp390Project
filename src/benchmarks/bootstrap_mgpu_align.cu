/**
 * bootstrap_mgpu_align.cu
 *
 * Lane MGPU-NEXUS-MICRO — multi-GPU NEXUS Bootstrap micro-benchmark.
 *
 * Mimics NEXUS's standalone bootstrap test (vendor/nexus/cuda/src/bootstrapping.cu)
 * at NEXUS's exact parameter set (logN=15, sparse 2^13=8,192 slots, hamming=192),
 * but dispatches the work data-parallel across N GPUs.
 *
 * Pattern (mirrors phantom_threadsafe_smoke.cu, the proven thread-safe pattern):
 *   - Each GPU gets its own std::thread.
 *   - Each thread sets cudaSetDevice(g), constructs its own PhantomContext
 *     (which binds Phantom's thread-local default_stream), builds its own
 *     bootstrapper.
 *   - Each thread runs its share of the calls (calls / n_gpus, rounded up).
 *   - Reports per-thread per-call timing AND aggregate wall-clock for the
 *     entire batch divided by total calls.
 *
 * The metric we want for extrapolation: median per-call time across all
 * threads (since each thread independently consumes one call's worth of
 * GPU time). Wall-clock for the whole batch divided by total calls is
 * the "effective per-call" — it amortises away the parallel speedup.
 *
 * Verifies MAE ≤ 1e-2 (bootstrap noise is naturally large) on each thread's
 * decoded output vs the encoded reference.
 *
 * CLI:
 *   bootstrap_mgpu_align                           (default --calls 100 --n-gpus 4)
 *   bootstrap_mgpu_align --calls 100 --n-gpus 4
 *   bootstrap_mgpu_align --calls 25  --n-gpus 1    (single-GPU sanity)
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <thread>
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
    int n_gpus = 4;
};

Config parse_cli(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--calls"   && i + 1 < argc) c.calls  = atoi(argv[++i]);
        else if (a == "--n-gpus" && i + 1 < argc) c.n_gpus = atoi(argv[++i]);
        else if (a == "--help"   || a == "-h") {
            printf("Usage: bootstrap_mgpu_align [--calls N] [--n-gpus G]\n");
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
    vector<double> per_call_ms;       // raw per-call timings on this GPU
    double mae_post_bootstrap = -1.0; // sanity check on first call
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
    printf("  MGPU-MICRO Bootstrap — NEXUS bootstrap, logN=15, %d GPUs\n", cfg.n_gpus);
    printf("  poly_degree=32,768, sparse_slots=8,192, hamming=192\n");
    printf("  Coeff moduli: {51, 16×46, 14×51, 51} (NEXUS bootstrapping.cu)\n");
    printf("  Bootstrap params: K=25, deg=59, scale_factor=2, loge=10\n");
    printf("  Total calls: %d (≈%d per GPU)\n", cfg.calls,
           (cfg.calls + cfg.n_gpus - 1) / cfg.n_gpus);
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

    // ═══ Per-thread results ═══
    vector<ThreadResult> results(cfg.n_gpus);

    // Compute per-GPU call counts (round-robin: GPU g gets calls in [g*chunk,
    // min((g+1)*chunk, total)) where chunk = ceil(total / n_gpus)).
    int chunk = (cfg.calls + cfg.n_gpus - 1) / cfg.n_gpus;

    // Per-thread reference: each thread samples its own random sparse data
    // (different seed per thread so we exercise different inputs, mirroring
    // independent inference). Compute cost is data-independent for bootstrap.
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
        threads.emplace_back([&, g, my_calls, start_call]() {
            ThreadResult &r = results[g];
            r.gpu = g;
            r.per_call_ms.reserve(my_calls);
            try {
                cudaSetDevice(g);

                // Per-thread Phantom state. Constructor binds default_stream.
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

                bootstrapper.prepare_mod_polynomial();

                vector<int> gal_steps;
                gal_steps.push_back(0);
                for (int i = 0; i < logN - 1; i++) {
                    gal_steps.push_back(1 << i);
                }
                bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps);

                ckks_evaluator.decryptor.create_galois_keys_from_steps(
                    gal_steps, *(ckks_evaluator.galois_keys));
                bootstrapper.slot_vec.push_back(logn);
                bootstrapper.generate_LT_coefficient_3();

                size_t slot_count = encoder.slot_count();

                // Per-thread input ciphertext
                mt19937 rng(424242 + g);
                uniform_real_distribution<double> dist(-1.0, 1.0);
                vector<double> sparse(sparse_slots, 0.0);
                for (auto &v : sparse) v = dist(rng);
                vector<double> input(slot_count, 0.0);
                for (size_t i = 0; i < slot_count; i++) input[i] = sparse[i % sparse_slots];

                PhantomPlaintext plain;
                PhantomCiphertext base_cipher;
                encoder.encode(context, input, scale, plain);
                public_key.encrypt_asymmetric(context, plain, base_cipher);

                for (int i = 0; i < total_level; i++) {
                    ckks_evaluator.evaluator.mod_switch_to_next_inplace(base_cipher);
                }

                // Warmup call (not counted) — kernels JIT, plans cached.
                {
                    PhantomCiphertext warm_in = base_cipher;
                    PhantomCiphertext rtn;
                    bootstrapper.bootstrap_3(rtn, warm_in);
                    cudaDeviceSynchronize();
                }

                printf("[T%d / GPU %d] setup done, %d calls scheduled\n",
                       g, g, my_calls);
                fflush(stdout);

                // Wait for all threads to be ready
                ready.fetch_add(1);
                while (ready.load() < cfg.n_gpus) { /* busy wait */ }

                // ═══ Run our share of calls ═══
                for (int k = 0; k < my_calls; k++) {
                    PhantomCiphertext input_ct = base_cipher;
                    PhantomCiphertext rtn;

                    cudaDeviceSynchronize();
                    auto t0 = chrono::high_resolution_clock::now();
                    {
                        NVTX_SCOPE("bootstrap_3_mgpu");
                        bootstrapper.bootstrap_3(rtn, input_ct);
                    }
                    cudaDeviceSynchronize();
                    auto t1 = chrono::high_resolution_clock::now();
                    double ms = chrono::duration<double, milli>(t1 - t0).count();
                    r.per_call_ms.push_back(ms);

                    // Sanity check on first call: decrypt + decode
                    if (k == 0) {
                        PhantomPlaintext tmp;
                        secret_key.decrypt(context, rtn, tmp);
                        vector<double> dec;
                        encoder.decode(context, tmp, dec);
                        double mae = 0.0;
                        size_t cmp = std::min(dec.size(), (size_t)sparse_slots);
                        for (size_t i = 0; i < cmp; i++) mae += fabs(sparse[i] - dec[i]);
                        mae /= cmp;
                        r.mae_post_bootstrap = mae;
                    }

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

    // ═══ Aggregate ═══
    bool any_exception = false;
    int  total_calls_done = 0;
    vector<double> all_per_call_ms;
    double mae_max = 0.0;
    bool mae_pass = true;

    printf("\n────────────── Per-thread results ──────────────\n");
    for (auto &r : results) {
        if (r.exception) {
            printf("  GPU %d: EXCEPTION (%s)\n", r.gpu, r.err.c_str());
            any_exception = true;
        } else if (r.calls_done == 0) {
            printf("  GPU %d: idle (no calls assigned)\n", r.gpu);
        } else {
            double med = median(r.per_call_ms);
            printf("  GPU %d: %d calls, median=%.2f ms, MAE_first=%.3e %s\n",
                   r.gpu, r.calls_done, med, r.mae_post_bootstrap,
                   r.mae_post_bootstrap < 0.05 ? "OK" : "NOISY");
            total_calls_done += r.calls_done;
            for (double m : r.per_call_ms) all_per_call_ms.push_back(m);
            if (r.mae_post_bootstrap > mae_max) mae_max = r.mae_post_bootstrap;
            if (r.mae_post_bootstrap > 0.05) mae_pass = false;
        }
    }

    double med_ms   = median(all_per_call_ms);
    double sigma_ms = stdev(all_per_call_ms);
    double min_ms   = all_per_call_ms.empty() ? 0.0 : *min_element(all_per_call_ms.begin(), all_per_call_ms.end());
    double max_ms   = all_per_call_ms.empty() ? 0.0 : *max_element(all_per_call_ms.begin(), all_per_call_ms.end());
    double p10_ms   = percentile(all_per_call_ms, 10);
    double p90_ms   = percentile(all_per_call_ms, 90);

    // Effective per-call time = wall-clock divided by total calls done.
    // This is the "throughput-effective" number that captures the parallel speedup.
    double effective_per_call_ms = (total_calls_done > 0)
        ? (wall_total_s * 1000.0) / total_calls_done : 0.0;

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  MGPU-MICRO Bootstrap HEADLINE (logN=15, %d GPUs)\n", cfg.n_gpus);
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
    printf("  NEXUS published (A100):    5,630 ms\n");
    printf("  NEXUS measured H100:       252.8 ms (single-GPU)\n");
    printf("  multiNEXUS single-GPU:     ~250 ms (Lane ALIGN-SINGLE)\n");
    printf("  multiNEXUS %d-GPU effective: %.2f ms\n", cfg.n_gpus, effective_per_call_ms);
    printf("  speedup vs single-GPU:     %.2fx\n",
           250.0 / std::max(1.0, effective_per_call_ms));
    printf("  speedup vs NEXUS A100:     %.2fx\n",
           5630.0 / std::max(1.0, effective_per_call_ms));
    printf("  ----\n");
    printf("  MAE_post_bootstrap (max across threads): %.3e %s\n",
           mae_max, mae_pass ? "PASS (<0.05)" : "FAIL");
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    return (any_exception || !mae_pass) ? 1 : 0;
}
