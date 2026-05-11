/**
 * argmax_align_n32k.cu
 *
 * Lane ALIGN — NEXUS Argmax alignment benchmark.
 *
 * Mimics NEXUS's argmax_test (vendor/nexus/cuda/src/main.cu lines 108-251)
 * at NEXUS's chosen Argmax parameter set (logN=15, poly_degree=32,768,
 * sparse_slots=2^13=8,192, main_mod_count=17, bs_mod_count=14).
 *
 * Lane ARGMAX-FIX (2026-05-10): added explicit scale-reset to the design
 * SCALE (2^46) before each bootstrap inside QuickMax. Without this reset,
 * scale drift from the per-round multiply+rescale-by-not-quite-2^46-prime
 * accumulates across rounds. By round 3 (vocab=8) the drifted scale fed
 * to the bootstrapper precomputed-LT-coefficient path triggered Phantom's
 * encode validation `log2(scale)+1 >= total_coeff_modulus_bit_count`
 * (vendor/phantom/src/ckks.cu line 107) somewhere inside `slottocoeff_3`.
 * NEXUS's vendored Phantom is byte-identical for ckks.cu encode, but
 * NEXUS's bundled `evaluate.cu` keeps scale-mismatch checks ENABLED
 * (vendor/nexus/cuda/thirdparty/phantom-fhe/src/evaluate.cu lines 276,
 * 1131, 1194), which forces scales to harmonize at every step. Our
 * vendor/phantom/ has those checks commented out, so drift accumulates
 * silently. The explicit `x.scale() = SCALE` before `set_final_scale`
 * breaks the drift-accumulation chain.
 *
 * The QuickMax tournament used by NEXUS (vendor/nexus/cuda/src/argmax.cu
 * lines 5-43) for input length L=2^k computes:
 *
 *   for i in 0..log2(L)-1:
 *     b = rotate(x, 2^i)
 *     a_plus_b = x + b
 *     a_minus_b = x - b
 *     sign = sgn_eval(a_minus_b, 2, 2)
 *     a_minus_b_sgn = a_minus_b * sign / 2
 *     half = encode(0.5)
 *     a_plus_b = a_plus_b * half
 *     x = a_plus_b + a_minus_b_sgn   // = max(a, b) elementwise
 *     bootstrap(x)
 *
 * Followed by a final mask-and-sgn step. The dominant cost is
 * log2(L) * (sgn_eval + bootstrap).
 *
 * We run two configurations:
 *
 *   --vocab 8       (NEXUS default test, log_step=3, 3 iterations)
 *   --vocab 30522   (BERT vocab; padded up to nearest power of two = 32,768,
 *                    log_step=15, 15 iterations)
 *
 * Multi-GPU strategy:
 *   The QuickMax tournament rounds are intrinsically sequential
 *   (round i+1 depends on round i's output), so we DO NOT split rounds
 *   across GPUs. Instead, in the multi-GPU run we exploit
 *   `--n-batches B`: B independent argmax computations (e.g. B different
 *   logits vectors from different sentences) are dispatched to N GPUs in
 *   parallel. With N=4 GPUs and B=4 we get a wall-clock 4× speedup at
 *   ideal scaling; the per-argmax latency stays the same as single-GPU.
 *
 * Decision rationale: the alternative (splitting rounds) would require
 * GPU-to-GPU ciphertext transfers between every round (a 32K×L ciphertext
 * transfer + sgn_eval re-init), which likely costs more than the round
 * itself. NEXUS itself runs argmax single-stream. Our multi-GPU lever
 * for argmax is throughput, not per-call latency.
 *
 * CLI:
 *   argmax_align_n32k                        (default vocab=8, n-gpus=4, batches=4, trials=3)
 *   argmax_align_n32k --vocab 30522 --n-gpus 4 --batches 4 --trials 3
 *   argmax_align_n32k --vocab 8 --n-gpus 1 --batches 1   (single-GPU, 1 batch)
 */

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>
#include <algorithm>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"
#include "galois.cuh"

#include "ckks_evaluator.cuh"
#include "bootstrapping/Bootstrapper.cuh"
#include "util/nvtx_tracer.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

// ---------------------------------------------------------------------------
namespace {

struct Config {
    int  n_gpus  = 4;
    int  trials  = 3;
    int  vocab   = 8;
    int  batches = 4;       // independent argmax problems for the multi-GPU run
};

Config parse_cli(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--n-gpus" && i + 1 < argc) c.n_gpus = atoi(argv[++i]);
        else if (a == "--trials" && i + 1 < argc) c.trials = atoi(argv[++i]);
        else if (a == "--vocab" && i + 1 < argc) c.vocab = atoi(argv[++i]);
        else if (a == "--batches" && i + 1 < argc) c.batches = atoi(argv[++i]);
        else if (a == "--help" || a == "-h") {
            printf("Usage: argmax_align_n32k [--n-gpus N] [--trials T] "
                   "[--vocab V] [--batches B]\n");
            exit(0);
        }
    }
    if (c.n_gpus  < 1) c.n_gpus  = 1;
    if (c.trials < 1) c.trials = 1;
    if (c.batches < 1) c.batches = 1;
    return c;
}

double median(vector<double> v) {
    if (v.empty()) return 0.0;
    sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1) ? v[n/2] : 0.5*(v[n/2-1] + v[n/2]);
}

double stdev(const vector<double> &v) {
    if (v.size() < 2) return 0.0;
    double m = 0.0; for (double x : v) m += x; m /= v.size();
    double s = 0.0; for (double x : v) s += (x-m)*(x-m);
    return sqrt(s / (v.size() - 1));
}

// Round to next power of two
int next_pow2(int x) {
    if (x <= 1) return 1;
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

// ---------------------------------------------------------------------------
// QuickMax inline (mirror of vendor/nexus/cuda/src/argmax.cu lines 5-43).
// We implement it inline here rather than depending on vendor/nexus/ at link
// time (vendor/nexus/cuda/ has a separate phantom-fhe build tree).
// ---------------------------------------------------------------------------
struct ArgmaxRunCtx {
    PhantomContext   *ctx;
    CKKSEvaluator    *eval;
    Bootstrapper     *bs;
    int sparse_slots;
    double SCALE;            // ARGMAX-FIX: design scale (2^logp) for explicit drift reset
};

void argmax_inline(
    ArgmaxRunCtx &rc,
    PhantomCiphertext &x,    // in/out
    PhantomCiphertext &x_copy,
    int len)
{
    NVTX_SCOPE("op:argmax");
    PhantomCiphertext tmp, b, sign, a_plus_b, a_minus_b, a_minus_b_sgn;
    PhantomPlaintext one, half;

    int log_step = (int)log2((double)len);

    // x = [a_0..a_n, 0..0] -> [a_0..a_n, a_0..a_n, 0..0]
    rc.eval->evaluator.rotate_vector(x, -len, *(rc.eval->galois_keys), tmp);
    rc.eval->evaluator.add_inplace(x, tmp);

    for (int i = 0; i < log_step; ++i) {
        NVTX_SCOPE_FMT("argmax:round_%d", i);
        rc.eval->evaluator.rotate_vector(x, (int)pow(2, i),
                                         *(rc.eval->galois_keys), b);

        rc.eval->evaluator.add(x, b, a_plus_b);
        rc.eval->evaluator.sub(x, b, a_minus_b);

        {
            NVTX_SCOPE("argmax:sgn_eval");
            sign = rc.eval->sgn_eval(a_minus_b, 2, 2);
        }

        // (a - b) * sgn(a - b) / 2
        rc.eval->evaluator.mod_switch_to_inplace(a_minus_b, sign.params_id());
        rc.eval->evaluator.multiply(a_minus_b, sign, a_minus_b_sgn);
        rc.eval->evaluator.relinearize_inplace(a_minus_b_sgn, *(rc.eval->relin_keys));
        rc.eval->evaluator.rescale_to_next_inplace(a_minus_b_sgn);

        // (a + b) / 2
        rc.eval->encoder.encode(0.5, a_plus_b.params_id(), a_plus_b.scale(), half);
        rc.eval->evaluator.multiply_plain_inplace(a_plus_b, half);
        rc.eval->evaluator.rescale_to_next_inplace(a_plus_b);

        // a = max(a, b)
        a_plus_b.scale() = a_minus_b_sgn.scale();
        rc.eval->evaluator.mod_switch_to_inplace(a_plus_b, a_minus_b_sgn.params_id());
        rc.eval->evaluator.add(a_plus_b, a_minus_b_sgn, x);

        // bootstrap x
        {
            NVTX_SCOPE("argmax:bootstrap");
            while (x.coeff_modulus_size() > 1) {
                rc.eval->evaluator.mod_switch_to_next_inplace(x);
            }
            // ARGMAX-FIX (2026-05-10): reset to design SCALE before bootstrap.
            // Without this, multiply+rescale rounds compound a small ratio
            // delta (prime / 2^logp ≠ 1 exactly) and by round 3 the drifted
            // scale exceeds Phantom's encode validation in slottocoeff_3.
            // Resetting to canonical SCALE is mathematically sound because
            // the actual cipher value's magnitude is independent of the
            // bookkeeping `scale_` field — what matters is that we tell the
            // bootstrapper a consistent value via set_final_scale() and that
            // the output ciphertext carries that same canonical scale.
            x.scale() = rc.SCALE;
            PhantomCiphertext rtn;
            rc.bs->set_final_scale(x.scale());
            rc.bs->bootstrap_3(rtn, x);
            x = rtn;
        }
    }

    // Final stage: x_copy = sgn(x_copy - x) + 1
    x_copy.scale() = x.scale();
    rc.eval->evaluator.mod_switch_to_inplace(x_copy, x.params_id());
    rc.eval->evaluator.sub_inplace(x_copy, x);
    {
        NVTX_SCOPE("argmax:final_sgn");
        x_copy = rc.eval->sgn_eval(x_copy, 2, 2, 1.0);
    }
    rc.eval->encoder.encode(1.0, x_copy.params_id(), x_copy.scale(), one);
    rc.eval->evaluator.add_plain_inplace(x_copy, one);
}

// ---------------------------------------------------------------------------
// Run one full argmax on this thread's GPU. Setup is hoisted out of the
// timed region (matches NEXUS argmax_test which times only argmax() in
// main.cu lines 244-249).
// ---------------------------------------------------------------------------
struct ArgmaxTrialResult {
    double argmax_ms = -1.0;
    vector<double> decoded;
};

void run_one_argmax_trial(
    const EncryptionParameters &parms,
    const string &sk_buf,
    long logN,
    int  vocab_padded,
    int  sparse_slots,
    long boundary_K, long deg, long scale_factor, long inverse_deg, long loge,
    int  total_level, int main_mod_count, int bs_mod_count, double SCALE,
    const vector<double> &input_data,
    bool decode_output,
    ArgmaxTrialResult &out)
{
    NVTX_SCOPE("argmax_trial");

    PhantomContext     ctx(parms);
    PhantomCKKSEncoder enc(ctx);
    PhantomSecretKey   sk;
    { stringstream ss(sk_buf); sk.load(ss); }
    PhantomPublicKey   pk = sk.gen_publickey(ctx);
    PhantomRelinKey    rk = sk.gen_relinkey(ctx);
    PhantomGaloisKey   gk;

    CKKSEvaluator eval(&ctx, &pk, &sk, &enc, &rk, &gk, SCALE);

    Bootstrapper bs(loge, logN - 2, logN - 1, total_level, SCALE,
                    boundary_K, deg, scale_factor, inverse_deg, &eval);
    bs.prepare_mod_polynomial();
    bs.slot_vec.push_back(logN - 2);

    // Generate galois steps as in NEXUS argmax_test
    vector<int> gal_steps;
    gal_steps.push_back(0);
    for (int i = 0; i < (int)logN - 1; i++) {
        gal_steps.push_back(1 << i);
    }
    bs.addLeftRotKeys_Linear_to_vector_3(gal_steps);

    // Argmax steps
    gal_steps.push_back(-vocab_padded);
    int log_step = (int)log2((double)vocab_padded);
    for (int i = 0; i < log_step; ++i) {
        gal_steps.push_back((int)pow(2, i));
    }

    // Deduplicate galois steps
    {
        std::sort(gal_steps.begin(), gal_steps.end());
        gal_steps.erase(std::unique(gal_steps.begin(), gal_steps.end()),
                        gal_steps.end());
    }

    eval.decryptor.create_galois_keys_from_steps(gal_steps, *(eval.galois_keys));
    bs.generate_LT_coefficient_3();

    // Encrypt input (replicated to fill all slots, NEXUS sparse pattern)
    size_t slot_count = enc.slot_count();
    vector<double> input_full(slot_count, 0.0);
    for (size_t i = 0; i < slot_count; i++) {
        input_full[i] = input_data[i % sparse_slots];
    }

    PhantomPlaintext plain_input;
    PhantomCiphertext cipher_input, cipher_output;
    eval.encoder.encode(input_full, SCALE, plain_input);
    eval.encryptor.encrypt(plain_input, cipher_input);
    eval.encryptor.encrypt(plain_input, cipher_output);

    // mod-switch input down to the lowest level (NEXUS does this before timing)
    for (int j = 0; j < bs_mod_count; j++) {
        eval.evaluator.mod_switch_to_next_inplace(cipher_input);
    }

    ArgmaxRunCtx rc{&ctx, &eval, &bs, sparse_slots, SCALE};

    cudaDeviceSynchronize();
    auto t0 = chrono::high_resolution_clock::now();
    argmax_inline(rc, cipher_input, cipher_output, vocab_padded);
    cudaDeviceSynchronize();
    auto t1 = chrono::high_resolution_clock::now();
    out.argmax_ms = chrono::duration<double, milli>(t1 - t0).count();

    if (decode_output) {
        PhantomPlaintext pt_out;
        sk.decrypt(ctx, cipher_output, pt_out);
        enc.decode(ctx, pt_out, out.decoded);
    }

    (void)main_mod_count;
}

}  // namespace

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    Config cfg = parse_cli(argc, argv);

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (cfg.n_gpus > dev_count) {
        fprintf(stderr,
                "[FATAL] requested %d GPUs but only %d visible — abort\n",
                cfg.n_gpus, dev_count);
        return 1;
    }

    // Pad vocab to next power of two for QuickMax
    int vocab_padded = next_pow2(cfg.vocab);
    int log_step = (int)log2((double)vocab_padded);

    printf("════════════════════════════════════════════════════════════\n");
    printf("  ALIGN-Argmax — NEXUS Argmax at logN=15 (poly_degree=32,768)\n");
    printf("  vocab=%d (padded to %d for QuickMax, log_step=%d)\n",
           cfg.vocab, vocab_padded, log_step);
    printf("  GPUs: %d, batches: %d, trials: %d\n",
           cfg.n_gpus, cfg.batches, cfg.trials);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    // ═══ NEXUS Argmax CKKS params (matches main.cu argmax_test) ═══
    long logN = 15;
    long logn = logN - 2;          // 13
    long sparse_slots = (1L << logn);  // 8,192
    int  logp = 46;
    int  logq = 51;
    int  log_special_prime = 51;
    int  main_mod_count = 17;       // QuickMax (NEXUS COEFF_MODULI[1])
    int  bs_mod_count   = 14;
    int  total_level    = main_mod_count + bs_mod_count;
    int  hamming_weight = 192;

    long boundary_K   = 25;
    long deg          = 59;
    long scale_factor = 2;
    long inverse_deg  = 1;
    long loge         = 10;

    vector<int> coeff_bits;
    coeff_bits.push_back(logq);
    for (int i = 0; i < main_mod_count; i++) coeff_bits.push_back(logp);
    for (int i = 0; i < bs_mod_count;   i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(log_special_prime);

    EncryptionParameters parms(scheme_type::ckks);
    size_t poly_modulus_degree = (size_t)(1 << logN);
    double SCALE = pow(2.0, logp);

    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bits));
    parms.set_secret_key_hamming_weight(hamming_weight);
    parms.set_sparse_slots(sparse_slots);

    printf("[Setup] CKKS N=%zu, L=%d (main %d + bs %d + boundary 2), slots=%ld\n",
           poly_modulus_degree, total_level + 2, main_mod_count, bs_mod_count,
           sparse_slots);
    fflush(stdout);

    // ═══ Synthesize argmax input data ═══
    // NEXUS reads from argmax_input_8.txt; we synthesize identical-shape
    // random data with a fixed seed (the wall-clock cost is independent of
    // input contents — sgn_eval is a polynomial evaluation at fixed depth).
    mt19937 rng(8675309);
    uniform_real_distribution<double> idist(-1.0, 1.0);

    vector<vector<double>> all_inputs(cfg.batches);
    for (int b = 0; b < cfg.batches; b++) {
        all_inputs[b].assign(sparse_slots, 0.0);
        for (int i = 0; i < cfg.vocab; i++) {
            all_inputs[b][i] = idist(rng);
        }
        // Padded slots [vocab, vocab_padded) stay 0; argmax over the
        // padded zeros is fine because real values are in [-1, 1].
    }

    // ═══ Generate SK on GPU 0 + serialize ═══
    cudaSetDevice(0);
    string sk_buf;
    {
        PhantomContext   ctx0(parms);
        PhantomSecretKey sk0(ctx0);
        stringstream ss; sk0.save(ss); sk_buf = ss.str();
    }
    cudaDeviceSynchronize();
    printf("[Setup] SK serialized (%zu bytes)\n", sk_buf.size());
    fflush(stdout);

    // ═══ Phase 1: Single-GPU argmax measurement (one batch, latency) ═══
    vector<double> single_gpu_trial_ms;
    printf("\n[Phase 1] Single-GPU argmax (1 batch, %d trials on GPU 0)...\n",
           cfg.trials);
    fflush(stdout);

    for (int t = 0; t < cfg.trials; t++) {
        cudaSetDevice(0);
        ArgmaxTrialResult tr;
        run_one_argmax_trial(parms, sk_buf, logN, vocab_padded, sparse_slots,
                             boundary_K, deg, scale_factor, inverse_deg, loge,
                             total_level, main_mod_count, bs_mod_count, SCALE,
                             all_inputs[0], /*decode=*/(t == 0), tr);
        single_gpu_trial_ms.push_back(tr.argmax_ms);
        printf("  trial %d/%d: argmax=%.1f ms (= %.3f s)\n",
               t + 1, cfg.trials, tr.argmax_ms, tr.argmax_ms / 1000.0);
        fflush(stdout);
    }
    double single_med = median(single_gpu_trial_ms);
    double single_sigma = stdev(single_gpu_trial_ms);
    printf("[Phase 1] single-GPU median = %.1f ms (σ=%.1f, %.3f s)\n",
           single_med, single_sigma, single_med / 1000.0);
    fflush(stdout);

    // ═══ Phase 2: Multi-GPU throughput (batches in parallel) ═══
    if (cfg.n_gpus >= 2 && cfg.batches >= 2) {
        printf("\n[Phase 2] Multi-GPU throughput: %d batches × %d trials, "
               "round-robin across %d GPUs...\n",
               cfg.batches, cfg.trials, cfg.n_gpus);
        fflush(stdout);

        vector<double> multi_gpu_wall_trial_ms;

        for (int t = 0; t < cfg.trials; t++) {
            // Distribute batches across GPUs: gpu g handles batches g, g+ngpus, ...
            atomic<int> ready{0};
            vector<thread> threads;
            vector<vector<double>> per_thread_argmax_ms(cfg.n_gpus);

            auto t0 = chrono::high_resolution_clock::now();
            for (int g = 0; g < cfg.n_gpus; g++) {
                threads.emplace_back([&, g]() {
                    cudaSetDevice(g);
                    ready.fetch_add(1);
                    while (ready.load() < cfg.n_gpus) { /* spin */ }
                    for (int b = g; b < cfg.batches; b += cfg.n_gpus) {
                        ArgmaxTrialResult tr;
                        run_one_argmax_trial(
                            parms, sk_buf, logN, vocab_padded, sparse_slots,
                            boundary_K, deg, scale_factor, inverse_deg, loge,
                            total_level, main_mod_count, bs_mod_count, SCALE,
                            all_inputs[b], false, tr);
                        per_thread_argmax_ms[g].push_back(tr.argmax_ms);
                    }
                });
            }
            for (auto &th : threads) th.join();
            auto t1 = chrono::high_resolution_clock::now();
            double wall_ms = chrono::duration<double, milli>(t1 - t0).count();
            multi_gpu_wall_trial_ms.push_back(wall_ms);

            // Per-thread totals
            double max_thread_total = 0.0;
            for (auto &v : per_thread_argmax_ms) {
                double sum = 0.0; for (double x : v) sum += x;
                max_thread_total = std::max(max_thread_total, sum);
            }

            printf("  trial %d/%d: wall=%.1f ms, slowest-GPU compute=%.1f ms, "
                   "per-batch wall = %.1f ms\n",
                   t + 1, cfg.trials, wall_ms, max_thread_total,
                   wall_ms / cfg.batches);
            fflush(stdout);
        }

        double multi_med = median(multi_gpu_wall_trial_ms);
        double multi_sigma = stdev(multi_gpu_wall_trial_ms);
        double per_batch = multi_med / cfg.batches;
        printf("[Phase 2] multi-GPU wall median = %.1f ms (σ=%.1f, %.3f s)\n",
               multi_med, multi_sigma, multi_med / 1000.0);
        printf("[Phase 2] per-batch effective time = %.1f ms (= %.3f s)\n",
               per_batch, per_batch / 1000.0);
        printf("[Phase 2] throughput speedup vs single-GPU per-call latency = %.2fx\n",
               single_med / per_batch);
        printf("  (interpretation: with %d GPUs serving %d concurrent users, each\n"
               "   user's argmax completes in %.3f s wall-clock vs single-GPU's %.3f s)\n",
               cfg.n_gpus, cfg.batches, per_batch / 1000.0, single_med / 1000.0);
        fflush(stdout);
    }

    // ═══ Headline summary ═══
    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  ALIGN-Argmax HEADLINE (logN=15, vocab=%d→padded=%d)\n",
           cfg.vocab, vocab_padded);
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Single-GPU H100 latency: %.1f ms (= %.3f s)\n",
           single_med, single_med / 1000.0);
    printf("  NEXUS published (A100):  2.48 s (vocab=30,522 BERT)\n");
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    return 0;
}
