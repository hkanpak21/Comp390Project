/**
 * bert_hp_multigpu.cu
 *
 * Slice S16 / S17 (HP-BERT track) — Head-Parallel BERT skeleton.
 *
 * Strategy: 1 head per GPU (S16), or N heads per GPU sequentially (S17).
 * Each GPU runs an entire head's compute graph (QKV → softmax → AV → out
 * → BS1 → LN1 → BS2 → FFN1 → GELU → FFN2 → BS3 → LN2 → BS4) in its own
 * std::thread, with its own PhantomContext / Bootstrapper / GaloisKeyStore
 * / encrypted ciphertext. No DKS — all rotations go through the
 * Phase 1 single-GPU pinned-host key streaming path
 * (`enable_key_streaming`).
 *
 * Verification: a "reference" pass on GPU 0 (run sequentially before the
 * threaded launch) computes the expected post-BS4 output for each head.
 * After the threaded run, each per-GPU head's plaintext is decoded and
 * compared element-wise; we report per-head MAE and PASS if all <1e-5.
 *
 * CLI:
 *   bert_hp_multigpu --n-gpus 4 --heads 4    (S16: 1 head per GPU)
 *   bert_hp_multigpu --n-gpus 4 --heads 12   (S17: 3 heads per GPU)
 *   bert_hp_multigpu --n-gpus 4 --heads 12 --layers 12 --skip-ref
 *     (S18: full 12-layer × 12-head HP-BERT measurement; --skip-ref
 *      because a 12-layer reference pass on GPU 0 would take ~12 × 10 s
 *      per head × 12 heads = ~24 min, dominated by serial bootstrap;
 *      verification is per-layer in S17)
 *   bert_hp_multigpu --n-gpus 4 --heads 12 --skip-ref  (skip reference)
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
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"
#include "galois.cuh"

#include "ckks_evaluator.cuh"
#include "galois_key_store.cuh"
#include "gelu.cuh"
#include "softmax.cuh"
#include "layer_norm.cuh"
#include "matrix_mul.cuh"
#include "bootstrapping/Bootstrapper.cuh"
#include "util/nvtx_tracer.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------
struct PerfTimer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        return chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
    }
};

// ---------------------------------------------------------------------------
// Per-head op-time totals (one set per GPU thread).
// ---------------------------------------------------------------------------
struct OpTimes {
    double qkv_matmul = 0, qk_matmul = 0, softmax_op = 0, av_matmul = 0,
           out_matmul = 0, bs1 = 0, ln1 = 0, bs2 = 0,
           ffn1 = 0, gelu_op = 0, ffn2 = 0, bs3 = 0, ln2 = 0, bs4 = 0;
    int heads = 0;
    void add(const OpTimes &o) {
        qkv_matmul += o.qkv_matmul; qk_matmul += o.qk_matmul; softmax_op += o.softmax_op;
        av_matmul += o.av_matmul; out_matmul += o.out_matmul;
        bs1 += o.bs1; ln1 += o.ln1; bs2 += o.bs2;
        ffn1 += o.ffn1; gelu_op += o.gelu_op; ffn2 += o.ffn2;
        bs3 += o.bs3; ln2 += o.ln2; bs4 += o.bs4;
        heads += o.heads;
    }
    double total() const {
        return qkv_matmul + qk_matmul + softmax_op + av_matmul + out_matmul +
               bs1 + ln1 + bs2 + ffn1 + gelu_op + ffn2 + bs3 + ln2 + bs4;
    }
};

// TIME_OP wraps each per-operator block with both:
//  (1) a CPU-timed pre/post cudaDeviceSynchronize for the times.<field> totals
//  (2) an NVTX range so the operator shows up as a named scope under
//      `nsys stats --report nvtxsum` in any traces collected via
//      slurm_hp_bert_nvtx_logN{15,16}.sh. The NVTX range is cheap (~100 ns
//      when not profiled) so leaving it on costs nothing for non-profiled
//      production runs.
#define TIME_OP(field, code) do {                       \
    cudaDeviceSynchronize();                            \
    NVTX_SCOPE("op:" #field);                           \
    PerfTimer _pt; _pt.start();                         \
    code;                                               \
    cudaDeviceSynchronize();                            \
    times.field += _pt.elapsed_ms();                    \
} while(0)

// ---------------------------------------------------------------------------
// Run one full BERT encoder layer (one head) using single-GPU pinned-host
// rotation. The caller is responsible for: cudaSetDevice(g), creating the
// per-thread context/keys/key-store, and providing pre-encrypted input
// ciphertext `ct_in`. Returns the post-BS4 ciphertext via `ct_out`.
// All operator evaluators are constructed inside (so they live & die on
// this thread's GPU).
// ---------------------------------------------------------------------------
static void run_one_head(
    PhantomContext   &ctx,
    CKKSEvaluator    &le,
    Bootstrapper     &lb,
    PhantomCiphertext &ct_in,
    PhantomCiphertext &ct_out,
    vector<vector<double>> &W_q,
    vector<vector<double>> &W_k,
    vector<vector<double>> &W_v,
    vector<vector<double>> &W_o,
    vector<vector<double>> &W_f1,
    vector<vector<double>> &W_f2,
    int hidden,
    int seq_len,
    OpTimes &times)
{
    (void)ctx;   // ctx is bound through `le` — not needed directly here

    NVTX_SCOPE("layer:run_one_head");
    GELUEvaluator    lg(le);
    SoftmaxEvaluator ls(le);
    LNEvaluator      ll(le);
    MMEvaluator      lm(le);

    vector<PhantomCiphertext> xi = {ct_in}, q, k, v;
    TIME_OP(qkv_matmul, {
        lm.matrix_mul_unified(xi, W_q, 1, q);
        lm.matrix_mul_unified(xi, W_k, 1, k);
        lm.matrix_mul_unified(xi, W_v, 1, v);
    });

    PhantomCiphertext as;
    TIME_OP(qk_matmul, {
        le.evaluator.mod_switch_to_inplace(k[0], q[0].chain_index());
        k[0].set_scale(q[0].scale());
        le.evaluator.multiply(q[0], k[0], as);
        le.evaluator.relinearize_inplace(as, *le.relin_keys);
        le.evaluator.rescale_to_next_inplace(as);
    });

    PhantomCiphertext aw;
    TIME_OP(softmax_op, { ls.softmax(as, aw, seq_len); });

    PhantomCiphertext ao;
    TIME_OP(av_matmul, {
        le.evaluator.mod_switch_to_inplace(v[0], aw.chain_index());
        v[0].set_scale(aw.scale());
        le.evaluator.multiply(aw, v[0], ao);
        le.evaluator.relinearize_inplace(ao, *le.relin_keys);
        le.evaluator.rescale_to_next_inplace(ao);
    });

    vector<PhantomCiphertext> pi = {ao}, po;
    TIME_OP(out_matmul, { lm.matrix_mul_unified(pi, W_o, 1, po); });

    // Lane MGPU-NSYS Pipeline-Overhead instrumentation:
    // Each TIME_OP(bsX) is split into MOD_SWITCH_DOWN (the chain
    // compaction `while` loop) + BOOTSTRAP_KERNEL (lb.bootstrap_3 itself).
    // Combined with the BS_MOD_RAISE/BS_SUBSUM/BS_MOD_REDUCTION/BS_SLOTTOCOEFF
    // scopes inside bootstrap_sparse_3 this lets us attribute the 1,032 ms
    // in-pipeline cost vs the 304 ms standalone cost.
    PhantomCiphertext b1;
    TIME_OP(bs1, {
        { NVTX_SCOPE("MOD_SWITCH_DOWN");
          while (po[0].coeff_modulus_size() > 1)
              le.evaluator.mod_switch_to_next_inplace(po[0]); }
        // FIX-BUG-04-02 (SCALE-CROSS-CUT): reset to canonical SCALE before
        // bootstrap to prevent silent scale drift across chained layers.
        // Mirrors argmax_align_n32k.cu:225. Required because our Phantom has
        // scale-mismatch checks commented out (CLAUDE.md lesson #7); without
        // this reset the second layer's bootstrap can encode a stale scale.
        po[0].scale() = le.scale;
        { NVTX_SCOPE("BOOTSTRAP_KERNEL");
          lb.bootstrap_3(b1, po[0]); }
    });

    PhantomCiphertext ln1o;
    TIME_OP(ln1, { ll.layer_norm(b1, ln1o, hidden); });

    PhantomCiphertext b2;
    TIME_OP(bs2, {
        { NVTX_SCOPE("MOD_SWITCH_DOWN");
          while (ln1o.coeff_modulus_size() > 1)
              le.evaluator.mod_switch_to_next_inplace(ln1o); }
        ln1o.scale() = le.scale;  // FIX-BUG-04-02 (SCALE-CROSS-CUT)
        { NVTX_SCOPE("BOOTSTRAP_KERNEL");
          lb.bootstrap_3(b2, ln1o); }
    });

    vector<PhantomCiphertext> fi = {b2}, fo;
    TIME_OP(ffn1, { lm.matrix_mul_unified(fi, W_f1, 1, fo); });

    PhantomCiphertext go;
    TIME_OP(gelu_op, { lg.gelu(fo[0], go); });

    vector<PhantomCiphertext> f2i = {go}, f2o;
    TIME_OP(ffn2, { lm.matrix_mul_unified(f2i, W_f2, 1, f2o); });

    PhantomCiphertext b3;
    TIME_OP(bs3, {
        { NVTX_SCOPE("MOD_SWITCH_DOWN");
          while (f2o[0].coeff_modulus_size() > 1)
              le.evaluator.mod_switch_to_next_inplace(f2o[0]); }
        f2o[0].scale() = le.scale;  // FIX-BUG-04-02 (SCALE-CROSS-CUT)
        { NVTX_SCOPE("BOOTSTRAP_KERNEL");
          lb.bootstrap_3(b3, f2o[0]); }
    });

    PhantomCiphertext ln2o;
    TIME_OP(ln2, { ll.layer_norm(b3, ln2o, hidden); });

    TIME_OP(bs4, {
        { NVTX_SCOPE("MOD_SWITCH_DOWN");
          while (ln2o.coeff_modulus_size() > 1)
              le.evaluator.mod_switch_to_next_inplace(ln2o); }
        ln2o.scale() = le.scale;  // FIX-BUG-04-02 (SCALE-CROSS-CUT)
        { NVTX_SCOPE("BOOTSTRAP_KERNEL");
          lb.bootstrap_3(ct_out, ln2o); }
    });

    times.heads++;
}

// ---------------------------------------------------------------------------
// Setup helper — runs once per thread / per reference. Builds context,
// keys, bootstrapper, and key store on the current GPU. Returns by
// out-params (everything is non-copyable so we use unique_ptr).
//
// The caller MUST have already called `cudaSetDevice(g)`.
// ---------------------------------------------------------------------------
static void setup_per_gpu(
    int g,
    const EncryptionParameters &parms,
    long logN, long logn, long logNh,
    int total_level, double SCALE,
    long boundary_K, long deg, long scale_factor, long inverse_deg, long loge,
    int seq_len, int hidden,
    const string &sk_buf,
    // out-params
    unique_ptr<PhantomContext>     &ctx_out,
    unique_ptr<PhantomCKKSEncoder> &enc_out,
    unique_ptr<PhantomSecretKey>   &sk_out,
    unique_ptr<PhantomPublicKey>   &pk_out,
    unique_ptr<PhantomRelinKey>    &rk_out,
    unique_ptr<PhantomGaloisKey>   &gk_out,
    unique_ptr<CKKSEvaluator>      &eval_out,
    unique_ptr<Bootstrapper>       &bs_out,
    unique_ptr<GaloisKeyStore>     &ks_out)
{
    ctx_out = make_unique<PhantomContext>(parms);
    enc_out = make_unique<PhantomCKKSEncoder>(*ctx_out);

    sk_out = make_unique<PhantomSecretKey>();
    { stringstream ss(sk_buf); sk_out->load(ss); }

    pk_out = make_unique<PhantomPublicKey>(sk_out->gen_publickey(*ctx_out));
    rk_out = make_unique<PhantomRelinKey>(sk_out->gen_relinkey(*ctx_out));
    gk_out = make_unique<PhantomGaloisKey>();

    eval_out = make_unique<CKKSEvaluator>(
        ctx_out.get(), pk_out.get(), sk_out.get(), enc_out.get(),
        rk_out.get(), gk_out.get(), SCALE);

    bs_out = make_unique<Bootstrapper>(
        loge, logn, logNh, total_level, SCALE,
        boundary_K, deg, scale_factor, inverse_deg, eval_out.get());
    bs_out->slot_vec.push_back(logn);
    bs_out->prepare_mod_polynomial();
    bs_out->generate_LT_coefficient_3();

    vector<int> gsteps;
    gsteps.push_back(0);
    for (int i = 0; i < logN - 1; i++) gsteps.push_back(1 << i);
    for (int i = 0; i < logN - 1; i++) gsteps.push_back(-(1 << i));
    gsteps.push_back(-seq_len);
    gsteps.push_back(-hidden);
    bs_out->addLeftRotKeys_Linear_to_vector_3(gsteps);

    {
        std::set<int> step_set(gsteps.begin(), gsteps.end());
        gsteps.assign(step_set.begin(), step_set.end());
    }
    auto gelts = ::get_elts_from_steps(gsteps, 1ULL << logN);
    ctx_out->setup_galois_tool(gelts);
    gk_out->resize_slots(gelts.size());

    ks_out = make_unique<GaloisKeyStore>();
    ks_out->generate_all_keys(*ctx_out, *sk_out, gelts.size());
    eval_out->evaluator.enable_key_streaming(ks_out.get(), gk_out.get());

    printf("[setup g=%d] context+keys ready (%zu rotation keys)\n",
           g, gelts.size());
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    int n_gpus = 4;
    int n_heads = 4;
    int inner = 64;
    int seq_len = 16;
    int hidden = 64;
    int n_layers = 1;        // S18: --layers 12 chains layers via output→input
    int n_trials = 1;        // S18: --trials 3 for median-of-3 measurement
    bool skip_ref = false;
    int  ring_N  = 65536;    // S29 PRIOR-ART: --N {32768,65536} for apples-to-apples

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc)
            n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads") && i+1 < argc)
            n_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc)
            inner = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq-len") && i+1 < argc)
            seq_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hidden") && i+1 < argc)
            hidden = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers") && i+1 < argc)
            n_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trials") && i+1 < argc)
            n_trials = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--skip-ref"))
            skip_ref = true;
        else if (!strcmp(argv[i], "--N") && i+1 < argc)
            ring_N = atoi(argv[++i]);
    }
    if (ring_N != 32768 && ring_N != 65536) {
        fprintf(stderr, "Unsupported --N %d (use 32768 or 65536)\n", ring_N);
        return 1;
    }

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < n_gpus) {
        fprintf(stderr, "Need %d GPUs, have %d\n", n_gpus, dev_count);
        return 1;
    }

    printf("════════════════════════════════════════════════════════════\n");
    printf("  HP-BERT (head-parallel) — N=%d, %d GPUs, %d heads, "
           "%d layer%s\n",
           ring_N, n_gpus, n_heads, n_layers, n_layers == 1 ? "" : "s");
    printf("  per-GPU: %d head%s sequentially, single-GPU pinned rotations\n",
           (n_heads + n_gpus - 1) / n_gpus,
           (n_heads + n_gpus - 1) / n_gpus > 1 ? "s" : "");
    printf("  hidden=%d, inner=%d, seq=%d\n", hidden, inner, seq_len);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    // ═══ CKKS params: N=65536 (default) or N=32768 (S29 PRIOR-ART) ═══
    // N=65536 mirrors bert_dks_multigpu (logN=16, main_mod=21, bs_mod=14).
    // N=32768 mirrors bert_encoder_layer.cu (logN=15, main_mod=21, bs_mod=14)
    // — the proven NEXUS-compatible parameter set used throughout this
    // codebase for apples-to-apples comparison vs NEXUS at N=32768.
    long logN     = (ring_N == 65536) ? 16 : 15;
    long logn     = logN - 2;
    long logNh    = logN - 1;
    size_t N      = 1ULL << logN;
    long sparse_slots_val = 1L << logn;
    int  logp     = 46;
    int  logq     = 51;
    int  log_special = 51;
    int  main_mod = 21;
    int  bs_mod   = 14;
    int  total_level = main_mod + bs_mod;
    double SCALE  = pow(2.0, logp);

    long boundary_K   = 25;
    long deg          = 59;
    long scale_factor = 2;
    long inverse_deg  = 1;
    long loge         = 10;

    vector<int> coeff_bits;
    coeff_bits.push_back(logq);
    for (int i = 0; i < main_mod; i++) coeff_bits.push_back(logp);
    for (int i = 0; i < bs_mod;   i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(log_special);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));
    parms.set_sparse_slots(sparse_slots_val);
    parms.set_secret_key_hamming_weight(192);

    // ═══ Setup on GPU 0: secret key + weights + per-head input ═══
    cudaSetDevice(0);
    PhantomContext   ctx0(parms);
    PhantomCKKSEncoder enc0(ctx0);
    PhantomSecretKey   sk0(ctx0);
    PhantomPublicKey   pk0 = sk0.gen_publickey(ctx0);
    PhantomRelinKey    rk0 = sk0.gen_relinkey(ctx0);
    PhantomGaloisKey   gk0_empty;
    size_t slots = enc0.slot_count();
    CKKSEvaluator eval0(&ctx0, &pk0, &sk0, &enc0, &rk0, &gk0_empty, SCALE);

    // Serialize SK so each per-GPU thread can load the same key.
    stringstream sk_buf;
    sk0.save(sk_buf);
    string sk_str = sk_buf.str();

    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.02, 0.02), idist(-0.5, 0.5);
    auto make_w = [&]() {
        vector<vector<double>> w(inner, vector<double>(slots, 0.0));
        for (auto &r : w)
            for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++)
                r[s] = wdist(rng);
        return w;
    };
    auto W_q = make_w(), W_k = make_w(), W_v = make_w(), W_o = make_w();
    auto W_f1 = make_w(), W_f2 = make_w();

    // Per-head input data (deterministic — both reference and threaded
    // runs use exactly the same bytes for head h).
    vector<vector<double>> head_inputs(n_heads, vector<double>(slots, 0.0));
    for (int h = 0; h < n_heads; h++) {
        for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++)
            head_inputs[h][s] = idist(rng);
    }

    // Encrypt all heads on GPU 0 then serialize.
    vector<string> ct_data(n_heads);
    for (int h = 0; h < n_heads; h++) {
        PhantomPlaintext pt;
        eval0.encoder.encode(head_inputs[h], SCALE, pt);
        PhantomCiphertext ct;
        eval0.encryptor.encrypt(pt, ct);
        for (int j = 0; j < bs_mod; j++)
            eval0.evaluator.mod_switch_to_next_inplace(ct);
        stringstream ss; ct.save(ss);
        ct_data[h] = ss.str();
    }
    printf("[setup] encrypted %d head input ciphertexts on GPU 0\n", n_heads);
    fflush(stdout);

    // ═══ Reference pass: run each head on GPU 0 sequentially ═══
    // Stores per-head decoded output for later element-wise compare.
    vector<vector<double>> ref_outputs(n_heads);
    if (!skip_ref) {
        printf("\n[ref] computing single-GPU reference for %d heads on GPU 0...\n", n_heads);
        fflush(stdout);

        unique_ptr<PhantomContext>     ref_ctx;
        unique_ptr<PhantomCKKSEncoder> ref_enc;
        unique_ptr<PhantomSecretKey>   ref_sk;
        unique_ptr<PhantomPublicKey>   ref_pk;
        unique_ptr<PhantomRelinKey>    ref_rk;
        unique_ptr<PhantomGaloisKey>   ref_gk;
        unique_ptr<CKKSEvaluator>      ref_eval;
        unique_ptr<Bootstrapper>       ref_bs;
        unique_ptr<GaloisKeyStore>     ref_ks;

        cudaSetDevice(0);
        setup_per_gpu(
            0, parms, logN, logn, logNh, total_level, SCALE,
            boundary_K, deg, scale_factor, inverse_deg, loge,
            seq_len, hidden, sk_str,
            ref_ctx, ref_enc, ref_sk, ref_pk, ref_rk, ref_gk,
            ref_eval, ref_bs, ref_ks);

        OpTimes ref_times;
        for (int h = 0; h < n_heads; h++) {
            PhantomCiphertext ct;
            { stringstream ss(ct_data[h]); ct.load(ss); }
            // S18: chain n_layers layers in the reference too, so the
            // per-head MAE compares like-for-like. Reset ct_out to a fresh
            // PhantomCiphertext each iteration (Phantom's bootstrap_3
            // asserts the dest is a fresh ciphertext, not a moved-from one).
            PhantomCiphertext ct_out;
            for (int layer = 0; layer < n_layers; ++layer) {
                ct_out = PhantomCiphertext();   // reset to fresh
                run_one_head(
                    *ref_ctx, *ref_eval, *ref_bs, ct, ct_out,
                    W_q, W_k, W_v, W_o, W_f1, W_f2,
                    hidden, seq_len, ref_times);
                ct = std::move(ct_out);
            }
            ct_out = std::move(ct);

            PhantomPlaintext pt;
            ref_sk->decrypt(*ref_ctx, ct_out, pt);
            ref_enc->decode(*ref_ctx, pt, ref_outputs[h]);
            printf("[ref] head %d done after %d layer(s) (decoded %zu slots)\n",
                   h, n_layers, ref_outputs[h].size());
            fflush(stdout);
        }
        printf("[ref] reference complete (sum of per-op times: %.1f ms)\n",
               ref_times.total());
        fflush(stdout);
    }

    // ═══ Distribute heads across GPUs (round-robin) ═══
    vector<vector<int>> gpu_heads(n_gpus);
    for (int i = 0; i < n_heads; i++) gpu_heads[i % n_gpus].push_back(i);
    for (int g = 0; g < n_gpus; g++) {
        printf("[dispatch] GPU %d: %zu head%s [", g, gpu_heads[g].size(),
               gpu_heads[g].size() == 1 ? "" : "s");
        for (size_t k = 0; k < gpu_heads[g].size(); k++)
            printf("%s%d", k == 0 ? "" : ",", gpu_heads[g][k]);
        printf("]\n");
    }
    fflush(stdout);

    // ═══ Threaded HP-BERT execution ═══
    printf("\n═══ Threaded HP-BERT on %d GPUs ═══\n", n_gpus);
    fflush(stdout);

    // Per-thread results: decoded outputs for each head this GPU owns.
    // Keyed by global head index.
    vector<vector<double>> head_outputs(n_heads);
    vector<bool>           head_done(n_heads, false);
    vector<string>         head_err(n_heads);
    OpTimes                global_times;
    mutex                  result_mtx;

    atomic<int> setup_done{0};
    PerfTimer compute_timer, total_timer;
    total_timer.start();

    vector<thread> threads;
    for (int g = 0; g < n_gpus; g++) {
        threads.emplace_back([&, g]() {
            try {
                cudaSetDevice(g);

                unique_ptr<PhantomContext>     ctx;
                unique_ptr<PhantomCKKSEncoder> enc;
                unique_ptr<PhantomSecretKey>   sk;
                unique_ptr<PhantomPublicKey>   pk;
                unique_ptr<PhantomRelinKey>    rk;
                unique_ptr<PhantomGaloisKey>   gk;
                unique_ptr<CKKSEvaluator>      eval;
                unique_ptr<Bootstrapper>       bs;
                unique_ptr<GaloisKeyStore>     ks;

                setup_per_gpu(
                    g, parms, logN, logn, logNh, total_level, SCALE,
                    boundary_K, deg, scale_factor, inverse_deg, loge,
                    seq_len, hidden, sk_str,
                    ctx, enc, sk, pk, rk, gk, eval, bs, ks);

                int my_count = setup_done.fetch_add(1) + 1;
                while (setup_done.load() < n_gpus) { /* spin */ }
                if (my_count == n_gpus) compute_timer.start();

                OpTimes times;
                for (int h_idx : gpu_heads[g]) {
                    PhantomCiphertext ct;
                    { stringstream ss(ct_data[h_idx]); ct.load(ss); }

                    // ── S18: chain n_layers layers per head. After each
                    //    layer's BS#4, ct_out is refreshed back to
                    //    coeff_modulus_size = bs_mod + main_mod + 1 - bs_mod
                    //    = main_mod + 1 (matches the input level after the
                    //    initial mod_switch loop on the fresh ct), so we
                    //    can feed ct_out directly into the next layer.
                    //    IMPORTANT: re-construct ct_out each iteration so
                    //    Phantom's `bootstrap_3` sees a fresh dest (it
                    //    asserts "Return cipher should initially be a new
                    //    ciphertext" — std::move alone leaves the ct_out
                    //    object in a moved-from but still-allocated state).
                    PhantomCiphertext ct_out;
                    for (int layer = 0; layer < n_layers; ++layer) {
                        ct_out = PhantomCiphertext();   // reset to fresh
                        run_one_head(
                            *ctx, *eval, *bs, ct, ct_out,
                            W_q, W_k, W_v, W_o, W_f1, W_f2,
                            hidden, seq_len, times);
                        ct = std::move(ct_out);
                        if (n_layers > 1) {
                            printf("[GPU %d] head %d layer %d/%d done "
                                   "(out level=%zu)\n",
                                   g, h_idx, layer + 1, n_layers,
                                   ct.coeff_modulus_size());
                            fflush(stdout);
                        }
                    }
                    ct_out = std::move(ct);  // restore ct_out for decode

                    // Decode for verification.
                    PhantomPlaintext pt;
                    sk->decrypt(*ctx, ct_out, pt);
                    vector<double> dec;
                    enc->decode(*ctx, pt, dec);
                    {
                        lock_guard<mutex> lk(result_mtx);
                        head_outputs[h_idx] = std::move(dec);
                        head_done[h_idx] = true;
                    }
                    printf("[GPU %d] head %d COMPLETE after %d layer(s) "
                           "(out level=%zu)\n",
                           g, h_idx, n_layers, ct_out.coeff_modulus_size());
                    fflush(stdout);
                }

                cudaDeviceSynchronize();
                {
                    lock_guard<mutex> lk(result_mtx);
                    global_times.add(times);
                }
            } catch (std::exception &e) {
                lock_guard<mutex> lk(result_mtx);
                fprintf(stderr, "[GPU %d] EXCEPTION: %s\n", g, e.what());
                for (int h_idx : gpu_heads[g])
                    if (!head_done[h_idx])
                        head_err[h_idx] = string("exception: ") + e.what();
            } catch (const char *s) {
                lock_guard<mutex> lk(result_mtx);
                fprintf(stderr, "[GPU %d] EXCEPTION (char*): %s\n", g, s ? s : "(null)");
                for (int h_idx : gpu_heads[g])
                    if (!head_done[h_idx])
                        head_err[h_idx] = "exception (char*)";
            } catch (...) {
                lock_guard<mutex> lk(result_mtx);
                fprintf(stderr, "[GPU %d] EXCEPTION (unknown)\n", g);
                for (int h_idx : gpu_heads[g])
                    if (!head_done[h_idx])
                        head_err[h_idx] = "exception (unknown)";
            }
        });
    }

    for (auto &t : threads) t.join();
    double compute_ms = compute_timer.elapsed_ms();
    double total_ms   = total_timer.elapsed_ms();

    // ═══ Verification ═══
    bool overall_pass = true;
    // FIX-BUG-02-01: tighten MAE gate from 1e-5 to the PRD spec 2.25e-6
    // (docs/prd/PRD-multiNEXUS-paper.md, Testing Decisions). The looser
    // 1e-5 was inherited from earlier per-op work; for the chained HP-BERT
    // pipeline at logN=15/16 the post-FIX-BUG-04-{01,02} reference run
    // produces MAE well within the tighter bound, so this is the floor the
    // paper should actually be gating on.
    double mae_threshold = 2.25e-6;
    int  cmp_n = static_cast<int>(sparse_slots_val);   // sparse-slot range
    if (skip_ref) {
        printf("\n[verify] --skip-ref set; skipping correctness check.\n");
        fflush(stdout);
    } else {
        printf("\n────────────── Per-head MAE (vs single-GPU reference) ──────────────\n");
        for (int h = 0; h < n_heads; h++) {
            if (!head_done[h]) {
                printf("  head %2d: MISSING (%s)\n", h,
                       head_err[h].empty() ? "no output" : head_err[h].c_str());
                overall_pass = false;
                continue;
            }
            const auto &ref = ref_outputs[h];
            const auto &got = head_outputs[h];
            int n = std::min({cmp_n,
                              static_cast<int>(ref.size()),
                              static_cast<int>(got.size())});
            if (n <= 0) {
                printf("  head %2d: empty decode (ref=%zu, got=%zu)\n",
                       h, ref.size(), got.size());
                overall_pass = false;
                continue;
            }
            double mae = 0.0;
            for (int i = 0; i < n; i++) mae += fabs(ref[i] - got[i]);
            mae /= n;
            const char *tag = (mae < mae_threshold) ? "PASS" : "FAIL";
            if (mae >= mae_threshold) overall_pass = false;
            printf("  head %2d: MAE=%.3e  (over %d slots)  %s\n",
                   h, mae, n, tag);
        }
    }

    // ═══ Headline ═══
    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  HP-BERT result — %d GPUs / %d heads / %d layer%s / N=%d\n",
           n_gpus, n_heads, n_layers, n_layers == 1 ? "" : "s", ring_N);
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Setup:   %8.1f ms\n", total_ms - compute_ms);
    printf("  Compute: %8.1f ms (concurrent across %d GPUs)\n",
           compute_ms, n_gpus);
    printf("  Total:   %8.1f ms = %.2f s\n", total_ms, total_ms / 1000.0);
    if (n_layers > 1) {
        printf("  Per-layer: %.1f ms (= compute / n_layers)\n",
               compute_ms / n_layers);
    }
    if (n_heads >= 12 && n_layers >= 12) {
        // Reference: CPU streaming, head-parallel = 249,600 ms = 249.6 s
        double speedup = 249600.0 / compute_ms;
        printf("  HP-BERT 12-layer × 12-head total compute: %.2f s\n",
               compute_ms / 1000.0);
        printf("  Speedup vs CPU streaming (249.6 s): %.2f×\n", speedup);
        printf("  Phase 4b champion (107.08 s, 2.33×): %.2f× over Phase 4b\n",
               107080.0 / compute_ms);
    }

    int H = global_times.heads > 0 ? global_times.heads : 1;
    double sum = global_times.total();
    auto row = [&](const char *name, double v) {
        printf("  %-18s %9.1f ms   %7.1f ms/head   %5.1f%%\n",
               name, v, v / H, sum > 0 ? 100.0 * v / sum : 0.0);
    };
    printf("\n─── Per-operation timing (summed across %d head×GPU completions) ───\n", H);
    row("QKV MatMul",    global_times.qkv_matmul);
    row("Q*K^T MatMul",  global_times.qk_matmul);
    row("Softmax",       global_times.softmax_op);
    row("Attn*V MatMul", global_times.av_matmul);
    row("Out MatMul",    global_times.out_matmul);
    row("Bootstrap #1",  global_times.bs1);
    row("LayerNorm #1",  global_times.ln1);
    row("Bootstrap #2",  global_times.bs2);
    row("FFN1 MatMul",   global_times.ffn1);
    row("GELU",          global_times.gelu_op);
    row("FFN2 MatMul",   global_times.ffn2);
    row("Bootstrap #3",  global_times.bs3);
    row("LayerNorm #2",  global_times.ln2);
    row("Bootstrap #4",  global_times.bs4);

    printf("\n══════════════════════════════════════════════\n");
    if (skip_ref) {
        printf("  HP-BERT verification SKIPPED (--skip-ref)\n");
    } else {
        printf("  HP-BERT verification: %s (threshold MAE < %.0e)\n",
               overall_pass ? "PASS" : "FAIL", mae_threshold);
    }
    printf("══════════════════════════════════════════════\n");
    fflush(stdout);

    return (skip_ref || overall_pass) ? 0 : 1;
}
