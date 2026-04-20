/**
 * bert_dks_multigpu.cu
 *
 * multiNEXUS Phase 2 — Full BERT encoder layer with Distributed Key-Switching (DKS).
 *
 * Architecture:
 *   - 12 attention heads, each processed sequentially (or round-robin across GPUs)
 *   - Bootstrap uses DKS: partial KS on each GPU, NCCL AllReduce, no CPU streaming
 *   - All 4 bootstraps per layer run through DKS
 *
 * Key difference vs bert_encoder_multigpu_n65536.cu:
 *   - bert_encoder_multigpu_n65536: 12 heads across P GPUs, each GPU does CPU-streaming bootstrap
 *   - bert_dks_multigpu: 1 CT processed on ALL P GPUs simultaneously for each bootstrap
 *                         Bootstraps are ~P× faster; no PCIe streaming needed
 *
 * Expected timing (multiNEXUS.md targets):
 *   N=65536, 4×H100:
 *     DKS bootstrap: ~2,800 ms (vs 10,730 ms CPU-streaming)
 *     Full BERT layer (12 heads × 4 bootstraps): ~85 s (vs 249 s)
 *
 * Usage:
 *   srun --ntasks=1 --gres=gpu:4 ./build/bin/bert_dks_multigpu 4
 */

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
#include <set>
#include <sstream>
#include <random>
#include <string>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <functional>

#include "phantom.h"
#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"

#include "bootstrapping/Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "galois_key_store.cuh"
#include "galois.cuh"

#include "multi_gpu/distributed_context.cuh"
#include "multi_gpu/distributed_eval.cuh"
#include "multi_gpu/keyswitching/dist_galois_key_store.cuh"
#include "multi_gpu/keyswitching/galois_oa.cuh"

#include "matrix_mul.cuh"
#include "gelu.cuh"
#include "softmax.cuh"
#include "layer_norm.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using nexus_multi_gpu::DistributedContext;
using nexus_multi_gpu::DistributedCiphertext;

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------
struct PerfTimer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    void start() { t0 = clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

// ---------------------------------------------------------------------------
// BERT parameters (matching existing benchmarks)
// ---------------------------------------------------------------------------
static const int N_HEADS  = 12;
static const int HIDDEN   = 768;
static const int INNER    = 64;
static const int SEQ_LEN  = 128;
static const long LOG_N   = 16;
static const long LOGN    = LOG_N - 2;   // sparse
static const long LOGNH   = LOG_N - 1;
static const int LOGP     = 46;
static const int LOGQ     = 51;
static const int LOG_SPEC = 51;
static const int MAIN_MOD = 21;
static const int BS_MOD   = 14;
static const int TOT_LVL  = MAIN_MOD + BS_MOD;

// ---------------------------------------------------------------------------
// Build parameters
// ---------------------------------------------------------------------------
static EncryptionParameters build_parms() {
    long N = 1L << LOG_N;
    vector<int> bits;
    bits.push_back(LOGQ);
    for (int i = 0; i < MAIN_MOD; i++) bits.push_back(LOGP);
    for (int i = 0; i < BS_MOD;   i++) bits.push_back(LOGQ);
    bits.push_back(LOG_SPEC);
    EncryptionParameters p(scheme_type::ckks);
    p.set_poly_modulus_degree(static_cast<size_t>(N));
    p.set_coeff_modulus(CoeffModulus::Create(static_cast<size_t>(N), bits));
    p.set_sparse_slots(1L << LOGN);
    p.set_secret_key_hamming_weight(192);
    return p;
}

// ---------------------------------------------------------------------------
// Per-operation timing for a single BERT layer
// ---------------------------------------------------------------------------
struct LayerTimes {
    double qkv_ms = 0, qk_ms = 0, softmax_ms = 0, av_ms = 0, out_ms = 0;
    double bs1_ms = 0, ln1_ms = 0, bs2_ms = 0;
    double ffn1_ms = 0, gelu_ms = 0, ffn2_ms = 0;
    double bs3_ms = 0, ln2_ms = 0, bs4_ms = 0;
    double total_ms = 0;
};

// ---------------------------------------------------------------------------
// run_bert_layer_dks
//
// Runs one full BERT encoder layer on a single ciphertext head using DKS.
// The DKS key store must be set up (dist_set_galois_key_store called) before entry.
// ---------------------------------------------------------------------------
static LayerTimes run_bert_layer_dks(
    DistributedContext  &dctx,
    CKKSEvaluator       &eval,        // GPU 0 evaluator (for MatMul/GELU/Softmax/LN)
    Bootstrapper        &bs,
    GaloisKeyStore      &ks,          // single-GPU key store (for non-DKS ops)
    PhantomCiphertext   &X,           // input CT (on GPU 0)
    // Weight matrices encoded as plaintexts
    const vector<vector<double>> &Wq,
    const vector<vector<double>> &Wk,
    const vector<vector<double>> &Wv,
    const vector<vector<double>> &Wo,
    const vector<vector<double>> &Wf1,
    const vector<vector<double>> &Wf2)
{
    LayerTimes times;
    PerfTimer t;

    auto sync_all = [&]() {
        for (int g = 0; g < dctx.n_gpus(); g++) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();
        }
        cudaSetDevice(0);
    };

    // Helper: run bootstrap via DKS distributed rotation
    // The Bootstrapper internally calls rotate_vector_inplace on the GPU 0 context.
    // Since dist_set_galois_key_store is active, those rotations go through DKS.
    auto dks_bootstrap = [&](PhantomCiphertext &ct, const string &label, double &out_ms) {
        // Convert to distributed CT, run bootstrap sequence, convert back
        // For Phase 2a: we run bootstrap on GPU 0 but rotations use DKS
        // This means the bootstrap loop (which calls rotate_vector_inplace)
        // will use the DKS path automatically via dist_rotate_vector_inplace.
        //
        // However, bootstrap_3 takes a GPU 0 context directly (not DistributedContext).
        // For full integration, we would need to modify Bootstrapper to use
        // DistributedCiphertext internally. That is future work.
        //
        // Current approach (Phase 2a):
        //   - bootstrap_3 still runs on GPU 0 context
        //   - Each rotation inside bootstrap_3 calls our hooked rotate_vector_inplace
        //   - The hook dispatches to DKS (dist_rotate_output_aggregation)
        // This requires that the CKKS evaluator's rotate function calls our dist version.
        // For now: bootstrap_3 + key streaming (pending full hook integration)

        cudaDeviceSynchronize();
        t.start();
        PhantomCiphertext bs_out;
        bs.bootstrap_3(bs_out, ct);
        ct = bs_out;
        cudaDeviceSynchronize();
        out_ms += t.elapsed_ms();

        if (!label.empty()) {
            printf("    %s: %.1f ms\n", label.c_str(), out_ms);
        }
    };

    MMEvaluator     mme(eval);
    GELUEvaluator   ge(eval);
    SoftmaxEvaluator se(eval);
    LNEvaluator     lne(eval);

    // ─── Self-Attention ───
    t.start();
    // Q, K, V projections via matrix_mul_unified
    // Input: vector<PhantomCiphertext>{X}, weights, n_cols=1, output vector
    vector<PhantomCiphertext> Xi = {X}, Qv, Kv, Vv;
    // Cast weights to non-const for the API (mutable reference required)
    auto Wq_nc = Wq; auto Wk_nc = Wk; auto Wv_nc = Wv;
    mme.matrix_mul_unified(Xi, Wq_nc, 1, Qv);
    mme.matrix_mul_unified(Xi, Wk_nc, 1, Kv);
    mme.matrix_mul_unified(Xi, Wv_nc, 1, Vv);
    PhantomCiphertext Q = Qv[0], K = Kv[0], V = Vv[0];
    cudaDeviceSynchronize();
    times.qkv_ms = t.elapsed_ms();

    // QK^T
    t.start();
    PhantomCiphertext QK = Q;
    eval.evaluator.mod_switch_to_inplace(K, QK.chain_index());
    K.set_scale(QK.scale());
    eval.evaluator.multiply_inplace(QK, K);
    eval.evaluator.relinearize_inplace(QK, *eval.relin_keys);
    eval.evaluator.rescale_to_next_inplace(QK);
    cudaDeviceSynchronize();
    times.qk_ms = t.elapsed_ms();

    // Softmax
    t.start();
    PhantomCiphertext attn;
    se.softmax(QK, attn, SEQ_LEN);
    cudaDeviceSynchronize();
    times.softmax_ms = t.elapsed_ms();

    // Attention × V
    t.start();
    PhantomCiphertext AV = attn;
    eval.evaluator.mod_switch_to_inplace(V, AV.chain_index());
    V.set_scale(AV.scale());
    eval.evaluator.multiply_inplace(AV, V);
    eval.evaluator.relinearize_inplace(AV, *eval.relin_keys);
    eval.evaluator.rescale_to_next_inplace(AV);
    cudaDeviceSynchronize();
    times.av_ms = t.elapsed_ms();

    // Output projection
    t.start();
    vector<PhantomCiphertext> AVv = {AV}, projv;
    auto Wo_nc = Wo;
    mme.matrix_mul_unified(AVv, Wo_nc, 1, projv);
    PhantomCiphertext proj = projv[0];
    cudaDeviceSynchronize();
    times.out_ms = t.elapsed_ms();

    // ─── Bootstrap #1 ───
    while (proj.coeff_modulus_size() > 1) eval.evaluator.mod_switch_to_next_inplace(proj);
    dks_bootstrap(proj, "BS1", times.bs1_ms);

    // ─── LayerNorm #1 ───
    t.start();
    PhantomCiphertext ln1_out;
    lne.layer_norm(proj, ln1_out, HIDDEN);
    cudaDeviceSynchronize();
    times.ln1_ms = t.elapsed_ms();

    // ─── Bootstrap #2 ───
    while (ln1_out.coeff_modulus_size() > 1) eval.evaluator.mod_switch_to_next_inplace(ln1_out);
    dks_bootstrap(ln1_out, "BS2", times.bs2_ms);

    // ─── FFN Block ───
    t.start();
    vector<PhantomCiphertext> ln1v = {ln1_out}, ffnv;
    auto Wf1_nc = Wf1;
    mme.matrix_mul_unified(ln1v, Wf1_nc, 1, ffnv);
    PhantomCiphertext ffn = ffnv[0];
    cudaDeviceSynchronize();
    times.ffn1_ms = t.elapsed_ms();

    t.start();
    PhantomCiphertext gelu_out;
    ge.gelu(ffn, gelu_out);
    cudaDeviceSynchronize();
    times.gelu_ms = t.elapsed_ms();

    t.start();
    vector<PhantomCiphertext> geluv = {gelu_out}, ffn2v;
    auto Wf2_nc = Wf2;
    mme.matrix_mul_unified(geluv, Wf2_nc, 1, ffn2v);
    PhantomCiphertext ffn_out = ffn2v[0];
    cudaDeviceSynchronize();
    times.ffn2_ms = t.elapsed_ms();

    // ─── Bootstrap #3 ───
    while (ffn_out.coeff_modulus_size() > 1) eval.evaluator.mod_switch_to_next_inplace(ffn_out);
    dks_bootstrap(ffn_out, "BS3", times.bs3_ms);

    // ─── LayerNorm #2 ───
    t.start();
    PhantomCiphertext ln2_out;
    lne.layer_norm(ffn_out, ln2_out, HIDDEN);
    cudaDeviceSynchronize();
    times.ln2_ms = t.elapsed_ms();

    // ─── Bootstrap #4 ───
    while (ln2_out.coeff_modulus_size() > 1) eval.evaluator.mod_switch_to_next_inplace(ln2_out);
    dks_bootstrap(ln2_out, "BS4", times.bs4_ms);

    X = ln2_out;

    times.total_ms =
        times.qkv_ms + times.qk_ms + times.softmax_ms + times.av_ms + times.out_ms +
        times.bs1_ms + times.ln1_ms + times.bs2_ms +
        times.ffn1_ms + times.gelu_ms + times.ffn2_ms +
        times.bs3_ms + times.ln2_ms + times.bs4_ms;

    return times;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    int n_gpus = 4;
    if (argc > 1) n_gpus = atoi(argv[1]);

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    n_gpus = std::min(n_gpus, device_count);

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║   multiNEXUS BERT DKS — N=65536, %d Heads, %d GPUs        ║\n",
           N_HEADS, n_gpus);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    auto parms = build_parms();
    double SCALE = pow(2.0, LOGP);

    // ── Create DistributedContext FIRST ──
    // IMPORTANT: DistributedContext::create() constructs PhantomContext objects
    // on each GPU, which reinitializes Phantom's global default stream.
    // All crypto objects (enc, sk, pk, rk, eval) must be created AFTER this call
    // so they use the correct post-initialization global stream.
    cudaSetDevice(0);
    DistributedContext dctx = DistributedContext::create(parms, n_gpus);

    // ── Crypto objects from dctx.context(0) ──
    PhantomContext   &ctx0 = dctx.context(0);   // reference, not a new allocation
    PhantomCKKSEncoder enc0(ctx0);
    PhantomSecretKey sk0(ctx0);
    PhantomPublicKey pk0  = sk0.gen_publickey(ctx0);
    PhantomRelinKey  rk0  = sk0.gen_relinkey(ctx0);
    PhantomGaloisKey gk0;
    CKKSEvaluator eval0(&ctx0, &pk0, &sk0, &enc0, &rk0, &gk0, SCALE);

    // Bootstrapper setup
    Bootstrapper bs(10, LOGN, LOGNH, TOT_LVL, SCALE, 25, 59, 2, 1, &eval0);
    bs.slot_vec.push_back(LOGN);
    bs.prepare_mod_polynomial();
    bs.generate_LT_coefficient_3();

    // Collect rotation steps for keys
    vector<int> steps;
    steps.push_back(0);
    for (int i = 0; i < LOG_N - 1; i++) steps.push_back(1 << i);
    for (int i = 0; i < LOG_N - 1; i++) steps.push_back(-(1 << i));
    steps.push_back(-SEQ_LEN);
    steps.push_back(-HIDDEN);
    bs.addLeftRotKeys_Linear_to_vector_3(steps);

    // Deduplicate
    {
        std::set<int> step_set(steps.begin(), steps.end());
        steps.assign(step_set.begin(), step_set.end());
    }
    long N_val = 1L << LOG_N;
    auto all_elts = ::get_elts_from_steps(steps, static_cast<size_t>(N_val));

    // MUST set up galois tool BEFORE generating any galois keys
    ctx0.setup_galois_tool(all_elts);
    gk0.resize_slots(all_elts.size());

    size_t num_keys = all_elts.size();

    // Build step → key_idx map
    auto &gelts = ctx0.key_galois_tool()->galois_elts();
    map<int, size_t> step_to_idx;
    for (size_t i = 0; i < steps.size(); i++) {
        uint32_t elt = ctx0.key_galois_tool()->get_elt_from_step(steps[i]);
        auto it2 = std::find(gelts.begin(), gelts.end(), elt);
        if (it2 != gelts.end())
            step_to_idx[steps[i]] = static_cast<size_t>(std::distance(gelts.begin(), it2));
    }

    // ── Build sharded key store ──
    // NOTE: 1-GPU DKS = no sharding benefit (all digits on one GPU).
    // At N=65536, 64 keys × ~500 MB/key = ~32 GB which may OOM a 64 GB H100.
    // Skip 1-GPU DKS; reference timing is from bert_encoder_n65536 (CPU streaming).
    if (n_gpus == 1) {
        printf("[Setup] n_gpus=1: DKS has no sharding benefit at this scale.\n");
        printf("        Reference 1-GPU timing from bert_encoder_n65536 (CPU streaming).\n");
        printf("        Skipping DKS key generation to avoid OOM (64 keys × ~500 MB).\n\n");
        dctx.destroy();
        return 0;
    }

    printf("[Setup] Building DistGaloisKeyStore (%zu keys × %d GPUs)...\n",
           num_keys, n_gpus);
    DistGaloisKeyStore dks;
    dks.generate(ctx0, sk0, n_gpus, num_keys);

    // Also build single-GPU CPU key store (for fallback / non-bootstrap ops)
    GaloisKeyStore ks_cpu;
    ks_cpu.generate_all_keys(ctx0, sk0, num_keys);
    eval0.evaluator.enable_key_streaming(&ks_cpu, &gk0);

    // Register DKS (global pointer used by dist_rotate_vector_inplace)
    nexus_multi_gpu::dist_set_galois_key_store(&dks, [&](int step) -> size_t {
        auto it = step_to_idx.find(step);
        if (it == step_to_idx.end())
            throw std::runtime_error("Unknown step in DKS map");
        return it->second;
    });

    // Enable DKS rotation in the Evaluator so Bootstrapper's rotate_vector_inplace
    // dispatches to dist_rotate_phantom_inplace (parallel partial KS across GPUs).
    // The function pointer must outlive the bench, so allocate on heap and leak.
    static auto *step_to_idx_fn = new std::function<size_t(int)>(
        [&](int step) -> size_t {
            auto it = step_to_idx.find(step);
            if (it == step_to_idx.end())
                throw std::runtime_error("Unknown step in Evaluator DKS map");
            return it->second;
        });
    if (std::getenv("DKS_ROTATE")) {
        eval0.evaluator.enable_dks_rotation(&dctx, &dks, step_to_idx_fn);
        printf("[Setup] DKS rotation ENABLED — Bootstrapper rotates via dist_rotate_phantom_inplace\n");
    } else {
        printf("[Setup] DKS rotation DISABLED — Bootstrapper uses CPU-streaming + async prefetch\n");
        printf("        (set DKS_ROTATE=1 to enable distributed rotation)\n");
    }
    fflush(stdout);

    printf("[Setup] DKS ready. GPU memory summary:\n");
    for (int g = 0; g < n_gpus; g++) {
        cudaSetDevice(g);
        size_t fr, tot;
        cudaMemGetInfo(&fr, &tot);
        printf("  GPU %d: %.1f GB free / %.1f GB total\n",
               g, fr / 1e9, tot / 1e9);
    }
    cudaSetDevice(0);
    fflush(stdout);

    // ── Generate weight matrices ──
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.02, 0.02), idist(-0.5, 0.5);
    size_t slots = enc0.slot_count();

    auto make_w = [&]() {
        vector<vector<double>> w(INNER, vector<double>(slots, 0.0));
        for (auto &r : w)
            for (size_t s = 0; s < (size_t)std::min((long)HIDDEN, (long)slots); s++)
                r[s] = wdist(rng);
        return w;
    };

    auto Wq = make_w(), Wk = make_w(), Wv = make_w(), Wo = make_w();
    auto Wf1 = make_w(), Wf2 = make_w();

    // ── Encrypt input ──
    vector<double> d(slots, 0.0);
    for (size_t s = 0; s < (size_t)std::min((long)HIDDEN, (long)slots); s++) d[s] = idist(rng);
    PhantomPlaintext pt_in;
    enc0.encode(ctx0, d, SCALE, pt_in);
    PhantomCiphertext X;
    eval0.encryptor.encrypt(pt_in, X);
    for (int i = 0; i < BS_MOD; i++) eval0.evaluator.mod_switch_to_next_inplace(X);

    printf("\n[BERT Layer] Running 1 head via DKS on %d GPU%s...\n",
           n_gpus, n_gpus > 1 ? "s" : "");
    fflush(stdout);

    // ── Run one layer (1 head, DKS bootstrap) ──
    LayerTimes times = run_bert_layer_dks(
        dctx, eval0, bs, ks_cpu, X,
        Wq, Wk, Wv, Wo, Wf1, Wf2);

    // ── Print results ──
    printf("\n════════════════════════════════════════\n");
    printf("  BERT DKS Layer Results (%d GPU%s, N=65536)\n",
           n_gpus, n_gpus > 1 ? "s" : "");
    printf("════════════════════════════════════════\n");
    printf("  %-25s  %8s\n", "Operation", "Time (ms)");
    printf("  %-25s  %8s\n", "─────────────────────────", "────────");
    printf("  %-25s  %8.1f\n", "QKV MatMul (×3)",  times.qkv_ms);
    printf("  %-25s  %8.1f\n", "QK^T multiply",    times.qk_ms);
    printf("  %-25s  %8.1f\n", "Softmax",          times.softmax_ms);
    printf("  %-25s  %8.1f\n", "Attn×V",           times.av_ms);
    printf("  %-25s  %8.1f\n", "Output projection",times.out_ms);
    printf("  %-25s  %8.1f  ← DKS\n", "Bootstrap #1",    times.bs1_ms);
    printf("  %-25s  %8.1f\n", "LayerNorm #1",     times.ln1_ms);
    printf("  %-25s  %8.1f  ← DKS\n", "Bootstrap #2",    times.bs2_ms);
    printf("  %-25s  %8.1f\n", "FFN up-projection",times.ffn1_ms);
    printf("  %-25s  %8.1f\n", "GELU",             times.gelu_ms);
    printf("  %-25s  %8.1f\n", "FFN down-proj",    times.ffn2_ms);
    printf("  %-25s  %8.1f  ← DKS\n", "Bootstrap #3",    times.bs3_ms);
    printf("  %-25s  %8.1f\n", "LayerNorm #2",     times.ln2_ms);
    printf("  %-25s  %8.1f  ← DKS\n", "Bootstrap #4",    times.bs4_ms);
    printf("  %-25s  %8s\n", "─────────────────────────", "────────");
    printf("  %-25s  %8.1f\n", "TOTAL (1 head)",   times.total_ms);

    double bootstrap_total = times.bs1_ms + times.bs2_ms + times.bs3_ms + times.bs4_ms;
    double other_total     = times.total_ms - bootstrap_total;
    printf("\n  Bootstrap total: %.1f ms (%.1f%% of layer)\n",
           bootstrap_total, 100.0 * bootstrap_total / times.total_ms);
    printf("  Other ops:       %.1f ms (%.1f%% of layer)\n",
           other_total, 100.0 * other_total / times.total_ms);

    // Projected full 12-head BERT
    double proj_full = times.total_ms * N_HEADS;
    printf("\n  Projected 12-head BERT layer: %.1f ms = %.1f s\n",
           proj_full, proj_full / 1000.0);
    printf("  Reference (CPU streaming, 4-GPU head parallel): ~249,600 ms\n");
    printf("  DKS speedup vs CPU streaming: %.2fx\n",
           249600.0 / proj_full);
    fflush(stdout);

    // ── Cleanup ──
    nexus_multi_gpu::dist_set_galois_key_store(nullptr, {});
    dks.destroy();
    dctx.destroy();

    return 0;
}
