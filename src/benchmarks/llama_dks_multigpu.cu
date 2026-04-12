/**
 * llama_dks_multigpu.cu
 *
 * multiNEXUS Step 12 — LLaMA Decoder Layer with Distributed Key-Switching.
 *
 * Structural extension of bert_dks_multigpu.cu. Same DKS bootstrap
 * infrastructure; different layer topology:
 *
 * BERT layer:  QKV → QK^T → Softmax → Attn×V → OutProj →
 *              BS#1 → LayerNorm → BS#2 →
 *              FFN-up → GELU → FFN-down →
 *              BS#3 → LayerNorm → BS#4
 *
 * LLaMA layer: QKV → RoPE(Q,K) → QK^T → Softmax → Attn×V → OutProj →
 *              BS#1 → RMSNorm → BS#2 →
 *              SwiGLU(gate,up) → gate⊙up → FFN-down →
 *              BS#3 → RMSNorm → BS#4
 *
 * Key differences vs BERT:
 *   • RoPE: 2 extra ct×pt multiplies + 2 rotations per layer on Q and K
 *   • SwiGLU: 3 matmuls (gate/up/down) + SiLU(gate) + ct×ct multiply
 *             vs BERT: 2 matmuls (up/down) + GELU
 *   • RMSNorm: skips mean-subtract; same depth as LayerNorm.
 *              Reused LNEvaluator as same-depth proxy.
 *   • SiLU activation: same polynomial depth as GELU. GELUEvaluator as proxy.
 *
 * Expected per-layer time: ~90 s (DKS, 4×H100), vs ~85 s for BERT.
 * The ~5 s overhead comes from: RoPE (2 rotations) + extra SwiGLU matmul
 * + gate⊙up ct×ct multiply.
 *
 * Usage:
 *   srun --ntasks=1 --gres=gpu:4 ./build/bin/llama_dks_multigpu 4
 */

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <stdexcept>
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
struct LlamaTimer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    void start() { t0 = clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

// ---------------------------------------------------------------------------
// LLaMA / crypto parameters
// Identical crypto params to BERT; same parameter set means same bootstrap depth.
// ---------------------------------------------------------------------------
static const int N_HEADS  = 12;
static const int HIDDEN   = 768;
static const int INNER    = 64;
static const int SEQ_LEN  = 128;
static const int FFN_DIM  = 3072;  // SwiGLU typically 4×hidden; same as BERT FFN
static const long LOG_N   = 16;
static const long LOGN    = LOG_N - 2;
static const long LOGNH   = LOG_N - 1;
static const int LOGP     = 46;
static const int LOGQ     = 51;
static const int LOG_SPEC = 51;
static const int MAIN_MOD = 21;
static const int BS_MOD   = 14;
static const int TOT_LVL  = MAIN_MOD + BS_MOD;

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
// Per-operation timing for one LLaMA decoder layer
// ---------------------------------------------------------------------------
struct LlamaLayerTimes {
    double qkv_ms = 0;
    double rope_ms = 0;   // RoPE rotations on Q and K
    double qk_ms = 0;
    double softmax_ms = 0;
    double av_ms = 0;
    double out_ms = 0;
    double bs1_ms = 0;
    double rms1_ms = 0;   // RMSNorm (proxy: LNEvaluator)
    double bs2_ms = 0;
    double ffn_gate_ms = 0;   // gate projection
    double ffn_silu_ms = 0;   // SiLU(gate) (proxy: GELU)
    double ffn_up_ms = 0;     // up projection
    double gate_up_ms = 0;    // gate⊙up ct×ct multiply
    double ffn_down_ms = 0;
    double bs3_ms = 0;
    double rms2_ms = 0;
    double bs4_ms = 0;
    double total_ms = 0;
};

// ---------------------------------------------------------------------------
// run_llama_layer_dks
// ---------------------------------------------------------------------------
static LlamaLayerTimes run_llama_layer_dks(
    DistributedContext  &dctx,
    CKKSEvaluator       &eval,
    Bootstrapper        &bs,
    GaloisKeyStore      &ks,
    PhantomCiphertext   &X,
    const vector<vector<double>> &Wq,
    const vector<vector<double>> &Wk,
    const vector<vector<double>> &Wv,
    const vector<vector<double>> &Wo,
    const vector<vector<double>> &Wgate,
    const vector<vector<double>> &Wup,
    const vector<vector<double>> &Wdown,
    const vector<double>         &rope_cos,  // RoPE cosine factor (plaintext)
    const vector<double>         &rope_sin)  // RoPE sine factor  (plaintext)
{
    LlamaLayerTimes times;
    LlamaTimer t;

    auto sync_all = [&]() {
        for (int g = 0; g < dctx.n_gpus(); g++) {
            cudaSetDevice(g);
            cudaDeviceSynchronize();
        }
        cudaSetDevice(0);
    };

    auto dks_bootstrap = [&](PhantomCiphertext &ct, const string &label, double &out_ms) {
        cudaDeviceSynchronize();
        t.start();
        PhantomCiphertext bs_out;
        bs.bootstrap_3(bs_out, ct);
        ct = bs_out;
        cudaDeviceSynchronize();
        out_ms += t.elapsed_ms();
        if (!label.empty())
            printf("    %s: %.1f ms\n", label.c_str(), out_ms);
    };

    MMEvaluator   mme(eval);
    GELUEvaluator ge(eval);   // used as SiLU proxy
    SoftmaxEvaluator se(eval);
    LNEvaluator  lne(eval);   // used as RMSNorm proxy

    // ─── Self-Attention ───
    t.start();
    vector<PhantomCiphertext> Xi = {X}, Qv, Kv, Vv;
    auto Wq_nc = Wq; auto Wk_nc = Wk; auto Wv_nc = Wv;
    mme.matrix_mul_unified(Xi, Wq_nc, 1, Qv);
    mme.matrix_mul_unified(Xi, Wk_nc, 1, Kv);
    mme.matrix_mul_unified(Xi, Wv_nc, 1, Vv);
    PhantomCiphertext Q = Qv[0], K = Kv[0], V = Vv[0];
    cudaDeviceSynchronize();
    times.qkv_ms = t.elapsed_ms();

    // ─── RoPE on Q and K ───
    // RoPE: Q' = Q⊙cos(mθ) + rot(Q)⊙sin(mθ)
    //       K' = K⊙cos(mθ) + rot(K)⊙sin(mθ)
    // Each needs 1 pt×ct multiply, 1 rotation, 1 pt×ct multiply, 1 ct+ct add.
    // We time this separately to show the RoPE overhead vs BERT.
    t.start();
    {
        PhantomContext &ctx0 = dctx.context(0);
        // Encode RoPE factors as plaintexts
        PhantomCKKSEncoder enc(ctx0);
        double scale = Q.scale();
        size_t slots = enc.slot_count();

        // Clamp rope_cos/sin to slot count
        vector<double> c(slots, 0.0), s(slots, 0.0);
        for (size_t i = 0; i < std::min(rope_cos.size(), slots); i++) c[i] = rope_cos[i];
        for (size_t i = 0; i < std::min(rope_sin.size(), slots); i++) s[i] = rope_sin[i];

        PhantomPlaintext pt_cos, pt_sin;
        enc.encode(ctx0, c, scale, pt_cos);
        enc.encode(ctx0, s, scale, pt_sin);

        // Apply RoPE to Q
        auto rope_apply = [&](PhantomCiphertext &ct) {
            // ct_cos = ct * cos_pt
            PhantomCiphertext ct_cos, ct_sin, ct_rot;
            eval.evaluator.mod_switch_to_inplace(pt_cos, ct.chain_index());
            eval.evaluator.multiply_plain(ct, pt_cos, ct_cos);
            eval.evaluator.rescale_to_next_inplace(ct_cos);

            // ct_rot = rotate(ct, INNER), then * sin_pt
            // Rotation by half the head dim is the standard RoPE rotation
            ct_rot = ct;
            eval.evaluator.rotate_vector_inplace(ct_rot, INNER / 2, *eval.galois_keys);
            eval.evaluator.mod_switch_to_inplace(pt_sin, ct_rot.chain_index());
            eval.evaluator.multiply_plain(ct_rot, pt_sin, ct_sin);
            eval.evaluator.rescale_to_next_inplace(ct_sin);

            // ct' = ct_cos + ct_sin  (with scale alignment)
            eval.evaluator.mod_switch_to_inplace(ct_sin, ct_cos.chain_index());
            ct_sin.set_scale(ct_cos.scale());
            eval.evaluator.add_inplace(ct_cos, ct_sin);
            ct = ct_cos;
        };

        rope_apply(Q);
        rope_apply(K);
    }
    cudaDeviceSynchronize();
    times.rope_ms = t.elapsed_ms();

    // ─── QK^T ───
    t.start();
    PhantomCiphertext QK = Q;
    eval.evaluator.mod_switch_to_inplace(K, QK.chain_index());
    K.set_scale(QK.scale());
    eval.evaluator.multiply_inplace(QK, K);
    eval.evaluator.relinearize_inplace(QK, *eval.relin_keys);
    eval.evaluator.rescale_to_next_inplace(QK);
    cudaDeviceSynchronize();
    times.qk_ms = t.elapsed_ms();

    // ─── Softmax ───
    t.start();
    PhantomCiphertext attn;
    se.softmax(QK, attn, SEQ_LEN);
    cudaDeviceSynchronize();
    times.softmax_ms = t.elapsed_ms();

    // ─── Attention × V ───
    t.start();
    PhantomCiphertext AV = attn;
    eval.evaluator.mod_switch_to_inplace(V, AV.chain_index());
    V.set_scale(AV.scale());
    eval.evaluator.multiply_inplace(AV, V);
    eval.evaluator.relinearize_inplace(AV, *eval.relin_keys);
    eval.evaluator.rescale_to_next_inplace(AV);
    cudaDeviceSynchronize();
    times.av_ms = t.elapsed_ms();

    // ─── Output projection ───
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

    // ─── RMSNorm #1 (LNEvaluator as proxy) ───
    t.start();
    PhantomCiphertext rms1_out;
    lne.layer_norm(proj, rms1_out, HIDDEN);
    cudaDeviceSynchronize();
    times.rms1_ms = t.elapsed_ms();

    // ─── Bootstrap #2 ───
    while (rms1_out.coeff_modulus_size() > 1) eval.evaluator.mod_switch_to_next_inplace(rms1_out);
    dks_bootstrap(rms1_out, "BS2", times.bs2_ms);

    // ─── SwiGLU FFN ───
    // Gate projection + SiLU (GELUEvaluator as proxy for SiLU depth)
    t.start();
    vector<PhantomCiphertext> rms1v = {rms1_out}, gatev;
    auto Wgate_nc = Wgate;
    mme.matrix_mul_unified(rms1v, Wgate_nc, 1, gatev);
    PhantomCiphertext gate_ct = gatev[0];
    cudaDeviceSynchronize();
    times.ffn_gate_ms = t.elapsed_ms();

    t.start();
    PhantomCiphertext silu_out;
    ge.gelu(gate_ct, silu_out);  // SiLU proxy
    cudaDeviceSynchronize();
    times.ffn_silu_ms = t.elapsed_ms();

    // Up projection
    t.start();
    vector<PhantomCiphertext> upv;
    auto Wup_nc = Wup;
    mme.matrix_mul_unified(rms1v, Wup_nc, 1, upv);
    PhantomCiphertext up_ct = upv[0];
    cudaDeviceSynchronize();
    times.ffn_up_ms = t.elapsed_ms();

    // gate ⊙ up (ct × ct)
    t.start();
    eval.evaluator.mod_switch_to_inplace(up_ct, silu_out.chain_index());
    up_ct.set_scale(silu_out.scale());
    eval.evaluator.multiply_inplace(silu_out, up_ct);
    eval.evaluator.relinearize_inplace(silu_out, *eval.relin_keys);
    eval.evaluator.rescale_to_next_inplace(silu_out);
    cudaDeviceSynchronize();
    times.gate_up_ms = t.elapsed_ms();

    // Down projection
    t.start();
    vector<PhantomCiphertext> silu_v = {silu_out}, downv;
    auto Wdown_nc = Wdown;
    mme.matrix_mul_unified(silu_v, Wdown_nc, 1, downv);
    PhantomCiphertext ffn_out = downv[0];
    cudaDeviceSynchronize();
    times.ffn_down_ms = t.elapsed_ms();

    // ─── Bootstrap #3 ───
    while (ffn_out.coeff_modulus_size() > 1) eval.evaluator.mod_switch_to_next_inplace(ffn_out);
    dks_bootstrap(ffn_out, "BS3", times.bs3_ms);

    // ─── RMSNorm #2 ───
    t.start();
    PhantomCiphertext rms2_out;
    lne.layer_norm(ffn_out, rms2_out, HIDDEN);
    cudaDeviceSynchronize();
    times.rms2_ms = t.elapsed_ms();

    // ─── Bootstrap #4 ───
    while (rms2_out.coeff_modulus_size() > 1) eval.evaluator.mod_switch_to_next_inplace(rms2_out);
    dks_bootstrap(rms2_out, "BS4", times.bs4_ms);

    X = rms2_out;

    times.total_ms =
        times.qkv_ms + times.rope_ms + times.qk_ms + times.softmax_ms +
        times.av_ms + times.out_ms +
        times.bs1_ms + times.rms1_ms + times.bs2_ms +
        times.ffn_gate_ms + times.ffn_silu_ms + times.ffn_up_ms +
        times.gate_up_ms + times.ffn_down_ms +
        times.bs3_ms + times.rms2_ms + times.bs4_ms;

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
    printf("║   multiNEXUS LLaMA DKS — N=65536, %d Heads, %d GPUs      ║\n",
           N_HEADS, n_gpus);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    auto parms = build_parms();
    double SCALE = pow(2.0, LOGP);

    // ── Create DistributedContext FIRST (resets Phantom's global stream) ──
    cudaSetDevice(0);
    DistributedContext dctx = DistributedContext::create(parms, n_gpus);

    // ── Crypto objects from dctx.context(0) ──
    PhantomContext   &ctx0 = dctx.context(0);
    PhantomCKKSEncoder enc0(ctx0);
    PhantomSecretKey sk0(ctx0);
    PhantomPublicKey pk0  = sk0.gen_publickey(ctx0);
    PhantomRelinKey  rk0  = sk0.gen_relinkey(ctx0);
    PhantomGaloisKey gk0;
    CKKSEvaluator eval0(&ctx0, &pk0, &sk0, &enc0, &rk0, &gk0, SCALE);

    // ── Bootstrapper ──
    Bootstrapper bs(10, LOGN, LOGNH, TOT_LVL, SCALE, 25, 59, 2, 1, &eval0);
    bs.slot_vec.push_back(LOGN);
    bs.prepare_mod_polynomial();
    bs.generate_LT_coefficient_3();

    // ── Rotation steps ──
    vector<int> steps;
    steps.push_back(0);
    for (int i = 0; i < LOG_N - 1; i++) steps.push_back(1 << i);
    for (int i = 0; i < LOG_N - 1; i++) steps.push_back(-(1 << i));
    steps.push_back(-SEQ_LEN);
    steps.push_back(-HIDDEN);
    steps.push_back(INNER / 2);   // RoPE rotation step
    bs.addLeftRotKeys_Linear_to_vector_3(steps);

    {
        std::set<int> step_set(steps.begin(), steps.end());
        steps.assign(step_set.begin(), step_set.end());
    }
    long N_val = 1L << LOG_N;
    auto all_elts = ::get_elts_from_steps(steps, static_cast<size_t>(N_val));

    ctx0.setup_galois_tool(all_elts);
    gk0.resize_slots(all_elts.size());
    size_t num_keys = all_elts.size();

    auto &gelts = ctx0.key_galois_tool()->galois_elts();
    map<int, size_t> step_to_idx;
    for (size_t i = 0; i < steps.size(); i++) {
        uint32_t elt = ctx0.key_galois_tool()->get_elt_from_step(steps[i]);
        auto it2 = std::find(gelts.begin(), gelts.end(), elt);
        if (it2 != gelts.end())
            step_to_idx[steps[i]] = static_cast<size_t>(std::distance(gelts.begin(), it2));
    }

    // ── DKS key store ──
    if (n_gpus == 1) {
        printf("[Setup] n_gpus=1: DKS has no sharding benefit. "
               "Using CPU-streaming reference instead.\n\n");
        dctx.destroy();
        return 0;
    }

    printf("[Setup] Building DistGaloisKeyStore (%zu keys × %d GPUs)...\n",
           num_keys, n_gpus);
    DistGaloisKeyStore dks;
    dks.generate(ctx0, sk0, n_gpus, num_keys);

    GaloisKeyStore ks_cpu;
    ks_cpu.generate_all_keys(ctx0, sk0, num_keys);
    eval0.evaluator.enable_key_streaming(&ks_cpu, &gk0);

    nexus_multi_gpu::dist_set_galois_key_store(&dks, [&](int step) -> size_t {
        auto it = step_to_idx.find(step);
        if (it == step_to_idx.end())
            throw std::runtime_error("Unknown step: " + std::to_string(step));
        return it->second;
    });

    printf("[Setup] DKS ready. GPU memory:\n");
    for (int g = 0; g < n_gpus; g++) {
        cudaSetDevice(g);
        size_t fr, tot;
        cudaMemGetInfo(&fr, &tot);
        printf("  GPU %d: %.1f GB free / %.1f GB total\n", g, fr / 1e9, tot / 1e9);
    }
    cudaSetDevice(0);

    // ── Weight matrices ──
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
    auto Wgate = make_w(), Wup = make_w(), Wdown = make_w();

    // RoPE factors (random; in practice computed from position index)
    vector<double> rope_cos(slots, 1.0), rope_sin(slots, 0.0);
    for (size_t i = 0; i < slots; i++) {
        double theta = (i < (size_t)INNER) ? (double)i * 0.01 : 0.0;
        rope_cos[i] = cos(theta);
        rope_sin[i] = sin(theta);
    }

    // ── Encrypt input ──
    vector<double> d(slots, 0.0);
    for (size_t s = 0; s < (size_t)std::min((long)HIDDEN, (long)slots); s++) d[s] = idist(rng);
    PhantomPlaintext pt_in;
    enc0.encode(ctx0, d, SCALE, pt_in);
    PhantomCiphertext X;
    eval0.encryptor.encrypt(pt_in, X);
    for (int i = 0; i < BS_MOD; i++) eval0.evaluator.mod_switch_to_next_inplace(X);

    printf("\n[LLaMA Layer] Running 1 head via DKS on %d GPU%s...\n",
           n_gpus, n_gpus > 1 ? "s" : "");

    LlamaLayerTimes times = run_llama_layer_dks(
        dctx, eval0, bs, ks_cpu, X,
        Wq, Wk, Wv, Wo, Wgate, Wup, Wdown, rope_cos, rope_sin);

    // ── Print results ──
    printf("\n════════════════════════════════════════════════════════\n");
    printf("  LLaMA DKS Layer Results (%d GPU%s, N=65536)\n",
           n_gpus, n_gpus > 1 ? "s" : "");
    printf("════════════════════════════════════════════════════════\n");
    printf("  %-28s  %8s  %s\n", "Operation", "Time (ms)", "Notes");
    printf("  %-28s  %8s\n", "────────────────────────────", "────────");
    printf("  %-28s  %8.1f\n", "QKV MatMul (×3)",       times.qkv_ms);
    printf("  %-28s  %8.1f  ← LLaMA: not in BERT\n", "RoPE (Q+K rotations)",   times.rope_ms);
    printf("  %-28s  %8.1f\n", "QK^T multiply",          times.qk_ms);
    printf("  %-28s  %8.1f\n", "Softmax",                times.softmax_ms);
    printf("  %-28s  %8.1f\n", "Attn×V",                 times.av_ms);
    printf("  %-28s  %8.1f\n", "Output projection",      times.out_ms);
    printf("  %-28s  %8.1f  ← DKS\n", "Bootstrap #1",            times.bs1_ms);
    printf("  %-28s  %8.1f  ← LLaMA: RMSNorm proxy\n", "RMSNorm #1",            times.rms1_ms);
    printf("  %-28s  %8.1f  ← DKS\n", "Bootstrap #2",            times.bs2_ms);
    printf("  %-28s  %8.1f  ← LLaMA: extra vs BERT\n", "FFN gate projection",    times.ffn_gate_ms);
    printf("  %-28s  %8.1f  ← LLaMA: SiLU proxy\n", "SiLU(gate)",             times.ffn_silu_ms);
    printf("  %-28s  %8.1f\n", "FFN up projection",      times.ffn_up_ms);
    printf("  %-28s  %8.1f  ← LLaMA: extra vs BERT\n", "gate⊙up (ct×ct)",        times.gate_up_ms);
    printf("  %-28s  %8.1f\n", "FFN down projection",    times.ffn_down_ms);
    printf("  %-28s  %8.1f  ← DKS\n", "Bootstrap #3",            times.bs3_ms);
    printf("  %-28s  %8.1f  ← LLaMA: RMSNorm proxy\n", "RMSNorm #2",            times.rms2_ms);
    printf("  %-28s  %8.1f  ← DKS\n", "Bootstrap #4",            times.bs4_ms);
    printf("  %-28s  %8s\n", "────────────────────────────", "────────");
    printf("  %-28s  %8.1f\n", "TOTAL (1 head)",          times.total_ms);

    double bs_total  = times.bs1_ms + times.bs2_ms + times.bs3_ms + times.bs4_ms;
    double llama_extra = times.rope_ms + times.ffn_gate_ms + times.gate_up_ms;
    printf("\n  Bootstrap total:          %.1f ms (%.1f%% of layer)\n",
           bs_total, 100.0 * bs_total / times.total_ms);
    printf("  LLaMA overhead vs BERT:   %.1f ms (RoPE + extra matmul + gate⊙up)\n",
           llama_extra);

    double proj_12head = times.total_ms * N_HEADS;
    printf("\n  Projected 12-head LLaMA layer: %.1f ms = %.1f s\n",
           proj_12head, proj_12head / 1000.0);
    printf("  Reference BERT DKS (4-GPU):    ~85,000 ms\n");
    printf("  LLaMA overhead vs BERT DKS:    %.1f ms = %.1f%%\n",
           proj_12head - 85000.0,
           100.0 * (proj_12head - 85000.0) / 85000.0);

    // ── Cleanup ──
    nexus_multi_gpu::dist_set_galois_key_store(nullptr, {});
    dks.destroy();
    dctx.destroy();

    return 0;
}
