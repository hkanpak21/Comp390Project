/**
 * llama_layer_baseline.cu  (PRD slice H1)
 *
 * Single-GPU LLaMA-7B decoder layer baseline at N=65,536.
 *
 * LLaMA-7B model dimensions:
 *   hidden_dim = 4096
 *   num_heads  = 32        (head_dim = hidden / heads = 128)
 *   num_layers = 32        (we run 1 for baseline)
 *   ffn_dim    = 11008     (SwiGLU; we approximate SiLU with GELU evaluator)
 *   seq_len    = 128       (default; configurable via CLI)
 *
 * What this benchmark measures:
 *   ONE LLaMA decoder layer (single head's worth of compute) end-to-end on
 *   ONE GPU, with CPU-streaming Galois keys (~62 GB key store does not fit
 *   on a single H100).  Mirrors the bert_encoder_n65536 single-GPU pattern
 *   but with LLaMA-7B per-head dimensions (head_dim = 128 vs BERT 64).
 *
 *   The structural cost difference vs BERT-base per-head:
 *     - QKV/QK/AV/Out matmul rows  : 128 vs 64   (≈2× more multiply_plain)
 *     - FFN gate/up/down rows      : 128 vs 64   (each, plus extra gate matmul)
 *     - RoPE                       : 2 ct×pt + 2 rotations on Q,K
 *     - SwiGLU                     : 3 matmuls + ct×ct gate⊙up
 *
 * Output: per-component breakdown (ms) and TOTAL per-layer time. Used as the
 * reference for HP-LLaMA (slice H2) and downstream LLaMA scaling work.
 *
 * Usage:
 *   ./build/bin/llama_layer_baseline             # default seq_len = 128
 *   ./build/bin/llama_layer_baseline --seq-len S
 *   ./build/bin/llama_layer_baseline --inner I   # head_dim override (default 128)
 *
 * Acceptance (PRD H1):
 *   "Single-GPU baseline measured" → file builds, runs one layer end-to-end,
 *   prints per-component timings; per-layer total recorded for downstream
 *   slices (H2..H6).
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>
#include <set>

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

static void print_mem(const char *label) {
    size_t free_m, total_m;
    cudaMemGetInfo(&free_m, &total_m);
    printf("[Mem] %-32s %.2f / %.2f GB used\n", label,
           (total_m - free_m) / (1024.0 * 1024.0 * 1024.0),
           total_m / (1024.0 * 1024.0 * 1024.0));
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// Per-layer timing record
// ---------------------------------------------------------------------------
struct LayerTimes {
    // attention
    double qkv = 0, rope = 0, qk = 0, softmax = 0, av = 0, out = 0;
    // bs/norm 1
    double bs1 = 0, rms1 = 0, bs2 = 0;
    // SwiGLU FFN
    double ffn_gate = 0, ffn_up = 0, ffn_silu = 0, ffn_gate_mul_up = 0, ffn_down = 0;
    // bs/norm 2
    double bs3 = 0, rms2 = 0, bs4 = 0;
    double total() const {
        return qkv + rope + qk + softmax + av + out +
               bs1 + rms1 + bs2 +
               ffn_gate + ffn_up + ffn_silu + ffn_gate_mul_up + ffn_down +
               bs3 + rms2 + bs4;
    }
};

// Time a block: cudaDeviceSynchronize, run code, sync again, accumulate ms.
// Variadic so commas inside { ... } code blocks don't trip the preprocessor.
#define TIME(field, ...) do {                                             \
    cudaDeviceSynchronize();                                              \
    PerfTimer _pt; _pt.start();                                           \
    __VA_ARGS__;                                                          \
    cudaDeviceSynchronize();                                              \
    times.field += _pt.elapsed_ms();                                      \
} while (0)

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    // ── LLaMA-7B parameters ──
    // Per-head compute width (head_dim) drives the matmul "inner" size in the
    // matrix_mul_unified API: inner_dim == weights.size(), one multiply_plain
    // per output row. For LLaMA-7B head_dim is 128 (BERT was 64).
    int inner   = 128;            // head_dim (LLaMA-7B)
    int hidden  = 4096;           // model hidden dim (only used to clamp slot fill)
    int seq_len = 128;            // sequence length (default per H1)
    int n_heads = 32;             // for projection only; we run 1 head baseline

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seq-len") && i+1 < argc) seq_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) inner = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hidden") && i+1 < argc) hidden = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads") && i+1 < argc) n_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: %s [--seq-len N] [--inner D] [--hidden H] [--heads K]\n",
                   argv[0]);
            return 0;
        }
    }

    printf("════════════════════════════════════════════════════════════\n");
    printf("  LLaMA-7B Layer Baseline (1 GPU, N=65536, CPU key streaming)\n");
    printf("  hidden=%d  heads=%d  head_dim(inner)=%d  seq_len=%d\n",
           hidden, n_heads, inner, seq_len);
    printf("  FFN=SwiGLU (3 matmuls + ct×ct), Norm=RMSNorm-proxy(LN), Act=SiLU-proxy(GELU)\n");
    printf("  RoPE: 2 rotations + 2 ct×pt on Q,K\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    cudaSetDevice(0);
    print_mem("Initial");

    // ── CKKS parameters (identical to BERT champion: N=65536, MAIN+BS=21+14) ──
    long logN = 16;
    long logn = logN - 2;        // sparse slots
    long logNh = logN - 1;
    size_t N = 1ULL << logN;
    long sparse_slots_val = 1L << logn;
    int logp = 46, logq = 51, log_special = 51;
    int main_mod = 21, bs_mod = 14;
    int total_level = main_mod + bs_mod;
    long boundary_K = 25, deg = 59, scale_factor = 2, inverse_deg = 1, loge = 10;
    double SCALE = pow(2.0, logp);

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

    // ── Crypto setup ──
    PhantomContext context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey sk(context);
    PhantomPublicKey pk = sk.gen_publickey(context);
    PhantomRelinKey  rk = sk.gen_relinkey(context);
    PhantomGaloisKey gk;          // populated via key streaming
    size_t slots = encoder.slot_count();

    CKKSEvaluator eval(&context, &pk, &sk, &encoder, &rk, &gk, SCALE);
    print_mem("After context + PK + RK");

    // ── Random LLaMA-7B per-head weight matrices ──
    // Each weight matrix has `inner` rows × `slots` cols (matrix_mul_unified
    // does `inner` multiply_plain + accumulate per output column).
    // For LLaMA-7B per-head: Wq/Wk/Wv/Wo, plus SwiGLU Wgate/Wup/Wdown.
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.02, 0.02), idist(-0.5, 0.5);
    auto make_w = [&]() {
        vector<vector<double>> w(inner, vector<double>(slots, 0.0));
        for (auto &r : w)
            for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++)
                r[s] = wdist(rng);
        return w;
    };
    auto W_q    = make_w();
    auto W_k    = make_w();
    auto W_v    = make_w();
    auto W_o    = make_w();
    auto W_gate = make_w();
    auto W_up   = make_w();
    auto W_down = make_w();

    // RoPE plaintext factors (cos / sin) — per-slot mask, encoded inside the
    // RoPE TIME block to track the encode cost too.
    vector<double> rope_cos(slots, 0.0), rope_sin(slots, 0.0);
    for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++) {
        rope_cos[s] = cos(0.01 * s);
        rope_sin[s] = sin(0.01 * s);
    }

    // ── Encrypt one head's input ciphertext ──
    vector<double> d(slots, 0.0);
    for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++)
        d[s] = idist(rng);
    PhantomPlaintext pt_in;
    eval.encoder.encode(d, SCALE, pt_in);
    PhantomCiphertext X;
    eval.encryptor.encrypt(pt_in, X);
    for (int i = 0; i < bs_mod; i++) eval.evaluator.mod_switch_to_next_inplace(X);
    printf("[Setup] Encrypted 1 head, level=%zu\n", X.coeff_modulus_size());

    // ── Bootstrap setup ──
    PerfTimer setup_timer; setup_timer.start();
    printf("\n[Setup] Initializing bootstrapper...\n"); fflush(stdout);
    Bootstrapper bootstrapper(loge, logn, logNh, total_level, SCALE,
                              boundary_K, deg, scale_factor, inverse_deg, &eval);
    bootstrapper.slot_vec.push_back(logn);
    bootstrapper.prepare_mod_polynomial();
    bootstrapper.generate_LT_coefficient_3();

    vector<int> all_steps;
    all_steps.push_back(0);
    for (int i = 0; i < logN - 1; i++) all_steps.push_back(1 << i);
    for (int i = 0; i < logN - 1; i++) all_steps.push_back(-(1 << i));
    all_steps.push_back(-seq_len);
    all_steps.push_back(-hidden);
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(all_steps);

    {
        std::set<int> step_set(all_steps.begin(), all_steps.end());
        all_steps.assign(step_set.begin(), step_set.end());
    }
    printf("[Setup] Total unique rotation steps: %zu\n", all_steps.size());

    auto all_elts = ::get_elts_from_steps(all_steps, N);
    context.setup_galois_tool(all_elts);
    gk.resize_slots(all_elts.size());

    printf("[KeyStore] Generating %zu Galois keys to host RAM (CPU streaming)...\n",
           all_elts.size());
    GaloisKeyStore key_store;
    key_store.generate_all_keys(context, sk, all_elts.size());
    eval.evaluator.enable_key_streaming(&key_store, &gk);

    printf("[Setup] Total setup: %.2f s\n", setup_timer.elapsed_ms() / 1000.0);
    print_mem("After setup + key store");

    // ── Operator evaluators ──
    GELUEvaluator    lg(eval);   // proxy for SiLU
    SoftmaxEvaluator ls(eval);
    LNEvaluator      ll(eval);   // proxy for RMSNorm
    MMEvaluator      lm(eval);

    // ── Run ONE LLaMA decoder layer end-to-end ──
    printf("\n═══ Running one LLaMA-7B decoder layer (1 head, 1 GPU) ═══\n");
    fflush(stdout);

    LayerTimes times;
    PerfTimer layer_timer; layer_timer.start();

    // ─── Self-Attention: QKV projections ───
    vector<PhantomCiphertext> xi = {X}, q, k, v;
    TIME(qkv, {
        lm.matrix_mul_unified(xi, W_q, 1, q);
        lm.matrix_mul_unified(xi, W_k, 1, k);
        lm.matrix_mul_unified(xi, W_v, 1, v);
    });
    printf("[Layer] QKV done    %.1f ms\n", times.qkv); fflush(stdout);

    // ─── RoPE on Q and K ───
    // 2 rotations + 2 ct×pt + 2 add + 2 rescale per Q,K.
    // ORDERING NOTE (regression workaround, 2026-05-10): the rotations MUST
    // happen BEFORE encoding/mod-switching the rope masks. The reverse order
    // (encode/mod-switch first, then rotate) segfaults on Phantom + the
    // CPU-streaming key path at deep chain_index — likely a stream-state
    // interaction in the encode→mod_switch→rotate sequence. Rotations-first
    // is the same order used implicitly by Bootstrapper's interleaved
    // baby-step/giant-step rotations and works reliably.
    TIME(rope, {
        // Rotations first (consumes no level).
        PhantomCiphertext q_rot, k_rot;
        eval.evaluator.rotate_vector(q[0], 1, *eval.galois_keys, q_rot);
        eval.evaluator.rotate_vector(k[0], 1, *eval.galois_keys, k_rot);

        // Then ct×pt with rope cos/sin masks.
        PhantomPlaintext pt_cos, pt_sin;
        eval.encoder.encode(rope_cos, q[0].scale(),   pt_cos);
        eval.encoder.encode(rope_sin, q_rot.scale(),  pt_sin);
        eval.evaluator.mod_switch_to_inplace(pt_cos, q[0].chain_index());
        eval.evaluator.mod_switch_to_inplace(pt_sin, q_rot.chain_index());

        PhantomCiphertext q_sin, k_sin, q_rot_pt, k_rot_pt;
        eval.evaluator.multiply_plain(q[0],  pt_cos, q_sin);
        eval.evaluator.multiply_plain(q_rot, pt_sin, q_rot_pt);
        eval.evaluator.add_inplace(q_sin, q_rot_pt);
        eval.evaluator.rescale_to_next_inplace(q_sin);

        eval.evaluator.multiply_plain(k[0],  pt_cos, k_sin);
        eval.evaluator.multiply_plain(k_rot, pt_sin, k_rot_pt);
        eval.evaluator.add_inplace(k_sin, k_rot_pt);
        eval.evaluator.rescale_to_next_inplace(k_sin);
        q[0] = q_sin;
        k[0] = k_sin;
    });
    printf("[Layer] RoPE done   %.1f ms\n", times.rope); fflush(stdout);

    // ─── QK^T ───
    PhantomCiphertext qk_ct;
    TIME(qk, {
        eval.evaluator.mod_switch_to_inplace(k[0], q[0].chain_index());
        k[0].set_scale(q[0].scale());
        eval.evaluator.multiply(q[0], k[0], qk_ct);
        eval.evaluator.relinearize_inplace(qk_ct, *eval.relin_keys);
        eval.evaluator.rescale_to_next_inplace(qk_ct);
    });
    printf("[Layer] QK^T done   %.1f ms\n", times.qk); fflush(stdout);

    // ─── Softmax ───
    PhantomCiphertext attn;
    TIME(softmax, { ls.softmax(qk_ct, attn, seq_len); });
    printf("[Layer] Softmax     %.1f ms\n", times.softmax); fflush(stdout);

    // ─── Attn × V ───
    PhantomCiphertext av_ct;
    TIME(av, {
        eval.evaluator.mod_switch_to_inplace(v[0], attn.chain_index());
        v[0].set_scale(attn.scale());
        eval.evaluator.multiply(attn, v[0], av_ct);
        eval.evaluator.relinearize_inplace(av_ct, *eval.relin_keys);
        eval.evaluator.rescale_to_next_inplace(av_ct);
    });
    printf("[Layer] Attn×V      %.1f ms\n", times.av); fflush(stdout);

    // ─── Output projection ───
    vector<PhantomCiphertext> ai = {av_ct}, ao;
    TIME(out, { lm.matrix_mul_unified(ai, W_o, 1, ao); });
    printf("[Layer] OutProj     %.1f ms\n", times.out); fflush(stdout);

    // ─── Bootstrap #1 ───
    while (ao[0].coeff_modulus_size() > 1)
        eval.evaluator.mod_switch_to_next_inplace(ao[0]);
    PhantomCiphertext b1;
    TIME(bs1, { bootstrapper.bootstrap_3(b1, ao[0]); });
    printf("[Layer] BS#1        %.1f ms\n", times.bs1); fflush(stdout);

    // ─── RMSNorm #1 (LayerNorm proxy) ───
    PhantomCiphertext rms1_out;
    TIME(rms1, { ll.layer_norm(b1, rms1_out, hidden); });
    printf("[Layer] RMSNorm#1   %.1f ms\n", times.rms1); fflush(stdout);

    // ─── Bootstrap #2 ───
    while (rms1_out.coeff_modulus_size() > 1)
        eval.evaluator.mod_switch_to_next_inplace(rms1_out);
    PhantomCiphertext b2;
    TIME(bs2, { bootstrapper.bootstrap_3(b2, rms1_out); });
    printf("[Layer] BS#2        %.1f ms\n", times.bs2); fflush(stdout);

    // ─── SwiGLU FFN: gate / up / SiLU(gate) / gate⊙up / down ───
    vector<PhantomCiphertext> fi = {b2}, gate_out, up_out;
    TIME(ffn_gate, { lm.matrix_mul_unified(fi, W_gate, 1, gate_out); });
    printf("[Layer] FFN-gate    %.1f ms\n", times.ffn_gate); fflush(stdout);

    TIME(ffn_up, { lm.matrix_mul_unified(fi, W_up, 1, up_out); });
    printf("[Layer] FFN-up      %.1f ms\n", times.ffn_up); fflush(stdout);

    PhantomCiphertext silu_gate;
    TIME(ffn_silu, { lg.gelu(gate_out[0], silu_gate); });
    printf("[Layer] SiLU(gate)  %.1f ms\n", times.ffn_silu); fflush(stdout);

    PhantomCiphertext gated;
    TIME(ffn_gate_mul_up, {
        eval.evaluator.mod_switch_to_inplace(up_out[0], silu_gate.chain_index());
        up_out[0].set_scale(silu_gate.scale());
        eval.evaluator.multiply(silu_gate, up_out[0], gated);
        eval.evaluator.relinearize_inplace(gated, *eval.relin_keys);
        eval.evaluator.rescale_to_next_inplace(gated);
    });
    printf("[Layer] gate⊙up     %.1f ms\n", times.ffn_gate_mul_up); fflush(stdout);

    vector<PhantomCiphertext> di = {gated}, down_out;
    TIME(ffn_down, { lm.matrix_mul_unified(di, W_down, 1, down_out); });
    printf("[Layer] FFN-down    %.1f ms\n", times.ffn_down); fflush(stdout);

    // ─── Bootstrap #3 ───
    while (down_out[0].coeff_modulus_size() > 1)
        eval.evaluator.mod_switch_to_next_inplace(down_out[0]);
    PhantomCiphertext b3;
    TIME(bs3, { bootstrapper.bootstrap_3(b3, down_out[0]); });
    printf("[Layer] BS#3        %.1f ms\n", times.bs3); fflush(stdout);

    // ─── RMSNorm #2 ───
    PhantomCiphertext rms2_out;
    TIME(rms2, { ll.layer_norm(b3, rms2_out, hidden); });
    printf("[Layer] RMSNorm#2   %.1f ms\n", times.rms2); fflush(stdout);

    // ─── Bootstrap #4 ───
    while (rms2_out.coeff_modulus_size() > 1)
        eval.evaluator.mod_switch_to_next_inplace(rms2_out);
    PhantomCiphertext b4;
    TIME(bs4, { bootstrapper.bootstrap_3(b4, rms2_out); });
    printf("[Layer] BS#4        %.1f ms\n", times.bs4); fflush(stdout);

    double layer_ms = layer_timer.elapsed_ms();
    print_mem("After full layer");

    // ── Sanity: output should be at top of chain (close to full level) ──
    size_t out_level = b4.coeff_modulus_size();
    printf("\n[Layer] Output ciphertext level after BS#4: %zu (no crash, layer complete)\n",
           out_level);

    // ── Report ──
    double sum = times.total();
    double bs_total = times.bs1 + times.bs2 + times.bs3 + times.bs4;
    double mm_total = times.qkv + times.qk + times.av + times.out +
                      times.ffn_gate + times.ffn_up + times.ffn_down;
    double nl_total = times.softmax + times.ffn_silu + times.rms1 + times.rms2;
    double extras   = times.rope + times.ffn_gate_mul_up;

    auto row = [&](const char *name, double v) {
        printf("  %-22s  %9.1f ms   %5.1f%%\n",
               name, v, sum > 0 ? 100.0 * v / sum : 0.0);
    };

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  LLaMA-7B Single-GPU Layer Baseline (1 head, N=65536)\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Wall time (layer):      %.1f ms (%.2f s)\n",
           layer_ms, layer_ms / 1000.0);
    printf("  Sum of measured ops:    %.1f ms (%.2f s)\n",
           sum, sum / 1000.0);
    printf("\n─── Per-operation breakdown ───\n");
    row("QKV MatMul (×3)",        times.qkv);
    row("RoPE (Q,K)",             times.rope);
    row("Q*K^T",                  times.qk);
    row("Softmax",                times.softmax);
    row("Attn*V",                 times.av);
    row("Out MatMul",             times.out);
    row("Bootstrap #1",           times.bs1);
    row("RMSNorm #1 (LN proxy)",  times.rms1);
    row("Bootstrap #2",           times.bs2);
    row("FFN gate MatMul",        times.ffn_gate);
    row("FFN up MatMul",          times.ffn_up);
    row("SiLU(gate) (GELU prx)",  times.ffn_silu);
    row("gate ⊙ up (ct×ct)",      times.ffn_gate_mul_up);
    row("FFN down MatMul",        times.ffn_down);
    row("Bootstrap #3",           times.bs3);
    row("RMSNorm #2 (LN proxy)",  times.rms2);
    row("Bootstrap #4",           times.bs4);
    printf("  %-22s  %9.1f ms   100.0%%\n", "TOTAL (work)", sum);

    printf("\n─── Category rollup ───\n");
    printf("  Bootstraps (4×):           %9.1f ms  (%.1f%%)\n",
           bs_total, sum>0?100.0*bs_total/sum:0.0);
    printf("  MatMuls (7×):              %9.1f ms  (%.1f%%)\n",
           mm_total, sum>0?100.0*mm_total/sum:0.0);
    printf("  Non-linear (4×):           %9.1f ms  (%.1f%%)\n",
           nl_total, sum>0?100.0*nl_total/sum:0.0);
    printf("  LLaMA extras (RoPE+gate⊙): %9.1f ms  (%.1f%%)\n",
           extras, sum>0?100.0*extras/sum:0.0);

    // Projection: full LLaMA-7B latency from this single-head, single-layer measurement.
    // 32 heads × 32 layers, all sequential on 1 GPU (worst-case single-GPU baseline).
    double proj_layer_all_heads = layer_ms * (double)n_heads;
    double proj_full_model      = proj_layer_all_heads * 32.0;
    printf("\n─── Projection (single GPU, sequential heads/layers) ───\n");
    printf("  Per-head per-layer:    %.1f ms\n", layer_ms);
    printf("  Per-layer (×%d heads): %.1f ms = %.1f s\n",
           n_heads, proj_layer_all_heads, proj_layer_all_heads / 1000.0);
    printf("  Full LLaMA-7B (32 layers × %d heads): %.1f s = %.1f min\n",
           n_heads, proj_full_model / 1000.0, proj_full_model / 60000.0);
    printf("════════════════════════════════════════════════════════════\n");

    return 0;
}
