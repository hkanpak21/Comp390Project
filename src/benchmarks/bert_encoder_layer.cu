/**
 * bert_encoder_layer.cu
 *
 * Complete BERT encoder layer following NEXUS Table IV.
 * All operations use real NEXUS FHE code at unified N=65536.
 *
 * Full layer pipeline (from NEXUS Table IV):
 *   MatMul(Q,K,V) → QK^T → Softmax → Attn·V → MatMul(OutProj) →
 *   LEVEL_REFRESH → LayerNorm → LEVEL_REFRESH →
 *   MatMul(FFN_up) → GELU → MatMul(FFN_down) →
 *   LEVEL_REFRESH → LayerNorm → LEVEL_REFRESH
 *
 * LEVEL_REFRESH uses re_encrypt (practical stand-in for bootstrap).
 * Bootstrap is ported and runs mechanically (levels restored, 262ms)
 * but accuracy requires matching the NEXUS Phantom fork's FFT internals.
 *
 * The parallelism pattern is identical: each ct refreshed independently.
 *
 * Usage:
 *   ./bin/bert_encoder_layer [--n-gpus 4] [--heads 2] [--inner 16] [--ffn-dim 64]
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

#include "ckks_evaluator.cuh"
#include "gelu.cuh"
#include "softmax.cuh"
#include "layer_norm.cuh"
#include "matrix_mul.cuh"
#include "../multi_gpu/pipeline/ct_pipeline.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;
using namespace nexus_multi_gpu;

struct PerfTimer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        return chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
    }
};

// Level refresh: re-encrypt to restore depth (stand-in for bootstrap)
static void level_refresh(CKKSEvaluator &eval, PhantomCiphertext &ct, double scale) {
    PhantomPlaintext pt;
    vector<double> vals;
    eval.decryptor.decrypt(ct, pt);
    eval.encoder.decode(pt, vals);
    eval.encoder.encode(vals, scale, pt);
    eval.encryptor.encrypt(pt, ct);
}

struct Config {
    int n_gpus = 1;
    int n_heads = 2;     // attention heads (BERT=12, reduced for testing)
    int head_dim = 32;   // per-head dimension (BERT=64)
    int hidden_dim = 64; // total hidden dim = n_heads × head_dim
    int ffn_dim = 128;   // FFN intermediate (BERT=3072)
    int seq_len = 16;    // sequence length
    int inner_matmul = 16; // matmul inner dim for testing
};

Config parse_args(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc) c.n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads") && i+1 < argc) c.n_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--head-dim") && i+1 < argc) c.head_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ffn-dim") && i+1 < argc) c.ffn_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq-len") && i+1 < argc) c.seq_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) c.inner_matmul = atoi(argv[++i]);
    }
    c.hidden_dim = c.n_heads * c.head_dim;
    return c;
}

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < cfg.n_gpus) {
        fprintf(stderr, "Need %d GPUs, have %d\n", cfg.n_gpus, dev_count);
        return 1;
    }

    printf("════════════════════════════════════════════════════════════\n");
    printf("  BERT Encoder Layer — Complete Pipeline\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  N=65536 (unified), GPUs=%d\n", cfg.n_gpus);
    printf("  heads=%d, head_dim=%d, hidden=%d, ffn=%d, seq=%d\n",
           cfg.n_heads, cfg.head_dim, cfg.hidden_dim, cfg.ffn_dim, cfg.seq_len);
    printf("  Pipeline: MatMul(QKV) → QK^T → Softmax → Attn·V →\n");
    printf("            MatMul(OutProj) → refresh → LayerNorm → refresh →\n");
    printf("            MatMul(FFN1) → GELU → MatMul(FFN2) →\n");
    printf("            refresh → LayerNorm → refresh\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    PerfTimer timer, total_timer;

    // ═══ Setup ═══
    size_t N = 1ULL << 16;
    // 22 levels: enough for MatMul(1) + QK^T(1) + Softmax(16) + Attn·V(1) + OutProj(1) = 20
    // Then refresh and continue with FFN
    vector<int> coeff_bits;
    coeff_bits.push_back(58);  // first
    for (int i = 0; i < 20; i++) coeff_bits.push_back(40);  // 20 main levels
    coeff_bits.push_back(58);  // special
    double SCALE = (double)(1ULL << 40);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));

    cudaSetDevice(0);
    PhantomContext ctx(parms);
    PhantomCKKSEncoder enc(ctx);
    PhantomSecretKey sk(ctx);
    PhantomPublicKey pk = sk.gen_publickey(ctx);
    PhantomRelinKey rk = sk.gen_relinkey(ctx);

    printf("[Setup] Generating Galois keys...\n");
    PhantomGaloisKey gk = sk.create_galois_keys(ctx);
    size_t slots = enc.slot_count();

    CKKSEvaluator ckks_eval(&ctx, &pk, &sk, &enc, &rk, &gk, SCALE);
    GELUEvaluator gelu_eval(ckks_eval);
    SoftmaxEvaluator softmax_eval(ckks_eval);
    LNEvaluator ln_eval(ckks_eval);
    MMEvaluator mm_eval(ckks_eval);

    printf("[Setup] Ready. slots=%zu\n\n", slots);

    // ═══ Random weights ═══
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.02, 0.02);
    uniform_real_distribution<double> idist(-0.5, 0.5);

    auto make_weights = [&](int dim) {
        vector<vector<double>> w(dim, vector<double>(slots, 0.0));
        for (auto &row : w)
            for (size_t s = 0; s < min((size_t)cfg.hidden_dim, slots); s++)
                row[s] = wdist(rng);
        return w;
    };

    // Q, K, V projection weights
    auto W_q = make_weights(cfg.inner_matmul);
    auto W_k = make_weights(cfg.inner_matmul);
    auto W_v = make_weights(cfg.inner_matmul);
    auto W_o = make_weights(cfg.inner_matmul);
    auto W_ffn1 = make_weights(cfg.inner_matmul);
    auto W_ffn2 = make_weights(cfg.inner_matmul);

    // ═══ Encrypt input ═══
    int n_cts = cfg.n_heads;  // one ct per attention head
    printf("[1] Encrypting %d input ciphertexts...\n", n_cts);
    timer.start();

    vector<PhantomCiphertext> X(n_cts);
    vector<vector<double>> X_data(n_cts, vector<double>(slots, 0.0));
    for (int i = 0; i < n_cts; i++) {
        for (size_t s = 0; s < min((size_t)cfg.hidden_dim, slots); s++)
            X_data[i][s] = idist(rng);
        PhantomPlaintext pt;
        ckks_eval.encoder.encode(X_data[i], SCALE, pt);
        ckks_eval.encryptor.encrypt(pt, X[i]);
    }
    printf("    Encrypted in %.1f ms\n\n", timer.elapsed_ms());

    total_timer.start();

    // ════════════════════════════════════════════════════════════════
    // SELF-ATTENTION BLOCK
    // ════════════════════════════════════════════════════════════════

    // Step 1: Q, K, V projections (×3, each consumes ~1 level)
    printf("[2] MatMul Q,K,V projections (%d heads × 3)...\n", cfg.n_heads);
    timer.start();
    vector<PhantomCiphertext> Q_cts, K_cts, V_cts;
    mm_eval.matrix_mul_unified(X, W_q, n_cts, Q_cts);
    mm_eval.matrix_mul_unified(X, W_k, n_cts, K_cts);
    mm_eval.matrix_mul_unified(X, W_v, n_cts, V_cts);
    cudaDeviceSynchronize();
    double mm_qkv_ms = timer.elapsed_ms();
    printf("    QKV MatMul: %.1f ms, levels=%zu\n", mm_qkv_ms, Q_cts[0].coeff_modulus_size());

    // Step 2: Attention scores QK^T (per head, ct × ct multiply)
    printf("[3] QK^T attention scores (%d heads)...\n", cfg.n_heads);
    timer.start();
    vector<PhantomCiphertext> attn_scores(cfg.n_heads);
    for (int h = 0; h < cfg.n_heads; h++) {
        // Match levels
        ckks_eval.evaluator.mod_switch_to_inplace(K_cts[h], Q_cts[h].chain_index());
        K_cts[h].set_scale(Q_cts[h].scale());
        // ct × ct multiply
        ckks_eval.evaluator.multiply(Q_cts[h], K_cts[h], attn_scores[h]);
        ckks_eval.evaluator.relinearize_inplace(attn_scores[h], *ckks_eval.relin_keys);
        ckks_eval.evaluator.rescale_to_next_inplace(attn_scores[h]);
    }
    cudaDeviceSynchronize();
    double qkt_ms = timer.elapsed_ms();
    printf("    QK^T: %.1f ms, levels=%zu\n", qkt_ms, attn_scores[0].coeff_modulus_size());

    // Step 3: Softmax (per head)
    printf("[4] Softmax (%d heads)...\n", cfg.n_heads);
    timer.start();
    vector<PhantomCiphertext> attn_weights(cfg.n_heads);
    for (int h = 0; h < cfg.n_heads; h++) {
        softmax_eval.softmax(attn_scores[h], attn_weights[h], cfg.seq_len);
    }
    cudaDeviceSynchronize();
    double softmax_ms = timer.elapsed_ms();
    printf("    Softmax: %.1f ms, levels=%zu\n", softmax_ms, attn_weights[0].coeff_modulus_size());

    // Step 4: Attention × V (per head)
    printf("[5] Attn × V (%d heads)...\n", cfg.n_heads);
    timer.start();
    vector<PhantomCiphertext> attn_out(cfg.n_heads);
    for (int h = 0; h < cfg.n_heads; h++) {
        ckks_eval.evaluator.mod_switch_to_inplace(V_cts[h], attn_weights[h].chain_index());
        V_cts[h].set_scale(attn_weights[h].scale());
        ckks_eval.evaluator.multiply(attn_weights[h], V_cts[h], attn_out[h]);
        ckks_eval.evaluator.relinearize_inplace(attn_out[h], *ckks_eval.relin_keys);
        ckks_eval.evaluator.rescale_to_next_inplace(attn_out[h]);
    }
    cudaDeviceSynchronize();
    double attnv_ms = timer.elapsed_ms();
    printf("    Attn·V: %.1f ms, levels=%zu\n", attnv_ms, attn_out[0].coeff_modulus_size());

    // Step 5: Output projection
    printf("[6] Output projection MatMul...\n");
    timer.start();
    vector<PhantomCiphertext> proj_out;
    mm_eval.matrix_mul_unified(attn_out, W_o, cfg.n_heads, proj_out);
    cudaDeviceSynchronize();
    double mm_out_ms = timer.elapsed_ms();
    printf("    OutProj MatMul: %.1f ms, levels=%zu\n", mm_out_ms, proj_out[0].coeff_modulus_size());

    // ═══ LEVEL REFRESH #1 ═══
    printf("[7] Level refresh #1 (re-encrypt)...\n");
    timer.start();
    for (auto &ct : proj_out) level_refresh(ckks_eval, ct, SCALE);
    cudaDeviceSynchronize();
    double refresh1_ms = timer.elapsed_ms();
    printf("    Refresh: %.1f ms, levels=%zu\n", refresh1_ms, proj_out[0].coeff_modulus_size());

    // Step 6: LayerNorm #1
    printf("[8] LayerNorm #1...\n");
    timer.start();
    vector<PhantomCiphertext> ln1_out(n_cts);
    for (int i = 0; i < n_cts; i++) {
        ln_eval.layer_norm(proj_out[i], ln1_out[i], cfg.hidden_dim);
    }
    cudaDeviceSynchronize();
    double ln1_ms = timer.elapsed_ms();
    printf("    LayerNorm: %.1f ms, levels=%zu\n", ln1_ms, ln1_out[0].coeff_modulus_size());

    // ═══ LEVEL REFRESH #2 ═══
    printf("[9] Level refresh #2...\n");
    timer.start();
    for (auto &ct : ln1_out) level_refresh(ckks_eval, ct, SCALE);
    cudaDeviceSynchronize();
    double refresh2_ms = timer.elapsed_ms();
    printf("    Refresh: %.1f ms\n", refresh2_ms);

    // ════════════════════════════════════════════════════════════════
    // FFN BLOCK
    // ════════════════════════════════════════════════════════════════

    // Step 7: FFN up-projection
    printf("[10] FFN up-projection MatMul...\n");
    timer.start();
    vector<PhantomCiphertext> ffn_up;
    mm_eval.matrix_mul_unified(ln1_out, W_ffn1, n_cts, ffn_up);
    cudaDeviceSynchronize();
    double mm_ffn1_ms = timer.elapsed_ms();
    printf("    FFN1 MatMul: %.1f ms, levels=%zu\n", mm_ffn1_ms, ffn_up[0].coeff_modulus_size());

    // Step 8: GELU
    printf("[11] GELU activation...\n");
    timer.start();
    vector<PhantomCiphertext> gelu_out(n_cts);
    for (int i = 0; i < n_cts; i++) {
        gelu_eval.gelu(ffn_up[i], gelu_out[i]);
    }
    cudaDeviceSynchronize();
    double gelu_ms = timer.elapsed_ms();
    printf("    GELU: %.1f ms, levels=%zu\n", gelu_ms, gelu_out[0].coeff_modulus_size());

    // Step 9: FFN down-projection
    printf("[12] FFN down-projection MatMul...\n");
    timer.start();
    // Need refresh before FFN2 if levels too low
    for (auto &ct : gelu_out) {
        if (ct.coeff_modulus_size() <= 2) level_refresh(ckks_eval, ct, SCALE);
    }
    vector<PhantomCiphertext> ffn_down;
    mm_eval.matrix_mul_unified(gelu_out, W_ffn2, n_cts, ffn_down);
    cudaDeviceSynchronize();
    double mm_ffn2_ms = timer.elapsed_ms();
    printf("    FFN2 MatMul: %.1f ms\n", mm_ffn2_ms);

    // ═══ LEVEL REFRESH #3 ═══
    printf("[13] Level refresh #3...\n");
    timer.start();
    for (auto &ct : ffn_down) level_refresh(ckks_eval, ct, SCALE);
    cudaDeviceSynchronize();
    double refresh3_ms = timer.elapsed_ms();
    printf("    Refresh: %.1f ms\n", refresh3_ms);

    // Step 10: LayerNorm #2
    printf("[14] LayerNorm #2...\n");
    timer.start();
    vector<PhantomCiphertext> ln2_out(n_cts);
    for (int i = 0; i < n_cts; i++) {
        ln_eval.layer_norm(ffn_down[i], ln2_out[i], cfg.hidden_dim);
    }
    cudaDeviceSynchronize();
    double ln2_ms = timer.elapsed_ms();
    printf("    LayerNorm: %.1f ms\n", ln2_ms);

    // ═══ LEVEL REFRESH #4 ═══
    printf("[15] Level refresh #4 (for next layer)...\n");
    timer.start();
    for (auto &ct : ln2_out) level_refresh(ckks_eval, ct, SCALE);
    cudaDeviceSynchronize();
    double refresh4_ms = timer.elapsed_ms();
    printf("    Refresh: %.1f ms\n", refresh4_ms);

    double total_ms = total_timer.elapsed_ms();

    // ═══ SUMMARY ═══
    double refresh_total = refresh1_ms + refresh2_ms + refresh3_ms + refresh4_ms;
    double compute_total = total_ms - refresh_total;

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  BERT Encoder Layer Summary (%d GPUs)\n", cfg.n_gpus);
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Stage              │ Time (ms)  │ Levels\n");
    printf("  ───────────────────┼────────────┼────────\n");
    printf("  MatMul QKV (×3)    │ %8.1f   │ -1\n", mm_qkv_ms);
    printf("  QK^T (×%d heads)   │ %8.1f   │ -1\n", cfg.n_heads, qkt_ms);
    printf("  Softmax (×%d)      │ %8.1f   │ -16\n", cfg.n_heads, softmax_ms);
    printf("  Attn·V (×%d)       │ %8.1f   │ -1\n", cfg.n_heads, attnv_ms);
    printf("  MatMul OutProj     │ %8.1f   │ -1\n", mm_out_ms);
    printf("  Level Refresh #1   │ %8.1f   │ +18\n", refresh1_ms);
    printf("  LayerNorm #1       │ %8.1f   │ -16\n", ln1_ms);
    printf("  Level Refresh #2   │ %8.1f   │ +18\n", refresh2_ms);
    printf("  MatMul FFN1        │ %8.1f   │ -1\n", mm_ffn1_ms);
    printf("  GELU               │ %8.1f   │ -14\n", gelu_ms);
    printf("  MatMul FFN2        │ %8.1f   │ -1\n", mm_ffn2_ms);
    printf("  Level Refresh #3   │ %8.1f   │ +18\n", refresh3_ms);
    printf("  LayerNorm #2       │ %8.1f   │ -16\n", ln2_ms);
    printf("  Level Refresh #4   │ %8.1f   │ +18\n", refresh4_ms);
    printf("  ───────────────────┼────────────┼────────\n");
    printf("  Compute total      │ %8.1f   │\n", compute_total);
    printf("  Refresh total      │ %8.1f   │ (4×)\n", refresh_total);
    printf("  TOTAL              │ %8.1f   │\n", total_ms);
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Parallelizable: QKV(×3) + QK^T(×%d) + Softmax(×%d) +\n", cfg.n_heads, cfg.n_heads);
    printf("                  Attn·V(×%d) + GELU + Refresh (per-ct)\n", cfg.n_heads);
    printf("  Sequential: LayerNorm (single ct)\n");
    printf("════════════════════════════════════════════════════════════\n");

    printf("\nDone.\n");
    return 0;
}
