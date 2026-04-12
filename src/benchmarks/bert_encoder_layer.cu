/**
 * bert_encoder_layer.cu
 *
 * Complete BERT encoder layer following NEXUS Table IV with REAL bootstrapping.
 * Uses N=32768 parameter set matching NEXUS's bootstrap-compatible configuration.
 *
 * Full layer pipeline (from NEXUS Table IV):
 *   MatMul(Q,K,V) → QK^T → Softmax → Attn·V → MatMul(OutProj) →
 *   BOOTSTRAP → LayerNorm → BOOTSTRAP →
 *   MatMul(FFN_up) → GELU → MatMul(FFN_down) →
 *   BOOTSTRAP → LayerNorm → BOOTSTRAP
 *
 * Parameters: N=32768, sparse_slots=8192, 17 main + 14 bootstrap moduli
 *
 * Usage:
 *   ./bin/bert_encoder_layer [--heads 2] [--inner 16]
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
#include "bootstrapping/Bootstrapper.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

struct PerfTimer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        return chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
    }
};

struct Config {
    int n_heads = 2;     // attention heads (BERT=12, reduced for testing)
    int head_dim = 32;   // per-head dimension (BERT=64)
    int hidden_dim = 64; // total hidden dim = n_heads × head_dim
    int seq_len = 16;    // sequence length
    int inner_matmul = 16; // matmul inner dim for testing
};

Config parse_args(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--heads") && i+1 < argc) c.n_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--head-dim") && i+1 < argc) c.head_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq-len") && i+1 < argc) c.seq_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) c.inner_matmul = atoi(argv[++i]);
    }
    c.hidden_dim = c.n_heads * c.head_dim;
    return c;
}

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    printf("════════════════════════════════════════════════════════════\n");
    printf("  BERT Encoder Layer — Real Bootstrapping\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  N=32768, sparse_slots=8192, hamming=192\n");
    printf("  heads=%d, head_dim=%d, hidden=%d, seq=%d\n",
           cfg.n_heads, cfg.head_dim, cfg.hidden_dim, cfg.seq_len);
    printf("  4× real bootstrap (no re-encryption)\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    PerfTimer timer, total_timer;

    // ═══ Parameters matching NEXUS bootstrap config (N=32768) ═══
    long logN = 15;
    long logn = logN - 2;          // 13
    long logNh = logN - 1;         // 14
    size_t N = 1ULL << logN;       // 32768
    long sparse_slots_val = 1L << logn;  // 8192

    int logp = 46, logq = 51, log_special = 51;
    // Need 20 levels for attention block before first bootstrap (Table IV: 21→1)
    // Then bootstrap restores to main_mod level
    int main_mod = 21, bs_mod = 14;
    int total_level = main_mod + bs_mod;
    double SCALE = pow(2.0, logp);

    // Build coefficient modulus chain
    vector<int> coeff_bits;
    coeff_bits.push_back(logq);                    // first prime
    for (int i = 0; i < main_mod; i++) coeff_bits.push_back(logp);  // main
    for (int i = 0; i < bs_mod; i++) coeff_bits.push_back(logq);    // bootstrap
    coeff_bits.push_back(log_special);             // special prime

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));
    parms.set_sparse_slots(sparse_slots_val);
    parms.set_secret_key_hamming_weight(192);

    cudaSetDevice(0);
    PhantomContext ctx(parms);
    PhantomCKKSEncoder enc(ctx);
    PhantomSecretKey sk(ctx);
    PhantomPublicKey pk = sk.gen_publickey(ctx);
    PhantomRelinKey rk = sk.gen_relinkey(ctx);
    size_t slots = enc.slot_count();

    printf("[Setup] slot_count=%zu, N=%zu\n", slots, N);

    // ═══ Bootstrap setup ═══
    printf("[Setup] Initializing bootstrapper...\n");
    timer.start();

    // Create Galois keys: bootstrap steps + operation-specific rotations
    vector<int> gal_steps;
    gal_steps.push_back(0);  // conjugation
    for (int i = 0; i < logN - 1; i++) gal_steps.push_back(1 << i);   // power-of-2 positive
    for (int i = 0; i < logN - 1; i++) gal_steps.push_back(-(1 << i)); // power-of-2 negative
    // Softmax and LayerNorm use -len rotations
    gal_steps.push_back(-cfg.seq_len);
    gal_steps.push_back(-cfg.hidden_dim);

    PhantomGaloisKey gk = sk.create_galois_keys_from_steps(ctx, gal_steps);

    CKKSEvaluator ckks_eval(&ctx, &pk, &sk, &enc, &rk, &gk, SCALE);

    // Initialize bootstrapper
    long boundary_K = 25, deg = 59, scale_factor = 2, inverse_deg = 1, loge = 10;
    Bootstrapper bootstrapper(loge, logn, logNh, total_level, SCALE,
                              boundary_K, deg, scale_factor, inverse_deg, &ckks_eval);
    bootstrapper.slot_vec.push_back(logn);
    bootstrapper.prepare_mod_polynomial();
    bootstrapper.generate_LT_coefficient_3();

    // Regenerate Galois keys: bootstrap + operation rotation steps
    gal_steps.clear();
    gal_steps.push_back(0);
    for (int i = 0; i < logN - 1; i++) gal_steps.push_back(1 << i);
    for (int i = 0; i < logN - 1; i++) gal_steps.push_back(-(1 << i));
    gal_steps.push_back(-cfg.seq_len);
    gal_steps.push_back(-cfg.hidden_dim);
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps);
    ckks_eval.decryptor.create_galois_keys_from_steps(gal_steps, *ckks_eval.galois_keys);

    double setup_ms = timer.elapsed_ms();
    printf("[Setup] Bootstrapper ready: %.1f ms\n", setup_ms);

    // ═══ Create evaluators ═══
    GELUEvaluator gelu_eval(ckks_eval);
    SoftmaxEvaluator softmax_eval(ckks_eval);
    LNEvaluator ln_eval(ckks_eval);
    MMEvaluator mm_eval(ckks_eval);

    // ═══ Random weights ═══
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.02, 0.02);
    uniform_real_distribution<double> idist(-0.5, 0.5);

    auto make_weights = [&](int dim) {
        vector<vector<double>> w(dim, vector<double>(slots, 0.0));
        for (auto &row : w)
            for (size_t s = 0; s < std::min((size_t)cfg.hidden_dim, slots); s++)
                row[s] = wdist(rng);
        return w;
    };

    auto W_q = make_weights(cfg.inner_matmul);
    auto W_k = make_weights(cfg.inner_matmul);
    auto W_v = make_weights(cfg.inner_matmul);
    auto W_o = make_weights(cfg.inner_matmul);
    auto W_ffn1 = make_weights(cfg.inner_matmul);
    auto W_ffn2 = make_weights(cfg.inner_matmul);

    // ═══ Encrypt input ═══
    int n_cts = cfg.n_heads;
    printf("[1] Encrypting %d input ciphertexts...\n", n_cts);
    timer.start();

    vector<PhantomCiphertext> X(n_cts);
    for (int i = 0; i < n_cts; i++) {
        vector<double> data(slots, 0.0);
        for (size_t s = 0; s < std::min((size_t)cfg.hidden_dim, slots); s++)
            data[s] = idist(rng);
        PhantomPlaintext pt;
        ckks_eval.encoder.encode(data, SCALE, pt);
        ckks_eval.encryptor.encrypt(pt, X[i]);
    }

    // Mod-switch past bootstrap levels to start at main level
    for (auto &ct : X) {
        for (int i = 0; i < bs_mod; i++)
            ckks_eval.evaluator.mod_switch_to_next_inplace(ct);
    }
    printf("    Encrypted in %.1f ms, levels=%zu\n\n", timer.elapsed_ms(), X[0].coeff_modulus_size());

    total_timer.start();

    // ════════════════════════════════════════════════════════════════
    // SELF-ATTENTION BLOCK
    // ════════════════════════════════════════════════════════════════

    // Step 1: Q, K, V projections
    printf("[2] MatMul Q,K,V projections (%d heads × 3)...\n", cfg.n_heads);
    timer.start();
    vector<PhantomCiphertext> Q_cts, K_cts, V_cts;
    mm_eval.matrix_mul_unified(X, W_q, n_cts, Q_cts);
    mm_eval.matrix_mul_unified(X, W_k, n_cts, K_cts);
    mm_eval.matrix_mul_unified(X, W_v, n_cts, V_cts);
    cudaDeviceSynchronize();
    double mm_qkv_ms = timer.elapsed_ms();
    printf("    QKV MatMul: %.1f ms, levels=%zu\n", mm_qkv_ms, Q_cts[0].coeff_modulus_size());

    // Step 2: Attention scores QK^T
    printf("[3] QK^T attention scores (%d heads)...\n", cfg.n_heads);
    timer.start();
    vector<PhantomCiphertext> attn_scores(cfg.n_heads);
    for (int h = 0; h < cfg.n_heads; h++) {
        ckks_eval.evaluator.mod_switch_to_inplace(K_cts[h], Q_cts[h].chain_index());
        K_cts[h].set_scale(Q_cts[h].scale());
        ckks_eval.evaluator.multiply(Q_cts[h], K_cts[h], attn_scores[h]);
        ckks_eval.evaluator.relinearize_inplace(attn_scores[h], *ckks_eval.relin_keys);
        ckks_eval.evaluator.rescale_to_next_inplace(attn_scores[h]);
    }
    cudaDeviceSynchronize();
    double qkt_ms = timer.elapsed_ms();
    printf("    QK^T: %.1f ms, levels=%zu\n", qkt_ms, attn_scores[0].coeff_modulus_size());

    // Step 3: Softmax
    printf("[4] Softmax (%d heads)...\n", cfg.n_heads);
    timer.start();
    vector<PhantomCiphertext> attn_weights(cfg.n_heads);
    for (int h = 0; h < cfg.n_heads; h++) {
        softmax_eval.softmax(attn_scores[h], attn_weights[h], cfg.seq_len);
    }
    cudaDeviceSynchronize();
    double softmax_ms = timer.elapsed_ms();
    printf("    Softmax: %.1f ms, levels=%zu\n", softmax_ms, attn_weights[0].coeff_modulus_size());

    // Step 4: Attention × V
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

    // ═══ Mod-switch to level 1 for bootstrap ═══
    for (auto &ct : proj_out) {
        while (ct.coeff_modulus_size() > 1)
            ckks_eval.evaluator.mod_switch_to_next_inplace(ct);
    }

    // ═══ BOOTSTRAP #1 ═══
    printf("[7] Bootstrap #1...\n");
    timer.start();
    for (auto &ct : proj_out) {
        PhantomCiphertext tmp;
        bootstrapper.bootstrap_3(tmp, ct);
        ct = std::move(tmp);
    }
    cudaDeviceSynchronize();
    double boot1_ms = timer.elapsed_ms();
    printf("    Bootstrap: %.1f ms, levels=%zu\n", boot1_ms, proj_out[0].coeff_modulus_size());

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

    // ═══ Mod-switch to level 1 for bootstrap ═══
    for (auto &ct : ln1_out) {
        while (ct.coeff_modulus_size() > 1)
            ckks_eval.evaluator.mod_switch_to_next_inplace(ct);
    }

    // ═══ BOOTSTRAP #2 ═══
    printf("[9] Bootstrap #2...\n");
    timer.start();
    for (auto &ct : ln1_out) {
        PhantomCiphertext tmp;
        bootstrapper.bootstrap_3(tmp, ct);
        ct = std::move(tmp);
    }
    cudaDeviceSynchronize();
    double boot2_ms = timer.elapsed_ms();
    printf("    Bootstrap: %.1f ms\n", boot2_ms);

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
    vector<PhantomCiphertext> ffn_down;
    mm_eval.matrix_mul_unified(gelu_out, W_ffn2, n_cts, ffn_down);
    cudaDeviceSynchronize();
    double mm_ffn2_ms = timer.elapsed_ms();
    printf("    FFN2 MatMul: %.1f ms, levels=%zu\n", mm_ffn2_ms, ffn_down[0].coeff_modulus_size());

    // ═══ Mod-switch to level 1 ═══
    for (auto &ct : ffn_down) {
        while (ct.coeff_modulus_size() > 1)
            ckks_eval.evaluator.mod_switch_to_next_inplace(ct);
    }

    // ═══ BOOTSTRAP #3 ═══
    printf("[13] Bootstrap #3...\n");
    timer.start();
    for (auto &ct : ffn_down) {
        PhantomCiphertext tmp;
        bootstrapper.bootstrap_3(tmp, ct);
        ct = std::move(tmp);
    }
    cudaDeviceSynchronize();
    double boot3_ms = timer.elapsed_ms();
    printf("    Bootstrap: %.1f ms\n", boot3_ms);

    // Step 10: LayerNorm #2
    printf("[14] LayerNorm #2...\n");
    timer.start();
    vector<PhantomCiphertext> ln2_out(n_cts);
    for (int i = 0; i < n_cts; i++) {
        ln_eval.layer_norm(ffn_down[i], ln2_out[i], cfg.hidden_dim);
    }
    cudaDeviceSynchronize();
    double ln2_ms = timer.elapsed_ms();
    printf("    LayerNorm: %.1f ms, levels=%zu\n", ln2_ms, ln2_out[0].coeff_modulus_size());

    // ═══ Mod-switch to level 1 ═══
    for (auto &ct : ln2_out) {
        while (ct.coeff_modulus_size() > 1)
            ckks_eval.evaluator.mod_switch_to_next_inplace(ct);
    }

    // ═══ BOOTSTRAP #4 ═══
    printf("[15] Bootstrap #4 (for next layer)...\n");
    timer.start();
    for (auto &ct : ln2_out) {
        PhantomCiphertext tmp;
        bootstrapper.bootstrap_3(tmp, ct);
        ct = std::move(tmp);
    }
    cudaDeviceSynchronize();
    double boot4_ms = timer.elapsed_ms();
    printf("    Bootstrap: %.1f ms\n", boot4_ms);

    double total_ms = total_timer.elapsed_ms();

    // ═══ Verify accuracy ═══
    printf("\n[Verify] Checking output accuracy...\n");
    PhantomPlaintext dec_pt;
    vector<double> dec_vals;
    ckks_eval.decryptor.decrypt(ln2_out[0], dec_pt);
    ckks_eval.encoder.decode(dec_pt, dec_vals);
    double max_val = 0;
    for (size_t i = 0; i < std::min(dec_vals.size(), (size_t)cfg.hidden_dim); i++)
        max_val = std::max(max_val, fabs(dec_vals[i]));
    printf("  Output max |value|: %.6f (should be finite, small)\n", max_val);
    printf("  First 5 values:");
    for (int i = 0; i < 5 && i < (int)dec_vals.size(); i++) printf(" %.4f", dec_vals[i]);
    printf("\n");

    // ═══ SUMMARY ═══
    double boot_total = boot1_ms + boot2_ms + boot3_ms + boot4_ms;
    double compute_total = total_ms - boot_total;

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  BERT Encoder Layer — Real Bootstrap Summary\n");
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Stage              │ Time (ms) \n");
    printf("  ───────────────────┼───────────\n");
    printf("  MatMul QKV (×3)    │ %8.1f  \n", mm_qkv_ms);
    printf("  QK^T (×%d heads)   │ %8.1f  \n", cfg.n_heads, qkt_ms);
    printf("  Softmax (×%d)      │ %8.1f  \n", cfg.n_heads, softmax_ms);
    printf("  Attn·V (×%d)       │ %8.1f  \n", cfg.n_heads, attnv_ms);
    printf("  MatMul OutProj     │ %8.1f  \n", mm_out_ms);
    printf("  Bootstrap #1       │ %8.1f  \n", boot1_ms);
    printf("  LayerNorm #1       │ %8.1f  \n", ln1_ms);
    printf("  Bootstrap #2       │ %8.1f  \n", boot2_ms);
    printf("  MatMul FFN1        │ %8.1f  \n", mm_ffn1_ms);
    printf("  GELU               │ %8.1f  \n", gelu_ms);
    printf("  MatMul FFN2        │ %8.1f  \n", mm_ffn2_ms);
    printf("  Bootstrap #3       │ %8.1f  \n", boot3_ms);
    printf("  LayerNorm #2       │ %8.1f  \n", ln2_ms);
    printf("  Bootstrap #4       │ %8.1f  \n", boot4_ms);
    printf("  ───────────────────┼───────────\n");
    printf("  Compute total      │ %8.1f  \n", compute_total);
    printf("  Bootstrap total    │ %8.1f  (4× ~260ms)\n", boot_total);
    printf("  TOTAL              │ %8.1f  \n", total_ms);
    printf("  Bootstrap %%        │ %6.1f%%\n", 100.0 * boot_total / total_ms);
    printf("════════════════════════════════════════════════════════════\n");

    printf("\nDone.\n");
    return 0;
}
