/**
 * bert_encoder_multigpu.cu
 *
 * Multi-GPU BERT encoder layer. Pipelines parallelizable stages:
 *   - QKV projections, QK^T, Softmax, Attn·V (per-head parallel)
 *   - GELU (per-ct parallel)
 *   - Level refreshes (per-ct parallel)
 *   - LayerNorm remains single-GPU (intra-ct rotations)
 *
 * Usage:
 *   ./bin/bert_encoder_multigpu --n-gpus 4 --heads 4 --inner 16
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

static void level_refresh(CKKSEvaluator &eval, PhantomCiphertext &ct, double scale) {
    PhantomPlaintext pt;
    vector<double> vals;
    eval.decryptor.decrypt(ct, pt);
    eval.encoder.decode(pt, vals);
    eval.encoder.encode(vals, scale, pt);
    eval.encryptor.encrypt(pt, ct);
}

int main(int argc, char **argv) {
    int n_gpus = 1, n_heads = 4, inner = 16, seq_len = 16;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc) n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads") && i+1 < argc) n_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) inner = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq-len") && i+1 < argc) seq_len = atoi(argv[++i]);
    }
    int hidden = n_heads * 32;

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < n_gpus) { fprintf(stderr, "Need %d GPUs\n", n_gpus); return 1; }

    printf("════════════════════════════════════════════════════════════\n");
    printf("  BERT Encoder Layer — Multi-GPU (%d GPUs)\n", n_gpus);
    printf("  heads=%d, hidden=%d, inner=%d\n", n_heads, hidden, inner);
    printf("════════════════════════════════════════════════════════════\n\n");

    PerfTimer timer, total_timer;

    // Setup
    size_t N = 1ULL << 16;
    vector<int> coeff_bits;
    coeff_bits.push_back(58);
    for (int i = 0; i < 20; i++) coeff_bits.push_back(40);
    coeff_bits.push_back(58);
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
    PhantomGaloisKey gk = sk.create_galois_keys(ctx);
    size_t slots = enc.slot_count();

    CKKSEvaluator eval0(&ctx, &pk, &sk, &enc, &rk, &gk, SCALE);
    GELUEvaluator gelu0(eval0);
    SoftmaxEvaluator softmax0(eval0);
    LNEvaluator ln0(eval0);
    MMEvaluator mm0(eval0);

    // Random data
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.02, 0.02), idist(-0.5, 0.5);

    auto make_w = [&]() {
        vector<vector<double>> w(inner, vector<double>(slots, 0.0));
        for (auto &r : w) for (size_t s = 0; s < (size_t)hidden; s++) r[s] = wdist(rng);
        return w;
    };
    auto W_q = make_w(), W_k = make_w(), W_v = make_w(), W_o = make_w();
    auto W_f1 = make_w(), W_f2 = make_w();

    // Encrypt
    vector<PhantomCiphertext> X(n_heads);
    for (int i = 0; i < n_heads; i++) {
        vector<double> d(slots, 0.0);
        for (size_t s = 0; s < (size_t)hidden; s++) d[s] = idist(rng);
        PhantomPlaintext pt;
        eval0.encoder.encode(d, SCALE, pt);
        eval0.encryptor.encrypt(pt, X[i]);
    }

    // ═══════════════════════════════════════════════════════════════
    // SINGLE GPU BASELINE
    // ═══════════════════════════════════════════════════════════════
    printf("═══ Single GPU Baseline ═══\n");
    total_timer.start();

    // Self-attention
    timer.start();
    vector<PhantomCiphertext> Q, K, V;
    mm0.matrix_mul_unified(X, W_q, n_heads, Q);
    mm0.matrix_mul_unified(X, W_k, n_heads, K);
    mm0.matrix_mul_unified(X, W_v, n_heads, V);
    cudaDeviceSynchronize();
    double t_qkv = timer.elapsed_ms();

    timer.start();
    vector<PhantomCiphertext> attn(n_heads);
    for (int h = 0; h < n_heads; h++) {
        eval0.evaluator.mod_switch_to_inplace(K[h], Q[h].chain_index());
        K[h].set_scale(Q[h].scale());
        eval0.evaluator.multiply(Q[h], K[h], attn[h]);
        eval0.evaluator.relinearize_inplace(attn[h], *eval0.relin_keys);
        eval0.evaluator.rescale_to_next_inplace(attn[h]);
    }
    cudaDeviceSynchronize();
    double t_qkt = timer.elapsed_ms();

    timer.start();
    vector<PhantomCiphertext> attn_w(n_heads);
    for (int h = 0; h < n_heads; h++)
        softmax0.softmax(attn[h], attn_w[h], seq_len);
    cudaDeviceSynchronize();
    double t_soft = timer.elapsed_ms();

    timer.start();
    vector<PhantomCiphertext> attn_out(n_heads);
    for (int h = 0; h < n_heads; h++) {
        eval0.evaluator.mod_switch_to_inplace(V[h], attn_w[h].chain_index());
        V[h].set_scale(attn_w[h].scale());
        eval0.evaluator.multiply(attn_w[h], V[h], attn_out[h]);
        eval0.evaluator.relinearize_inplace(attn_out[h], *eval0.relin_keys);
        eval0.evaluator.rescale_to_next_inplace(attn_out[h]);
    }
    cudaDeviceSynchronize();
    double t_attnv = timer.elapsed_ms();

    timer.start();
    vector<PhantomCiphertext> proj;
    mm0.matrix_mul_unified(attn_out, W_o, n_heads, proj);
    cudaDeviceSynchronize();
    double t_proj = timer.elapsed_ms();

    timer.start();
    for (auto &ct : proj) level_refresh(eval0, ct, SCALE);
    cudaDeviceSynchronize();
    double t_r1 = timer.elapsed_ms();

    timer.start();
    vector<PhantomCiphertext> ln1(n_heads);
    for (int i = 0; i < n_heads; i++) ln0.layer_norm(proj[i], ln1[i], hidden);
    cudaDeviceSynchronize();
    double t_ln1 = timer.elapsed_ms();

    timer.start();
    for (auto &ct : ln1) level_refresh(eval0, ct, SCALE);
    cudaDeviceSynchronize();
    double t_r2 = timer.elapsed_ms();

    // FFN
    timer.start();
    vector<PhantomCiphertext> ffn1;
    mm0.matrix_mul_unified(ln1, W_f1, n_heads, ffn1);
    cudaDeviceSynchronize();
    double t_ffn1 = timer.elapsed_ms();

    timer.start();
    vector<PhantomCiphertext> gelu_res(n_heads);
    for (int i = 0; i < n_heads; i++) gelu0.gelu(ffn1[i], gelu_res[i]);
    cudaDeviceSynchronize();
    double t_gelu = timer.elapsed_ms();

    timer.start();
    for (auto &ct : gelu_res)
        if (ct.coeff_modulus_size() <= 2) level_refresh(eval0, ct, SCALE);
    vector<PhantomCiphertext> ffn2;
    mm0.matrix_mul_unified(gelu_res, W_f2, n_heads, ffn2);
    cudaDeviceSynchronize();
    double t_ffn2 = timer.elapsed_ms();

    timer.start();
    for (auto &ct : ffn2) level_refresh(eval0, ct, SCALE);
    cudaDeviceSynchronize();
    double t_r3 = timer.elapsed_ms();

    timer.start();
    vector<PhantomCiphertext> ln2(n_heads);
    for (int i = 0; i < n_heads; i++) ln0.layer_norm(ffn2[i], ln2[i], hidden);
    cudaDeviceSynchronize();
    double t_ln2 = timer.elapsed_ms();

    timer.start();
    for (auto &ct : ln2) level_refresh(eval0, ct, SCALE);
    cudaDeviceSynchronize();
    double t_r4 = timer.elapsed_ms();

    double total_1gpu = total_timer.elapsed_ms();
    double refresh_1gpu = t_r1 + t_r2 + t_r3 + t_r4;
    double compute_1gpu = total_1gpu - refresh_1gpu;

    printf("  QKV=%.1f QK^T=%.1f Soft=%.1f AV=%.1f Proj=%.1f\n",
           t_qkv, t_qkt, t_soft, t_attnv, t_proj);
    printf("  R1=%.1f LN1=%.1f R2=%.1f FFN1=%.1f GELU=%.1f FFN2=%.1f\n",
           t_r1, t_ln1, t_r2, t_ffn1, t_gelu, t_ffn2);
    printf("  R3=%.1f LN2=%.1f R4=%.1f\n", t_r3, t_ln2, t_r4);
    printf("  Total: %.1f ms (compute=%.1f, refresh=%.1f)\n\n", total_1gpu, compute_1gpu, refresh_1gpu);

    // ═══════════════════════════════════════════════════════════════
    // MULTI-GPU PIPELINE
    // ═══════════════════════════════════════════════════════════════
    if (n_gpus > 1) {
        printf("═══ Multi-GPU Pipeline (%d GPUs) ═══\n", n_gpus);

        CtPipeline pipe = CtPipeline::create(parms, n_gpus, sk);
        pipe.enable_galois_keys();

        // Re-encrypt fresh input
        cudaSetDevice(0);
        vector<PhantomCiphertext> X2(n_heads);
        for (int i = 0; i < n_heads; i++) {
            vector<double> d(slots, 0.0);
            mt19937 rng2(42 + i);
            for (size_t s = 0; s < (size_t)hidden; s++) d[s] = idist(rng2);
            PhantomPlaintext pt;
            eval0.encoder.encode(d, SCALE, pt);
            eval0.encryptor.encrypt(pt, X2[i]);
        }

        total_timer.start();

        // Pipeline: scatter n_heads cts, each GPU processes its heads
        pipe.scatter(X2);
        timer.start();
        pipe.execute_full([&](int gpu, PhantomContext &c, PhantomSecretKey &lsk,
                              PhantomPublicKey &lpk, PhantomRelinKey &lrk,
                              PhantomGaloisKey &lgk, PhantomCKKSEncoder &e,
                              vector<PhantomCiphertext> &local) {
            CKKSEvaluator leval(&c, &lpk, &lsk, &e, &lrk, &lgk, SCALE);
            GELUEvaluator lgelu(leval);
            SoftmaxEvaluator lsoft(leval);
            LNEvaluator lln(leval);
            MMEvaluator lmm(leval);

            // Per-head: QKV → QK^T → Softmax → AV → OutProj → refresh → LN → refresh
            for (auto &ct : local) {
                // Q, K, V projections
                vector<PhantomCiphertext> q_tmp, k_tmp, v_tmp;
                {
                    vector<PhantomCiphertext> x_vec = {ct};
                    lmm.matrix_mul_unified(x_vec, W_q, 1, q_tmp);
                    lmm.matrix_mul_unified(x_vec, W_k, 1, k_tmp);
                    lmm.matrix_mul_unified(x_vec, W_v, 1, v_tmp);
                }

                // QK^T
                leval.evaluator.mod_switch_to_inplace(k_tmp[0], q_tmp[0].chain_index());
                k_tmp[0].set_scale(q_tmp[0].scale());
                PhantomCiphertext attn_s;
                leval.evaluator.multiply(q_tmp[0], k_tmp[0], attn_s);
                leval.evaluator.relinearize_inplace(attn_s, *leval.relin_keys);
                leval.evaluator.rescale_to_next_inplace(attn_s);

                // Softmax
                PhantomCiphertext attn_w;
                lsoft.softmax(attn_s, attn_w, seq_len);

                // Attn·V
                leval.evaluator.mod_switch_to_inplace(v_tmp[0], attn_w.chain_index());
                v_tmp[0].set_scale(attn_w.scale());
                PhantomCiphertext a_out;
                leval.evaluator.multiply(attn_w, v_tmp[0], a_out);
                leval.evaluator.relinearize_inplace(a_out, *leval.relin_keys);
                leval.evaluator.rescale_to_next_inplace(a_out);

                // OutProj
                vector<PhantomCiphertext> p_in = {a_out}, p_out;
                lmm.matrix_mul_unified(p_in, W_o, 1, p_out);

                // Refresh + LN
                level_refresh(leval, p_out[0], SCALE);
                PhantomCiphertext ln_out;
                lln.layer_norm(p_out[0], ln_out, hidden);

                // Refresh + FFN
                level_refresh(leval, ln_out, SCALE);
                vector<PhantomCiphertext> f1_in = {ln_out}, f1_out;
                lmm.matrix_mul_unified(f1_in, W_f1, 1, f1_out);

                // GELU
                PhantomCiphertext g_out;
                lgelu.gelu(f1_out[0], g_out);

                // FFN2
                if (g_out.coeff_modulus_size() <= 2) level_refresh(leval, g_out, SCALE);
                vector<PhantomCiphertext> f2_in = {g_out}, f2_out;
                lmm.matrix_mul_unified(f2_in, W_f2, 1, f2_out);

                // Refresh + LN + Refresh
                level_refresh(leval, f2_out[0], SCALE);
                PhantomCiphertext ln2_out;
                lln.layer_norm(f2_out[0], ln2_out, hidden);
                level_refresh(leval, ln2_out, SCALE);

                ct = std::move(ln2_out);
            }
        });
        cudaSetDevice(0);
        cudaDeviceSynchronize();
        double total_pipe = total_timer.elapsed_ms();

        printf("  Total (%d GPUs): %.1f ms\n", n_gpus, total_pipe);
        double speedup = total_1gpu / total_pipe;

        printf("\n════════════════════════════════════════════════\n");
        printf("  BERT Encoder Layer Scaling\n");
        printf("════════════════════════════════════════════════\n");
        printf("  1 GPU:     %8.1f ms\n", total_1gpu);
        printf("  %d GPUs:    %8.1f ms\n", n_gpus, total_pipe);
        printf("  Speedup:   %.2fx\n", speedup);
        printf("  Efficiency: %.1f%%\n", speedup / n_gpus * 100.0);
        printf("════════════════════════════════════════════════\n");

        pipe.destroy();
    }

    printf("\nDone.\n");
    return 0;
}
