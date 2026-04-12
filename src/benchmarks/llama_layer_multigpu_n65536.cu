/**
 * llama_layer_multigpu_n65536.cu
 *
 * LLaMA-style decoder layer at N=65536 with CPU-side Galois key streaming.
 *
 * Structural differences from BERT (what matters for FHE cost):
 *   • Norm: RMSNorm instead of LayerNorm. In FHE both need one inv-sqrt
 *     polynomial evaluation; RMSNorm skips the mean-subtract (saves ~1 sum-tree).
 *     We reuse the LayerNorm evaluator as a same-depth proxy.
 *   • Activation: SwiGLU FFN = 3 matmuls (gate, up, down) + elementwise
 *     mul(SiLU(gate), up) + down-projection. BERT FFN = 2 matmuls + GELU.
 *     This adds one extra ciphertext matmul and one ciphertext×ciphertext
 *     multiply per layer — the largest structural cost bump.
 *   • Activation function: SiLU (x · σ(x)) has nearly identical polynomial
 *     approximation depth to GELU at the ranges used after bootstrap. We
 *     reuse the GELU evaluator as a same-depth proxy.
 *   • RoPE: two extra rotations + two ct×pt multiplies per layer (on Q and K).
 *     Included explicitly below via `rope_rotate`.
 *
 * Same head distribution, key streaming, and per-op instrumentation as
 * bert_encoder_multigpu_n65536.cu, so numbers are directly comparable.
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
#include <thread>
#include <sstream>
#include <atomic>
#include <mutex>
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

struct PerfTimer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        return chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
    }
};

struct OpTimes {
    // Attention
    double qkv_matmul = 0, rope = 0, qk_matmul = 0, softmax = 0,
           av_matmul = 0, out_matmul = 0;
    // Norm + bootstrap after attention
    double bs1 = 0, rms1 = 0, bs2 = 0;
    // SwiGLU FFN
    double ffn_gate = 0, ffn_up = 0, ffn_silu = 0,
           ffn_gate_mul_up = 0, ffn_down = 0;
    // Norm + bootstrap after FFN
    double bs3 = 0, rms2 = 0, bs4 = 0;
    int heads = 0;
    void add(const OpTimes &o) {
        qkv_matmul += o.qkv_matmul; rope += o.rope; qk_matmul += o.qk_matmul;
        softmax += o.softmax; av_matmul += o.av_matmul; out_matmul += o.out_matmul;
        bs1 += o.bs1; rms1 += o.rms1; bs2 += o.bs2;
        ffn_gate += o.ffn_gate; ffn_up += o.ffn_up; ffn_silu += o.ffn_silu;
        ffn_gate_mul_up += o.ffn_gate_mul_up; ffn_down += o.ffn_down;
        bs3 += o.bs3; rms2 += o.rms2; bs4 += o.bs4;
        heads += o.heads;
    }
    double total() const {
        return qkv_matmul + rope + qk_matmul + softmax + av_matmul + out_matmul +
               bs1 + rms1 + bs2 + ffn_gate + ffn_up + ffn_silu +
               ffn_gate_mul_up + ffn_down + bs3 + rms2 + bs4;
    }
};

#define TIME_OP(field, code) do { \
    cudaDeviceSynchronize(); \
    PerfTimer _pt; _pt.start(); \
    code; \
    cudaDeviceSynchronize(); \
    times.field += _pt.elapsed_ms(); \
} while(0)

int main(int argc, char **argv) {
    int n_gpus = 4, n_heads = 12, inner = 64, seq_len = 16;
    int hidden = 64;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc) n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads") && i+1 < argc) n_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) inner = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq-len") && i+1 < argc) seq_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hidden") && i+1 < argc) hidden = atoi(argv[++i]);
    }

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < n_gpus) { fprintf(stderr, "Need %d GPUs, have %d\n", n_gpus, dev_count); return 1; }

    printf("════════════════════════════════════════════════════════════\n");
    printf("  Multi-GPU LLaMA Layer N=65536 (%d GPUs, key streaming)\n", n_gpus);
    printf("  heads=%d, hidden=%d, inner=%d, seq=%d\n", n_heads, hidden, inner, seq_len);
    printf("  FFN=SwiGLU (3 matmuls), Norm=RMSNorm-proxy, Act=SiLU-proxy, RoPE=on\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    long logN = 16;
    long logn = logN - 2;
    long logNh = logN - 1;
    size_t N = 1ULL << logN;
    long sparse_slots_val = 1L << logn;
    int logp = 46, logq = 51, log_special = 51;
    int main_mod = 21, bs_mod = 14;
    int total_level = main_mod + bs_mod;
    double SCALE = pow(2.0, logp);

    vector<int> coeff_bits;
    coeff_bits.push_back(logq);
    for (int i = 0; i < main_mod; i++) coeff_bits.push_back(logp);
    for (int i = 0; i < bs_mod; i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(log_special);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));
    parms.set_sparse_slots(sparse_slots_val);
    parms.set_secret_key_hamming_weight(192);

    long boundary_K = 25, deg = 59, scale_factor = 2, inverse_deg = 1, loge = 10;

    cudaSetDevice(0);
    PhantomContext ctx0(parms);
    PhantomCKKSEncoder enc0(ctx0);
    PhantomSecretKey sk0(ctx0);
    PhantomPublicKey pk0 = sk0.gen_publickey(ctx0);
    PhantomRelinKey rk0 = sk0.gen_relinkey(ctx0);
    PhantomGaloisKey gk0_empty;
    size_t slots = enc0.slot_count();
    CKKSEvaluator eval0(&ctx0, &pk0, &sk0, &enc0, &rk0, &gk0_empty, SCALE);

    stringstream sk_buf;
    sk0.save(sk_buf);

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
    // SwiGLU: gate + up + down (three weight matrices, vs BERT's two)
    auto W_gate = make_w(), W_up = make_w(), W_down = make_w();

    // RoPE cos/sin plaintext masks (approximated as two random plaintext rotations;
    // structurally equivalent to ct×pt multiplies + rotations).
    vector<double> rope_cos(slots, 0.0), rope_sin(slots, 0.0);
    for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++) {
        rope_cos[s] = cos(0.01 * s);
        rope_sin[s] = sin(0.01 * s);
    }

    vector<PhantomCiphertext> X(n_heads);
    for (int i = 0; i < n_heads; i++) {
        vector<double> d(slots, 0.0);
        for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++) d[s] = idist(rng);
        PhantomPlaintext pt;
        eval0.encoder.encode(d, SCALE, pt);
        eval0.encryptor.encrypt(pt, X[i]);
        for (int j = 0; j < bs_mod; j++) eval0.evaluator.mod_switch_to_next_inplace(X[i]);
    }
    printf("[Setup] Encrypted %d heads on GPU 0\n\n", n_heads);

    vector<vector<int>> gpu_heads(n_gpus);
    for (int i = 0; i < n_heads; i++) gpu_heads[i % n_gpus].push_back(i);

    vector<string> ct_data(n_heads);
    for (int i = 0; i < n_heads; i++) {
        stringstream ss; X[i].save(ss);
        ct_data[i] = ss.str();
    }

    printf("═══ Running on %d GPUs (key streaming at N=65536) ═══\n", n_gpus);

    vector<thread> threads;
    atomic<int> setup_done{0};
    PerfTimer compute_timer, total_timer;
    total_timer.start();

    OpTimes global_times;
    mutex times_mtx;

    for (int g = 0; g < n_gpus; g++) {
        threads.emplace_back([&, g]() {
            cudaSetDevice(g);

            PhantomContext ctx(parms);
            PhantomCKKSEncoder enc(ctx);

            PhantomSecretKey sk;
            { stringstream ss(sk_buf.str()); sk.load(ss); }
            PhantomPublicKey pk = sk.gen_publickey(ctx);
            PhantomRelinKey rk = sk.gen_relinkey(ctx);
            PhantomGaloisKey gk;

            CKKSEvaluator le(&ctx, &pk, &sk, &enc, &rk, &gk, SCALE);

            Bootstrapper lb(loge, logn, logNh, total_level, SCALE,
                            boundary_K, deg, scale_factor, inverse_deg, &le);
            lb.slot_vec.push_back(logn);
            lb.prepare_mod_polynomial();
            lb.generate_LT_coefficient_3();

            vector<int> gsteps;
            gsteps.push_back(0);
            for (int i = 0; i < logN - 1; i++) gsteps.push_back(1 << i);
            for (int i = 0; i < logN - 1; i++) gsteps.push_back(-(1 << i));
            gsteps.push_back(-seq_len);
            gsteps.push_back(-hidden);
            lb.addLeftRotKeys_Linear_to_vector_3(gsteps);

            std::set<int> step_set(gsteps.begin(), gsteps.end());
            gsteps.assign(step_set.begin(), step_set.end());

            auto gelts = ::get_elts_from_steps(gsteps, N);
            ctx.setup_galois_tool(gelts);
            gk.resize_slots(gelts.size());

            GaloisKeyStore key_store;
            key_store.generate_all_keys(ctx, sk, gelts.size());
            le.evaluator.enable_key_streaming(&key_store, &gk);

            printf("[GPU %d] Setup complete (%zu heads, %zu keys)\n",
                   g, gpu_heads[g].size(), gelts.size());
            fflush(stdout);

            int my_count = setup_done.fetch_add(1) + 1;
            while (setup_done.load() < n_gpus) { /* spin */ }
            if (my_count == n_gpus) compute_timer.start();

            GELUEvaluator lg(le);        // proxy for SiLU (same depth)
            SoftmaxEvaluator ls(le);
            LNEvaluator ll(le);          // proxy for RMSNorm (same depth)
            MMEvaluator lm(le);

            OpTimes times;

            for (int h_idx : gpu_heads[g]) {
                PhantomCiphertext ct;
                { stringstream ss(ct_data[h_idx]); ct.load(ss); }

                // ─── Attention ───
                vector<PhantomCiphertext> xi = {ct}, q, k, v;
                TIME_OP(qkv_matmul, {
                    lm.matrix_mul_unified(xi, W_q, 1, q);
                    lm.matrix_mul_unified(xi, W_k, 1, k);
                    lm.matrix_mul_unified(xi, W_v, 1, v);
                });

                // RoPE: apply cos/sin rotation to Q and K.
                // Structurally = 2 rotations + 2 ct×pt multiplies + 2 adds per Q,K.
                TIME_OP(rope, {
                    PhantomPlaintext pt_cos;
                    PhantomPlaintext pt_sin;
                    le.encoder.encode(rope_cos, q[0].scale(), pt_cos);
                    le.encoder.encode(rope_sin, q[0].scale(), pt_sin);
                    le.evaluator.mod_switch_to_inplace(pt_cos, q[0].chain_index());
                    le.evaluator.mod_switch_to_inplace(pt_sin, q[0].chain_index());
                    PhantomCiphertext q_rot;
                    PhantomCiphertext k_rot;
                    PhantomCiphertext q_sin;
                    PhantomCiphertext k_sin;
                    le.evaluator.rotate_vector(q[0], 1, *le.galois_keys, q_rot);
                    le.evaluator.rotate_vector(k[0], 1, *le.galois_keys, k_rot);
                    le.evaluator.multiply_plain(q[0], pt_cos, q_sin);
                    le.evaluator.multiply_plain(q_rot, pt_sin, q_rot);
                    le.evaluator.add_inplace(q_sin, q_rot);
                    le.evaluator.rescale_to_next_inplace(q_sin);
                    le.evaluator.multiply_plain(k[0], pt_cos, k_sin);
                    le.evaluator.multiply_plain(k_rot, pt_sin, k_rot);
                    le.evaluator.add_inplace(k_sin, k_rot);
                    le.evaluator.rescale_to_next_inplace(k_sin);
                    q[0] = q_sin;
                    k[0] = k_sin;
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
                TIME_OP(softmax, { ls.softmax(as, aw, seq_len); });

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

                PhantomCiphertext b1;
                TIME_OP(bs1, {
                    while (po[0].coeff_modulus_size() > 1) le.evaluator.mod_switch_to_next_inplace(po[0]);
                    lb.bootstrap_3(b1, po[0]);
                });

                PhantomCiphertext ln1o;
                TIME_OP(rms1, { ll.layer_norm(b1, ln1o, hidden); });  // RMSNorm proxy

                PhantomCiphertext b2;
                TIME_OP(bs2, {
                    while (ln1o.coeff_modulus_size() > 1) le.evaluator.mod_switch_to_next_inplace(ln1o);
                    lb.bootstrap_3(b2, ln1o);
                });

                // ─── SwiGLU FFN: 3 matmuls + elementwise mul(SiLU(gate), up) + down ───
                vector<PhantomCiphertext> fi = {b2}, gate_out, up_out;
                TIME_OP(ffn_gate, { lm.matrix_mul_unified(fi, W_gate, 1, gate_out); });
                TIME_OP(ffn_up,   { lm.matrix_mul_unified(fi, W_up,   1, up_out);   });

                PhantomCiphertext silu_gate;
                TIME_OP(ffn_silu, { lg.gelu(gate_out[0], silu_gate); });  // SiLU proxy

                PhantomCiphertext gated;
                TIME_OP(ffn_gate_mul_up, {
                    le.evaluator.mod_switch_to_inplace(up_out[0], silu_gate.chain_index());
                    up_out[0].set_scale(silu_gate.scale());
                    le.evaluator.multiply(silu_gate, up_out[0], gated);
                    le.evaluator.relinearize_inplace(gated, *le.relin_keys);
                    le.evaluator.rescale_to_next_inplace(gated);
                });

                vector<PhantomCiphertext> di = {gated}, down_out;
                TIME_OP(ffn_down, { lm.matrix_mul_unified(di, W_down, 1, down_out); });

                PhantomCiphertext b3;
                TIME_OP(bs3, {
                    while (down_out[0].coeff_modulus_size() > 1) le.evaluator.mod_switch_to_next_inplace(down_out[0]);
                    lb.bootstrap_3(b3, down_out[0]);
                });

                PhantomCiphertext ln2o;
                TIME_OP(rms2, { ll.layer_norm(b3, ln2o, hidden); });  // RMSNorm proxy

                PhantomCiphertext b4;
                TIME_OP(bs4, {
                    while (ln2o.coeff_modulus_size() > 1) le.evaluator.mod_switch_to_next_inplace(ln2o);
                    lb.bootstrap_3(b4, ln2o);
                });
                times.heads++;
                printf("[GPU %d] head %d COMPLETE (out level=%zu)\n",
                       g, h_idx, b4.coeff_modulus_size());
                fflush(stdout);
            }

            cudaDeviceSynchronize();
            { lock_guard<mutex> lk(times_mtx); global_times.add(times); }
        });
    }

    for (auto &t : threads) t.join();
    double compute_ms = compute_timer.elapsed_ms();
    double total_ms = total_timer.elapsed_ms();

    printf("\n════════════════════════════════════════════════\n");
    printf("  N=65536 Multi-GPU LLaMA Layer — Results\n");
    printf("════════════════════════════════════════════════\n");
    printf("  GPUs: %d, Heads: %d, N=65536\n", n_gpus, n_heads);
    printf("  Setup:   %8.1f ms\n", total_ms - compute_ms);
    printf("  Compute: %8.1f ms (pipeline parallelism)\n", compute_ms);
    printf("  Total:   %8.1f ms\n", total_ms);
    printf("════════════════════════════════════════════════\n");

    int H = global_times.heads > 0 ? global_times.heads : 1;
    double sum = global_times.total();
    auto row = [&](const char *name, double v) {
        printf("  %-20s %9.1f ms   %7.1f ms/head   %5.1f%%\n",
               name, v, v / H, 100.0 * v / sum);
    };
    printf("\n─── Per-operation timing (summed across %d heads) ───\n", H);
    row("QKV MatMul",         global_times.qkv_matmul);
    row("RoPE (Q,K)",         global_times.rope);
    row("Q*K^T MatMul",       global_times.qk_matmul);
    row("Softmax",            global_times.softmax);
    row("Attn*V MatMul",      global_times.av_matmul);
    row("Out MatMul",         global_times.out_matmul);
    row("Bootstrap #1",       global_times.bs1);
    row("RMSNorm #1",         global_times.rms1);
    row("Bootstrap #2",       global_times.bs2);
    row("FFN gate MatMul",    global_times.ffn_gate);
    row("FFN up MatMul",      global_times.ffn_up);
    row("SiLU(gate)",         global_times.ffn_silu);
    row("gate ⊙ up (ct×ct)",  global_times.ffn_gate_mul_up);
    row("FFN down MatMul",    global_times.ffn_down);
    row("Bootstrap #3",       global_times.bs3);
    row("RMSNorm #2",         global_times.rms2);
    row("Bootstrap #4",       global_times.bs4);
    printf("  %-20s %9.1f ms   %7.1f ms/head   100.0%%\n",
           "TOTAL (work)", sum, sum / H);
    double bs_total = global_times.bs1 + global_times.bs2 + global_times.bs3 + global_times.bs4;
    double mm_total = global_times.qkv_matmul + global_times.qk_matmul + global_times.av_matmul +
                      global_times.out_matmul + global_times.ffn_gate + global_times.ffn_up +
                      global_times.ffn_down;
    double nl_total = global_times.softmax + global_times.ffn_silu + global_times.rms1 + global_times.rms2;
    double extra   = global_times.rope + global_times.ffn_gate_mul_up;
    printf("\n─── Category rollup ───\n");
    printf("  Bootstraps (4×):      %8.1f ms   (%.1f%%)\n", bs_total, 100.0*bs_total/sum);
    printf("  MatMuls (7×):         %8.1f ms   (%.1f%%)\n", mm_total, 100.0*mm_total/sum);
    printf("  Non-linear (4×):      %8.1f ms   (%.1f%%)\n", nl_total, 100.0*nl_total/sum);
    printf("  LLaMA extras (RoPE+⊙):%8.1f ms   (%.1f%%)\n", extra, 100.0*extra/sum);
    printf("════════════════════════════════════════════════\n");
    return 0;
}
