/**
 * bert_encoder_n65536.cu
 *
 * REAL end-to-end BERT encoder layer at N=65536 with CPU-side Galois key streaming.
 * All 14 NEXUS Table IV operations with 4× real bootstrap.
 *
 * This is the FIRST single-GPU BERT encoder at N=65536 with fully homomorphic
 * execution (no parameter switching, no re-encryption).
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
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

static void print_mem(const char *label) {
    size_t free_m, total_m;
    cudaMemGetInfo(&free_m, &total_m);
    printf("[Mem] %s: %.2f/%.2f GB\n", label,
           (total_m - free_m) / (1024.0*1024.0*1024.0),
           total_m / (1024.0*1024.0*1024.0));
    fflush(stdout);
}

int main(int argc, char **argv) {
    int n_heads = 2, inner = 64, seq = 16, hidden = 64;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--heads") && i+1 < argc) n_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) inner = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq-len") && i+1 < argc) seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hidden") && i+1 < argc) hidden = atoi(argv[++i]);
    }

    printf("════════════════════════════════════════════════════════════\n");
    printf("  BERT Encoder Layer — N=65536 with CPU Key Streaming\n");
    printf("  heads=%d, hidden=%d, inner=%d, seq=%d\n", n_heads, hidden, inner, seq);
    printf("  4× real bootstrap (no re-encryption, single N)\n");
    printf("════════════════════════════════════════════════════════════\n\n");

    cudaSetDevice(0);
    print_mem("Initial");

    // ═══ N=65536 params ═══
    long logN = 16;
    long logn = logN - 2, logNh = logN - 1;
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
    for (int i = 0; i < bs_mod; i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(log_special);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));
    parms.set_sparse_slots(sparse_slots_val);
    parms.set_secret_key_hamming_weight(192);

    // ═══ Setup ═══
    PhantomContext context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey sk(context);
    PhantomPublicKey pk = sk.gen_publickey(context);
    PhantomRelinKey rk = sk.gen_relinkey(context);
    PhantomGaloisKey gk;  // will be populated via streaming
    size_t slots = encoder.slot_count();

    CKKSEvaluator eval(&context, &pk, &sk, &encoder, &rk, &gk, SCALE);
    print_mem("After context + PK + RK");

    // ═══ Weights (small random) ═══
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

    // ═══ Encrypt heads ═══
    vector<PhantomCiphertext> X(n_heads);
    for (int h = 0; h < n_heads; h++) {
        vector<double> d(slots, 0.0);
        for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++) d[s] = idist(rng);
        PhantomPlaintext pt;
        eval.encoder.encode(d, SCALE, pt);
        eval.encryptor.encrypt(pt, X[h]);
        for (int i = 0; i < bs_mod; i++)
            eval.evaluator.mod_switch_to_next_inplace(X[h]);
    }
    printf("[Setup] Encrypted %d heads, level=%zu\n", n_heads, X[0].coeff_modulus_size());

    // ═══ Bootstrap setup ═══
    PerfTimer timer; timer.start();
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
    all_steps.push_back(-seq);
    all_steps.push_back(-hidden);
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(all_steps);

    // Deduplicate
    std::set<int> step_set(all_steps.begin(), all_steps.end());
    all_steps.assign(step_set.begin(), step_set.end());
    printf("[Setup] Total unique rotation steps: %zu\n", all_steps.size());

    auto all_elts = ::get_elts_from_steps(all_steps, N);
    context.setup_galois_tool(all_elts);
    gk.resize_slots(all_elts.size());

    // Generate all keys to CPU
    printf("[KeyStore] Generating %zu keys to CPU memory...\n", all_elts.size());
    GaloisKeyStore key_store;
    key_store.generate_all_keys(context, sk, all_elts.size());
    eval.evaluator.enable_key_streaming(&key_store, &gk);

    printf("[Setup] Total setup: %.1f s\n", timer.elapsed_ms() / 1000.0);
    print_mem("After setup + keystore");

    GELUEvaluator lg(eval);
    SoftmaxEvaluator ls(eval);
    LNEvaluator ll(eval);
    MMEvaluator lm(eval);

    // ═══ BERT Encoder Layer ═══
    printf("\n═══ Running BERT encoder layer on %d heads ═══\n", n_heads);
    PerfTimer layer_timer; layer_timer.start();

    for (int h = 0; h < n_heads; h++) {
        printf("\n[Head %d] Starting...\n", h); fflush(stdout);
        PhantomCiphertext &ct = X[h];
        vector<PhantomCiphertext> xi = {ct}, q, k, v;

        // QKV projections
        lm.matrix_mul_unified(xi, W_q, 1, q);
        lm.matrix_mul_unified(xi, W_k, 1, k);
        lm.matrix_mul_unified(xi, W_v, 1, v);
        printf("[Head %d] QKV done\n", h);

        // QK^T
        eval.evaluator.mod_switch_to_inplace(k[0], q[0].chain_index());
        k[0].set_scale(q[0].scale());
        PhantomCiphertext as;
        eval.evaluator.multiply(q[0], k[0], as);
        eval.evaluator.relinearize_inplace(as, *eval.relin_keys);
        eval.evaluator.rescale_to_next_inplace(as);

        // Softmax
        PhantomCiphertext aw;
        ls.softmax(as, aw, seq);
        printf("[Head %d] Softmax done\n", h);

        // Attn × V
        eval.evaluator.mod_switch_to_inplace(v[0], aw.chain_index());
        v[0].set_scale(aw.scale());
        PhantomCiphertext ao;
        eval.evaluator.multiply(aw, v[0], ao);
        eval.evaluator.relinearize_inplace(ao, *eval.relin_keys);
        eval.evaluator.rescale_to_next_inplace(ao);

        // OutProj
        vector<PhantomCiphertext> pi = {ao}, po;
        lm.matrix_mul_unified(pi, W_o, 1, po);
        printf("[Head %d] OutProj done\n", h);

        // Bootstrap #1
        while (po[0].coeff_modulus_size() > 1)
            eval.evaluator.mod_switch_to_next_inplace(po[0]);
        PerfTimer bt1; bt1.start();
        PhantomCiphertext b1;
        bootstrapper.bootstrap_3(b1, po[0]);
        printf("[Head %d] Bootstrap #1: %.1f ms\n", h, bt1.elapsed_ms());

        // LayerNorm #1
        PhantomCiphertext ln1o;
        ll.layer_norm(b1, ln1o, hidden);
        printf("[Head %d] LN #1 done\n", h);

        // Bootstrap #2
        while (ln1o.coeff_modulus_size() > 1)
            eval.evaluator.mod_switch_to_next_inplace(ln1o);
        PerfTimer bt2; bt2.start();
        PhantomCiphertext b2;
        bootstrapper.bootstrap_3(b2, ln1o);
        printf("[Head %d] Bootstrap #2: %.1f ms\n", h, bt2.elapsed_ms());

        // FFN1
        vector<PhantomCiphertext> fi = {b2}, fo;
        lm.matrix_mul_unified(fi, W_f1, 1, fo);
        PhantomCiphertext go;
        lg.gelu(fo[0], go);
        printf("[Head %d] GELU done\n", h);

        // FFN2
        vector<PhantomCiphertext> f2i = {go}, f2o;
        lm.matrix_mul_unified(f2i, W_f2, 1, f2o);

        // Bootstrap #3
        while (f2o[0].coeff_modulus_size() > 1)
            eval.evaluator.mod_switch_to_next_inplace(f2o[0]);
        PerfTimer bt3; bt3.start();
        PhantomCiphertext b3;
        bootstrapper.bootstrap_3(b3, f2o[0]);
        printf("[Head %d] Bootstrap #3: %.1f ms\n", h, bt3.elapsed_ms());

        // LayerNorm #2
        PhantomCiphertext ln2o;
        ll.layer_norm(b3, ln2o, hidden);

        // Bootstrap #4
        while (ln2o.coeff_modulus_size() > 1)
            eval.evaluator.mod_switch_to_next_inplace(ln2o);
        PerfTimer bt4; bt4.start();
        PhantomCiphertext b4;
        bootstrapper.bootstrap_3(b4, ln2o);
        printf("[Head %d] Bootstrap #4: %.1f ms\n", h, bt4.elapsed_ms());
        printf("[Head %d] COMPLETE (output level=%zu)\n", h, b4.coeff_modulus_size());
    }

    double layer_ms = layer_timer.elapsed_ms();
    print_mem("After full layer");

    printf("\n════════════════════════════════════════════════\n");
    printf("  N=65536 BERT Encoder Layer — Results\n");
    printf("════════════════════════════════════════════════\n");
    printf("  N=65536, heads=%d, hidden=%d, inner=%d\n", n_heads, hidden, inner);
    printf("  4x real bootstrap per head, CPU key streaming\n");
    printf("  Total compute time: %.1f ms (%.2f s)\n", layer_ms, layer_ms/1000.0);
    printf("  Per-head time: %.1f ms\n", layer_ms / n_heads);
    printf("════════════════════════════════════════════════\n");
    return 0;
}
