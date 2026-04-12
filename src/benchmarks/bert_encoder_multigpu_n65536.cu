/**
 * bert_encoder_multigpu_n65536.cu
 *
 * Multi-GPU BERT encoder layer at N=65536 with CPU-side Galois key streaming.
 * Each GPU has its own PhantomContext + key store (CPU) + reusable GPU buffer.
 * Heads distributed round-robin across GPUs.
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
    printf("  Multi-GPU BERT Encoder N=65536 (%d GPUs, key streaming)\n", n_gpus);
    printf("  heads=%d, hidden=%d, inner=%d, seq=%d\n", n_heads, hidden, inner, seq_len);
    printf("════════════════════════════════════════════════════════════\n\n");

    // ═══ N=65536 params ═══
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

    // ═══ Setup on GPU 0: context, keys, encrypt input ═══
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
    auto W_f1 = make_w(), W_f2 = make_w();

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

    // Distribute heads: round-robin
    vector<vector<int>> gpu_heads(n_gpus);
    for (int i = 0; i < n_heads; i++) gpu_heads[i % n_gpus].push_back(i);

    // Serialize ciphertexts for scatter
    vector<string> ct_data(n_heads);
    for (int i = 0; i < n_heads; i++) {
        stringstream ss; X[i].save(ss);
        ct_data[i] = ss.str();
    }

    // ═══ Multi-GPU execution ═══
    printf("═══ Running on %d GPUs (key streaming at N=65536) ═══\n", n_gpus);

    vector<thread> threads;
    atomic<int> setup_done{0};
    PerfTimer compute_timer, total_timer;
    total_timer.start();

    for (int g = 0; g < n_gpus; g++) {
        threads.emplace_back([&, g]() {
            cudaSetDevice(g);

            // Each GPU creates its OWN PhantomContext (sets thread-local default_stream)
            PhantomContext ctx(parms);
            PhantomCKKSEncoder enc(ctx);

            PhantomSecretKey sk;
            { stringstream ss(sk_buf.str()); sk.load(ss); }
            PhantomPublicKey pk = sk.gen_publickey(ctx);
            PhantomRelinKey rk = sk.gen_relinkey(ctx);
            PhantomGaloisKey gk;

            CKKSEvaluator le(&ctx, &pk, &sk, &enc, &rk, &gk, SCALE);

            // Per-GPU bootstrapper setup
            Bootstrapper lb(loge, logn, logNh, total_level, SCALE,
                            boundary_K, deg, scale_factor, inverse_deg, &le);
            lb.slot_vec.push_back(logn);
            lb.prepare_mod_polynomial();
            lb.generate_LT_coefficient_3();

            // Collect rotation steps
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

            // Per-GPU key store — each GPU has ITS OWN CPU key store
            GaloisKeyStore key_store;
            key_store.generate_all_keys(ctx, sk, gelts.size());
            le.evaluator.enable_key_streaming(&key_store, &gk);

            printf("[GPU %d] Setup complete (%zu heads, %zu keys)\n",
                   g, gpu_heads[g].size(), gelts.size());
            fflush(stdout);

            int my_count = setup_done.fetch_add(1) + 1;
            while (setup_done.load() < n_gpus) { /* spin */ }
            if (my_count == n_gpus) compute_timer.start();

            GELUEvaluator lg(le);
            SoftmaxEvaluator ls(le);
            LNEvaluator ll(le);
            MMEvaluator lm(le);

            // Process assigned heads
            for (int h_idx : gpu_heads[g]) {
                PhantomCiphertext ct;
                { stringstream ss(ct_data[h_idx]); ct.load(ss); }

                vector<PhantomCiphertext> xi = {ct}, q, k, v;
                lm.matrix_mul_unified(xi, W_q, 1, q);
                lm.matrix_mul_unified(xi, W_k, 1, k);
                lm.matrix_mul_unified(xi, W_v, 1, v);

                le.evaluator.mod_switch_to_inplace(k[0], q[0].chain_index());
                k[0].set_scale(q[0].scale());
                PhantomCiphertext as;
                le.evaluator.multiply(q[0], k[0], as);
                le.evaluator.relinearize_inplace(as, *le.relin_keys);
                le.evaluator.rescale_to_next_inplace(as);

                PhantomCiphertext aw;
                ls.softmax(as, aw, seq_len);

                le.evaluator.mod_switch_to_inplace(v[0], aw.chain_index());
                v[0].set_scale(aw.scale());
                PhantomCiphertext ao;
                le.evaluator.multiply(aw, v[0], ao);
                le.evaluator.relinearize_inplace(ao, *le.relin_keys);
                le.evaluator.rescale_to_next_inplace(ao);

                vector<PhantomCiphertext> pi = {ao}, po;
                lm.matrix_mul_unified(pi, W_o, 1, po);

                while (po[0].coeff_modulus_size() > 1) le.evaluator.mod_switch_to_next_inplace(po[0]);
                PhantomCiphertext b1; lb.bootstrap_3(b1, po[0]);

                PhantomCiphertext ln1o; ll.layer_norm(b1, ln1o, hidden);
                while (ln1o.coeff_modulus_size() > 1) le.evaluator.mod_switch_to_next_inplace(ln1o);
                PhantomCiphertext b2; lb.bootstrap_3(b2, ln1o);

                vector<PhantomCiphertext> fi = {b2}, fo;
                lm.matrix_mul_unified(fi, W_f1, 1, fo);
                PhantomCiphertext go; lg.gelu(fo[0], go);

                vector<PhantomCiphertext> f2i = {go}, f2o;
                lm.matrix_mul_unified(f2i, W_f2, 1, f2o);
                while (f2o[0].coeff_modulus_size() > 1) le.evaluator.mod_switch_to_next_inplace(f2o[0]);
                PhantomCiphertext b3; lb.bootstrap_3(b3, f2o[0]);

                PhantomCiphertext ln2o; ll.layer_norm(b3, ln2o, hidden);
                while (ln2o.coeff_modulus_size() > 1) le.evaluator.mod_switch_to_next_inplace(ln2o);
                PhantomCiphertext b4; lb.bootstrap_3(b4, ln2o);
                printf("[GPU %d] head %d COMPLETE (out level=%zu)\n",
                       g, h_idx, b4.coeff_modulus_size());
                fflush(stdout);
            }

            cudaDeviceSynchronize();
        });
    }

    for (auto &t : threads) t.join();
    double compute_ms = compute_timer.elapsed_ms();
    double total_ms = total_timer.elapsed_ms();

    printf("\n════════════════════════════════════════════════\n");
    printf("  N=65536 Multi-GPU BERT Encoder — Results\n");
    printf("════════════════════════════════════════════════\n");
    printf("  GPUs: %d, Heads: %d, N=65536\n", n_gpus, n_heads);
    printf("  Setup:   %8.1f ms\n", total_ms - compute_ms);
    printf("  Compute: %8.1f ms (pipeline parallelism)\n", compute_ms);
    printf("  Total:   %8.1f ms\n", total_ms);
    printf("════════════════════════════════════════════════\n");
    return 0;
}
