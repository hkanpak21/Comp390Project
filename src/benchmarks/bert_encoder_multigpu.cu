/**
 * bert_encoder_multigpu.cu
 *
 * Multi-GPU BERT encoder layer with REAL bootstrapping.
 * Pipelines per-head processing across GPUs via CtPipeline.
 * Each GPU runs complete attention+FFN per head with 4× bootstrap.
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

int main(int argc, char **argv) {
    int n_gpus = 1, n_heads = 4, inner = 16, seq_len = 16;
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
    if (dev_count < n_gpus) { fprintf(stderr, "Need %d GPUs\n", n_gpus); return 1; }

    printf("════════════════════════════════════════════════════════════\n");
    printf("  BERT Encoder Layer — Multi-GPU Real Bootstrap (%d GPUs)\n", n_gpus);
    printf("  heads=%d, hidden=%d, inner=%d, seq=%d\n", n_heads, hidden, inner, seq_len);
    printf("════════════════════════════════════════════════════════════\n\n");

    PerfTimer timer;

    // ═══ Parameters matching NEXUS bootstrap config ═══
    long logN = 15;
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

    cudaSetDevice(0);
    PhantomContext ctx(parms);
    PhantomCKKSEncoder enc(ctx);
    PhantomSecretKey sk(ctx);
    PhantomPublicKey pk = sk.gen_publickey(ctx);
    PhantomRelinKey rk = sk.gen_relinkey(ctx);
    size_t slots = enc.slot_count();

    // Galois keys: bootstrap + operations
    vector<int> gal_steps;
    gal_steps.push_back(0);
    for (int i = 0; i < logN - 1; i++) gal_steps.push_back(1 << i);
    for (int i = 0; i < logN - 1; i++) gal_steps.push_back(-(1 << i));
    gal_steps.push_back(-seq_len);
    gal_steps.push_back(-hidden);

    long boundary_K = 25, deg = 59, scale_factor = 2, inverse_deg = 1, loge = 10;

    // Minimal evaluator for encryption only (no Galois keys needed)
    PhantomGaloisKey gk; // empty
    CKKSEvaluator eval0(&ctx, &pk, &sk, &enc, &rk, &gk, SCALE);

    // ═══ Weights (shared) ═══
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.02, 0.02), idist(-0.5, 0.5);

    auto make_w = [&]() {
        vector<vector<double>> w(inner, vector<double>(slots, 0.0));
        for (auto &r : w) for (size_t s = 0; s < std::min((size_t)hidden, slots); s++) r[s] = wdist(rng);
        return w;
    };
    auto W_q = make_w(), W_k = make_w(), W_v = make_w(), W_o = make_w();
    auto W_f1 = make_w(), W_f2 = make_w();

    // ═══ Encrypt input ═══
    vector<PhantomCiphertext> X(n_heads);
    for (int i = 0; i < n_heads; i++) {
        vector<double> d(slots, 0.0);
        for (size_t s = 0; s < std::min((size_t)hidden, slots); s++) d[s] = idist(rng);
        PhantomPlaintext pt;
        eval0.encoder.encode(d, SCALE, pt);
        eval0.encryptor.encrypt(pt, X[i]);
    }
    // Mod-switch past bootstrap levels
    for (auto &ct : X)
        for (int i = 0; i < bs_mod; i++)
            eval0.evaluator.mod_switch_to_next_inplace(ct);

    printf("[Setup] Encrypted %d heads, levels=%zu\n\n", n_heads, X[0].coeff_modulus_size());

    // ═══════════════════════════════════════════════════════════════
    // MULTI-GPU PIPELINE
    // ═══════════════════════════════════════════════════════════════
    {
        printf("═══ Multi-GPU Pipeline (%d GPUs) ═══\n", n_gpus);

        CtPipeline pipe = CtPipeline::create(parms, n_gpus, sk);

        // Galois keys will be generated per-GPU inside execute_full
        // after bootstrapper adds its rotation steps

        // Re-encrypt fresh input
        vector<PhantomCiphertext> X2(n_heads);
        for (int i = 0; i < n_heads; i++) {
            vector<double> d(slots, 0.0);
            for (size_t s = 0; s < std::min((size_t)hidden, slots); s++) d[s] = idist(rng);
            PhantomPlaintext pt;
            eval0.encoder.encode(d, SCALE, pt);
            eval0.encryptor.encrypt(pt, X2[i]);
        }
        for (auto &ct : X2)
            for (int i = 0; i < bs_mod; i++)
                eval0.evaluator.mod_switch_to_next_inplace(ct);

        timer.start();
        pipe.scatter(X2);

        pipe.execute_full([&](int gpu, PhantomContext &c, PhantomSecretKey &lsk,
                              PhantomPublicKey &lpk, PhantomRelinKey &lrk,
                              PhantomGaloisKey &lgk, PhantomCKKSEncoder &e,
                              vector<PhantomCiphertext> &local) {

            CKKSEvaluator le(&c, &lpk, &lsk, &e, &lrk, &lgk, SCALE);

            // Per-GPU bootstrapper setup
            Bootstrapper lb(loge, logn, logNh, total_level, SCALE,
                            boundary_K, deg, scale_factor, inverse_deg, &le);
            lb.slot_vec.push_back(logn);
            lb.prepare_mod_polynomial();
            lb.generate_LT_coefficient_3();

            // Generate Galois keys with all rotation steps (bootstrap + operations)
            vector<int> gpu_gal_steps;
            gpu_gal_steps.push_back(0);
            for (int i = 0; i < logN - 1; i++) gpu_gal_steps.push_back(1 << i);
            for (int i = 0; i < logN - 1; i++) gpu_gal_steps.push_back(-(1 << i));
            gpu_gal_steps.push_back(-seq_len);
            gpu_gal_steps.push_back(-hidden);
            lb.addLeftRotKeys_Linear_to_vector_3(gpu_gal_steps);
            le.decryptor.create_galois_keys_from_steps(gpu_gal_steps, *le.galois_keys);

            GELUEvaluator lg(le);
            SoftmaxEvaluator ls(le);
            LNEvaluator ll(le);
            MMEvaluator lm(le);

            for (auto &ct : local) {
                // QKV
                vector<PhantomCiphertext> xi = {ct}, q, k, v;
                lm.matrix_mul_unified(xi, W_q, 1, q);
                lm.matrix_mul_unified(xi, W_k, 1, k);
                lm.matrix_mul_unified(xi, W_v, 1, v);

                // QK^T
                le.evaluator.mod_switch_to_inplace(k[0], q[0].chain_index());
                k[0].set_scale(q[0].scale());
                PhantomCiphertext as;
                le.evaluator.multiply(q[0], k[0], as);
                le.evaluator.relinearize_inplace(as, *le.relin_keys);
                le.evaluator.rescale_to_next_inplace(as);

                // Softmax
                PhantomCiphertext aw;
                ls.softmax(as, aw, seq_len);

                // Attn·V
                le.evaluator.mod_switch_to_inplace(v[0], aw.chain_index());
                v[0].set_scale(aw.scale());
                PhantomCiphertext ao;
                le.evaluator.multiply(aw, v[0], ao);
                le.evaluator.relinearize_inplace(ao, *le.relin_keys);
                le.evaluator.rescale_to_next_inplace(ao);

                // OutProj
                vector<PhantomCiphertext> pi = {ao}, po;
                lm.matrix_mul_unified(pi, W_o, 1, po);

                // Bootstrap #1
                while (po[0].coeff_modulus_size() > 1)
                    le.evaluator.mod_switch_to_next_inplace(po[0]);
                PhantomCiphertext b1;
                lb.bootstrap_3(b1, po[0]);

                // LayerNorm
                PhantomCiphertext ln1o;
                ll.layer_norm(b1, ln1o, hidden);

                // Bootstrap #2
                while (ln1o.coeff_modulus_size() > 1)
                    le.evaluator.mod_switch_to_next_inplace(ln1o);
                PhantomCiphertext b2;
                lb.bootstrap_3(b2, ln1o);

                // FFN1
                vector<PhantomCiphertext> fi = {b2}, fo;
                lm.matrix_mul_unified(fi, W_f1, 1, fo);

                // GELU
                PhantomCiphertext go;
                lg.gelu(fo[0], go);

                // FFN2
                vector<PhantomCiphertext> f2i = {go}, f2o;
                lm.matrix_mul_unified(f2i, W_f2, 1, f2o);

                // Bootstrap #3
                while (f2o[0].coeff_modulus_size() > 1)
                    le.evaluator.mod_switch_to_next_inplace(f2o[0]);
                PhantomCiphertext b3;
                lb.bootstrap_3(b3, f2o[0]);

                // LayerNorm #2
                PhantomCiphertext ln2o;
                ll.layer_norm(b3, ln2o, hidden);

                // Bootstrap #4
                while (ln2o.coeff_modulus_size() > 1)
                    le.evaluator.mod_switch_to_next_inplace(ln2o);
                PhantomCiphertext b4;
                lb.bootstrap_3(b4, ln2o);

                ct = std::move(b4);
            }
        });

        cudaSetDevice(0);
        auto results = pipe.gather();
        cudaDeviceSynchronize();
        double total_pipe = timer.elapsed_ms();

        printf("\n════════════════════════════════════════════════\n");
        printf("  BERT Encoder Layer — Real Bootstrap (%d GPUs)\n", n_gpus);
        printf("════════════════════════════════════════════════\n");
        printf("  Heads: %d, GPUs: %d\n", n_heads, n_gpus);
        printf("  Total:     %8.1f ms\n", total_pipe);
        printf("  Per-head:  %8.1f ms\n", total_pipe);  // all heads parallel
        printf("════════════════════════════════════════════════\n");

        pipe.destroy();
    }

    printf("\nDone.\n");
    return 0;
}
