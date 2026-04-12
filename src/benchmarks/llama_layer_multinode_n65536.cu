/**
 * llama_layer_multinode_n65536.cu
 *
 * Multi-node LLaMA decoder layer at N=65536 with CPU key streaming.
 * MPI scatter/gather of heads across nodes, each node runs multi-GPU
 * with per-GPU CPU key stores (same pattern as bert_encoder_multinode_n65536.cu).
 *
 * LLaMA-vs-BERT structural differences modeled:
 *   • RoPE on Q and K  (ct×pt multiplies + rotations)
 *   • SwiGLU FFN       (3 matmuls + SiLU-proxy + gate⊙up ct×ct multiply)
 *   • RMSNorm          (LayerNorm proxy — same polynomial depth)
 */

#ifdef USE_MPI

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <random>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>
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

struct PerfTimer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() { return chrono::duration<double, milli>(chrono::high_resolution_clock::now() - t0).count(); }
};

static string ser_cts(const vector<PhantomCiphertext> &c) {
    stringstream s; int n = c.size(); s.write((char *)&n, 4);
    for (auto &ct : c) ct.save(s);
    return s.str();
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    int gpn = 4, heads = 12, inner = 64, seq = 16, hidden = 64;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpus-per-node") && i+1 < argc) gpn = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads") && i+1 < argc) heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) inner = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq-len") && i+1 < argc) seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hidden") && i+1 < argc) hidden = atoi(argv[++i]);
    }
    int total_gpus = world * gpn;

    vector<int> heads_per_rank(world);
    for (int i = 0; i < heads; i++) heads_per_rank[i % world]++;
    int my_heads = heads_per_rank[rank];

    if (rank == 0) {
        printf("════════════════════════════════════════════════════════════\n");
        printf("  Multi-Node N=65536 LLaMA Layer (key streaming)\n");
        printf("  %d nodes × %d GPUs = %d GPUs, heads=%d\n", world, gpn, total_gpus, heads);
        printf("  FFN=SwiGLU (3 matmuls), Norm=RMSNorm-proxy, RoPE=on\n");
        printf("  Head distribution:");
        for (int r = 0; r < world; r++) printf(" node%d=%d", r, heads_per_rank[r]);
        printf("\n════════════════════════════════════════════════════════════\n\n");
    }

    PerfTimer timer;

    long logN = 16, logn = logN - 2, logNh = logN - 1;
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
    PhantomSecretKey sk_local(ctx0);
    size_t slots = enc0.slot_count();

    // Broadcast SK from rank 0
    stringstream skb;
    if (rank == 0) sk_local.save(skb);
    {
        string d;
        if (rank == 0) d = skb.str();
        int sz = d.size();
        MPI_Bcast(&sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) d.resize(sz);
        MPI_Bcast(&d[0], sz, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (rank != 0) { skb.str(d); skb.clear(); }
    }
    PhantomSecretKey sk; skb.seekg(0); sk.load(skb);
    PhantomPublicKey pk = sk.gen_publickey(ctx0);
    PhantomRelinKey rk = sk.gen_relinkey(ctx0);
    PhantomGaloisKey gk0;
    CKKSEvaluator eval0(&ctx0, &pk, &sk, &enc0, &rk, &gk0, SCALE);

    stringstream sk_ser;
    sk.save(sk_ser);
    string sk_str = sk_ser.str();

    // Weights — same seed so every node gets identical W
    mt19937 rng(42);
    uniform_real_distribution<double> wd(-0.02, 0.02), id(-0.5, 0.5);
    auto mkw = [&]() {
        vector<vector<double>> w(inner, vector<double>(slots, 0.0));
        for (auto &row : w)
            for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++)
                row[s] = wd(rng);
        return w;
    };
    auto Wq = mkw(), Wk = mkw(), Wv = mkw(), Wo = mkw();
    auto Wgate = mkw(), Wup = mkw(), Wdown = mkw();

    // RoPE masks
    vector<double> rope_cos(slots, 0.0), rope_sin(slots, 0.0);
    for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++) {
        rope_cos[s] = cos(0.01 * s);
        rope_sin[s] = sin(0.01 * s);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Scatter heads from rank 0
    timer.start();
    vector<string> local_ct_data;
    if (rank == 0) {
        vector<PhantomCiphertext> all(heads);
        for (int i = 0; i < heads; i++) {
            vector<double> d(slots, 0.0);
            for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++) d[s] = id(rng);
            PhantomPlaintext pt;
            eval0.encoder.encode(d, SCALE, pt);
            eval0.encryptor.encrypt(pt, all[i]);
            for (int j = 0; j < bs_mod; j++) eval0.evaluator.mod_switch_to_next_inplace(all[i]);
        }
        for (int i = 0; i < my_heads; i++) {
            stringstream ss; all[i].save(ss);
            local_ct_data.push_back(ss.str());
        }
        int offset = heads_per_rank[0];
        for (int r = 1; r < world; r++) {
            int cnt = heads_per_rank[r];
            if (cnt > 0) {
                vector<PhantomCiphertext> b(all.begin() + offset, all.begin() + offset + cnt);
                string d = ser_cts(b); int sz = d.size();
                MPI_Send(&sz, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                MPI_Send(d.data(), sz, MPI_CHAR, r, 1, MPI_COMM_WORLD);
            } else {
                int sz = 0; MPI_Send(&sz, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
            }
            offset += cnt;
        }
    } else {
        int sz; MPI_Recv(&sz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (sz > 0) {
            string d(sz, '\0'); MPI_Recv(&d[0], sz, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            stringstream ss(d); int n; ss.read((char *)&n, 4);
            for (int i = 0; i < n; i++) {
                PhantomCiphertext ct; ct.load(ss);
                stringstream s2; ct.save(s2);
                local_ct_data.push_back(s2.str());
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double scatter_ms = timer.elapsed_ms();
    if (rank == 0) printf("[1] Scatter: %.1f ms\n", scatter_ms);

    // Per-node multi-GPU (LLaMA ops)
    timer.start();
    int local_heads = (int)local_ct_data.size();
    int active_gpus = std::min(gpn, local_heads);
    vector<vector<int>> gpu_heads(active_gpus);
    for (int i = 0; i < local_heads; i++) gpu_heads[i % active_gpus].push_back(i);

    vector<thread> threads;
    atomic<int> setup_done{0};
    PerfTimer compute_timer;

    for (int g = 0; g < active_gpus; g++) {
        threads.emplace_back([&, g]() {
            cudaSetDevice(g);
            PhantomContext ctx(parms);
            PhantomCKKSEncoder enc(ctx);

            PhantomSecretKey lsk;
            { stringstream ss(sk_str); lsk.load(ss); }
            PhantomPublicKey lpk = lsk.gen_publickey(ctx);
            PhantomRelinKey lrk = lsk.gen_relinkey(ctx);
            PhantomGaloisKey lgk;

            CKKSEvaluator le(&ctx, &lpk, &lsk, &enc, &lrk, &lgk, SCALE);
            Bootstrapper lb(loge, logn, logNh, total_level, SCALE,
                            boundary_K, deg, scale_factor, inverse_deg, &le);
            lb.slot_vec.push_back(logn);
            lb.prepare_mod_polynomial();
            lb.generate_LT_coefficient_3();

            vector<int> gsteps;
            gsteps.push_back(0);
            for (int i = 0; i < logN - 1; i++) gsteps.push_back(1 << i);
            for (int i = 0; i < logN - 1; i++) gsteps.push_back(-(1 << i));
            gsteps.push_back(-seq);
            gsteps.push_back(-hidden);
            lb.addLeftRotKeys_Linear_to_vector_3(gsteps);
            std::set<int> step_set(gsteps.begin(), gsteps.end());
            gsteps.assign(step_set.begin(), step_set.end());

            auto gelts = ::get_elts_from_steps(gsteps, N);
            ctx.setup_galois_tool(gelts);
            lgk.resize_slots(gelts.size());

            GaloisKeyStore key_store;
            key_store.generate_all_keys(ctx, lsk, gelts.size());
            le.evaluator.enable_key_streaming(&key_store, &lgk);

            int my_count = setup_done.fetch_add(1) + 1;
            while (setup_done.load() < active_gpus) { /* spin */ }
            if (my_count == active_gpus) compute_timer.start();

            GELUEvaluator lg_eval(le);   // SiLU proxy
            SoftmaxEvaluator lsf(le);
            LNEvaluator ll(le);          // RMSNorm proxy
            MMEvaluator lm(le);

            for (int h_idx : gpu_heads[g]) {
                PhantomCiphertext ct;
                { stringstream ss2(local_ct_data[h_idx]); ct.load(ss2); }

                // ── Attention: QKV ──────────────────────────────────────────
                vector<PhantomCiphertext> xi = {ct}, q, k, v;
                lm.matrix_mul_unified(xi, Wq, 1, q);
                lm.matrix_mul_unified(xi, Wk, 1, k);
                lm.matrix_mul_unified(xi, Wv, 1, v);

                // ── RoPE on Q and K ──────────────────────────────────────────
                {
                    PhantomPlaintext pt_cos;
                    PhantomPlaintext pt_sin;
                    le.encoder.encode(rope_cos, q[0].scale(), pt_cos);
                    le.encoder.encode(rope_sin, q[0].scale(), pt_sin);
                    le.evaluator.mod_switch_to_inplace(pt_cos, q[0].chain_index());
                    le.evaluator.mod_switch_to_inplace(pt_sin, q[0].chain_index());
                    PhantomCiphertext q_rot;
                    PhantomCiphertext k_rot;
                    le.evaluator.rotate_vector(q[0], 1, *le.galois_keys, q_rot);
                    le.evaluator.rotate_vector(k[0], 1, *le.galois_keys, k_rot);
                    PhantomCiphertext q_cos_part;
                    PhantomCiphertext q_sin_part;
                    le.evaluator.multiply_plain(q[0], pt_cos, q_cos_part);
                    le.evaluator.multiply_plain(q_rot, pt_sin, q_sin_part);
                    le.evaluator.add_inplace(q_cos_part, q_sin_part);
                    le.evaluator.rescale_to_next_inplace(q_cos_part);
                    PhantomCiphertext k_cos_part;
                    PhantomCiphertext k_sin_part;
                    le.evaluator.multiply_plain(k[0], pt_cos, k_cos_part);
                    le.evaluator.multiply_plain(k_rot, pt_sin, k_sin_part);
                    le.evaluator.add_inplace(k_cos_part, k_sin_part);
                    le.evaluator.rescale_to_next_inplace(k_cos_part);
                    q[0] = q_cos_part;
                    k[0] = k_cos_part;
                }

                // ── Q·K^T ────────────────────────────────────────────────────
                le.evaluator.mod_switch_to_inplace(k[0], q[0].chain_index());
                k[0].set_scale(q[0].scale());
                PhantomCiphertext as;
                le.evaluator.multiply(q[0], k[0], as);
                le.evaluator.relinearize_inplace(as, *le.relin_keys);
                le.evaluator.rescale_to_next_inplace(as);

                // ── Softmax ──────────────────────────────────────────────────
                PhantomCiphertext aw; lsf.softmax(as, aw, seq);

                // ── Attn·V ───────────────────────────────────────────────────
                le.evaluator.mod_switch_to_inplace(v[0], aw.chain_index());
                v[0].set_scale(aw.scale());
                PhantomCiphertext ao;
                le.evaluator.multiply(aw, v[0], ao);
                le.evaluator.relinearize_inplace(ao, *le.relin_keys);
                le.evaluator.rescale_to_next_inplace(ao);

                // ── Output projection ────────────────────────────────────────
                vector<PhantomCiphertext> pi = {ao}, po;
                lm.matrix_mul_unified(pi, Wo, 1, po);

                // ── Bootstrap #1 ─────────────────────────────────────────────
                while (po[0].coeff_modulus_size() > 1) le.evaluator.mod_switch_to_next_inplace(po[0]);
                PhantomCiphertext b1; lb.bootstrap_3(b1, po[0]);

                // ── RMSNorm #1 ───────────────────────────────────────────────
                PhantomCiphertext rms1o; ll.layer_norm(b1, rms1o, hidden);

                // ── Bootstrap #2 ─────────────────────────────────────────────
                while (rms1o.coeff_modulus_size() > 1) le.evaluator.mod_switch_to_next_inplace(rms1o);
                PhantomCiphertext b2; lb.bootstrap_3(b2, rms1o);

                // ── SwiGLU FFN: gate + up + SiLU + gate⊙up + down ────────────
                vector<PhantomCiphertext> fi = {b2};
                vector<PhantomCiphertext> gate_out;
                vector<PhantomCiphertext> up_out;
                lm.matrix_mul_unified(fi, Wgate, 1, gate_out);
                lm.matrix_mul_unified(fi, Wup,   1, up_out);

                PhantomCiphertext silu_gate;
                lg_eval.gelu(gate_out[0], silu_gate);   // SiLU proxy

                le.evaluator.mod_switch_to_inplace(up_out[0], silu_gate.chain_index());
                up_out[0].set_scale(silu_gate.scale());
                PhantomCiphertext gated;
                le.evaluator.multiply(silu_gate, up_out[0], gated);
                le.evaluator.relinearize_inplace(gated, *le.relin_keys);
                le.evaluator.rescale_to_next_inplace(gated);

                vector<PhantomCiphertext> di = {gated};
                vector<PhantomCiphertext> down_out;
                lm.matrix_mul_unified(di, Wdown, 1, down_out);

                // ── Bootstrap #3 ─────────────────────────────────────────────
                while (down_out[0].coeff_modulus_size() > 1) le.evaluator.mod_switch_to_next_inplace(down_out[0]);
                PhantomCiphertext b3; lb.bootstrap_3(b3, down_out[0]);

                // ── RMSNorm #2 ───────────────────────────────────────────────
                PhantomCiphertext rms2o; ll.layer_norm(b3, rms2o, hidden);

                // ── Bootstrap #4 ─────────────────────────────────────────────
                while (rms2o.coeff_modulus_size() > 1) le.evaluator.mod_switch_to_next_inplace(rms2o);
                PhantomCiphertext b4; lb.bootstrap_3(b4, rms2o);
            }
            cudaDeviceSynchronize();
        });
    }
    for (auto &t : threads) t.join();
    double compute_ms = (active_gpus > 0) ? compute_timer.elapsed_ms() : 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    double node_ms = timer.elapsed_ms();

    double max_compute = 0;
    MPI_Reduce(&compute_ms, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n════════════════════════════════════════════════\n");
        printf("  N=65536 Multi-Node LLaMA Layer — Results\n");
        printf("════════════════════════════════════════════════\n");
        printf("  Nodes=%d, GPUs=%d, Heads=%d\n", world, total_gpus, heads);
        printf("  Scatter:          %8.1f ms\n", scatter_ms);
        printf("  Compute (max):    %8.1f ms\n", max_compute);
        printf("  Compute (r0):     %8.1f ms\n", compute_ms);
        printf("  Node total:       %8.1f ms (includes setup)\n", node_ms);
        printf("════════════════════════════════════════════════\n");
    }

    MPI_Finalize();
    return 0;
}

#else
#include <cstdio>
int main() { printf("ERROR: Compiled without USE_MPI — rerun cmake with MPI enabled.\n"); return 1; }
#endif
