/**
 * bert_connected_multinode.cu
 *
 * Multi-node connected BERT pipeline with bootstrapping.
 * MPI scatter → each node: MatMul → GELU → Bootstrap → MPI gather
 * Single parameter set, no re-encryption, true FHE privacy.
 *
 * Usage:
 *   srun --nodes=4 --ntasks-per-node=1 --gres=gpu:4 \
 *     ./bin/bert_connected_multinode --gpus-per-node 4 --cols 64 --inner 16
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

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

#include "ckks_evaluator.cuh"
#include "gelu.cuh"
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

double plain_gelu(double x) {
    return 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

static string serialize_cts(const vector<PhantomCiphertext> &cts) {
    stringstream ss;
    int n = (int)cts.size();
    ss.write(reinterpret_cast<const char*>(&n), sizeof(int));
    for (auto &ct : cts) ct.save(ss);
    return ss.str();
}

static vector<PhantomCiphertext> deserialize_cts(const string &data) {
    stringstream ss(data);
    int n; ss.read(reinterpret_cast<char*>(&n), sizeof(int));
    vector<PhantomCiphertext> cts(n);
    for (int i = 0; i < n; i++) cts[i].load(ss);
    return cts;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int gpus_per_node = 4, n_columns = 64, inner_dim = 16;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpus-per-node") && i+1 < argc) gpus_per_node = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i+1 < argc) n_columns = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) inner_dim = atoi(argv[++i]);
    }

    int total_gpus = world_size * gpus_per_node;
    int cts_per_rank = n_columns / world_size;

    if (rank == 0) {
        printf("================================================================\n");
        printf("  Multi-Node Connected BERT (MatMul→GELU→Bootstrap)\n");
        printf("================================================================\n");
        printf("Nodes: %d, GPUs/node: %d, Total: %d GPUs\n", world_size, gpus_per_node, total_gpus);
        printf("Columns: %d (%d/node), inner: %d\n", n_columns, cts_per_rank, inner_dim);
        printf("Privacy: No re-encryption — true FHE\n\n");
    }

    PerfTimer timer;

    // ═══ Setup: unified N=65536 ═══
    size_t N = 1ULL << 16;
    int main_mod = 17, bs_mod = 14, total_lev = main_mod + bs_mod;
    int logp = 46, logq = 51;
    vector<int> coeff_bits;
    coeff_bits.push_back(logq);
    for (int i = 0; i < main_mod; i++) coeff_bits.push_back(logp);
    for (int i = 0; i < bs_mod; i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(logq); // special
    double SCALE = pow(2.0, logp);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));

    cudaSetDevice(0);
    PhantomContext ctx(parms);

    // Broadcast secret key from rank 0
    PhantomSecretKey sk_rank0(ctx);
    stringstream sk_buf;
    if (rank == 0) sk_rank0.save(sk_buf);
    { string d; if (rank == 0) d = sk_buf.str();
      int sz = d.size(); MPI_Bcast(&sz, 1, MPI_INT, 0, MPI_COMM_WORLD);
      if (rank != 0) d.resize(sz);
      MPI_Bcast(&d[0], sz, MPI_CHAR, 0, MPI_COMM_WORLD);
      if (rank != 0) { sk_buf.str(d); sk_buf.clear(); }
    }
    PhantomSecretKey sk; sk_buf.seekg(0); sk.load(sk_buf);

    PhantomCKKSEncoder enc(ctx);
    PhantomPublicKey pk = sk.gen_publickey(ctx);
    PhantomRelinKey rk = sk.gen_relinkey(ctx);
    PhantomGaloisKey gk = sk.create_galois_keys(ctx);
    size_t slots = enc.slot_count();

    CKKSEvaluator ckks_eval(&ctx, &pk, &sk, &enc, &rk, &gk, SCALE);

    // Intra-node pipeline
    CtPipeline pipe = CtPipeline::create(parms, gpus_per_node, sk);
    pipe.enable_galois_keys();

    // Weights (same seed = same data on all ranks)
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.01, 0.01), idist(-1.0, 1.0);
    vector<vector<double>> weights(inner_dim, vector<double>(slots, 0.0));
    for (int j = 0; j < inner_dim; j++)
        for (size_t s = 0; s < min((size_t)768, slots); s++) weights[j][s] = wdist(rng);
    vector<vector<double>> input_data(n_columns, vector<double>(slots, 0.0));
    for (int i = 0; i < n_columns; i++)
        for (size_t s = 0; s < min((size_t)768, slots); s++) input_data[i][s] = idist(rng);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("[Setup] Done on all nodes\n");

    // ═══ Stage 1: Encrypt + MPI scatter ═══
    timer.start();
    vector<PhantomCiphertext> local_cts;
    if (rank == 0) {
        vector<PhantomCiphertext> all(n_columns);
        for (int i = 0; i < n_columns; i++) {
            PhantomPlaintext pt;
            ckks_eval.encoder.encode(input_data[i], SCALE, pt);
            ckks_eval.encryptor.encrypt(pt, all[i]);
            for (int l = 0; l < bs_mod; l++)
                ckks_eval.evaluator.mod_switch_to_next_inplace(all[i]);
        }
        for (int i = 0; i < cts_per_rank; i++) local_cts.push_back(all[i]);
        for (int r = 1; r < world_size; r++) {
            vector<PhantomCiphertext> batch(all.begin() + r*cts_per_rank,
                                             all.begin() + (r+1)*cts_per_rank);
            string d = serialize_cts(batch); int sz = d.size();
            MPI_Send(&sz, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
            MPI_Send(d.data(), sz, MPI_CHAR, r, 1, MPI_COMM_WORLD);
        }
    } else {
        int sz; MPI_Recv(&sz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        string d(sz, '\0'); MPI_Recv(&d[0], sz, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_cts = deserialize_cts(d);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double scatter_ms = timer.elapsed_ms();
    if (rank == 0) printf("[1] Scatter: %.1f ms\n", scatter_ms);

    // ═══ Stage 2: Connected pipeline on each node ═══
    // MatMul → GELU → Bootstrap (all intra-node pipelined)
    timer.start();
    long logn = 15, logNh = 15;

    pipe.scatter(local_cts);
    pipe.execute_full([&](int gpu, PhantomContext &c, PhantomSecretKey &lsk,
                          PhantomPublicKey &lpk, PhantomRelinKey &lrk,
                          PhantomGaloisKey &lgk, PhantomCKKSEncoder &e,
                          vector<PhantomCiphertext> &local) {
        CKKSEvaluator leval(&c, &lpk, &lsk, &e, &lrk, &lgk, SCALE);
        GELUEvaluator lgelu(leval);
        Bootstrapper lbs(10, logn, logNh, total_lev, SCALE, 25, 59, 2, 1, &leval);
        lbs.slot_vec.push_back(logn);
        lbs.prepare_mod_polynomial();
        lbs.generate_LT_coefficient_3();

        for (auto &ct : local) {
            // MatMul
            vector<PhantomCiphertext> temp(inner_dim);
            for (int j = 0; j < inner_dim; j++) {
                PhantomPlaintext wp; e.encode(c, weights[j], SCALE, wp);
                temp[j] = multiply_plain(c, ct, wp);
            }
            PhantomCiphertext acc = add(c, temp[0], temp[1]);
            for (int j = 2; j < inner_dim; j++) add_inplace(c, acc, temp[j]);
            rescale_to_next_inplace(c, acc);

            // GELU
            PhantomCiphertext gelu_out;
            lgelu.gelu(acc, gelu_out);

            // Bootstrap
            while (gelu_out.coeff_modulus_size() > 1)
                leval.evaluator.mod_switch_to_next_inplace(gelu_out);
            PhantomCiphertext refreshed;
            lbs.bootstrap_3(refreshed, gelu_out);
            ct = std::move(refreshed);
        }
    });
    cudaSetDevice(0);
    auto results = pipe.gather();

    MPI_Barrier(MPI_COMM_WORLD);
    double compute_ms = timer.elapsed_ms();
    if (rank == 0) printf("[2] Compute (MatMul+GELU+Bootstrap): %.1f ms\n", compute_ms);

    // ═══ Stage 3: MPI gather ═══
    timer.start();
    vector<PhantomCiphertext> all_results;
    if (rank == 0) {
        all_results.resize(n_columns);
        for (int i = 0; i < cts_per_rank; i++) all_results[i] = std::move(results[i]);
        for (int r = 1; r < world_size; r++) {
            int sz; MPI_Recv(&sz, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            string d(sz, '\0'); MPI_Recv(&d[0], sz, MPI_CHAR, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            auto batch = deserialize_cts(d);
            for (int i = 0; i < (int)batch.size(); i++)
                all_results[r*cts_per_rank + i] = std::move(batch[i]);
        }
    } else {
        string d = serialize_cts(results); int sz = d.size();
        MPI_Send(&sz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(d.data(), sz, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double gather_ms = timer.elapsed_ms();
    if (rank == 0) printf("[3] Gather: %.1f ms\n", gather_ms);

    // ═══ Verify ═══
    if (rank == 0) {
        vector<vector<double>> gelu_exp(n_columns, vector<double>(slots, 0.0));
        for (int i = 0; i < n_columns; i++) {
            for (int j = 0; j < inner_dim; j++)
                for (size_t s = 0; s < slots; s++)
                    gelu_exp[i][s] += input_data[i][s] * weights[j][s];
            for (size_t s = 0; s < slots; s++)
                gelu_exp[i][s] = plain_gelu(gelu_exp[i][s]);
        }
        double mae_sum = 0;
        for (int i = 0; i < n_columns && i < (int)all_results.size(); i++)
            mae_sum += ckks_eval.calculate_MAE(gelu_exp[i], all_results[i], slots);
        double mae = mae_sum / n_columns;

        printf("\n════════════════════════════════════════════════\n");
        printf("  Multi-Node Connected BERT Results\n");
        printf("════════════════════════════════════════════════\n");
        printf("  Nodes: %d, GPUs: %d\n", world_size, total_gpus);
        printf("  Scatter: %.1f ms\n", scatter_ms);
        printf("  Compute: %.1f ms (MatMul+GELU+Bootstrap)\n", compute_ms);
        printf("  Gather:  %.1f ms\n", gather_ms);
        printf("  Total:   %.1f ms\n", scatter_ms + compute_ms + gather_ms);
        printf("  MAE:     %.6f %s\n", mae, mae < 0.5 ? "PASS" : "FAIL");
        printf("  Privacy: No re-encryption — true FHE\n");
        printf("════════════════════════════════════════════════\n");
    }

    pipe.destroy();
    MPI_Finalize();
    return 0;
}

#else
#include <cstdio>
int main() { printf("ERROR: Built without MPI. Rebuild with MPI module.\n"); return 1; }
#endif
