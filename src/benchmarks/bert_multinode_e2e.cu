/**
 * bert_multinode_e2e.cu
 *
 * Full multi-node BERT encoder layer: MatMul → re-encrypt → GELU
 * using MPI for inter-node distribution and CtPipeline for intra-node.
 *
 * Architecture:
 *   1. Rank 0: encrypt input ciphertexts (MatMul params, N=8192)
 *   2. MPI scatter serialized ciphertexts to all nodes
 *   3. Each node: CtPipeline MatMul across local GPUs
 *   4. Each node: local re-encrypt (MatMul→GELU params, N=65536)
 *   5. Each node: CtPipeline GELU across local GPUs (real NEXUS GELU)
 *   6. MPI gather GELU results back to rank 0
 *   7. Rank 0: verify correctness via MAE
 *
 * Usage:
 *   srun --nodes=4 --ntasks-per-node=1 --gres=gpu:4 \
 *     ./bin/bert_multinode_e2e --gpus-per-node 4 --cols 64 --inner 32
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
#include "../multi_gpu/pipeline/ct_pipeline.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;
using namespace nexus_multi_gpu;

struct Timer {
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

// Serialize a batch of ciphertexts into a single buffer
static string serialize_cts(const vector<PhantomCiphertext> &cts) {
    stringstream ss;
    int n = (int)cts.size();
    ss.write(reinterpret_cast<const char*>(&n), sizeof(int));
    for (auto &ct : cts) {
        ct.save(ss);
    }
    return ss.str();
}

// Deserialize a batch of ciphertexts from a buffer
static vector<PhantomCiphertext> deserialize_cts(const string &data) {
    stringstream ss(data);
    int n;
    ss.read(reinterpret_cast<char*>(&n), sizeof(int));
    vector<PhantomCiphertext> cts(n);
    for (int i = 0; i < n; i++) {
        cts[i].load(ss);
    }
    return cts;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int gpus_per_node = 4;
    int n_columns = 64;
    int inner_dim = 32;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpus-per-node") && i+1 < argc) gpus_per_node = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cols") && i+1 < argc) n_columns = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc) inner_dim = atoi(argv[++i]);
    }

    int total_gpus = world_size * gpus_per_node;
    int cts_per_rank = n_columns / world_size;

    if (rank == 0) {
        printf("================================================================\n");
        printf("  BERT Multi-Node E2E: MatMul → re-encrypt → GELU\n");
        printf("================================================================\n");
        printf("Nodes: %d, GPUs/node: %d, Total GPUs: %d\n", world_size, gpus_per_node, total_gpus);
        printf("Columns: %d (%d per node), inner_dim: %d\n\n", n_columns, cts_per_rank, inner_dim);
    }

    Timer timer, total_timer;

    // ══════════════════════════════════════════════════════════════════════
    // SETUP: Both parameter sets on every rank
    // ══════════════════════════════════════════════════════════════════════

    // MatMul: N=8192, L=3
    size_t N_mm = 8192;
    vector<int> mm_bits = {60, 40, 60};
    const double SCALE = (double)(1ULL << 40);

    EncryptionParameters mm_parms(scheme_type::ckks);
    mm_parms.set_poly_modulus_degree(N_mm);
    mm_parms.set_coeff_modulus(CoeffModulus::Create(N_mm, mm_bits));

    cudaSetDevice(0);
    PhantomContext mm_ctx(mm_parms);

    // Generate MatMul key on rank 0, broadcast to all ranks
    PhantomSecretKey mm_sk_rank0(mm_ctx);
    stringstream mm_sk_buf;
    if (rank == 0) mm_sk_rank0.save(mm_sk_buf);
    {
        string sk_data;
        if (rank == 0) sk_data = mm_sk_buf.str();
        int sk_size = (int)sk_data.size();
        MPI_Bcast(&sk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) sk_data.resize(sk_size);
        MPI_Bcast(&sk_data[0], sk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (rank != 0) { mm_sk_buf.str(sk_data); mm_sk_buf.clear(); }
    }
    PhantomSecretKey mm_sk;
    mm_sk_buf.seekg(0);
    mm_sk.load(mm_sk_buf);

    PhantomCKKSEncoder mm_enc(mm_ctx);
    size_t mm_slots = N_mm / 2;

    // GELU: N=65536, L=20
    size_t N_gelu = 1ULL << 16;
    vector<int> gelu_bits = {58, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                              40, 40, 40, 40, 40, 40, 40, 40, 40, 58};

    EncryptionParameters gelu_parms(scheme_type::ckks);
    gelu_parms.set_poly_modulus_degree(N_gelu);
    gelu_parms.set_coeff_modulus(CoeffModulus::Create(N_gelu, gelu_bits));

    PhantomContext gelu_ctx(gelu_parms);

    // Generate GELU key on rank 0, broadcast to all ranks
    PhantomSecretKey gelu_sk_rank0(gelu_ctx);
    stringstream gelu_sk_buf;
    if (rank == 0) gelu_sk_rank0.save(gelu_sk_buf);
    {
        string sk_data;
        if (rank == 0) sk_data = gelu_sk_buf.str();
        int sk_size = (int)sk_data.size();
        MPI_Bcast(&sk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) sk_data.resize(sk_size);
        MPI_Bcast(&sk_data[0], sk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        if (rank != 0) { gelu_sk_buf.str(sk_data); gelu_sk_buf.clear(); }
    }
    PhantomSecretKey gelu_sk;
    gelu_sk_buf.seekg(0);
    gelu_sk.load(gelu_sk_buf);

    PhantomPublicKey gelu_pk = gelu_sk.gen_publickey(gelu_ctx);
    PhantomRelinKey gelu_rk = gelu_sk.gen_relinkey(gelu_ctx);
    PhantomGaloisKey gelu_gk;  // GELU doesn't need rotations
    PhantomCKKSEncoder gelu_enc(gelu_ctx);
    size_t gelu_slots = N_gelu / 2;

    // Random weights (same on all ranks via same seed)
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.01, 0.01);
    uniform_real_distribution<double> idist(-1.0, 1.0);

    vector<vector<double>> weight_data(inner_dim, vector<double>(mm_slots));
    for (int j = 0; j < inner_dim; j++)
        for (size_t s = 0; s < mm_slots; s++) weight_data[j][s] = wdist(rng);

    // Input data (same on all ranks for verification)
    vector<vector<double>> input_data(n_columns, vector<double>(mm_slots));
    for (int i = 0; i < n_columns; i++)
        for (size_t s = 0; s < mm_slots; s++) input_data[i][s] = idist(rng);

    // Plaintext ground truth (all ranks can compute this)
    vector<vector<double>> mm_expected(n_columns, vector<double>(mm_slots, 0.0));
    for (int i = 0; i < n_columns; i++)
        for (int j = 0; j < inner_dim; j++)
            for (size_t s = 0; s < mm_slots; s++)
                mm_expected[i][s] += input_data[i][s] * weight_data[j][s];

    vector<vector<double>> gelu_expected(n_columns, vector<double>(mm_slots));
    for (int i = 0; i < n_columns; i++)
        for (size_t s = 0; s < mm_slots; s++)
            gelu_expected[i][s] = plain_gelu(mm_expected[i][s]);

    // Create intra-node MatMul pipeline
    CtPipeline mm_pipe = CtPipeline::create(mm_parms, gpus_per_node, mm_sk);

    // Create intra-node GELU pipeline
    CtPipeline gelu_pipe = CtPipeline::create(gelu_parms, gpus_per_node, gelu_sk);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("[Setup] Pipelines created on all nodes\n");

    total_timer.start();

    // ══════════════════════════════════════════════════════════════════════
    // STAGE 1: Encrypt + MPI scatter (MatMul ciphertexts)
    // ══════════════════════════════════════════════════════════════════════
    timer.start();

    vector<PhantomCiphertext> local_mm_cts;

    if (rank == 0) {
        // Encrypt all ciphertexts
        PhantomPublicKey mm_pk = mm_sk.gen_publickey(mm_ctx);
        vector<PhantomCiphertext> all_cts(n_columns);
        for (int i = 0; i < n_columns; i++) {
            PhantomPlaintext pt;
            mm_enc.encode(mm_ctx, input_data[i], SCALE, pt);
            mm_pk.encrypt_asymmetric(mm_ctx, pt, all_cts[i]);
        }

        // Keep rank 0's batch
        for (int i = 0; i < cts_per_rank; i++)
            local_mm_cts.push_back(all_cts[i]);

        // Send to other ranks
        for (int r = 1; r < world_size; r++) {
            vector<PhantomCiphertext> batch;
            for (int i = r * cts_per_rank; i < (r + 1) * cts_per_rank && i < n_columns; i++)
                batch.push_back(all_cts[i]);
            string data = serialize_cts(batch);
            int sz = (int)data.size();
            MPI_Send(&sz, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
            MPI_Send(data.data(), sz, MPI_CHAR, r, 1, MPI_COMM_WORLD);
        }
    } else {
        // Receive batch from rank 0
        int sz;
        MPI_Recv(&sz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        string data(sz, '\0');
        MPI_Recv(&data[0], sz, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_mm_cts = deserialize_cts(data);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double scatter_ms = timer.elapsed_ms();
    if (rank == 0) printf("[1] Encrypt + scatter: %.1f ms (%d cts/node)\n", scatter_ms, cts_per_rank);

    // ══════════════════════════════════════════════════════════════════════
    // STAGE 2: MatMul on each node (intra-node pipeline)
    // ══════════════════════════════════════════════════════════════════════
    timer.start();

    cudaSetDevice(0);
    mm_pipe.scatter(local_mm_cts);
    mm_pipe.execute([&](int gpu, PhantomContext &c, PhantomRelinKey &r,
                        PhantomCKKSEncoder &e, vector<PhantomCiphertext> &local) {
        for (auto &ct : local) {
            vector<PhantomCiphertext> temp(inner_dim);
            for (int j = 0; j < inner_dim; j++) {
                PhantomPlaintext wp;
                e.encode(c, weight_data[j], SCALE, wp);
                temp[j] = multiply_plain(c, ct, wp);
            }
            PhantomCiphertext acc = add(c, temp[0], temp[1]);
            for (int j = 2; j < inner_dim; j++)
                add_inplace(c, acc, temp[j]);
            rescale_to_next_inplace(c, acc);
            ct = std::move(acc);
        }
    });
    cudaSetDevice(0);
    auto mm_results = mm_pipe.gather();

    MPI_Barrier(MPI_COMM_WORLD);
    double mm_ms = timer.elapsed_ms();
    if (rank == 0) printf("[2] MatMul: %.1f ms\n", mm_ms);

    // ══════════════════════════════════════════════════════════════════════
    // STAGE 3: Local re-encrypt (MatMul N=8192 → GELU N=65536)
    // ══════════════════════════════════════════════════════════════════════
    timer.start();

    cudaSetDevice(0);
    vector<PhantomCiphertext> local_gelu_cts(cts_per_rank);
    for (int i = 0; i < cts_per_rank; i++) {
        PhantomPlaintext pt;
        vector<double> vals;
        mm_sk.decrypt(mm_ctx, mm_results[i], pt);
        mm_enc.decode(mm_ctx, pt, vals);
        vals.resize(gelu_slots, 0.0);

        PhantomPlaintext gelu_pt;
        gelu_enc.encode(gelu_ctx, vals, SCALE, gelu_pt);
        gelu_pk.encrypt_asymmetric(gelu_ctx, gelu_pt, local_gelu_cts[i]);
    }
    cudaDeviceSynchronize();

    MPI_Barrier(MPI_COMM_WORLD);
    double reenc_ms = timer.elapsed_ms();
    if (rank == 0) printf("[3] Re-encrypt: %.1f ms\n", reenc_ms);

    // ══════════════════════════════════════════════════════════════════════
    // STAGE 4: GELU on each node (intra-node pipeline, real NEXUS GELU)
    // ══════════════════════════════════════════════════════════════════════
    timer.start();

    cudaSetDevice(0);
    gelu_pipe.scatter(local_gelu_cts);
    gelu_pipe.execute_full([&](int gpu, PhantomContext &c, PhantomSecretKey &sk,
                               PhantomPublicKey &pk, PhantomRelinKey &rk,
                               PhantomGaloisKey &gk, PhantomCKKSEncoder &e,
                               vector<PhantomCiphertext> &local) {
        nexus::CKKSEvaluator local_eval(&c, &pk, &sk, &e, &rk, &gk, SCALE);
        nexus::GELUEvaluator local_gelu(local_eval);

        for (auto &ct : local) {
            PhantomCiphertext result;
            local_gelu.gelu(ct, result);
            ct = std::move(result);
        }
    });
    cudaSetDevice(0);
    auto gelu_results = gelu_pipe.gather();

    MPI_Barrier(MPI_COMM_WORLD);
    double gelu_ms = timer.elapsed_ms();
    if (rank == 0) printf("[4] GELU: %.1f ms\n", gelu_ms);

    // ══════════════════════════════════════════════════════════════════════
    // STAGE 5: MPI gather GELU results back to rank 0
    // ══════════════════════════════════════════════════════════════════════
    timer.start();

    vector<PhantomCiphertext> all_gelu_results;
    if (rank == 0) {
        all_gelu_results.resize(n_columns);
        // Copy rank 0's results
        for (int i = 0; i < cts_per_rank; i++)
            all_gelu_results[i] = std::move(gelu_results[i]);

        // Receive from other ranks
        for (int r = 1; r < world_size; r++) {
            int sz;
            MPI_Recv(&sz, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            string data(sz, '\0');
            MPI_Recv(&data[0], sz, MPI_CHAR, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            auto batch = deserialize_cts(data);
            for (int i = 0; i < (int)batch.size(); i++)
                all_gelu_results[r * cts_per_rank + i] = std::move(batch[i]);
        }
    } else {
        // Send results to rank 0
        string data = serialize_cts(gelu_results);
        int sz = (int)data.size();
        MPI_Send(&sz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(data.data(), sz, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double gather_ms = timer.elapsed_ms();
    double total_ms = total_timer.elapsed_ms();
    if (rank == 0) printf("[5] Gather: %.1f ms\n", gather_ms);

    // ══════════════════════════════════════════════════════════════════════
    // VERIFY CORRECTNESS (rank 0 only)
    // ══════════════════════════════════════════════════════════════════════
    if (rank == 0) {
        // Verify MatMul (check a few from rank 0's batch)
        double mm_mae_sum = 0;
        PhantomRelinKey mm_rk = mm_sk.gen_relinkey(mm_ctx);
        PhantomGaloisKey mm_gk;
        PhantomPublicKey mm_pk = mm_sk.gen_publickey(mm_ctx);
        CKKSEvaluator mm_eval(&mm_ctx, &mm_pk, &mm_sk, &mm_enc, &mm_rk, &mm_gk, SCALE);

        for (int i = 0; i < cts_per_rank; i++)
            mm_mae_sum += mm_eval.calculate_MAE(mm_expected[i], mm_results[i], mm_slots);
        double mm_mae = mm_mae_sum / cts_per_rank;

        // Verify GELU (all columns gathered)
        CKKSEvaluator gelu_eval(&gelu_ctx, &gelu_pk, &gelu_sk, &gelu_enc, &gelu_rk, &gelu_gk, SCALE);
        double gelu_mae_sum = 0;
        for (int i = 0; i < n_columns && i < (int)all_gelu_results.size(); i++) {
            vector<double> exp_padded = gelu_expected[i];
            exp_padded.resize(gelu_slots, 0.0);
            gelu_mae_sum += gelu_eval.calculate_MAE(exp_padded, all_gelu_results[i], mm_slots);
        }
        double gelu_mae = gelu_mae_sum / n_columns;

        // ═══ Summary ═══
        printf("\n════════════════════════════════════════════════════════════\n");
        printf("  Multi-Node BERT E2E: MatMul → GELU\n");
        printf("════════════════════════════════════════════════════════════\n");
        printf("  Nodes: %d, GPUs/node: %d, Total GPUs: %d\n", world_size, gpus_per_node, total_gpus);
        printf("  Columns: %d, inner_dim: %d\n", n_columns, inner_dim);
        printf("  ──────────────────────────────────────────────────────────\n");
        printf("  Stage              │ Time (ms)\n");
        printf("  ───────────────────┼──────────\n");
        printf("  Encrypt + scatter  │ %8.1f\n", scatter_ms);
        printf("  MatMul             │ %8.1f\n", mm_ms);
        printf("  Re-encrypt         │ %8.1f\n", reenc_ms);
        printf("  GELU               │ %8.1f\n", gelu_ms);
        printf("  Gather             │ %8.1f\n", gather_ms);
        printf("  ───────────────────┼──────────\n");
        printf("  Total              │ %8.1f\n", total_ms);
        printf("  Compute only       │ %8.1f  (MatMul + GELU)\n", mm_ms + gelu_ms);
        printf("  ──────────────────────────────────────────────────────────\n");
        printf("  MatMul MAE: %.6f %s\n", mm_mae, mm_mae < 0.01 ? "PASS" : "FAIL");
        printf("  GELU MAE:   %.6f %s\n", gelu_mae, gelu_mae < 0.05 ? "PASS" : "FAIL");
        printf("  Overall:    %s\n",
               (mm_mae < 0.01 && gelu_mae < 0.05) ? "ALL PASS" : "FAIL");
        printf("════════════════════════════════════════════════════════════\n");
    }

    mm_pipe.destroy();
    gelu_pipe.destroy();
    MPI_Finalize();
    return 0;
}

#else
#include <cstdio>
int main() { printf("ERROR: Built without MPI support. Rebuild with -DUSE_MPI=ON\n"); return 1; }
#endif
