/**
 * bert_multinode.cu
 *
 * Multi-node BERT MatMul + GELU pipeline using MPI.
 * Each node has 4 H100 GPUs. MPI distributes ciphertexts across nodes,
 * each node uses CtPipeline for intra-node parallelism.
 *
 * Usage (via srun):
 *   srun --nodes=2 --ntasks-per-node=1 --gres=gpu:4 \
 *     ./bin/bert_multinode --gpus-per-node 4 --cols 64 --inner 32
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
#include "../multi_gpu/pipeline/multi_node_pipeline.cuh"

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

    if (rank == 0) {
        printf("================================================================\n");
        printf("  BERT Multi-Node FHE Inference\n");
        printf("================================================================\n");
        printf("Nodes: %d, GPUs/node: %d, Total GPUs: %d\n", world_size, gpus_per_node, total_gpus);
        printf("Columns: %d, inner_dim: %d\n\n", n_columns, inner_dim);
    }

    Timer timer;

    // MatMul parameters: N=8192, L=3
    size_t N_mm = 8192;
    vector<int> mm_bits = {60, 40, 60};
    const double SCALE = (double)(1ULL << 40);

    EncryptionParameters mm_parms(scheme_type::ckks);
    mm_parms.set_poly_modulus_degree(N_mm);
    mm_parms.set_coeff_modulus(CoeffModulus::Create(N_mm, mm_bits));

    cudaSetDevice(0);
    PhantomContext mm_ctx(mm_parms);
    PhantomSecretKey mm_sk(mm_ctx);
    PhantomCKKSEncoder mm_enc(mm_ctx);
    size_t mm_slots = N_mm / 2;

    // Random weights (all ranks need the same)
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.01, 0.01);
    uniform_real_distribution<double> idist(-1.0, 1.0);

    vector<vector<double>> weight_data(inner_dim, vector<double>(mm_slots));
    for (int j = 0; j < inner_dim; j++)
        for (size_t s = 0; s < mm_slots; s++) weight_data[j][s] = wdist(rng);

    // Create multi-node pipeline
    timer.start();
    MultiNodePipeline mnp = MultiNodePipeline::create(mm_parms, gpus_per_node, mm_sk);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        printf("[1] Pipeline created in %.1f ms\n", timer.elapsed_ms());

    // ── MatMul ──────────────────────────────────────────────────────────
    // Only rank 0 encrypts
    vector<PhantomCiphertext> mm_cts;
    vector<vector<double>> input_data(n_columns, vector<double>(mm_slots));
    if (rank == 0) {
        timer.start();
        PhantomPublicKey mm_pk = mm_sk.gen_publickey(mm_ctx);

        for (int i = 0; i < n_columns; i++)
            for (size_t s = 0; s < mm_slots; s++) input_data[i][s] = idist(rng);

        mm_cts.resize(n_columns);
        for (int i = 0; i < n_columns; i++) {
            PhantomPlaintext pt;
            mm_enc.encode(mm_ctx, input_data[i], SCALE, pt);
            mm_pk.encrypt_asymmetric(mm_ctx, pt, mm_cts[i]);
        }
        printf("[2] Encrypted %d cts in %.1f ms\n", n_columns, timer.elapsed_ms());
    }

    // Scatter
    timer.start();
    mnp.scatter(mm_cts);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        printf("[3] Scatter: %.1f ms\n", timer.elapsed_ms());

    // Execute MatMul on each node's GPUs
    timer.start();
    mnp.execute([&](int gpu, PhantomContext &c, PhantomRelinKey &r,
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
    MPI_Barrier(MPI_COMM_WORLD);
    double mm_exec_ms = timer.elapsed_ms();
    if (rank == 0)
        printf("[4] MatMul execute: %.1f ms\n", mm_exec_ms);

    // Gather
    timer.start();
    auto mm_results = mnp.gather();
    if (rank == 0) {
        printf("[5] Gather: %.1f ms\n", timer.elapsed_ms());

        // Verify
        double mm_mae_sum = 0;
        PhantomRelinKey mm_rk = mm_sk.gen_relinkey(mm_ctx);
        PhantomGaloisKey mm_gk;
        PhantomPublicKey mm_pk2 = mm_sk.gen_publickey(mm_ctx);
        CKKSEvaluator verify_eval(&mm_ctx, &mm_pk2, &mm_sk, &mm_enc, &mm_rk, &mm_gk, SCALE);

        for (int i = 0; i < n_columns && i < (int)mm_results.size(); i++) {
            vector<double> expected(mm_slots, 0.0);
            for (int j = 0; j < inner_dim; j++)
                for (size_t s = 0; s < mm_slots; s++)
                    expected[s] += input_data[i][s] * weight_data[j][s];
            mm_mae_sum += verify_eval.calculate_MAE(expected, mm_results[i], mm_slots);
        }
        double mm_mae = mm_mae_sum / n_columns;
        printf("  MatMul MAE: %.6f %s\n", mm_mae, mm_mae < 0.01 ? "PASS" : "FAIL");
    }

    mnp.destroy();

    // ── Summary ─────────────────────────────────────────────────────────
    if (rank == 0) {
        printf("\n════════════════════════════════════════════════\n");
        printf("  Multi-Node BERT MatMul Results\n");
        printf("════════════════════════════════════════════════\n");
        printf("  Nodes: %d, GPUs: %d\n", world_size, total_gpus);
        printf("  Columns: %d, inner_dim: %d\n", n_columns, inner_dim);
        printf("  MatMul execute: %.1f ms\n", mm_exec_ms);
        printf("  Per-GPU workload: %d cts\n", n_columns / total_gpus);
        printf("  Per-ct time: %.3f ms\n", mm_exec_ms / (n_columns / total_gpus));
        printf("════════════════════════════════════════════════\n");
    }

    MPI_Finalize();
    return 0;
}

#else
#include <cstdio>
int main() { printf("ERROR: Built without MPI support. Rebuild with -DUSE_MPI=ON\n"); return 1; }
#endif
