/**
 * multi_node_bench.cu
 *
 * Multi-node ciphertext pipeline benchmark using MPI.
 *
 * Usage (via srun):
 *   srun --nodes=2 --ntasks-per-node=1 --gres=gpu:4 \
 *     ./bin/multi_node_bench --gpus-per-node 4 --N 8192 --L 10 --n-cts 128
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

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

#include "../multi_gpu/pipeline/multi_node_pipeline.cuh"

using namespace std;
using namespace phantom;
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

    // Parse args
    int gpus_per_node = 4;
    size_t poly_degree = 8192;
    size_t n_moduli = 10;
    int n_cts = 128;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpus-per-node") && i+1 < argc) gpus_per_node = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i+1 < argc) poly_degree = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--L") && i+1 < argc) n_moduli = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-cts") && i+1 < argc) n_cts = atoi(argv[++i]);
    }

    int total_gpus = world_size * gpus_per_node;

    if (rank == 0) {
        printf("=== Multi-Node Pipeline Benchmark ===\n");
        printf("Nodes: %d, GPUs/node: %d, Total GPUs: %d\n", world_size, gpus_per_node, total_gpus);
        printf("N=%zu, L=%zu, ciphertexts=%d\n\n", poly_degree, n_moduli, n_cts);
    }

    // Setup
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_degree);
    vector<int> bits = {60};
    for (size_t i = 0; i < n_moduli - 2; i++) bits.push_back(40);
    bits.push_back(60);
    parms.set_coeff_modulus(arith::CoeffModulus::Create(poly_degree, bits));
    const double SCALE = (double)(1ULL << 40);

    // Each rank creates context on its GPU 0
    cudaSetDevice(0);
    PhantomContext ctx(parms);
    PhantomSecretKey sk(ctx);
    PhantomCKKSEncoder enc(ctx);

    // Workload
    size_t slots = poly_degree / 2;
    vector<double> plain_data(slots, 0.5);

    auto process_fn = [&](int gpu, PhantomContext &c, PhantomRelinKey &r,
                          PhantomCKKSEncoder &e, vector<PhantomCiphertext> &local) {
        for (auto &ct : local) {
            PhantomPlaintext lp;
            e.encode(c, plain_data, ct.scale(), lp);
            multiply_plain_inplace(c, ct, lp);
            rescale_to_next_inplace(c, ct);
            e.encode(c, plain_data, ct.scale(), lp);
            for (int rep = 0; rep < 10; rep++)
                add_plain_inplace(c, ct, lp);
        }
    };

    // Create pipeline
    Timer timer;
    timer.start();
    MultiNodePipeline mnp = MultiNodePipeline::create(parms, gpus_per_node, sk);
    if (rank == 0)
        printf("[1] Pipeline created in %.1f ms\n", timer.elapsed_ms());

    // Create ciphertexts (only on rank 0)
    vector<PhantomCiphertext> cts;
    if (rank == 0) {
        timer.start();
        PhantomPlaintext pt;
        vector<double> input(slots, 1.0);
        enc.encode(ctx, input, SCALE, pt);
        cts.resize(n_cts);
        for (int i = 0; i < n_cts; i++)
            sk.encrypt_symmetric(ctx, pt, cts[i]);
        printf("[2] Encrypted %d cts in %.1f ms\n", n_cts, timer.elapsed_ms());
    }

    // Scatter
    timer.start();
    mnp.scatter(cts);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        printf("[3] Scatter: %.1f ms\n", timer.elapsed_ms());

    // Execute
    timer.start();
    mnp.execute(process_fn);
    MPI_Barrier(MPI_COMM_WORLD);
    double exec_ms = timer.elapsed_ms();
    if (rank == 0)
        printf("[4] Execute: %.1f ms\n", exec_ms);

    // Gather
    timer.start();
    auto results = mnp.gather();
    if (rank == 0)
        printf("[5] Gather: %.1f ms\n", timer.elapsed_ms());

    // Summary
    if (rank == 0) {
        printf("\n=== Results ===\n");
        printf("Nodes: %d, GPUs: %d, Ciphertexts: %d\n", world_size, total_gpus, n_cts);
        printf("Execute time: %.2f ms\n", exec_ms);
        printf("Per-GPU: %d cts, %.3f ms/ct\n",
               n_cts / total_gpus, exec_ms / (n_cts / total_gpus));
    }

    mnp.destroy();
    MPI_Finalize();
    return 0;
}

#else
#include <cstdio>
int main() { printf("ERROR: Built without MPI support. Rebuild with -DUSE_MPI=ON\n"); return 1; }
#endif
