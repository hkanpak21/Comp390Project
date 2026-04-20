/**
 * bert_dks_multinode.cu
 *
 * multiNEXUS Step 11 — Multi-node BERT DKS Bootstrap Scaling.
 *
 * Extends bert_dks_multigpu.cu to span multiple MN5 nodes.
 * Architecture:
 *   - 1 MPI rank per node, each rank manages 4 H100 GPUs
 *   - NCCL AllReduce spans ALL GPUs across ALL nodes
 *   - Key shards distributed by digit % total_gpus (global GPU rank)
 *   - Each extra node → fewer owned digits → proportionally smaller keys
 *
 * Expected scaling (multiNEXUS Table 4.1 targets):
 *   1 node  / 4  GPUs:  ~2,800 ms/bootstrap
 *   2 nodes / 8  GPUs:  ~1,500 ms/bootstrap
 *   4 nodes / 16 GPUs:  ~  800 ms/bootstrap
 *
 * Key memory scaling:
 *   1 node  / 4  GPUs:  ~16 GB/GPU
 *   2 nodes / 8  GPUs:  ~ 8 GB/GPU
 *   4 nodes / 16 GPUs:  ~ 4 GB/GPU
 *
 * Usage (on MN5):
 *   srun --nodes=2 --ntasks-per-node=1 --gres=gpu:4 \
 *     ./build/bin/bert_dks_multinode --gpus-per-node 4
 */

#ifdef USE_MPI

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <stdexcept>
#include <functional>

#include "phantom.h"
#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"

#include "bootstrapping/Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "galois_key_store.cuh"
#include "galois.cuh"

#include "multi_gpu/distributed_context.cuh"
#include "multi_gpu/distributed_eval.cuh"
#include "multi_gpu/keyswitching/dist_galois_key_store.cuh"
#include "multi_gpu/keyswitching/galois_oa.cuh"

#include "matrix_mul.cuh"
#include "gelu.cuh"
#include "softmax.cuh"
#include "layer_norm.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using nexus_multi_gpu::DistributedContext;

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------
struct MNTimer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    void start() { t0 = clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

// ---------------------------------------------------------------------------
// BERT / crypto parameters (same as bert_dks_multigpu.cu)
// ---------------------------------------------------------------------------
static const long LOG_N   = 16;
static const long LOGN    = LOG_N - 2;
static const long LOGNH   = LOG_N - 1;
static const int  LOGP    = 46;
static const int  LOGQ    = 51;
static const int  LOG_SPEC = 51;
static const int  MAIN_MOD = 21;
static const int  BS_MOD   = 14;
static const int  TOT_LVL  = MAIN_MOD + BS_MOD;
static const int  SEQ_LEN  = 128;
static const int  HIDDEN   = 768;

static EncryptionParameters build_parms() {
    long N = 1L << LOG_N;
    vector<int> bits;
    bits.push_back(LOGQ);
    for (int i = 0; i < MAIN_MOD; i++) bits.push_back(LOGP);
    for (int i = 0; i < BS_MOD;   i++) bits.push_back(LOGQ);
    bits.push_back(LOG_SPEC);
    EncryptionParameters p(scheme_type::ckks);
    p.set_poly_modulus_degree(static_cast<size_t>(N));
    p.set_coeff_modulus(CoeffModulus::Create(static_cast<size_t>(N), bits));
    p.set_sparse_slots(1L << LOGN);
    p.set_secret_key_hamming_weight(192);
    return p;
}

// ---------------------------------------------------------------------------
// Bootstrap timing on this node (all 4 GPUs via DKS)
// ---------------------------------------------------------------------------
static double time_dks_bootstrap(
    Bootstrapper &bs,
    PhantomCiphertext &ct,
    int n_warmup = 1,
    int n_trials = 3)
{
    MNTimer t;
    // Warmup
    for (int w = 0; w < n_warmup; w++) {
        PhantomCiphertext bs_out;
        bs.bootstrap_3(bs_out, ct);
        ct = bs_out;
        cudaDeviceSynchronize();
    }
    // Timed trials
    double total = 0.0;
    for (int r = 0; r < n_trials; r++) {
        t.start();
        PhantomCiphertext bs_out;
        bs.bootstrap_3(bs_out, ct);
        ct = bs_out;
        cudaDeviceSynchronize();
        total += t.elapsed_ms();
    }
    return total / n_trials;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    int gpus_per_node = 4;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gpus-per-node") && i + 1 < argc)
            gpus_per_node = atoi(argv[++i]);
    }

    // Clamp to available devices
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    gpus_per_node = std::min(gpus_per_node, dev_count);

    int total_gpus = mpi_size * gpus_per_node;

    if (mpi_rank == 0) {
        printf("╔══════════════════════════════════════════════════════════════╗\n");
        printf("║   multiNEXUS BERT DKS Multi-Node — N=65536                   ║\n");
        printf("║   Nodes: %2d  GPUs/node: %d  Total GPUs: %2d                   ║\n",
               mpi_size, gpus_per_node, total_gpus);
        printf("╚══════════════════════════════════════════════════════════════╝\n\n");
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Build FHE parameters ──
    auto parms = build_parms();
    double SCALE = pow(2.0, LOGP);

    // ── Create multi-node DistributedContext FIRST (resets Phantom global stream) ──
    cudaSetDevice(0);
    DistributedContext dctx = DistributedContext::create_multinode(
        parms, gpus_per_node, MPI_COMM_WORLD);

    // ── Crypto objects from dctx.context(0) (created after global stream init) ──
    PhantomContext   &ctx0 = dctx.context(0);
    PhantomCKKSEncoder enc0(ctx0);
    PhantomSecretKey sk0(ctx0);
    PhantomPublicKey pk0  = sk0.gen_publickey(ctx0);
    PhantomRelinKey  rk0  = sk0.gen_relinkey(ctx0);
    PhantomGaloisKey gk0;
    CKKSEvaluator eval0(&ctx0, &pk0, &sk0, &enc0, &rk0, &gk0, SCALE);

    // ── Bootstrapper ──
    Bootstrapper bs(10, LOGN, LOGNH, TOT_LVL, SCALE, 25, 59, 2, 1, &eval0);
    bs.slot_vec.push_back(LOGN);
    bs.prepare_mod_polynomial();
    bs.generate_LT_coefficient_3();

    // ── Rotation steps ──
    vector<int> steps;
    steps.push_back(0);
    for (int i = 0; i < LOG_N - 1; i++) steps.push_back(1 << i);
    for (int i = 0; i < LOG_N - 1; i++) steps.push_back(-(1 << i));
    steps.push_back(-SEQ_LEN);
    steps.push_back(-HIDDEN);
    bs.addLeftRotKeys_Linear_to_vector_3(steps);

    {
        std::set<int> step_set(steps.begin(), steps.end());
        steps.assign(step_set.begin(), step_set.end());
    }
    long N_val = 1L << LOG_N;
    auto all_elts = ::get_elts_from_steps(steps, static_cast<size_t>(N_val));

    // MUST set up galois tool BEFORE key generation
    ctx0.setup_galois_tool(all_elts);
    gk0.resize_slots(all_elts.size());
    size_t num_keys = all_elts.size();

    // Build step → key_idx map
    auto &gelts = ctx0.key_galois_tool()->galois_elts();
    map<int, size_t> step_to_idx;
    for (size_t i = 0; i < steps.size(); i++) {
        uint32_t elt = ctx0.key_galois_tool()->get_elt_from_step(steps[i]);
        auto it2 = std::find(gelts.begin(), gelts.end(), elt);
        if (it2 != gelts.end())
            step_to_idx[steps[i]] = static_cast<size_t>(std::distance(gelts.begin(), it2));
    }

    if (mpi_rank == 0)
        printf("[Rank 0] %zu rotation keys to shard across %d total GPUs\n",
               num_keys, total_gpus);

    int rank_offset = dctx.global_rank_offset();

    // ── Build DKS key store (multinode variant) ──
    if (mpi_rank == 0) {
        printf("[Setup] Building DistGaloisKeyStore multinode "
               "(%zu keys, rank_offset=%d, total_gpus=%d)...\n",
               num_keys, rank_offset, total_gpus);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    DistGaloisKeyStore dks;
    dks.generate_multinode(ctx0, sk0, total_gpus, rank_offset, gpus_per_node, num_keys);

    // ── CPU streaming key store (for non-DKS ops — same on every rank) ──
    GaloisKeyStore ks_cpu;
    ks_cpu.generate_all_keys(ctx0, sk0, num_keys);
    eval0.evaluator.enable_key_streaming(&ks_cpu, &gk0);

    // ── Register DKS hook ──
    nexus_multi_gpu::dist_set_galois_key_store(&dks, [&](int step) -> size_t {
        auto it = step_to_idx.find(step);
        if (it == step_to_idx.end())
            throw std::runtime_error("Unknown step in DKS map: " + std::to_string(step));
        return it->second;
    });

    MPI_Barrier(MPI_COMM_WORLD);

    // ── Memory report ──
    if (mpi_rank == 0) {
        printf("[Rank 0] GPU memory after DKS setup:\n");
        for (int g = 0; g < gpus_per_node; g++) {
            cudaSetDevice(g);
            size_t fr, tot;
            cudaMemGetInfo(&fr, &tot);
            printf("  Node 0 GPU %d (global %d): %.1f GB free / %.1f GB total\n",
                   g, rank_offset + g, fr / 1e9, tot / 1e9);
        }
        cudaSetDevice(0);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Encrypt a test ciphertext (same seed on all ranks) ──
    mt19937 rng(42);
    uniform_real_distribution<double> idist(-0.5, 0.5);
    size_t slots = enc0.slot_count();
    vector<double> d(slots, 0.0);
    for (size_t s = 0; s < (size_t)std::min((long)HIDDEN, (long)slots); s++) d[s] = idist(rng);

    PhantomPlaintext pt_in;
    enc0.encode(ctx0, d, SCALE, pt_in);
    PhantomCiphertext X;
    eval0.encryptor.encrypt(pt_in, X);
    // Bootstrap requires coeff_modulus_size() == 1
    while (X.coeff_modulus_size() > 1) eval0.evaluator.mod_switch_to_next_inplace(X);

    // ── Bootstrap timing sweep ──
    if (mpi_rank == 0) {
        printf("\n[Benchmark] DKS bootstrap timing — %d-node × %d-GPU config\n\n",
               mpi_size, gpus_per_node);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Each rank times its own bootstrap (same work, all participate in AllReduce)
    double local_bs_ms = time_dks_bootstrap(bs, X, /*warmup=*/1, /*trials=*/3);

    // Collect all ranks' timings at rank 0
    vector<double> all_times(mpi_size);
    MPI_Gather(&local_bs_ms, 1, MPI_DOUBLE, all_times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        double avg_ms = 0, max_ms = 0;
        for (int r = 0; r < mpi_size; r++) {
            avg_ms += all_times[r];
            max_ms = std::max(max_ms, all_times[r]);
        }
        avg_ms /= mpi_size;

        printf("════════════════════════════════════════════════════════\n");
        printf("  DKS Bootstrap Timing — %d nodes × %d GPUs = %d total\n",
               mpi_size, gpus_per_node, total_gpus);
        printf("════════════════════════════════════════════════════════\n");
        for (int r = 0; r < mpi_size; r++)
            printf("  Node %2d: %.1f ms\n", r, all_times[r]);
        printf("  ──────────────────────────────────────────────────────\n");
        printf("  Average: %.1f ms\n", avg_ms);
        printf("  Max:     %.1f ms  (wall-clock bound)\n", max_ms);

        // Reference comparison
        double single_node_ref = 10730.0;  // 1-GPU CPU-streaming
        double four_gpu_ref    = 2800.0;   // 4-GPU DKS target
        printf("\n  Reference — 1-GPU CPU streaming:   %.0f ms\n", single_node_ref);
        printf("  Reference — 4-GPU DKS (1 node):    %.0f ms\n", four_gpu_ref);
        printf("  This config speedup vs 1-GPU:       %.2fx\n", single_node_ref / max_ms);
        printf("  Key mem/GPU: ~%.1f GB (est., sharding %d/%d digits)\n",
               16.0 * 4.0 / total_gpus,   // scales inversely with total GPUs
               9 / total_gpus + 1, 9);    // rough: beta=9, ceil(9/total_gpus) per GPU

        printf("════════════════════════════════════════════════════════\n");

        // ── Projected 4-bootstrap BERT layer timing ──
        double layer_ms = max_ms * 4;  // 4 bootstraps per BERT layer
        printf("\nProjected BERT layer (4 bootstraps, %d total GPUs):\n", total_gpus);
        printf("  Bootstrap total:   %.1f ms = %.1f s\n", layer_ms, layer_ms / 1000.0);
        printf("  Full 12-head BERT: %.1f s (assuming 12 heads sequential)\n",
               layer_ms / 1000.0 * 12.0 + 5.0);  // 5s for other ops
        fflush(stdout);
    }

    // ── Cleanup ──
    nexus_multi_gpu::dist_set_galois_key_store(nullptr, {});
    dks.destroy();
    dctx.destroy();

    MPI_Finalize();
    return 0;
}

#else  // !USE_MPI

#include <cstdio>
int main() {
    fprintf(stderr, "bert_dks_multinode requires MPI. Build with USE_MPI=ON.\n");
    return 1;
}

#endif // USE_MPI
