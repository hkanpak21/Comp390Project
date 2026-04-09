/**
 * spmd_keyswitch_bench.cu
 *
 * TRUE SPMD multi-GPU key-switching benchmark.
 *
 * Architecture: one std::thread per GPU. Each thread:
 *   1. cudaSetDevice(gpu_id)
 *   2. Creates its own PhantomContext (GPU-local NTT tables, RNS tools)
 *   3. Generates keys independently (same secret → same keys)
 *   4. Encrypts, multiplies, gets c2
 *   5. Scatters c2 limbs to its local buffer
 *   6. ALL threads call keyswitching_input_broadcast SIMULTANEOUSLY
 *      (each with its own gpu_id → all participate in same NCCL collective)
 *   7. GPU 0 gathers result, decrypts, validates against ground truth
 *
 * This is the correct SPMD pattern that avoids NCCL deadlocks.
 *
 * Usage:
 *   ./spmd_keyswitch_bench --n-gpus 4 [--verbose]
 */

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <sstream>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

#include "../multi_gpu/comm/nccl_comm.cuh"
#include "../multi_gpu/partition/rns_partition.cuh"
#include "../multi_gpu/keyswitching/input_broadcast.cuh"
#include "../multi_gpu/keyswitching/output_aggregation.cuh"

using namespace std;
using namespace phantom;
using namespace nexus_multi_gpu;

// Simple barrier (C++17 compatible)
class Barrier {
    int n_, count_;
    int gen_ = 0;
    mutex mtx_;
    condition_variable cv_;
public:
    explicit Barrier(int n) : n_(n), count_(n) {}
    void wait() {
        unique_lock<mutex> lk(mtx_);
        int g = gen_;
        if (--count_ == 0) { gen_++; count_ = n_; cv_.notify_all(); }
        else { cv_.wait(lk, [&]{ return g != gen_; }); }
    }
};

struct Config {
    int n_gpus = 2;
    bool verbose = false;
    size_t poly_degree = 8192;
    size_t n_moduli = 5;
    int iters = 10;
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc) cfg.n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose")) cfg.verbose = true;
        else if (!strcmp(argv[i], "--N") && i+1 < argc) cfg.poly_degree = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--L") && i+1 < argc) cfg.n_moduli = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) cfg.iters = atoi(argv[++i]);
    }
    return cfg;
}

struct Timer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        return chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
    }
};

// Per-GPU state
struct GpuState {
    int gpu_id;
    PhantomContext *ctx;          // owned, GPU-local
    PhantomSecretKey *sk;
    PhantomRelinKey *rk;
    PhantomCKKSEncoder *encoder;

    // Ciphertext after multiply (size=3, has c2)
    PhantomCiphertext ct_mul;

    // Local limb buffers for distributed c2
    uint64_t *local_c2 = nullptr;
    size_t local_limbs = 0;
    size_t total_limbs = 0;

    // Timing results
    double ib_time_ms = 0;
    double oa_time_ms = 0;

    ~GpuState() {
        delete encoder;
        delete rk;
        delete sk;
        delete ctx;
        if (local_c2) { cudaSetDevice(gpu_id); cudaFree(local_c2); }
    }
};

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < cfg.n_gpus) {
        fprintf(stderr, "Need %d GPUs, have %d\n", cfg.n_gpus, dev_count);
        return 1;
    }

    printf("=== SPMD Multi-GPU Key-Switching Benchmark ===\n");
    printf("GPUs: %d, N=%zu, L=%zu, iters=%d\n\n", cfg.n_gpus,
           cfg.poly_degree, cfg.n_moduli, cfg.iters);

    // ---- Build encryption parameters ----
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(cfg.poly_degree);
    vector<int> bits = {60};
    for (size_t i = 0; i < cfg.n_moduli - 2; i++) bits.push_back(40);
    bits.push_back(60);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(cfg.poly_degree, bits));

    // ---- NCCL init (must be before per-GPU PhantomContext creation) ----
    printf("[1] Initializing NCCL for %d GPUs...\n", cfg.n_gpus);
    vector<int> dev_ids(cfg.n_gpus);
    for (int i = 0; i < cfg.n_gpus; i++) dev_ids[i] = i;
    MultiGpuContext mgpu = MultiGpuContext::create(dev_ids);
    printf("   NCCL OK\n");

    // ---- Enable peer access ----
    for (int i = 0; i < cfg.n_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < cfg.n_gpus; j++) {
            if (i != j) {
                int can = 0;
                cudaDeviceCanAccessPeer(&can, i, j);
                if (can) cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }

    // ---- Create per-GPU state (PhantomContext + keys) ----
    printf("[2] Creating per-GPU PhantomContexts and keys...\n");
    Timer timer;
    timer.start();

    vector<GpuState*> states(cfg.n_gpus);
    size_t slots = cfg.poly_degree / 2;
    const double SCALE = (double)(1ULL << 40);

    // Random input (same on all GPUs for validation)
    vector<double> input_a(slots), input_b(slots);
    srand(42);
    for (size_t i = 0; i < slots; i++) {
        input_a[i] = (rand() % 1000) / 100.0 - 5.0;
        input_b[i] = (rand() % 1000) / 100.0 - 5.0;
    }

    // Initialize each GPU sequentially (PhantomContext creation is not thread-safe)
    // Key insight: ALL GPUs must use the SAME secret key. We generate on GPU 0,
    // serialize, then load on each other GPU.
    stringstream sk_stream;
    for (int g = 0; g < cfg.n_gpus; g++) {
        cudaSetDevice(g);
        auto *st = new GpuState();
        st->gpu_id = g;
        st->ctx = new PhantomContext(parms);

        if (g == 0) {
            // Generate secret key on GPU 0 and serialize
            st->sk = new PhantomSecretKey(*st->ctx);
            st->sk->save(sk_stream);
        } else {
            // Load same secret key on other GPUs
            sk_stream.seekg(0);
            st->sk = new PhantomSecretKey();
            st->sk->load(sk_stream);
        }

        st->rk = new PhantomRelinKey(st->sk->gen_relinkey(*st->ctx));
        st->encoder = new PhantomCKKSEncoder(*st->ctx);

        st->encoder = new PhantomCKKSEncoder(*st->ctx);
        states[g] = st;
    }

    // Encrypt and multiply on GPU 0 ONLY, then copy the full ciphertext to each GPU
    cudaSetDevice(0);
    PhantomCiphertext ct_mul_gpu0;
    {
        PhantomPlaintext pa, pb;
        states[0]->encoder->encode(*states[0]->ctx, input_a, SCALE, pa);
        states[0]->encoder->encode(*states[0]->ctx, input_b, SCALE, pb);
        PhantomCiphertext ca, cb;
        states[0]->sk->encrypt_symmetric(*states[0]->ctx, pa, ca);
        states[0]->sk->encrypt_symmetric(*states[0]->ctx, pb, cb);
        multiply_inplace(*states[0]->ctx, ca, cb);
        ct_mul_gpu0 = std::move(ca);
    }

    auto chain_idx_init = ct_mul_gpu0.chain_index();
    size_t total_limbs_init = states[0]->ctx->get_context_data(chain_idx_init).gpu_rns_tool().base_Ql().size();
    size_t ct_data_size = ct_mul_gpu0.size() * total_limbs_init * cfg.poly_degree;

    // Copy the full ciphertext to each GPU and scatter local c2 limbs
    for (int g = 0; g < cfg.n_gpus; g++) {
        auto *st = states[g];
        cudaSetDevice(g);

        // Copy full ciphertext to this GPU
        st->ct_mul.resize(*st->ctx, chain_idx_init, ct_mul_gpu0.size(), cudaStreamPerThread);
        st->ct_mul.set_scale(ct_mul_gpu0.scale());
        st->ct_mul.set_ntt_form(ct_mul_gpu0.is_ntt_form());
        cudaMemcpyPeer(st->ct_mul.data(), g, ct_mul_gpu0.data(), 0,
                       ct_data_size * sizeof(uint64_t));

        st->total_limbs = total_limbs_init;
        st->local_limbs = n_local_limbs(g, cfg.n_gpus, st->total_limbs);

        // Scatter c2 limbs (c2 = poly index 2)
        size_t local_bytes = st->local_limbs * cfg.poly_degree * sizeof(uint64_t);
        if (local_bytes > 0) {
            cudaMalloc(&st->local_c2, local_bytes);
            uint64_t *c2_full = st->ct_mul.data() + 2 * st->total_limbs * cfg.poly_degree;
            size_t loc = 0;
            for (size_t j = 0; j < st->total_limbs; j++) {
                if (owner_of_limb(j, cfg.n_gpus) != g) continue;
                cudaMemcpy(st->local_c2 + loc * cfg.poly_degree,
                           c2_full + j * cfg.poly_degree,
                           cfg.poly_degree * sizeof(uint64_t),
                           cudaMemcpyDeviceToDevice);
                loc++;
            }
        }
    }
    cudaSetDevice(0);
    printf("   Per-GPU init in %.1f ms\n", timer.elapsed_ms());

    // ---- Ground truth on GPU 0 ----
    printf("[3] Computing ground truth (GPU 0, single-GPU relinearize)...\n");
    timer.start();
    PhantomCiphertext ct_gt;
    {
        cudaSetDevice(0);
        PhantomPlaintext pa, pb;
        states[0]->encoder->encode(*states[0]->ctx, input_a, SCALE, pa);
        states[0]->encoder->encode(*states[0]->ctx, input_b, SCALE, pb);
        PhantomCiphertext ca, cb;
        states[0]->sk->encrypt_symmetric(*states[0]->ctx, pa, ca);
        states[0]->sk->encrypt_symmetric(*states[0]->ctx, pb, cb);
        multiply_inplace(*states[0]->ctx, ca, cb);
        ct_gt = std::move(ca);
    }
    double gt_time = 0;
    for (int it = 0; it < cfg.iters; it++) {
        PhantomCiphertext tmp = ct_gt;
        // Need to re-create since relinearize modifies in-place
        {
            PhantomPlaintext pa, pb;
            states[0]->encoder->encode(*states[0]->ctx, input_a, SCALE, pa);
            states[0]->encoder->encode(*states[0]->ctx, input_b, SCALE, pb);
            PhantomCiphertext ca, cb;
            states[0]->sk->encrypt_symmetric(*states[0]->ctx, pa, ca);
            states[0]->sk->encrypt_symmetric(*states[0]->ctx, pb, cb);
            multiply_inplace(*states[0]->ctx, ca, cb);
            tmp = std::move(ca);
        }
        timer.start();
        relinearize_inplace(*states[0]->ctx, tmp, *states[0]->rk);
        cudaDeviceSynchronize();
        gt_time += timer.elapsed_ms();
        if (it == cfg.iters - 1) ct_gt = std::move(tmp);
    }
    gt_time /= cfg.iters;
    printf("   Ground truth: %.3f ms avg\n", gt_time);

    // Decrypt ground truth
    PhantomPlaintext pt_gt;
    states[0]->sk->decrypt(*states[0]->ctx, ct_gt, pt_gt);
    vector<double> result_gt;
    states[0]->encoder->decode(*states[0]->ctx, pt_gt, result_gt);

    // ---- SPMD Input Broadcast benchmark ----
    printf("[4] SPMD Input Broadcast (%d GPUs, %d iters)...\n", cfg.n_gpus, cfg.iters);

    Barrier bar(cfg.n_gpus);
    atomic<double> total_ib_time{0};

    // Each iteration: all GPUs re-create ciphertext, then keyswitch in parallel
    for (int it = 0; it < cfg.iters; it++) {
        // Re-create ct_mul on GPU 0 only, then copy to all GPUs
        cudaSetDevice(0);
        {
            PhantomPlaintext pa, pb;
            states[0]->encoder->encode(*states[0]->ctx, input_a, SCALE, pa);
            states[0]->encoder->encode(*states[0]->ctx, input_b, SCALE, pb);
            PhantomCiphertext ca, cb;
            states[0]->sk->encrypt_symmetric(*states[0]->ctx, pa, ca);
            states[0]->sk->encrypt_symmetric(*states[0]->ctx, pb, cb);
            multiply_inplace(*states[0]->ctx, ca, cb);
            ct_mul_gpu0 = std::move(ca);
        }

        // Distribute to all GPUs
        for (int g = 0; g < cfg.n_gpus; g++) {
            auto *st = states[g];
            cudaSetDevice(g);
            cudaMemcpyPeer(st->ct_mul.data(), g, ct_mul_gpu0.data(), 0,
                           ct_data_size * sizeof(uint64_t));
            st->ct_mul.set_scale(ct_mul_gpu0.scale());

            // Re-scatter c2 limbs
            uint64_t *c2_full = st->ct_mul.data() + 2 * st->total_limbs * cfg.poly_degree;
            size_t loc = 0;
            for (size_t j = 0; j < st->total_limbs; j++) {
                if (owner_of_limb(j, cfg.n_gpus) != g) continue;
                cudaMemcpy(st->local_c2 + loc * cfg.poly_degree,
                           c2_full + j * cfg.poly_degree,
                           cfg.poly_degree * sizeof(uint64_t),
                           cudaMemcpyDeviceToDevice);
                loc++;
            }
        }

        // SPMD: all GPUs call keyswitching_input_broadcast simultaneously
        timer.start();
        vector<thread> threads;
        for (int g = 0; g < cfg.n_gpus; g++) {
            threads.emplace_back([&, g]() {
                auto *st = states[g];
                cudaSetDevice(g);

                bar.wait();  // synchronize start

                keyswitching_input_broadcast(
                    mgpu, *st->ctx, g,
                    st->ct_mul, st->local_c2, *st->rk,
                    st->total_limbs, cfg.poly_degree, cfg.n_gpus);

                bar.wait();  // synchronize end
            });
        }
        for (auto &t : threads) t.join();
        double elapsed = timer.elapsed_ms();
        total_ib_time = total_ib_time.load() + elapsed;
    }

    double avg_ib = total_ib_time.load() / cfg.iters;
    printf("   Input Broadcast: %.3f ms avg\n", avg_ib);

    // Validate: decrypt GPU 0's result
    {
        cudaSetDevice(0);
        auto *st = states[0];
        st->ct_mul.resize(*st->ctx, st->ct_mul.chain_index(), 2, cudaStreamPerThread);
        PhantomPlaintext pt;
        st->sk->decrypt(*st->ctx, st->ct_mul, pt);
        vector<double> result;
        st->encoder->decode(*st->ctx, pt, result);

        double mae = 0;
        for (size_t i = 0; i < slots; i++) mae += fabs(result[i] - result_gt[i]);
        mae /= slots;
        printf("   MAE vs ground truth: %.2e  %s\n", mae, mae < 1e-3 ? "PASS" : "FAIL");
    }

    // ---- Summary ----
    printf("\n=== Results ===\n");
    printf("Ground truth (1 GPU):        %.3f ms\n", gt_time);
    printf("Input Broadcast (%d GPUs):   %.3f ms  (%.2fx)\n",
           cfg.n_gpus, avg_ib, gt_time / avg_ib);

    // Cleanup
    for (auto *st : states) delete st;
    mgpu.destroy();

    printf("\nDone.\n");
    return 0;
}
