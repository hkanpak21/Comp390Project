/**
 * spmd_keyswitch_bench.cu
 *
 * TRUE SPMD multi-GPU key-switching benchmark with persistent threads.
 *
 * Architecture:
 *   - Worker threads are created ONCE and stay alive for all iterations
 *   - Setup (encrypt, multiply, scatter) happens ONCE before timing
 *   - Timing measures ONLY the key-switching call (AllGather + keyswitch_inplace)
 *   - Barriers synchronize start/end of each timed iteration
 *
 * This eliminates thread creation overhead (~1ms), ciphertext setup (~5-30ms),
 * and cudaMemcpyPeer (~1ms) from the timing.
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

// --- Barrier ---
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
    int warmup = 5;
    int iters = 50;
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc) cfg.n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose")) cfg.verbose = true;
        else if (!strcmp(argv[i], "--N") && i+1 < argc) cfg.poly_degree = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--L") && i+1 < argc) cfg.n_moduli = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc) cfg.warmup = atoi(argv[++i]);
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

// Per-GPU persistent state
struct GpuState {
    int gpu_id;
    PhantomContext *ctx = nullptr;
    PhantomSecretKey *sk = nullptr;
    PhantomRelinKey *rk = nullptr;
    PhantomCKKSEncoder *encoder = nullptr;
    PhantomCiphertext ct_mul;         // size=3 ciphertext
    PhantomCiphertext ct_mul_backup;  // backup for re-use across iterations
    uint64_t *local_c2 = nullptr;
    size_t local_limbs = 0;
    size_t total_limbs = 0;
};

// Worker thread function: stays alive, waits for signals
void worker_fn(
    GpuState *st,
    MultiGpuContext *mgpu,
    Barrier *bar,
    atomic<int> *phase,  // 0=wait, 1=keyswitch, 2=exit
    int n_gpus,
    size_t poly_degree)
{
    cudaSetDevice(st->gpu_id);

    while (true) {
        bar->wait();  // wait for signal from main thread
        int p = phase->load();
        if (p == 2) break;  // exit

        if (p == 1) {
            // Restore ciphertext from backup (key-switching modifies in-place)
            size_t ct_bytes = st->ct_mul_backup.size() * st->total_limbs * poly_degree * sizeof(uint64_t);
            cudaMemcpy(st->ct_mul.data(), st->ct_mul_backup.data(), ct_bytes, cudaMemcpyDeviceToDevice);

            // Run key-switching
            keyswitching_input_broadcast(
                *mgpu, *st->ctx, st->gpu_id,
                st->ct_mul, st->local_c2, *st->rk,
                st->total_limbs, poly_degree, n_gpus);
        }

        bar->wait();  // signal completion
    }
}

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < cfg.n_gpus) {
        fprintf(stderr, "Need %d GPUs, have %d\n", cfg.n_gpus, dev_count);
        return 1;
    }

    printf("=== SPMD Multi-GPU Key-Switching Benchmark ===\n");
    printf("GPUs: %d, N=%zu, L=%zu, warmup=%d, iters=%d\n\n",
           cfg.n_gpus, cfg.poly_degree, cfg.n_moduli, cfg.warmup, cfg.iters);

    // --- Parameters ---
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(cfg.poly_degree);
    vector<int> bits = {60};
    for (size_t i = 0; i < cfg.n_moduli - 2; i++) bits.push_back(40);
    bits.push_back(60);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(cfg.poly_degree, bits));

    const double SCALE = (double)(1ULL << 40);
    size_t slots = cfg.poly_degree / 2;
    vector<double> input_a(slots), input_b(slots);
    srand(42);
    for (size_t i = 0; i < slots; i++) {
        input_a[i] = (rand() % 1000) / 100.0 - 5.0;
        input_b[i] = (rand() % 1000) / 100.0 - 5.0;
    }

    // --- NCCL init ---
    printf("[1] NCCL init...\n");
    vector<int> dev_ids(cfg.n_gpus);
    for (int i = 0; i < cfg.n_gpus; i++) dev_ids[i] = i;
    MultiGpuContext mgpu = MultiGpuContext::create(dev_ids);

    // Enable peer access
    for (int i = 0; i < cfg.n_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < cfg.n_gpus; j++) {
            if (i != j) { int c=0; cudaDeviceCanAccessPeer(&c,i,j); if(c) cudaDeviceEnablePeerAccess(j,0); }
        }
    }

    // --- Per-GPU setup (one-time) ---
    printf("[2] Per-GPU setup (contexts, keys, ciphertexts)...\n");
    Timer timer;
    timer.start();

    // Secret key: generate on GPU 0, serialize, load on all GPUs
    vector<GpuState> states(cfg.n_gpus);
    stringstream sk_buf;

    for (int g = 0; g < cfg.n_gpus; g++) {
        cudaSetDevice(g);
        states[g].gpu_id = g;
        states[g].ctx = new PhantomContext(parms);
        if (g == 0) {
            states[g].sk = new PhantomSecretKey(*states[g].ctx);
            states[g].sk->save(sk_buf);
        } else {
            sk_buf.seekg(0);
            states[g].sk = new PhantomSecretKey();
            states[g].sk->load(sk_buf);
        }
        states[g].rk = new PhantomRelinKey(states[g].sk->gen_relinkey(*states[g].ctx));
        states[g].encoder = new PhantomCKKSEncoder(*states[g].ctx);
    }

    // Encrypt + multiply on GPU 0 only
    cudaSetDevice(0);
    PhantomCiphertext ct_mul_src;
    {
        PhantomPlaintext pa, pb;
        states[0].encoder->encode(*states[0].ctx, input_a, SCALE, pa);
        states[0].encoder->encode(*states[0].ctx, input_b, SCALE, pb);
        PhantomCiphertext ca, cb;
        states[0].sk->encrypt_symmetric(*states[0].ctx, pa, ca);
        states[0].sk->encrypt_symmetric(*states[0].ctx, pb, cb);
        multiply_inplace(*states[0].ctx, ca, cb);
        ct_mul_src = std::move(ca);
    }

    auto chain_idx = ct_mul_src.chain_index();
    size_t total_limbs = states[0].ctx->get_context_data(chain_idx).gpu_rns_tool().base_Ql().size();
    size_t ct_data_elems = ct_mul_src.size() * total_limbs * cfg.poly_degree;

    // Copy ciphertext to each GPU and scatter c2 limbs
    for (int g = 0; g < cfg.n_gpus; g++) {
        cudaSetDevice(g);
        auto &st = states[g];
        st.total_limbs = total_limbs;
        st.local_limbs = n_local_limbs(g, cfg.n_gpus, total_limbs);

        // Full ciphertext copy
        st.ct_mul.resize(*st.ctx, chain_idx, ct_mul_src.size(), cudaStreamPerThread);
        st.ct_mul.set_scale(ct_mul_src.scale());
        st.ct_mul.set_ntt_form(ct_mul_src.is_ntt_form());
        cudaMemcpyPeer(st.ct_mul.data(), g, ct_mul_src.data(), 0,
                       ct_data_elems * sizeof(uint64_t));

        // Backup copy (for restoring between iterations)
        st.ct_mul_backup.resize(*st.ctx, chain_idx, ct_mul_src.size(), cudaStreamPerThread);
        st.ct_mul_backup.set_scale(ct_mul_src.scale());
        st.ct_mul_backup.set_ntt_form(ct_mul_src.is_ntt_form());
        cudaMemcpy(st.ct_mul_backup.data(), st.ct_mul.data(),
                   ct_data_elems * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

        // Scatter c2 limbs
        size_t max_local = (total_limbs + cfg.n_gpus - 1) / cfg.n_gpus;
        size_t local_bytes = max_local * cfg.poly_degree * sizeof(uint64_t);
        cudaMalloc(&st.local_c2, local_bytes);
        cudaMemset(st.local_c2, 0, local_bytes);

        uint64_t *c2_full = st.ct_mul.data() + 2 * total_limbs * cfg.poly_degree;
        size_t loc = 0;
        for (size_t j = 0; j < total_limbs; j++) {
            if (owner_of_limb(j, cfg.n_gpus) != g) continue;
            cudaMemcpy(st.local_c2 + loc * cfg.poly_degree,
                       c2_full + j * cfg.poly_degree,
                       cfg.poly_degree * sizeof(uint64_t),
                       cudaMemcpyDeviceToDevice);
            loc++;
        }
    }
    cudaSetDevice(0);
    printf("   Setup in %.1f ms\n", timer.elapsed_ms());

    // --- Ground truth ---
    printf("[3] Ground truth (single-GPU relinearize, %d iters)...\n", cfg.iters);
    // Warmup
    for (int i = 0; i < cfg.warmup; i++) {
        PhantomCiphertext tmp;
        tmp.resize(*states[0].ctx, chain_idx, ct_mul_src.size(), cudaStreamPerThread);
        cudaMemcpy(tmp.data(), ct_mul_src.data(), ct_data_elems * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        tmp.set_scale(ct_mul_src.scale()); tmp.set_ntt_form(true);
        relinearize_inplace(*states[0].ctx, tmp, *states[0].rk);
        cudaDeviceSynchronize();
    }
    // Timed
    timer.start();
    for (int i = 0; i < cfg.iters; i++) {
        PhantomCiphertext tmp;
        tmp.resize(*states[0].ctx, chain_idx, ct_mul_src.size(), cudaStreamPerThread);
        cudaMemcpy(tmp.data(), ct_mul_src.data(), ct_data_elems * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        tmp.set_scale(ct_mul_src.scale()); tmp.set_ntt_form(true);
        relinearize_inplace(*states[0].ctx, tmp, *states[0].rk);
        cudaDeviceSynchronize();
    }
    double gt_total = timer.elapsed_ms();
    double gt_avg = gt_total / cfg.iters;
    printf("   Ground truth: %.3f ms avg (%.1f ms total)\n", gt_avg, gt_total);

    // Decrypt ground truth for validation
    PhantomCiphertext ct_gt;
    ct_gt.resize(*states[0].ctx, chain_idx, ct_mul_src.size(), cudaStreamPerThread);
    cudaMemcpy(ct_gt.data(), ct_mul_src.data(), ct_data_elems * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    ct_gt.set_scale(ct_mul_src.scale()); ct_gt.set_ntt_form(true);
    relinearize_inplace(*states[0].ctx, ct_gt, *states[0].rk);
    ct_gt.resize(*states[0].ctx, chain_idx, 2, cudaStreamPerThread);
    PhantomPlaintext pt_gt;
    states[0].sk->decrypt(*states[0].ctx, ct_gt, pt_gt);
    vector<double> result_gt;
    states[0].encoder->decode(*states[0].ctx, pt_gt, result_gt);

    // --- SPMD Input Broadcast with persistent threads ---
    printf("[4] SPMD Input Broadcast (%d GPUs, %d warmup + %d timed)...\n",
           cfg.n_gpus, cfg.warmup, cfg.iters);

    Barrier bar(cfg.n_gpus);
    atomic<int> phase{0};

    // Launch persistent worker threads for GPUs 1..n-1
    vector<thread> workers;
    for (int g = 1; g < cfg.n_gpus; g++) {
        workers.emplace_back(worker_fn, &states[g], &mgpu, &bar, &phase,
                             cfg.n_gpus, cfg.poly_degree);
    }

    // GPU 0 acts as the main thread + worker for GPU 0
    auto gpu0_keyswitch = [&]() {
        cudaSetDevice(0);
        auto &st = states[0];
        size_t ct_bytes = st.ct_mul_backup.size() * st.total_limbs * cfg.poly_degree * sizeof(uint64_t);
        cudaMemcpy(st.ct_mul.data(), st.ct_mul_backup.data(), ct_bytes, cudaMemcpyDeviceToDevice);
        keyswitching_input_broadcast(
            mgpu, *st.ctx, 0, st.ct_mul, st.local_c2, *st.rk,
            st.total_limbs, cfg.poly_degree, cfg.n_gpus);
    };

    // Warmup
    for (int i = 0; i < cfg.warmup; i++) {
        phase.store(1);
        bar.wait();       // release workers
        gpu0_keyswitch(); // GPU 0 does its part
        bar.wait();       // wait for all
    }

    // Timed iterations
    timer.start();
    for (int i = 0; i < cfg.iters; i++) {
        phase.store(1);
        bar.wait();
        gpu0_keyswitch();
        bar.wait();
    }
    cudaDeviceSynchronize();
    double ib_total = timer.elapsed_ms();
    double ib_avg = ib_total / cfg.iters;

    // Shutdown workers
    phase.store(2);
    bar.wait();
    for (auto &w : workers) w.join();

    printf("   Input Broadcast: %.3f ms avg (%.1f ms total)\n", ib_avg, ib_total);

    // Validate
    {
        cudaSetDevice(0);
        auto &st = states[0];
        st.ct_mul.resize(*st.ctx, chain_idx, 2, cudaStreamPerThread);
        PhantomPlaintext pt;
        st.sk->decrypt(*st.ctx, st.ct_mul, pt);
        vector<double> result;
        st.encoder->decode(*st.ctx, pt, result);
        double mae = 0;
        for (size_t i = 0; i < slots; i++) mae += fabs(result[i] - result_gt[i]);
        mae /= slots;
        printf("   MAE vs ground truth: %.2e  %s\n", mae, mae < 1e-3 ? "PASS" : "FAIL");
    }

    // --- Summary ---
    double speedup = gt_avg / ib_avg;
    printf("\n=== Results ===\n");
    printf("Ground truth (1 GPU):        %.3f ms avg\n", gt_avg);
    printf("Input Broadcast (%d GPUs):   %.3f ms avg\n", cfg.n_gpus, ib_avg);
    printf("Speedup:                     %.2fx\n", speedup);
    printf("Efficiency:                  %.1f%%\n", speedup / cfg.n_gpus * 100.0);

    // Cleanup
    for (auto &st : states) {
        cudaSetDevice(st.gpu_id);
        if (st.local_c2) cudaFree(st.local_c2);
        delete st.encoder; delete st.rk; delete st.sk; delete st.ctx;
    }
    mgpu.destroy();

    printf("\nDone.\n");
    return 0;
}
