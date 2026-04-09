/**
 * spmd_oa_bench.cu
 *
 * SPMD Output Aggregation benchmark — measures REAL compute speedup.
 *
 * Key difference from Input Broadcast:
 *   IB: AllGather c2 → EVERY GPU does FULL keyswitch → no speedup
 *   OA: EACH GPU does PARTIAL inner product (its digits only) → AllReduce → speedup!
 *
 * With beta digits and n GPUs, each GPU processes beta/n digits.
 * The inner product kernel work scales as 1/n → real compute reduction.
 *
 * Pipeline per GPU:
 *   1. mod-up (full — each GPU has all c2)     — same work per GPU
 *   2. partial_key_switch_inner_prod            — 1/n work per GPU ← SPEEDUP
 *   3. AllReduce partial results                — communication
 *   4. mod_reduce + mod-down + add_to_ct        — same work per GPU
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
#include "../multi_gpu/keyswitching/output_aggregation.cuh"

using namespace std;
using namespace phantom;
using namespace nexus_multi_gpu;

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
    size_t poly_degree = 8192;
    size_t n_moduli = 5;
    int warmup = 5;
    int iters = 50;
    bool verbose = false;
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc) cfg.n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i+1 < argc) cfg.poly_degree = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--L") && i+1 < argc) cfg.n_moduli = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc) cfg.warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) cfg.iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose")) cfg.verbose = true;
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

struct GpuState {
    int gpu_id;
    PhantomContext *ctx = nullptr;
    PhantomSecretKey *sk = nullptr;
    PhantomRelinKey *rk = nullptr;
    PhantomCKKSEncoder *encoder = nullptr;
    PhantomCiphertext ct_mul;
    PhantomCiphertext ct_backup;
    size_t total_limbs = 0;
};

// Worker: persistent thread, calls OA keyswitch on signal
void oa_worker(
    GpuState *st,
    MultiGpuContext *mgpu,
    Barrier *bar,
    atomic<int> *phase,
    int n_gpus)
{
    cudaSetDevice(st->gpu_id);

    while (true) {
        bar->wait();
        int p = phase->load();
        if (p == 2) break;

        if (p == 1) {
            // Restore ciphertext (must reset size=3 since OA may have modified it)
            size_t N = st->ctx->get_context_data(0).parms().poly_modulus_degree();
            auto ci = st->ct_backup.chain_index();
            size_t ct_bytes = st->ct_backup.size() * st->total_limbs * N * sizeof(uint64_t);
            st->ct_mul.resize(*st->ctx, ci, st->ct_backup.size(), cudaStreamPerThread);
            cudaMemcpy(st->ct_mul.data(), st->ct_backup.data(), ct_bytes, cudaMemcpyDeviceToDevice);
            st->ct_mul.set_scale(st->ct_backup.scale());
            st->ct_mul.set_ntt_form(true);

            // Extract c2 (3rd polynomial)
            uint64_t *c2 = st->ct_mul.data() + 2 * st->total_limbs * N;

            // Run Output Aggregation
            keyswitching_output_aggregation(
                *mgpu, *st->ctx, st->gpu_id,
                st->ct_mul, c2, *st->rk, n_gpus);
        }

        bar->wait();
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

    printf("=== SPMD Output Aggregation Benchmark (REAL SPEEDUP) ===\n");
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

    // --- NCCL ---
    printf("[1] NCCL init...\n");
    vector<int> dev_ids(cfg.n_gpus);
    for (int i = 0; i < cfg.n_gpus; i++) dev_ids[i] = i;
    MultiGpuContext mgpu = MultiGpuContext::create(dev_ids);
    for (int i = 0; i < cfg.n_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < cfg.n_gpus; j++)
            if (i!=j) { int c=0; cudaDeviceCanAccessPeer(&c,i,j); if(c) cudaDeviceEnablePeerAccess(j,0); }
    }

    // --- Per-GPU setup ---
    printf("[2] Per-GPU setup...\n");
    Timer timer;
    timer.start();

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

    // Encrypt + multiply on GPU 0
    cudaSetDevice(0);
    PhantomCiphertext ct_src;
    {
        PhantomPlaintext pa, pb;
        states[0].encoder->encode(*states[0].ctx, input_a, SCALE, pa);
        states[0].encoder->encode(*states[0].ctx, input_b, SCALE, pb);
        PhantomCiphertext ca, cb;
        states[0].sk->encrypt_symmetric(*states[0].ctx, pa, ca);
        states[0].sk->encrypt_symmetric(*states[0].ctx, pb, cb);
        multiply_inplace(*states[0].ctx, ca, cb);
        ct_src = std::move(ca);
    }

    auto chain_idx = ct_src.chain_index();
    size_t total_limbs = states[0].ctx->get_context_data(chain_idx).gpu_rns_tool().base_Ql().size();
    size_t ct_elems = ct_src.size() * total_limbs * cfg.poly_degree;

    // Print beta (number of key decomposition digits)
    auto &rns_tool = states[0].ctx->get_context_data(1 + (chain_idx - 1)).gpu_rns_tool();
    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();
    printf("   beta (digits) = %zu, total_limbs = %zu\n", beta, total_limbs);
    printf("   With %d GPUs: each GPU processes %zu/%d = ~%zu digits\n",
           cfg.n_gpus, beta, cfg.n_gpus, (beta + cfg.n_gpus - 1) / cfg.n_gpus);

    // Copy ciphertext to all GPUs
    for (int g = 0; g < cfg.n_gpus; g++) {
        cudaSetDevice(g);
        auto &st = states[g];
        st.total_limbs = total_limbs;

        st.ct_mul.resize(*st.ctx, chain_idx, ct_src.size(), cudaStreamPerThread);
        st.ct_mul.set_scale(ct_src.scale());
        st.ct_mul.set_ntt_form(ct_src.is_ntt_form());
        cudaMemcpyPeer(st.ct_mul.data(), g, ct_src.data(), 0, ct_elems * sizeof(uint64_t));

        st.ct_backup.resize(*st.ctx, chain_idx, ct_src.size(), cudaStreamPerThread);
        st.ct_backup.set_scale(ct_src.scale());
        st.ct_backup.set_ntt_form(ct_src.is_ntt_form());
        cudaMemcpy(st.ct_backup.data(), st.ct_mul.data(), ct_elems * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    }
    printf("   Setup in %.1f ms\n", timer.elapsed_ms());

    // --- Ground truth ---
    printf("[3] Ground truth (single-GPU Phantom relinearize)...\n");
    cudaSetDevice(0);
    for (int i = 0; i < cfg.warmup; i++) {
        // Must re-create size=3 ciphertext each time (relinearize changes size to 2)
        states[0].ct_mul.resize(*states[0].ctx, chain_idx, ct_src.size(), cudaStreamPerThread);
        cudaMemcpy(states[0].ct_mul.data(), states[0].ct_backup.data(), ct_elems * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        states[0].ct_mul.set_scale(ct_src.scale()); states[0].ct_mul.set_ntt_form(true);
        relinearize_inplace(*states[0].ctx, states[0].ct_mul, *states[0].rk);
        cudaDeviceSynchronize();
    }
    timer.start();
    for (int i = 0; i < cfg.iters; i++) {
        states[0].ct_mul.resize(*states[0].ctx, chain_idx, ct_src.size(), cudaStreamPerThread);
        cudaMemcpy(states[0].ct_mul.data(), states[0].ct_backup.data(), ct_elems * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        states[0].ct_mul.set_scale(ct_src.scale()); states[0].ct_mul.set_ntt_form(true);
        relinearize_inplace(*states[0].ctx, states[0].ct_mul, *states[0].rk);
        cudaDeviceSynchronize();
    }
    double gt_avg = timer.elapsed_ms() / cfg.iters;
    printf("   Ground truth: %.3f ms avg\n", gt_avg);

    // Decrypt for validation
    PhantomCiphertext ct_gt;
    ct_gt.resize(*states[0].ctx, chain_idx, ct_src.size(), cudaStreamPerThread);
    cudaMemcpy(ct_gt.data(), ct_src.data(), ct_elems * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    ct_gt.set_scale(ct_src.scale()); ct_gt.set_ntt_form(true);
    relinearize_inplace(*states[0].ctx, ct_gt, *states[0].rk);
    ct_gt.resize(*states[0].ctx, chain_idx, 2, cudaStreamPerThread);
    PhantomPlaintext pt_gt;
    states[0].sk->decrypt(*states[0].ctx, ct_gt, pt_gt);
    vector<double> result_gt;
    states[0].encoder->decode(*states[0].ctx, pt_gt, result_gt);

    // --- SPMD Output Aggregation ---
    printf("[4] SPMD Output Aggregation (%d GPUs)...\n", cfg.n_gpus);

    Barrier bar(cfg.n_gpus);
    atomic<int> phase{0};

    // Launch persistent workers
    vector<thread> workers;
    for (int g = 1; g < cfg.n_gpus; g++) {
        workers.emplace_back(oa_worker, &states[g], &mgpu, &bar, &phase, cfg.n_gpus);
    }

    auto gpu0_oa = [&]() {
        cudaSetDevice(0);
        auto &st = states[0];
        size_t N = cfg.poly_degree;
        size_t ct_bytes = st.ct_backup.size() * st.total_limbs * N * sizeof(uint64_t);
        st.ct_mul.resize(*st.ctx, chain_idx, ct_src.size(), cudaStreamPerThread);
        cudaMemcpy(st.ct_mul.data(), st.ct_backup.data(), ct_bytes, cudaMemcpyDeviceToDevice);
        st.ct_mul.set_scale(ct_src.scale());
        st.ct_mul.set_ntt_form(true);
        uint64_t *c2 = st.ct_mul.data() + 2 * st.total_limbs * N;
        keyswitching_output_aggregation(mgpu, *st.ctx, 0, st.ct_mul, c2, *st.rk, cfg.n_gpus);
    };

    // Warmup
    for (int i = 0; i < cfg.warmup; i++) {
        phase.store(1);
        bar.wait();
        gpu0_oa();
        bar.wait();
    }

    // Timed
    timer.start();
    for (int i = 0; i < cfg.iters; i++) {
        phase.store(1);
        bar.wait();
        gpu0_oa();
        bar.wait();
    }
    cudaDeviceSynchronize();
    double oa_avg = timer.elapsed_ms() / cfg.iters;

    // Shutdown
    phase.store(2);
    bar.wait();
    for (auto &w : workers) w.join();

    printf("   Output Aggregation: %.3f ms avg\n", oa_avg);

    // Validate
    {
        cudaSetDevice(0);
        states[0].ct_mul.resize(*states[0].ctx, chain_idx, 2, cudaStreamPerThread);
        PhantomPlaintext pt;
        states[0].sk->decrypt(*states[0].ctx, states[0].ct_mul, pt);
        vector<double> result;
        states[0].encoder->decode(*states[0].ctx, pt, result);
        double mae = 0;
        for (size_t i = 0; i < slots; i++) mae += fabs(result[i] - result_gt[i]);
        mae /= slots;
        printf("   MAE: %.2e  %s\n", mae, mae < 1e-3 ? "PASS" : "FAIL");
    }

    // --- Summary ---
    double speedup = gt_avg / oa_avg;
    printf("\n=== Results ===\n");
    printf("beta (digits):               %zu\n", beta);
    printf("Ground truth (1 GPU):        %.3f ms\n", gt_avg);
    printf("Output Aggregation (%d GPUs): %.3f ms\n", cfg.n_gpus, oa_avg);
    printf("Speedup:                     %.2fx\n", speedup);
    printf("Efficiency:                  %.1f%%\n", speedup / cfg.n_gpus * 100.0);
    printf("Compute per GPU:             ~%zu/%d digits = %.1f%% of single-GPU\n",
           beta, cfg.n_gpus, 100.0 / cfg.n_gpus);

    // Cleanup
    for (auto &st : states) {
        delete st.encoder; delete st.rk; delete st.sk; delete st.ctx;
    }
    mgpu.destroy();
    printf("\nDone.\n");
    return 0;
}
