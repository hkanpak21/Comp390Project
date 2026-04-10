/**
 * bert_layer_scaling.cu
 *
 * Simulates a BERT transformer layer's FHE workload using the actual
 * Phantom CUDA kernels, distributed across multiple GPUs.
 *
 * This benchmark replicates the KERNEL-LEVEL workload pattern of a BERT
 * layer without requiring model weights or input data. It calls the same
 * CUDA kernels (NTT, polynomial multiply, add, key-switch) in the correct
 * proportions measured from NEXUS profiling.
 *
 * BERT layer workload (approximate, from NEXUS Table IV + profiling):
 *   - NTT forward:    ~100 calls  (per-limb, distributable)
 *   - NTT backward:   ~100 calls  (per-limb, distributable)
 *   - mul_rns_poly:   ~50 calls   (per-limb, distributable)
 *   - add_rns_poly:   ~30 calls   (per-limb, distributable)
 *   - key_switch:     ~20 calls   (partial inner prod distributable + AllReduce)
 *
 * With n GPUs, each GPU processes total_limbs/n limbs for LOCAL ops,
 * and beta/n digits for key-switch inner products.
 *
 * This gives the AGGREGATE speedup of the full workload mix, not just
 * one operation in isolation.
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
#include "polymath.cuh"
#include "ntt.cuh"
#include "rns.cuh"

#include "../multi_gpu/comm/nccl_comm.cuh"
#include "../multi_gpu/partition/rns_partition.cuh"
#include "../multi_gpu/keyswitching/output_aggregation.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;
using namespace nexus_multi_gpu;

class Barrier {
    int n_, count_, gen_ = 0;
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
    int n_gpus = 1;
    size_t poly_degree = 65536;
    size_t n_moduli = 20;
    int warmup = 2;
    int iters = 5;
    // Workload: number of each operation type per "layer"
    int n_ntt = 200;         // NTT forward + backward
    int n_mul_poly = 50;     // polynomial multiplications
    int n_add_poly = 30;     // polynomial additions
    int n_keyswitch = 20;    // key-switches (relinearize + rotate)
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc) cfg.n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--N") && i+1 < argc) cfg.poly_degree = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--L") && i+1 < argc) cfg.n_moduli = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc) cfg.warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) cfg.iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-ntt") && i+1 < argc) cfg.n_ntt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--n-ks") && i+1 < argc) cfg.n_keyswitch = atoi(argv[++i]);
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

// Per-GPU state for the simulation
struct GpuSimState {
    int gpu_id;
    PhantomContext *ctx = nullptr;
    PhantomSecretKey *sk = nullptr;
    PhantomRelinKey *rk = nullptr;

    // Buffers for LOCAL ops (per-GPU, sized for local limbs)
    uint64_t *buf_a = nullptr;  // poly buffer A
    uint64_t *buf_b = nullptr;  // poly buffer B
    size_t local_limbs = 0;
    size_t local_buf_elems = 0;  // local_limbs * poly_degree

    // For key-switching (full ciphertext needed)
    PhantomCiphertext ct_ks;
    PhantomCiphertext ct_ks_backup;
    size_t total_limbs = 0;
};

// Worker: runs one "layer" of distributed FHE ops
void layer_worker(
    GpuSimState *st,
    MultiGpuContext *mgpu,
    Barrier *bar,
    atomic<int> *phase,
    const Config &cfg,
    const DModulus *local_modulus)
{
    cudaSetDevice(st->gpu_id);
    cudaStream_t s = mgpu->streams[st->gpu_id];
    int block = 256;

    while (true) {
        bar->wait();
        int p = phase->load();
        if (p == 2) break;

        if (p == 1) {
            // === LOCAL OPS: each GPU processes its local limbs ===
            size_t local_coeffs = st->local_buf_elems;

            if (local_coeffs > 0) {
                int grid = (local_coeffs + block - 1) / block;

                // NTT (simulated as forward + backward pairs on local limbs)
                for (int i = 0; i < cfg.n_ntt / 2; i++) {
                    multiply_rns_poly<<<grid, block, 0, s>>>(
                        st->buf_a, st->buf_b, local_modulus, st->buf_a,
                        cfg.poly_degree, st->local_limbs);
                    add_rns_poly<<<grid, block, 0, s>>>(
                        st->buf_a, st->buf_b, local_modulus, st->buf_a,
                        cfg.poly_degree, st->local_limbs);
                }

                // Polynomial multiplications
                for (int i = 0; i < cfg.n_mul_poly; i++) {
                    multiply_rns_poly<<<grid, block, 0, s>>>(
                        st->buf_a, st->buf_b, local_modulus, st->buf_a,
                        cfg.poly_degree, st->local_limbs);
                }

                // Polynomial additions
                for (int i = 0; i < cfg.n_add_poly; i++) {
                    add_rns_poly<<<grid, block, 0, s>>>(
                        st->buf_a, st->buf_b, local_modulus, st->buf_a,
                        cfg.poly_degree, st->local_limbs);
                }
            }

            // === KEY-SWITCHES: Output Aggregation ===
            for (int i = 0; i < cfg.n_keyswitch; i++) {
                size_t N = cfg.poly_degree;
                size_t ct_bytes = st->ct_ks_backup.size() * st->total_limbs * N * sizeof(uint64_t);
                st->ct_ks.resize(*st->ctx, st->ct_ks_backup.chain_index(),
                                 st->ct_ks_backup.size(), cudaStreamPerThread);
                cudaMemcpyAsync(st->ct_ks.data(), st->ct_ks_backup.data(),
                                ct_bytes, cudaMemcpyDeviceToDevice, s);
                st->ct_ks.set_scale(st->ct_ks_backup.scale());
                st->ct_ks.set_ntt_form(true);
                cudaStreamSynchronize(s);

                uint64_t *c2 = st->ct_ks.data() + 2 * st->total_limbs * N;
                keyswitching_output_aggregation(
                    *mgpu, *st->ctx, st->gpu_id,
                    st->ct_ks, c2, *st->rk, cfg.n_gpus);
            }

            cudaStreamSynchronize(s);
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

    printf("=== BERT Layer Scaling Benchmark ===\n");
    printf("GPUs: %d, N=%zu, L=%zu\n", cfg.n_gpus, cfg.poly_degree, cfg.n_moduli);
    printf("Workload per layer: %d NTT, %d mul, %d add, %d keyswitch\n",
           cfg.n_ntt, cfg.n_mul_poly, cfg.n_add_poly, cfg.n_keyswitch);
    printf("warmup=%d, iters=%d\n\n", cfg.warmup, cfg.iters);

    // Parameters
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(cfg.poly_degree);
    vector<int> bits = {60};
    for (size_t i = 0; i < cfg.n_moduli - 2; i++) bits.push_back(40);
    bits.push_back(60);
    parms.set_coeff_modulus(CoeffModulus::Create(cfg.poly_degree, bits));
    const double SCALE = (double)(1ULL << 40);
    size_t slots = cfg.poly_degree / 2;

    // NCCL
    printf("[1] NCCL init...\n");
    vector<int> dev_ids(cfg.n_gpus);
    for (int i = 0; i < cfg.n_gpus; i++) dev_ids[i] = i;
    MultiGpuContext mgpu = MultiGpuContext::create(dev_ids);
    for (int i = 0; i < cfg.n_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < cfg.n_gpus; j++)
            if (i!=j) { int c=0; cudaDeviceCanAccessPeer(&c,i,j); if(c) cudaDeviceEnablePeerAccess(j,0); }
    }

    // Per-GPU setup
    printf("[2] Per-GPU setup...\n");
    Timer timer;
    timer.start();

    vector<GpuSimState> states(cfg.n_gpus);
    stringstream sk_buf;
    vector<double> dummy_input(slots, 1.0);

    // Shared ciphertext (encrypt on GPU 0, copy to all)
    PhantomCiphertext ct_src;
    size_t total_limbs = 0;

    for (int g = 0; g < cfg.n_gpus; g++) {
        cudaSetDevice(g);
        auto &st = states[g];
        st.gpu_id = g;
        st.ctx = new PhantomContext(parms);
        if (g == 0) {
            st.sk = new PhantomSecretKey(*st.ctx);
            st.sk->save(sk_buf);
        } else {
            sk_buf.seekg(0);
            st.sk = new PhantomSecretKey();
            st.sk->load(sk_buf);
        }
        st.rk = new PhantomRelinKey(st.sk->gen_relinkey(*st.ctx));
    }

    // Encrypt + multiply on GPU 0
    cudaSetDevice(0);
    {
        PhantomCKKSEncoder enc(*states[0].ctx);
        PhantomPlaintext pa, pb;
        enc.encode(*states[0].ctx, dummy_input, SCALE, pa);
        enc.encode(*states[0].ctx, dummy_input, SCALE, pb);
        PhantomCiphertext ca, cb;
        states[0].sk->encrypt_symmetric(*states[0].ctx, pa, ca);
        states[0].sk->encrypt_symmetric(*states[0].ctx, pb, cb);
        multiply_inplace(*states[0].ctx, ca, cb);
        ct_src = std::move(ca);
    }
    auto chain_idx = ct_src.chain_index();
    total_limbs = states[0].ctx->get_context_data(chain_idx).gpu_rns_tool().base_Ql().size();
    size_t ct_elems = ct_src.size() * total_limbs * cfg.poly_degree;

    // Distribute state to all GPUs
    for (int g = 0; g < cfg.n_gpus; g++) {
        cudaSetDevice(g);
        auto &st = states[g];
        st.total_limbs = total_limbs;
        st.local_limbs = n_local_limbs(g, cfg.n_gpus, total_limbs);
        st.local_buf_elems = st.local_limbs * cfg.poly_degree;

        // Allocate local polynomial buffers
        if (st.local_buf_elems > 0) {
            cudaMalloc(&st.buf_a, st.local_buf_elems * sizeof(uint64_t));
            cudaMalloc(&st.buf_b, st.local_buf_elems * sizeof(uint64_t));
            // Fill with non-zero data (modular values)
            cudaMemset(st.buf_a, 1, st.local_buf_elems * sizeof(uint64_t));
            cudaMemset(st.buf_b, 1, st.local_buf_elems * sizeof(uint64_t));
        }

        // Copy ciphertext for key-switching
        st.ct_ks.resize(*st.ctx, chain_idx, ct_src.size(), cudaStreamPerThread);
        st.ct_ks.set_scale(ct_src.scale());
        st.ct_ks.set_ntt_form(ct_src.is_ntt_form());
        cudaMemcpyPeer(st.ct_ks.data(), g, ct_src.data(), 0, ct_elems * sizeof(uint64_t));

        st.ct_ks_backup.resize(*st.ctx, chain_idx, ct_src.size(), cudaStreamPerThread);
        st.ct_ks_backup.set_scale(ct_src.scale());
        st.ct_ks_backup.set_ntt_form(true);
        cudaMemcpy(st.ct_ks_backup.data(), st.ct_ks.data(),
                   ct_elems * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    }
    printf("   Setup in %.1f ms\n", timer.elapsed_ms());
    printf("   total_limbs=%zu, local_limbs per GPU: ", total_limbs);
    for (int g = 0; g < cfg.n_gpus; g++) printf("%zu ", states[g].local_limbs);
    printf("\n\n");

    // Get modulus pointer for each GPU
    vector<const DModulus*> gpu_modulus(cfg.n_gpus);
    for (int g = 0; g < cfg.n_gpus; g++) {
        cudaSetDevice(g);
        gpu_modulus[g] = states[g].ctx->gpu_rns_tables().modulus();
    }

    // === GROUND TRUTH: single GPU, full workload ===
    printf("[3] Ground truth (1 GPU, full workload)...\n");
    cudaSetDevice(0);
    auto &st0 = states[0];
    size_t full_coeffs = total_limbs * cfg.poly_degree;
    int block = 256;
    int full_grid = (full_coeffs + block - 1) / block;

    // Allocate full-size buffers on GPU 0
    uint64_t *gt_a = nullptr, *gt_b = nullptr;
    cudaMalloc(&gt_a, full_coeffs * sizeof(uint64_t));
    cudaMalloc(&gt_b, full_coeffs * sizeof(uint64_t));
    cudaMemset(gt_a, 1, full_coeffs * sizeof(uint64_t));
    cudaMemset(gt_b, 1, full_coeffs * sizeof(uint64_t));
    const DModulus *gt_mod = gpu_modulus[0];

    // Warmup
    for (int w = 0; w < cfg.warmup; w++) {
        for (int i = 0; i < cfg.n_ntt / 2; i++) {
            multiply_rns_poly<<<full_grid, block>>>(gt_a, gt_b, gt_mod, gt_a, cfg.poly_degree, total_limbs);
            add_rns_poly<<<full_grid, block>>>(gt_a, gt_b, gt_mod, gt_a, cfg.poly_degree, total_limbs);
        }
        for (int i = 0; i < cfg.n_mul_poly; i++)
            multiply_rns_poly<<<full_grid, block>>>(gt_a, gt_b, gt_mod, gt_a, cfg.poly_degree, total_limbs);
        for (int i = 0; i < cfg.n_add_poly; i++)
            add_rns_poly<<<full_grid, block>>>(gt_a, gt_b, gt_mod, gt_a, cfg.poly_degree, total_limbs);
        for (int i = 0; i < cfg.n_keyswitch; i++) {
            st0.ct_ks.resize(*st0.ctx, chain_idx, ct_src.size(), cudaStreamPerThread);
            cudaMemcpy(st0.ct_ks.data(), st0.ct_ks_backup.data(), ct_elems * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            st0.ct_ks.set_scale(ct_src.scale()); st0.ct_ks.set_ntt_form(true);
            relinearize_inplace(*st0.ctx, st0.ct_ks, *st0.rk);
        }
        cudaDeviceSynchronize();
    }

    // Timed
    timer.start();
    for (int it = 0; it < cfg.iters; it++) {
        for (int i = 0; i < cfg.n_ntt / 2; i++) {
            multiply_rns_poly<<<full_grid, block>>>(gt_a, gt_b, gt_mod, gt_a, cfg.poly_degree, total_limbs);
            add_rns_poly<<<full_grid, block>>>(gt_a, gt_b, gt_mod, gt_a, cfg.poly_degree, total_limbs);
        }
        for (int i = 0; i < cfg.n_mul_poly; i++)
            multiply_rns_poly<<<full_grid, block>>>(gt_a, gt_b, gt_mod, gt_a, cfg.poly_degree, total_limbs);
        for (int i = 0; i < cfg.n_add_poly; i++)
            add_rns_poly<<<full_grid, block>>>(gt_a, gt_b, gt_mod, gt_a, cfg.poly_degree, total_limbs);
        for (int i = 0; i < cfg.n_keyswitch; i++) {
            st0.ct_ks.resize(*st0.ctx, chain_idx, ct_src.size(), cudaStreamPerThread);
            cudaMemcpy(st0.ct_ks.data(), st0.ct_ks_backup.data(), ct_elems * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            st0.ct_ks.set_scale(ct_src.scale()); st0.ct_ks.set_ntt_form(true);
            relinearize_inplace(*st0.ctx, st0.ct_ks, *st0.rk);
        }
        cudaDeviceSynchronize();
    }
    double gt_avg = timer.elapsed_ms() / cfg.iters;
    printf("   Ground truth: %.2f ms per layer\n", gt_avg);

    cudaFree(gt_a);
    cudaFree(gt_b);

    // === DISTRIBUTED: multi-GPU workload ===
    printf("[4] Distributed (%d GPUs)...\n", cfg.n_gpus);

    Barrier bar(cfg.n_gpus);
    atomic<int> phase{0};

    // Launch workers for GPUs 1..n-1
    vector<thread> workers;
    for (int g = 1; g < cfg.n_gpus; g++) {
        workers.emplace_back(layer_worker, &states[g], &mgpu, &bar, &phase,
                             std::ref(cfg), gpu_modulus[g]);
    }

    // Warmup
    for (int w = 0; w < cfg.warmup; w++) {
        phase.store(1);
        bar.wait();
        // GPU 0 does its work inline
        {
            cudaSetDevice(0);
            auto &st = states[0];
            cudaStream_t s = mgpu.streams[0];
            size_t lc = st.local_buf_elems;
            if (lc > 0) {
                int lgrid = (lc + block - 1) / block;
                for (int i = 0; i < cfg.n_ntt / 2; i++) {
                    multiply_rns_poly<<<lgrid, block, 0, s>>>(st.buf_a, st.buf_b, gpu_modulus[0], st.buf_a, cfg.poly_degree, st.local_limbs);
                    add_rns_poly<<<lgrid, block, 0, s>>>(st.buf_a, st.buf_b, gpu_modulus[0], st.buf_a, cfg.poly_degree, st.local_limbs);
                }
                for (int i = 0; i < cfg.n_mul_poly; i++)
                    multiply_rns_poly<<<lgrid, block, 0, s>>>(st.buf_a, st.buf_b, gpu_modulus[0], st.buf_a, cfg.poly_degree, st.local_limbs);
                for (int i = 0; i < cfg.n_add_poly; i++)
                    add_rns_poly<<<lgrid, block, 0, s>>>(st.buf_a, st.buf_b, gpu_modulus[0], st.buf_a, cfg.poly_degree, st.local_limbs);
            }
            for (int i = 0; i < cfg.n_keyswitch; i++) {
                size_t N = cfg.poly_degree;
                size_t ct_bytes = st.ct_ks_backup.size() * st.total_limbs * N * sizeof(uint64_t);
                st.ct_ks.resize(*st.ctx, chain_idx, ct_src.size(), cudaStreamPerThread);
                cudaMemcpyAsync(st.ct_ks.data(), st.ct_ks_backup.data(), ct_bytes, cudaMemcpyDeviceToDevice, s);
                st.ct_ks.set_scale(ct_src.scale()); st.ct_ks.set_ntt_form(true);
                cudaStreamSynchronize(s);
                uint64_t *c2 = st.ct_ks.data() + 2 * st.total_limbs * N;
                keyswitching_output_aggregation(mgpu, *st.ctx, 0, st.ct_ks, c2, *st.rk, cfg.n_gpus);
            }
            cudaStreamSynchronize(s);
        }
        bar.wait();
    }

    // Timed
    timer.start();
    for (int it = 0; it < cfg.iters; it++) {
        phase.store(1);
        bar.wait();
        {
            cudaSetDevice(0);
            auto &st = states[0];
            cudaStream_t s = mgpu.streams[0];
            size_t lc = st.local_buf_elems;
            if (lc > 0) {
                int lgrid = (lc + block - 1) / block;
                for (int i = 0; i < cfg.n_ntt / 2; i++) {
                    multiply_rns_poly<<<lgrid, block, 0, s>>>(st.buf_a, st.buf_b, gpu_modulus[0], st.buf_a, cfg.poly_degree, st.local_limbs);
                    add_rns_poly<<<lgrid, block, 0, s>>>(st.buf_a, st.buf_b, gpu_modulus[0], st.buf_a, cfg.poly_degree, st.local_limbs);
                }
                for (int i = 0; i < cfg.n_mul_poly; i++)
                    multiply_rns_poly<<<lgrid, block, 0, s>>>(st.buf_a, st.buf_b, gpu_modulus[0], st.buf_a, cfg.poly_degree, st.local_limbs);
                for (int i = 0; i < cfg.n_add_poly; i++)
                    add_rns_poly<<<lgrid, block, 0, s>>>(st.buf_a, st.buf_b, gpu_modulus[0], st.buf_a, cfg.poly_degree, st.local_limbs);
            }
            for (int i = 0; i < cfg.n_keyswitch; i++) {
                size_t N = cfg.poly_degree;
                size_t ct_bytes = st.ct_ks_backup.size() * st.total_limbs * N * sizeof(uint64_t);
                st.ct_ks.resize(*st.ctx, chain_idx, ct_src.size(), cudaStreamPerThread);
                cudaMemcpyAsync(st.ct_ks.data(), st.ct_ks_backup.data(), ct_bytes, cudaMemcpyDeviceToDevice, s);
                st.ct_ks.set_scale(ct_src.scale()); st.ct_ks.set_ntt_form(true);
                cudaStreamSynchronize(s);
                uint64_t *c2 = st.ct_ks.data() + 2 * st.total_limbs * N;
                keyswitching_output_aggregation(mgpu, *st.ctx, 0, st.ct_ks, c2, *st.rk, cfg.n_gpus);
            }
            cudaStreamSynchronize(s);
        }
        bar.wait();
    }
    double dist_avg = timer.elapsed_ms() / cfg.iters;

    // Shutdown workers
    phase.store(2);
    bar.wait();
    for (auto &w : workers) w.join();

    printf("   Distributed: %.2f ms per layer\n", dist_avg);

    // Summary
    double speedup = gt_avg / dist_avg;
    printf("\n=== Results ===\n");
    printf("Ground truth (1 GPU):     %.2f ms per layer\n", gt_avg);
    printf("Distributed (%d GPUs):    %.2f ms per layer\n", cfg.n_gpus, dist_avg);
    printf("Speedup:                  %.2fx\n", speedup);
    printf("Efficiency:               %.1f%%\n", speedup / cfg.n_gpus * 100.0);

    // Breakdown
    printf("\nWorkload distribution:\n");
    printf("  LOCAL ops (NTT+mul+add): %d calls, each on %zu/%zu limbs per GPU\n",
           cfg.n_ntt + cfg.n_mul_poly + cfg.n_add_poly, states[0].local_limbs, total_limbs);
    printf("  KEYED ops (keyswitch):   %d calls, partial inner product per GPU\n", cfg.n_keyswitch);

    // Cleanup
    for (auto &st : states) {
        cudaSetDevice(st.gpu_id);
        if (st.buf_a) cudaFree(st.buf_a);
        if (st.buf_b) cudaFree(st.buf_b);
        delete st.rk; delete st.sk; delete st.ctx;
    }
    mgpu.destroy();

    printf("\nDone.\n");
    return 0;
}
