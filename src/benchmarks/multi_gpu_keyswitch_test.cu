/**
 * multi_gpu_keyswitch_test.cu
 *
 * End-to-end validation of multi-GPU key-switching algorithms.
 *
 * Architecture: GPU 0 does all FHE work (encrypt, multiply, keyswitch, decrypt).
 * For multi-GPU NCCL collectives, ALL GPUs must participate. We use std::thread
 * to launch one thread per GPU. Non-zero GPUs just participate in the NCCL
 * collective with dummy buffers; GPU 0 does the real work.
 *
 * Usage:
 *   ./multi_gpu_keyswitch_test --n-gpus 2 [--verbose]
 */

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

// Simple barrier for C++17 (std::barrier requires C++20)
class SimpleBarrier {
    int n_, count_;
    int generation_ = 0;
    mutex mtx_;
    condition_variable cv_;
public:
    explicit SimpleBarrier(int n) : n_(n), count_(n) {}
    void arrive_and_wait() {
        unique_lock<mutex> lk(mtx_);
        int gen = generation_;
        if (--count_ == 0) {
            generation_++;
            count_ = n_;
            cv_.notify_all();
        } else {
            cv_.wait(lk, [&] { return gen != generation_; });
        }
    }
};

// Phantom FHE
#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

// Our multi-GPU code
#include "../multi_gpu/comm/nccl_comm.cuh"
#include "../multi_gpu/partition/rns_partition.cuh"
#include "../multi_gpu/keyswitching/input_broadcast.cuh"
#include "../multi_gpu/keyswitching/output_aggregation.cuh"

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

struct Config {
    int n_gpus = 2;
    bool verbose = false;
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i + 1 < argc)
            cfg.n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--verbose"))
            cfg.verbose = true;
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Multi-GPU helper thread function
// Non-zero GPUs participate in NCCL collectives with allocated buffers.
// ---------------------------------------------------------------------------
struct GpuWorkerArgs {
    MultiGpuContext *mgpu;
    int gpu_id;
    size_t buf_elems;     // total_limbs * degree for allgather/allreduce
    size_t local_elems;   // local_limbs * degree
    // Synchronization
    SimpleBarrier *sync_bar;
    atomic<int> *phase;   // 0=wait, 1=IB_allgather, 2=OA_allreduce, 3=done
};

void gpu_worker_thread(GpuWorkerArgs args) {
    cudaSetDevice(args.gpu_id);

    // Allocate local and full buffers for NCCL participation
    uint64_t *local_buf = nullptr, *full_buf = nullptr;
    cudaMalloc(&local_buf, args.local_elems * sizeof(uint64_t));
    cudaMalloc(&full_buf, args.buf_elems * sizeof(uint64_t));
    cudaMemset(local_buf, 0, args.local_elems * sizeof(uint64_t));
    cudaMemset(full_buf, 0, args.buf_elems * sizeof(uint64_t));

    // Also for OA: need partial_cx buffer (2 * size_QlP_n)
    // We over-allocate: use buf_elems * 2 which is enough
    uint64_t *partial_cx = nullptr;
    cudaMalloc(&partial_cx, 2 * args.buf_elems * sizeof(uint64_t));
    cudaMemset(partial_cx, 0, 2 * args.buf_elems * sizeof(uint64_t));

    while (true) {
        args.sync_bar->arrive_and_wait();  // wait for main thread signal

        int p = args.phase->load();
        if (p == 3) break;  // done

        if (p == 1) {
            // Input Broadcast: participate in AllGather
            // GPU 0 does the real AllGather; we just call our side
            ncclGroupStart();
            ncclAllGather(local_buf, full_buf,
                         args.local_elems, ncclUint64,
                         args.mgpu->comms[args.gpu_id],
                         args.mgpu->streams[args.gpu_id]);
            ncclGroupEnd();
            cudaStreamSynchronize(args.mgpu->streams[args.gpu_id]);
        }
        else if (p == 2) {
            // Output Aggregation: participate in AllReduce
            ncclGroupStart();
            ncclAllReduce(partial_cx, partial_cx,
                         2 * args.buf_elems, ncclUint64, ncclSum,
                         args.mgpu->comms[args.gpu_id],
                         args.mgpu->streams[args.gpu_id]);
            ncclGroupEnd();
            cudaStreamSynchronize(args.mgpu->streams[args.gpu_id]);
        }

        args.sync_bar->arrive_and_wait();  // signal completion
    }

    cudaFree(local_buf);
    cudaFree(full_buf);
    cudaFree(partial_cx);
}

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count < cfg.n_gpus) {
        fprintf(stderr, "ERROR: requested %d GPUs but only %d available\n",
                cfg.n_gpus, device_count);
        return 1;
    }

    printf("=== Multi-GPU Key-Switching Validation Test ===\n");
    printf("GPUs: %d\n\n", cfg.n_gpus);

    // ---- NCCL init ----
    printf("[0/6] Initializing NCCL...\n");
    vector<int> dev_ids(cfg.n_gpus);
    for (int i = 0; i < cfg.n_gpus; i++) dev_ids[i] = i;
    MultiGpuContext mgpu_ctx = MultiGpuContext::create(dev_ids);
    cudaSetDevice(0);
    printf("   NCCL initialized for %d GPUs\n\n", cfg.n_gpus);

    // ---- Phantom context on GPU 0 ----
    printf("[1/6] Initializing Phantom CKKS context...\n");
    Timer timer;
    timer.start();

    constexpr size_t POLY_DEGREE = 8192;
    const double SCALE = (double)(1ULL << 40);
    constexpr size_t N_MODULI = 5;

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(POLY_DEGREE);
    vector<int> bit_sizes = {60};
    for (size_t i = 0; i < N_MODULI - 2; i++) bit_sizes.push_back(40);
    bit_sizes.push_back(60);
    parms.set_coeff_modulus(phantom::arith::CoeffModulus::Create(POLY_DEGREE, bit_sizes));
    PhantomContext context(parms);
    printf("   Context created in %.1f ms\n", timer.elapsed_ms());

    printf("[2/6] Generating keys...\n");
    timer.start();
    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    printf("   Keys generated in %.1f ms\n", timer.elapsed_ms());

    printf("[3/6] Encrypting and multiplying...\n");
    timer.start();
    PhantomCKKSEncoder encoder(context);
    size_t slot_count = POLY_DEGREE / 2;
    vector<double> input_a(slot_count), input_b(slot_count);
    srand(42);
    for (size_t i = 0; i < slot_count; i++) {
        input_a[i] = (rand() % 1000) / 100.0 - 5.0;
        input_b[i] = (rand() % 1000) / 100.0 - 5.0;
    }
    PhantomPlaintext plain_a, plain_b;
    encoder.encode(context, input_a, SCALE, plain_a);
    encoder.encode(context, input_b, SCALE, plain_b);
    printf("   Encrypt + multiply in %.1f ms\n", timer.elapsed_ms());

    // ---- Ground truth ----
    printf("[4/6] Computing ground truth...\n");
    timer.start();
    PhantomCiphertext ct_gt;
    {
        PhantomCiphertext a, b;
        secret_key.encrypt_symmetric(context, plain_a, a);
        secret_key.encrypt_symmetric(context, plain_b, b);
        multiply_inplace(context, a, b);
        ct_gt = std::move(a);
    }
    relinearize_inplace(context, ct_gt, relin_keys);
    printf("   Single-GPU relinearize in %.1f ms\n", timer.elapsed_ms());

    PhantomPlaintext plain_gt;
    secret_key.decrypt(context, ct_gt, plain_gt);
    vector<double> result_gt;
    encoder.decode(context, plain_gt, result_gt);

    // ---- Get sizes for worker threads ----
    PhantomCiphertext ct_tmp;
    {
        PhantomCiphertext a, b;
        secret_key.encrypt_symmetric(context, plain_a, a);
        secret_key.encrypt_symmetric(context, plain_b, b);
        multiply_inplace(context, a, b);
        ct_tmp = std::move(a);
    }
    auto chain_idx = ct_tmp.chain_index();
    auto &ctx_data = context.get_context_data(chain_idx);
    size_t size_Ql = ctx_data.gpu_rns_tool().base_Ql().size();
    size_t total_elems = size_Ql * POLY_DEGREE;

    // ---- Launch worker threads for non-zero GPUs ----
    SimpleBarrier sync_bar(cfg.n_gpus);  // all GPUs including 0
    atomic<int> phase{0};

    vector<thread> workers;
    vector<GpuWorkerArgs> worker_args(cfg.n_gpus);
    for (int g = 1; g < cfg.n_gpus; g++) {
        size_t local_n = n_local_limbs(g, cfg.n_gpus, size_Ql);
        worker_args[g] = {&mgpu_ctx, g, total_elems, local_n * POLY_DEGREE,
                          &sync_bar, &phase};
        workers.emplace_back(gpu_worker_thread, worker_args[g]);
    }

    // ---- Step 5: Input Broadcast (GPU 0 thread) ----
    printf("[5/6] Testing Input Broadcast key-switching (%d GPUs)...\n", cfg.n_gpus);
    timer.start();

    PhantomCiphertext ct_ib;
    {
        PhantomCiphertext a, b;
        secret_key.encrypt_symmetric(context, plain_a, a);
        secret_key.encrypt_symmetric(context, plain_b, b);
        multiply_inplace(context, a, b);
        ct_ib = std::move(a);
    }
    uint64_t *c2_ib = ct_ib.data() + 2 * size_Ql * POLY_DEGREE;

    if (cfg.n_gpus > 1) {
        phase.store(1);           // tell workers: IB allgather
        sync_bar.arrive_and_wait(); // release workers
    }

    keyswitching_input_broadcast(mgpu_ctx, context, 0,
                                 ct_ib, c2_ib, relin_keys,
                                 size_Ql, POLY_DEGREE, cfg.n_gpus);

    if (cfg.n_gpus > 1) {
        sync_bar.arrive_and_wait(); // wait for workers
    }

    ct_ib.resize(context, chain_idx, 2, cudaStreamPerThread);
    double ib_time = timer.elapsed_ms();

    PhantomPlaintext plain_ib;
    secret_key.decrypt(context, ct_ib, plain_ib);
    vector<double> result_ib;
    encoder.decode(context, plain_ib, result_ib);

    double ib_mae = 0.0;
    for (size_t i = 0; i < slot_count; i++)
        ib_mae += fabs(result_ib[i] - result_gt[i]);
    ib_mae /= slot_count;

    printf("   Input Broadcast: %.1f ms, MAE=%.2e  %s\n",
           ib_time, ib_mae, ib_mae < 1e-3 ? "PASS" : "FAIL");

    // ---- Step 6: Output Aggregation (GPU 0 thread) ----
    printf("[6/6] Testing Output Aggregation key-switching (%d GPUs)...\n", cfg.n_gpus);
    timer.start();

    PhantomCiphertext ct_oa;
    {
        PhantomCiphertext a, b;
        secret_key.encrypt_symmetric(context, plain_a, a);
        secret_key.encrypt_symmetric(context, plain_b, b);
        multiply_inplace(context, a, b);
        ct_oa = std::move(a);
    }
    uint64_t *c2_oa = ct_oa.data() + 2 * size_Ql * POLY_DEGREE;

    if (cfg.n_gpus > 1) {
        phase.store(2);           // tell workers: OA allreduce
        sync_bar.arrive_and_wait();
    }

    keyswitching_output_aggregation(mgpu_ctx, context, 0,
                                    ct_oa, c2_oa, relin_keys,
                                    cfg.n_gpus);

    if (cfg.n_gpus > 1) {
        sync_bar.arrive_and_wait();
    }

    ct_oa.resize(context, chain_idx, 2, cudaStreamPerThread);
    double oa_time = timer.elapsed_ms();

    PhantomPlaintext plain_oa;
    secret_key.decrypt(context, ct_oa, plain_oa);
    vector<double> result_oa;
    encoder.decode(context, plain_oa, result_oa);

    double oa_mae = 0.0;
    for (size_t i = 0; i < slot_count; i++)
        oa_mae += fabs(result_oa[i] - result_gt[i]);
    oa_mae /= slot_count;

    printf("   Output Aggregation: %.1f ms, MAE=%.2e  %s\n",
           oa_time, oa_mae, oa_mae < 1e-3 ? "PASS" : "FAIL");

    // ---- Cleanup workers ----
    if (cfg.n_gpus > 1) {
        phase.store(3);
        sync_bar.arrive_and_wait();
        for (auto &w : workers) w.join();
    }

    // ---- Summary ----
    printf("\n=== Summary ===\n");
    printf("Single-GPU relinearize:       ground truth\n");
    printf("Input Broadcast (%d GPUs):    MAE=%.2e  %s\n",
           cfg.n_gpus, ib_mae, ib_mae < 1e-3 ? "PASS" : "FAIL");
    printf("Output Aggregation (%d GPUs): MAE=%.2e  %s\n",
           cfg.n_gpus, oa_mae, oa_mae < 1e-3 ? "PASS" : "FAIL");

    if (cfg.verbose) {
        printf("\nFirst 5 slots:\n");
        for (int i = 0; i < 5; i++)
            printf("  [%d] GT=%.6f  IB=%.6f  OA=%.6f\n",
                   i, result_gt[i], result_ib[i], result_oa[i]);
    }

    mgpu_ctx.destroy();
    bool passed = (ib_mae < 1e-3) && (oa_mae < 1e-3);
    printf("\n%s\n", passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    return passed ? 0 : 1;
}
