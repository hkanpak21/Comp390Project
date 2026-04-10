/**
 * ks_breakdown_bench.cu
 *
 * Micro-benchmark that measures each stage of key-switching separately:
 *   1. modup (base conversion + NTT)
 *   2. inner product (key_switch_inner_prod_c2_and_evk)
 *   3. moddown (INTT + base conversion)
 *   4. add_to_ct
 *
 * This tells us exactly what fraction of keyswitch time is distributable.
 * Only the inner product scales with number of digits (distributable).
 * modup/moddown are shared work that every GPU must do.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <sstream>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"
#include "rns.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

struct Config {
    size_t poly_degree = 65536;
    size_t n_moduli = 20;
    int warmup = 5;
    int iters = 20;
};

Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--N") && i+1 < argc) cfg.poly_degree = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--L") && i+1 < argc) cfg.n_moduli = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc) cfg.iters = atoi(argv[++i]);
    }
    return cfg;
}

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    printf("=== Key-Switching Stage Breakdown ===\n");
    printf("N=%zu, L=%zu, iters=%d\n\n", cfg.poly_degree, cfg.n_moduli, cfg.iters);

    // Setup
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(cfg.poly_degree);
    vector<int> bits = {60};
    for (size_t i = 0; i < cfg.n_moduli - 2; i++) bits.push_back(40);
    bits.push_back(60);
    parms.set_coeff_modulus(CoeffModulus::Create(cfg.poly_degree, bits));

    PhantomContext context(parms);
    PhantomSecretKey sk(context);
    PhantomRelinKey rk = sk.gen_relinkey(context);
    PhantomCKKSEncoder encoder(context);

    const double SCALE = (double)(1ULL << 40);
    size_t slots = cfg.poly_degree / 2;
    vector<double> input(slots);
    srand(42);
    for (size_t i = 0; i < slots; i++) input[i] = (rand() % 1000) / 100.0 - 5.0;

    PhantomPlaintext pa, pb;
    encoder.encode(context, input, SCALE, pa);
    encoder.encode(context, input, SCALE, pb);
    PhantomCiphertext ca, cb;
    sk.encrypt_symmetric(context, pa, ca);
    sk.encrypt_symmetric(context, pb, cb);
    multiply_inplace(context, ca, cb);

    // Extract parameters
    auto chain_idx = ca.chain_index();
    auto &key_context_data = context.get_context_data(0);
    auto &key_parms = key_context_data.parms();
    auto n = key_parms.poly_modulus_degree();
    auto &key_modulus = key_parms.coeff_modulus();
    size_t size_P = key_parms.special_modulus_size();
    size_t size_QP = key_modulus.size();

    uint32_t levelsDropped = ca.chain_index() - 1;
    auto &rns_tool = context.get_context_data(1 + levelsDropped).gpu_rns_tool();
    auto modulus_QP = context.gpu_rns_tables().modulus();

    size_t size_Ql = rns_tool.base_Ql().size();
    size_t size_Q = size_QP - size_P;
    size_t size_QlP = size_Ql + size_P;
    auto size_Ql_n = size_Ql * n;
    auto size_QlP_n = size_QlP * n;
    size_t beta = rns_tool.v_base_part_Ql_to_compl_part_QlP_conv().size();

    printf("Parameters:\n");
    printf("  size_Ql = %zu, size_P = %zu, size_QlP = %zu\n", size_Ql, size_P, size_QlP);
    printf("  beta (digits) = %zu\n", beta);
    printf("  inner product work per digit: %zu * %zu = %zu elements\n",
           size_QlP, n, size_QlP_n);
    printf("\n");

    // Backup c2
    uint64_t *c2_backup = nullptr;
    cudaMalloc(&c2_backup, size_Ql * n * sizeof(uint64_t));
    uint64_t *c2 = ca.data() + 2 * size_Ql_n;
    cudaMemcpy(c2_backup, c2, size_Ql * n * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

    // Pre-allocate buffers
    auto t_mod_up = make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, cudaStreamPerThread);
    auto cx = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, cudaStreamPerThread);
    auto reduction_threshold =
        (1 << (bits_per_uint64 - static_cast<uint64_t>(log2(key_modulus.front().value())) - 1)) - 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms;

    // ---- Benchmark modup ----
    for (int i = 0; i < cfg.warmup; i++) {
        cudaMemcpy(c2, c2_backup, size_Ql * n * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        rns_tool.modup(t_mod_up.get(), c2, context.gpu_rns_tables(), scheme_type::ckks, cudaStreamPerThread);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(start);
    for (int i = 0; i < cfg.iters; i++) {
        cudaMemcpy(c2, c2_backup, size_Ql * n * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        rns_tool.modup(t_mod_up.get(), c2, context.gpu_rns_tables(), scheme_type::ckks, cudaStreamPerThread);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float modup_avg = ms / cfg.iters;
    printf("  modup:                %.3f ms avg  (NOT distributable — every GPU does full)\n", modup_avg);

    // ---- Benchmark inner product (full — all beta digits) ----
    for (int i = 0; i < cfg.warmup; i++) {
        key_switch_inner_prod(cx.get(), t_mod_up.get(), rk.public_keys_ptr(), rns_tool, modulus_QP,
                              reduction_threshold, cudaStreamPerThread);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(start);
    for (int i = 0; i < cfg.iters; i++) {
        key_switch_inner_prod(cx.get(), t_mod_up.get(), rk.public_keys_ptr(), rns_tool, modulus_QP,
                              reduction_threshold, cudaStreamPerThread);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float inner_prod_avg = ms / cfg.iters;
    printf("  inner_product (full): %.3f ms avg  (DISTRIBUTABLE — splits across GPUs)\n", inner_prod_avg);

    // ---- Benchmark moddown ----
    for (int i = 0; i < cfg.warmup; i++) {
        // Re-run inner prod to get valid cx
        key_switch_inner_prod(cx.get(), t_mod_up.get(), rk.public_keys_ptr(), rns_tool, modulus_QP,
                              reduction_threshold, cudaStreamPerThread);
        auto cx_copy = make_cuda_auto_ptr<uint64_t>(2 * size_QlP_n, cudaStreamPerThread);
        cudaMemcpy(cx_copy.get(), cx.get(), 2 * size_QlP_n * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        for (size_t j = 0; j < 2; j++) {
            auto cx_j = cx_copy.get() + j * size_QlP_n;
            rns_tool.moddown_from_NTT(cx_j, cx_j, context.gpu_rns_tables(), scheme_type::ckks, cudaStreamPerThread);
        }
        cudaDeviceSynchronize();
    }
    // Timed
    cudaEventRecord(start);
    for (int i = 0; i < cfg.iters; i++) {
        key_switch_inner_prod(cx.get(), t_mod_up.get(), rk.public_keys_ptr(), rns_tool, modulus_QP,
                              reduction_threshold, cudaStreamPerThread);
        for (size_t j = 0; j < 2; j++) {
            auto cx_j = cx.get() + j * size_QlP_n;
            rns_tool.moddown_from_NTT(cx_j, cx_j, context.gpu_rns_tables(), scheme_type::ckks, cudaStreamPerThread);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float moddown_avg = ms / cfg.iters - inner_prod_avg;  // subtract inner prod time
    printf("  moddown:              %.3f ms avg  (NOT distributable — every GPU does full)\n", moddown_avg);

    // ---- Total ----
    float total = modup_avg + inner_prod_avg + moddown_avg;
    printf("\n  Total keyswitch:      %.3f ms\n", total);
    printf("  Inner prod fraction:  %.1f%%\n", inner_prod_avg / total * 100);
    printf("  Distributable:        %.1f%%\n", inner_prod_avg / total * 100);
    printf("  Non-distributable:    %.1f%% (modup + moddown)\n", (modup_avg + moddown_avg) / total * 100);

    printf("\n  Theoretical max speedup:\n");
    for (int g : {2, 4, 8}) {
        float distributed = inner_prod_avg / g;
        float predicted = modup_avg + distributed + moddown_avg;
        printf("    %d GPUs: %.3f ms (%.2fx speedup)\n", g, predicted, total / predicted);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(c2_backup);

    printf("\nDone.\n");
    return 0;
}
