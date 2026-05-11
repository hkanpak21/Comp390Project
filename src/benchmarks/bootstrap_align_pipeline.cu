/**
 * bootstrap_align_pipeline.cu
 *
 * Apples-to-apples per-bootstrap microbenchmark at the EXACT HP-BERT
 * pipeline workload. Single GPU, no DKS, no concurrent heads.
 *
 * Why this exists:
 *   - NEXUS's standalone `bootstrapping.cu` test runs at logN=15, sparse_slots=2^13,
 *     coeff chain {logq, 16×logp, 14×logq, log_special}  =>  total_level=30.
 *   - HP-BERT's in-pipeline bootstrap (same logN=15) runs at sparse_slots=2^13
 *     with chain {logq, 21×logp, 14×logq, log_special} => total_level=35,
 *     plus a much larger Galois-key set (HP-BERT linear ops use many extra
 *     rotation steps), driving cache pressure & key-streaming traffic.
 *   - The 252.8 ms NEXUS-standalone vs 1,017.7 ms HP-BERT in-pipeline number is
 *     therefore comparing two different workloads. To isolate the workload-vs-
 *     orchestration contribution, we measure THIS code (single isolated boot,
 *     same kernel, same parameters as HP-BERT) and compare to the modified
 *     NEXUS standalone test (Phase 2 of the alignment plan).
 *
 * What this measures:
 *   - logN, total_level, sparse_slots, scale, hamming weight, all bootstrapper
 *     hyperparameters (boundary_K, deg, scale_factor, inverse_deg, loge) match
 *     bert_hp_multigpu.cu exactly (lines 354-388). The same `bootstrap_3` call
 *     is used. Galois keys include only the bootstrap rotation steps (no QKV /
 *     softmax / FFN extra steps), since this is an isolated bootstrap test —
 *     adding HP-BERT's pipeline-specific keys only inflates the key store and
 *     would re-introduce the cache-pressure component we're trying to factor
 *     OUT.
 *
 * CLI:
 *   bootstrap_align_pipeline [--N {32768,65536}] [--iters N] [--warmup N]
 *
 * Default: N=32768 (logN=15), 100 iters + 5 warmup, single GPU 0.
 *   N=32768 mirrors the NEXUS standalone bootstrap test (their logN=15) so we
 *   can compare like-for-like; N=65536 lets us also report the pipeline's full
 *   default workload.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"
#include "galois.cuh"

#include "ckks_evaluator.cuh"
#include "bootstrapping/Bootstrapper.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

struct WallTimer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        return chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
    }
};

int main(int argc, char **argv) {
    int  ring_N = 32768;     // logN=15 default — matches NEXUS standalone test ring
    int  iters  = 100;
    int  warmup = 5;
    int  device = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--N") && i+1 < argc)
            ring_N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i+1 < argc)
            iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1 < argc)
            warmup = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--device") && i+1 < argc)
            device = atoi(argv[++i]);
    }
    if (ring_N != 32768 && ring_N != 65536) {
        fprintf(stderr, "Unsupported --N %d (use 32768 or 65536)\n", ring_N);
        return 1;
    }

    cudaSetDevice(device);

    // ───── HP-BERT pipeline workload (mirrors bert_hp_multigpu.cu:354-388) ─────
    long logN     = (ring_N == 65536) ? 16 : 15;
    long logn     = logN - 2;
    long logNh    = logN - 1;
    size_t N      = 1ULL << logN;
    long sparse_slots_val = 1L << logn;
    int  logp     = 46;
    int  logq     = 51;
    int  log_special = 51;
    int  main_mod = 21;          // HP-BERT pipeline depth (NEXUS standalone uses 16)
    int  bs_mod   = 14;
    int  total_level = main_mod + bs_mod;
    double SCALE  = pow(2.0, logp);

    long boundary_K   = 25;
    long deg          = 59;
    long scale_factor = 2;
    long inverse_deg  = 1;
    long loge         = 10;

    vector<int> coeff_bits;
    coeff_bits.push_back(logq);
    for (int i = 0; i < main_mod; i++) coeff_bits.push_back(logp);
    for (int i = 0; i < bs_mod;   i++) coeff_bits.push_back(logq);
    coeff_bits.push_back(log_special);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, coeff_bits));
    parms.set_sparse_slots(sparse_slots_val);
    parms.set_secret_key_hamming_weight(192);

    printf("════════════════════════════════════════════════════════════\n");
    printf("  bootstrap_align_pipeline — HP-BERT workload, single GPU %d\n", device);
    printf("    logN=%ld  logn=%ld  logNh=%ld  sparse_slots=%ld\n",
           logN, logn, logNh, sparse_slots_val);
    printf("    main_mod=%d  bs_mod=%d  total_level=%d\n",
           main_mod, bs_mod, total_level);
    printf("    SCALE=2^%d   logp=%d  logq=%d  log_special=%d\n",
           logp, logp, logq, log_special);
    printf("    Hamming weight=192   K=%ld deg=%ld sf=%ld ideg=%ld loge=%ld\n",
           boundary_K, deg, scale_factor, inverse_deg, loge);
    printf("    iters=%d  warmup=%d\n", iters, warmup);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    PhantomContext      context(parms);
    PhantomCKKSEncoder  encoder(context);
    PhantomSecretKey    sk(context);
    PhantomPublicKey    pk = sk.gen_publickey(context);
    PhantomRelinKey     rk = sk.gen_relinkey(context);
    PhantomGaloisKey    gk;
    CKKSEvaluator       eval(&context, &pk, &sk, &encoder, &rk, &gk, SCALE);

    Bootstrapper bs(loge, logn, logNh, total_level, SCALE,
                    boundary_K, deg, scale_factor, inverse_deg, &eval);

    bs.slot_vec.push_back(logn);
    bs.prepare_mod_polynomial();

    // Bootstrap-only Galois keys (matches what NEXUS standalone test installs).
    // We deliberately do NOT add HP-BERT's QKV/softmax/FFN rotation steps so
    // this benchmark isolates the kernel cost — adding those would re-introduce
    // the cache-pressure / key-streaming overhead we are trying to factor out.
    vector<int> gal_steps;
    gal_steps.push_back(0);
    for (int i = 0; i < logN - 1; i++) gal_steps.push_back(1 << i);
    bs.addLeftRotKeys_Linear_to_vector_3(gal_steps);

    // Deduplicate steps before installing.
    {
        std::set<int> step_set(gal_steps.begin(), gal_steps.end());
        gal_steps.assign(step_set.begin(), step_set.end());
    }
    eval.decryptor.create_galois_keys_from_steps(gal_steps, *(eval.galois_keys));
    bs.generate_LT_coefficient_3();
    printf("[setup] %zu Galois rotation keys installed (bootstrap-only)\n",
           gal_steps.size());
    fflush(stdout);

    // Build a depleted ciphertext at the same starting level HP-BERT bootstrap
    // sees (after the `while coeff_modulus_size>1` mod-switch loop in
    // run_one_head). HP-BERT's bs1/bs2/bs3/bs4 all enter at chain_index=1.
    size_t slots = encoder.slot_count();
    vector<double> input(slots, 0.0);
    for (size_t s = 0; s < (size_t)sparse_slots_val; s++) {
        input[s] = sin(0.01 * (double)s);
    }
    PhantomPlaintext  pt;
    PhantomCiphertext ct_template;
    eval.encoder.encode(input, SCALE, pt);
    eval.encryptor.encrypt(pt, ct_template);
    while (ct_template.coeff_modulus_size() > 1)
        eval.evaluator.mod_switch_to_next_inplace(ct_template);
    printf("[setup] template ct prepared at chain_index=%zu, "
           "coeff_modulus_size=%zu\n",
           ct_template.chain_index(), ct_template.coeff_modulus_size());
    fflush(stdout);

    // ───── Warmup ─────
    printf("\n[warmup] %d untimed bootstraps\n", warmup);
    fflush(stdout);
    for (int i = 0; i < warmup; i++) {
        PhantomCiphertext ct = ct_template;
        PhantomCiphertext rtn;
        bs.bootstrap_3(rtn, ct);
        cudaDeviceSynchronize();
    }

    // ───── Timed iterations ─────
    printf("\n[time] %d timed isolated bootstraps\n", iters);
    fflush(stdout);
    vector<double> times(iters, 0.0);
    for (int it = 0; it < iters; it++) {
        PhantomCiphertext ct = ct_template;     // restore depleted state
        PhantomCiphertext rtn;
        cudaDeviceSynchronize();
        WallTimer t; t.start();
        bs.bootstrap_3(rtn, ct);
        cudaDeviceSynchronize();
        times[it] = t.elapsed_ms();
        if ((it+1) % 10 == 0) {
            printf("[time] iter %d: %.2f ms\n", it+1, times[it]);
            fflush(stdout);
        }
    }

    // ───── Statistics ─────
    vector<double> sorted_times = times;
    std::sort(sorted_times.begin(), sorted_times.end());
    double median = sorted_times[iters/2];
    double min_v = sorted_times.front();
    double max_v = sorted_times.back();
    double sum = 0.0; for (double v : times) sum += v;
    double mean = sum / iters;
    double var = 0.0; for (double v : times) var += (v - mean)*(v - mean);
    double sd = sqrt(var / iters);
    double p10 = sorted_times[(int)(iters * 0.10)];
    double p90 = sorted_times[(int)(iters * 0.90)];

    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  bootstrap_align_pipeline — RESULT (single isolated bootstrap)\n");
    printf("    workload: HP-BERT pipeline @ logN=%ld, total_level=%d\n",
           logN, total_level);
    printf("    iters=%d  warmup=%d\n", iters, warmup);
    printf("    median=%.2f ms  mean=%.2f ms  σ=%.2f ms\n",
           median, mean, sd);
    printf("    min=%.2f  p10=%.2f  p90=%.2f  max=%.2f ms\n",
           min_v, p10, p90, max_v);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    return 0;
}
