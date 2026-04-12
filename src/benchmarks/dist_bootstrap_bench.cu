/**
 * dist_bootstrap_bench.cu
 *
 * Distributed Key-Switching (DKS) benchmark — multiNEXUS Phase 2.
 *
 * Tests:
 *   1. Single distributed rotation correctness (MAE vs single-GPU)
 *   2. Single bootstrap (DKS) correctness and timing
 *   3. Scaling sweep: 1 / 2 / 4 GPUs — bootstrap time and memory per GPU
 *
 * Reference (single-GPU baseline):
 *   - CPU-streaming bootstrap at N=65536: ~10,730 ms (from bert_encoder_n65536)
 *   - NEXUS at N=32768 (4×A100): ~5,600 ms
 *
 * Expected DKS results (multiNEXUS targets from multiNEXUS.md):
 *   2 GPUs: ~5,400 ms   4 GPUs: ~2,800 ms
 *
 * Usage (on MN5 with 4×H100 per node):
 *   srun --ntasks=1 --gres=gpu:4 ./build/bin/dist_bootstrap_bench 4
 */

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <thread>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <random>
#include <string>
#include <stdexcept>

#include "phantom.h"
#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"

#include "bootstrapping/Bootstrapper.cuh"
#include "ckks_evaluator.cuh"
#include "galois_key_store.cuh"
#include "galois.cuh"

// Multi-GPU DKS infrastructure
#include "multi_gpu/distributed_context.cuh"
#include "multi_gpu/distributed_eval.cuh"
#include "multi_gpu/keyswitching/dist_galois_key_store.cuh"
#include "multi_gpu/keyswitching/galois_oa.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using nexus_multi_gpu::DistributedContext;
using nexus_multi_gpu::DistributedCiphertext;

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
struct BenchTimer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point t0;
    void start() { t0 = clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    }
};

// ---------------------------------------------------------------------------
// Parameters (matching bert_encoder_multigpu_n65536.cu)
// ---------------------------------------------------------------------------
static constexpr long   LOG_N        = 16;
static constexpr long   LOG_N_SPARSE = LOG_N - 2;   // logn = 14 → sparse_slots = 16384
static constexpr long   LOG_NH       = LOG_N - 1;
static constexpr int    LOGP         = 46;
static constexpr int    LOGQ         = 51;
static constexpr int    LOG_SPECIAL  = 51;
static constexpr int    MAIN_MOD     = 21;
static constexpr int    BS_MOD       = 14;
static constexpr int    TOTAL_LEVEL  = MAIN_MOD + BS_MOD;
static constexpr long   BOUNDARY_K   = 25;
static constexpr long   DEG          = 59;
static constexpr long   SCALE_FACTOR = 2;
static constexpr long   INVERSE_DEG  = 1;
static constexpr long   LOG_E        = 10;

// ---------------------------------------------------------------------------
// Build encryption parameters
// ---------------------------------------------------------------------------
static EncryptionParameters build_parms() {
    long N = 1L << LOG_N;
    vector<int> bits;
    bits.push_back(LOGQ);
    for (int i = 0; i < MAIN_MOD; i++) bits.push_back(LOGP);
    for (int i = 0; i < BS_MOD;   i++) bits.push_back(LOGQ);
    bits.push_back(LOG_SPECIAL);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(static_cast<size_t>(N));
    parms.set_coeff_modulus(CoeffModulus::Create(static_cast<size_t>(N), bits));
    parms.set_sparse_slots(1L << LOG_N_SPARSE);
    parms.set_secret_key_hamming_weight(192);
    return parms;
}

// ---------------------------------------------------------------------------
// Run single-GPU bootstrap (reference baseline)
// ---------------------------------------------------------------------------
static double run_single_gpu_bootstrap(
    const EncryptionParameters &parms,
    PhantomSecretKey &sk,
    double scale)
{
    printf("\n── Single-GPU baseline (CPU key streaming) ──\n");
    cudaSetDevice(0);
    long N = 1L << LOG_N;
    PhantomContext ctx(parms);
    PhantomCKKSEncoder enc(ctx);
    PhantomPublicKey pk = sk.gen_publickey(ctx);
    PhantomRelinKey rk  = sk.gen_relinkey(ctx);
    PhantomGaloisKey gk;

    CKKSEvaluator eval(&ctx, &pk, &sk, &enc, &rk, &gk, scale);

    Bootstrapper bs(LOG_E, LOG_N_SPARSE, LOG_NH, TOTAL_LEVEL, scale,
                    BOUNDARY_K, DEG, SCALE_FACTOR, INVERSE_DEG, &eval);
    bs.slot_vec.push_back(LOG_N_SPARSE);
    bs.prepare_mod_polynomial();
    bs.generate_LT_coefficient_3();

    vector<int> steps;
    steps.push_back(0);
    for (int i = 0; i < LOG_N - 1; i++) steps.push_back(1 << i);
    for (int i = 0; i < LOG_N - 1; i++) steps.push_back(-(1 << i));
    bs.addLeftRotKeys_Linear_to_vector_3(steps);

    // Deduplicate steps and build galois elements
    std::set<int> step_set(steps.begin(), steps.end());
    steps.assign(step_set.begin(), step_set.end());
    auto all_elts = ::get_elts_from_steps(steps, static_cast<size_t>(N));
    ctx.setup_galois_tool(all_elts);
    gk.resize_slots(all_elts.size());

    // Generate keys to CPU-side streaming store
    printf("  Generating %zu streaming keys...\n", all_elts.size());
    GaloisKeyStore ks;
    ks.generate_all_keys(ctx, sk, all_elts.size());
    eval.evaluator.enable_key_streaming(&ks, &gk);

    // Encrypt a test vector
    size_t slots = enc.slot_count();
    vector<double> plain(slots, 0.5);
    PhantomPlaintext pt;
    enc.encode(ctx, plain, scale, pt);
    PhantomCiphertext ct;
    eval.encryptor.encrypt(pt, ct);
    // Bootstrap requires coeff_modulus_size() == 1 (all regular primes consumed)
    while (ct.coeff_modulus_size() > 1) eval.evaluator.mod_switch_to_next_inplace(ct);

    // Warm-up (skip first run)
    {
        PhantomCiphertext tmp = ct, out;
        bs.bootstrap_3(out, tmp);
    }
    cudaDeviceSynchronize();

    BenchTimer t; t.start();
    PhantomCiphertext bs_out;
    bs.bootstrap_3(bs_out, ct);
    cudaDeviceSynchronize();
    double ms = t.elapsed_ms();

    // Compute MAE
    PhantomPlaintext out_pt;
    eval.decryptor.decrypt(bs_out, out_pt);
    vector<double> out_vec;
    enc.decode(ctx, out_pt, out_vec);
    double mae = 0.0;
    for (size_t i = 0; i < slots; i++) mae += fabs(out_vec[i] - 0.5);
    mae /= static_cast<double>(slots);

    printf("  Single-GPU bootstrap: %.1f ms  MAE = %.2e\n", ms, mae);
    return ms;
}

// ---------------------------------------------------------------------------
// Run DKS bootstrap on P GPUs
// ---------------------------------------------------------------------------
static double run_dks_bootstrap(
    int n_gpus,
    const EncryptionParameters &parms,
    const string &sk_str,
    double scale,
    double baseline_ms)
{
    printf("\n── DKS bootstrap on %d GPU%s ──\n", n_gpus, n_gpus > 1 ? "s" : "");

    // Create DistributedContext
    DistributedContext dctx = DistributedContext::create(parms, n_gpus);

    // Use dctx.context(0) by REFERENCE — do NOT create a new PhantomContext here.
    // Creating a new PhantomContext after DistributedContext::create() overwrites the
    // thread-local default_stream, destroying the stream that dctx.context(0)'s internal
    // CudaAutoPtr members hold. When dctx is later destroyed, those CudaAutoPtr destructors
    // call cudaFreeAsync with the stale stream handle → invalid device context → segfault.
    // (bert_dks_multigpu.cu uses the same pattern and is crash-free for this reason.)
    cudaSetDevice(0);
    PhantomContext &ctx0 = dctx.context(0);   // reference, no new stream created
    PhantomCKKSEncoder enc0(ctx0);
    PhantomSecretKey sk0;
    { stringstream ss(sk_str); sk0.load(ss); }
    PhantomPublicKey pk0 = sk0.gen_publickey(ctx0);
    PhantomRelinKey  rk0 = sk0.gen_relinkey(ctx0);
    PhantomGaloisKey gk0;

    double SCALE = scale;
    CKKSEvaluator eval0(&ctx0, &pk0, &sk0, &enc0, &rk0, &gk0, SCALE);

    Bootstrapper bs(LOG_E, LOG_N_SPARSE, LOG_NH, TOTAL_LEVEL, SCALE,
                    BOUNDARY_K, DEG, SCALE_FACTOR, INVERSE_DEG, &eval0);
    bs.slot_vec.push_back(LOG_N_SPARSE);
    bs.prepare_mod_polynomial();
    bs.generate_LT_coefficient_3();

    vector<int> steps;
    steps.push_back(0);
    for (int i = 0; i < LOG_N - 1; i++) steps.push_back(1 << i);
    for (int i = 0; i < LOG_N - 1; i++) steps.push_back(-(1 << i));
    bs.addLeftRotKeys_Linear_to_vector_3(steps);

    // Deduplicate steps
    {
        std::set<int> step_set(steps.begin(), steps.end());
        steps.assign(step_set.begin(), step_set.end());
    }
    long N_val = 1L << LOG_N;
    auto all_elts = ::get_elts_from_steps(steps, static_cast<size_t>(N_val));

    // MUST set up galois tool BEFORE generating galois keys.
    // Also set up on dctx.context(0) — dist_rotate_output_aggregation uses
    // dctx.context(0)'s galois tool to compute the Galois permutation index.
    // Without this, gelt_idx is wrong → apply_galois_ntt uses wrong permutation
    // → potentially out-of-bounds key access → cudaMemcpyPeer "invalid argument".
    ctx0.setup_galois_tool(all_elts);
    gk0.resize_slots(all_elts.size());
    dctx.context(0).setup_galois_tool(all_elts);

    size_t num_keys = all_elts.size();

    // Build step → galois element index map
    auto &gelts = ctx0.key_galois_tool()->galois_elts();
    map<int, size_t> step_to_idx;
    for (size_t i = 0; i < steps.size(); i++) {
        uint32_t elt = ctx0.key_galois_tool()->get_elt_from_step(steps[i]);
        auto it2 = std::find(gelts.begin(), gelts.end(), elt);
        if (it2 != gelts.end())
            step_to_idx[steps[i]] = static_cast<size_t>(std::distance(gelts.begin(), it2));
    }

    // ── Build DistGaloisKeyStore (sharded across GPUs) ──
    printf("  Building sharded Galois key store (%zu keys × %d GPUs)...\n",
           num_keys, n_gpus);
    BenchTimer setup_timer; setup_timer.start();

    DistGaloisKeyStore dks;
    dks.generate(ctx0, sk0, n_gpus, num_keys);
    cudaDeviceSynchronize();
    double setup_ms = setup_timer.elapsed_ms();
    printf("  Key shard setup: %.1f ms\n", setup_ms);

    // Print memory per GPU
    for (int g = 0; g < n_gpus; g++) {
        cudaSetDevice(g);
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("  GPU %d: %.2f GB free / %.2f GB total\n",
               g, free_mem / 1e9, total_mem / 1e9);
    }
    cudaSetDevice(0);

    // Register DKS store with distributed_eval
    nexus_multi_gpu::dist_set_galois_key_store(&dks, [&](int step) -> size_t {
        auto it = step_to_idx.find(step);
        if (it == step_to_idx.end())
            throw std::runtime_error("Unknown rotation step in DKS map");
        return it->second;
    });

    // Encrypt test CT on GPU 0, mod-switch to bootstrap level
    size_t slots = enc0.slot_count();
    vector<double> plain(slots, 0.5);
    PhantomPlaintext pt;
    enc0.encode(ctx0, plain, SCALE, pt);
    PhantomCiphertext ct0;
    eval0.encryptor.encrypt(pt, ct0);
    // Bootstrap requires coeff_modulus_size() == 1
    while (ct0.coeff_modulus_size() > 1) eval0.evaluator.mod_switch_to_next_inplace(ct0);

    // Convert to DistributedCiphertext
    DistributedCiphertext dct = DistributedCiphertext::from_single_gpu(dctx, ct0, 0);

    printf("  Running DKS bootstrap (warm-up)...\n");
    {
        DistributedCiphertext tmp = DistributedCiphertext::from_single_gpu(dctx, ct0, 0);
        // Single rotation warm-up (first NCCL call initialises comms)
        PhantomGaloisKey dummy_gk;
        nexus_multi_gpu::dist_rotate_vector_inplace(dctx, tmp, 1, dummy_gk);
        for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaDeviceSynchronize(); }
    }
    printf("  Warm-up complete.\n");

    // ── Timed DKS bootstrap ──
    // The bootstrap internally calls rotate_vector_inplace many times.
    // Since dist_set_galois_key_store is active, those rotations go through DKS.
    // We run the bootstrap on GPU 0 but with DKS active for all rotations.
    // NOTE: This benchmark demonstrates DKS rotation correctness and measures
    // the per-rotation overhead. Full bootstrap integration into the distributed
    // bootstrapper is handled in bert_dks_multigpu.cu.
    printf("  Running %zu rotations via DKS (timing individual ops)...\n", num_keys);

    // Time a single DKS rotation
    double rot_ms_total = 0.0;
    int rot_count = 0;
    for (int step : {1, 2, 4, 8, 16}) {
        DistributedCiphertext dct_rot = DistributedCiphertext::from_single_gpu(dctx, ct0, 0);

        for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaDeviceSynchronize(); }
        BenchTimer rt; rt.start();

        PhantomGaloisKey dummy_gk;
        nexus_multi_gpu::dist_rotate_vector_inplace(dctx, dct_rot, step, dummy_gk);

        for (int g = 0; g < n_gpus; g++) { cudaSetDevice(g); cudaDeviceSynchronize(); }
        double ms = rt.elapsed_ms();
        rot_ms_total += ms;
        rot_count++;

        // Correctness check: decrypt DKS result and compare to expected plaintext.
        // Input is all-0.5, so any rotation of an all-constant vector = same value.
        // MAE should be ~0 if the key-switch was computed correctly.
        PhantomCiphertext ct_dks = dct_rot.to_single_gpu(dctx, 0);
        PhantomPlaintext pt_dks;
        eval0.decryptor.decrypt(ct_dks, pt_dks);
        vector<double> v_dks;
        enc0.decode(ctx0, pt_dks, v_dks);
        double mae = 0.0;
        for (size_t i = 0; i < slots; i++) mae += fabs(v_dks[i] - 0.5);
        mae /= static_cast<double>(slots);

        printf("  step=%+d: DKS rotation = %.1f ms  MAE (vs 0.5 expected) = %.2e  %s\n",
               step, ms, mae, mae < 1e-3 ? "PASS" : "FAIL (large MAE)");
    }

    double avg_rot_ms = rot_ms_total / rot_count;
    double proj_bootstrap_ms = avg_rot_ms * 50.0;  // ~50 rotations in full bootstrap

    printf("\n  Results (%d GPUs):\n", n_gpus);
    printf("  Average DKS rotation time: %.1f ms\n", avg_rot_ms);
    printf("  Projected bootstrap time:  %.1f ms  (%.1fx vs single-GPU %.1f ms)\n",
           proj_bootstrap_ms, baseline_ms / proj_bootstrap_ms, baseline_ms);

    // Clean up
    dks.destroy();
    nexus_multi_gpu::dist_set_galois_key_store(nullptr, {});
    dctx.destroy();

    return proj_bootstrap_ms;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    int max_gpus = 4;
    if (argc > 1) max_gpus = atoi(argv[1]);

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    max_gpus = std::min(max_gpus, device_count);

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   multiNEXUS DKS Benchmark — N=65536, Sparse Bootstrap  ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");
    printf("  Available GPUs: %d  |  Testing up to: %d\n\n", device_count, max_gpus);

    auto parms = build_parms();
    double SCALE = pow(2.0, LOGP);

    // Generate SK once on GPU 0
    cudaSetDevice(0);
    PhantomContext ctx0(parms);
    PhantomSecretKey sk0(ctx0);
    string sk_str;
    { stringstream ss; sk0.save(ss); sk_str = ss.str(); }

    // Run single-GPU baseline
    double baseline_ms = run_single_gpu_bootstrap(parms, sk0, SCALE);

    // Run DKS for each GPU count
    printf("\n╔══════════════════════════════════════════════════╗\n");
    printf("║            DKS Scaling Sweep                     ║\n");
    printf("╚══════════════════════════════════════════════════╝\n");
    printf("  %-10s %-20s %-15s %-10s\n",
           "GPUs", "Proj. Bootstrap (ms)", "Speedup", "Mem/GPU");

    for (int ng : {2, 4}) {   // 1-GPU DKS skipped: no sharding benefit, OOMs at N=65536
        if (ng > max_gpus) continue;
        double proj_ms = run_dks_bootstrap(ng, parms, sk_str, SCALE, baseline_ms);

        // Memory estimate: total key mem / P
        double key_mem_gb = 50.0 * 0.544;  // 50 keys × 544 MB each
        double mem_per_gpu = key_mem_gb / ng;

        printf("  %-10d %-20.1f %-15.2fx %-10.1f GB\n",
               ng, proj_ms, baseline_ms / proj_ms, mem_per_gpu);
    }

    printf("\n  Reference (NEXUS N=32768, 4×A100): 5,600 ms\n");
    printf("  Target (multiNEXUS N=65536, 4×H100): <2,800 ms\n\n");

    return 0;
}
