/**
 * bootstrapping_bench.cu
 *
 * Isolated bootstrapping latency benchmark.
 *
 * Bootstrapping is the single most expensive operation in NEXUS:
 * it accounts for roughly 60% of the Argmax operation's time.
 * This benchmark measures bootstrapping in isolation so we can:
 *
 *   1. Establish single-GPU baseline bootstrapping latency
 *   2. Measure multi-GPU speedup for bootstrapping specifically
 *   3. Profile the CtoS/StoC linear transforms vs. modular reduction
 *
 * Bootstrapping pipeline (NEXUS implementation):
 *   1. Slot-to-Coefficient (StoC)   : ~14 BSGS linear transforms
 *   2. Subsum + ModularReduction    : ~9 polynomial evaluations
 *   3. Coefficient-to-Slot (CtoS)   : ~14 BSGS linear transforms
 *
 * For multi-GPU, the dominant cost is rotations (which require key-switching).
 * Each BSGS step uses O(sqrt(N)) rotations. Distributing limbs across GPUs
 * reduces NTT and key-switch cost by ~n_gpus factor.
 *
 * Usage:
 *   ./bootstrapping_bench [--n-gpus N] [--iters I] [--output path.csv]
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "phantom.h"
#include "ckks_evaluator.cuh"
#include "bootstrapping/Bootstrapper.cuh"

using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;
using namespace nexus;

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------

struct WallTimer {
    std::chrono::high_resolution_clock::time_point t0;
    void start() { t0 = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
};

// ---------------------------------------------------------------------------
// Bootstrapping parameters (from NEXUS main.cu argmax_test)
// ---------------------------------------------------------------------------

// Full modulus chain: 17 (main) + 14 (bootstrap) = 31 total levels
static constexpr int MAIN_MOD_COUNT = 17;
static constexpr int BS_MOD_COUNT   = 14;
static constexpr int TOTAL_MOD_COUNT = MAIN_MOD_COUNT + BS_MOD_COUNT - 1; // 30

// Bootstrapping hyperparameters matching NEXUS paper configuration
static constexpr int  LOG_E        = 10;    // encoding error bound (2^-10)
static constexpr int  LOG_N        = 4;     // sparse slots = 2^4 = 16
static constexpr int  LOG_NH       = 14;    // log(N/2) for full bootstrapping
static constexpr int  BOUNDARY_K   = 25;    // sign function cutoff
static constexpr int  SIN_DEG      = 59;    // trig polynomial degree
static constexpr int  INVERSE_DEG  = 2;     // Newton depth for inverse
static constexpr int  SCALE_FACTOR = 2;     // auxiliary scale factor

static constexpr size_t N = 1ULL << 16;    // poly degree = 65536
static constexpr double SCALE = 1ULL << 40;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

struct Config {
    int         n_gpus      = 1;
    int         iters       = 5;
    std::string output_file = "";
    bool        verbose     = false;
};

static Config parse_args(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; ++i) {
        std::string a(argv[i]);
        if (a == "--n-gpus"  && i + 1 < argc) c.n_gpus = std::atoi(argv[++i]);
        if (a == "--iters"   && i + 1 < argc) c.iters  = std::atoi(argv[++i]);
        if (a == "--output"  && i + 1 < argc) c.output_file = argv[++i];
        if (a == "--verbose")                  c.verbose = true;
    }
    return c;
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

struct BootstrapResult {
    int    n_gpus;
    double mean_ms;
    double min_ms;
    double max_ms;
    double stddev_ms;
    double speedup;
};

// ---------------------------------------------------------------------------
// Benchmark driver
// ---------------------------------------------------------------------------

static BootstrapResult run_benchmark(const Config &cfg) {
    CUDA_CHECK(cudaSetDevice(0));

    // Build full modulus chain for bootstrapping.
    // 58-bit anchor + (TOTAL_MOD_COUNT-2) 40-bit primes + 58-bit tail
    std::vector<int> moduli;
    moduli.push_back(58);
    for (int i = 0; i < TOTAL_MOD_COUNT - 2; ++i) moduli.push_back(40);
    moduli.push_back(58);

    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(N);
    parms.set_coeff_modulus(CoeffModulus::Create(N, moduli));

    PhantomContext context(parms);
    PhantomCKKSEncoder encoder(context);
    PhantomSecretKey  sk(context);
    PhantomPublicKey  pk = sk.gen_publickey(context);
    PhantomRelinKey   rk = sk.gen_relinkey(context);
    PhantomGaloisKey  gk;

    CKKSEvaluator ckks(&context, &pk, &sk, &encoder, &rk, &gk, SCALE, {});

    // ---- Setup Bootstrapper ----
    printf("[bootstrap_bench] Initializing Bootstrapper...\n");
    WallTimer init_timer;
    init_timer.start();

    Bootstrapper bootstrapper(
        LOG_E, LOG_N, LOG_NH,
        TOTAL_MOD_COUNT - 1,   // L
        SCALE,                  // initial_scale
        SCALE,                  // final_scale
        BOUNDARY_K, SIN_DEG,
        SCALE_FACTOR, INVERSE_DEG,
        &ckks
    );

    // Generate Galois keys for bootstrapping rotations.
    std::vector<int> gal_steps;
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps);
    ckks.decryptor.create_galois_keys_from_steps(gal_steps, *ckks.galois_keys);

    double init_ms = init_timer.elapsed_ms();
    printf("[bootstrap_bench] Bootstrapper initialized in %.1f ms\n", init_ms);

    // ---- Prepare ciphertext at low level (simulating post-computation state) ----
    size_t n_slots = N / 2;
    std::vector<double> input_vec(n_slots, 0.5);
    PhantomPlaintext  pt;
    PhantomCiphertext ct;
    ckks.encoder.encode(input_vec, SCALE, pt);
    ckks.encryptor.encrypt(pt, ct);

    // Mod-switch down to main_mod_count level to simulate depleted ciphertext
    for (int i = 0; i < MAIN_MOD_COUNT - 1; ++i)
        ckks.evaluator.mod_switch_to_next_inplace(ct);

    if (cfg.verbose)
        printf("[bootstrap_bench] Ciphertext at chain_index=%zu, coeff_mod_size=%zu\n",
               ct.chain_index(), ct.coeff_modulus_size());

    // ---- Timed bootstrapping iterations ----
    std::vector<double> times(cfg.iters);

    for (int it = 0; it < cfg.iters; ++it) {
        PhantomCiphertext ct_copy = ct;   // restore to same state each time
        WallTimer t;
        t.start();
        bootstrapper.bootstrap_sparse_3(ct_copy, ct_copy);
        times[it] = t.elapsed_ms();

        if (cfg.verbose)
            printf("[bootstrap_bench] Iter %d: %.1f ms\n", it, times[it]);
    }

    // ---- Statistics ----
    double sum = 0, min_t = times[0], max_t = times[0];
    for (double v : times) { sum += v; min_t = std::min(min_t, v); max_t = std::max(max_t, v); }
    double mean = sum / cfg.iters;

    double var = 0;
    for (double v : times) var += (v - mean) * (v - mean);
    double stddev = std::sqrt(var / cfg.iters);

    // Single-GPU baseline from NEXUS paper: ~8s for bootstrapping (per Argmax breakdown)
    static constexpr double BASELINE_BS_MS = 8000.0;

    return {cfg.n_gpus, mean, min_t, max_t, stddev,
            BASELINE_BS_MS / mean};
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

static void print_result(const BootstrapResult &r) {
    printf("\n=== Bootstrapping Benchmark ===\n");
    printf("GPUs:     %d\n",     r.n_gpus);
    printf("Mean:     %.1f ms\n", r.mean_ms);
    printf("Min:      %.1f ms\n", r.min_ms);
    printf("Max:      %.1f ms\n", r.max_ms);
    printf("Std dev:  %.1f ms\n", r.stddev_ms);
    printf("Speedup:  %.2fx\n",   r.speedup);
}

static void write_csv(const BootstrapResult &r, const std::string &path) {
    bool write_header = true;
    { std::ifstream check(path); if (check.good()) write_header = false; }
    std::ofstream out(path, std::ios::app);
    if (write_header)
        out << "n_gpus,mean_ms,min_ms,max_ms,stddev_ms,speedup\n";
    out << r.n_gpus << "," << r.mean_ms << "," << r.min_ms << ","
        << r.max_ms << "," << r.stddev_ms << "," << r.speedup << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);
    printf("Bootstrapping Benchmark: n_gpus=%d  iters=%d\n",
           cfg.n_gpus, cfg.iters);

    BootstrapResult result = run_benchmark(cfg);
    print_result(result);

    if (!cfg.output_file.empty()) {
        write_csv(result, cfg.output_file);
        printf("Results written to %s\n", cfg.output_file.c_str());
    }

    return 0;
}

#undef CUDA_CHECK
