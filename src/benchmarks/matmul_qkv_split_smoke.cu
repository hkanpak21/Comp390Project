/**
 * matmul_qkv_split_smoke.cu  (PRD slice F2)
 *
 * Split QKV matmul across 4 GPUs.
 *
 * BERT QKV: 3 separate matmuls (Q, K, V), each producing 64 output
 * columns. Total 192 column-matmuls across all three. Distribute the 192
 * across 4 GPUs (48 cols/GPU) — each GPU computes a contiguous block of
 * cols spanning whichever of Q/K/V it lands in.
 *
 * Acceptance (PRD F2):
 *   - Per-GPU matmul time drops from ~44 ms to ~11 ms (4× speedup target).
 *   - Per-column MAE preserved between split and single-GPU reference
 *     (< 1e-6).
 *
 * Pattern is the same per-thread Phantom design as F1
 * (`matmul_split_smoke.cu`): SK + per-column input ciphertext serialised
 * on GPU 0 and broadcast to every worker so cross-GPU MAE comparison
 * isolates the split. Warmup discard matmul on each thread to avoid the
 * first-matmul lazy-init garbage.
 *
 * CLI:
 *   matmul_qkv_split_smoke              (default: 4 GPUs, 64 cols/output, 768 inner)
 *   matmul_qkv_split_smoke --n-gpus 2
 *   matmul_qkv_split_smoke --cols-per-output 32 --inner 384
 */

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"

#include "ckks_evaluator.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

namespace {

struct Config {
    int n_gpus = 4;
    int cols_per_output = 64;   // BERT-base: 64 output cols per of Q/K/V
    int inner_dim = 768;        // BERT-base hidden dim
    int n_outputs = 3;          // Q, K, V
};

Config parse_cli(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--n-gpus" && i + 1 < argc) c.n_gpus = atoi(argv[++i]);
        else if (a == "--cols-per-output" && i + 1 < argc) c.cols_per_output = atoi(argv[++i]);
        else if (a == "--inner" && i + 1 < argc) c.inner_dim = atoi(argv[++i]);
        else if (a == "--n-outputs" && i + 1 < argc) c.n_outputs = atoi(argv[++i]);
        else if (a == "--help" || a == "-h") {
            printf("Usage: matmul_qkv_split_smoke [--n-gpus N] [--cols-per-output C] "
                   "[--inner D] [--n-outputs O]\n");
            exit(0);
        }
    }
    return c;
}

struct ThreadResult {
    int gpu = -1;
    int col_lo = -1;            // global column index (0..n_total_cols)
    int col_hi = -1;
    bool finished = false;
    bool exception = false;
    string err;
    double matmul_ms = -1.0;
    vector<vector<double>> decoded_cols;
};

// One column matmul: inner_dim multiply_plain + add_many + rescale.
PhantomCiphertext matmul_column(PhantomContext &ctx, CKKSEvaluator &eval,
                                const PhantomCiphertext &col_ct,
                                const vector<PhantomPlaintext> &weights) {
    int inner = (int)weights.size();
    vector<PhantomCiphertext> temp_cts(inner);
    for (int j = 0; j < inner; j++)
        temp_cts[j] = multiply_plain(ctx, col_ct, weights[j]);
    PhantomCiphertext acc;
    acc = add(ctx, temp_cts[0], temp_cts[1]);
    for (int j = 2; j < inner; j++)
        add_inplace(ctx, acc, temp_cts[j]);
    rescale_to_next_inplace(ctx, acc);
    return acc;
}

}  // namespace

int main(int argc, char **argv) {
    Config cfg = parse_cli(argc, argv);

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < cfg.n_gpus) {
        fprintf(stderr,
                "[FATAL] requested %d GPUs but only %d visible\n",
                cfg.n_gpus, dev_count);
        return 1;
    }

    int n_total_cols = cfg.cols_per_output * cfg.n_outputs;
    if (n_total_cols % cfg.n_gpus != 0) {
        fprintf(stderr,
                "[FATAL] n_total_cols (%d = %d outputs × %d cols) must be "
                "divisible by n_gpus (%d)\n",
                n_total_cols, cfg.n_outputs, cfg.cols_per_output, cfg.n_gpus);
        return 1;
    }

    printf("════════════════════════════════════════════════════════════\n");
    printf("  F2 — Split QKV matmul (3 outputs × %d cols across %d GPUs)\n",
           cfg.cols_per_output, cfg.n_gpus);
    printf("  n_total_cols = %d (Q/K/V × %d each), inner_dim = %d\n",
           n_total_cols, cfg.cols_per_output, cfg.inner_dim);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    // Small CKKS for smoke speed
    size_t poly_degree = 8192;
    vector<int> coeff_bits = {52, 40, 40, 40, 52};
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_degree, coeff_bits));
    const double SCALE = (double)(1ULL << 40);

    size_t slots = poly_degree / 2;
    printf("[Setup] CKKS N=%zu, L=%d, slots=%zu, scale=2^40\n",
           poly_degree, (int)coeff_bits.size() - 1, slots);
    fflush(stdout);

    // Generate SK + per-column input ciphertexts on GPU 0
    cudaSetDevice(0);
    string sk_buf;
    vector<string> col_ct_buf(n_total_cols);
    vector<vector<double>> input_data(n_total_cols, vector<double>(slots));
    // Three separate weight matrices for Q, K, V
    vector<vector<vector<double>>> weights_qkv(cfg.n_outputs);
    for (int o = 0; o < cfg.n_outputs; o++)
        weights_qkv[o].assign(cfg.inner_dim, vector<double>(slots));

    {
        PhantomContext     ctx0(parms);
        PhantomCKKSEncoder enc0(ctx0);
        PhantomSecretKey   sk0(ctx0);
        PhantomPublicKey   pk0 = sk0.gen_publickey(ctx0);
        PhantomRelinKey    rk0 = sk0.gen_relinkey(ctx0);
        PhantomGaloisKey   gk0_empty;
        CKKSEvaluator      eval0(&ctx0, &pk0, &sk0, &enc0, &rk0, &gk0_empty,
                                 SCALE);

        mt19937 rng(12345);
        uniform_real_distribution<double> idist(-1.0, 1.0);
        uniform_real_distribution<double> wdist(-0.01, 0.01);

        // Generate 3 weight matrices (different seeds for Q, K, V)
        for (int o = 0; o < cfg.n_outputs; o++)
            for (int j = 0; j < cfg.inner_dim; j++)
                for (size_t s = 0; s < slots; s++)
                    weights_qkv[o][j][s] = wdist(rng);

        for (int c = 0; c < n_total_cols; c++) {
            for (size_t s = 0; s < slots; s++)
                input_data[c][s] = idist(rng);

            PhantomPlaintext pt;
            enc0.encode(ctx0, input_data[c], SCALE, pt);
            PhantomCiphertext ct;
            eval0.encryptor.encrypt(pt, ct);

            stringstream ss; ct.save(ss); col_ct_buf[c] = ss.str();
        }
        {
            stringstream ss; sk0.save(ss); sk_buf = ss.str();
        }
        printf("[Setup] SK + %d input ciphertexts serialised (sk=%zu bytes)\n",
               n_total_cols, sk_buf.size());
        fflush(stdout);
    }
    cudaDeviceSynchronize();

    // Helper: which Q/K/V output does global column c belong to?
    auto output_of = [&](int c) { return c / cfg.cols_per_output; };

    // Single-GPU reference
    printf("[Phase 1] Single-GPU reference (all %d cols on GPU 0)...\n",
           n_total_cols);
    fflush(stdout);
    vector<vector<double>> ref_decoded(n_total_cols);
    double ref_ms = -1.0;
    {
        cudaSetDevice(0);
        PhantomContext     ctx(parms);
        PhantomCKKSEncoder enc(ctx);
        PhantomSecretKey   sk;
        { stringstream ss(sk_buf); sk.load(ss); }
        PhantomPublicKey   pk = sk.gen_publickey(ctx);
        PhantomRelinKey    rk = sk.gen_relinkey(ctx);
        PhantomGaloisKey   gk_empty;
        CKKSEvaluator      eval(&ctx, &pk, &sk, &enc, &rk, &gk_empty, SCALE);

        // Encode all 3 weight matrices on this context
        vector<vector<PhantomPlaintext>> wpts(cfg.n_outputs);
        for (int o = 0; o < cfg.n_outputs; o++) {
            wpts[o].resize(cfg.inner_dim);
            for (int j = 0; j < cfg.inner_dim; j++)
                enc.encode(ctx, weights_qkv[o][j], SCALE, wpts[o][j]);
        }

        // Warmup
        {
            PhantomCiphertext ct_warm;
            { stringstream ss(col_ct_buf[0]); ct_warm.load(ss); }
            PhantomCiphertext out_warm = matmul_column(ctx, eval, ct_warm, wpts[0]);
            cudaDeviceSynchronize();
            (void)out_warm;
        }

        auto t0 = chrono::high_resolution_clock::now();
        for (int c = 0; c < n_total_cols; c++) {
            int o = output_of(c);
            PhantomCiphertext ct_in;
            { stringstream ss(col_ct_buf[c]); ct_in.load(ss); }
            PhantomCiphertext out = matmul_column(ctx, eval, ct_in, wpts[o]);
            cudaDeviceSynchronize();

            PhantomPlaintext pt_out;
            sk.decrypt(ctx, out, pt_out);
            enc.decode(ctx, pt_out, ref_decoded[c]);
        }
        auto t1 = chrono::high_resolution_clock::now();
        ref_ms = chrono::duration<double, milli>(t1 - t0).count();
        printf("[Phase 1] reference done in %.1f ms (%.2f ms/col)\n",
               ref_ms, ref_ms / n_total_cols);
        fflush(stdout);
    }

    // Split path: each GPU owns a contiguous block of columns
    printf("[Phase 2] Split-path (%d cols across %d GPUs in parallel)...\n",
           n_total_cols, cfg.n_gpus);
    fflush(stdout);
    vector<ThreadResult> results(cfg.n_gpus);
    int cols_per_gpu = n_total_cols / cfg.n_gpus;
    atomic<int> ready{0};
    auto split_t0 = chrono::high_resolution_clock::now();

    vector<thread> threads;
    for (int g = 0; g < cfg.n_gpus; g++) {
        threads.emplace_back([&, g]() {
            ThreadResult &r = results[g];
            r.gpu = g;
            r.col_lo = g * cols_per_gpu;
            r.col_hi = (g + 1) * cols_per_gpu;
            try {
                cudaSetDevice(g);

                PhantomContext     ctx(parms);
                PhantomCKKSEncoder enc(ctx);
                PhantomSecretKey   sk;
                { stringstream ss(sk_buf); sk.load(ss); }
                PhantomPublicKey   pk = sk.gen_publickey(ctx);
                PhantomRelinKey    rk = sk.gen_relinkey(ctx);
                PhantomGaloisKey   gk_empty;
                CKKSEvaluator      eval(&ctx, &pk, &sk, &enc, &rk, &gk_empty,
                                        SCALE);

                vector<vector<PhantomPlaintext>> wpts(cfg.n_outputs);
                for (int o = 0; o < cfg.n_outputs; o++) {
                    wpts[o].resize(cfg.inner_dim);
                    for (int j = 0; j < cfg.inner_dim; j++)
                        enc.encode(ctx, weights_qkv[o][j], SCALE, wpts[o][j]);
                }

                r.decoded_cols.resize(r.col_hi - r.col_lo);

                // Warmup discard
                {
                    PhantomCiphertext ct_warm;
                    { stringstream ss(col_ct_buf[r.col_lo]); ct_warm.load(ss); }
                    PhantomCiphertext out_warm =
                        matmul_column(ctx, eval, ct_warm, wpts[output_of(r.col_lo)]);
                    cudaDeviceSynchronize();
                    (void)out_warm;
                }

                ready.fetch_add(1);
                while (ready.load() < cfg.n_gpus) { /* spin */ }

                auto t0 = chrono::high_resolution_clock::now();
                for (int c = r.col_lo; c < r.col_hi; c++) {
                    int o = output_of(c);
                    PhantomCiphertext ct_in;
                    { stringstream ss(col_ct_buf[c]); ct_in.load(ss); }
                    PhantomCiphertext out = matmul_column(ctx, eval, ct_in, wpts[o]);
                    cudaDeviceSynchronize();

                    PhantomPlaintext pt_out;
                    sk.decrypt(ctx, out, pt_out);
                    enc.decode(ctx, pt_out, r.decoded_cols[c - r.col_lo]);
                }
                auto t1 = chrono::high_resolution_clock::now();
                r.matmul_ms = chrono::duration<double, milli>(t1 - t0).count();

                // Print per-GPU breakdown by Q/K/V coverage
                int q_lo = std::max(r.col_lo, 0);
                int q_hi = std::min(r.col_hi, cfg.cols_per_output);
                int k_lo = std::max(r.col_lo, cfg.cols_per_output);
                int k_hi = std::min(r.col_hi, 2 * cfg.cols_per_output);
                int v_lo = std::max(r.col_lo, 2 * cfg.cols_per_output);
                int v_hi = std::min(r.col_hi, 3 * cfg.cols_per_output);
                printf("[T%d / GPU %d] cols=[%d,%d) "
                       "(Q:%d, K:%d, V:%d) matmul=%.1f ms (%.2f ms/col)\n",
                       g, g, r.col_lo, r.col_hi,
                       std::max(0, q_hi - q_lo),
                       std::max(0, k_hi - k_lo),
                       std::max(0, v_hi - v_lo),
                       r.matmul_ms,
                       r.matmul_ms / std::max(1, r.col_hi - r.col_lo));
                fflush(stdout);
                r.finished = true;
            } catch (std::exception &e) {
                r.exception = true;
                r.err = e.what();
                fprintf(stderr, "[T%d / GPU %d] EXCEPTION: %s\n", g, g, e.what());
            } catch (...) {
                r.exception = true;
                r.err = "(unknown)";
                fprintf(stderr, "[T%d / GPU %d] EXCEPTION (unknown)\n", g, g);
            }
        });
    }
    for (auto &t : threads) t.join();
    auto split_t1 = chrono::high_resolution_clock::now();
    double split_wall_ms =
        chrono::duration<double, milli>(split_t1 - split_t0).count();
    printf("[Phase 2] split wall (slowest GPU) = %.1f ms\n", split_wall_ms);
    fflush(stdout);

    // Verification
    printf("\n────────────── Per-column MAE (split vs single-GPU ref) "
           "──────────────\n");
    int ok_count = 0, fail_count = 0;
    double max_mae = 0.0;
    int max_mae_col = -1;

    bool any_thread_failed = false;
    for (auto &r : results)
        if (!r.finished || r.exception) any_thread_failed = true;

    if (any_thread_failed) {
        for (auto &r : results) {
            printf("  GPU %d: %s%s\n", r.gpu,
                   r.exception ? "EXCEPTION " : "",
                   r.exception ? r.err.c_str() : "(unfinished)");
        }
        printf("\n════════════════════════════════════════════════════════════\n");
        printf("  F2 RESULT: FAIL (thread error)\n");
        printf("════════════════════════════════════════════════════════════\n");
        return 1;
    }

    for (int c = 0; c < n_total_cols; c++) {
        int g = c / cols_per_gpu;
        int local = c % cols_per_gpu;
        const auto &ref = ref_decoded[c];
        const auto &got = results[g].decoded_cols[local];

        if (got.size() != ref.size()) {
            printf("  col %3d: SIZE MISMATCH (ref=%zu, got=%zu) FAIL\n",
                   c, ref.size(), got.size());
            fail_count++;
            continue;
        }
        double s = 0.0;
        for (size_t i = 0; i < ref.size(); i++)
            s += fabs(ref[i] - got[i]);
        double mae = s / ref.size();
        if (mae > max_mae) { max_mae = mae; max_mae_col = c; }

        bool pass = mae < 1e-6;
        if (pass) ok_count++; else fail_count++;

        // Print first/last per output (Q, K, V)
        if (c < 4 || c >= n_total_cols - 4 ||
            c == cfg.cols_per_output || c == 2 * cfg.cols_per_output) {
            const char *out_label = (output_of(c) == 0) ? "Q"
                                  : (output_of(c) == 1) ? "K" : "V";
            printf("  col %3d (%s, GPU %d): MAE=%.3e %s\n",
                   c, out_label, g, mae, pass ? "OK" : "FAIL");
        }
    }

    printf("\n────────────── Summary ──────────────\n");
    printf("  pass=%d / %d (max MAE = %.3e at col %d)\n",
           ok_count, n_total_cols, max_mae, max_mae_col);
    printf("  ref wall = %.1f ms (single GPU, all %d cols)\n",
           ref_ms, n_total_cols);
    printf("  split wall = %.1f ms (%d GPUs, %d cols each)\n",
           split_wall_ms, cfg.n_gpus, cols_per_gpu);
    printf("  parallel speedup vs reference: %.2f×\n",
           ref_ms / std::max(1.0, split_wall_ms));
    double avg_per_gpu_matmul = 0.0;
    int finished_gpus = 0;
    for (auto &r : results)
        if (r.finished) { avg_per_gpu_matmul += r.matmul_ms; finished_gpus++; }
    if (finished_gpus > 0) avg_per_gpu_matmul /= finished_gpus;
    printf("  per-GPU matmul time (avg): %.1f ms (target: ~%.1f ms = ref/%d)\n",
           avg_per_gpu_matmul, ref_ms / cfg.n_gpus, cfg.n_gpus);

    bool overall_pass = (fail_count == 0);
    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  F2 RESULT: %s\n", overall_pass ? "PASS" : "FAIL");
    printf("  threshold: per-column MAE < 1e-6\n");
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    return overall_pass ? 0 : 1;
}
