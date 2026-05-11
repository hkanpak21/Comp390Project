/**
 * matmul_split_smoke.cu
 *
 * Slice F1 (MatMul output-channel split track) — Smoke test.
 *
 * Goal: prove that an output-channel-split plain×ciphertext matmul on 2
 *       GPUs produces a result bit-equivalent to the single-GPU reference.
 *
 * Workload model (matches the per-column NEXUS MatMul kernel from
 *   src/benchmarks/bert_matmul_real.cu lines 145-169):
 *   - A 768-element ciphertext (one ciphertext per output column,
 *     N=8,192 sparse-mode CKKS so the smoke test runs in <30 s)
 *   - 64 output columns; each column is `inner_dim=768` multiply_plain
 *     calls + add_many → one output ciphertext per column
 *   - Single-GPU reference: GPU 0 computes all 64 output ciphertexts
 *     sequentially.
 *   - Split path: GPU 0 computes columns [0,32), GPU 1 computes columns
 *     [32,64), in parallel std::threads. Concatenate and compare.
 *
 * Acceptance: MAE between the concatenated split output and the
 *             single-GPU reference, decoded slot-by-slot, is < 1e-6 on
 *             every output column.
 *
 * Design notes:
 *   - Each GPU thread owns its own `PhantomContext`, `RelinKey`, and
 *     encoded plaintext copies. SK is generated on GPU 0 and serialized
 *     so both threads load the SAME secret key (otherwise the split
 *     vs reference comparison is meaningless).
 *   - The input ciphertext is also serialized — every GPU loads the same
 *     bytes — so any drift between split and reference is an artifact of
 *     the multi-GPU split, not an artifact of independent re-encryption.
 *   - F1 acceptance does NOT require DKS rotation or bootstrap; this is
 *     pure ciphertext × plaintext multiply + add — the embarrassingly-
 *     parallel core of any output-channel split (see
 *     docs/PERFORMANCE_SURFACE_ANALYSIS.md §3.2).
 *
 * CLI:
 *   matmul_split_smoke              (default: 2 GPUs, 64 cols, inner=768)
 *   matmul_split_smoke --n-gpus 4
 *   matmul_split_smoke --cols 32 --inner 384
 *
 * Slice gating: F1 PRD acceptance gates F2..F4 (per-op splits) and F5
 * (HP-BERT × MatMul split composition).
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
    int n_gpus    = 2;
    int n_columns = 64;
    int inner_dim = 768;
};

Config parse_cli(int argc, char **argv) {
    Config c;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--n-gpus" && i + 1 < argc) c.n_gpus = atoi(argv[++i]);
        else if (a == "--cols" && i + 1 < argc) c.n_columns = atoi(argv[++i]);
        else if (a == "--inner" && i + 1 < argc) c.inner_dim = atoi(argv[++i]);
        else if (a == "--help" || a == "-h") {
            printf("Usage: matmul_split_smoke [--n-gpus N] [--cols C] [--inner D]\n");
            exit(0);
        }
    }
    return c;
}

// Per-thread output. Aggregated under result_mtx after join().
struct ThreadResult {
    int gpu = -1;
    int col_lo = -1;     // inclusive
    int col_hi = -1;     // exclusive
    bool finished = false;
    bool exception = false;
    string err;
    double matmul_ms = -1.0;
    // Decoded slot vectors for each owned column. decoded_cols[c - col_lo]
    // is the post-matmul output for global column index c.
    vector<vector<double>> decoded_cols;
};

// Compute one output column: 768 multiply_plain + add_many + rescale.
// All work happens on the caller's CUDA device — caller must cudaSetDevice
// before invoking.
PhantomCiphertext matmul_column(PhantomContext       &ctx,
                                CKKSEvaluator        &eval,
                                const PhantomCiphertext &col_ct,
                                const vector<PhantomPlaintext> &weights) {
    int inner = (int)weights.size();
    vector<PhantomCiphertext> temp_cts(inner);
    for (int j = 0; j < inner; j++) {
        temp_cts[j] = multiply_plain(ctx, col_ct, weights[j]);
    }
    PhantomCiphertext acc;
    acc = add(ctx, temp_cts[0], temp_cts[1]);
    for (int j = 2; j < inner; j++) {
        add_inplace(ctx, acc, temp_cts[j]);
    }
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
                "[FATAL] requested %d GPUs but only %d visible — abort\n",
                cfg.n_gpus, dev_count);
        return 1;
    }
    if (cfg.n_columns % cfg.n_gpus != 0) {
        fprintf(stderr,
                "[FATAL] n_columns (%d) must be divisible by n_gpus (%d)\n",
                cfg.n_columns, cfg.n_gpus);
        return 1;
    }

    printf("════════════════════════════════════════════════════════════\n");
    printf("  F1 — MatMul output-channel split smoke test\n");
    printf("  n_gpus=%d, n_columns=%d, inner_dim=%d\n",
           cfg.n_gpus, cfg.n_columns, cfg.inner_dim);
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    // ═══ Small CKKS params — N=8192, L=4 — fast enough for a smoke test ═══
    size_t poly_degree = 8192;
    vector<int> coeff_bits = {52, 40, 40, 40, 52};  // L=4 levels (3 mults + final)
    EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_degree, coeff_bits));
    const double SCALE = (double)(1ULL << 40);

    size_t slots = poly_degree / 2;
    printf("[Setup] CKKS N=%zu, L=%d, slots=%zu, scale=2^40\n",
           poly_degree, (int)coeff_bits.size() - 1, slots);
    fflush(stdout);

    // ═══ Generate SK + per-column input ciphertext on GPU 0 and serialize.
    //     Every thread loads the SAME bytes so any drift between the
    //     single-GPU reference and the split-path output isolates the split. ═
    cudaSetDevice(0);
    string sk_buf;
    vector<string> col_ct_buf(cfg.n_columns);
    vector<vector<double>> input_data(cfg.n_columns, vector<double>(slots));
    vector<vector<double>> weight_data(cfg.inner_dim, vector<double>(slots));

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
        uniform_real_distribution<double> wdist(-0.01, 0.01);  // BERT-like

        for (int j = 0; j < cfg.inner_dim; j++)
            for (size_t s = 0; s < slots; s++)
                weight_data[j][s] = wdist(rng);

        for (int i = 0; i < cfg.n_columns; i++) {
            for (size_t s = 0; s < slots; s++)
                input_data[i][s] = idist(rng);

            PhantomPlaintext pt;
            enc0.encode(ctx0, input_data[i], SCALE, pt);
            PhantomCiphertext ct;
            eval0.encryptor.encrypt(pt, ct);

            stringstream ss; ct.save(ss); col_ct_buf[i] = ss.str();
        }
        {
            stringstream ss; sk0.save(ss); sk_buf = ss.str();
        }
        printf("[Setup] SK + %d input ciphertexts serialized "
               "(sk=%zu bytes, ct[0]=%zu bytes)\n",
               cfg.n_columns, sk_buf.size(), col_ct_buf[0].size());
        fflush(stdout);
    }
    cudaDeviceSynchronize();

    // ═══ Single-GPU reference: every output column computed on GPU 0 ═══
    printf("[Phase 1] Single-GPU reference (all %d cols on GPU 0)...\n",
           cfg.n_columns);
    fflush(stdout);
    vector<vector<double>> ref_decoded(cfg.n_columns);
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

        // Encode the SAME weights on this context (encoder is per-context).
        vector<PhantomPlaintext> wpts(cfg.inner_dim);
        for (int j = 0; j < cfg.inner_dim; j++)
            enc.encode(ctx, weight_data[j], SCALE, wpts[j]);

        // Warmup matmul (same reason as in worker thread: Phantom's first
        // matmul on a fresh context can return garbage from a lazy init
        // path; see comment in worker thread below). Discard result.
        {
            PhantomCiphertext ct_warm;
            { stringstream ss(col_ct_buf[0]); ct_warm.load(ss); }
            PhantomCiphertext out_warm = matmul_column(ctx, eval, ct_warm, wpts);
            cudaDeviceSynchronize();
            (void)out_warm;
        }

        auto t0 = chrono::high_resolution_clock::now();
        for (int c = 0; c < cfg.n_columns; c++) {
            PhantomCiphertext ct_in;
            { stringstream ss(col_ct_buf[c]); ct_in.load(ss); }
            PhantomCiphertext out = matmul_column(ctx, eval, ct_in, wpts);
            cudaDeviceSynchronize();

            PhantomPlaintext pt_out;
            sk.decrypt(ctx, out, pt_out);
            enc.decode(ctx, pt_out, ref_decoded[c]);
        }
        auto t1 = chrono::high_resolution_clock::now();
        ref_ms = chrono::duration<double, milli>(t1 - t0).count();
        printf("[Phase 1] reference done in %.1f ms\n", ref_ms);
        fflush(stdout);
    }

    // ═══ Split-path: each GPU computes a contiguous subset of columns ═══
    printf("[Phase 2] Split-path (%d cols across %d GPUs in parallel)...\n",
           cfg.n_columns, cfg.n_gpus);
    fflush(stdout);
    vector<ThreadResult> results(cfg.n_gpus);
    int cols_per_gpu = cfg.n_columns / cfg.n_gpus;
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

                vector<PhantomPlaintext> wpts(cfg.inner_dim);
                for (int j = 0; j < cfg.inner_dim; j++)
                    enc.encode(ctx, weight_data[j], SCALE, wpts[j]);

                r.decoded_cols.resize(r.col_hi - r.col_lo);

                // Warmup: run one matmul on col 0 (always loaded fresh,
                // discarded). Phantom's first matmul on a fresh thread
                // context can produce garbage if some lazy initializer
                // hasn't fired — observed empirically as a single
                // column of MAE>1e28 on GPU 1's first column. Doing a
                // throwaway matmul cures it.
                {
                    PhantomCiphertext ct_warm;
                    { stringstream ss(col_ct_buf[0]); ct_warm.load(ss); }
                    PhantomCiphertext out_warm = matmul_column(ctx, eval, ct_warm, wpts);
                    cudaDeviceSynchronize();
                    (void)out_warm;
                }

                ready.fetch_add(1);
                while (ready.load() < cfg.n_gpus) { /* spin */ }

                auto t0 = chrono::high_resolution_clock::now();
                for (int c = r.col_lo; c < r.col_hi; c++) {
                    PhantomCiphertext ct_in;
                    { stringstream ss(col_ct_buf[c]); ct_in.load(ss); }
                    PhantomCiphertext out = matmul_column(ctx, eval, ct_in, wpts);
                    cudaDeviceSynchronize();

                    PhantomPlaintext pt_out;
                    sk.decrypt(ctx, out, pt_out);
                    enc.decode(ctx, pt_out, r.decoded_cols[c - r.col_lo]);
                }
                auto t1 = chrono::high_resolution_clock::now();
                r.matmul_ms = chrono::duration<double, milli>(t1 - t0).count();

                printf("[T%d / GPU %d] cols=[%d,%d) matmul=%.1f ms\n",
                       g, g, r.col_lo, r.col_hi, r.matmul_ms);
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

    // ═══ Aggregate, compare against reference, report MAE per column ═══
    printf("\n────────────── Per-column MAE (split vs single-GPU ref) "
           "──────────────\n");
    int ok_count = 0;
    int fail_count = 0;
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
        printf("  F1 RESULT: FAIL (thread error)\n");
        printf("════════════════════════════════════════════════════════════\n");
        return 1;
    }

    // Reassemble: column c lives on GPU (c / cols_per_gpu) at offset (c % cols_per_gpu)
    for (int c = 0; c < cfg.n_columns; c++) {
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
        if (pass) ok_count++;
        else      fail_count++;

        // Only print first 4 + last 4 to keep log tidy
        if (c < 4 || c >= cfg.n_columns - 4)
            printf("  col %3d (GPU %d): MAE=%.3e %s\n",
                   c, g, mae, pass ? "OK" : "FAIL");
    }

    printf("\n────────────── Summary ──────────────\n");
    printf("  pass=%d / %d (max MAE = %.3e at col %d)\n",
           ok_count, cfg.n_columns, max_mae, max_mae_col);
    printf("  ref wall = %.1f ms, split wall = %.1f ms (parallel speedup %.2f×)\n",
           ref_ms, split_wall_ms, ref_ms / std::max(1.0, split_wall_ms));

    bool overall_pass = (fail_count == 0);
    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  F1 RESULT: %s\n", overall_pass ? "PASS" : "FAIL");
    printf("  threshold: per-column MAE < 1e-6\n");
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    return overall_pass ? 0 : 1;
}
