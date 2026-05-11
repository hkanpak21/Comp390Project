/**
 * llama_hp_multigpu.cu  (PRD slice H2)
 *
 * Head-Parallel LLaMA decoder layer on 4 GPUs.
 *
 * Models supported (--model):
 *   llama-7b    : hidden=4096, heads=32, head_dim=128, ffn=11008 (SwiGLU)
 *                 default seq_len=16, ffn-inner-per-head = 11008/32 = 344
 *   llama-3-8b  : hidden=4096, heads=32, head_dim=128, ffn=14336 (SwiGLU)
 *                 default seq_len=8 (NEXUS Table IV), ffn-inner-per-head = 14336/32 = 448
 *
 *                 Single layer per invocation; 32 heads distributed
 *                 round-robin across 4 GPUs (8 heads per GPU sequential).
 *
 * Per-head compute (per Lane F's H1 baseline, 1 head ≈ 10.35 s on 1×H100):
 *   QKV → RoPE → QK^T → Softmax → Attn×V → OutProj
 *   → BS#1 → RMSNorm#1 (LN proxy)
 *   → BS#2
 *   → SwiGLU FFN: gate / up / SiLU(gate) / gate⊙up / down
 *   → BS#3 → RMSNorm#2
 *   → BS#4
 *
 * Pattern mirrors `bert_hp_multigpu.cu` (Lane D, S15-S17): each GPU thread
 * owns its own `PhantomContext + Bootstrapper + GaloisKeyStore`, runs N
 * heads sequentially, no DKS — single-GPU pinned-host key streaming via
 * Phase-1 `enable_key_streaming`. SK + per-head ciphertext bytes are
 * serialized on GPU 0 main thread and loaded by every worker so cross-GPU
 * MAE comparison isolates the head-parallel split.
 *
 * Acceptance (PRD H2):
 *   HP-LLaMA runs on 4 GPUs (8 heads per GPU), one layer end-to-end.
 *   MAE preserved vs single-GPU baseline (threshold 1e-5 per head).
 *
 * CLI:
 *   llama_hp_multigpu --n-gpus 4 --heads 32                         (default H2 config; LLaMA-7B dims)
 *   llama_hp_multigpu --n-gpus 4 --heads 4                          (smoke 1 head per GPU)
 *   llama_hp_multigpu --n-gpus 4 --heads 32 --skip-ref
 *   llama_hp_multigpu --n-gpus 4 --heads 32 --model llama-3-8b      (NEXUS Table IV apples-to-apples)
 *   llama_hp_multigpu --n-gpus 1 --heads 32 --layers 32 --model llama-3-8b --skip-ref --N 65536
 *
 * --model llama-3-8b sets seq_len=8 and FFN inner-per-head=448 by default.
 * --seq-len, --inner, --ffn-inner override those defaults individually.
 *
 * Note: keeps Lane F's RoPE-rotations-FIRST workaround (encoding the rope
 * masks AFTER rotating Q,K avoids the deep-chain-index segfault path).
 *
 * Note on attention: LLaMA-3 uses GQA (8 KV heads vs 32 Q heads) but for
 * FHE inference we (and presumably NEXUS's published Table IV) treat
 * attention as standard MHA — each head computes a self-contained QKV.
 */

#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "context.cuh"
#include "secretkey.h"
#include "evaluate.cuh"
#include "ckks.h"
#include "ciphertext.h"
#include "galois.cuh"

#include "ckks_evaluator.cuh"
#include "galois_key_store.cuh"
#include "gelu.cuh"
#include "softmax.cuh"
#include "layer_norm.cuh"
#include "matrix_mul.cuh"
#include "bootstrapping/Bootstrapper.cuh"

using namespace std;
using namespace phantom;
using namespace phantom::arith;
using namespace nexus;

// ---------------------------------------------------------------------------
// Timer
// ---------------------------------------------------------------------------
struct PerfTimer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        return chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
    }
};

// Per-head per-op timing accumulator. heads counts how many head completions
// contributed to the sums.
struct OpTimes {
    double qkv = 0, rope = 0, qk = 0, softmax = 0, av = 0, out = 0;
    double bs1 = 0, rms1 = 0, bs2 = 0;
    double ffn_gate = 0, ffn_up = 0, ffn_silu = 0, ffn_gate_mul_up = 0, ffn_down = 0;
    double bs3 = 0, rms2 = 0, bs4 = 0;
    int heads = 0;

    double total() const {
        return qkv + rope + qk + softmax + av + out
             + bs1 + rms1 + bs2
             + ffn_gate + ffn_up + ffn_silu + ffn_gate_mul_up + ffn_down
             + bs3 + rms2 + bs4;
    }
    void add(const OpTimes &o) {
        qkv += o.qkv; rope += o.rope; qk += o.qk; softmax += o.softmax;
        av += o.av; out += o.out;
        bs1 += o.bs1; rms1 += o.rms1; bs2 += o.bs2;
        ffn_gate += o.ffn_gate; ffn_up += o.ffn_up; ffn_silu += o.ffn_silu;
        ffn_gate_mul_up += o.ffn_gate_mul_up; ffn_down += o.ffn_down;
        bs3 += o.bs3; rms2 += o.rms2; bs4 += o.bs4;
        heads += o.heads;
    }
};

#define TIME_OP(field, ...) do {                        \
    PerfTimer _pt; _pt.start();                         \
    { __VA_ARGS__; }                                    \
    cudaDeviceSynchronize();                            \
    times.field += _pt.elapsed_ms();                    \
} while(0)

// ---------------------------------------------------------------------------
// SIMD slot folding for HP-LLaMA: pack N per-head ciphertexts into ONE
// ciphertext at layer boundaries, bootstrap once, then unpack.
//
// PACK LAYOUT:
//   packed slot [h*per_head_slots + s] ← head h's slot s, for s in [0, per_head_slots)
//   trailing slots in slot_count are zero-padded.
//
// CORRECTNESS NOTE: this uses decrypt → re-encode → re-encrypt (we hold the
// secret key on the server in this benchmark setup). For non-interactive
// deployment, NEXUS Algorithm 3 (rotation+masking SIMD fold) would be needed.
//
// Bootstrap a packed ciphertext is one bootstrap call instead of N — that's
// the entire point of this optimization.
// ---------------------------------------------------------------------------
static void pack_heads_for_bootstrap(
    PhantomContext        &ctx,
    CKKSEvaluator         &le,
    PhantomSecretKey      &sk,
    const vector<PhantomCiphertext> &per_head_cts,
    int per_head_slots,    // typically `hidden` (= head_dim, e.g. 128)
    PhantomCiphertext     &packed_out)
{
    const size_t H = per_head_cts.size();
    const size_t slot_count = le.encoder.slot_count();
    if (H == 0) throw std::runtime_error("pack_heads: empty input");
    if (H * (size_t)per_head_slots > slot_count)
        throw std::runtime_error("pack_heads: heads × per_head_slots > slot_count "
                                 "(SIMD packing won't fit)");

    // Decrypt each head, gather the first per_head_slots, place into packed[h*per_head_slots ..]
    vector<double> packed(slot_count, 0.0);
    for (size_t h = 0; h < H; h++) {
        PhantomCiphertext ct = per_head_cts[h];   // copy (decrypt is non-const)
        PhantomPlaintext pt;
        sk.decrypt(ctx, ct, pt);
        vector<double> dec;
        le.encoder.decode(pt, dec);
        size_t take = std::min((size_t)per_head_slots, dec.size());
        for (size_t s = 0; s < take; s++) {
            packed[h * (size_t)per_head_slots + s] = dec[s];
        }
    }

    // Use the scale of the first head's ciphertext so chain index/scale are
    // consistent with a fresh encrypt (post-bootstrap, the bootstrap replaces
    // chain_index anyway, so the input chain doesn't matter much beyond the
    // bootstrap entry conditions which are handled by the caller).
    double encode_scale = per_head_cts[0].scale();
    PhantomPlaintext pt_packed;
    le.encoder.encode(packed, encode_scale, pt_packed);
    le.encryptor.encrypt(pt_packed, packed_out);
}

// Inverse of pack_heads_for_bootstrap. Splits a packed ciphertext back into N
// per-head ciphertexts. Each output ct holds head h's data in slots [0, per_head_slots).
//
// We decrypt the packed ct, extract each per-head slice, and re-encrypt one ct
// per head. This preserves the property that downstream per-head compute
// expects head data in slots [0..per_head_slots).
static void unpack_heads_after_bootstrap(
    PhantomContext        &ctx,
    CKKSEvaluator         &le,
    PhantomSecretKey      &sk,
    PhantomCiphertext     &packed_in,
    int per_head_slots,
    int n_heads,
    vector<PhantomCiphertext> &per_head_out)
{
    const size_t slot_count = le.encoder.slot_count();
    if ((size_t)n_heads * (size_t)per_head_slots > slot_count)
        throw std::runtime_error("unpack_heads: heads × per_head_slots > slot_count");

    PhantomPlaintext pt;
    sk.decrypt(ctx, packed_in, pt);
    vector<double> dec;
    le.encoder.decode(pt, dec);

    per_head_out.assign(n_heads, PhantomCiphertext());
    double encode_scale = packed_in.scale();
    for (int h = 0; h < n_heads; h++) {
        vector<double> head_vec(slot_count, 0.0);
        for (int s = 0; s < per_head_slots; s++) {
            size_t src_idx = (size_t)h * (size_t)per_head_slots + (size_t)s;
            if (src_idx < dec.size()) head_vec[s] = dec[src_idx];
        }
        PhantomPlaintext pt_h;
        le.encoder.encode(head_vec, encode_scale, pt_h);
        le.encryptor.encrypt(pt_h, per_head_out[h]);
    }
}

// ---------------------------------------------------------------------------
// Run one full LLaMA decoder layer (one head) using single-GPU pinned-host
// rotation. Mirrors run_one_head in bert_hp_multigpu.cu but with LLaMA
// extras (RoPE on Q/K, SwiGLU FFN, RMSNorm proxy via LayerNorm evaluator).
//
// Returns the post-BS4 ciphertext via `ct_out` when `skip_final_bs4=false`,
// or the pre-BS4 (post-RMSNorm#2) ciphertext when `skip_final_bs4=true`.
// The latter mode enables the SIMD slot-folded layer-boundary bootstrap
// (pack heads → bootstrap once → unpack).
// ---------------------------------------------------------------------------
static void run_one_head_llama(
    PhantomContext   &ctx,
    CKKSEvaluator    &le,
    Bootstrapper     &lb,
    PhantomCiphertext &ct_in,
    PhantomCiphertext &ct_out,
    vector<vector<double>> &W_q,
    vector<vector<double>> &W_k,
    vector<vector<double>> &W_v,
    vector<vector<double>> &W_o,
    vector<vector<double>> &W_gate,
    vector<vector<double>> &W_up,
    vector<vector<double>> &W_down,
    vector<double>         &rope_cos,
    vector<double>         &rope_sin,
    int hidden,
    int seq_len,
    OpTimes &times,
    bool skip_final_bs4 = false)
{
    (void)ctx;
    GELUEvaluator    lg(le);   // SiLU proxy
    SoftmaxEvaluator ls(le);
    LNEvaluator      ll(le);   // RMSNorm proxy
    MMEvaluator      lm(le);

    // ── QKV projections ──
    vector<PhantomCiphertext> xi = {ct_in}, q, k, v;
    TIME_OP(qkv, {
        lm.matrix_mul_unified(xi, W_q, 1, q);
        lm.matrix_mul_unified(xi, W_k, 1, k);
        lm.matrix_mul_unified(xi, W_v, 1, v);
    });

    // ── RoPE on Q and K ──
    // Rotations FIRST, then encode/mod-switch (Lane F H1 ordering workaround
    // for the deep-chain-index segfault path).
    TIME_OP(rope, {
        PhantomCiphertext q_rot, k_rot;
        le.evaluator.rotate_vector(q[0], 1, *le.galois_keys, q_rot);
        le.evaluator.rotate_vector(k[0], 1, *le.galois_keys, k_rot);

        PhantomPlaintext pt_cos, pt_sin;
        le.encoder.encode(rope_cos, q[0].scale(),   pt_cos);
        le.encoder.encode(rope_sin, q_rot.scale(),  pt_sin);
        le.evaluator.mod_switch_to_inplace(pt_cos, q[0].chain_index());
        le.evaluator.mod_switch_to_inplace(pt_sin, q_rot.chain_index());

        PhantomCiphertext q_sin, k_sin, q_rot_pt, k_rot_pt;
        le.evaluator.multiply_plain(q[0],  pt_cos, q_sin);
        le.evaluator.multiply_plain(q_rot, pt_sin, q_rot_pt);
        le.evaluator.add_inplace(q_sin, q_rot_pt);
        le.evaluator.rescale_to_next_inplace(q_sin);

        le.evaluator.multiply_plain(k[0],  pt_cos, k_sin);
        le.evaluator.multiply_plain(k_rot, pt_sin, k_rot_pt);
        le.evaluator.add_inplace(k_sin, k_rot_pt);
        le.evaluator.rescale_to_next_inplace(k_sin);
        q[0] = q_sin;
        k[0] = k_sin;
    });

    // ── QK^T ──
    PhantomCiphertext qk_ct;
    TIME_OP(qk, {
        le.evaluator.mod_switch_to_inplace(k[0], q[0].chain_index());
        k[0].set_scale(q[0].scale());
        le.evaluator.multiply(q[0], k[0], qk_ct);
        le.evaluator.relinearize_inplace(qk_ct, *le.relin_keys);
        le.evaluator.rescale_to_next_inplace(qk_ct);
    });

    // ── Softmax ──
    PhantomCiphertext attn;
    TIME_OP(softmax, { ls.softmax(qk_ct, attn, seq_len); });

    // ── Attn × V ──
    PhantomCiphertext av_ct;
    TIME_OP(av, {
        le.evaluator.mod_switch_to_inplace(v[0], attn.chain_index());
        v[0].set_scale(attn.scale());
        le.evaluator.multiply(attn, v[0], av_ct);
        le.evaluator.relinearize_inplace(av_ct, *le.relin_keys);
        le.evaluator.rescale_to_next_inplace(av_ct);
    });

    // ── Output projection ──
    vector<PhantomCiphertext> ai = {av_ct}, ao;
    TIME_OP(out, { lm.matrix_mul_unified(ai, W_o, 1, ao); });

    // ── BS#1 ──
    PhantomCiphertext b1;
    TIME_OP(bs1, {
        while (ao[0].coeff_modulus_size() > 1)
            le.evaluator.mod_switch_to_next_inplace(ao[0]);
        lb.bootstrap_3(b1, ao[0]);
    });

    // ── RMSNorm #1 (LN proxy) ──
    PhantomCiphertext rms1_out;
    TIME_OP(rms1, { ll.layer_norm(b1, rms1_out, hidden); });

    // ── BS#2 ──
    PhantomCiphertext b2;
    TIME_OP(bs2, {
        while (rms1_out.coeff_modulus_size() > 1)
            le.evaluator.mod_switch_to_next_inplace(rms1_out);
        lb.bootstrap_3(b2, rms1_out);
    });

    // ── SwiGLU FFN: gate, up, SiLU(gate), gate⊙up, down ──
    vector<PhantomCiphertext> fi = {b2}, gate_out, up_out;
    TIME_OP(ffn_gate, { lm.matrix_mul_unified(fi, W_gate, 1, gate_out); });
    TIME_OP(ffn_up,   { lm.matrix_mul_unified(fi, W_up,   1, up_out);   });

    PhantomCiphertext silu_gate;
    TIME_OP(ffn_silu, { lg.gelu(gate_out[0], silu_gate); });

    PhantomCiphertext gated;
    TIME_OP(ffn_gate_mul_up, {
        le.evaluator.mod_switch_to_inplace(up_out[0], silu_gate.chain_index());
        up_out[0].set_scale(silu_gate.scale());
        le.evaluator.multiply(silu_gate, up_out[0], gated);
        le.evaluator.relinearize_inplace(gated, *le.relin_keys);
        le.evaluator.rescale_to_next_inplace(gated);
    });

    vector<PhantomCiphertext> di = {gated}, down_out;
    TIME_OP(ffn_down, { lm.matrix_mul_unified(di, W_down, 1, down_out); });

    // ── BS#3 ──
    PhantomCiphertext b3;
    TIME_OP(bs3, {
        while (down_out[0].coeff_modulus_size() > 1)
            le.evaluator.mod_switch_to_next_inplace(down_out[0]);
        lb.bootstrap_3(b3, down_out[0]);
    });

    // ── RMSNorm #2 ──
    PhantomCiphertext rms2_out;
    TIME_OP(rms2, { ll.layer_norm(b3, rms2_out, hidden); });

    // ── BS#4 ──
    if (skip_final_bs4) {
        // Drop levels to the bootstrap entry point and emit pre-BS4 ct.
        // The packed bootstrap will be applied once across all heads by the
        // caller (SIMD slot-folded layer-boundary bootstrap).
        TIME_OP(bs4, {
            while (rms2_out.coeff_modulus_size() > 1)
                le.evaluator.mod_switch_to_next_inplace(rms2_out);
        });
        ct_out = std::move(rms2_out);
    } else {
        TIME_OP(bs4, {
            while (rms2_out.coeff_modulus_size() > 1)
                le.evaluator.mod_switch_to_next_inplace(rms2_out);
            lb.bootstrap_3(ct_out, rms2_out);
        });
    }

    times.heads++;
}

// ---------------------------------------------------------------------------
// Per-GPU setup helper. Mirrors bert_hp_multigpu.cu's setup_per_gpu but
// the rotation steps include the full bidirectional LLaMA range.
// ---------------------------------------------------------------------------
static void setup_per_gpu(
    int g,
    const EncryptionParameters &parms,
    long logN, long logn, long logNh,
    int total_level, double SCALE,
    long boundary_K, long deg, long scale_factor, long inverse_deg, long loge,
    int seq_len, int hidden,
    const string &sk_buf,
    unique_ptr<PhantomContext>     &ctx_out,
    unique_ptr<PhantomCKKSEncoder> &enc_out,
    unique_ptr<PhantomSecretKey>   &sk_out,
    unique_ptr<PhantomPublicKey>   &pk_out,
    unique_ptr<PhantomRelinKey>    &rk_out,
    unique_ptr<PhantomGaloisKey>   &gk_out,
    unique_ptr<CKKSEvaluator>      &eval_out,
    unique_ptr<Bootstrapper>       &bs_out,
    unique_ptr<GaloisKeyStore>     &ks_out)
{
    ctx_out = make_unique<PhantomContext>(parms);
    enc_out = make_unique<PhantomCKKSEncoder>(*ctx_out);

    sk_out = make_unique<PhantomSecretKey>();
    { stringstream ss(sk_buf); sk_out->load(ss); }

    pk_out = make_unique<PhantomPublicKey>(sk_out->gen_publickey(*ctx_out));
    rk_out = make_unique<PhantomRelinKey>(sk_out->gen_relinkey(*ctx_out));
    gk_out = make_unique<PhantomGaloisKey>();

    eval_out = make_unique<CKKSEvaluator>(
        ctx_out.get(), pk_out.get(), sk_out.get(), enc_out.get(),
        rk_out.get(), gk_out.get(), SCALE);

    bs_out = make_unique<Bootstrapper>(
        loge, logn, logNh, total_level, SCALE,
        boundary_K, deg, scale_factor, inverse_deg, eval_out.get());
    bs_out->slot_vec.push_back(logn);
    bs_out->prepare_mod_polynomial();
    bs_out->generate_LT_coefficient_3();

    vector<int> gsteps;
    gsteps.push_back(0);
    for (int i = 0; i < logN - 1; i++) gsteps.push_back(1 << i);
    for (int i = 0; i < logN - 1; i++) gsteps.push_back(-(1 << i));
    gsteps.push_back(-seq_len);
    gsteps.push_back(-hidden);
    bs_out->addLeftRotKeys_Linear_to_vector_3(gsteps);

    {
        std::set<int> step_set(gsteps.begin(), gsteps.end());
        gsteps.assign(step_set.begin(), step_set.end());
    }
    auto gelts = ::get_elts_from_steps(gsteps, 1ULL << logN);
    ctx_out->setup_galois_tool(gelts);
    gk_out->resize_slots(gelts.size());

    ks_out = make_unique<GaloisKeyStore>();
    ks_out->generate_all_keys(*ctx_out, *sk_out, gelts.size());
    eval_out->evaluator.enable_key_streaming(ks_out.get(), gk_out.get());

    printf("[setup g=%d] context+keys ready (%zu rotation keys)\n",
           g, gelts.size());
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    int n_gpus  = 4;
    int n_heads = 32;
    int hidden  = 128;   // LLaMA head_dim (matches Lane F H1 baseline)
    int inner   = 128;   // attention matmul inner dim per head (= head_dim)
    int ffn_inner = -1;  // FFN matmul inner dim per head; if <0 derived from --model
    int seq_len = -1;    // sequence length; if <0 derived from --model
    int n_layers = 1;
    int n_trials = 1;
    bool skip_ref = false;
    int  ring_N  = 65536; // LLAMA-FULL: --N 32768 or 65536 for parity
    string model = "llama-7b";   // {llama-7b, llama-3-8b}
    bool pack_bootstrap = false; // SIMD slot folding: pack heads → 1 BS#4 per layer per GPU

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--n-gpus") && i+1 < argc)
            n_gpus = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--heads") && i+1 < argc)
            n_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc)
            inner = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ffn-inner") && i+1 < argc)
            ffn_inner = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq-len") && i+1 < argc)
            seq_len = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--hidden") && i+1 < argc)
            hidden = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--layers") && i+1 < argc)
            n_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trials") && i+1 < argc)
            n_trials = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--skip-ref"))
            skip_ref = true;
        else if (!strcmp(argv[i], "--N") && i+1 < argc)
            ring_N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--model") && i+1 < argc)
            model = argv[++i];
        else if (!strcmp(argv[i], "--pack-bootstrap"))
            pack_bootstrap = true;
    }
    if (ring_N != 32768 && ring_N != 65536) {
        fprintf(stderr, "Unsupported --N %d (use 32768 or 65536)\n", ring_N);
        return 1;
    }
    // Per-model defaults for seq_len and ffn_inner (per-head). CLI overrides win.
    int model_seq_len, model_ffn_inner_total;
    if (model == "llama-3-8b") {
        model_seq_len = 8;       // NEXUS Table IV LLaMA-3-8B
        model_ffn_inner_total = 14336;
    } else if (model == "llama-7b") {
        model_seq_len = 16;      // historical HP-LLaMA default (Lane F/H baseline)
        model_ffn_inner_total = 11008;
    } else {
        fprintf(stderr, "Unsupported --model %s (use llama-7b or llama-3-8b)\n",
                model.c_str());
        return 1;
    }
    if (seq_len < 0) seq_len = model_seq_len;
    if (ffn_inner < 0) {
        // Per-head FFN inner = ceil(model_ffn_inner_total / n_heads).
        ffn_inner = (model_ffn_inner_total + n_heads - 1) / n_heads;
    }
    (void)n_trials; // trial loop happens in SLURM script (fresh CUDA state per trial)

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count < n_gpus) {
        fprintf(stderr, "Need %d GPUs, have %d\n", n_gpus, dev_count);
        return 1;
    }

    printf("════════════════════════════════════════════════════════════\n");
    printf("  HP-LLaMA (head-parallel) — model=%s, N=%d, %d GPUs, %d heads, %d layer%s\n",
           model.c_str(), ring_N, n_gpus, n_heads, n_layers, n_layers == 1 ? "" : "s");
    printf("  per-GPU: %d head%s sequentially, single-GPU pinned rotations\n",
           (n_heads + n_gpus - 1) / n_gpus,
           (n_heads + n_gpus - 1) / n_gpus > 1 ? "s" : "");
    printf("  hidden=%d, attn_inner=%d, ffn_inner=%d, seq=%d\n",
           hidden, inner, ffn_inner, seq_len);
    printf("  ffn_total≈%d (per-head ffn_inner=%d × %d heads)\n",
           ffn_inner * n_heads, ffn_inner, n_heads);
    printf("  pack_bootstrap=%s (SIMD slot-folded layer-boundary BS#4)\n",
           pack_bootstrap ? "ON" : "off");
    printf("════════════════════════════════════════════════════════════\n");
    fflush(stdout);

    // ═══ CKKS params: N=65536 (default) or N=32768 (NEXUS-comparable) ═══
    long logN     = (ring_N == 65536) ? 16 : 15;
    long logn     = logN - 2;
    long logNh    = logN - 1;
    size_t N      = 1ULL << logN;
    long sparse_slots_val = 1L << logn;
    int  logp     = 46;
    int  logq     = 51;
    int  log_special = 51;
    int  main_mod = 21;
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

    // ═══ Setup on GPU 0: SK + weights + per-head input ═══
    cudaSetDevice(0);
    PhantomContext   ctx0(parms);
    PhantomCKKSEncoder enc0(ctx0);
    PhantomSecretKey   sk0(ctx0);
    PhantomPublicKey   pk0 = sk0.gen_publickey(ctx0);
    PhantomRelinKey    rk0 = sk0.gen_relinkey(ctx0);
    PhantomGaloisKey   gk0_empty;
    size_t slots = enc0.slot_count();
    CKKSEvaluator eval0(&ctx0, &pk0, &sk0, &enc0, &rk0, &gk0_empty, SCALE);

    stringstream sk_buf;
    sk0.save(sk_buf);
    string sk_str = sk_buf.str();

    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.02, 0.02), idist(-0.5, 0.5);
    auto make_w = [&](int rows) {
        vector<vector<double>> w(rows, vector<double>(slots, 0.0));
        for (auto &r : w)
            for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++)
                r[s] = wdist(rng);
        return w;
    };
    // Attention projections use attention inner = head_dim per head
    auto W_q    = make_w(inner);
    auto W_k    = make_w(inner);
    auto W_v    = make_w(inner);
    auto W_o    = make_w(inner);
    // FFN uses ffn_inner per head (= ffn_total / n_heads)
    auto W_gate = make_w(ffn_inner);
    auto W_up   = make_w(ffn_inner);
    auto W_down = make_w(ffn_inner);

    vector<double> rope_cos(slots, 0.0), rope_sin(slots, 0.0);
    for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++) {
        rope_cos[s] = cos(0.01 * s);
        rope_sin[s] = sin(0.01 * s);
    }

    vector<vector<double>> head_inputs(n_heads, vector<double>(slots, 0.0));
    for (int h = 0; h < n_heads; h++)
        for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++)
            head_inputs[h][s] = idist(rng);

    vector<string> ct_data(n_heads);
    for (int h = 0; h < n_heads; h++) {
        PhantomPlaintext pt;
        eval0.encoder.encode(head_inputs[h], SCALE, pt);
        PhantomCiphertext ct;
        eval0.encryptor.encrypt(pt, ct);
        for (int j = 0; j < bs_mod; j++)
            eval0.evaluator.mod_switch_to_next_inplace(ct);
        stringstream ss; ct.save(ss);
        ct_data[h] = ss.str();
    }
    printf("[setup] encrypted %d head input ciphertexts on GPU 0\n", n_heads);
    fflush(stdout);

    // ═══ Reference pass on GPU 0 ═══
    vector<vector<double>> ref_outputs(n_heads);
    if (!skip_ref) {
        printf("\n[ref] computing single-GPU reference for %d heads on GPU 0...\n", n_heads);
        fflush(stdout);

        unique_ptr<PhantomContext>     ref_ctx;
        unique_ptr<PhantomCKKSEncoder> ref_enc;
        unique_ptr<PhantomSecretKey>   ref_sk;
        unique_ptr<PhantomPublicKey>   ref_pk;
        unique_ptr<PhantomRelinKey>    ref_rk;
        unique_ptr<PhantomGaloisKey>   ref_gk;
        unique_ptr<CKKSEvaluator>      ref_eval;
        unique_ptr<Bootstrapper>       ref_bs;
        unique_ptr<GaloisKeyStore>     ref_ks;

        cudaSetDevice(0);
        setup_per_gpu(
            0, parms, logN, logn, logNh, total_level, SCALE,
            boundary_K, deg, scale_factor, inverse_deg, loge,
            seq_len, hidden, sk_str,
            ref_ctx, ref_enc, ref_sk, ref_pk, ref_rk, ref_gk,
            ref_eval, ref_bs, ref_ks);

        OpTimes ref_times;
        for (int h = 0; h < n_heads; h++) {
            PhantomCiphertext ct;
            { stringstream ss(ct_data[h]); ct.load(ss); }
            PhantomCiphertext ct_out;
            for (int layer = 0; layer < n_layers; ++layer) {
                ct_out = PhantomCiphertext();   // reset to fresh
                run_one_head_llama(
                    *ref_ctx, *ref_eval, *ref_bs, ct, ct_out,
                    W_q, W_k, W_v, W_o, W_gate, W_up, W_down,
                    rope_cos, rope_sin, hidden, seq_len, ref_times);
                ct = std::move(ct_out);
            }
            ct_out = std::move(ct);

            PhantomPlaintext pt;
            ref_sk->decrypt(*ref_ctx, ct_out, pt);
            ref_enc->decode(*ref_ctx, pt, ref_outputs[h]);
            printf("[ref] head %d done after %d layer(s) (decoded %zu slots)\n",
                   h, n_layers, ref_outputs[h].size());
            fflush(stdout);
        }
        printf("[ref] reference complete (sum of per-op times: %.1f ms)\n",
               ref_times.total());
        fflush(stdout);
    }

    // ═══ Distribute heads across GPUs ═══
    vector<vector<int>> gpu_heads(n_gpus);
    for (int i = 0; i < n_heads; i++) gpu_heads[i % n_gpus].push_back(i);
    for (int g = 0; g < n_gpus; g++) {
        printf("[dispatch] GPU %d: %zu head%s [", g, gpu_heads[g].size(),
               gpu_heads[g].size() == 1 ? "" : "s");
        for (size_t k = 0; k < gpu_heads[g].size(); k++)
            printf("%s%d", k == 0 ? "" : ",", gpu_heads[g][k]);
        printf("]\n");
    }
    fflush(stdout);

    // ═══ Threaded HP-LLaMA execution ═══
    printf("\n═══ Threaded HP-LLaMA on %d GPUs ═══\n", n_gpus);
    fflush(stdout);

    vector<vector<double>> head_outputs(n_heads);
    vector<bool>           head_done(n_heads, false);
    vector<string>         head_err(n_heads);
    OpTimes                global_times;
    mutex                  result_mtx;
    atomic<int> setup_done{0};
    PerfTimer compute_timer, total_timer;
    total_timer.start();

    vector<thread> threads;
    for (int g = 0; g < n_gpus; g++) {
        threads.emplace_back([&, g]() {
            try {
                cudaSetDevice(g);
                unique_ptr<PhantomContext>     ctx;
                unique_ptr<PhantomCKKSEncoder> enc;
                unique_ptr<PhantomSecretKey>   sk;
                unique_ptr<PhantomPublicKey>   pk;
                unique_ptr<PhantomRelinKey>    rk;
                unique_ptr<PhantomGaloisKey>   gk;
                unique_ptr<CKKSEvaluator>      eval;
                unique_ptr<Bootstrapper>       bs;
                unique_ptr<GaloisKeyStore>     ks;

                setup_per_gpu(
                    g, parms, logN, logn, logNh, total_level, SCALE,
                    boundary_K, deg, scale_factor, inverse_deg, loge,
                    seq_len, hidden, sk_str,
                    ctx, enc, sk, pk, rk, gk, eval, bs, ks);

                int my_count = setup_done.fetch_add(1) + 1;
                while (setup_done.load() < n_gpus) { /* spin */ }
                if (my_count == n_gpus) compute_timer.start();

                OpTimes times;
                if (!pack_bootstrap) {
                    // ── ORIGINAL: run heads sequentially, BS#4 per-head ──
                    for (int h_idx : gpu_heads[g]) {
                        PhantomCiphertext ct;
                        { stringstream ss(ct_data[h_idx]); ct.load(ss); }
                        PhantomCiphertext ct_out;
                        for (int layer = 0; layer < n_layers; ++layer) {
                            ct_out = PhantomCiphertext();   // reset to fresh
                            run_one_head_llama(
                                *ctx, *eval, *bs, ct, ct_out,
                                W_q, W_k, W_v, W_o, W_gate, W_up, W_down,
                                rope_cos, rope_sin, hidden, seq_len, times);
                            ct = std::move(ct_out);
                            if (n_layers > 1) {
                                printf("[GPU %d] head %d layer %d/%d done "
                                       "(out level=%zu)\n",
                                       g, h_idx, layer + 1, n_layers,
                                       ct.coeff_modulus_size());
                                fflush(stdout);
                            }
                        }
                        ct_out = std::move(ct);

                        PhantomPlaintext pt;
                        sk->decrypt(*ctx, ct_out, pt);
                        vector<double> dec;
                        enc->decode(*ctx, pt, dec);
                        {
                            lock_guard<mutex> lk(result_mtx);
                            head_outputs[h_idx] = std::move(dec);
                            head_done[h_idx] = true;
                        }
                        printf("[GPU %d] head %d COMPLETE after %d layer(s) "
                               "(out level=%zu)\n",
                               g, h_idx, n_layers, ct_out.coeff_modulus_size());
                        fflush(stdout);
                    }
                } else {
                    // ── PACKED: SIMD slot-folded layer-boundary BS#4 ──
                    // Maintain one ciphertext per head; for each layer,
                    // run pre-BS4 per head, then pack→bootstrap→unpack ONCE.
                    const int H = (int)gpu_heads[g].size();
                    vector<PhantomCiphertext> head_cts(H);
                    for (int i = 0; i < H; i++) {
                        stringstream ss(ct_data[gpu_heads[g][i]]);
                        head_cts[i].load(ss);
                    }
                    for (int layer = 0; layer < n_layers; ++layer) {
                        // 1) Pre-BS4 per head (everything through RMSNorm#2,
                        //    levels dropped to 1).
                        vector<PhantomCiphertext> pre_bs4(H);
                        for (int i = 0; i < H; i++) {
                            PhantomCiphertext ct_out;
                            run_one_head_llama(
                                *ctx, *eval, *bs, head_cts[i], ct_out,
                                W_q, W_k, W_v, W_o, W_gate, W_up, W_down,
                                rope_cos, rope_sin, hidden, seq_len, times,
                                /*skip_final_bs4=*/true);
                            pre_bs4[i] = std::move(ct_out);
                        }

                        // 2) Pack H heads' pre-BS4 cts into one ciphertext.
                        PhantomCiphertext packed;
                        PerfTimer _pt_pack; _pt_pack.start();
                        pack_heads_for_bootstrap(
                            *ctx, *eval, *sk,
                            pre_bs4, /*per_head_slots=*/hidden, packed);
                        cudaDeviceSynchronize();
                        double pack_ms = _pt_pack.elapsed_ms();

                        // 3) Bootstrap the packed ciphertext (one bootstrap
                        //    instead of H per layer per GPU).
                        PhantomCiphertext bs_out;
                        PerfTimer _pt_bs; _pt_bs.start();
                        // Drop levels in case re-encrypt landed at full level.
                        while (packed.coeff_modulus_size() > 1)
                            eval->evaluator.mod_switch_to_next_inplace(packed);
                        bs->bootstrap_3(bs_out, packed);
                        cudaDeviceSynchronize();
                        double bs_ms = _pt_bs.elapsed_ms();
                        // Account this packed bootstrap into the bs4 bucket
                        // so totals stay comparable.
                        times.bs4 += bs_ms;

                        // 4) Unpack the packed-bootstrapped ct back to H heads.
                        PerfTimer _pt_unpack; _pt_unpack.start();
                        unpack_heads_after_bootstrap(
                            *ctx, *eval, *sk,
                            bs_out, /*per_head_slots=*/hidden,
                            /*n_heads=*/H, head_cts);
                        cudaDeviceSynchronize();
                        double unpack_ms = _pt_unpack.elapsed_ms();

                        if (n_layers > 1) {
                            printf("[GPU %d] layer %d/%d packed-BS done "
                                   "(pack=%.1fms bs=%.1fms unpack=%.1fms "
                                   "out level=%zu)\n",
                                   g, layer + 1, n_layers, pack_ms, bs_ms,
                                   unpack_ms, head_cts[0].coeff_modulus_size());
                            fflush(stdout);
                        }
                    }

                    // Decode each head's final ct.
                    for (int i = 0; i < H; i++) {
                        int h_idx = gpu_heads[g][i];
                        PhantomPlaintext pt;
                        sk->decrypt(*ctx, head_cts[i], pt);
                        vector<double> dec;
                        enc->decode(*ctx, pt, dec);
                        {
                            lock_guard<mutex> lk(result_mtx);
                            head_outputs[h_idx] = std::move(dec);
                            head_done[h_idx] = true;
                        }
                        printf("[GPU %d] head %d COMPLETE after %d layer(s) "
                               "(packed-BS, out level=%zu)\n",
                               g, h_idx, n_layers,
                               head_cts[i].coeff_modulus_size());
                        fflush(stdout);
                    }
                }

                cudaDeviceSynchronize();
                {
                    lock_guard<mutex> lk(result_mtx);
                    global_times.add(times);
                }
            } catch (std::exception &e) {
                lock_guard<mutex> lk(result_mtx);
                fprintf(stderr, "[GPU %d] EXCEPTION: %s\n", g, e.what());
                for (int h_idx : gpu_heads[g])
                    if (!head_done[h_idx])
                        head_err[h_idx] = string("exception: ") + e.what();
            } catch (const char *s) {
                lock_guard<mutex> lk(result_mtx);
                fprintf(stderr, "[GPU %d] EXCEPTION (char*): %s\n", g, s ? s : "(null)");
                for (int h_idx : gpu_heads[g])
                    if (!head_done[h_idx])
                        head_err[h_idx] = "exception (char*)";
            } catch (...) {
                lock_guard<mutex> lk(result_mtx);
                fprintf(stderr, "[GPU %d] EXCEPTION (unknown)\n", g);
                for (int h_idx : gpu_heads[g])
                    if (!head_done[h_idx])
                        head_err[h_idx] = "exception (unknown)";
            }
        });
    }
    for (auto &t : threads) t.join();
    double compute_ms = compute_timer.elapsed_ms();
    double total_ms   = total_timer.elapsed_ms();

    // ═══ Verification ═══
    bool overall_pass = true;
    double mae_threshold = 1e-5;
    int  cmp_n = static_cast<int>(sparse_slots_val);
    if (skip_ref) {
        printf("\n[verify] --skip-ref set; skipping correctness check.\n");
    } else {
        printf("\n────────────── Per-head MAE (vs single-GPU reference) ──────────────\n");
        for (int h = 0; h < n_heads; h++) {
            if (!head_done[h]) {
                printf("  head %2d: MISSING (%s)\n", h,
                       head_err[h].empty() ? "no output" : head_err[h].c_str());
                overall_pass = false;
                continue;
            }
            const auto &ref = ref_outputs[h];
            const auto &got = head_outputs[h];
            int n = std::min({cmp_n,
                              static_cast<int>(ref.size()),
                              static_cast<int>(got.size())});
            if (n <= 0) {
                printf("  head %2d: empty decode (ref=%zu, got=%zu)\n",
                       h, ref.size(), got.size());
                overall_pass = false;
                continue;
            }
            double mae = 0.0;
            for (int i = 0; i < n; i++) mae += fabs(ref[i] - got[i]);
            mae /= n;
            const char *tag = (mae < mae_threshold) ? "PASS" : "FAIL";
            if (mae >= mae_threshold) overall_pass = false;
            printf("  head %2d: MAE=%.3e  (over %d slots)  %s\n",
                   h, mae, n, tag);
        }
    }

    // ═══ Headline ═══
    printf("\n════════════════════════════════════════════════════════════\n");
    printf("  HP-LLaMA result — %d GPUs / %d heads / %d layer%s / N=%d\n",
           n_gpus, n_heads, n_layers, n_layers == 1 ? "" : "s", ring_N);
    printf("════════════════════════════════════════════════════════════\n");
    printf("  Setup:   %8.1f ms\n", total_ms - compute_ms);
    printf("  Compute: %8.1f ms (concurrent across %d GPUs)\n",
           compute_ms, n_gpus);
    printf("  Total:   %8.1f ms = %.2f s\n", total_ms, total_ms / 1000.0);
    if (n_layers > 1) {
        printf("  Per-layer: %.1f ms = %.2f s (= compute / n_layers)\n",
               compute_ms / n_layers, compute_ms / n_layers / 1000.0);
    }

    int H = global_times.heads > 0 ? global_times.heads : 1;
    double sum = global_times.total();
    auto row = [&](const char *name, double v) {
        printf("  %-22s %9.1f ms   %7.1f ms/head   %5.1f%%\n",
               name, v, v / H, sum > 0 ? 100.0 * v / sum : 0.0);
    };
    printf("\n─── Per-operation timing (summed across %d head completions) ───\n", H);
    row("QKV MatMul",           global_times.qkv);
    row("RoPE (Q,K)",           global_times.rope);
    row("Q*K^T",                global_times.qk);
    row("Softmax",              global_times.softmax);
    row("Attn*V",               global_times.av);
    row("Out MatMul",           global_times.out);
    row("Bootstrap #1",         global_times.bs1);
    row("RMSNorm #1",           global_times.rms1);
    row("Bootstrap #2",         global_times.bs2);
    row("FFN gate MatMul",      global_times.ffn_gate);
    row("FFN up MatMul",        global_times.ffn_up);
    row("SiLU(gate)",           global_times.ffn_silu);
    row("gate ⊙ up (ct×ct)",    global_times.ffn_gate_mul_up);
    row("FFN down MatMul",      global_times.ffn_down);
    row("Bootstrap #3",         global_times.bs3);
    row("RMSNorm #2",           global_times.rms2);
    row("Bootstrap #4",         global_times.bs4);

    // Speedup vs Lane F H1 baseline (10.35 s/head sequential single-GPU)
    if (H > 0) {
        double per_head_compute = compute_ms / H * n_gpus;  // approx parallel
        double single_gpu_serial_proj = 10350.0 * n_heads * n_layers;
        printf("\n─── HP-LLaMA scaling (vs Lane F H1 single-GPU baseline) ───\n");
        printf("  H1 baseline (1 head, 1 layer, 1 GPU):        10.35 s\n");
        printf("  Single-GPU sequential (%d heads × %d layers): %.1f s\n",
               n_heads, n_layers, single_gpu_serial_proj / 1000.0);
        printf("  HP-LLaMA observed (%d heads, %d GPUs):       %.1f s\n",
               n_heads, n_gpus, compute_ms / 1000.0);
        if (compute_ms > 0) {
            printf("  Speedup vs single-GPU sequential:            %.2f×\n",
                   single_gpu_serial_proj / compute_ms);
        }
    }

    printf("\n══════════════════════════════════════════════\n");
    if (skip_ref) {
        printf("  HP-LLaMA verification SKIPPED (--skip-ref)\n");
    } else {
        printf("  HP-LLaMA verification: %s (threshold MAE < %.0e)\n",
               overall_pass ? "PASS" : "FAIL", mae_threshold);
    }
    printf("══════════════════════════════════════════════\n");
    fflush(stdout);

    return (skip_ref || overall_pass) ? 0 : 1;
}
