/**
 * llama_hp_multinode.cu  (Lane LLAMA-FULL)
 *
 * Multi-node Head-Parallel HP-LLaMA — analog of bert_hp_multinode.cu for
 * LLaMA layer compute. Scale-out HP-LLaMA to 4 nodes × 4 GPUs (16 GPUs).
 *
 * Models: --model {llama-7b, llama-3-8b}
 *   llama-7b   : ffn_total=11008, default seq_len=16
 *   llama-3-8b : ffn_total=14336, default seq_len=8 (NEXUS Table IV apples-to-apples)
 * --seq-len, --inner, --ffn-inner override per-model defaults individually.
 *
 * NCCL EDITION (no MPI). Uses a shared-filesystem bootstrap of ncclUniqueId
 * and SLURM env vars (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID, SLURM_NODEID)
 * for rank discovery. One process per GPU (launched by `srun --mpi=none`).
 *
 * Strategy: 1 process per GPU. World size = `nodes * gpus-per-node`. Each
 * rank owns N heads sequentially (default: 32 heads / 16 ranks = 2 heads
 * per rank for the 16-GPU multi-node measurement; 32 heads / 32 ranks = 1
 * head per rank if 32 ranks are launched). The single-GPU pinned-host key
 * streaming pipeline is reused per rank (same as bert_hp_multinode).
 *
 * Per-head compute mirrors `run_one_head_llama` in llama_hp_multigpu.cu:
 *   QKV → RoPE → QK^T → Softmax → Attn×V → OutProj → BS#1 → RMSNorm#1
 *   → BS#2 → SwiGLU FFN (gate, up, SiLU(gate), gate⊙up, down) → BS#3
 *   → RMSNorm#2 → BS#4
 *
 * No cross-rank communication during a layer (HP heads are independent);
 * only NCCL barriers at trial-start / trial-end and an allreduce(MAX/SUM)
 * for timing aggregation.
 *
 * Usage (on MN5):
 *   srun --nodes=4 --ntasks-per-node=4 --gres=gpu:4 --mpi=none \
 *     ./build/bin/llama_hp_multinode \
 *       --N 65536 --heads 32 --layers 32 --skip-ref
 *
 * CLI:
 *   --N {32768,65536}   CKKS ring degree (default 65536; matches HP-LLaMA)
 *   --heads N           LLaMA head count (default 32)
 *   --layers N          LLaMA layer count (default 32 = LLaMA-7B)
 *   --inner N           inner MatMul dim (default 128 = head_dim)
 *   --seq-len N         sequence length (default 16)
 *   --hidden N          hidden dim (default 128 = head_dim)
 *   --trials N          trial counter (only used by SLURM script for label)
 *   --skip-ref          (always implicitly true for multinode)
 *   --bootstrap-dir P   shared-FS dir for ncclUniqueId
 */

#include <cuda_runtime.h>
#include <nccl.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
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

#define NCCL_CHECK(cmd) do {                                                  \
    ncclResult_t _r = (cmd);                                                  \
    if (_r != ncclSuccess) {                                                  \
        fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__,      \
                ncclGetErrorString(_r));                                      \
        std::exit(1);                                                         \
    }                                                                         \
} while(0)

#define CUDA_CHECK(cmd) do {                                                  \
    cudaError_t _e = (cmd);                                                   \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(_e));                                      \
        std::exit(1);                                                         \
    }                                                                         \
} while(0)

// ─── SLURM env helpers ─────────────────────────────────────────────────────
static int env_int(const char *name, int dflt) {
    const char *v = std::getenv(name);
    if (!v || !*v) return dflt;
    return std::atoi(v);
}

// ─── ncclUniqueId bootstrap via shared filesystem ──────────────────────────
static std::string bootstrap_path(const std::string &dir) {
    const char *jobid = std::getenv("SLURM_JOB_ID");
    if (!jobid || !*jobid) jobid = "local";
    return dir + "/ncclid_" + std::string(jobid) + ".bin";
}

static void write_unique_id(const ncclUniqueId &id, const std::string &path) {
    std::string tmp = path + ".tmp";
    {
        std::ofstream of(tmp, std::ios::binary | std::ios::trunc);
        if (!of) {
            fprintf(stderr, "[bootstrap] cannot open %s for write\n", tmp.c_str());
            std::exit(1);
        }
        of.write(reinterpret_cast<const char*>(&id), sizeof(ncclUniqueId));
        of.flush();
    }
    int fd = ::open(tmp.c_str(), O_RDONLY);
    if (fd >= 0) { ::fsync(fd); ::close(fd); }
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        fprintf(stderr, "[bootstrap] rename %s -> %s failed: %s\n",
                tmp.c_str(), path.c_str(), std::strerror(errno));
        std::exit(1);
    }
}

static void read_unique_id(ncclUniqueId &id, const std::string &path,
                           int max_wait_sec = 60) {
    auto t0 = std::chrono::steady_clock::now();
    while (true) {
        struct stat st;
        if (::stat(path.c_str(), &st) == 0 &&
            st.st_size == (off_t)sizeof(ncclUniqueId)) {
            std::ifstream f(path, std::ios::binary);
            if (f) {
                f.read(reinterpret_cast<char*>(&id), sizeof(ncclUniqueId));
                if (f.gcount() == (std::streamsize)sizeof(ncclUniqueId)) return;
            }
        }
        auto dt = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - t0).count();
        if (dt > max_wait_sec) {
            fprintf(stderr, "[bootstrap] timed out waiting for %s\n", path.c_str());
            std::exit(1);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

// ─── NCCL collectives ──────────────────────────────────────────────────────
static void nccl_barrier(ncclComm_t comm, cudaStream_t s) {
    int *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
    int one = 1;
    CUDA_CHECK(cudaMemcpyAsync(d, &one, sizeof(int), cudaMemcpyHostToDevice, s));
    NCCL_CHECK(ncclAllReduce(d, d, 1, ncclInt, ncclSum, comm, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(d));
}

static void nccl_bcast_bytes(ncclComm_t comm, cudaStream_t s,
                             void *buf, size_t bytes, int root) {
    if (bytes == 0) return;
    void *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, bytes));
    int rank, _ws;
    NCCL_CHECK(ncclCommUserRank(comm, &rank));
    NCCL_CHECK(ncclCommCount(comm, &_ws));
    if (rank == root) {
        CUDA_CHECK(cudaMemcpyAsync(d, buf, bytes, cudaMemcpyHostToDevice, s));
    }
    NCCL_CHECK(ncclBroadcast(d, d, bytes, ncclChar, root, comm, s));
    if (rank != root) {
        CUDA_CHECK(cudaMemcpyAsync(buf, d, bytes, cudaMemcpyDeviceToHost, s));
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(d));
}

static void nccl_allreduce_doubles(ncclComm_t comm, cudaStream_t s,
                                   double *buf, size_t n,
                                   ncclRedOp_t op) {
    if (n == 0) return;
    double *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(d, buf, n * sizeof(double),
                               cudaMemcpyHostToDevice, s));
    NCCL_CHECK(ncclAllReduce(d, d, n, ncclDouble, op, comm, s));
    CUDA_CHECK(cudaMemcpyAsync(buf, d, n * sizeof(double),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(d));
}

static void nccl_allreduce_ints(ncclComm_t comm, cudaStream_t s,
                                int *buf, size_t n, ncclRedOp_t op) {
    if (n == 0) return;
    int *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(d, buf, n * sizeof(int),
                               cudaMemcpyHostToDevice, s));
    NCCL_CHECK(ncclAllReduce(d, d, n, ncclInt, op, comm, s));
    CUDA_CHECK(cudaMemcpyAsync(buf, d, n * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(d));
}

static void nccl_bcast_size(ncclComm_t comm, cudaStream_t s,
                            size_t *v, int root) {
    nccl_bcast_bytes(comm, s, v, sizeof(size_t), root);
}

// ─── Timer ─────────────────────────────────────────────────────────────────
struct PerfTimer {
    chrono::high_resolution_clock::time_point t0;
    void start() { t0 = chrono::high_resolution_clock::now(); }
    double elapsed_ms() const {
        return chrono::duration<double, milli>(
            chrono::high_resolution_clock::now() - t0).count();
    }
};

// ─── Per-head op-time totals (mirror llama_hp_multigpu) ────────────────────
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
};

#define TIME_OP(field, ...) do {                        \
    cudaDeviceSynchronize();                            \
    PerfTimer _pt; _pt.start();                         \
    { __VA_ARGS__; }                                    \
    cudaDeviceSynchronize();                            \
    times.field += _pt.elapsed_ms();                    \
} while(0)

// ─── One LLaMA decoder layer (one head) — mirrors llama_hp_multigpu ────────
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
    OpTimes &times)
{
    (void)ctx;
    GELUEvaluator    lg(le);   // SiLU proxy
    SoftmaxEvaluator ls(le);
    LNEvaluator      ll(le);   // RMSNorm proxy
    MMEvaluator      lm(le);

    // QKV projections
    vector<PhantomCiphertext> xi = {ct_in}, q, k, v;
    TIME_OP(qkv, {
        lm.matrix_mul_unified(xi, W_q, 1, q);
        lm.matrix_mul_unified(xi, W_k, 1, k);
        lm.matrix_mul_unified(xi, W_v, 1, v);
    });

    // RoPE on Q and K (rotations FIRST workaround for chain-index segfault)
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

    PhantomCiphertext qk_ct;
    TIME_OP(qk, {
        le.evaluator.mod_switch_to_inplace(k[0], q[0].chain_index());
        k[0].set_scale(q[0].scale());
        le.evaluator.multiply(q[0], k[0], qk_ct);
        le.evaluator.relinearize_inplace(qk_ct, *le.relin_keys);
        le.evaluator.rescale_to_next_inplace(qk_ct);
    });

    PhantomCiphertext attn;
    TIME_OP(softmax, { ls.softmax(qk_ct, attn, seq_len); });

    PhantomCiphertext av_ct;
    TIME_OP(av, {
        le.evaluator.mod_switch_to_inplace(v[0], attn.chain_index());
        v[0].set_scale(attn.scale());
        le.evaluator.multiply(attn, v[0], av_ct);
        le.evaluator.relinearize_inplace(av_ct, *le.relin_keys);
        le.evaluator.rescale_to_next_inplace(av_ct);
    });

    vector<PhantomCiphertext> ai = {av_ct}, ao;
    TIME_OP(out, { lm.matrix_mul_unified(ai, W_o, 1, ao); });

    PhantomCiphertext b1;
    TIME_OP(bs1, {
        while (ao[0].coeff_modulus_size() > 1)
            le.evaluator.mod_switch_to_next_inplace(ao[0]);
        lb.bootstrap_3(b1, ao[0]);
    });

    PhantomCiphertext rms1_out;
    TIME_OP(rms1, { ll.layer_norm(b1, rms1_out, hidden); });

    PhantomCiphertext b2;
    TIME_OP(bs2, {
        while (rms1_out.coeff_modulus_size() > 1)
            le.evaluator.mod_switch_to_next_inplace(rms1_out);
        lb.bootstrap_3(b2, rms1_out);
    });

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

    PhantomCiphertext b3;
    TIME_OP(bs3, {
        while (down_out[0].coeff_modulus_size() > 1)
            le.evaluator.mod_switch_to_next_inplace(down_out[0]);
        lb.bootstrap_3(b3, down_out[0]);
    });

    PhantomCiphertext rms2_out;
    TIME_OP(rms2, { ll.layer_norm(b3, rms2_out, hidden); });

    TIME_OP(bs4, {
        while (rms2_out.coeff_modulus_size() > 1)
            le.evaluator.mod_switch_to_next_inplace(rms2_out);
        lb.bootstrap_3(ct_out, rms2_out);
    });

    times.heads++;
}

// ─── Per-rank setup ────────────────────────────────────────────────────────
static void setup_per_rank(
    int local_gpu,
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
    unique_ptr<GaloisKeyStore>     &ks_out,
    int rank)
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

    printf("[setup r=%d gpu=%d] context+keys ready (%zu rotation keys)\n",
           rank, local_gpu, gelts.size());
    fflush(stdout);
}

// ─── main ──────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    int rank = env_int("SLURM_PROCID", 0);
    int world_size = env_int("SLURM_NTASKS", 1);
    int local_rank = env_int("SLURM_LOCALID", rank);
    int node_id = env_int("SLURM_NODEID", 0);
    (void)node_id;

    int n_heads  = 32;       // LLaMA
    int n_layers = 32;       // LLaMA
    int inner    = 128;      // attention inner = head_dim per head
    int ffn_inner = -1;      // FFN inner per head; if <0 derived from --model
    int seq_len  = -1;       // sequence length; if <0 derived from --model
    int hidden   = 128;
    int n_trials = 1;
    bool skip_ref = true;
    int  ring_N  = 65536;    // LLaMA default — uniform logN=16
    string bootstrap_dir = "/gpfs/projects/etur02/hkanpak/scratch";
    string model = "llama-7b";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--heads") && i+1 < argc)
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
        else if (!strcmp(argv[i], "--bootstrap-dir") && i+1 < argc)
            bootstrap_dir = argv[++i];
        else if (!strcmp(argv[i], "--model") && i+1 < argc)
            model = argv[++i];
    }
    if (ring_N != 32768 && ring_N != 65536) {
        if (rank == 0)
            fprintf(stderr, "Unsupported --N %d (use 32768 or 65536)\n", ring_N);
        return 1;
    }
    int model_seq_len, model_ffn_inner_total;
    if (model == "llama-3-8b") {
        model_seq_len = 8;
        model_ffn_inner_total = 14336;
    } else if (model == "llama-7b") {
        model_seq_len = 16;
        model_ffn_inner_total = 11008;
    } else {
        if (rank == 0)
            fprintf(stderr, "Unsupported --model %s (use llama-7b or llama-3-8b)\n",
                    model.c_str());
        return 1;
    }
    if (seq_len < 0) seq_len = model_seq_len;
    if (ffn_inner < 0)
        ffn_inner = (model_ffn_inner_total + n_heads - 1) / n_heads;
    (void)n_trials;

    // ─── Pin rank to local GPU ─────────────────────────────────────────────
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    int local_gpu = (dev_count > 0) ? (local_rank % dev_count) : 0;
    CUDA_CHECK(cudaSetDevice(local_gpu));

    // ─── Bootstrap NCCL ────────────────────────────────────────────────────
    ncclUniqueId nccl_id;
    std::string id_path = bootstrap_path(bootstrap_dir);
    if (rank == 0) {
        ::mkdir(bootstrap_dir.c_str(), 0755);
        ::unlink(id_path.c_str());
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        write_unique_id(nccl_id, id_path);
        printf("[bootstrap r=0] wrote ncclUniqueId to %s\n", id_path.c_str());
        fflush(stdout);
    } else {
        read_unique_id(nccl_id, id_path);
    }

    ncclComm_t world_comm = nullptr;
    cudaStream_t world_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&world_stream, cudaStreamNonBlocking));
    NCCL_CHECK(ncclCommInitRank(&world_comm, world_size, nccl_id, rank));

    if (rank == 0) {
        ::unlink(id_path.c_str());
    }

    // ─── Distribute heads across ranks (round-robin) ───────────────────────
    // If world_size >= n_heads: ranks 0..n_heads-1 each own 1 head.
    // If world_size < n_heads: each rank owns ceil(n_heads/world_size) heads
    // sequentially (e.g. 32 heads / 16 ranks = 2 heads/rank).
    vector<int> my_heads;
    for (int h = rank; h < n_heads; h += world_size)
        my_heads.push_back(h);
    bool is_active = !my_heads.empty();

    if (rank == 0) {
        printf("════════════════════════════════════════════════════════════\n");
        printf("  HP-LLaMA MULTINODE (NCCL) — model=%s, N=%d, %d ranks, %d heads, %d layer%s\n",
               model.c_str(), ring_N, world_size, n_heads, n_layers, n_layers == 1 ? "" : "s");
        printf("  Heads per rank: %d (round-robin distribution)\n",
               (n_heads + world_size - 1) / world_size);
        printf("  Per-rank: single-GPU pinned-host rotation\n");
        printf("  hidden=%d, attn_inner=%d, ffn_inner=%d, seq=%d\n",
               hidden, inner, ffn_inner, seq_len);
        printf("  ffn_total≈%d (per-head ffn_inner=%d × %d heads)\n",
               ffn_inner * n_heads, ffn_inner, n_heads);
        printf("════════════════════════════════════════════════════════════\n");
        fflush(stdout);
    }
    nccl_barrier(world_comm, world_stream);

    // ─── CKKS params (mirror llama_hp_multigpu / bert_hp_multinode) ────────
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

    // ─── Generate SK on rank 0, broadcast to all ranks ─────────────────────
    string sk_str;
    {
        stringstream sk_buf;
        if (rank == 0) {
            PhantomContext   ctx_tmp(parms);
            PhantomSecretKey sk_tmp(ctx_tmp);
            sk_tmp.save(sk_buf);
            sk_str = sk_buf.str();
            printf("[rank 0] generated and serialized secret key "
                   "(%zu bytes)\n", sk_str.size());
            fflush(stdout);
        }
        size_t sk_size = sk_str.size();
        nccl_bcast_size(world_comm, world_stream, &sk_size, /*root=*/0);
        if (rank != 0) sk_str.resize(sk_size);
        nccl_bcast_bytes(world_comm, world_stream,
                         const_cast<char*>(sk_str.data()), sk_size, /*root=*/0);
        nccl_barrier(world_comm, world_stream);
    }

    // ─── Per-rank weights ──────────────────────────────────────────────────
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.02, 0.02), idist(-0.5, 0.5);
    size_t slots = (size_t)sparse_slots_val;

    auto make_w = [&](int rows) {
        vector<vector<double>> w(rows, vector<double>(slots, 0.0));
        for (auto &r : w)
            for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++)
                r[s] = wdist(rng);
        return w;
    };

    vector<vector<double>> W_q, W_k, W_v, W_o, W_gate, W_up, W_down;
    vector<double> rope_cos(slots, 0.0), rope_sin(slots, 0.0);
    vector<vector<double>> head_inputs;

    if (is_active) {
        W_q = make_w(inner); W_k = make_w(inner); W_v = make_w(inner); W_o = make_w(inner);
        W_gate = make_w(ffn_inner); W_up = make_w(ffn_inner); W_down = make_w(ffn_inner);

        for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++) {
            rope_cos[s] = cos(0.01 * s);
            rope_sin[s] = sin(0.01 * s);
        }

        // Generate per-head inputs deterministically; only keep the heads
        // this rank owns (must consume RNG identically on every rank).
        head_inputs.assign(n_heads, vector<double>(slots, 0.0));
        for (int h = 0; h < n_heads; h++)
            for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++)
                head_inputs[h][s] = idist(rng);
    }

    // ─── Per-rank context setup ────────────────────────────────────────────
    unique_ptr<PhantomContext>     ctx;
    unique_ptr<PhantomCKKSEncoder> enc;
    unique_ptr<PhantomSecretKey>   sk;
    unique_ptr<PhantomPublicKey>   pk;
    unique_ptr<PhantomRelinKey>    rk;
    unique_ptr<PhantomGaloisKey>   gk;
    unique_ptr<CKKSEvaluator>      eval;
    unique_ptr<Bootstrapper>       bs;
    unique_ptr<GaloisKeyStore>     ks;

    PerfTimer setup_timer;
    setup_timer.start();

    setup_per_rank(
        local_gpu, parms, logN, logn, logNh, total_level, SCALE,
        boundary_K, deg, scale_factor, inverse_deg, loge,
        seq_len, hidden, sk_str,
        ctx, enc, sk, pk, rk, gk, eval, bs, ks, rank);

    double setup_ms_local = setup_timer.elapsed_ms();

    // ─── Encrypt this rank's input(s) ──────────────────────────────────────
    vector<PhantomCiphertext> ct_in_per_head;
    if (is_active) {
        ct_in_per_head.resize(my_heads.size());
        for (size_t i = 0; i < my_heads.size(); i++) {
            PhantomPlaintext pt;
            eval->encoder.encode(head_inputs[my_heads[i]], SCALE, pt);
            eval->encryptor.encrypt(pt, ct_in_per_head[i]);
            for (int j = 0; j < bs_mod; j++)
                eval->evaluator.mod_switch_to_next_inplace(ct_in_per_head[i]);
        }
    }

    nccl_barrier(world_comm, world_stream);

    if (rank == 0) {
        printf("\n═══ All ranks setup; starting compute ═══\n");
        fflush(stdout);
    }

    // ─── Compute (LLaMA layer chain, per head sequentially) ────────────────
    OpTimes times;
    PerfTimer compute_timer;
    compute_timer.start();
    double compute_ms_local = 0;

    if (is_active) {
        for (size_t i = 0; i < my_heads.size(); i++) {
            int h_idx = my_heads[i];
            PhantomCiphertext ct = ct_in_per_head[i];
            PhantomCiphertext ct_out;
            for (int layer = 0; layer < n_layers; ++layer) {
                ct_out = PhantomCiphertext();
                run_one_head_llama(*ctx, *eval, *bs, ct, ct_out,
                                   W_q, W_k, W_v, W_o, W_gate, W_up, W_down,
                                   rope_cos, rope_sin, hidden, seq_len, times);
                ct = std::move(ct_out);
                if (n_layers > 1) {
                    printf("[rank %d head %d] layer %d/%d done (out level=%zu)\n",
                           rank, h_idx, layer + 1, n_layers,
                           ct.coeff_modulus_size());
                    fflush(stdout);
                }
            }
            cudaDeviceSynchronize();
            printf("[rank %d head %d] COMPLETE after %d layer(s)\n",
                   rank, h_idx, n_layers);
            fflush(stdout);
        }
    }
    compute_ms_local = compute_timer.elapsed_ms();
    if (!is_active) compute_ms_local = 0.0;

    nccl_barrier(world_comm, world_stream);

    // ─── Aggregate timings via NCCL allreduce ──────────────────────────────
    double maxbuf[2] = { compute_ms_local, setup_ms_local };
    nccl_allreduce_doubles(world_comm, world_stream, maxbuf, 2, ncclMax);
    double compute_ms_max = maxbuf[0];
    double setup_ms_max   = maxbuf[1];

    // 17 op fields packed as SUM
    double sumbuf[17] = {
        times.qkv, times.rope, times.qk, times.softmax, times.av, times.out,
        times.bs1, times.rms1, times.bs2,
        times.ffn_gate, times.ffn_up, times.ffn_silu, times.ffn_gate_mul_up, times.ffn_down,
        times.bs3, times.rms2, times.bs4
    };
    nccl_allreduce_doubles(world_comm, world_stream, sumbuf, 17, ncclSum);

    int hbuf[1] = { times.heads };
    nccl_allreduce_ints(world_comm, world_stream, hbuf, 1, ncclSum);
    int sum_h = hbuf[0];

    if (rank == 0) {
        // Total bootstrap instances = n_heads × n_layers × 4 bootstraps
        int n_bootstrap_inst = n_heads * n_layers * 4;
        double bootstrap_total_ms = sumbuf[6] + sumbuf[8] + sumbuf[14] + sumbuf[16];
        double per_bootstrap_ms = (n_bootstrap_inst > 0)
            ? bootstrap_total_ms / n_bootstrap_inst
            : 0.0;

        double sum_all = 0;
        for (int k = 0; k < 17; k++) sum_all += sumbuf[k];

        printf("\n════════════════════════════════════════════════════════════\n");
        printf("  HP-LLaMA MULTINODE (NCCL) result — %d ranks / %d heads / "
               "%d layer%s / N=%d\n",
               world_size, n_heads, n_layers, n_layers == 1 ? "" : "s", ring_N);
        printf("════════════════════════════════════════════════════════════\n");
        printf("  Setup (max across ranks):   %8.1f ms = %.2f s\n",
               setup_ms_max, setup_ms_max / 1000.0);
        printf("  Compute (max across ranks): %8.1f ms = %.2f s\n",
               compute_ms_max, compute_ms_max / 1000.0);
        if (n_layers > 0) {
            printf("  Per-layer (compute / n_layers): %.1f ms = %.2f s\n",
                   compute_ms_max / n_layers, compute_ms_max / n_layers / 1000.0);
        }
        printf("  Per-bootstrap (avg of %d instances): %.1f ms\n",
               n_bootstrap_inst, per_bootstrap_ms);

        printf("\n─── Per-operation totals (summed across %d head completions) ───\n", sum_h);
        auto row = [&](const char *name, double v) {
            printf("  %-22s %9.1f ms   %7.1f ms/head   %5.1f%%\n",
                   name, v, v / std::max(1, sum_h),
                   sum_all > 0 ? 100.0 * v / sum_all : 0.0);
        };
        row("QKV MatMul",        sumbuf[0]);
        row("RoPE (Q,K)",        sumbuf[1]);
        row("Q*K^T",             sumbuf[2]);
        row("Softmax",           sumbuf[3]);
        row("Attn*V",            sumbuf[4]);
        row("Out MatMul",        sumbuf[5]);
        row("Bootstrap #1",      sumbuf[6]);
        row("RMSNorm #1",        sumbuf[7]);
        row("Bootstrap #2",      sumbuf[8]);
        row("FFN gate MatMul",   sumbuf[9]);
        row("FFN up MatMul",     sumbuf[10]);
        row("SiLU(gate)",        sumbuf[11]);
        row("gate * up",         sumbuf[12]);
        row("FFN down MatMul",   sumbuf[13]);
        row("Bootstrap #3",      sumbuf[14]);
        row("RMSNorm #2",        sumbuf[15]);
        row("Bootstrap #4",      sumbuf[16]);

        printf("\n══════════════════════════════════════════════\n");
        if (skip_ref) {
            printf("  HP-LLaMA verification SKIPPED (--skip-ref)\n");
        } else {
            printf("  HP-LLaMA verification not implemented in multinode binary "
                   "(use llama_hp_multigpu --heads N for verification)\n");
        }
        printf("══════════════════════════════════════════════\n");
        fflush(stdout);
    }

    nccl_barrier(world_comm, world_stream);

    NCCL_CHECK(ncclCommDestroy(world_comm));
    cudaStreamDestroy(world_stream);

    return 0;
}
