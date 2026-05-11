/**
 * bert_hp_multinode.cu
 *
 * Multi-node Head-Parallel BERT — scale-out HP-BERT to 4 nodes × 4 GPUs (16 GPUs).
 *
 * NCCL EDITION (no MPI). Uses a shared-filesystem bootstrap of ncclUniqueId
 * and SLURM env vars (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID, SLURM_NODEID)
 * for rank discovery. One process per GPU (launched by `srun --mpi=none`).
 *
 * Strategy (multi-node analogue of bert_hp_multigpu.cu):
 *   - 1 process per GPU. World size = `nodes * gpus-per-node`. We launch with
 *     16 ranks and run BERT-base (12 heads × 12 layers); the first 12 ranks run
 *     1 head each in parallel, the remaining 4 ranks are idle (head_id >= 12).
 *   - Each rank uses the same single-GPU pinned-host key streaming pipeline
 *     used by `bert_hp_multigpu.cu` (no DKS, no T-MODUP dependence). All
 *     bootstrap rotation keys live in CPU pinned memory and stream into GPU
 *     on demand (the proven Phase 1 path).
 *   - Heads in HP-BERT are computationally independent — there is no
 *     cross-head communication during a layer. The only multi-node sync is
 *     a small ncclAllReduce barrier at trial start (so the parallel-time
 *     measurement starts after every rank has finished setup) and
 *     ncclAllReduce(MAX/SUM) at trial end (so the wall-clock parallel time
 *     is `max over ranks` of per-rank compute time).
 *   - Per-bootstrap timing (the headline) is averaged across all
 *     12 head × n_layers × 4 bootstraps = 576 bootstrap instances.
 *
 * Why NCCL (not MPI):
 *   - GPU-to-GPU communication should be GPU-native. MPI requires a CPU-side
 *     copy (or CUDA-aware MPI which still bottlenecks on host orchestration).
 *   - We already use NCCL for all single-node multi-GPU paths
 *     (src/multi_gpu/comm/nccl_comm.{cu,cuh}). One transport stack to tune.
 *   - The "MPI is needed for multi-node bootstrap" claim is folklore — NCCL
 *     can bootstrap from any out-of-band channel. We use a small file on
 *     the shared GPFS at /gpfs/projects/etur02/hkanpak/scratch/ncclid_<JOBID>.
 *
 * Why 16 ranks for 12 heads:
 *   - The 4× H100 ACC node × 4 nodes ceiling = 16 GPUs. BERT-base has 12 heads.
 *     We choose to fully populate the allocation (16 ranks) rather than
 *     under-subscribe, so SLURM scheduling matches the standard 4-node
 *     allocation pattern. The 4 idle ranks add ~0% overhead.
 *
 * Usage (on MN5):
 *   srun --nodes=4 --ntasks-per-node=4 --gres=gpu:4 --mpi=none \
 *     ./build/bin/bert_hp_multinode \
 *       --N 32768 --heads 12 --layers 12 --skip-ref
 *
 * CLI flags (subset of bert_hp_multigpu.cu — multinode binary always treats
 * --n-gpus as `1`, since each rank owns exactly 1 GPU):
 *   --N {32768,65536}         CKKS ring degree (default 32768)
 *   --heads N                 BERT head count (default 12)
 *   --layers N                BERT layer count (default 12)
 *   --inner N                 inner MatMul dim (default 64)
 *   --seq-len N               sequence length (default 16)
 *   --hidden N                hidden dim (default 64)
 *   --trials N                trial count for median-of-N (default 1; use 3 in SLURM)
 *   --skip-ref                skip the single-GPU reference pass (recommended)
 *   --bootstrap-dir PATH      directory on shared FS for ncclUniqueId file
 *                             (default: /gpfs/projects/etur02/hkanpak/scratch)
 *
 * Trial loop is in the SLURM script (3 sequential runs) for fresh CUDA state
 * per trial, mirroring the single-node `slurm_bert_hp_n32768.sh`.
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

// ---------------------------------------------------------------------------
// SLURM env helpers — single source of truth for rank/size discovery.
// ---------------------------------------------------------------------------
static int env_int(const char *name, int dflt) {
    const char *v = std::getenv(name);
    if (!v || !*v) return dflt;
    return std::atoi(v);
}

// ---------------------------------------------------------------------------
// ncclUniqueId bootstrap via shared filesystem.
//
//   Rank 0 generates the unique id, writes it atomically to
//     <bootstrap_dir>/ncclid_<jobid>.bin
//   All other ranks poll for the file, then read 128 bytes.
//
// Atomic write: write to <name>.tmp, fsync, rename. POSIX rename is atomic
// on the same filesystem — guarantees readers never see a partial id.
// ---------------------------------------------------------------------------
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
    // sync to disk so other nodes see the bytes (GPFS semantics)
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

// ---------------------------------------------------------------------------
// Tiny NCCL-based collectives over scalar/byte buffers (replaces MPI_*).
// All collectives operate on world-comm; transient device buffers are
// allocated/freed inside (cheap relative to bootstrap latency).
// ---------------------------------------------------------------------------
static void nccl_barrier(ncclComm_t comm, cudaStream_t s) {
    // Smallest meaningful collective: 1-int allreduce. NCCL has no native
    // barrier primitive; this is the canonical idiom.
    int *d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
    int one = 1;
    CUDA_CHECK(cudaMemcpyAsync(d, &one, sizeof(int), cudaMemcpyHostToDevice, s));
    NCCL_CHECK(ncclAllReduce(d, d, 1, ncclInt, ncclSum, comm, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaFree(d));
}

// Broadcast `bytes` from root to all ranks. `buf` is host memory.
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

// AllReduce a vector of doubles in place (host side). Returns the reduced
// vector on every rank — caller can pick rank 0's view if only root needs it.
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

// AllReduce a vector of ints in place (host side).
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

// Broadcast a single size_t (number of bytes follow-up payload).
static void nccl_bcast_size(ncclComm_t comm, cudaStream_t s,
                            size_t *v, int root) {
    nccl_bcast_bytes(comm, s, v, sizeof(size_t), root);
}

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

// ---------------------------------------------------------------------------
// Per-head op-time totals (one set per rank).
// ---------------------------------------------------------------------------
struct OpTimes {
    double qkv_matmul = 0, qk_matmul = 0, softmax_op = 0, av_matmul = 0,
           out_matmul = 0, bs1 = 0, ln1 = 0, bs2 = 0,
           ffn1 = 0, gelu_op = 0, ffn2 = 0, bs3 = 0, ln2 = 0, bs4 = 0;
    int heads = 0;
    double total() const {
        return qkv_matmul + qk_matmul + softmax_op + av_matmul + out_matmul +
               bs1 + ln1 + bs2 + ffn1 + gelu_op + ffn2 + bs3 + ln2 + bs4;
    }
};

#define TIME_OP(field, code) do {                       \
    cudaDeviceSynchronize();                            \
    PerfTimer _pt; _pt.start();                         \
    code;                                               \
    cudaDeviceSynchronize();                            \
    times.field += _pt.elapsed_ms();                    \
} while(0)

// ---------------------------------------------------------------------------
// Run one full BERT encoder layer (one head). Mirrors run_one_head in
// bert_hp_multigpu.cu.
// ---------------------------------------------------------------------------
static void run_one_head(
    PhantomContext   &ctx,
    CKKSEvaluator    &le,
    Bootstrapper     &lb,
    PhantomCiphertext &ct_in,
    PhantomCiphertext &ct_out,
    vector<vector<double>> &W_q,
    vector<vector<double>> &W_k,
    vector<vector<double>> &W_v,
    vector<vector<double>> &W_o,
    vector<vector<double>> &W_f1,
    vector<vector<double>> &W_f2,
    int hidden,
    int seq_len,
    OpTimes &times)
{
    (void)ctx;

    GELUEvaluator    lg(le);
    SoftmaxEvaluator ls(le);
    LNEvaluator      ll(le);
    MMEvaluator      lm(le);

    vector<PhantomCiphertext> xi = {ct_in}, q, k, v;
    TIME_OP(qkv_matmul, {
        lm.matrix_mul_unified(xi, W_q, 1, q);
        lm.matrix_mul_unified(xi, W_k, 1, k);
        lm.matrix_mul_unified(xi, W_v, 1, v);
    });

    PhantomCiphertext as;
    TIME_OP(qk_matmul, {
        le.evaluator.mod_switch_to_inplace(k[0], q[0].chain_index());
        k[0].set_scale(q[0].scale());
        le.evaluator.multiply(q[0], k[0], as);
        le.evaluator.relinearize_inplace(as, *le.relin_keys);
        le.evaluator.rescale_to_next_inplace(as);
    });

    PhantomCiphertext aw;
    TIME_OP(softmax_op, { ls.softmax(as, aw, seq_len); });

    PhantomCiphertext ao;
    TIME_OP(av_matmul, {
        le.evaluator.mod_switch_to_inplace(v[0], aw.chain_index());
        v[0].set_scale(aw.scale());
        le.evaluator.multiply(aw, v[0], ao);
        le.evaluator.relinearize_inplace(ao, *le.relin_keys);
        le.evaluator.rescale_to_next_inplace(ao);
    });

    vector<PhantomCiphertext> pi = {ao}, po;
    TIME_OP(out_matmul, { lm.matrix_mul_unified(pi, W_o, 1, po); });

    PhantomCiphertext b1;
    TIME_OP(bs1, {
        while (po[0].coeff_modulus_size() > 1)
            le.evaluator.mod_switch_to_next_inplace(po[0]);
        // FIX-BUG-04-02 (SCALE-CROSS-CUT): reset to canonical SCALE before
        // bootstrap to prevent silent scale drift across chained layers.
        // Mirrors argmax_align_n32k.cu:225. Required because our Phantom has
        // scale-mismatch checks commented out (CLAUDE.md lesson #7).
        po[0].scale() = le.scale;
        lb.bootstrap_3(b1, po[0]);
    });

    PhantomCiphertext ln1o;
    TIME_OP(ln1, { ll.layer_norm(b1, ln1o, hidden); });

    PhantomCiphertext b2;
    TIME_OP(bs2, {
        while (ln1o.coeff_modulus_size() > 1)
            le.evaluator.mod_switch_to_next_inplace(ln1o);
        ln1o.scale() = le.scale;  // FIX-BUG-04-02 (SCALE-CROSS-CUT)
        lb.bootstrap_3(b2, ln1o);
    });

    vector<PhantomCiphertext> fi = {b2}, fo;
    TIME_OP(ffn1, { lm.matrix_mul_unified(fi, W_f1, 1, fo); });

    PhantomCiphertext go;
    TIME_OP(gelu_op, { lg.gelu(fo[0], go); });

    vector<PhantomCiphertext> f2i = {go}, f2o;
    TIME_OP(ffn2, { lm.matrix_mul_unified(f2i, W_f2, 1, f2o); });

    PhantomCiphertext b3;
    TIME_OP(bs3, {
        while (f2o[0].coeff_modulus_size() > 1)
            le.evaluator.mod_switch_to_next_inplace(f2o[0]);
        f2o[0].scale() = le.scale;  // FIX-BUG-04-02 (SCALE-CROSS-CUT)
        lb.bootstrap_3(b3, f2o[0]);
    });

    PhantomCiphertext ln2o;
    TIME_OP(ln2, { ll.layer_norm(b3, ln2o, hidden); });

    TIME_OP(bs4, {
        while (ln2o.coeff_modulus_size() > 1)
            le.evaluator.mod_switch_to_next_inplace(ln2o);
        ln2o.scale() = le.scale;  // FIX-BUG-04-02 (SCALE-CROSS-CUT)
        lb.bootstrap_3(ct_out, ln2o);
    });

    times.heads++;
}

// ---------------------------------------------------------------------------
// Setup helper — one PhantomContext + bootstrap polys + Galois key store on
// the local GPU. Identical to bert_hp_multigpu.cu's setup_per_gpu, except
// each rank only sets up for its own (single) GPU.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// main — multinode HP-BERT (NCCL edition)
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    // ─── Rank discovery from SLURM env (no MPI_Init) ───────────────────────
    int rank = env_int("SLURM_PROCID", 0);
    int world_size = env_int("SLURM_NTASKS", 1);
    int local_rank = env_int("SLURM_LOCALID", rank);
    int node_id = env_int("SLURM_NODEID", 0);
    (void)node_id;

    int n_heads  = 12;
    int n_layers = 12;
    int inner    = 64;
    int seq_len  = 16;
    int hidden   = 64;
    int n_trials = 1;
    bool skip_ref = true;       // multinode: never run reference
    int  ring_N  = 32768;       // multinode default = NEXUS-comparable N
    string bootstrap_dir = "/gpfs/projects/etur02/hkanpak/scratch";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--heads") && i+1 < argc)
            n_heads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--inner") && i+1 < argc)
            inner = atoi(argv[++i]);
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
    }
    if (ring_N != 32768 && ring_N != 65536) {
        if (rank == 0)
            fprintf(stderr, "Unsupported --N %d (use 32768 or 65536)\n", ring_N);
        return 1;
    }

    // ─── Each rank pinned to its local GPU via SLURM_LOCALID ───────────────
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    int local_gpu = (dev_count > 0) ? (local_rank % dev_count) : 0;
    CUDA_CHECK(cudaSetDevice(local_gpu));

    // ─── Bootstrap NCCL via shared-FS uniqueId ─────────────────────────────
    ncclUniqueId nccl_id;
    std::string id_path = bootstrap_path(bootstrap_dir);
    if (rank == 0) {
        // Ensure the directory exists; mkdir() returns -1 with EEXIST if it
        // already does — that's fine.
        ::mkdir(bootstrap_dir.c_str(), 0755);
        // Best-effort cleanup of stale id from an earlier abort.
        ::unlink(id_path.c_str());
        NCCL_CHECK(ncclGetUniqueId(&nccl_id));
        write_unique_id(nccl_id, id_path);
        printf("[bootstrap r=0] wrote ncclUniqueId to %s\n", id_path.c_str());
        fflush(stdout);
    } else {
        read_unique_id(nccl_id, id_path);
    }

    // Initialize the world communicator (one comm per process).
    ncclComm_t world_comm = nullptr;
    cudaStream_t world_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&world_stream, cudaStreamNonBlocking));
    NCCL_CHECK(ncclCommInitRank(&world_comm, world_size, nccl_id, rank));

    if (rank == 0) {
        // Now that all peers have entered ncclCommInitRank, the id file is
        // safe to delete (so it won't be picked up by a future job).
        ::unlink(id_path.c_str());
    }

    // Each rank gets one head_id by rank order. Rank r owns head r if r < n_heads.
    int head_id = rank;
    bool is_active = (rank < n_heads);

    if (rank == 0) {
        printf("════════════════════════════════════════════════════════════\n");
        printf("  HP-BERT MULTINODE (NCCL) — N=%d, %d ranks, %d heads, %d layer%s\n",
               ring_N, world_size, n_heads, n_layers, n_layers == 1 ? "" : "s");
        printf("  Each rank = 1 GPU = 1 head (1 head per rank)\n");
        printf("  Active ranks: %d  /  Idle ranks: %d\n",
               n_heads, world_size - n_heads);
        printf("  Per-rank: single-GPU pinned-host rotation\n");
        printf("  hidden=%d, inner=%d, seq=%d\n", hidden, inner, seq_len);
        printf("════════════════════════════════════════════════════════════\n");
        fflush(stdout);
    }
    nccl_barrier(world_comm, world_stream);

    // ═══ CKKS params: matches bert_hp_multigpu.cu ═══
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

    // ═══ Generate the secret key on rank 0, broadcast to all ranks via NCCL ═══
    string sk_str;
    {
        stringstream sk_buf;
        if (rank == 0) {
            // Use a temporary GPU 0 context just to generate the SK.
            PhantomContext   ctx_tmp(parms);
            PhantomSecretKey sk_tmp(ctx_tmp);
            sk_tmp.save(sk_buf);
            sk_str = sk_buf.str();
            printf("[rank 0] generated and serialized secret key "
                   "(%zu bytes)\n", sk_str.size());
            fflush(stdout);
        }
        // 1) Broadcast SK byte count.
        size_t sk_size = sk_str.size();
        nccl_bcast_size(world_comm, world_stream, &sk_size, /*root=*/0);
        // 2) Broadcast SK payload.
        if (rank != 0) sk_str.resize(sk_size);
        nccl_bcast_bytes(world_comm, world_stream,
                         const_cast<char*>(sk_str.data()), sk_size, /*root=*/0);
        nccl_barrier(world_comm, world_stream);
    }

    // Per-rank weights — deterministic, identical across ranks via the same
    // RNG seed (`42`). Mirrors bert_hp_multigpu.cu's per-head seed approach.
    mt19937 rng(42);
    uniform_real_distribution<double> wdist(-0.02, 0.02), idist(-0.5, 0.5);

    size_t slots = (size_t)sparse_slots_val;

    auto make_w = [&]() {
        vector<vector<double>> w(inner, vector<double>(slots, 0.0));
        for (auto &r : w)
            for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++)
                r[s] = wdist(rng);
        return w;
    };

    vector<vector<double>> W_q, W_k, W_v, W_o, W_f1, W_f2;
    vector<double> my_head_input(slots, 0.0);

    if (is_active) {
        W_q = make_w(); W_k = make_w(); W_v = make_w();
        W_o = make_w(); W_f1 = make_w(); W_f2 = make_w();

        for (int h = 0; h < n_heads; h++) {
            vector<double> hi(slots, 0.0);
            for (size_t s = 0; s < (size_t)std::min((long)hidden, (long)slots); s++)
                hi[s] = idist(rng);
            if (h == head_id) my_head_input = std::move(hi);
        }
    }

    // ═══ Per-rank setup ═══
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

    // ═══ Encrypt this rank's input ═══
    PhantomCiphertext ct_in;
    if (is_active) {
        PhantomPlaintext pt;
        eval->encoder.encode(my_head_input, SCALE, pt);
        eval->encryptor.encrypt(pt, ct_in);
        for (int j = 0; j < bs_mod; j++)
            eval->evaluator.mod_switch_to_next_inplace(ct_in);
    }

    nccl_barrier(world_comm, world_stream);

    if (rank == 0) {
        printf("\n═══ All ranks setup; starting compute ═══\n");
        fflush(stdout);
    }

    // ═══ Compute ═══
    OpTimes times;
    PerfTimer compute_timer;
    compute_timer.start();
    double compute_ms_local = 0;

    if (is_active) {
        PhantomCiphertext ct = ct_in;
        PhantomCiphertext ct_out;
        for (int layer = 0; layer < n_layers; ++layer) {
            ct_out = PhantomCiphertext();
            run_one_head(*ctx, *eval, *bs, ct, ct_out,
                         W_q, W_k, W_v, W_o, W_f1, W_f2,
                         hidden, seq_len, times);
            ct = std::move(ct_out);
            if (n_layers > 1) {
                printf("[rank %d head %d] layer %d/%d done (out level=%zu)\n",
                       rank, head_id, layer + 1, n_layers,
                       ct.coeff_modulus_size());
                fflush(stdout);
            }
        }
        cudaDeviceSynchronize();
    }
    compute_ms_local = compute_timer.elapsed_ms();
    if (!is_active) {
        compute_ms_local = 0.0;
    }

    nccl_barrier(world_comm, world_stream);

    // ═══ Aggregate timing across ranks via NCCL allreduce ═══
    // Pack {compute_ms, setup_ms} as MAX, then 14 op fields as SUM.
    // We use ncclAllReduce so every rank ends with the answer (rank 0 prints).
    double maxbuf[2] = { compute_ms_local, setup_ms_local };
    nccl_allreduce_doubles(world_comm, world_stream, maxbuf, 2, ncclMax);
    double compute_ms_max = maxbuf[0];
    double setup_ms_max   = maxbuf[1];

    double sumbuf[14] = {
        times.qkv_matmul, times.qk_matmul, times.softmax_op, times.av_matmul,
        times.out_matmul, times.bs1, times.ln1, times.bs2,
        times.ffn1, times.gelu_op, times.ffn2, times.bs3, times.ln2, times.bs4
    };
    nccl_allreduce_doubles(world_comm, world_stream, sumbuf, 14, ncclSum);
    double sum_qkv = sumbuf[0];
    double sum_qk  = sumbuf[1];
    double sum_sm  = sumbuf[2];
    double sum_av  = sumbuf[3];
    double sum_ou  = sumbuf[4];
    double sum_b1  = sumbuf[5];
    double sum_l1  = sumbuf[6];
    double sum_b2  = sumbuf[7];
    double sum_f1  = sumbuf[8];
    double sum_ge  = sumbuf[9];
    double sum_f2  = sumbuf[10];
    double sum_b3  = sumbuf[11];
    double sum_l2  = sumbuf[12];
    double sum_b4  = sumbuf[13];

    int hbuf[1] = { times.heads };
    nccl_allreduce_ints(world_comm, world_stream, hbuf, 1, ncclSum);
    int sum_h = hbuf[0];

    if (rank == 0) {
        // Total bootstrap instances = n_active_heads × n_layers × 4 bootstraps
        int n_active = n_heads;       // first n_heads ranks active
        int n_bootstrap_inst = n_active * n_layers * 4;
        double bootstrap_total_ms = sum_b1 + sum_b2 + sum_b3 + sum_b4;
        double per_bootstrap_ms = (n_bootstrap_inst > 0)
            ? bootstrap_total_ms / n_bootstrap_inst
            : 0.0;

        double sum_all = sum_qkv + sum_qk + sum_sm + sum_av + sum_ou
                       + sum_b1 + sum_l1 + sum_b2
                       + sum_f1 + sum_ge + sum_f2 + sum_b3 + sum_l2 + sum_b4;

        printf("\n════════════════════════════════════════════════════════════\n");
        printf("  HP-BERT MULTINODE (NCCL) result — %d ranks (%d active) / %d heads / "
               "%d layer%s / N=%d\n",
               world_size, n_active, n_heads, n_layers, n_layers == 1 ? "" : "s", ring_N);
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

        printf("\n─── Per-operation totals (summed across %d active head completions) ───\n", sum_h);
        auto row = [&](const char *name, double v) {
            printf("  %-18s %9.1f ms   %7.1f ms/head   %5.1f%%\n",
                   name, v, v / std::max(1, sum_h), sum_all > 0 ? 100.0 * v / sum_all : 0.0);
        };
        row("QKV MatMul",    sum_qkv);
        row("Q*K^T MatMul",  sum_qk);
        row("Softmax",       sum_sm);
        row("Attn*V MatMul", sum_av);
        row("Out MatMul",    sum_ou);
        row("Bootstrap #1",  sum_b1);
        row("LayerNorm #1",  sum_l1);
        row("Bootstrap #2",  sum_b2);
        row("FFN1 MatMul",   sum_f1);
        row("GELU",          sum_ge);
        row("FFN2 MatMul",   sum_f2);
        row("Bootstrap #3",  sum_b3);
        row("LayerNorm #2",  sum_l2);
        row("Bootstrap #4",  sum_b4);

        printf("\n══════════════════════════════════════════════\n");
        if (skip_ref) {
            printf("  HP-BERT verification SKIPPED (--skip-ref)\n");
        } else {
            printf("  HP-BERT verification not implemented in multinode binary "
                   "(use bert_hp_multigpu --heads N for verification)\n");
        }
        printf("══════════════════════════════════════════════\n");
        fflush(stdout);
    }

    nccl_barrier(world_comm, world_stream);

    // ─── Tear down ─────────────────────────────────────────────────────────
    NCCL_CHECK(ncclCommDestroy(world_comm));
    cudaStreamDestroy(world_stream);

    // unused now
    (void)n_trials;

    return 0;
}
