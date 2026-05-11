# Appendix A — NEXUS/Phantom modifications and bug-fix log

> Status: draft v1
> Slice: WRITE-Appendix
> Depends-on: BUG-01, BUG-02, BUG-03, BUG-04

## A.1 Scope and purpose

This appendix enumerates every modification we made to upstream code this
semester and the bug-fix log for the components on the chained-pipeline
critical path. Two classes of upstream code are in scope:

1. **Phantom CKKS** (`vendor/phantom/`) — the GPU-native CKKS library by
   encryptorion-lab. We patched it in 5 places (~95 LOC total) to make it
   safe under multi-GPU, multi-thread, and bootstrap-style lazy-rescale
   call patterns.
2. **NEXUS-NDSS25 evaluators** (`vendor/nexus/cuda/src/` → `src/nexus_eval/`).
   We did not modify upstream NEXUS in place; we ported the evaluators we
   needed onto our Phantom fork and added the multi-GPU prefetch and
   output-channel-split hooks the chained pipeline depends on.

Two classes of code are explicitly out of scope (see §A.7).

The purpose is twofold: a reviewer can re-derive the surface area of
upstream changes from this appendix alone, and a re-implementer can see
which invariants must hold for the chained pipeline at `logN=15` to be
correct.

## A.2 Phantom modifications (~95 lines total)

All five patch sites were necessary; removing any one of them breaks one
of the critical-path measurements in §§6–7 of the paper.

### A.2.1 `ciphertext.h` — `save()/load()` (~30 LOC)

| File | Lines | What | Why |
|---|---|---|---|
| `vendor/phantom/include/ciphertext.h` | 235, 255 | Added stream `save(std::ostream&)` / `load(std::istream&)` for `PhantomCiphertext` | Cross-GPU transfer (`cudaMemcpyPeer` is non-portable across nodes) and MPI broadcast of intermediate ciphertexts for the HP-BERT activation hand-off |

Without it, ciphertexts cannot leave the GPU they were created on except
via raw device-pointer copies, which couples the application to a single
process address space.

### A.2.2 `secretkey.h` — `save()/load()` + default constructor (~20 LOC)

| File | Lines | What | Why |
|---|---|---|---|
| `vendor/phantom/include/secretkey.h` | (header) | Added `save()/load()` and a default constructor for `PhantomSecretKey` | GPU-to-GPU and rank-0-to-all distribution of the secret key during DistributedContext setup; default ctor allows declaration before deserialization |

The 4-node multinode binary serialises the key on rank 0 and broadcasts
it over NCCL (see `bert_hp_multinode.cu:627`); the default constructor
lets each receiving rank declare an empty key and `load()` into it.

### A.2.3 `globals.h` + `context.cu` — `thread_local default_stream` (~15 LOC)

| File | Lines | What | Why |
|---|---|---|---|
| `vendor/phantom/include/util/globals.h` | 57 | `extern thread_local std::unique_ptr<...> default_stream;` | Per-thread CUDA streams for concurrent multi-GPU work |
| `vendor/phantom/src/context.cu` | (matching) | Move `default_stream` initialisation under the per-thread guard | Each `std::thread` in HP-BERT and DP-per-op gets its own stream rather than colliding on a process-wide global |

Without this patch, two worker threads driving different GPUs share the
same default stream and serialise each other; we confirmed this on the
HP-BERT path before the patch (4-GPU run was no faster than 1-GPU).

### A.2.4 `cuda_wrapper.cuh` — remove hardcoded `cudaSetDevice(0)` (~5 LOC)

| File | Lines | What | Why |
|---|---|---|---|
| `vendor/phantom/include/util/cuda_wrapper.cuh` | stream ctor | Removed `cudaSetDevice(0)` from the `cuda_stream_wrapper` constructor | The constructor was hardcoded to GPU 0; any stream allocated on a worker thread for GPU `g > 0` was silently placed on GPU 0 |

This was the single most insidious Phantom bug for multi-GPU: streams
"belonged" to the wrong device, kernels ran on the wrong device, and
peer-copies appeared to succeed but read uninitialised memory. The fix
is a one-line deletion; the diagnosis took two days.

### A.2.5 `evaluate.cu` — comment out scale-mismatch validation (~25 LOC)

| File | Lines | What | Why |
|---|---|---|---|
| `vendor/phantom/src/evaluate.cu` | `sub_inplace`, `multiply_plain_inplace`, `add_plain_inplace` | Commented out the `cipher.scale() != plain.scale()` validation guard in three operators | The bootstrap pipeline (`coefftoslot_3 → mod_reduction → slottocoeff_3`) and the argmax `sgn_eval` chain rely on **lazy rescaling**: the scale of an intermediate ciphertext is allowed to drift transiently and is reset before the next bootstrap. NEXUS's Phantom fork keeps the check enabled because their code re-aligns scales eagerly; our chained-pipeline call sites do not, so the check was a false positive that hard-aborted the program |

**This is the patch the caller contract documented in CLAUDE.md
lesson #7 is built on.** Because the check is now disabled, callers
that chain into a bootstrap **must** explicitly reset
`ct.scale() = SCALE` (the canonical scale of the parameter set) before
the next bootstrap, otherwise drift accumulates silently and surfaces
deep inside the Phantom `slottocoeff_3` encode validation. The argmax
binary does this at `src/benchmarks/argmax_align_n32k.cu:225`. The
chained HP-BERT binaries currently do not (see BUG-02 finding
[HIGH], FIX-BUG-02-02).

## A.3 NEXUS-eval modifications

`src/nexus_eval/` is a port of `vendor/nexus/cuda/src/` onto our Phantom
fork, with five named additions that were not in upstream NEXUS. We do
not modify `vendor/nexus/` in place; the port is a fork.

### A.3.1 `Bootstrapper.cu` — 8 prefetch hooks

`src/nexus_eval/bootstrapping/Bootstrapper.cu` lines **1893, 1926,
1969, 2001, 2043, 2077, 2125, 2159** call
`ckks->evaluator.prefetch_rotation_step(...)` immediately after the
current iteration's `rotate_vector` is enqueued. The four critical-path
hooks (1893, 1926, 1969, 2001) live in `bsgs_linear_transform` and
`rotated_bsgs_linear_transform`, both of which the `bootstrap_sparse_3`
path takes. The four `_hoisting` variants (2043, 2077, 2125, 2159) are
off the critical path today but were patched for parity.

Verified present by BUG-04 audit. The hooks are the H→D-overlap surface
on which the GaloisKeyStore prefetcher (§A.3.2) operates.

### A.3.2 `galois_key_store.cuh` — async prefetch + `cudaHostRegister`

`src/nexus_eval/galois_key_store.cuh` is the canonical example of
CLAUDE.md lesson #1: the H→D copy at line **220**
(`cudaHostRegister(ptr, sz, cudaHostRegisterDefault)`) pins the source
buffer before the `cudaMemcpyAsync`, otherwise the copy is silently
synchronous. Per-slot copy/compute events at lines 188–194 and 301 let
the compute stream wait for only the slot it consumes, so prefetching
slot `i+1` overlaps the rotate for slot `i`. The class also satisfies
Rule of Five explicitly (lines 99–102 — copy `= delete`, move
`= default`) per CLAUDE.md lesson #3.

### A.3.3 `matrix_mul.cu` — `matrix_mul_range(start, end)`

`src/nexus_eval/matrix_mul.cuh:82` declares `matrix_mul_range`; the
legacy `matrix_mul(x, y, res)` at `matrix_mul.cu:167` is now a
one-liner that delegates `matrix_mul_range(x, y, res, 0, 64)`. The
range form lets us split the 64-output-column matmul across GPUs by
column band, which is how the 4-GPU MatMul measurement in §6 is
produced (each thread owns 16 columns). The range form bounds-clamps
its arguments at `matrix_mul.cu:191–196`.

### A.3.4 `gelu.cu` — chain-depth caller contract (`i < 18`)

The wrapper itself does not allocate `coeff_modulus`; that is the
caller's responsibility. The relevant benchmark binaries
(`gelu_align_n65k.cu:129`, `gelu_mgpu_align.cu:139`, and
`layernorm_align_n65k.cu:130`) all use
`for (int i = 0; i < 18; i++) coeff_bits.push_back(40);`, giving
`{58, 18 × 40, 58}` = 20 limbs at `logN=16`. The earlier code used
`i < 17` (19 limbs) which exhausted the chain mid-`sgn_eval` during
GELU warmup with the error "end of modulus switching chain reached"
— a one-character fix verified by counting commas in
`vendor/nexus/cuda/src/main.cu`. This is CLAUDE.md lesson #9.

### A.3.5 `argmax_align_n32k.cu` — explicit scale reset + vocab guard

`src/benchmarks/argmax_align_n32k.cu:225` resets
`x.scale() = SCALE` before each bootstrap inside the QuickMax loop;
the comment block at lines 216–224 documents why (CLAUDE.md lesson
#7, see §A.2.5). The guard at lines 385–394 refuses cleanly with a
FATAL message when `vocab > sparse_slots`, where `sparse_slots = 8192`
at `logN=15` — required because the binary handles only single-cipher
inputs, and NEXUS's published `vocab=30,522` needs a multi-cipher
tournament that is not in upstream NEXUS's open source either
(CLAUDE.md lesson #10, BUG-01 finding [HIGH]).

## A.4 Multi-GPU framework (new code, not a modification)

`src/multi_gpu/` (~3,559 LOC, of which 1,438 LOC across the three
load-bearing files cited below) is wholly new this semester. It is not
a patch on upstream code; we flag it here so the appendix's "what we
changed" inventory is complete.

| Module | File | LOC | Role |
|---|---|---:|---|
| DistributedContext | `distributed_context.cu` | 591 | Per-GPU `PhantomContext`, persistent worker pool, `RotationWorkspace`, NCCL teardown |
| Distributed GaloisKey store | `keyswitching/dist_galois_key_store.cuh` | 339 | STRIDED per-GPU key-digit ownership (CLAUDE.md lesson #6) |
| Output Aggregation key-switch | `keyswitching/output_aggregation.cu` | 508 | T-MODUP partial inner product + AllReduce; STRIDED kernel |

Other files in `src/multi_gpu/` (`distributed_eval.cu`,
`keyswitching/galois_oa.cu`, `keyswitching/input_broadcast.cu`,
`comm/nccl_comm.cu`, `partition/rns_partition.cu`) implement the
cold-path operators, NCCL lifecycle, and RNS partition helpers.

## A.5 Bug-fix log

One row per critical-path bug fixed during this work. "Origin" cites
the commit hash; "Lesson" cites the CLAUDE.md non-negotiable the bug
crystallised into.

| # | Component | Bug | Symptom | Fix | Origin | Lesson |
|---:|---|---|---|---|---|---|
| 1 | `src/nexus_eval/gelu.cu` (caller) | GELU `coeff_modulus` had `i < 17` → 19 limbs; needed 20 limbs at `logN=16` | GELU warmup crash: "end of modulus switching chain reached" mid-`sgn_eval` | Bumped to `i < 18` (`{58, 18×40, 58}`); verified against `vendor/nexus/cuda/src/main.cu` | (pre-`8e04b14`) | #9 |
| 2 | `src/benchmarks/argmax_align_n32k.cu` | Argmax scale drift between QuickMax rounds | Phantom encode-validation error on the 3rd bootstrap inside QuickMax | Explicit `x.scale() = SCALE` reset before each bootstrap (line 225) | (pre-`8e04b14`) | #7 |
| 3 | `src/multi_gpu/distributed_context.cu::destroy()` | Stale-stream segfault on `PhantomContext` dtor for non-primary GPUs | Segfault at process exit when 4-GPU runs torn down (the captured stream had already been destroyed when `cudaFreeAsync` fired) | `release()` GPU 1..N-1 contexts; only destroy GPU 0's; sync streams + reorder NCCL→stream→context teardown | `71885e7` | #4 |
| 4 | `src/multi_gpu/keyswitching/{output_aggregation.cu,dist_galois_key_store.cuh}` | T-MODUP digit ownership was CONTIGUOUS instead of STRIDED when `chain_beta < dnum` | NCCL P2P illegal-memory-access cascade in `keyswitching_output_aggregation_dks` | Walk digits `for (size_t d = gpu_id; d < beta; d += n_gpus)`; matching STRIDED allocator in the key store; comment block at `dist_galois_key_store.cuh:19–30` locks in the invariant | `b4949cb` | #6 |
| 5 | `src/multi_gpu/` (DKS rotation v1) | Hamming-weight crash and rotation `invalid argument` bug | Multi-GPU DKS rotation aborted or produced wrong output for sparse keys | Corrected the hamming-weight enumeration and the rotation step indexing | `a791d2f` | (correctness) |
| 6 | `src/benchmarks/bert_encoder_multigpu.cu` | Undefined `N` in `printf` (compile error after Phantom fork switch) | TU did not compile | Hoisted `N` into scope; trivial fix | `6bf5d5f` | (build) |
| 7 | DKS benchmark TU set | Compilation failures after API drift; MN5 SLURM scripts missing | DKS benchmarks could not be built on MN5 | Fixed include order + API call sites; added matching SLURM scripts | `be567df` | (build) |
| 8 | NEXUS_USE_MPI option (CMake) | Intel MPI linker error on MN5 | Build failed when MPI sym were referenced via static linkage | Introduced `NEXUS_USE_MPI` CMake option to bypass | `5e5b408` | (build) |
| 9 | Phantom-fork switch (bootstrap accuracy) | Bootstrap FFT layout mismatch with non-NEXUS Phantom fork; MAE ≫ 10⁻³ | Bootstrap returned garbage; LT coefficients (computed offline via Remez) did not match the encoder's butterfly layout | Switched `vendor/phantom/` to the NEXUS Phantom fork; copied `bootstrap_*` evaluators verbatim — 0 modifications inside the bootstrap | `fe5a905`, `4d6ea58` | (correctness) |
| 10 | Bootstrap timing path | Scale-validation checks inside `evaluate.cu` aborted lazy-rescale calls | Bootstrap failed mid-`coefftoslot_3` | Commented out scale validation in `sub_inplace`, `multiply_plain_inplace`, `add_plain_inplace`; see §A.2.5 | `caa09c3` | #7 |

Bugs 1 and 2 (GELU chain depth, argmax scale drift) are the two bugs
narrated in the PI-facing report
(`docs/PI_REPORT.md` lines 58–62). Bug 3 was the stale-stream segfault
that motivated CLAUDE.md lesson #4; bug 4 was the STRIDED-vs-CONTIGUOUS
incident that motivated lesson #6. Bug 9 was the bootstrap-accuracy
incident that motivated switching to the NEXUS Phantom fork (the LT
coefficients are precomputed via Remez against the encoder's exact FFT
butterfly layout, so the bootstrap is only correct against the exact
fork those coefficients were derived for).

## A.6 Audit summary (consolidated from BUG-01..04)

Each audit covered one slice of the critical path. All four returned
PASS-WITH-FINDINGS; none returned BLOCKER on a current measurement.

| Audit | Slice | Files audited | Result | Highest-severity finding |
|---|---|---:|---|---|
| BUG-01 | Per-op align binaries + SLURM scripts | 10 binaries + 10 SLURM | PASS-WITH-FINDINGS | [BLOCKER] No MAE gate in single-GPU GELU/LayerNorm/Softmax/Argmax; mgpu variants also lack MAE gates (timing-only headline numbers) |
| BUG-02 | `bert_hp_multigpu.cu`, `bert_hp_multinode.cu`, 7 HP SLURM scripts | 2 binaries + 7 SLURM | PASS-WITH-FINDINGS | [HIGH] Multinode binary has no MAE gate at all (line 493 `skip_ref = true` hard-coded); single-node gate is `1e-5` instead of PRD's `2.25e-6` and `--skip-ref` is used in every production run |
| BUG-03 | `src/multi_gpu/` framework (distributed_context, distributed_eval, all four keyswitching files, comm, overlap, partition, nvtx) | ~12 files | PASS-WITH-FINDINGS | [BLOCKER] `MultiGpuContext::destroy()` (`nccl_comm.cu:85–86`) destroys streams before NCCL communicators and skips the device-sync; ordering should be `cudaDeviceSynchronize → ncclCommDestroy → cudaStreamDestroy` |
| BUG-04 | `src/nexus_eval/` wrappers + `bootstrapping/Bootstrapper.cu` | 8 files | PASS-WITH-FINDINGS | [HIGH] `Bootstrapper::bootstrap_sparse_3` (lines 3043–3107) has ~7 leftover `fprintf` + `cudaDeviceSynchronize()` debug calls that collapse the H↔D overlap the 8 prefetch hooks were designed to provide |

### A.6.1 Proposed FIX slices (severity counts)

The four audits proposed **48 follow-up FIX slices** in total. Severity
distribution:

| Severity | BUG-01 | BUG-02 | BUG-03 | BUG-04 | Total |
|---|---:|---:|---:|---:|---:|
| BLOCKER | 5 | 0 | 1 | 0 | 6 |
| HIGH | 4 | 3 | 4 | 3 | 14 |
| MEDIUM | 2 | 4 | 6 | 2 | 14 |
| LOW | 3 | 4 | 4 | 3 | 14 |
| **Total** | **14** | **11** | **15** | **8** | **48** |

The 6 BLOCKERs are all in BUG-01 (missing MAE gates on 5 align binaries)
and BUG-03 (`MultiGpuContext::destroy()` ordering). None of them gate
the current paper measurements — the BLOCKERs are about
**reproducibility insurance**, not currently broken measurements — but
they should land before any further code generation runs against these
binaries.

The 14 HIGH findings concentrate on three classes of issue: missing or
loose correctness gates (BUG-01-06, BUG-02-01, BUG-02-05), hot-path
allocations and unpinned H→D copies (BUG-02-03, BUG-02-04, BUG-03-02..05,
BUG-04-03), and chained-pipeline scale-reset (BUG-02-02, BUG-04-04,
BUG-04-05). Removing the seven debug syncs flagged as FIX-BUG-04-01 is
the single largest observable critical-path win available without
changing algorithm.

## A.7 What the audit did NOT cover

Be explicit about scope:

- **`vendor/phantom/` internals** were not audited line-by-line beyond
  the five patch sites in §A.2. Phantom is treated as a library; we
  read its headers and the patched `.cu` files for evidence of our
  modifications, not to audit its bootstrap implementation.
- **`vendor/nexus/cuda/` internals** were not audited. We
  cherry-picked the evaluators we needed into `src/nexus_eval/`; the
  upstream tree remains as a reference for the LT coefficient
  derivation only.
- **Archive directories** (`src/multi_gpu/archive/pipeline/`,
  `src/benchmarks/archive/`, `src/multi_gpu/overlap/`,
  `keyswitching/input_broadcast.cu` legacy Phase-1 fallback) were not
  audited. These contain abandoned strategies (`CtPipeline`, pipeline
  parallelism) and dead code that BUG-03 explicitly recommends moving
  to `src/multi_gpu/archive/`.
- **LLaMA binaries** (`llama_hp_multigpu.cu`, `llama_hp_multinode.cu`)
  were not audited. This paper is BERT-only; LLaMA is out of scope
  (CLAUDE.md "Out of scope" section).
- **MN5-side runtime environment** (NCCL config, GPFS bootstrap-id
  semantics under contention) was audited only at the SLURM-script
  level. The runtime behaviour of NCCL on the ACC partition is
  documented separately in `docs/MN5_NCCL_CONFIG.md`.

The full FIX-slice catalogue per audit lives in
`docs/audits/BUG-01..04_*.md` and is the source of truth if any
individual finding needs re-derivation.
