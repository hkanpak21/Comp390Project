# BUG-03 — src/multi_gpu/ framework audit

**Date**: 2026-05-11
**Scope**: `distributed_context.{cu,cuh}`, `distributed_eval.{cu,cuh}`, `keyswitching/{galois_oa,output_aggregation,input_broadcast,dist_galois_key_store}`, `comm/nccl_comm.{cu,cuh}`, `overlap/stream_manager.{cu,cuh}`, `partition/rns_partition.{cu,cuh}`, `nvtx_ranges.cuh`
**Result**: **PASS-WITH-FINDINGS** — hot rotation path (`dist_rotate_phantom_inplace`) is correct and lesson-compliant; several MEDIUM/HIGH issues live on non-hot or legacy paths plus two HIGH cleanup-order risks in `destroy()`.

## Summary table per file

Legend: `✓` OK · `△` partial / non-hot · `✗` violates rule · `n/a` not applicable.

| File | Rule of 5 | Hot-path malloc | Pinned H2D | Stream order | STRIDED | NCCL lifecycle | NVTX | Notes |
|---|---|---|---|---|---|---|---|---|
| `distributed_context.cuh/.cu` | △ (DistributedCiphertext ✓; DistributedContext move-only OK; `RotationWorkspace`, `Worker` have raw owners + implicit move) | ✓ (only at first-call grow) | n/a | △ (see HIGH-1, HIGH-2) | n/a | ✓ (`ncclCommDestroy` before context teardown) | n/a | Lazy worker spawn inside `ensure_rotation_workspace`; T-OVERLAP events are inert |
| `distributed_eval.cu` | n/a | ✗ (per-call `cudaMalloc(local_plain)` in `dist_multiply_plain_inplace`, `dist_add_plain_inplace`) | n/a | △ (per-op `cudaStreamSynchronize` is correct but coarse) | n/a | n/a | ✗ (no NVTX) | Spawns `std::thread` per LOCAL-op call instead of using `dispatch_to_all_gpus` |
| `keyswitching/galois_oa.cu` (`dist_rotate_phantom_inplace`) | n/a | ✓ (uses `RotationWorkspace`) | n/a | ✓ (event-gated writeback) | ✓ (passes through to OA-dks) | n/a | ✓ (NVTX_SCOPE phases) | Hot path; lesson-clean |
| `keyswitching/galois_oa.cu` (`dist_rotate_output_aggregation`) | n/a | ✗ (`cudaMalloc(c0/c2_gal_dev)` per call, plus `broadcast_buffer` allocates per peer per call) | n/a | △ (host sync at scatter) | ✓ | n/a | ✗ | Legacy DCT path — kept around but every call mallocs |
| `keyswitching/output_aggregation.cu` | n/a | △ (`make_cuda_auto_ptr` for `t_mod_up`, `cx`, `t_cks` — async-pool, but still per-call) | n/a | ✓ | ✓ (STRIDED kernel + comment locked in) | ✓ | △ (NVTX on `modup`/`moddown` but not on partial-KS kernel or AllReduce) | T-MODUP STRIDED invariant correctly enforced; no leftover CONTIGUOUS branch in the kernel |
| `keyswitching/dist_galois_key_store.cuh` | △ (own dtor `destroy()`; non-copyable; **move ctor/assign not defined** — implicit move *should* be generated since copy is `delete`d, but rule is fragile, see MED-1) | n/a (setup-only) | ✗ (HIGH-3: H2D `cudaMemcpyAsync` from `std::vector<uint64_t>::data()` — pageable) | △ (per-key sync) | ✓ (STRIDED, locked in by comment + invariant) | n/a | n/a | `generate_multinode()` has the same pinned-source issue |
| `keyswitching/input_broadcast.cu` | n/a | △ (`thread_local` static caches grow on demand — OK as a workspace) | n/a | ✓ | n/a | n/a | ✗ | Legacy Phase-1 path; not invoked on hot path |
| `keyswitching/output_aggregation.cuh`, `galois_oa.cuh`, `input_broadcast.cuh` | headers only | n/a | n/a | n/a | n/a | n/a | n/a | Header comments still reference "CONTIGUOUS" in places (see LOW-2) |
| `comm/nccl_comm.cu/.cuh` | △ (`MultiGpuContext` plain struct; no copy/move discipline — see MED-2) | n/a | n/a | ✓ | n/a | ✓ (`ncclCommDestroy` paired) | n/a | `destroy()` does **not** sync streams before destroying them — see HIGH-4 |
| `overlap/stream_manager.cu/.cuh` | △ (`StreamManager` has dtor but `PerGpuStreams` is held by value — copy/move not explicitly disabled; see MED-3) | n/a | n/a | n/a | n/a | n/a | n/a | **DEAD CODE** — no non-archive caller |
| `partition/rns_partition.cu/.cuh` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | Active and correct; inline helpers; `kernel_scatter_limbs` / `kernel_gather_limbs` are exported but unused (see LOW-3) |
| `nvtx_ranges.cuh` | n/a | n/a | n/a | n/a | n/a | n/a | n/a | Single caller (`dist_bert_layer_bench.cu`); hot-path rotation uses `src/util/nvtx_tracer.cuh` instead — name overlap (see LOW-1) |

## Per-file findings

### `distributed_context.cuh` / `distributed_context.cu`

**`destroy()` body (lines 336–413)** — Verified against CLAUDE.md lesson #4:
- Joins worker threads first (lines 348–359). ✓
- Resets `RotationWorkspace::local_cts[g]` slot-by-slot under correct `cudaSetDevice(device_ids[g])` (lines 364–369) so that each `PhantomCiphertext` destructor's `cudaFreeAsync` runs against its captured stream while that stream still exists. ✓
- Frees `c0_gal[g]` / `c2_gal[g]` with correct device set (lines 371–375). ✓
- Destroys per-GPU events, streams, `ncclComm_t` in that order (lines 378–394). ✓
- Final loop **releases** `contexts_[g]` for `g >= 1` and only lets GPU 0's `PhantomContext` actually destruct (lines 409–412). ✓ — matches the commit 71885e7 fix.

**`RotationWorkspace` lifecycle**:
- Lazy: first allocated inside `ensure_rotation_workspace(n_bytes)` (line 260). Grows on demand; old buffers `cudaFree`'d before re-`cudaMalloc`. **Single allocation point**, never freed except in `destroy()`. ✓
- The "grow" path (lines 269–272) does `cudaFree` followed by `cudaMalloc` synchronously without any `cudaStreamSynchronize` — fine because Phase 4b never grows mid-bootstrap (single chain level → fixed `poly_bytes`).

**Persistent worker threads** (lines 280–304 spawn, 307–334 dispatch, 348–359 shutdown):
- Spawn is *inside* `ensure_rotation_workspace`, not `create()`. That is intentional (comment line 277), but means: **(a)** first call into any rotation path is the spawn point, **(b)** any path that uses `dispatch_to_all_gpus` without first having called `ensure_rotation_workspace` throws. Currently safe since `dist_rotate_phantom_inplace` calls them in order, but fragile — see MED-4.
- Mutex+CV handoff, `has_work/done` flags, exception capture via `std::exception_ptr` — correct and conventional.
- Join order before stream/context destruction is correct (workers cannot still be touching device pointers).

**Findings**:
- **[HIGH-1]** (`distributed_context.cu:260–273`) `ensure_rotation_workspace` does NOT `cudaSetDevice(device_ids[0])` at the very top — it relies on the caller's current device. The for-loop sets each GPU correctly, then resets to `device_ids[0]` at the end (line 274). Re-entrant within a single thread is fine. But: workers (spawned lines 286–301) capture the *raw `Worker*`* and call `cudaSetDevice(raw->device_id)` once at start; subsequent calls into `ensure_rotation_workspace` from the main thread while workers are idle is OK. No bug, but the contract is undocumented.
- **[HIGH-2]** (`distributed_context.cu:269`) The growth path issues `cudaFree(rot_ws_.c0_gal[g])` on a pointer that may have been used asynchronously on `ctx.stream(g)` without a prior synchronize. If `ensure_rotation_workspace` is ever called between two rotations with growth (different `poly_bytes` → different chain level), the prior `cudaMemcpyPeerAsync` / kernel may still be in flight. Practically never triggered in the current bootstrap (uniform `poly_bytes`), but a latent BLOCKER once `bert_hp_multigpu` uses multi-chain workloads. Recommend `cudaStreamSynchronize(streams_[g])` before the free.
- **[MED-1]** (`distributed_context.cuh:129–138`) `RotationWorkspace` has raw `uint64_t*` and `std::vector<PhantomCiphertext>` but no explicit copy/move-control. It's *embedded* in `DistributedContext` as a value member, never independently copied — relying on `DistributedContext`'s `= delete` copy declarations. Add `= delete` on `RotationWorkspace` copy ops defensively (Rule of Five lesson #3).
- **[MED-2]** (`distributed_context.cuh:151–161`) `Worker` struct holds `std::thread`, `std::mutex`, `std::condition_variable` — `std::thread` is move-only, the rest are non-copyable. The struct itself is non-copyable in practice, but no `= delete` is declared. Wrapping in `std::unique_ptr<Worker>` (line 232) avoids the issue. ✓ practically; cosmetic only.
- **[MED-3]** (`distributed_context.cu:235, 239`) `distribute_relin_keys` / `distribute_galois_keys` are no-ops with `printf`. They lie about distributing keys. Either rename to `note_relin_keys()` or delete. Currently misleading — a reader following CLAUDE.md "shallow copy keys to each GPU" expects real work here.
- **[LOW-1]** (`distributed_context.cu:483, 497, 449`) `DistributedCiphertext::from_single_gpu` and `::to_single_gpu` allocate per-call with `cudaMalloc` and run a `cudaMemcpyPeer` *for every limb of every poly* (lines 453–469). Two paths off the hot one, but expensive. Pre-allocate scatter buffer in `RotationWorkspace`.

### `distributed_eval.cu` / `distributed_eval.cuh`

**Findings**:
- **[HIGH-3]** (`distributed_eval.cu:179, 224`) `dist_multiply_plain_inplace` and `dist_add_plain_inplace` each `cudaMalloc(&local_plain, ...)` and `cudaFree(local_plain)` **inside the per-GPU thread function, every call**. Direct CLAUDE.md lesson #2 violation. Either:
  1. Reuse a `RotationWorkspace`-style persistent plain buffer (one per GPU), OR
  2. If the plain-op path is no longer hot, archive these files.
  These ops are called per BERT inference; per-op overhead at logN=15 is non-trivial.
- **[MED-4]** (`distributed_eval.cu:59, 73, 103, 132, 164, 210`) Every LOCAL op spawns a new `std::vector<std::thread>` and joins it. `DistributedContext` has `dispatch_to_all_gpus` exactly for this — but `distributed_eval.cu` never calls it. Inconsistent with the Phase 4b worker design. CPU overhead of `std::thread` spawn ≈ ~80 µs/op × ~hundreds of ops per BERT layer = measurable.
- **[MED-5]** (`distributed_eval.cu:267–285, 310–318`) `dist_rescale_to_next_inplace`, `dist_mod_switch_to_next_inplace`, `dist_relinearize_inplace`, `dist_square_and_relin_inplace` all fall through `gather_op_scatter` — i.e. distributed scatter → GPU 0 work → scatter back. Stated as "Phase 1 fallback" in the comments. Confirm whether this code path is still reached anywhere on the headline benchmarks; if not, move under `archive/`.
- **[LOW-2]** No NVTX annotations anywhere in `distributed_eval.cu`. Major op entry points (`dist_add_inplace`, etc.) are invisible to nsys — limits PROFILE-slice usefulness.
- **[LOW-3]** (`distributed_eval.cu:323–331`) Module-level globals `g_dks_store`, `g_dks_key_idx_fn`. Reasonable for a benchmark, but two `DistributedContext`s in the same process clobber each other silently. Document or move into `DistributedContext`.

### `keyswitching/galois_oa.cu`

**Hot path (`dist_rotate_phantom_inplace`, lines 272–414)** — Verified clean:
- Uses `RotationWorkspace` (line 302). ✓
- Persistent per-GPU `local_cts` reused across rotations (lines 332–354). ✓
- Phase 4b worker dispatch (line 393). ✓
- Writeback gated on `oa_done_events[0]` via `cudaStreamWaitEvent(stream0, ...)` before `cudaMemcpyAsync` — correct order for cross-stream ordering (lines 406–411). ✓ Matches the comment block describing the prior bug.
- Final `cudaStreamSynchronize(stream0)` (line 412) serializes the write into the caller's `ct`. ✓
- NVTX scopes around each phase (lines 307, 318, 381, 398). ✓

**Cold path (`dist_rotate_output_aggregation`, lines 88–262)**:
- Per-call `cudaMalloc(c0_gal_dev, c2_gal_dev)` (lines 126–127) plus `broadcast_buffer` peer-allocs (line 77). HIGH-4 below.
- After Phase 3 worker join, calls `cudaEventSynchronize(oa_evts[0])` on host (lines 238–241) — correct (host-side wait before reading via `from_single_gpu`).
- Cleanup loop frees per-GPU temp peer buffers (lines 249–258). Free order vs `cudaStreamSynchronize` is OK because the workers `cudaStreamSynchronize(s)` inside `keyswitching_output_aggregation_dks` (line 504 of `output_aggregation.cu`, taken via the `else` branch when events absent — but here events ARE set, so the else is skipped; cleanup relies on `cudaEventSynchronize` having already waited).

**Findings**:
- **[HIGH-4]** (`galois_oa.cu:74–82, 126–127`) `dist_rotate_output_aggregation` allocates `c0_gal_dev` / `c2_gal_dev` plus per-peer broadcast buffers via `cudaMalloc` every call. CLAUDE.md lesson #2 violation on the cold path. Same fix as HIGH-3.
- **[MED-6]** (`galois_oa.cu:151–164`) Per-call `local_cts[g].resize(...)` constructs PhantomCiphertexts in a temporary `std::vector` that is destroyed at end of function — invokes `PhantomCiphertext` dtor on non-GPU-0 contexts under the *current* device set (line 153). Lesson #4 says these destructors call `cudaFreeAsync` on the captured stream. Here the stream is still alive (within the same call), so safe. But the workers must have completed (they do — `threads[].join()` at line 218) before this vector goes out of scope. ✓ subtle but correct.
- **[MED-7]** (`galois_oa.cu:420–438`) `dist_relinearize_output_aggregation` is a stub that falls back to gather-operate-scatter without actually calling any KS function. The `(void)relin_evks; (void)beta;` plus the "TODO" comment make this clear; flag for either implementation or removal.

### `keyswitching/output_aggregation.cu`

**STRIDED ownership** (lines 70–141): kernel signature `partial_key_switch_inner_prod` walks `for (size_t d = gpu_id; d < beta; d += n_gpus)` — STRIDED. ✓ Locked in by the "(T-MODUP-FIX-2 2026-05-10)" comment block. No CONTIGUOUS branch remains in the kernel.

Both `keyswitching_output_aggregation` (lines 220–373) and `keyswitching_output_aggregation_dks` (lines 382–506) use the same STRIDED kernel and guard with `if (d_count > 0)` for the small-beta edge case. ✓ Matches `dist_galois_key_store.cuh` STRIDED allocator. Invariant is consistent end-to-end.

**Findings**:
- **[MED-8]** (`output_aggregation.cu:282, 290, 424, 432`) Per-call `make_cuda_auto_ptr<uint64_t>(beta * size_QlP_n, s)` for `t_mod_up` and `cx`. Uses `cudaMallocAsync` under the hood (Phantom's `make_cuda_auto_ptr`), so it hits the stream pool — but still allocation in the hot path. At N=65536 / beta=20, `t_mod_up` ≈ 220 MB. Even pool-backed, the stream-ordered alloc can stall the worker if the pool fragments. Persistent workspace would eliminate the risk entirely.
- **[MED-9]** (`output_aggregation.cu:372, 504`) The non-DKS variant always ends with `cudaStreamSynchronize(s)` (line 372). The DKS variant ends with `cudaStreamSynchronize(s)` only in the *else* branch (line 504); when `oa_done_events[gpu_id]` is set (which it always is in the deployed Phase 4b setup), the function returns **without any sync**. The caller (`dist_rotate_phantom_inplace`) does `cudaStreamWaitEvent(stream0, oa_evts[0])` — but ONLY on event index 0, not on the other workers' events. Other workers' streams may still have in-flight kernels (`oa_add_to_ct`) when the workers return and `dispatch_to_all_gpus` `cv.wait` releases the main thread. The next rotation's `ensure_rotation_workspace` (a no-op in steady state) doesn't sync either. The next rotation's `apply_galois_ntt` is on GPU 0's stream0, not on other GPUs' streams. So in steady state: **the next rotation's `cudaMemcpyPeerAsync` (line 323) writes into `ws.c0_gal[g] / c2_gal[g]` on GPU g without waiting for the prior rotation's `oa_add_to_ct` on `ctx.stream(g)` to finish.** `c0_gal/c2_gal` is a DIFFERENT buffer than `local_cts[g].data()`, so no RAW hazard exists; but `local_cts[g]` is reused (Phase 4a) and `local_cts[g].data()` will be `cudaMemset`'d (line 351) and written via `cudaMemcpy` (line 352) on the **default** stream while the previous rotation's `oa_add_to_ct` on `ctx.stream(g)` may still be writing it. This is a **HIGH stream-ordering risk** that hides behind `cudaDeviceSynchronize()` on line 357 — which DOES sync all streams on each GPU. So the inter-rotation sync IS present, but via an expensive device-wide barrier. Worth flagging that the per-GPU `cudaDeviceSynchronize` (line 357) is load-bearing.
- **[LOW-4]** (`output_aggregation.cu:327–334, 459–466`) T-STRAGGLER barrier event recording + waits are kept for "future tracing" but are inert (multiple comments explain). Acceptable. They DO add ~1 µs of overhead per rotation. If PROFILE shows that's negligible, leave; otherwise gate with a runtime flag.
- **[LOW-5]** (`output_aggregation.cu:251–259`) `mul_tech == hps_overq_leveled` branch allocates `t_cks` per call via `make_cuda_auto_ptr`. CKKS path doesn't hit this (we use `mul_tech_type::none` for CKKS), so cold; leave.

### `keyswitching/dist_galois_key_store.cuh`

**STRIDED ownership**: ✓ verified at lines 148–164 (single-node) and 252–266 (multi-node). Comment block at lines 19–30 documents the prior CONTIGUOUS bug and why STRIDED is mandatory. Invariant is now load-bearing and locked in.

**Findings**:
- **[HIGH-5]** (`dist_galois_key_store.cuh:159–161, 260–262`) The H2D copy uses `cudaMemcpyAsync(dptr, comps[d].data(), n_elem * sizeof(uint64_t), cudaMemcpyHostToDevice, sg)` where `comps[d]` is `std::vector<uint64_t>` — **pageable** host memory. CLAUDE.md lesson #1 says this is silently synchronous. Setup-only path (called once at startup), so not a perf killer, but the lesson is explicit and easy to fix: `cudaHostRegister(comps[d].data(), n_elem * sizeof(uint64_t), cudaHostRegisterDefault)` immediately before the copy, then `cudaHostUnregister` after the per-key `cudaStreamSynchronize`. At 50 keys × beta digits × ~50 MB each, the startup cost ≈ 1–2 s of avoidable sync-stall.
- **[MED-10]** (`dist_galois_key_store.cuh:82–87`) `DistGaloisKeyStore` has user-declared destructor `~DistGaloisKeyStore() { destroy(); }` and `= delete`s copy ops. Move ops are **not** explicitly declared. With copy `= delete`, the compiler implicitly does NOT generate a move (it generates `= delete`'d moves). So the class is *neither copyable nor movable*. If any caller `auto ks = build_store(...)` relies on NRVO failing → compile error. Check current usage: callers heap-allocate (`std::unique_ptr<DistGaloisKeyStore>`) or stack-allocate by name then call `generate(...)`. ✓ practically OK, but explicitly declaring `= default` move ops would document intent. Pure cosmetic, downgrade to LOW.
- **[LOW-6]** (`dist_galois_key_store.cuh:307–320`) `estimated_gpu_memory_gb()` uses a hardcoded `0.032` GB constant. Misleading; either compute from the actual buffer sizes (`comps[d].size()` is reachable) or drop.

### `keyswitching/input_broadcast.cu`

**Findings**:
- **[MED-11]** (`input_broadcast.cu:88–107, 135–152`) Uses `thread_local` static caches for `tl_padded`, `tl_gathered`, `tl_full_c2`. Grows on demand via `cudaFree`+`cudaMalloc`. Same HIGH-2 risk: the grow path `cudaFree`s a pointer that may still be in-flight on `ctx.streams[gpu_id]`. Not on the hot rotation path (Phase 1 fallback), so MEDIUM not HIGH. Persistent-workspace fix recommended.
- **[MED-12]** Thread-local static GPU buffers are never freed at program exit. `thread_local` destructors run, but raw `uint64_t*` has none. Process-exit reclaim handles it, but flagged.

### `comm/nccl_comm.cu` / `.cuh`

**`MultiGpuContext::destroy()` (lines 70–93)**:
- Iterates GPUs, sets device, destroys events, **destroys stream BEFORE destroying NCCL communicator**:
  ```
  cudaStreamDestroy(streams[g]);   // line 85
  ncclCommDestroy(comms[g]);       // line 86
  ```
- CLAUDE.md lesson #6 says `ncclCommDestroy` before CUDA context destroy. Stream destroy ≠ context destroy, but **NCCL is still holding handles to the stream when `ncclCommDestroy` is called**. With non-blocking streams this is usually fine, but it is the wrong order. ⇒

**Findings**:
- **[HIGH-6]** (`nccl_comm.cu:85–86`) Destroy order: stream destroyed *before* its NCCL communicator. Should be reversed: `ncclCommDestroy(comms[g])` first, then `cudaStreamDestroy(streams[g])`. Also missing a `cudaStreamSynchronize(streams[g])` (or `cudaDeviceSynchronize()`) before either destroy — any in-flight NCCL kernel on the stream is now operating on a destroyed comm/stream. Note `DistributedContext::destroy()` does the same order at `distributed_context.cu:390–391`. (And `distributed_context.cu` does `cudaDeviceSynchronize()` at line 343 before the teardown, so the sync hazard is covered THERE; but `MultiGpuContext::destroy()` skips the sync.) Both call sites should be aligned: `cudaDeviceSynchronize()` → `ncclCommDestroy` → `cudaStreamDestroy`.
- **[MED-13]** (`nccl_comm.cuh:42–68`) `MultiGpuContext` is a plain struct with `vector<>` and `cudaStream_t/cudaEvent_t/ncclComm_t`. No copy/move discipline declared. It's value-copied in `galois_oa.cu:177–190` (a per-call `MultiGpuContext mgctx;` populated as shallow copies of the persistent comm handles). Shallow copy of `ncclComm_t` is OK (they're opaque handles, not owning), but for safety, document that this struct is "non-owning view when copied". Or split into `OwnedMultiGpuContext` and `MultiGpuContextView`.

### `overlap/stream_manager.cu` / `.cuh`

**Status**: **DEAD CODE.** Only caller is `src/benchmarks/archive/bert_inference.cu`. No live benchmark or framework file includes `stream_manager.cuh`.

**Findings**:
- **[MED-14]** Move to `src/multi_gpu/archive/` next to `pipeline/`. The `OverlapScheduler::schedule_compute_comm_overlap` API embodies a since-abandoned design (the T-OVERLAP comments in `output_aggregation.cu` explain why event-driven overlap proved inert on H100×4 NVSwitch).

### `partition/rns_partition.cu` / `.cuh`

**Status**: Active. Inline helpers `owner_of_limb`, `n_local_limbs`, etc. are used throughout. The CUDA kernels `kernel_scatter_limbs` / `kernel_gather_limbs` and their host wrappers are defined but **never called** by any live code (`grep` returns no hits outside the file itself).

**Findings**:
- **[LOW-7]** Kernels and wrappers are dead. Inline helpers must stay. Either delete the kernel definitions or keep with a comment noting they're library code retained for future paths.

### `nvtx_ranges.cuh`

**Status**: Single live caller (`dist_bert_layer_bench.cu`). The hot rotation path uses `src/util/nvtx_tracer.cuh` (different file, similar purpose) — see `galois_oa.cu:43`.

**Findings**:
- **[LOW-8]** Two NVTX header files with overlapping APIs (`nvtx_ranges.cuh` vs `util/nvtx_tracer.cuh`). Pick one. The `multi_gpu/nvtx_ranges.cuh` macros (NVTX_NTT etc.) are not used; only `NvtxRange` / `NVTX_PUSH` / `NVTX_POP` are referenced.

## Test coverage gaps

### `phantom_threadsafe_smoke.cu` (location: `src/benchmarks/phantom_threadsafe_smoke.cu`)
- Tests: 2-thread, 2-GPU bootstrap via per-thread `PhantomContext`. MAE-vs-thread-0 acceptance gate at 1e-5.
- **Does NOT exercise**: `DistributedContext`, `RotationWorkspace`, persistent workers, NCCL comms, `DistGaloisKeyStore`, T-MODUP STRIDED, OA kernel. It only validates Phantom *thread-safety*, not the multi_gpu framework.
- **Gap**: there is no test that hits `DistributedContext::destroy()` after a real rotation — i.e. no test exists that would have caught the stale-stream segfault from commit 71885e7 if it regressed.

### `multi_gpu_keyswitch_test.cu` (location: `src/benchmarks/multi_gpu_keyswitch_test.cu`)
- Tests: `MultiGpuContext::create/destroy`, `keyswitching_input_broadcast`, `keyswitching_output_aggregation` — all on a single GPU (line 263–264 forces `n_gpus = 1` whenever > 1, with a printf note explaining "true multi-GPU testing requires distributed Phantom contexts").
- **Does NOT exercise**: real multi-GPU NCCL collective behavior, STRIDED ownership across GPUs, `keyswitching_output_aggregation_dks`, `DistGaloisKeyStore`, `DistributedContext`, `dist_rotate_phantom_inplace`, persistent workers.
- The `MultiGpuContext::destroy()` HIGH-6 ordering bug would never trip this test (single GPU = trivial NCCL).
- **Gap**: zero coverage of the hot rotation path or STRIDED invariant. A regression on either is only caught by the full BERT benchmarks (slow, MN5-only).

### Recommended test additions (for later FIX slices, not now)
- A 4-GPU smoke that calls `dist_rotate_phantom_inplace` 10× and `DistributedContext::destroy()` — would assert no segfault and stable MAE.
- A unit test for `DistGaloisKeyStore::generate` on 2+ GPUs that verifies `evks[d]` is non-null iff `d % n_gpus == gpu_id` (catches the CONTIGUOUS regression directly).
- A teardown ordering test that calls `MultiGpuContext::destroy()` after queuing an `ncclAllReduce` without an intervening sync — to surface the HIGH-6 hazard.

## Dead / archived code candidates

| Path | Status | Recommendation |
|---|---|---|
| `src/multi_gpu/overlap/` | Dead (only archive caller) | Move to `src/multi_gpu/archive/overlap/` (parallel to existing `archive/pipeline/`) |
| `src/multi_gpu/partition/rns_partition.cu` (kernel section) | Inline helpers live; kernels + host wrappers dead | Delete kernel + host-wrapper section, keep `.cuh` inline helpers |
| `keyswitching/input_broadcast.{cu,cuh}` | Only Phase-1 fallback path; not invoked from any live benchmark in headline measurements (per `dist_set_galois_key_store` gating in `distributed_eval.cu:339–348`) | Verify; if confirmed dead, move under `multi_gpu/archive/keyswitching/` |
| `keyswitching/galois_oa.cu::dist_relinearize_output_aggregation` (stub at line 420) | TODO stub | Delete or implement |
| `distributed_eval.cu::distribute_relin_keys`, `distribute_galois_keys` | No-op printf | Delete (see MED-3) |
| `multi_gpu/nvtx_ranges.cuh` | One caller; macros unused | Consolidate with `util/nvtx_tracer.cuh` |

## Recommended follow-up FIX slices

- **FIX-BUG-03-01** [BLOCKER] `MultiGpuContext::destroy()` — sync streams + reorder to `ncclCommDestroy → cudaStreamDestroy` (HIGH-6). One-file, ~5 lines. Apply same fix logic at `distributed_context.cu:390–391` for parity (already-synced via line 343, so cosmetic there).
- **FIX-BUG-03-02** [HIGH] `dist_galois_key_store.cuh::generate` and `generate_multinode` — `cudaHostRegister` the `comps[d].data()` source before the H2D `cudaMemcpyAsync` (HIGH-5). Saves ~1–2 s of startup-sync per inference.
- **FIX-BUG-03-03** [HIGH] `distributed_eval.cu::dist_multiply_plain_inplace` / `dist_add_plain_inplace` — move `local_plain` into a per-GPU persistent buffer in `DistributedContext` or in a `PlaintextWorkspace` (HIGH-3). Or, if these paths are dead, archive them.
- **FIX-BUG-03-04** [HIGH] `distributed_context.cu::ensure_rotation_workspace` — `cudaStreamSynchronize(streams_[g])` before `cudaFree` on the grow path (HIGH-2). Pre-emptive against multi-chain `bert_hp_multigpu` workloads.
- **FIX-BUG-03-05** [HIGH] `galois_oa.cu::dist_rotate_output_aggregation` — persistent c0/c2_gal_dev + persistent broadcast buffers in `RotationWorkspace`, OR explicitly mark the function as cold-path-only and add a single `printf` warning on first invocation (HIGH-4).
- **FIX-BUG-03-06** [MEDIUM] `distributed_eval.cu` — replace per-op `std::thread` spawn with `DistributedContext::dispatch_to_all_gpus` (MED-4). Wire through; saves ~80 µs/op × N ops.
- **FIX-BUG-03-07** [MEDIUM] `output_aggregation.cu` — persistent `t_mod_up`, `cx` in `RotationWorkspace` (MED-8). Eliminates stream-ordered alloc fragmentation risk.
- **FIX-BUG-03-08** [MEDIUM] Archive `src/multi_gpu/overlap/` (MED-14) — pure cleanup; mirrors existing `archive/pipeline/`.
- **FIX-BUG-03-09** [MEDIUM] `nccl_comm.cuh::MultiGpuContext` — declare copy/move discipline; or split owning vs view (MED-13).
- **FIX-BUG-03-10** [MEDIUM] Delete `distribute_relin_keys` / `distribute_galois_keys` no-ops in `distributed_context.cu` (MED-3).
- **FIX-BUG-03-11** [MEDIUM] Delete or implement `dist_relinearize_output_aggregation` stub (MED-7).
- **FIX-BUG-03-12** [LOW] Add NVTX coverage to `distributed_eval.cu` major ops (LOW-2).
- **FIX-BUG-03-13** [LOW] Consolidate `multi_gpu/nvtx_ranges.cuh` with `util/nvtx_tracer.cuh` (LOW-8).
- **FIX-BUG-03-14** [LOW] Delete dead scatter/gather kernels in `partition/rns_partition.cu` (LOW-7).
- **FIX-BUG-03-15** [LOW] Add a 4-GPU rotate-then-destroy smoke test to guard against stale-stream regression (test gap).

## Cross-reference verdict against CLAUDE.md non-negotiables

| Lesson | Status in `src/multi_gpu/` |
|---|---|
| #1 pinned H2D | ✗ violated in `dist_galois_key_store.cuh` setup (HIGH-5) |
| #2 no hot-path `cudaMalloc` | ✗ violated in `distributed_eval.cu` (HIGH-3) + cold-path `galois_oa.cu` (HIGH-4); ✓ in hot `dist_rotate_phantom_inplace` |
| #3 Rule of Five | ✓ for `DistributedCiphertext` (explicit); ✓ in spirit for others; minor gaps flagged MED-1/2/13 |
| #4 `release()` non-primary contexts | ✓ correctly done in `DistributedContext::destroy()` (lines 409–412) |
| #6 T-MODUP STRIDED | ✓ enforced in kernel + store; comments lock it in |
| NCCL teardown order | ✗ wrong in `MultiGpuContext::destroy()` (HIGH-6); ✓ effectively safe in `DistributedContext::destroy()` via prior device-sync |
