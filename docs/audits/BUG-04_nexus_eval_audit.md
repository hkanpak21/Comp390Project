# BUG-04 — `src/nexus_eval/` wrappers audit

**Date**: 2026-05-11
**Scope**: `ckks_evaluator{.cu,.cuh}`, `gelu{.cu,.cuh}`, `layer_norm{.cu,.cuh}`,
`softmax{.cu,.cuh}`, `matrix_mul{.cu,.cuh}`, `galois_key_store.cuh`,
`utils.cuh`, `bootstrapping/Bootstrapper.{cu,cuh}`
**Result**: **PASS-WITH-FINDINGS**

The wrappers are functionally correct on the headline single-GPU and DP-multi-GPU
paths. `GaloisKeyStore` is the canonical example of how to do
`cudaHostRegister` + persistent workspaces + per-slot copy/compute events
correctly — every non-negotiable lesson from CLAUDE.md is honoured there.

Findings concentrate in three areas:

1. **`Bootstrapper` has no destructor and no R5** — it owns a heap pointer
   (`mod_reducer`, `new ModularReducer` in the ctor) that is never freed,
   and the implicit copy ctor would shallow-copy it. A user that copies a
   `Bootstrapper` (none today, but nothing prevents it) gets immediate UB
   on dtor of the second instance. **MEDIUM, latent.**
2. **`bsgs_linear_transform_hoisting` and the two-`new`-per-iter hoisting
   variants** in `Bootstrapper.cu` use raw `new PhantomCiphertext`/`delete`
   pairs that are not exception-safe and have one outright leak path
   (`rotated_nobsgs_linear_transform_hoisting`, line ~2211, leaks `tmpct`).
   These paths are not on the bootstrap_3 critical path (we use
   `bsgs_linear_transform`, not `_hoisting`) but they remain in the binary.
   **LOW (off critical path), but should be flagged.**
3. **`softmax.cu` and `layer_norm.cu` never explicitly reset scale before
   chained Phantom calls** (no `x.scale() = SCALE` anywhere in either file
   and no caller-side reset before calling these wrappers). Combined with
   `Evaluator::add_inplace` / `multiply_inplace` *silently* mutating `ct1`'s
   scale to match `ct2`'s (CKKSEvaluator.cuh lines 448-453, 480-484), this
   is the canonical scale-drift footgun from lesson #7 and #6. Today this
   is masked because softmax/layernorm at logN=16 in the per-op binaries
   never chain into a bootstrap, but the moment HP-BERT chains
   `gelu → bootstrap → layernorm → softmax` in one head, drift can
   accumulate. **MEDIUM, latent.**

The `gelu.cu` `coeff_modulus` allocation is correct on the **benchmark side**
(all three per-op binaries: `gelu_align_n65k.cu:129`, `gelu_mgpu_align.cu:139`,
`layernorm_align_n65k.cu:130` all use `i < 18` → 20 limbs total at logN=16,
matching CLAUDE.md lesson #9). `softmax_align_n65k.cu:130` correctly uses
`i < 16` (18 limbs) because NEXUS COEFF_MODULI[2] has only 16 forties for
softmax. **The wrapper file (`gelu.cu`) itself does not construct coeff_modulus**
— that is the caller's responsibility — so there is no `i < 18` loop here to
verify; the comment in CLAUDE.md lesson #9 refers to the benchmark binaries.

## Summary table per file

Legend: ✓ = correct/present; ✗ = absent/broken; n/a = not applicable.
"R5" = Rule of Five status of the dominant class in the file.

| File | Rule of 5 | Pinned H2D | Hot-path malloc | Chain depth | Scale reset | In-place semantics | Notes |
|---|---|---|---|---|---|---|---|
| `galois_key_store.cuh` | ✓ explicit (lines 99-102) | ✓ `cudaHostRegister` line 220 | ✓ persistent slot workspaces line 188-194 | n/a | n/a | n/a | Canonical reference for lessons #1-3 |
| `ckks_evaluator.cuh` (`CKKSEvaluator`) | ✗ no dtor declared, all owned ptrs are raw non-owning (memory managed outside) | n/a | low (per-op `multiply_const_inplace`/`add_const_inplace` allocate a fresh `PhantomPlaintext`) | n/a | ✗ `add_inplace`/`multiply_inplace`/`sub_inplace` SILENTLY normalize `ct1.scale() = ct2.scale()` — masks drift | comments document NEXUS strict-scale workaround | drift hazard, see SCALE-CROSS-CUT below |
| `ckks_evaluator.cuh` (`Evaluator::RemoteGPU`) | ✗ raw owner of heap `PhantomContext*`/`PhantomGaloisKey*`; deleted by worker thread on shutdown, not dtor | n/a | n/a | n/a | n/a | thread + mutex + cv → R5 implicitly suppressed | shared_ptr keeps Evaluator copyable |
| `gelu.cu` | n/a (free function on PhantomCiphertext) | n/a | per-call: many `PhantomPlaintext` + `PhantomCiphertext` temps; no persistent workspace | logN=16 needs 20 limbs — caller responsibility, ✓ in `gelu_align_*` | inputs `x` are mutated by `mod_switch_to_inplace` at lines 51, 62, 88, 97, 105, 110 | **in-place mutation NOT in header signature** (signature is `gelu(x, res)` not `gelu(x_inout, res)`) | see GELU-MUTATION below |
| `layer_norm.cu` | n/a | n/a | per-call temps; no workspace | needs ≥18 limbs (paper uses 18 = 20-2-rescales) | ✗ no scale reset; relies on Evaluator silent renorm | mutates `a` in-place via `rotate_vector + add_inplace` AND `mod_switch_to_inplace(a, …)` line 37 | header signature `layer_norm(x, res, len)` does not signal mutation of `x` |
| `softmax.cu` | n/a | n/a | per-call temps; no workspace | n/a | ✗ no scale reset | mutates `x` in-place via `add_inplace(x, tmp)` line 16 | header does not signal `x` mutation |
| `matrix_mul.cu` | n/a (`MMEvaluator` holds only a non-owning `CKKSEvaluator*`) | n/a | `multiply_power_of_x` does `new uint64_t[…]` + `cudaMemcpyAsync` + `cudaStreamSynchronize` **on every call** (lines 58-83) — host alloc on hot path | depends on `b_compressed_cts` decompress chain; rescales until `coeff_modulus_size() > 1` line 281-285 | n/a | `matrix_mul_range` correctly bounds-clamps `cols_lo`/`cols_hi` lines 191-196 ✓ | see MATMUL-NEW-PER-CALL below |
| `bootstrapping/Bootstrapper.cu` | ✗ **no dtor, no copy/move declarations**; owns raw `ModularReducer*` allocated by `new` in ctor | n/a (key prefetch routed via Evaluator → GaloisKeyStore which IS pinned) | hoisting paths (lines 2026-2211) raw `new`/`delete` per giant-step iter | mature path: `bootstrap_sparse_3` works at logN=15 with 22-limb context | `bootstrap_3` does not reset `cipher.scale()` itself — caller must (argmax does at `argmax_align_n32k.cu:225` ✓; chained HP-BERT call sites do not — see SCALE-CROSS-CUT) | `bootstrap_inplace_3` calls `bootstrap_3` then `cipher = rtncipher` line 3424-3425; `bootstrap_3` also calls `cudaStreamSynchronize(stream)` at end (line 3419) | see BOOT-RAW-OWN, BOOT-HOISTING-LEAK below |
| `utils.cuh` | n/a (stack-only `Timer`) | n/a | n/a | n/a | n/a | n/a | trivially safe; no globals, no singletons |

## Per-file findings

### `src/nexus_eval/galois_key_store.cuh`

- **`cudaHostRegister` usage**: line 220, inside `generate_all_keys` after host
  data is populated, on each component of each key. Loop iterates over all
  `num_keys_ × dnum_` slices. Reports `pinned_bytes` and `pinned_failures` —
  failure path tolerated, not aborted. CLAUDE.md lesson #1 ✓.
- **Async overlap**: dedicated `copy_stream_` (line 199, `cudaStreamNonBlocking`)
  distinct from the compute `alloc_stream_` captured at `generate_all_keys()`
  time. Per-slot `copy_done_event_` lets the compute stream wait for only the
  slot it consumes (line 301). Per-slot `compute_done_event_` blocks the next
  prefetch into the same slot (line 259-261).
- **Rule of Five**: explicit (lines 99-102) — copy deleted, move defaulted.
  Comment correctly cites lesson #3.
- **Destructor stream-safety**: `alloc_stream_` is captured **once** at
  `generate_all_keys()` time (lines 137-138) and reused in the dtor (line 116)
  rather than re-reading `phantom::util::global_variables::default_stream` at
  destruction. The class-doc comment (lines 78-82) explicitly cites CLAUDE.md
  lesson #4 ("the default stream may have been reset…"). ✓
- **Findings**:
  - [LOW] (galois_key_store.cuh:139-153) — `generate_all_keys` synchronises
    the stream once per key (`cudaStreamSynchronize(stream)` line 141) before
    `rk.copy_to_host`. This serialises key generation across all `num_keys`
    keys (~50 ops); for `num_keys = 39` at logN=16 this is acceptable
    (setup-time only) but could be batched with a `cudaMemcpyAsync` pipeline.
    Not on the inference critical path.
  - [LOW] (galois_key_store.cuh:189-194) — `cudaMallocAsync` for the
    `cache_size_` slot buffers is correctly issued on the `alloc_stream_` so
    the dtor's `cudaFreeAsync` on the same stream is well-ordered. However
    there is no `cudaStreamSynchronize` before `cudaMallocAsync` is recorded
    — relying on the pool allocator to handle ordering. This is correct per
    CUDA semantics but worth a paper-time NVTX scope to confirm in nsys.
  - [LOW] (galois_key_store.cuh:107) — early-return guard
    `if (alloc_stream_ == nullptr) return;` skips ALL cleanup (events,
    pinned-host-unregister) if `generate_all_keys` was never called. Safe
    when the object was never used; subtly wrong if a future caller
    allocates events directly without calling `generate_all_keys`.

### `src/nexus_eval/ckks_evaluator.cuh` / `.cu`

- **`CKKSEvaluator` ownership**: header line 765 — `"Memory managed outside
  of the evaluator"` comment. All four raw pointers (`context`, `relin_keys`,
  `galois_keys`, `evaluator's encoder`) are non-owning. ✓ no R5 needed.
- **`Evaluator::RemoteGPU`**: heap-allocated via `make_shared` (line 193, 274),
  worker thread allocates a `PhantomContext` and `PhantomGaloisKey` via `new`
  inside the worker (lines 210, 216) and `delete`s them on shutdown signal
  (lines 262-263). **The dtor for `RemoteGPU` is the implicit one**; if the
  shutdown signal is never sent, the worker thread keeps running and the
  GPU resources leak. `shutdown_remote_gpu()` (line 323) must be called
  manually. **No automatic teardown via dtor.**
- **Scale management primitives** (lines 448-493): every `add_inplace`,
  `multiply_inplace`, `add_plain_inplace`, `multiply_plain_inplace`,
  `sub_plain_inplace`, `sub_inplace` silently mutates either `ct1.scale()`
  (when both are ct) or `plain.scale()` (when one is plain) to make them
  match. Comment lines 487-491 acknowledges this is a workaround for our
  Phantom having strict checks commented out. **This is exactly the
  silent-drift surface CLAUDE.md lesson #7 warns about.**
- **Findings**:
  - [MEDIUM] (ckks_evaluator.cuh:448-453, 480-484, 489-492, 510-513, 527-530)
    — `add_inplace`/`multiply_inplace`/`sub_inplace` silently overwrite
    `ct1.scale() = ct2.scale()` when they differ. NEXUS upstream throws
    (because its Phantom keeps the check enabled). Our impl swallows the
    mismatch. Callers downstream of a chained wrapper sequence will see
    a scale that drifted to whichever operand was on the right of the most
    recent inplace op. The bookkeeping eventually triggers the
    `Phantom::slottocoeff_3` encode validation that argmax already worked
    around (`argmax_align_n32k.cu:225`).
  - [LOW] (ckks_evaluator.cuh:153) `local_device` defaulted to -1 but
    `Evaluator(ctx, enc)` calls `cudaGetDevice(&local_device)` line 186 —
    fine, but `Evaluator()` (the default ctor, line 181) leaves it at -1.
    A default-constructed `Evaluator` assigned-into by `CKKSEvaluator`'s
    ctor (line 800-801 `Evaluator ckks_evaluator(...); this->evaluator = ckks_evaluator;`)
    survives because the parameterised ctor runs first. Confirmed safe.
  - [LOW] (ckks_evaluator.cu:53-62) `prefetch_rotation_step` early-returns
    silently if the step is not in the galois_tool's element table. Correct
    behaviour (the rotation will fall back to a synchronous load via
    `ensure_key_loaded`), but no NVTX scope or log → unobservable in nsys
    if the early-return frequency is high.

### `src/nexus_eval/gelu.cu` / `.cuh`

- **In-place mutation documented?**: **No.** Header (`gelu.cuh:17`) signature
  is `gelu(PhantomCiphertext &x, PhantomCiphertext &res)` — both args are
  non-const refs but conventionally one would read `x` as "input" and `res`
  as "output". The implementation mutates `x` via `mod_switch_to_inplace`
  at line 105 (`mod_switch_to_inplace(Ax, a1.chain_index())` — Ax is a
  local, OK), line 110 (`mod_switch_to_inplace(x, a2.chain_index())` —
  **mutates the input `x`**), line 51/62 (mutates `x_2`, local), line 88/97
  (`mod_switch_to_inplace(coeff_A[i], …)` and `(cts[i], …)` — locals). The
  only input-mutation is line 110; combined with the per-call expense of
  building `b_expanded_cts` from `x` repeatedly, this is precisely the
  hazard lesson #8 calls out.
- **`coeff_modulus` construction (caller-side, cross-reference)**: per
  `src/benchmarks/gelu_align_n65k.cu:129`, `gelu_mgpu_align.cu:139`, and
  `layernorm_align_n65k.cu:130` — all three use `for (int i = 0; i < 18;
  i++) coeff_bits.push_back(40);` between the two 58s, giving 20 limbs
  total at logN=16. CLAUDE.md lesson #9 ✓ (the wrapper itself does not
  construct coeff_modulus — that's caller responsibility, and all relevant
  callers are correct).
- **Findings**:
  - [HIGH] (gelu.cu:110, gelu.cuh:17) — `gelu()` mutates input `x` via
    `mod_switch_to_inplace`. This is undocumented in the header. Per-call
    benchmarks must re-encrypt or deep-copy each iteration. **The single-GPU
    `gelu_align_n65k.cu:205` does `PhantomCiphertext input_ct = base_cipher;`
    per iter, which is a copy-assignment of PhantomCiphertext. If
    PhantomCiphertext copy-assign is a deep device copy, this is safe; if
    it's a shallow-ish copy (typical for vendored Phantom forks), warmup
    will deplete the chain.** The `gelu_mgpu_align.cu` re-encrypts every
    iter (intentional, per its own comment) — that path is unambiguously
    safe.
  - [MEDIUM] (gelu.cu:18-20) — three `encode` calls back-to-back to build
    `p0`, `p1`, `delta`. Each encode rebuilds a slot-count-sized vector on
    the host. At logN=16, slot_count=32768, that's 24KB host alloc per
    encode × 3 encodes × per gelu call. Not on the multi-GPU critical
    path's dominant 30-40ms, but a candidate persistent-encode-cache for
    the per-call latency Y figure.
  - [LOW] (gelu.cu:74-76) — `vector<PhantomPlaintext> coeff_A(8)` is
    rebuilt per call. Coefficients are constants; could be cached on the
    `GELUEvaluator` instance (it has `d_g`, `d_f` already). Would shave
    ~1ms per call (rough estimate from nsys traces).

### `src/nexus_eval/softmax.cu` / `.cuh`

- **Chain-depth requirements**: input `x` is added to its `-len` rotation
  (consumes nothing extra), then `exp(x)` (chain-heavy: implements an exp
  approximation), then `inverse(res)` (Newton iterations, multiple levels).
  At logN=16 NEXUS COEFF_MODULI[2] = `{58, 16×40, 58}` = 18 limbs total,
  which is what `softmax_align_n65k.cu:130` provides. ✓
- **Scale reset before bootstrap**: **the softmax wrapper does not call
  bootstrap and does not reset scale**. Callers that chain
  `gelu → softmax → bootstrap` must reset between. No such caller exists
  in the per-op binaries (softmax doesn't chain into bootstrap there). HP-BERT
  `bert_hp_multigpu.cu` is out of scope for this audit (would be BUG-02).
- **Hot-path allocations**: per-call `PhantomPlaintext delta` (line 28) and
  per-iter `res` reassignment inside the log-step loop (lines 22-24). The
  loop bounds `log_step = log2(len)` — for `len=128` that's 7 iters, for
  `len=4096` that's 12. Each iter does one rotate + one add + one assign
  (`tmp = res`). Assign of PhantomCiphertext is the same shallow-copy
  question as gelu.
- **Findings**:
  - [MEDIUM] (softmax.cu:11-45) — Input `x` is mutated in-place
    (`add_inplace(x, tmp)` line 16). Header (softmax.cuh:15)
    `softmax(PhantomCiphertext &x, PhantomCiphertext &res, int len)` does
    not signal `x`-mutation. Same hazard as gelu.cu:110.
  - [MEDIUM] (softmax.cu:31, 37-39) — Two consecutive `multiply_plain` +
    `rescale_to_next_inplace` separated by an `inverse(res)` call.
    `inverse` consumes ~`iter*2` levels (default iter=4 → 8 levels). After
    those + 2 rescales the chain depth is consumed deep into the second
    58-limb. If a caller chained-in pre-consumed chain (e.g., came from
    gelu output without rescale check), this would silently exhaust mid-
    `inverse`.
  - [LOW] (softmax.cu:14) — `log2(len)` of a runtime int, no explicit
    check that `len` is a power of 2. Argmax-style hostile inputs (`len=129`)
    would do 7 iters and silently produce wrong values.

### `src/nexus_eval/layer_norm.cu` / `.cuh`

- **Chain-depth requirements**: input squared (1 level), accumulate log_step
  rotations (no level cost), multiply by `1/768` plain (no level), rescale
  (1 level), `invert_sqrt(y, 4, 2)` — `d_newt=4` Newton iters + `d_gold=2`
  Goldschmidt → ~10 levels, then multiply with `a` (1 level + rescale).
  Total ≈ 13-14 levels. Fits comfortably in the 20-limb gelu-shaped chain.
- **Scale reset**: same as softmax — no internal reset, no chain into
  bootstrap from this wrapper.
- **Findings**:
  - [MEDIUM] (layer_norm.cu:11-41) — Same in-place mutation of `a`
    (`add_inplace(a, tmp)` line 16, `mod_switch_to_inplace(a, y.chain_index())`
    line 37). Header does not signal mutation.
  - [LOW] (layer_norm.cu:35) — `invert_sqrt(y, 4, 2)` hardcoded; would be
    cleaner as `LNEvaluator` member tuneable, but functional.

### `src/nexus_eval/matrix_mul.cu` / `.cuh`

- **`matrix_mul_range` correctness**: bounds-clamped at lines 191-196
  (`cols_lo<0 → 0`, `cols_hi>64 → 64`, empty-range early-return). Index
  math (lines 234-238) computes the *minimal* compressed-ct slice
  `[k_lo, k_hi)` needed for the requested column range. Decompress is
  restricted; compress + per-column inner loop both honour the range.
  Single-GPU fallback (called by `matrix_mul()` line 167 with `0, 64`)
  exercises the full path identically to the legacy NEXUS code. ✓
- **Allocation pattern in `multiply_power_of_x`**: lines 58-59
  `auto dest_data = new uint64_t[…]; auto dest_data_copy = new uint64_t[…]`
  on every call. Both ~`encrypted_count * rns_coeff_count * 8 bytes` =
  for logN=13 ct: 2 × 10 × 8192 × 8 = 1.28 MB × 2 = **2.56 MB host alloc
  per `multiply_power_of_x` call**. `decompress_ciphertext` (line 142) calls
  it `2 × logN = 26 times per ct` × ~6 compressed cts = **156 host
  allocs of ~2.56 MB each per full single-GPU matrix_mul call**.
- **Findings**:
  - [HIGH] (matrix_mul.cu:58-59, 85-86) — `new uint64_t[…]` + matching
    `delete[]` on a hot path (called O(100×) per matrix_mul). At
    ~2.56 MB per pair this is ~400 MB churn through the host allocator
    per matmul. Lesson #2. Persistent workspace would also let us drop
    the two `cudaStreamSynchronize` calls (lines 61, 83) by reusing a
    pinned host staging buffer per thread.
  - [MEDIUM] (matrix_mul.cu:60, 82) — `cudaMemcpyAsync` from
    `destination.data()` (device) to `dest_data` (heap-pageable
    host) and back. Per CLAUDE.md lesson #1, **the host side is not
    pinned**, so the "async" copy is silently synchronous. With pinning
    and persistent buffer this halves the wall-clock of
    `multiply_power_of_x`.
  - [MEDIUM] (matrix_mul.cu:245) — `vector<PhantomCiphertext>
    b_expanded_cts(64 * 768)` — 49152-element default-constructed vector
    allocated per `matrix_mul_range` call. The comment (lines 240-244)
    acknowledges the over-allocation for index symmetry; the cost is one
    heap alloc + 49152 default ctors per call. Per-thread `MMEvaluator`
    workspace would amortize this.
  - [LOW] (matrix_mul.cu:266-279) — per-column inner loop builds
    `vector<PhantomCiphertext> temp_cts(768)` per output column.
    `add_many` then reads them sequentially. For (cols_hi - cols_lo) = 16
    (4-GPU split), that's 16 × 768 = 12288 temp ciphertexts per
    `matrix_mul_range` call.
  - [LOW] (matrix_mul.cu:62) — `std::copy(dest_data, dest_data + ...)`
    after `cudaStreamSynchronize` — semantically fine but `std::memcpy`
    on raw `uint64_t` would be marginally faster.

### `src/nexus_eval/bootstrapping/Bootstrapper.cu` / `.cuh`

- **`bootstrap_3` location (line 3408)**: dispatches to
  `bootstrap_full_3` or `bootstrap_sparse_3` based on `logn == logNh`,
  calls `cudaStreamSynchronize(cipher.data_ptr().get_stream())` at end
  (line 3419) — single-stream serialization point.
- **8 prefetch hooks** (CLAUDE.md non-negotiable claim): grep finds
  exactly 8 `prefetch_rotation_step` calls, at lines 1893, 1926, 1969,
  2001, 2043, 2077, 2125, 2159. They are paired: lines 1893/1926 are the
  baby-step and giant-step prefetch in `bsgs_linear_transform`; 1969/2001
  in `rotated_bsgs_linear_transform`; 2043/2077 in
  `bsgs_linear_transform_hoisting`; 2125/2159 in
  `rotated_bsgs_linear_transform_hoisting`. Placement is correct: each
  prefetch is issued *immediately after* the current iteration's
  `rotate_vector` is enqueued on the compute stream, so the H→D for
  iteration `i+1` overlaps the rotate kernel for iteration `i`.
- **bsgs paths used by `bootstrap_sparse_3`**: the sparse_3 path (line
  3041) feeds `coefftoslot_3` → `slottocoeff_3` which (via grep) call
  the **non-hoisting** `bsgs_linear_transform` and
  `rotated_bsgs_linear_transform`. So the four prefetches at 1893, 1926,
  1969, 2001 are on the bootstrap_3 critical path; 2043, 2077, 2125,
  2159 (hoisting variants) are not.
- **Stream / async semantics in `bootstrap_3`**: ends with an explicit
  `cudaStreamSynchronize` (line 3419) on the input ct's captured stream.
  This is the synchronization point that makes the whole bootstrap_3
  blocking from the caller's perspective. No leftover synchronous paths
  inside the LT loops; the four critical prefetch hooks fire correctly.
- **Stream-ordering hazard, modraise/subsum**: `bootstrap_sparse_3` has
  six **`cudaDeviceSynchronize()`** calls + matching `fprintf` debug
  prints scattered through `BS_MOD_RAISE`, `BS_SUBSUM`, the
  `coefftoslot_3` call, the `BS_MOD_REDUCTION` call (lines 3048, 3066,
  3069, 3094, 3105). Each is a full-device flush. **These are debug
  prints left in production code; each `cudaDeviceSynchronize` collapses
  the H↔D overlap that the prefetch hooks set up.**
- **Findings**:
  - [HIGH] (Bootstrapper.cu:3043, 3048, 3050, 3064, 3067, 3070, 3073,
    3092, 3094, 3095, 3100, 3105, 3107) — `fprintf(stderr, …)` +
    `cudaDeviceSynchronize()` debug calls remain in `bootstrap_sparse_3`.
    Each `cudaDeviceSynchronize` is a full-device barrier that destroys
    the H↔D overlap delivered by the eight prefetch hooks. Removing
    these is the single biggest critical-path win available without
    touching algorithm. **This is the most concrete BUG-04 fix
    candidate.**
  - [MEDIUM] (Bootstrapper.cuh:43, Bootstrapper.cu:18-19) — `Bootstrapper`
    owns a raw `ModularReducer *mod_reducer` allocated by `new` in the
    ctor; **the class has no destructor declared, no copy/move
    declarations, no `delete mod_reducer`**. The pointer leaks on
    `Bootstrapper` dtor. If anyone ever copies a `Bootstrapper`, both
    copies share the same `mod_reducer` and the first dtor wouldn't free
    it anyway. R5: not satisfied. CLAUDE.md lesson #3 violated for this
    class.
  - [MEDIUM] (Bootstrapper.cu:2026-2095, 2102-2176, 2179-2211) — three
    `_hoisting` variants use raw `new PhantomCiphertext` / `delete` in
    per-iteration loops. Not exception-safe: an exception thrown by
    `multiply_vector_reduced_error` between the `new` and the matching
    `delete` leaks. The `rotated_nobsgs_linear_transform_hoisting` path
    at lines 2179-2212 has **no `delete tmpct` at all** at the end of
    the function (line 2211 just reads `*tmpct` then returns) — leaks
    one `PhantomCiphertext` per call. These paths are off the
    bootstrap_3 critical path today (`_3` uses non-hoisting LTs) but
    remain in the binary for future call sites.
  - [LOW] (Bootstrapper.cu:3419) — single-stream barrier at end of
    `bootstrap_3` makes the entire op blocking. Fine for benchmarking
    (every per-op binary cudaDeviceSynchronize's anyway) but means
    chained HP-BERT calls cannot overlap the tail of bootstrap_N with
    the head of layer-N+1's matmul.
  - [LOW] (Bootstrapper.cuh:33-43) — `vector<long> slot_vec`,
    `vector<vector<vector<vector<complex<double>>>>>` etc. — all
    automatic-storage members, fine. But the FFT-coefficient cubes are
    large (~tens of MB) and are rebuilt every time `generate_LT_coefficient_3`
    is called.

### `src/nexus_eval/utils.cuh`

- **Globals / singletons**: none. The file defines a single `nexus::Timer`
  class (header-only, stack-only, no static state). The `using namespace
  std; using namespace std::chrono;` inside the namespace is a minor
  hygiene smell (pulled into every TU that includes it) but not a race
  hazard.
- **Findings**:
  - [LOW] (utils.cuh:7-8) — `using namespace std` in a header is a
    namespace-pollution risk. Functional, but `std::chrono::` and
    `std::cout`-style qualifications inline would let callers include
    `utils.cuh` without polluting their namespace.

## Scale management cross-cut

Across `src/nexus_eval/`:

| File | `x.scale() = SCALE` calls | Relies on `Evaluator` silent renorm? |
|---|---|---|
| ckks_evaluator.cuh | 0 (Evaluator class IS the silent renormalizer, lines 448, 480, 489, 510, 527) | n/a — it provides the silent renorm |
| gelu.cu | 0 | yes — line 91 (`cts[i].set_scale(ckks->scale)`) and line 115-116 (`s1.set_scale(ckks->scale); s2.set_scale(ckks->scale)`) explicitly normalize, ✓ |
| layer_norm.cu | 0 | yes (no explicit reset; depends on `Evaluator::add_inplace` etc.) |
| softmax.cu | 0 | yes (no explicit reset) |
| matrix_mul.cu | 0 (uses `set_scale` for plain on line 119, 274, 277 — different purpose) | n/a (plain-cipher) |
| Bootstrapper.cu | line 3053 (`cipher.scale() = modulus[0].value()` — required by ModRaise per CKKS theory, NOT a drift-reset) | bootstrap_3 expects caller to have already set the canonical scale |

**Cross-cut conclusion**: the chained-path scale-reset invariant from
CLAUDE.md lesson #7 (`x.scale() = SCALE` before bootstrap) is honoured
**only at the call sites** (argmax: `argmax_align_n32k.cu:225` ✓,
bert_hp: out of audit scope). **None of the wrappers in
`src/nexus_eval/` reset scale internally**, and the `Evaluator` class
*silently* re-aligns scales on every inplace op, which means drift is
invisible until it triggers a Phantom-internal validation deep inside
`slottocoeff_3` (where argmax's bug originally manifested). For
chained HP-BERT (`gelu → bootstrap → layernorm → softmax`), the
contract is: **the caller must `x.scale() = SCALE` between each wrapper
that consumes levels and the next bootstrap**. This contract is not
documented anywhere in `src/nexus_eval/`.

## Rule of Five cross-cut

| Class | Owns device/heap memory? | Dtor declared? | Copy declared? | Move declared? | R5 status |
|---|---|---|---|---|---|
| `GaloisKeyStore` (galois_key_store.cuh:38) | yes (per-slot GPU buffers, events, stream, pinned host registrations) | yes (line 106) | deleted (line 99-100) | defaulted (line 101-102) | ✓ Compliant |
| `CKKSEvaluator` (ckks_evaluator.cuh:745) | no (comment line 765 "memory managed outside") | no (implicit) | implicit (shallow on non-owned ptrs is safe) | implicit | n/a Non-owner; safe |
| `Evaluator` (ckks_evaluator.cuh:120) | partially — via `std::shared_ptr<RemoteGPU> remote` (line 152, refcounted ✓), and indirectly via void* `key_store_`/`dks_*_` raw pointers (non-owning, comment line 152 confirms) | no | implicit (shallow-copy of `shared_ptr` is correct) | implicit | ✓ via `shared_ptr` hack |
| `Evaluator::RemoteGPU` (ckks_evaluator.cuh:127) | yes — owns `PhantomContext*`, `PhantomGaloisKey*`, `std::thread worker` | no | implicit | implicit | **✗ relies on `shutdown_remote_gpu()` being called manually** — worker thread will block dtor of containing `shared_ptr` if not shutdown |
| `Encoder` / `Encryptor` / `Decryptor` (ckks_evaluator.cuh:30, 102, 719) | no (raw non-owning pointers) | no | implicit | implicit | ✓ Non-owners; safe |
| `MMEvaluator` (matrix_mul.cuh:14) | no (just `CKKSEvaluator*`) | no | implicit | implicit | ✓ Non-owner; safe |
| `GELUEvaluator` / `LNEvaluator` / `SoftmaxEvaluator` | no | no | implicit | implicit | ✓ Non-owners; safe |
| `Bootstrapper` (Bootstrapper.cuh:16) | **yes — raw `ModularReducer *mod_reducer` line 43**, allocated by `new` in ctor | **no** | **no** | **no** | **✗ Leaks `mod_reducer` on dtor. Implicit copy ctor would double-own.** |
| `Timer` (utils.cuh:10) | no (stack-only) | no | implicit | implicit | ✓ |

**Cross-cut**: `Bootstrapper` is the one non-compliant R5 class in this
slice. `Evaluator::RemoteGPU` is technically R5-deficient but the
`shared_ptr<RemoteGPU>` wrapper + manual `shutdown_remote_gpu()`
contract works as long as call sites honour the contract.

## Recommended follow-up FIX slices

- **FIX-BUG-04-01** — Remove the `fprintf(stderr, …)` + `cudaDeviceSynchronize()`
  debug calls left in `Bootstrapper::bootstrap_sparse_3` (lines 3043-3107,
  ~7 sync calls). **HIGH** — these collapse the H↔D overlap the eight
  prefetch hooks were designed to provide; this is the single largest
  observable win on the bootstrap_3 critical path.

- **FIX-BUG-04-02** — Give `Bootstrapper` an explicit destructor that
  `delete`s `mod_reducer`, plus `Bootstrapper(const &) = delete` /
  `operator=(const &) = delete` to honour Rule of Five (and prevent
  future shallow-copy of `mod_reducer`). **MEDIUM** — currently latent
  (no one copies it), one leak per `Bootstrapper` instance per process
  lifetime.

- **FIX-BUG-04-03** — Persist `multiply_power_of_x` host scratch
  (`dest_data`, `dest_data_copy`) as `MMEvaluator` members allocated
  once via `cudaHostAlloc` (pinned). Drop the
  `new uint64_t[…]`/`delete[]` per call and the two `cudaStreamSynchronize`
  hops. **HIGH** — ~400 MB host-allocator churn per matmul call, plus
  the lesson-#1 silently-sync `cudaMemcpyAsync` from pageable host.

- **FIX-BUG-04-04** — Document `gelu()`, `softmax()`, `layer_norm()`
  in-place input mutation in their headers — either rename the parameter
  to `x_inout` and add a `// Note: mutates x via mod_switch_to_inplace`,
  or provide a `gelu_const(const PhantomCiphertext &x, PhantomCiphertext &res)`
  overload that deep-copies internally. **HIGH** — currently the only
  warning is in CLAUDE.md, which means each new wrapper user has to
  rediscover it; the existing `gelu_align_n65k.cu` mitigates with a
  copy-assign per iter but that assumes Phantom's copy-assign is deep.

- **FIX-BUG-04-05** — Document the "caller must reset scale before
  chaining into bootstrap" contract on `bootstrap_3` (Bootstrapper.cuh:190).
  Add a one-line comment + cross-reference to `argmax_align_n32k.cu:225`
  as the canonical example. **MEDIUM** — currently institutional knowledge
  only.

- **FIX-BUG-04-06** — Cache `coeff_A` plaintext vector across `gelu()`
  calls (it's 8 constant coefficients re-encoded every call). Either
  promote to `GELUEvaluator` member with lazy first-call init, or
  precompute in the ctor. **LOW** — ~1ms per call, non-critical-path.

- **FIX-BUG-04-07** — Fix the `delete tmpct` leak in
  `Bootstrapper::rotated_nobsgs_linear_transform_hoisting` (line 2211).
  Convert the three `_hoisting` LT functions to `std::unique_ptr` /
  `std::optional<PhantomCiphertext>` to be exception-safe.
  **LOW** — off bootstrap_3 critical path.

- **FIX-BUG-04-08** — Add an NVTX scope around `prefetch_rotation_step`'s
  early-return path in `ckks_evaluator.cu:53-62` so nsys can show when
  prefetches are no-ops vs hit/miss. **LOW** — observability only.
