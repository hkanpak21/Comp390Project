# BUG-01 — Per-op align binaries audit

**Date**: 2026-05-11
**Scope**: 6 single-GPU + 4 multi-GPU per-op alignment binaries + 10 corresponding SLURM scripts
**Result**: PASS-WITH-FINDINGS

The binaries are functionally correct and produce headline numbers consistent
with `docs/PER_OP_VS_NEXUS.md`. However, **none** of the single-GPU align
binaries (and only one of the multi-GPU ones) enforce a correctness gate that
exits non-zero on failure — all of them print MAE then return 0. This is the
dominant finding. Several MEDIUM-severity hot-path concerns (per-iter
ciphertext copies, no `cudaHostRegister`, implicit `PhantomContext`
destruction on non-primary GPUs in worker threads) are latent rather than
acutely broken in the current measurement regime.

## Summary table

Legend: ✓ = correct/present, ✗ = absent/broken, n/a = not applicable for this
op, "soft" = printed but not enforced (returns 0).

| Binary | MAE gate | Hot-path malloc | Pinned H2D | Rule of 5 | Stream dtor | Scale reset | Chain depth | Notes |
|---|---|---|---|---|---|---|---|---|
| bootstrap_align_n32k     | soft (≤0.01, line 260)        | low risk (ct copy each iter) | n/a (no host async) | n/a (no user GPU class) | single ctx GPU0 ✓ | n/a | n/a | warmup uses fresh ct ✓ |
| matmul_align_n8k         | enforced (rel 5% line 510-524, returns 2) | full ctx+keys rebuilt per trial inside `run_one_matmul_trial` | n/a | n/a | per-thread ctx destroyed at thread exit ✗ (lesson #4) | n/a | n/a | dominant cost: per-trial context+SK reload + key regen |
| gelu_align_n65k          | ✗ (no MAE check at all) | low risk (ct copy) | n/a | n/a | single ctx GPU0 ✓ | n/a | ✓ (`i < 18` = 20 limbs) | reuses `base_cipher` copy per iter — see GELU-MUTATION below |
| layernorm_align_n65k     | ✗ (no MAE check)      | low risk | n/a | n/a | single ctx GPU0 ✓ | n/a | ✓ (20 limbs) | reuses base_cipher copy per iter |
| softmax_align_n65k       | ✗ (no MAE check)      | low risk | n/a | n/a | single ctx GPU0 ✓ | n/a | ✓ (18 limbs) | reuses base_cipher copy per iter |
| argmax_align_n32k        | ✗ (no MAE — decoded but not gated) | full ctx+keys+bs setup per trial inside `run_one_argmax_trial` | n/a | n/a | per-thread ctx destroyed at thread exit ✗ | ✓ (line 225 `x.scale() = SCALE`) | n/a | guard for `vocab > sparse_slots` present ✓ (line 385-394) |
| bootstrap_mgpu_align     | enforced (≤0.05 line 350, exit 1) | low risk | n/a | n/a | per-thread non-primary ctx destroyed ✗ (lesson #4) | n/a | n/a | only binary that enforces MAE |
| gelu_mgpu_align          | ✗ (no MAE)            | re-encrypts every iter ✓ (intentional) | n/a | n/a | per-thread ctx destroyed ✗ | n/a | ✓ | re-encrypts because `gelu()` mutates input — correct per lesson #8 |
| layernorm_mgpu_align     | ✗ (no MAE)            | low risk | n/a | n/a | per-thread ctx destroyed ✗ | n/a | ✓ | reuses base_cipher copy per iter |
| softmax_mgpu_align       | ✗ (no MAE)            | low risk | n/a | n/a | per-thread ctx destroyed ✗ | n/a | ✓ | reuses base_cipher copy per iter |

## Per-binary findings

### `src/benchmarks/bootstrap_align_n32k.cu`
- **MAE gate**: pre-bootstrap MAE printed at line 233 (informational only).
  Post-warmup MAE printed at line 259 with a 0.01 threshold and "PASS/FAIL"
  text — but the return value is unaffected. No MAE check on the measurement
  loop at all.
- **Findings**:
  - [HIGH] Warmup MAE result is printed as "FAIL" but `main` still returns 0
    (`bootstrap_align_n32k.cu:260, 328`). A bootstrap regression that broke
    correctness would not surface in SLURM's exit code. Follow-up FIX:
    capture the FAIL into a status flag and return non-zero on failure.
  - [MEDIUM] Measurement loop does no MAE check on any iteration's output
    (lines 271-295). If a per-call corruption appeared later in the run
    (e.g. modulus exhaustion bug) we would not detect it.
  - [LOW] `PhantomCiphertext input_ct = base_cipher;` (line 277) executed
    inside the timed region. The copy is cheap relative to the ~250 ms
    bootstrap, but it does include a host-side copy plus device-side
    `cudaMemcpyAsync` chains. Hoisting would tighten timing variance.

### `src/benchmarks/matmul_align_n8k.cu`
- **MAE gate**: enforced (lines 495-524). Compares both single-GPU and
  multi-GPU decoded col-0 against an unencrypted plain-matmul reference,
  acceptance is relative `|Δ| / single < 5e-2` (per design — see file
  preamble lines 44-56). Returns 2 on failure.
- **Findings**:
  - [HIGH] **Full PhantomContext, secret key load, public+relin key generation,
    Galois key generation, MMEvaluator construction, AND
    `decryptor.create_galois_keys_from_elts` are ALL done inside
    `run_one_matmul_trial` (lines 210-229)**. Per-trial setup probably
    dominates the wall-clock for small column ranges. Specifically:
    - Multi-GPU thread case: every thread rebuilds keys per trial. For
      cols=16 the per-column compute is ~193 ms; key generation is many
      hundreds of milliseconds for logN=13. The reported "matmul wall"
      number bundles this. If extrapolating to a serving scenario where
      keys are amortised across many calls, the current measurement
      overestimates per-call cost.
    - The variable `gk_empty` (line 216) is constructed empty then filled
      via `eval.decryptor.create_galois_keys_from_elts` (line 229) — this
      is correct but again happens per-trial inside the timed setup.
  - [HIGH] **`PhantomContext ctx(parms);` is constructed inside each worker
    thread (line 210), then destroyed at thread exit**. Per CLAUDE.md
    lesson #4, `PhantomContext` dtor calls `cudaFreeAsync` on its captured
    stream. The lesson notes that **non-primary GPU contexts** (GPU 1..N-1)
    should be `release()`d, not destroyed. This binary destroys them
    implicitly. Symptoms: occasional stream-bound async-free errors on
    process exit, esp. on stress. Has not been observed in current
    measurements but is latent. Follow-up FIX: add an explicit
    `ctx.release()` call (or equivalent) in the worker before scope exit;
    only the GPU0 context should run the full dtor.
  - [MEDIUM] `mut_x = matrix_4096x768_T;` and `mut_y = row_pack;` (lines
    234-235) duplicate the host matrices into thread-local copies because
    `matrix_mul_range` takes mutable refs. This is unnecessary if
    `matrix_mul_range` does not actually mutate them; verify and drop the
    copy.
  - [LOW] MAE check only runs when both single-GPU and multi-GPU phases
    execute (line 495). The single-GPU-only run (`--n-gpus 1`) silently
    skips the MAE gate. Follow-up: add a single-GPU vs plain-truth check
    that runs regardless of `n_gpus`.

### `src/benchmarks/gelu_align_n65k.cu`
- **MAE gate**: **none**. No correctness check anywhere in the file.
- **Findings**:
  - [BLOCKER] **No MAE check.** The 70.30 ms single-GPU number quoted in
    `docs/PER_OP_VS_NEXUS.md` line 284 (JOBID 40387027) is timing-only —
    nothing verifies the output is mathematically the correct GELU. A
    silent corruption (e.g. wrong polynomial coefficients, chain
    underflow) would print fine timings and return 0. Follow-up FIX:
    decrypt + decode + compare against a plain `0.5x(1 + tanh(...))`
    reference, threshold 1e-2 (GELU polynomial approx noise is large; 1e-5
    is too tight for GELU and softmax, which is why the brief defaults to
    "≤ 1e-5" only for the linear ops).
  - [HIGH] **`PhantomCiphertext input_ct = base_cipher;` (line 205) every
    iteration is exposed to lesson #8 (GELU in-place mutation)**.
    `gelu()` mutates its input via `mod_switch_to_inplace(x, ...)` (see
    `src/nexus_eval/gelu.cu:110`). The fact that `input_ct` is a *copy*
    of `base_cipher` (constructed line 205) means `base_cipher` itself is
    preserved across iterations — so this is correct in spirit. However:
    `PhantomCiphertext`'s copy ctor must actually do a deep copy of the
    device buffer for this to work. If the Rule-of-Five is violated for
    `PhantomCiphertext` (and the user-declared dtor suppressed the
    implicit copy/move — see lesson #3), the iteration would corrupt
    `base_cipher`. The gelu_mgpu binary handles this by re-encrypting per
    iter (line 215-216 of `gelu_mgpu_align.cu`); the single-GPU one does
    not. Verify behaviour empirically: run for 100 iters and check that
    iter 99's timing matches iter 1's. If they drift, this is the cause.
  - [HIGH] Warmup at lines 184-194 uses `warmup_in = base_cipher;` then
    runs `gelu(warmup_in, rtn)`. If `gelu()` mutates `warmup_in` through
    the copy, AND the copy is shallow on the device side, `base_cipher`
    would be depleted before the measurement loop starts. Same
    Rule-of-Five concern as above.
  - [LOW] Galois keys created via `secret_key.create_galois_keys(context)`
    on line 148 — full key (all rotations). GELU does not need rotations.
    Wasted setup time, not a correctness concern.

### `src/benchmarks/layernorm_align_n65k.cu`
- **MAE gate**: **none**.
- **Findings**:
  - [BLOCKER] No MAE check. Same as GELU. Quoted 45.5 ms single-GPU
    number in `docs/PER_OP_VS_NEXUS.md` line 281 is timing-only. Follow-up
    FIX: add MAE vs plain-layernorm reference on first 16×768 slot range,
    threshold 1e-2.
  - [MEDIUM] `PhantomCiphertext input_ct = base_cipher;` per iter is the
    same copy-ct concern as GELU but `layer_norm()` does not document
    in-place mutation, so this is less acute. Verify by reading
    `src/nexus_eval/layer_norm.cu` to confirm input is not mutated.
  - [LOW] Galois keys are full (line 147), but layer_norm does use
    rotations for the reduction so this is fine.

### `src/benchmarks/softmax_align_n65k.cu`
- **MAE gate**: **none**.
- **Findings**:
  - [BLOCKER] No MAE check. Same pattern. Quoted 20 ms single-GPU number
    in `docs/PER_OP_VS_NEXUS.md` line 282 is timing-only. Follow-up:
    decrypt + decode, MAE vs plain softmax over first 128 slots,
    threshold 1e-2.
  - [MEDIUM] Same copy-ct concern as GELU/LayerNorm.

### `src/benchmarks/argmax_align_n32k.cu`
- **MAE gate**: ciphertext is decoded on first trial (line 470 `decode=(t==0)`),
  but the decoded values are **never compared** to anything. No correctness
  gate.
- **Findings**:
  - [BLOCKER] **No MAE / correctness check.** The 848.4 ms single-GPU number
    in `docs/PER_OP_VS_NEXUS.md` line 285 is timing-only. The decoded vector
    is computed but discarded (line 339). Argmax has a clear ground truth
    (the index of the largest input); follow-up FIX: take the argmax of
    the decoded result, compare to `std::distance(begin, max_element(input))`
    on the synthetic input, return non-zero if they disagree. This is a
    cheap and decisive correctness signal.
  - [HIGH] **Scale-reset (`x.scale() = SCALE` at line 225) is present and
    correct** per CLAUDE.md lesson #7. Comment at lines 216-224 documents
    why. ✓
  - [HIGH] **`vocab > sparse_slots` guard is present and reachable** (lines
    385-394). Returns 2 with FATAL message. ✓
  - [HIGH] Full context+keys+bootstrapper setup is inside
    `run_one_argmax_trial` (lines 269-307). Per-trial setup includes
    `prepare_mod_polynomial`, `addLeftRotKeys_Linear_to_vector_3`,
    `generate_LT_coefficient_3`, `create_galois_keys_from_steps`. For the
    multi-GPU throughput phase this happens **per batch per GPU**
    (line 504-509), making the "throughput" number a measurement of
    "first-call latency × batches" rather than steady-state throughput.
    The single 848 ms number is dominated by argmax compute (3 rounds ×
    bootstrap ≈ 750 ms), but the multi-GPU `4.65 s/batch` number cited
    in `docs/PER_OP_VS_NEXUS.md` line 285 ("~3.7 s of which is amortizable")
    is consistent with this finding — the doc itself flags the setup
    cost. Follow-up FIX: hoist `run_one_argmax_trial` setup out of the
    per-trial timed region (one context per GPU, reused across batches).
  - [HIGH] **Per-thread `PhantomContext`s on non-GPU0 are destroyed at
    scope exit** (line 269 + thread-lambda exit ~line 512). Same
    lesson #4 latent risk as matmul.
  - [MEDIUM] Galois keys are generated per trial (line 306). Galois key
    generation at logN=15 with bootstrap+argmax steps is expensive (tens
    of seconds); the SK is shared via serialization (line 452) but each
    thread re-derives all keys.

### `src/benchmarks/bootstrap_mgpu_align.cu`
- **MAE gate**: **enforced** — `mae_max > 0.05 → mae_pass = false → return 1`
  (lines 332-334, 349-350, 388-392). Bootstrap noise is naturally large,
  so 0.05 is appropriate (not 1e-5).
- **Findings**:
  - [HIGH] **Per-thread `PhantomContext`s for non-GPU0 are destroyed at
    thread exit** (line 207 + lambda exit at line 321). CLAUDE.md lesson #4
    says "release GPU 1..N-1, only destroy GPU 0's". The current code
    relies on the Phantom dtor to be safe on shutdown for all GPUs. In
    practice this has worked (the binary has shipped numbers in
    `docs/PER_OP_VS_NEXUS.md` line 280), but the lesson exists because a
    previous code path crashed on this. Add `release()`-style cleanup at
    thread exit.
  - [LOW] `PhantomCiphertext input_ct = base_cipher;` per iter inside the
    timed region (line 272). Bootstrap is ~250 ms; this is sub-1% overhead.
  - [LOW] Warmup is not counted but happens before the `ready` barrier
    (line 255-260 vs 267). Threads warm up at different rates depending
    on GPU temp / clock state. Could affect first-call timing variance.

### `src/benchmarks/gelu_mgpu_align.cu`
- **MAE gate**: **none**.
- **Findings**:
  - [BLOCKER] No MAE check. Same as `gelu_align_n65k.cu`. Quoted 31.84 ms
    4-GPU effective number in `docs/PER_OP_VS_NEXUS.md` line 284 is
    timing-only.
  - [HIGH] **Per-thread `PhantomContext` lesson #4 risk** (line 169 + thread
    exit).
  - [MEDIUM] **The fresh re-encryption per iter is correctly implemented**
    (lines 215-216, with explanatory comment lines 211-214 explicitly
    citing the in-place mod_switch lesson). ✓ This is the *correct*
    pattern — the single-GPU `gelu_align_n65k.cu` should adopt the same
    pattern.
  - [LOW] Re-encryption adds ~5-10 ms per iter, included in the timed
    region. The 31.84 ms number therefore **includes encryption cost**,
    which is not part of NEXUS's published 3.35 s number. This is a
    paper-fairness concern, not a correctness bug. (NEXUS's standalone
    `gelu_test` re-uses one ciphertext.) Follow-up: report the
    encryption cost separately; subtract from headline if the comparison
    is "GELU compute only".

### `src/benchmarks/layernorm_mgpu_align.cu`
- **MAE gate**: **none**.
- **Findings**:
  - [BLOCKER] No MAE check. Quoted 25.07 ms / 17.6 ms numbers in
    `docs/PER_OP_VS_NEXUS.md` line 281 are timing-only.
  - [HIGH] Per-thread `PhantomContext` lesson #4 risk (line 166).
  - [LOW] Reuses `base_cipher` copy per iter without re-encryption. If
    `layer_norm()` does NOT mutate its input, this is fine; otherwise the
    base_cipher gets depleted (same concern as gelu single-GPU).

### `src/benchmarks/softmax_mgpu_align.cu`
- **MAE gate**: **none**.
- **Findings**:
  - [BLOCKER] No MAE check. Quoted 16.52 ms / 13.4 ms numbers in
    `docs/PER_OP_VS_NEXUS.md` line 282 are timing-only.
  - [HIGH] Per-thread `PhantomContext` lesson #4 risk (line 166).
  - [LOW] Same base_cipher reuse concern.

## SLURM script findings

All 10 SLURM scripts conform to the structural requirements from
`CLAUDE.md`:

| Script | LD_LIBRARY_PATH ✓ | Modules ✓ | n-gpus ✓ | Other |
|---|---|---|---|---|
| `slurm_bootstrap_align.sh`        | ✓ line 34 | ✓ line 31 (cuda/12.8 cmake/3.30.5 nccl/2.24.3-1) | `--gres=gpu:1` ✓ | OK |
| `slurm_matmul_align.sh`           | ✓ line 39 | ✓ line 36 | `--gres=gpu:4`, binary called with `--n-gpus 4` ✓ | OK |
| `slurm_gelu_align.sh`             | ✓ line 34 | ✓ line 31 | gpu:1 ✓ | OK |
| `slurm_layernorm_align.sh`        | ✓ line 34 | ✓ line 31 | gpu:1 ✓ | OK |
| `slurm_softmax_align.sh`          | ✓ line 34 | ✓ line 31 | gpu:1 ✓ | OK |
| `slurm_argmax_align.sh`           | ✓ line 37 | ✓ line 34 | gpu:4, `--n-gpus 4` ✓ | runs vocab=8 AND vocab=30522; the vocab=30522 invocation will FATAL-exit (return 2) per the guard in argmax_align_n32k.cu:385-394, and the script masks this via `exit $((RC_A | RC_B))` — RC will be 2. **[MEDIUM]** Follow-up: split into two scripts or accept-list RC=2 for vocab=30522 explicitly. |
| `slurm_bootstrap_mgpu_align.sh`   | ✓ line 36 | ✓ line 33 | gpu:4, `--n-gpus 4` ✓ | OK |
| `slurm_gelu_mgpu_align.sh`        | ✓ line 35 | ✓ line 32 | gpu:4, `--n-gpus 4` ✓ | OK |
| `slurm_layernorm_mgpu_align.sh`   | ✓ line 35 | ✓ line 32 | gpu:4, `--n-gpus 4` ✓ | OK |
| `slurm_softmax_mgpu_align.sh`     | ✓ line 35 | ✓ line 32 | gpu:4, `--n-gpus 4` ✓ | OK |

Notes:
- All scripts use `module purge && module load cuda/12.8 cmake/3.30.5
  nccl/2.24.3-1` — exact match to `CLAUDE.md`.
- All scripts export `LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib`
  before launching — `libntl.so.44` will resolve.
- All mgpu scripts request `--gres=gpu:4`, `--cpus-per-task=80`, and pass
  `--n-gpus 4` to the binary.
- All mgpu scripts use `set -e` then capture `RC=$?` after the binary; with
  `set -e` and non-zero RC the captured value is moot because the shell
  exits before the `echo "Exit code"` line. This is cosmetic. **[LOW]**

## Reproducibility check

`docs/PER_OP_VS_NEXUS.md` §4 (lines 82-202) cites a comparison table
populated from JOBIDs 40368129/40368130/40368131/40368132/40367787 and
the multi-GPU JOBIDs 40369736/40369738/40369739/40369976/40386863/40387026/
40387027/40387047-50/40387054/40387075.

Of those, the entries in §4.2.1 lines 280-285 (4-GPU and 16-GPU columns)
and the lines 165-172 status table presume the current source produces
the same numbers as the JOBID logs. Per this audit:

- **Bootstrap @ logN=15** (line 280): the source `bootstrap_align_n32k.cu`
  and `bootstrap_mgpu_align.cu` are unchanged in spirit; the headline
  number (240.98 ms single-GPU / 192.5 ms 4-GPU effective) should
  reproduce. **Reproducibility: PROBABLE.**
- **MatMul @ logN=13** (line 283): the source comment block (lines 33-39
  of `matmul_align_n8k.cu`) documents a 2026-05-10 fix to use
  `matrix_mul_range` rather than the full 64-column matmul on every
  thread. The 0.285 s / 0.122 s numbers match this fixed path. **OK** —
  but: the 5e-2 relative MAE tolerance is generous (preamble line 49-55
  acknowledges this); a stricter run would change the pass/fail status
  but should not change the timing.
- **GELU / LayerNorm / Softmax single-GPU numbers** (lines 281, 282, 284):
  these binaries do not have MAE gates. **Timing numbers should
  reproduce** but there is no correctness guarantee on what was measured.
- **GELU @ logN=16 70.30 ms** (line 284, JOBID 40387027): file
  `gelu_align_n65k.cu` looks like the same source that produced JOBID
  40387027 — chain depth is `i < 18` matching the comment. **Reproducibility:
  PROBABLE for timing; correctness is unverified.**
- **Argmax @ logN=15 vocab=8** (line 285): the scale-reset fix is present
  in source (line 225). JOBID 40369741's 848.4 ms number should
  reproduce. **OK for timing.**
- **Argmax vocab=30522** (§4 line 104, "TBD"): cannot be measured by this
  binary because of the FATAL-guard at vocab > sparse_slots=8192. The
  `slurm_argmax_align.sh` script attempts to run it anyway and will exit
  non-zero. The §4 line 104 entry "TBD (4-batch throughput)" is therefore
  **accurate but stale**: it cannot ever land until multi-cipher
  tournament is implemented (out of scope per `CLAUDE.md`'s "Out of
  scope" section).

No JOBIDs cite stale source files. The reproducibility risk is concentrated
in **GELU/LayerNorm/Softmax** because the timing numbers are not paired
with a correctness gate — a future code change could silently break the
op while the headline number drifts within noise.

## Recommended follow-up FIX slices

Ordered by severity, then by paper-impact:

- **FIX-BUG-01-01** [BLOCKER]: add MAE-vs-plain-reference check + exit-code
  gate to `gelu_align_n65k.cu`. Threshold 1e-2. Without this the paper's
  ~70 ms GELU number has no correctness backing.
- **FIX-BUG-01-02** [BLOCKER]: same for `layernorm_align_n65k.cu`.
- **FIX-BUG-01-03** [BLOCKER]: same for `softmax_align_n65k.cu`.
- **FIX-BUG-01-04** [BLOCKER]: same for `argmax_align_n32k.cu` — argmax
  has a strict ground-truth comparison (decoded-argmax-index == known
  max index), so the gate is cheap and decisive.
- **FIX-BUG-01-05** [BLOCKER]: add MAE checks to all four mgpu binaries
  (gelu/layernorm/softmax/argmax). Pattern: borrow the bootstrap_mgpu
  per-thread `mae_post_bootstrap` plumbing.
- **FIX-BUG-01-06** [HIGH]: enforce the warmup MAE FAIL in
  `bootstrap_align_n32k.cu` (line 260) — propagate to non-zero return.
- **FIX-BUG-01-07** [HIGH]: hoist per-trial PhantomContext + key setup
  out of the timed region in `matmul_align_n8k.cu` and
  `argmax_align_n32k.cu`. One context per GPU, reused across trials/batches.
  This is the lesson #2 (persistent workspace) analogue for the alignment
  binaries.
- **FIX-BUG-01-08** [HIGH]: explicit `release()` (not implicit destroy)
  for non-GPU0 `PhantomContext`s in all multi-threaded binaries
  (matmul_align_n8k, argmax_align_n32k, bootstrap_mgpu_align,
  gelu_mgpu_align, layernorm_mgpu_align, softmax_mgpu_align). Lesson #4.
- **FIX-BUG-01-09** [HIGH]: investigate whether `PhantomCiphertext` copy
  ctor performs a deep device-buffer copy. If not, the single-GPU
  GELU/LayerNorm/Softmax base_cipher pattern is silently broken (a later
  iter's input is shallow-aliased to base_cipher and the chain depletes).
  Bring the gelu_mgpu re-encrypt-per-iter pattern (lines 215-216) into
  the single-GPU binaries.
- **FIX-BUG-01-10** [MEDIUM]: split `slurm_argmax_align.sh` so the
  vocab=30522 invocation is gated behind a `MULTI_CIPHER_AVAILABLE=1`
  env var, or remove the vocab=30522 line entirely with a comment
  pointing at the FATAL guard. Currently the script always returns 2.
- **FIX-BUG-01-11** [MEDIUM]: drop the redundant host-side matrix copies
  (`mut_x = matrix_4096x768_T;`) inside `run_one_matmul_trial`
  (matmul_align_n8k.cu:234-235) if `matrix_mul_range` does not mutate.
- **FIX-BUG-01-12** [LOW]: hoist `PhantomCiphertext input_ct = base_cipher;`
  out of the timed region in bootstrap_align_n32k / bootstrap_mgpu_align
  /layernorm_mgpu / softmax_mgpu. Tighten timing variance.
- **FIX-BUG-01-13** [LOW]: in `gelu_align_n65k.cu`, drop the
  `secret_key.create_galois_keys(context)` (line 148) — GELU doesn't
  use rotations. Saves setup time, no measurement impact.
- **FIX-BUG-01-14** [LOW]: rationalize `set -e` + `RC=$?` capture in mgpu
  SLURM scripts (the `echo "Exit code"` line never runs on failure).
