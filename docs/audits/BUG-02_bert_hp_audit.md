# BUG-02 — bert_hp_multigpu / bert_hp_multinode audit

**Date**: 2026-05-11
**Scope**: `src/benchmarks/bert_hp_multigpu.cu`, `src/benchmarks/bert_hp_multinode.cu`, and all `scripts/mn5/slurm_bert_hp*.sh` variants
**Result**: PASS-WITH-FINDINGS

The two HP-BERT binaries are structurally sound for the Goal 2 unit run (`--n-gpus 1 --heads 1 --layers 2 --N 32768`) and for 4× / 16× strong-scaling measurements. No BLOCKERs found. Several HIGH/MEDIUM findings are listed below — none gate MEASURE-01, but the HP-MAE gate is materially weaker than the PRD's `2.25e-6` target and the multinode binary has no MAE gate at all.

## Summary table

| Binary | HP MAE gate | NCCL clean | Persist WS | Ctx release | STRIDED | Scale reset | Pinned H2D | CLI plumbing | Notes |
|---|---|---|---|---|---|---|---|---|---|
| `bert_hp_multigpu.cu` | Partial (1e-5, not 2.25e-6) | N/A (no NCCL) | N/A (no DKS) | N/A (no DistributedContext) | N/A (no T-MODUP) | Partial (intra-head only, not at layer boundary) | OK (via `enable_key_streaming` → `cudaHostRegister` in `GaloisKeyStore`) | OK | `--skip-ref` allowed; default is N=65536 not N=32768 |
| `bert_hp_multinode.cu` | **Missing entirely** | OK (init+destroy paired) | N/A (no DKS) | N/A (no DistributedContext) | N/A (no T-MODUP) | Partial (same as above) | **`cudaMemcpyAsync` from pageable host in NCCL helpers** | OK (no `--n-gpus`; locked to 1 GPU/rank) | Default N is 32768 here (correct for Goal 2) |

Cross-references checked:
- `src/multi_gpu/keyswitching/dist_galois_key_store.cuh:148,252` — STRIDED ownership confirmed (not used by HP path)
- `src/multi_gpu/keyswitching/output_aggregation.cu:70,116,289,431` — STRIDED inner product confirmed (not used by HP path)
- `src/multi_gpu/distributed_context.cu:336-415` — `destroy()` correctly `release()`s GPU 1..n-1 contexts (not used by HP path)
- `src/nexus_eval/galois_key_store.cuh:220` — `cudaHostRegister` confirmed inside `GaloisKeyStore` (HP path inherits this).

## Per-binary findings

### bert_hp_multigpu.cu

- **HP MAE gate**: a reference pass on GPU 0 IS computed (lines 458–509, `setup_per_gpu` + `run_one_head` chained `n_layers` times). Per-head MAE is printed and compared at lines 654–680. Threshold is `mae_threshold = 1e-5` (line 648), NOT the PRD's `2.25e-6`. `--skip-ref` skips the gate entirely (lines 322, 651, 738) and the SLURM scripts (`slurm_bert_hp_12layer.sh:41`, `slurm_bert_hp_n32768.sh:50`) all run with `--skip-ref`, so the PRD MAE gate is never actually enforced in production runs.
- **NCCL setup**: no NCCL; pure `std::thread` per GPU. No MPI either. Clean.
- **Persistent workspace**: not applicable — HP path does not use DKS or the `RotationWorkspace` pattern. Rotations go through the single-GPU pinned-host streaming path (`enable_key_streaming`, line 304), which itself does the pinning via `cudaHostRegister` inside `GaloisKeyStore` (`src/nexus_eval/galois_key_store.cuh:220`). No per-call `cudaMalloc` in the hot path was found in the HP source itself.
- **Per-GPU context lifetime**: each per-thread `PhantomContext` lives inside a `unique_ptr` (line 545) that goes out of scope at thread end on the thread's GPU. Because `cudaSetDevice(g)` is called at the top of each thread (line 543) and the `unique_ptr`s are local, the dtor runs while the right device is current — this is the correct pattern for HP (not the DistributedContext shared-stream hazard described in CLAUDE.md lesson #4, which only applies when contexts are owned by a single thread that creates multiple in reverse order). No action needed.
- **STRIDED ownership for T-MODUP**: not applicable — HP path never invokes T-MODUP.
- **Scale drift across the chained pipeline (CLAUDE.md lesson #7)**: this is the highest-risk finding. The HP path chains `n_layers` layers via `ct = std::move(ct_out)` (lines 588, 715). Inside `run_one_head` the only `scale()` resets are intra-head (lines 159, 171) before the `Q*K^T` and `Attn*V` multiplies. There is **no explicit `ct.scale() = SCALE` reset between layers** before BS#4's output becomes BS#1's input of the next layer. The bootstrapper's output scale is set internally and matches `SCALE` in steady state, so this currently works; however per CLAUDE.md lesson #7 ("ours has scale-mismatch checks commented out … chained Phantom paths must reset scale explicitly before bootstrap"), an explicit `ct.scale() = SCALE` at the layer boundary would harden the chain against silent drift. The MAE gate at `1e-5` is loose enough to mask drift up to ~10⁻⁵ per layer — at 12 layers this could accumulate.
- **Async H→D pinned memory (lesson #1)**: not directly applicable — the HP binary has no explicit `cudaMemcpyAsync` calls. All H→D traffic goes via `GaloisKeyStore` which `cudaHostRegister`s its buffers, so the pinning lesson is satisfied transitively.
- **Stream usage**: each thread calls `cudaSetDevice(g)` and `cudaDeviceSynchronize()` inside `TIME_OP` (line 109). No explicit stream creation in the HP binary itself — work uses the per-device default stream. No stream-lifetime hazard.
- **CLI surface**: `--n-gpus`, `--heads`, `--layers`, `--N`, `--inner`, `--seq-len`, `--hidden`, `--trials`, `--skip-ref` are all parsed (lines 326–344). `--N` validated against `{32768, 65536}` (line 345). `--trials` is parsed (line 338) but **never read** — the binary always does a single inner run and relies on the SLURM script's outer trial loop. This is intentional per the multinode binary's design note but is undocumented and confusing.
- **Argmax integration**: not present in the HP binary. The pipeline stops at post-BS4 and decodes for verification (line 601). No final classifier/argmax. CLAUDE.md lesson #10's vocab≤8192 constraint is not propagated because it is not yet engaged.
- **Findings**:
  - [HIGH] MAE gate is `1e-5`, not the PRD's `2.25e-6` (line 648) — and `--skip-ref` (used in every production SLURM script) bypasses the gate entirely. Recommend: tighten threshold AND introduce a single-layer reference run in the 12-layer SLURM script so the gate is enforced before the timing measurement. Follow-up FIX slice.
  - [HIGH] No explicit `ct.scale() = SCALE` reset between layers (around `ct = std::move(ct_out)` at lines 495 and 588). At 12 chained layers + loose MAE gate, drift could silently accumulate. Per CLAUDE.md lesson #7 this is mandatory.
  - [MEDIUM] Default `--N` is 65536 (line 323), but Goal 2's reference parameter set is `N=32768` (logN=15). The SLURM scripts explicitly pass `--N 32768`, but a developer invoking the binary directly will get the wrong ring degree. Recommend flipping the default.
  - [MEDIUM] `--trials` is parsed but unused (line 338, `(void)n_trials` cast nowhere — actually the variable is read into but never consumed in `bert_hp_multigpu.cu`). Either implement an in-binary trial loop or strip the flag.
  - [LOW] `slurm_bert_hp_n32768.sh` does NOT set any `NCCL_*` env vars. This is fine because the single-node HP binary does no NCCL, but the absence is worth noting if someone later adds DKS to this binary.
  - [LOW] Reference-pass `setup_per_gpu` is called on GPU 0 while the main thread also already created a `PhantomContext ctx0` on GPU 0 (line 406). Two contexts on the same GPU is wasteful but not incorrect. Could be deduped.

### bert_hp_multinode.cu

- **HP MAE gate**: **completely absent**. `skip_ref` is hard-coded to `true` at line 493 and the verification branch at lines 815–820 prints "verification not implemented in multinode binary". This means the 16-GPU multinode result that goes into Goal 2 has zero correctness verification — we rely entirely on the single-node binary's gate (which is itself `1e-5`, not `2.25e-6`, see HIGH above).
- **NCCL setup**: `ncclGetUniqueId` (line 538), `ncclCommInitRank` (line 550), `ncclCommDestroy` (line 828), `cudaStreamCreateWithFlags` (line 549), `cudaStreamDestroy` (line 829). Init and destroy are paired. ncclUniqueId bootstrap via GPFS file (lines 144–192) with atomic rename — correct. No `MPI_*` calls (confirmed by grep). srun uses `--mpi=none` in all SLURM scripts. Clean.
- **Persistent workspace**: not applicable (no DKS).
- **Per-GPU context lifetime**: each rank owns exactly one GPU and exactly one `PhantomContext` in a `unique_ptr` (line 663). The `unique_ptr`s go out of scope at `main` exit while the correct device is current. Correct.
- **STRIDED ownership**: not applicable.
- **Scale drift across the chained pipeline (lesson #7)**: same finding as `bert_hp_multigpu.cu`. Lines 707–724 chain via `ct = std::move(ct_out)` with no explicit `ct.scale() = SCALE` reset between layers.
- **Async H→D pinned memory (lesson #1)**: **VIOLATION**. The NCCL helper functions `nccl_bcast_bytes` (line 212), `nccl_allreduce_doubles` (line 233), `nccl_allreduce_ints` (line 249), and `nccl_barrier` (line 199) all use `cudaMemcpyAsync` with raw host-side `buf` / `&one` (lines 205, 221, 225, 239, 242, 254, 257) that have NOT been `cudaHostRegister`'d. Per CLAUDE.md lesson #1 these silently degrade to synchronous copies. Critically:
  - `nccl_bcast_bytes` is called with the secret-key string buffer (line 627) — a multi-hundred-KB transfer that will run synchronously.
  - The per-rank `cudaMalloc`/`cudaFree` inside each helper (e.g. line 203, 216, 238, 253) is a hot-path malloc on the barrier — violates lesson #2. These run once per setup/per-trial-tail so the absolute cost is small (~µs each), but the pattern itself is exactly what lesson #2 prohibits and would compound if these helpers were ever called per-layer.
- **Stream usage**: a single non-blocking stream `world_stream` (line 549) is created per rank and destroyed at line 829. NCCL helpers all use this stream and `cudaStreamSynchronize` after each collective. Clean.
- **CLI surface**: `--heads`, `--layers`, `--inner`, `--seq-len`, `--hidden`, `--trials`, `--skip-ref`, `--N`, `--bootstrap-dir` all parsed (lines 497–516). No `--n-gpus` — by design, since each rank owns one GPU. `--N` validated. `--trials` parsed but unused (same as multigpu). Goal 2 unit run is not directly expressible because the multinode binary requires `world_size = SLURM_NTASKS`; one would need to allocate 1 rank.
- **Argmax integration**: not present. Same as multigpu.
- **Findings**:
  - [HIGH] No HP MAE gate at all (line 493, 815–820). Multinode results that go into the 16-GPU strong-scaling story have no enforced correctness. Recommend: optionally re-enable a `skip_ref=false` path for rank-0 single-head verification, or run a single-node MAE-gate trial as a precondition for accepting multinode timings.
  - [HIGH] NCCL helpers use `cudaMemcpyAsync` from un-pinned host memory (`bert_hp_multinode.cu:205, 221, 225, 239, 242, 254, 257`). Per CLAUDE.md lesson #1 these are silently synchronous. Most impact is on `nccl_bcast_bytes` for the secret-key broadcast (line 627). Fix: `cudaHostRegister` the buffer before the copy, or stage through a small pinned scratch buffer.
  - [MEDIUM] `cudaMalloc`/`cudaFree` inside every NCCL helper (lines 203/208, 216/228, 238/245, 253/260). Hot-path mallocs are forbidden by lesson #2. Allocate a single `world_scratch` device buffer once at NCCL init and reuse.
  - [MEDIUM] Same scale-drift finding as multigpu — no explicit `ct.scale() = SCALE` reset between layers (line 715).
  - [LOW] `ncclUniqueId` GPFS bootstrap polls every 50 ms with a 60 s timeout (lines 172–192). Adequate for ACC partition, but no error if rank 0 crashes between `write_unique_id` and `unlink`. A successor SLURM job could pick up a stale id with the same JOBID modulus collision (very low probability). Consider a UUID suffix.
  - [LOW] Rank 0 `unlink`s the id file immediately after `ncclCommInitRank` (line 555). If a non-zero rank is slow to read, it may already have read in time, but on a heavily loaded GPFS this is racy. Move the unlink AFTER the first `nccl_barrier`.

## SLURM script findings

All 7 scripts examined: `slurm_bert_hp_n32768.sh`, `slurm_bert_hp_logN15_4node.sh`, `slurm_bert_hp_n32768_4node.sh`, `slurm_bert_hp_smoke.sh`, `slurm_bert_hp_layer.sh`, `slurm_bert_hp_12layer.sh`, `slurm_bert_hp_nccl_smoke.sh`.

Common, correct elements (all 7):
- `export LD_LIBRARY_PATH=/gpfs/projects/etur02/hkanpak/local/lib:$LD_LIBRARY_PATH` (NTL runtime) — present in all 7.
- `cuda/12.8 cmake/3.30.5 nccl/2.24.3-1` module load — present in all 7.
- `#SBATCH --account=etur02 --partition=acc --qos=acc_ehpc` — consistent.

Per-script:

- **`slurm_bert_hp_n32768.sh`** (single-node, 4 GPUs, the Goal 2 strong-scaling-at-4 script):
  - Wired with `--N 32768 --n-gpus 4 --heads 12 --layers 12 --skip-ref`. Three-trial loop. OK.
  - [LOW] `--skip-ref` bypasses MAE gate (see HIGH on multigpu binary).
  - [LOW] No `NCCL_*` env vars (intentional — no NCCL in single-node HP path).

- **`slurm_bert_hp_logN15_4node.sh`** (4 nodes × 4 GPUs = 16 GPUs, the Goal 2 strong-scaling-at-16 script):
  - `#SBATCH --nodes=4 --ntasks-per-node=4 --gres=gpu:4` → 16 ranks. Matches `--ntasks=16 --ntasks-per-node=4` in the srun. Correct.
  - `srun --mpi=none ...` correctly disables MPI launcher.
  - Adds `gcc/11.4.0` to module load (not in single-node script) — needed because NCCL bootstrap path uses C++17 stdlib features built against that gcc.
  - All required NCCL env vars set per `docs/MN5_NCCL_CONFIG.md`: `NCCL_DEBUG=WARN`, `NCCL_IB_DISABLE=0`, `NCCL_IB_HCA=mlx5`, `NCCL_IB_CUDA_SUPPORT=1`, `NCCL_SOCKET_IFNAME=ib0`, `NCCL_P2P_DISABLE=0`, `NCCL_SHM_DISABLE=0`. Correct.
  - `BOOTSTRAP_DIR=/gpfs/projects/etur02/hkanpak/scratch` created with `mkdir -p`. Correct.
  - [LOW] `--skip-ref` mandatory (multinode binary refuses non-skip), so this configuration has zero correctness gate (see HIGH on multinode binary).

- **`slurm_bert_hp_n32768_4node.sh`** (apparently a duplicate of the logN15_4node script with a different job name):
  - Functionally identical to `slurm_bert_hp_logN15_4node.sh` (same module set, same NCCL env, same srun, same 3-trial loop). Time limit is 01:30:00 vs 00:30:00.
  - [LOW] Duplicate scripts — recommend collapsing to one or marking the older one as archived.

- **`slurm_bert_hp_smoke.sh`** (S16 smoke, 4 GPUs / 4 heads / 1 layer):
  - Runs WITHOUT `--skip-ref`, so it actually exercises the MAE gate. This is the one place the gate runs. Good.

- **`slurm_bert_hp_layer.sh`** (S17, 4 GPUs / 12 heads / 1 layer):
  - Also without `--skip-ref`. MAE gate active.

- **`slurm_bert_hp_12layer.sh`** (S18, 4 GPUs / 12 heads / 12 layers):
  - With `--skip-ref` (justified in the header: 12-layer reference would take ~24 min).

- **`slurm_bert_hp_nccl_smoke.sh`** (NCCL wireup smoke, 2 nodes × 2 ranks, 1 layer):
  - `NCCL_DEBUG=INFO` (not WARN) for first-run rail-topology visibility. Correct for smoke purposes.

- **SLURM findings**:
  - [MEDIUM] None of the 16-GPU production scripts enforces any MAE gate. Recommend a "MAE-gate precondition" job: run `slurm_bert_hp_smoke.sh` (or a single-layer 1-rank multinode run) and fail the pipeline if it doesn't pass before kicking off the 16-GPU timing.
  - [LOW] `slurm_bert_hp_n32768_4node.sh` and `slurm_bert_hp_logN15_4node.sh` are near-duplicates.

## Goal 2 unit-run readiness

Can we run `bert_hp_multigpu --n-gpus 1 --heads 1 --layers 2 --N 32768` today?

**YES — it should run.** The CLI parses all four flags, and the binary code path with `n_gpus = 1, n_heads = 1, n_layers = 2` is exercised inside the same `for (int g = 0; g < n_gpus; g++)` thread loop and the `for (int layer = 0; layer < n_layers; ++layer)` inner loop. The reference pass will also run (skip_ref=false by default), producing the MAE check at the `1e-5` threshold.

Blockers / caveats for MEASURE-01:
1. The MAE threshold is `1e-5` — measurably weaker than the PRD's `2.25e-6`. If MEASURE-01 is to formally cite the gate, either tighten the threshold or document the deviation.
2. There is no SLURM wrapper for the unit run — no `slurm_bert_hp_unit_logN15.sh` or equivalent. The closest is `slurm_bert_hp_smoke.sh`, which runs `--heads 4` and 1 layer (not the unit run). Recommend writing a tiny SLURM script.
3. Saturation check (`time(layer 1) ≈ time(layer 2) within 5%`) is implicit: the binary prints per-layer compute via the `[GPU %d] head %d layer %d/%d done` lines (line 590), but no automatic ≤5% saturation assertion is emitted. Manual postprocess.

None of these are BLOCKERs — they are tightening / packaging items.

## Recommended follow-up FIX slices

- **FIX-BUG-02-01 (HIGH)**: Tighten HP MAE threshold from `1e-5` to `2.25e-6` in `bert_hp_multigpu.cu:648` and document the alignment with the PRD's Goal 2 spec.
- **FIX-BUG-02-02 (HIGH)**: Add explicit `ct.scale() = SCALE` reset at the layer boundary in both `bert_hp_multigpu.cu` (line 588) and `bert_hp_multinode.cu` (line 715), per CLAUDE.md lesson #7.
- **FIX-BUG-02-03 (HIGH)**: In `bert_hp_multinode.cu`, `cudaHostRegister` (or use pinned scratch) the host buffers passed to `cudaMemcpyAsync` inside `nccl_bcast_bytes`, `nccl_allreduce_doubles`, `nccl_allreduce_ints`, and `nccl_barrier`. Per CLAUDE.md lesson #1.
- **FIX-BUG-02-04 (MEDIUM)**: Replace per-collective `cudaMalloc`/`cudaFree` in NCCL helpers with a single persistent device scratch buffer allocated once at NCCL init. Per CLAUDE.md lesson #2.
- **FIX-BUG-02-05 (MEDIUM)**: Add a rank-0 single-head MAE gate to `bert_hp_multinode.cu` (run a reference head on rank 0 before the world compute begins) so the 16-GPU result carries some correctness signal.
- **FIX-BUG-02-06 (MEDIUM)**: Flip default `ring_N` to `32768` in `bert_hp_multigpu.cu:323` to match Goal 2; or refuse to run without an explicit `--N`.
- **FIX-BUG-02-07 (MEDIUM)**: Either implement the in-binary `--trials` loop or strip the unused flag from both HP binaries.
- **FIX-BUG-02-08 (LOW)**: Add `slurm_bert_hp_unit_logN15.sh` that wraps the Goal 2 unit run (`--n-gpus 1 --heads 1 --layers 2 --N 32768`) including a built-in saturation assertion (time(layer 1) vs time(layer 2) within 5%).
- **FIX-BUG-02-09 (LOW)**: Deduplicate `slurm_bert_hp_n32768_4node.sh` vs `slurm_bert_hp_logN15_4node.sh`.
- **FIX-BUG-02-10 (LOW)**: In `bert_hp_multinode.cu`, move the `unlink(id_path)` (line 555) AFTER the first `nccl_barrier` so slow ranks can't miss the file under GPFS contention.
- **FIX-BUG-02-11 (LOW)**: Remove the redundant GPU 0 `PhantomContext ctx0` in `bert_hp_multigpu.cu:406` since `setup_per_gpu` builds another GPU 0 context for the reference pass.
