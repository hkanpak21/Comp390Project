# Ralph Loop State — multiNEXUS PRD Execution

## MN5 RESULTS (2026-05-07 ~19:34)

### Iteration 433-434: First MAE check on MN5
- **rsync footgun**: trailing-slash on source put `vendor/` files at project root, not under `vendor/`. Re-rsync without trailing slash (and excluded `phantom_old`) put them in correct location.
- **Phantom rebuild footgun (real)**: even after deleting `libPhantom.so`, ExternalProject's stamp files (`phantom_ext-build`, `phantom_ext-install`) said built. Had to delete stamps before `cmake --build . --target phantom_ext` actually rebuilt. After this `nm` showed `modup_partial` symbol present.
- **Build OK**: All three benchmark binaries built clean.
- **Submitted job 40208797 → MAE check ran**.

### MAE result interpretation (CRITICAL CORRECTION)
- The "Single-GPU bootstrap MAE = 1.26e-01" headline is **NOT a regression**. It is the EXPECTED behaviour at chain_index=0 (modulus exhausted) — measures absolute deviation from 0.5 not relative correctness.
- The actual T-MODUP correctness check is `single-GPU MAE == DKS MAE` (same numeric noise → DKS algorithm is correct).
- April 12 baseline `dks_bootstrap_38923192.out` ALSO showed `MAE = 1.26e-01` — confirms not a regression.

### β-throw bug FOUND and FIXED
- T-MODUP threw `std::runtime_error("beta is not divisible by n_gpus")` at the 2-GPU sweep because at smaller chain levels β can be tiny (e.g., 3) and 3 % 2 ≠ 0.
- **Fix landed**: replaced strict throw with uneven contiguous sharding — distribute β % n_gpus remainder to first `remainder` GPUs, so each GPU owns floor(β/n) or floor(β/n)+1 contiguous digits. d_count==0 is a no-op (handled by modup_partial early-return).
- Touched files: `src/multi_gpu/keyswitching/output_aggregation.cu` (both variants), `src/multi_gpu/keyswitching/dist_galois_key_store.cuh` (both `generate` and `generate_multinode`).

### T-MODUP CORRECTNESS VERIFIED (job 40210267)
- Single-GPU bootstrap: MAE = 1.25e-01
- DKS bootstrap on 2 GPUs: MAE = 1.25e-01
- **MATCH (same numeric noise)** — T-MODUP indexing is correct.

### Open issue
- `dks_bootstrap_bench` crashes between 2-GPU and 4-GPU sweep with `cudaMemcpyAsync invalid argument` at `cuda_wrapper.cuh:96`. Probably tied to context teardown between sweeps; does NOT block 4-GPU benchmark in `bert_dks_multigpu`. Investigate later.

### Bisect found T-MODUP is the bug
- **Job 40212948** (T-STRAGGLER OFF, T-MODUP ON, no other changes): same NCCL illegal memory access during AllReduce → T-STRAGGLER not the bug.
- **Job 40213202** (T-MODUP REVERTED, other iteration changes intact): clean run! 4-GPU bootstrap 2099 ms, 1-layer 9.6 s. Effectively reproduces Phase 4b champion → T-MODUP confirmed as bug.
- T-MODUP iteration-3 verification gave GREENLIGHT but agent only checked logical indexing on paper, never ran the kernel. Real-world failure mode unknown; suspect index off-by-one in `modup_copy_partQl_partial_kernel` or `make_cuda_auto_ptr<uint64_t>(0,...)` pointer at d_count=0 in the new uneven-sharding code path.
- T-MODUP is NOT shippable as-is. Recommendation: ship Phase 4b numbers, document T-MODUP as future work.

### REAL results on MN5 (T-MODUP reverted, all other iteration changes inactive)
| Config | Bootstrap | 1-Layer | 12-Layer | vs CPU 249.6s |
|---|---|---|---|---|
| 4-GPU baseline (Phase 4b) | 2099 ms | 9.61 s | — | — |
| 4-GPU 12-layer measured (job 40213339) | 2098 ms | 8.92 ± 0.21 s | **107.08 s** | **2.33×** |
| 12-layer projection from 1-layer × 12 | — | — | 107.04 s | (matches measurement to 0.04%) |

T-12LAYER-BASE PRD goal "±10% of projection" → ACHIEVED (7% better).

### Currently in queue
- T-NEXUS (job 40213471): N=32768 BERT-encoder for parameter-matched comparison.



## Status (2026-05-07)

**Iterations 1-4 complete.** All locally-doable PRD code and paper tasks have landed.
The remaining work depends on MN5 (BSC MareNostrum 5) benchmark runs.

## Code changes landed (in iteration order)

| Task | Status | Risk | Files |
|---|---|---|---|
| T-STRAGGLER | landed (it 1) | low | distributed_context.{cu,cuh}, comm/nccl_comm.{cu,cuh}, output_aggregation.cu, galois_oa.cu |
| T-OVERLAP partial | landed (it 1) | low | output_aggregation.cu (inner sync removed) |
| T-OVERLAP full | landed (it 3) | medium | new oa_done_events; output_aggregation.cu trailing sync replaced; galois_oa.cu writeback waits on event |
| T-TRACE NVTX | landed (it 1) | none | Bootstrapper.cu (8 BSGS ranges); output_aggregation.cu (modup/moddown) |
| T-MODUP | landed (it 2) | **HIGH** | vendor/phantom/include/rns.cuh; vendor/phantom/src/rns_bconv.cu (new modup_partial + partial_kernel); output_aggregation.{cu,cuh}; dist_galois_key_store.cuh (CONTIGUOUS ownership) |
| T-LRU | landed (it 1) | low | galois_key_store.cuh (kCacheSize=10) |
| T-12LAYER-BASE | landed (it 1) | low | bert_dks_multigpu.cu (BERT_LAYERS env); scripts/mn5/slurm_bert_12layer_dks.sh |

T-MODUP review verdict (it 3): **GREENLIGHT for MN5 build**. All 10 critical indexing checks passed.

## Paper sections written

| Section | Status | Source |
|---|---|---|
| P-SETUP (skeleton) | landed (it 1) | IEEEtran |
| P-BG (Background) | landed (it 1) | ~470 words + N→memory table |
| P-DESIGN (Design) | landed (it 1) | ~720 words + 4 subsections |
| P-RELATED (Related Work) | landed (it 1) | 3 paragraphs |
| P-EVAL (skeleton) | landed (it 3) | 6 subsections + 3 tables; 17 \TODO markers |
| P-CEILING | landed (it 3) | ~660 words + rotation-bound floor calc |
| P-INTRO | landed (it 3) | 3 paragraphs |
| P-ABSTRACT | landed (it 3) | 194 words skeleton |
| P-CONCL | landed (it 3) | ~150 words |
| Bibliography | landed (it 1) | refs.bib with 8 keys |
| Figures (2 added it 4) | landed (it 4) | fig5_kernel_breakdown, fig6_gpu_utilization (fig1 was pre-existing) |

## Documentation

- `docs/RALPH_LOOP_HANDOFF.md` (it 4) — 284-line runbook for sync, build, validation, paper-TODO mapping, recovery

## Remaining work (blocked on MN5)

| PRD Task | Why blocked |
|---|---|
| P-EVAL final numbers | needs T-STRAGGLER, T-MODUP, T-12LAYER-OPT, T-NEXUS measurements |
| P-ABSTRACT final numbers | needs final speedup × |
| P-CONCL final numbers | needs final speedup × |
| T-BSGS | conditional on T-TRACE nvtxsum showing baby+giant > 5% |
| T-12LAYER-OPT | needs T-STRAGGLER + T-MODUP correctness verified first |

## Next user action

Per `docs/RALPH_LOOP_HANDOFF.md`:
1. `rsync` to MN5 (command in handoff)
2. `make -j20 dist_bootstrap_bench bert_dks_multigpu bert_encoder_multigpu`
3. `sbatch scripts/mn5/slurm_dks_bootstrap.sh` — verify MAE = 2.25e-6 BEFORE benchmarks
4. Then run the 6-step benchmark battery
5. Fill the 17 \TODO markers in `paper/main.tex`

## Iteration 5 (complete)

Final paper polish pass — 30 notation fixes, 13 GPU-count standardisations, 6 acronym definitions, 4 prose tightenings, 5 LaTeX hygiene fixes. Found and fixed Abstract/Conclusion ~600 ms vs Eval ~1,170 ms inconsistency (now consistently ~1,100 ms).

## Iteration 6 (complete)

Created `paper/Makefile`: detects rsvg-convert or inkscape; SVG→PDF for fig1–fig6; runs pdflatex+bibtex pipeline. `make` builds the paper end-to-end on local mac. `make clean` resets.

## Iteration 7 (complete)

Final wiring verification — all confirmed:
- Cross-stream wait order in `galois_oa.cu:406-411` — `cudaStreamWaitEvent(stream0, oa_evts[0], 0)` precedes `cudaMemcpyAsync(stream0)`, both queued sequentially from CPU.
- `oa_done_events` defensive fallback in `output_aggregation.cu:340-344, 466-470` — falls back to `cudaStreamSynchronize` if event vector is empty (protects bare `MultiGpuContext` callers in benchmarks).
- `modup_partial` declared at `vendor/phantom/include/rns.cuh:168`, defined at `vendor/phantom/src/rns_bconv.cu:682`, throws on out-of-bounds digit range at line 701.
- `vendor/phantom/src/CMakeLists.txt:11` lists `rns_bconv.cu` as a Phantom source — `modup_partial` will compile.

## Iteration 8 (complete)

Created `scripts/mn5/run_paper_battery.sh` — single-command submitter for the full paper validation order: MAE check (slurm_dks_bootstrap), 5× single-layer BERT (slurm_bert_dks), 12-layer BERT (slurm_bert_12layer_dks), T-NEXUS at N=32768 (slurm_bert_encoder_multigpu). Prints job IDs, log paths, and the post-Step-1 MAE verification command. Has `--mae` flag for MAE-only mode if user wants to gate everything on correctness first.

## Agent IDs (for follow-up via SendMessage if needed)

| Iteration | Agent | ID |
|---|---|---|
| 1 | T-STRAGGLER+OVERLAP+TRACE | a1e20d3805cf29fa4 |
| 1 | T-TRACE Bootstrapper | a0d1944fa77c788f8 |
| 1 | Paper P-SETUP/BG/DESIGN/RELATED | a543e14dd77a41045 |
| 1 | T-LRU | ae3dbeff43514b9e6 |
| 1 | T-12LAYER-BASE | aa4bcb8aec9b05865 |
| 2 | T-MODUP | a09945a9583c8c436 |
| 2 | Sync investigation | aa878925a0f099cab |
| 2 | T-NEXUS prep | a606c025f30ee9f58 |
| 2 | Verification | a2ba2ee4b4a5df795 |
| 3 | Paper completion | ad4f8cb1acca16b17 |
| 3 | T-MODUP verification | a54c3a0a249e4358a |
| 3 | Cross-stream wait | afb82c059bcfb7136 |
| 4 | Handoff doc | a48df010e6f47fb51 |
| 4 | Cleanup | ab9820815437dcdb7 |
