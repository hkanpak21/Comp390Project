---
title: "PRD — multiNEXUS Paper: Per-Op Multi-GPU Typology + End-to-End at uniform logN=15"
status: in-progress
owner: hkanpak21
created: 2026-05-11
last_updated: 2026-05-11 (evening)
labels: [in-progress, paper, multi-gpu, fhe, ralph-pickup]
branch: paper/multinexus
---

> **Status (2026-05-11 evening):** 25 of ~33 slices have committed work; first FIX
> commit landed (`FIX-BUG-04-01` removes Bootstrapper debug syncs); both
> `MODULE-02` and `MODULE-03` from Module Sketches now exist with tests; assembled
> `paper/paper.md` is 21,334 words across 10 sections. Critical gap: **no
> PROFILE-NN or MEASURE-NN job has actually run on MN5 yet**, so 8 `[TODO:
> confirm with PROFILE-NN trace]` markers in §6 and the entire numerical body of
> §7 are still placeholders. See "Execution Status" and "Pending slices for the
> Ralph loop" below.

# PRD: multiNEXUS Paper — Per-Op Multi-GPU Typology + End-to-End at uniform logN=15

## Problem Statement

Halil has spent the semester building multi-GPU FHE infrastructure on top of NEXUS (Zhang et al., NDSS 2025) and accumulated measurements but no coherent paper-ready story. Two specific contributions need to be packaged into a paper deliverable for submission to the PI:

1. We can *identify* NEXUS's per-operation kernels on H100 (build their code from source on our hardware, run their own benchmarks) and demonstrate clean multi-GPU acceleration with provable parallelization efficiency on each operation.
2. We can run *real end-to-end BERT inference* at a chosen logN, using a small unit measurement plus a saturation check to honestly extrapolate to full BERT, then show parallelization performance under both head-parallel (latency, strong scaling) and data-parallel (throughput, weak scaling).

The current state has the per-op data in `docs/PER_OP_VS_NEXUS.md` §4.4 but is missing:
- Profiling-grounded ceiling explanations for each operation (most ops have only stdout, not nsys traces, so we cannot say *why* the ceiling is where it is).
- A small-unit end-to-end measurement at uniform logN=15 with explicit saturation verification.
- Throughput (data-parallel weak scaling) measurements for the chained pipeline at 4 and 16 GPUs.
- A bug-audit pass on the critical-path code so paper claims rest on bug-free binaries.
- A coherent paper structure that walks the reader from "we identified NEXUS" through "we beat it on multi-GPU" to "we can do end-to-end" without leaving holes.

Without these, we have a collection of measurements but not a paper.

## Solution

Produce a paper-ready writeup with two evaluation sections plus an appendix:

- **Goal 1**: per-operation multi-GPU typology, baselined against NEXUS-on-H100 single-GPU. Six op subsections, each following a 6-field template (aim / strategy / implementation / result / profiling-grounded explanation / profiling-grounded ceiling). Operations are grouped by parallelization regime (compute-parallel / data-parallel-throughput / transitional) so the heterogeneity becomes the story rather than a caveat.
- **Goal 2**: end-to-end BERT inference at uniform logN=15. Demonstrated via a 1-head × 2-layer unit measurement on 1 GPU, with explicit saturation check (layer 1 timing ≈ layer 2 timing). Extrapolated to full BERT by multiplication. Then shown under both head-parallel (HP-BERT, strong scaling, per-inference latency) and data-parallel (G concurrent independent inferences, weak scaling, throughput).
- **Appendix A**: every modification we made to NEXUS and Phantom this semester, plus the bug-fix log, structured for the PI to walk through.

The work is divided into vertical slices. Each slice produces one self-contained measurement or document update plus exactly one commit. Slices declare their upstream dependencies in commit message footers so the dependency graph is reconstructable from `git log`.

## User Stories

1. As Halil, I want a paper-ready writeup of the semester's multi-GPU FHE work, so that I can submit it to my PI for review.
2. As Halil, I want every per-op result claim in the paper to be backed by a profiling trace, so that the parallelization-efficiency argument is defensible to a reviewer.
3. As Halil, I want the PI to see how I built NEXUS on H100 and ran their own benchmarks, so that the comparison column eliminates A100→H100 hardware-uplift confusion.
4. As Halil, I want to demonstrate that our framework can chain BERT operations end-to-end at uniform logN=15, so that we can claim a deliverable NEXUS's open source cannot produce.
5. As Halil, I want a small "1 head × 2 layers" measurement with saturation verification, so that I can multiply up to project full BERT performance honestly without paying the wall-clock cost of running 12 layers × 12 heads from scratch on a single GPU.
6. As Halil, I want both strong-scaling (HP-BERT, latency) and weak-scaling (data-parallel, throughput) results for the end-to-end pipeline, so that I cover both parallelization regimes a reviewer might ask about.
7. As Halil, I want a critical-path bug audit before paper writing begins, so that the paper's claims rest on bug-free code rather than on numbers from a binary with a latent issue.
8. As Halil, I want all NEXUS/Phantom modifications documented in an appendix, so that the PI can see exactly what was changed and re-derive the modifications from the appendix alone.
9. As Halil, I want each piece of work to land as a meaningful, well-written commit, so that I can review the development trajectory from `git log` and explain it during paper review.
10. As Halil, I want vertical job slicing with explicit upstream-dependency declarations in commit footers, so that I can track which slices block which downstream slices and identify the critical path.
11. As Halil, I want CLAUDE.md to reflect the current paper plan, so that fresh agents starting work on the repo have correct context and do not waste time re-deriving outdated framings.
12. As Halil, I want MD files that contradict the current plan (older PI presentation drafts) to be archived rather than left in active docs, so that fresh agents do not act on outdated context.
13. As an agent picking up a slice, I want to see the upstream-dependency status of my slice declared in the commit footer of the slice that blocked me, so that I know whether I'm blocked or unblocked.
14. As an agent committing work, I want a clear convention for commit messages (subject ≤ 72 chars, imperative mood, body explains why-then-what-then-where, footer lists upstream slice IDs), so that the history reads cleanly.
15. As Halil, I want each per-op subsection of the paper to follow the same 6-field template, so that the typology comparison is structurally consistent across operations and the reader can scan easily.
16. As Halil, I want the per-op typology to group operations into compute-parallel / data-parallel-throughput / transitional buckets, so that the heterogeneity story is the paper's strength rather than a caveat about Bootstrap and Softmax not scaling well.
17. As Halil, I want the saturation check (layer 1 ≈ layer 2 within stated tolerance) explicitly verified and reported in the paper, so that the extrapolation from 2 layers to 12 has a stated validity argument rather than being asserted.
18. As Halil, I want the multi-cipher argmax gap honestly disclosed in the discussion section, so that reviewers cannot accuse us of hiding a limitation that is in fact a real architectural constraint.
19. As Halil, I want every measurement in the paper reproducible from a SLURM script in `scripts/mn5/`, so that anyone with MN5 access can rerun any number from the paper.
20. As Halil, I want a JOBID + log path mapping for every number that appears in the paper, so that any number can be traced to its raw output for review or debug.
21. As Halil, I want the appendix to include the patch summaries for `vendor/phantom/` (≈95 lines) and `vendor/nexus/` modifications, so that the PI can see the surface area of upstream changes at a glance.
22. As Halil, I want the bug-audit output to be a structured checklist (one row per critical-path component), so that I can see what was checked, what passed, what failed, and what was fixed.
23. As an agent reviewing CLAUDE.md, I want the file to declare the paper plan, the 9-section structure, the per-op 6-field template, and the slice convention up front, so that I have the operational context within the first paragraph.

## Implementation Decisions

**Paper structure (9 sections + Appendix A):**
1. Abstract
2. Introduction (problem, two contributions, multi-GPU FHE landscape gap)
3. Background (CKKS basics, RNS-CKKS bootstrap, NEXUS, prior multi-GPU FHE: Cerium, Cinnamon)
4. Identifying NEXUS on H100 (build from source, per-op measurement, the fair-comparison column)
5. Multi-GPU strategies (DKS, head-parallel, data-parallel-per-op — explained, no measurements yet)
6. Goal 1 — Per-op multi-GPU typology (six op subsections, 6-field template each)
7. Goal 2 — End-to-end at uniform logN=15 (unit + saturation + extrapolation + HP-BERT strong scaling + data-parallel weak scaling)
8. Discussion (multi-cipher argmax gap, no SIMD packing, position vs Cerium)
9. Conclusion + future work
10. Appendix A — NEXUS/Phantom modifications + bug-fix log

**Per-op 6-field template (Goal 1):**
For each of Bootstrap, MatMul, GELU, LayerNorm, Softmax, Argmax, the paper subsection contains:
1. Aim — what this operation does in BERT and what we want to measure.
2. Parallelization strategy — what kind (data-parallel, output-channel split, etc.) and why this kind suits this op.
3. Implementation — high-level wiring (no code).
4. Result — measured single-GPU + 4-GPU + 16-GPU per-call latency + speedup vs single-GPU.
5. Profiling-grounded explanation — what nsys/NCU shows that justifies the result (e.g. "kernel utilization is 92% so compute-bound; output-channel split distributes that compute evenly").
6. Profiling-grounded ceiling — what the trace tells us about why we cannot push further (e.g. "context-setup time per rank dominates for ops with per-call compute under 100 ms").

**Goal 2 measurement methodology:**
- Unit run: 1-head × 2-layer chained pipeline at uniform logN=15 on 1 GPU.
- Saturation check: time(layer 1) ≈ time(layer 2) within stated tolerance (5% recommended).
- Extrapolation: full BERT projected time = 12 × 12 × per-head-per-layer time.
- Strong scaling (latency): HP-BERT at 4, 16 GPUs vs the extrapolated 1-GPU baseline.
- Weak scaling (throughput): G concurrent independent single-GPU inferences vs 1 inference time.
- All numbers in this section reproducible from one SLURM script per measurement.

**Bug audit scope:**
Critical-path code only — six per-op alignment binaries, `bert_hp_multigpu` and `bert_hp_multinode`, the multi-GPU framework layer (key sharding, distributed context, output aggregation), and the NEXUS evaluator wrappers. Audit deliverables:
- Reading each binary's correctness gate (MAE thresholds vs single-GPU reference).
- Running smoke tests on each.
- Scanning for known-suspect patterns: silent scale checks, modulus chain depth, NCCL digit-shard ownership, per-call memory allocation in hot paths, missing `LD_LIBRARY_PATH` in SLURM scripts.
- Validating that current numbers in `docs/PER_OP_VS_NEXUS.md` §4 are reproducible from current source.

**Vertical slice convention:**
Each slice is one focused unit of work that produces exactly one commit. Format:

- **Slice ID format**: `<phase>-<NN>` (e.g., `BUG-01`, `PROFILE-01`, `MEASURE-03`, `WRITE-S6`, `DOC-01`).
- **Phase vocabulary**: `BUG` (bug audit + fix), `PROFILE` (nsys/NCU generation), `MEASURE` (new measurements), `WRITE` (paper sections), `DOC` (CLAUDE / README / MD updates), `APPENDIX` (Appendix A content).
- **Commit message format**:
  - Subject: `<phase>(<area>): <imperative summary, ≤72 chars>`
  - Body: motivation paragraph, then technical detail, then JOBID/log references, then upstream-dependency footer
  - Footer line: `Slice: <slice-id>; Depends-on: <upstream-slice-ids comma-separated, or "none">`
- **One slice per commit** — never bundle two slices into one commit, even if both are small.

**Profiling artifact organization:**
All nsys traces live under `experiments/results/<date>_h100x<G>_<op>-mgpu-nsys/raw/` (matches existing pattern). Each `.nsys-rep` has accompanying NVTX summary, GPU-sum, stdout, and stderr files. The paper's Section 6 references the trace by JOBID; the appendix lists JOBID → trace path mapping.

**MD file disposition:**
- Keep in active `docs/`: `PI_REPORT.md`, `PER_OP_VS_NEXUS.md`, `HPC_PRIMER.md`, `MN5_NCCL_CONFIG.md`, `NSIGHT_GUIDE.md`.
- Keep in active root / `paper/`: `README.md`, `CLAUDE.md`, `paper/architecture_guide.md` (already has legacy-framing note).
- Move to `docs/archive/`: `docs/PI_PRESENTATION.md` and `docs/PI_PRESENTATION_SLIDES.md` — both predate this PRD and contain framings (per-op-only, slide outlines tied to a different evaluation structure) that would mislead a fresh agent.
- Update `CLAUDE.md` to declare the paper plan, the 9-section structure, the appendix, the 6-field per-op template, and the vertical-slice + commit conventions, all up front.

## Module Sketches

The following are the deeper testable modules surfaced by this PRD. Each is a candidate for a small, focused implementation that sits behind a stable interface.

1. **Saturation analyzer.** Pure function: given two ordered timing measurements (e.g. `t_layer_1`, `t_layer_2`), returns `{saturated: bool, relative_delta: float, threshold: float}`. Stable interface, easy to test with synthetic timing inputs; will be used by Goal 2 to justify the extrapolation. Recommended for tests.
2. **Per-op result aggregator.** Pure function on file inputs: given a JOBID and a `experiments/results/<dir>/raw/` path, returns a structured per-op record `{op, gpus, per_call_ms, throughput_inferences_per_s, mae, jobid, log_path, nvtx_breakdown}`. Used by the paper-table generation. Recommended for tests with checked-in fixture files.
3. **Vertical-slice dependency tracker.** Given a list of slice declarations (parsed from commit message footers), returns the topological order, identifies blocked slices, and renders a dependency graph. Used by Halil to track progress. Recommended for tests on a small synthetic slice graph.
4. **Throughput driver script.** SLURM-level orchestrator: launches G concurrent single-GPU `bert_hp_multigpu --n-gpus 1` inferences and reports aggregate throughput. Stable interface (just `--n-instances G`); not pure but deterministic on identical input. Tests not required if the per-instance binary is already gated.

The first three are pure functions and easy to slot into `scripts/regression/` style. The fourth is operational tooling.

## Testing Decisions

A good test for this project tests the *external behavior* of measurement infrastructure, not implementation details. For paper-writing work, "testing" mostly means correctness gates on the binaries that produce paper numbers:
- Each per-op alignment binary must report MAE under its threshold (typically ≤ 1e-5 vs single-GPU reference; ≤ 5% relative for MatMul).
- HP-BERT must produce a layer-2 output that matches the single-GPU reference within MAE ≤ 2.25e-6.
- The data-parallel throughput driver must produce identical per-rank outputs (cross-rank MAE = 0) since each rank runs an independent identical input.

**Modules to test (recommended):**
- Saturation analyzer (pure function on timing inputs).
- Per-op result aggregator (pure function on file fixtures).
- Vertical-slice dependency tracker (pure function on slice graph).

**Modules NOT to test:**
- The per-op alignment binaries themselves — already gated by MAE thresholds at runtime.
- The multi-GPU framework — already exercised by `phantom_threadsafe_smoke.cu` and `multi_gpu_keyswitch_test.cu`.

**Prior art:** `scripts/regression/run_correctness.sh` is the existing pattern for MAE-gated regression tests. New testable modules should slot into this harness with their fixtures under `scripts/regression/fixtures/`.

## Out of Scope

Explicitly NOT in this PRD:
- **Slot-axis SIMD packing for HP-BERT.** Required to beat NEXUS's published 37.3 s end-to-end on a fair workload. Multi-day refactor of the chain. Listed as "What is left" item 1; not in this paper.
- **Multi-cipher argmax tournament.** Required for an apples-to-apples vs NEXUS's published vocab=30,522 number. Disclosed as a limitation in Section 8 of the paper; not implemented here.
- **Per-rank context pooling.** Would lift small-op 16-GPU efficiency from 9–22% to ~30–50%. Out of scope; current efficiency reported honestly with profiling-grounded justification.
- **Layer-pipeline parallelism.** Different parallelization scheme (different layers on different GPUs). Not implemented; not measured. Mentioned only as alternative future work.
- **CPU baseline rerun.** The historical 249 s CPU streaming baseline is not re-measured.
- **Cerium head-to-head numbers.** Cerium code is not public; comparison is qualitative.
- **HP-LLaMA results.** This paper focuses on BERT. LLaMA work is out of scope for this PRD.
- **End-to-end inference at logN=13 or logN=16.** End-to-end demo is at uniform logN=15 only. logN=13 doesn't fit chain depth without bootstrap (and NEXUS doesn't ship a logN=13 bootstrap); logN=16 is HP-BERT's already-measured setting and was deprioritized in favor of the smaller logN=15 demo.

## Further Notes

### Vertical slice map (with status as of 2026-05-11 evening)

Phase order is: BUG → FIX → PROFILE / MEASURE → WRITE → APPENDIX → MODULE/DOC. DOC slices have no upstream and can run any time. Status legend: ✅ done with commit, 🟡 partial (script committed, job not yet run on MN5), ❌ not started, 📋 placeholder (TODOs to backfill once data lands).

| Slice ID | Description | Depends on | Status | Commit |
|---|---|---|---|---|
| `DOC-01` | Update CLAUDE.md with paper plan, 9-section structure, per-op template, slice convention | none | ✅ | 8e04b14 |
| `DOC-02` | Archive `docs/PI_PRESENTATION.md` and `docs/PI_PRESENTATION_SLIDES.md` | none | ✅ | 8e04b14 |
| `DOC-03` | Publish this PRD | none | ✅ | 8e04b14 |
| `DOC-04` | Assemble `paper/paper.md` from section drafts | WRITE-S1..S9, WRITE-Appendix | ✅ | 2a053fd |
| `BUG-01` | Bug audit pass on six align binaries | none | ✅ | 19b07c1 |
| `BUG-02` | Bug audit pass on `bert_hp_multigpu` + `bert_hp_multinode` | none | ✅ | 4ec55b0 |
| `BUG-03` | Bug audit pass on `src/multi_gpu/` framework | none | ✅ | f2c9087 |
| `BUG-04` | Bug audit pass on `src/nexus_eval/` evaluator wrappers | none | ✅ | 877c32a |
| `FIX-BUG-04-01` | Remove debug fprintf+sync from Bootstrapper hot path | BUG-04 | ✅ | 7bb9bf3 |
| `FIX-BUG-04-02` | SCALE-CROSS-CUT: add `cipher.scale() = SCALE` reset before bootstrap at HP-BERT chained call sites (mirrors `argmax_align_n32k.cu:225`) | BUG-04 | ❌ | — |
| `FIX-BUG-04-03` | BOOT-RAW-OWN: give `Bootstrapper` an explicit destructor for the raw-`new` `ModularReducer*` (Rule of Five, lesson #3) | BUG-04 | ❌ | — |
| `FIX-BUG-04-04` | MATMUL-NEW-PER-CALL: hoist `new uint64_t[…]` + `cudaMemcpyAsync` out of `multiply_power_of_x` into a persistent staging buffer | BUG-04 | ❌ | — |
| `FIX-BUG-01-01` | Add MAE gates to all six single-GPU align binaries (currently none gate) | BUG-01 | ❌ | — |
| `FIX-BUG-02-01` | Tighten HP-BERT MAE gate to PRD `2.25e-6` target; add gate to multinode binary | BUG-02 | ❌ | — |
| `FIX-BUG-03-01` | Resolve two HIGH cleanup-order risks in `DistributedContext::destroy()` | BUG-03 | ❌ | — |
| `PROFILE-01` | 4-GPU nsys for matmul_align_n8k | BUG-01 (matmul), FIX-BUG-04-01 | 🟡 ae54775 (script only — not submitted) | — |
| `PROFILE-02` | 4-GPU nsys for gelu_mgpu_align | BUG-01 (gelu), FIX-BUG-04-01 | 🟡 867e13f (script only — not submitted) | — |
| `PROFILE-03` | 4-GPU nsys for softmax_mgpu_align | BUG-01 (softmax), FIX-BUG-04-01 | ❌ **script not yet written** | — |
| `PROFILE-04` | 4-GPU nsys for argmax_align_n32k | BUG-01 (argmax), FIX-BUG-04-01 | 🟡 cce11c9 (script only — not submitted) | — |
| `MEASURE-01` | Goal 2 unit run: `bert_hp_multigpu --n-gpus 1 --heads 1 --layers 2 --N 32768` | BUG-02, FIX-BUG-04-01, FIX-BUG-04-02 | 🟡 e7e8e5c (script only — not submitted) | — |
| `MEASURE-02` | Saturation analyzer (Module Sketch #1) + tests | none (pure module) | ✅ | 43c1753 |
| `MEASURE-03` | Goal 2 data-parallel throughput at 4-GPU | MEASURE-01, BUG-02 | 🟡 11a6f4e (script only — not submitted) | — |
| `MEASURE-04` | Goal 2 data-parallel throughput at 16-GPU | MEASURE-03 | 🟡 8d400da (script only — not submitted) | — |
| `MODULE-02` | Per-op result aggregator (Module Sketch #2) + tests | none (pure module) | ✅ | 1f702e6 |
| `MODULE-03` | Vertical-slice dependency tracker (Module Sketch #3) + tests | none (pure module) | ✅ | 1f52771 |
| `WRITE-S1` | Section 1 (Abstract) | WRITE-S6, WRITE-S7 (numerical anchors) | ✅ | 1cb494b — but **needs refresh after backfills** |
| `WRITE-S2` | Section 2 (Introduction) | DOC-01 | ✅ | 9e12c0a |
| `WRITE-S3` | Section 3 (Background) | DOC-01 | ✅ | 93fe523 |
| `WRITE-S4` | Section 4 (Identifying NEXUS on H100) | none (existing data) | ✅ | c3af0ce |
| `WRITE-S5` | Section 5 (Multi-GPU strategies — DKS, HP, DP) | none (architectural) | ✅ | 41022ce |
| `WRITE-S6` | Section 6 (Goal 1 per-op typology) — six op subsections | PROFILE-01..04 + BUG-01 | 📋 93d2b71 — **8 `[TODO: confirm with PROFILE-NN trace]` markers awaiting backfill** |
| `WRITE-S7` | Section 7 (Goal 2 end-to-end) | MEASURE-01..04 | 📋 e3132c1 — **skeleton only, numerical cells empty** |
| `WRITE-S8` | Section 8 (Discussion) | WRITE-S6, WRITE-S7 | ✅ | 5e9b9fd |
| `WRITE-S9` | Section 9 (Conclusion + future work) | WRITE-S8 | ✅ | 6205489 |
| `WRITE-Appendix` | Appendix A (NEXUS/Phantom mods + bug-fix log) | BUG-01..04, FIX-* | ✅ b494963 — **needs refresh as new FIX-* commits land** |
| `BACKFILL-S6` | Replace 8 `[TODO: PROFILE-NN]` markers in §6 with real trace numbers; re-run MODULE-02 aggregator to refresh §6.4 table | PROFILE-01..04 outputs | ❌ — |
| `BACKFILL-S7` | Fill §7 numerical cells (unit time, layer-1 vs layer-2, saturation pass/fail, 1/4/16-GPU latency, 4/16-GPU throughput) | MEASURE-01..04 outputs | ❌ — |
| `REFRESH-S1` | Update Abstract numbers after BACKFILL-S6/S7 | BACKFILL-S6, BACKFILL-S7 | ❌ — |
| `REFRESH-paper-md` | Re-run `scripts/assemble_paper.sh` after any BACKFILL/REFRESH | any preceding REFRESH | ❌ — |

**Critical path now:** FIX-BUG-04-02 → MEASURE-01 → MEASURE-03 → MEASURE-04 → BACKFILL-S7. In parallel: PROFILE-03 → submit PROFILE-01..04 → BACKFILL-S6. Then REFRESH-S1 + REFRESH-paper-md.

### Execution status as of 2026-05-11 evening

- 25 commits on local branch `paper/multinexus`. **Branch not yet pushed to origin** — origin still on `multiNEXUS` (~25 commits behind).
- All 4 BUG audits done (~13.5K words across `docs/audits/BUG-{01..04}_*.md`).
- 1 of 7 expected FIX slices landed (`FIX-BUG-04-01`, the highest-impact).
- All 3 PRD module sketches implemented + unit-tested (Sketch #1 as `MEASURE-02`, Sketches #2-3 as `MODULE-02/03`).
- All 10 paper section drafts exist; `paper/paper.md` assembled (21,334 words).
- 6 of 7 PROFILE/MEASURE SLURM scripts committed; **0 jobs submitted to MN5** since 10:54 this morning (the argmax-v8192/v30k retry).
- §6 carries 8 `[TODO]` markers; §7 is a numerical skeleton.
- Branch reflects work BEFORE the Bootstrapper sync removal landed in MN5 binaries — every cell in `docs/PER_OP_VS_NEXUS.md` §4.4 will need re-checking once jobs run on the patched build.

### Pending slices for the Ralph loop (in dependency-correct order)

The Ralph loop should pick the **first unblocked slice from this list per iteration**, produce one commit (slice ID + `Depends-on` footer), then exit. Promise phrase for completion: `<promise>PRD-RALPH-COMPLETE</promise>` after the final REFRESH-paper-md commit.

**Phase A — push and lock in current work (do first, before any other slice):**
1. `OPS-push` — push `paper/multinexus` to `origin` so all in-flight work is durable. One-line operation; not a code commit but worth a status note.

**Phase B — FIX slices that block measurements (must land before MEASURE-01 runs):**
2. `FIX-BUG-04-02` — SCALE-CROSS-CUT for HP-BERT bootstrap call sites. Mirror the `cipher.scale() = SCALE` reset from `argmax_align_n32k.cu:225` at every `bs.bootstrap_3(...)` call site inside `bert_hp_multigpu.cu` (and `bert_hp_multinode.cu` if applicable). Without this, `MEASURE-01` may fail at layer 2 with the same Phantom encode-validation error that broke Argmax before `FIX-ARGMAX`.
3. `FIX-BUG-01-01` — add MAE gates to the six single-GPU align binaries. Each binary already prints MAE; wrap with `assert(mae < threshold)` style check. Threshold = `1e-5` for non-MatMul; `5%` relative for MatMul.
4. `PROFILE-03` — write `scripts/mn5/slurm_softmax_mgpu_nsys.sh` mirroring the existing `slurm_gelu_mgpu_nsys.sh` pattern. The only structurally-incomplete PROFILE deliverable.

**Phase C — submit jobs on the patched binaries (each is its own slice; one commit per submission, body includes JOBID):**
5. `RUN-PROFILE-01` — rebuild + submit matmul nsys on MN5; commit captures JOBID and log path.
6. `RUN-PROFILE-02` — rebuild + submit gelu nsys; commit captures JOBID and log path.
7. `RUN-PROFILE-03` — submit softmax nsys (after PROFILE-03 lands); commit captures JOBID and log path.
8. `RUN-PROFILE-04` — submit argmax nsys; commit captures JOBID and log path.
9. `RUN-MEASURE-01` — submit HP-BERT 1-head × 2-layer unit run; commit captures JOBID, log path, and the saturation-check output (`MODULE-01` analyzer applied to the layer-1/layer-2 timings).
10. `RUN-MEASURE-03` — submit DP-4 throughput run; commit captures JOBID and aggregate throughput.
11. `RUN-MEASURE-04` — submit DP-16 throughput run; commit captures JOBID and aggregate throughput.

**Phase D — additional FIX slices that don't block MEASURE but tighten the paper:**
12. `FIX-BUG-04-03` — Bootstrapper destructor (BOOT-RAW-OWN).
13. `FIX-BUG-04-04` — MatMul host-alloc out of `multiply_power_of_x`.
14. `FIX-BUG-02-01` — tighten HP-BERT MAE gate; add multinode gate.
15. `FIX-BUG-03-01` — `DistributedContext::destroy()` cleanup-order fixes.

**Phase E — backfill paper sections with the new measured numbers:**
16. `BACKFILL-S6` — replace each of the 8 `[TODO: PROFILE-NN]` markers in `paper/sections/06_per_op_typology.md` with the corresponding measured number from the RUN-PROFILE-NN logs. Re-run `scripts/MODULE-02-per-op-aggregator` to refresh the §6.4 cross-bucket summary table.
17. `BACKFILL-S7` — fill the numerical cells in `paper/sections/07_end_to_end.md` from RUN-MEASURE-01..04. Saturation check pass/fail; 1-GPU baseline; 4/16-GPU strong-scaling; 4/16-GPU throughput.
18. `REFRESH-S1` — update `paper/sections/01_abstract.md` headline numbers if BACKFILL-S6/S7 changed them (likely yes, given the post-FIX-BUG-04-01 baseline shift).
19. `REFRESH-Appendix` — append the new FIX-* commit hashes to the bug-fix log in `paper/sections/appendix_a.md`.
20. `REFRESH-paper-md` — re-run `scripts/assemble_paper.sh` to regenerate `paper/paper.md`. Final commit before promise.

**Completion criterion (Ralph promise):**
After slice 20 (`REFRESH-paper-md`) commits successfully, output:

```
<promise>PRD-RALPH-COMPLETE</promise>
```

This signals the loop to terminate.

**Per-iteration constraints:**
- One slice per iteration, one commit per iteration. Never bundle.
- Every commit follows the format from "Commit cadence guidance for agents" below: `<slice-id>(<area>): <imperative>` subject; why-then-what-then-where body; `Slice: <id>; Depends-on: <upstream ids>` footer.
- If the picked slice's upstream dependency is not yet ✅ in this map, skip and pick the next unblocked slice. Update the map entry's status column when the slice lands.
- For RUN-* slices: the SLURM job may still be PENDING when the iteration ends. In that case, commit the submission record (JOBID + log path) and let a later iteration check the result. Add an inline note in the relevant paper section if numbers will land later.
- For FIX-* slices: rebuild the binary on MN5 in the same commit body (capture the build log) so RUN-* slices can rely on the patched binary.
- All commits must be on branch `paper/multinexus`.

### Estimated wall-clock for SLURM jobs
- 4 nsys jobs × 5–15 min each, plus MN5 queue priority delay.
- 1 unit run: ~15–60 s actual run time.
- 2 throughput runs: ~12 min wall each (single-inference time, run G concurrently).
- Total wall, queue + run: 1–4 hours depending on queue.

### MD file harm assessment
- `docs/PI_PRESENTATION.md` — older brief from before today's grilling decisions. Mentions per-op-only strategy; does not discuss Goal 2's end-to-end at logN=15. A fresh agent reading this might re-derive an outdated plan. **Action: archive.**
- `docs/PI_PRESENTATION_SLIDES.md` — 10-slide outline tied to the older brief. Same risk. **Action: archive.**
- `paper/architecture_guide.md` — has a "(legacy framing)" note already added; safe to keep.
- All others audited and current.

### Commit cadence guidance for agents
- One commit per slice. No bundling.
- Subject ≤ 72 chars, imperative mood, includes the slice ID prefix in `()` form for grep-ability.
- Body explains the why first, then the what, then the where (file refs / JOBID / log path).
- Footer lists upstream slice dependencies that this commit consumes (`Slice: BUG-01; Depends-on: none`).
- Trailer `Co-Authored-By:` only if an agent did the work.
