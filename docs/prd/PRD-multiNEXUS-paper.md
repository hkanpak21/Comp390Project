---
title: "PRD — multiNEXUS Paper: Per-Op Multi-GPU Typology + End-to-End at uniform logN=15"
status: needs-triage
owner: hkanpak21
created: 2026-05-11
labels: [needs-triage, paper, multi-gpu, fhe]
---

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

### Vertical slice initial map (with dependencies)

Phase order is: BUG → PROFILE / MEASURE → WRITE → APPENDIX. DOC slices have no upstream and can run any time.

| Slice ID | Description | Depends on |
|---|---|---|
| `DOC-01` | Update CLAUDE.md with paper plan, 9-section structure, per-op template, slice convention, commit convention | none |
| `DOC-02` | Audit MD files; archive `docs/PI_PRESENTATION.md` and `docs/PI_PRESENTATION_SLIDES.md` | none |
| `DOC-03` | Publish this PRD (the file you are reading) | none |
| `BUG-01` | Bug audit pass on six align binaries | none |
| `BUG-02` | Bug audit pass on `bert_hp_multigpu` + `bert_hp_multinode` | none |
| `BUG-03` | Bug audit pass on `src/multi_gpu/` framework | none |
| `BUG-04` | Bug audit pass on `src/nexus_eval/` evaluator wrappers | none |
| `PROFILE-01` | Generate 4-GPU nsys for matmul_align_n8k | BUG-01 (matmul) |
| `PROFILE-02` | Generate 4-GPU nsys for gelu_mgpu_align | BUG-01 (gelu) |
| `PROFILE-03` | Generate 4-GPU nsys for softmax_mgpu_align | BUG-01 (softmax) |
| `PROFILE-04` | Generate 4-GPU nsys for argmax_align_n32k | BUG-01 (argmax) |
| `MEASURE-01` | Goal 2 unit run: 1-head × 2-layer @ logN=15 on 1 GPU | BUG-02 |
| `MEASURE-02` | Saturation check from MEASURE-01 output (analyzer) | MEASURE-01 |
| `MEASURE-03` | Goal 2 data-parallel throughput at 4-GPU | MEASURE-01, BUG-02 |
| `MEASURE-04` | Goal 2 data-parallel throughput at 16-GPU | MEASURE-03 |
| `WRITE-S2` | Section 2 (Introduction) draft | DOC-01 |
| `WRITE-S3` | Section 3 (Background) draft | DOC-01 |
| `WRITE-S4` | Section 4 (Identifying NEXUS on H100) | none (existing data) |
| `WRITE-S5` | Section 5 (Multi-GPU strategies — DKS, HP, DP) | none (architectural) |
| `WRITE-S6.{op}` | Six per-op subsections of Section 6 (Goal 1 typology) | corresponding PROFILE-NN + BUG-01 |
| `WRITE-S7` | Section 7 (Goal 2 end-to-end) | MEASURE-01..04 |
| `WRITE-S8` | Section 8 (Discussion) | WRITE-S6, WRITE-S7 |
| `WRITE-S9` | Section 9 (Conclusion + future work) | WRITE-S8 |
| `WRITE-Appendix` | Appendix A (NEXUS/Phantom mods + bug log) | BUG-01..04 |

**Critical path:** DOC-01 → BUG-01..04 → (PROFILE-01..04 + MEASURE-01..04 in parallel) → WRITE-S6.{op} + WRITE-S7 → WRITE-S8 → WRITE-S9.

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
