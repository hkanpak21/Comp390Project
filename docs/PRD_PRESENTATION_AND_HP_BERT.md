# PRD — multiNEXUS HPC Story: Presentation Defense + HP-BERT Implementation

## North Star

Demonstrate non-trivial HPC engineering on a real workload (BERT inference
under FHE), through a multi-axis multi-GPU parallelization story. The PI
presentation tomorrow shows the foundation (DKS at N=65,536) and the
roadmap (head-parallel BERT → ~3.75× over current). The semester's
implementation work delivers the head-parallel measurements that turn
the roadmap into a published result.

## Definition of Done (presentation-ready, end of day tomorrow)

- Paper compiles cleanly with all P0 issues fixed (see Group A)
- Primer (`docs/HPC_PRIMER.md`) covers Q1–Q6 of the grilling session
- Multi-GPU scaling plan (`docs/MULTIGPU_SCALING_PLAN.md`) is the
  defense + roadmap story
- Presentation deck (10 slides) covers: problem, contribution, mechanics,
  results, the multi-axis story, future-work HP-BERT pitch
- User has rehearsed the Q1–Q6 answers from the primer

## Definition of Done (HP-BERT shipped, end of week 2)

- `src/benchmarks/bert_hp_multigpu.cu` exists, builds, passes correctness
  vs Phase 4b (MAE ≤ 2.25e-6)
- Full 12-layer × 12-head measurement on 4× H100, 5 trials, captured in
  `experiments/RESULTS.md` and `docs/RESULTS_SUMMARY.md`
- Speedup ≥ 2.5× over Phase 4b on full BERT-base (370–500 s vs 1,388 s)
- Paper updated with HP-BERT row in Tables 1, 2, and a new figure

---

## How to use this PRD

Each slice below is **independent** (can be picked off in any order
unless `Depends on:` says otherwise) and **atomic** (single sitting,
single tangible artifact). Slices are tagged by **Group** —
slices in different groups can be worked in parallel by separate Ralph
Loop runs (or by you on different terminals).

Format per slice:
```
### S<N>: <title>
**Group:** A | B | C | D | E
**Priority:** P0 | P1 | P2
**Effort:** <minutes>
**Depends on:** <slice IDs or "none">
**Acceptance:** <single binary criterion>
```

---

# GROUP A — Paper Polish (independent, parallelizable, do tonight)

These can ALL be done in parallel. None depends on another. Cumulative
effort ~3 hours.

### S1: Regenerate figures from SVG
**Group:** A
**Priority:** P0
**Effort:** 10 min
**Depends on:** none
**Acceptance:** `pdfinfo paper/fig1_multigpu_scaling.pdf` shows pages > 0
and file size > 10 KB. Same for fig5, fig6.

**Steps:**
1. `cd paper/`
2. `rsvg-convert -f pdf fig1_multigpu_scaling.svg -o fig1_multigpu_scaling.pdf`
3. Same for fig5_kernel_breakdown.svg and fig6_gpu_utilization.svg
4. If `rsvg-convert` not installed: `inkscape --export-type=pdf fig1_multigpu_scaling.svg`
5. Recompile main.tex; verify figures render

### S2: Fix \TODO{not delivered} red text in published PDF
**Group:** A
**Priority:** P0
**Effort:** 5 min
**Depends on:** none
**Acceptance:** `grep -n "TODO" paper/main.tex` returns no rendered TODO
markers (only commented-out lines OK).

**Steps:**
1. Edit paper/main.tex line 382: replace `\TODO{not delivered}` with
   `--- (future work)`
2. Recompile; verify no red text in the opt-evolution table

### S3: Replace dangling "Table 6 of supplementary results"
**Group:** A
**Priority:** P0
**Effort:** 10 min
**Depends on:** none
**Acceptance:** No reference to "supplementary" or "Table 6" in
paper/main.tex.

**Steps:**
1. Edit paper/main.tex line 586: rewrite as inline citation, e.g.,
   "(measured at $29.5$~ms median on a single GPU; see
   `docs/RESULTS_SUMMARY.md` Table 6)" or just inline the value.

### S4: Replace tab:syscompare draft caption
**Group:** A
**Priority:** P0
**Effort:** 10 min
**Depends on:** none
**Acceptance:** Caption no longer mentions "projections after T-STRAGGLER
and T-MODUP land" or "filled after MN5 runs."

**Steps:**
1. Edit paper/main.tex line 488 caption with: "Comparison with prior
   systems on bootstrap and full BERT inference. multiNEXUS rows are
   directly measured Phase 4b values; NEXUS, Cerium, Cinnamon are as
   reported by the cited works."

### S5: Replace blackwell citation for H100 launch latency
**Group:** A
**Priority:** P0
**Effort:** 20 min
**Depends on:** none
**Acceptance:** `\cite{blackwell}` no longer used to support a
launch-latency claim. Either remove the claim or replace the citation.

**Steps:**
1. Either: change to `~5\,\mu\mathrm{s}` (no citation, common knowledge)
2. Or: cite NVIDIA CUDA Best Practices Guide
3. Or: cite a specific microbenchmark paper

### S6: Reconcile 2.33× vs 2.16× speedup inconsistency
**Group:** A
**Priority:** P0
**Effort:** 30 min
**Depends on:** none
**Acceptance:** Paper consistently uses one speedup as headline; the
other is contextualized in a footnote or table.

**Steps:**
1. Decide: headline = 2.33× (directly measured 12-layer)
2. Edit §4.3 to clearly say: "Per-head projection: 115.7 s (2.16×).
   Directly measured 12-layer: 107.08 s (2.33×). The directly measured
   value validates the projection methodology and is our headline."
3. Verify abstract, conclusion, and §4.3 all align

### S7: Bibliography cleanup — remove duplicates and placeholders
**Group:** A
**Priority:** P1
**Effort:** 20 min
**Depends on:** none
**Acceptance:** `grep -E "nicepaper|bootstrappingCKKS|^@misc\{Nexus,"
paper/refs.bib` returns no entries.

**Steps:**
1. Delete `nicepaper1`, `nicepaper2`, `nicepaper3` entries
2. Delete `bootstrappingCKKS` (duplicate of `bsgs`)
3. Delete `Nexus` (duplicate of `Zhang2025`)
4. Optionally delete clearly irrelevant entries (Spectre, Meltdown,
   Samsung breaches, Intel TMD, GDPR, etc.) — but only if not cited

### S8: Add CPU baseline 249.6 s provenance
**Group:** A
**Priority:** P1
**Effort:** 15 min
**Depends on:** none
**Acceptance:** Paper has either a table row or footnote explaining how
249.6 s was derived.

**Steps:**
1. Add a footnote to §4.3 or to Table 1: "CPU streaming reference
   measured on [hardware] at [date] with [config]; see
   `docs/RESULTS_SUMMARY.md` §0 for details."

### S9: Include fig2 (layer breakdown) and fig3 (bootstrap phases)
**Group:** A
**Priority:** P1
**Effort:** 30 min
**Depends on:** S1
**Acceptance:** main.tex has `\includegraphics` for fig2 and fig3 in
appropriate places (probably §4.4 layer breakdown and §4.2 where-time-
goes).

**Steps:**
1. Convert `fig2_layer_breakdown.svg` and `fig3_bootstrap_phases.svg`
   to PDF
2. Add `\begin{figure}` blocks in relevant sections of main.tex
3. Recompile

### S10: Add DKS data-flow figure to paper
**Group:** A
**Priority:** P1
**Effort:** 15 min
**Depends on:** S1
**Acceptance:** `paper/fig_dks_dataflow.pdf` exists and is included in
the paper §3 (Design).

**Steps:**
1. Convert `paper/fig_dks_dataflow.svg` (just created) to PDF
2. Replace fig1 (which is mislabeled as DKS) or add as a new figure
3. Update caption and reference

---

# GROUP B — Presentation Materials (mostly independent, do tomorrow morning)

### S11: Outline 10-slide presentation deck
**Group:** B
**Priority:** P0
**Effort:** 45 min
**Depends on:** none (can leverage primer + scaling plan)
**Acceptance:** `paper/presentation_outline.md` exists with 10 slides,
each with title + 3-5 bullet points + speaker notes.

**Steps:**
1. Slide 1: Title + the one-sentence contribution
2. Slide 2: The problem (62 GB key store at N=65,536, doesn't fit on H100)
3. Slide 3: CKKS background (limbs, digits, key-switching, bootstrap — VERY brief)
4. Slide 4: The four parallelism axes (digit, head, layer, batch)
5. Slide 5: DKS — what we built, the data flow figure
6. Slide 6: Phase 4b results (table + figure)
7. Slide 7: The honest 7%-over-single-GPU story + reframe (feasibility, not speedup)
8. Slide 8: Multi-axis roadmap (HP-BERT, T-MODUP, multi-node)
9. Slide 9: Expected HP-BERT result (370 s) vs NEXUS (37 s at smaller N)
10. Slide 10: Summary + ask (advisor input on multi-node priority)

### S12: Memorize the Q1–Q6 one-liners from the primer
**Group:** B
**Priority:** P0
**Effort:** 30 min
**Depends on:** S11 (or independent)
**Acceptance:** User can recite all 6 one-liners from memory without
checking the primer.

**Steps:**
1. Open `docs/HPC_PRIMER.md`
2. Find the "One-liner for the PI" sentence in each section
3. Recite each 3 times
4. Test self by writing each from memory

### S13: Print or have the primer accessible during the talk
**Group:** B
**Priority:** P1
**Effort:** 5 min
**Depends on:** all primer sections complete
**Acceptance:** Primer is on second monitor, printed, or on phone.

### S14: Generate one big "money shot" figure
**Group:** B
**Priority:** P1
**Effort:** 30 min
**Depends on:** none
**Acceptance:** A single figure showing the four-parallelism-axes story,
saved as `paper/fig_parallelism_axes.svg` and converted to PDF.

**Steps:**
1. Sketch: x-axis = parallelism axis (digit, head, layer, batch), y-axis
   = projected speedup; current Phase 4b is one bar; HP-BERT projection
   is a second bar; future strategies are dashed bars
2. Use Sanzo Wada palette (indigo + persimmon + cream)
3. Save SVG, convert to PDF

---

# GROUP C — HP-BERT Implementation (sequential within group, post-presentation)

These run AFTER tomorrow's talk. Each depends on the previous.

### S15: Phantom thread-safety smoke test
**Group:** C
**Priority:** P0 (for HP-BERT path)
**Effort:** 90 min
**Depends on:** none
**Acceptance:** `src/benchmarks/phantom_threadsafe_smoke.cu` exists,
builds, runs 2 simultaneous bootstraps in 2 std::threads on 2 GPUs
without crashing. Output MAE matches single-GPU reference.

**Steps:**
1. Create new file `src/benchmarks/phantom_threadsafe_smoke.cu`
2. In main(): spawn 2 std::threads, each calling `bootstrap_3` on its
   own GPU with its own context
3. Use simple ciphertext (constant value 1.0)
4. Verify both threads complete without exception
5. Compare decrypted output to reference; assert MAE < 1e-5
6. Add to CMakeLists.txt
7. Submit SLURM job, verify runs cleanly

### S16: Skeleton for bert_hp_multigpu
**Group:** C
**Priority:** P0 (for HP-BERT path)
**Effort:** 2 hours
**Depends on:** S15
**Acceptance:** `src/benchmarks/bert_hp_multigpu.cu` exists, builds,
runs 1 head per GPU on 4 GPUs (no DKS), produces 4 head outputs that
each match single-GPU reference (MAE ≤ 2.25e-6).

**Steps:**
1. `cp src/benchmarks/bert_dks_multigpu.cu src/benchmarks/bert_hp_multigpu.cu`
2. Strip out DKS rotation calls; replace with single-GPU pinned rotations
   per Phase 1
3. Replace head-loop body with std::thread dispatch — one head per GPU
4. Each thread holds its own PhantomContext, key store, ciphertext
5. Add CMakeLists.txt entry
6. Build with `make -j20 bert_hp_multigpu`
7. Single SLURM run, 4 heads on 4 GPUs, verify outputs

### S17: Scale to 12 heads on 4 GPUs (3 per GPU sequential)
**Group:** C
**Priority:** P0
**Effort:** 90 min
**Depends on:** S16
**Acceptance:** bert_hp_multigpu runs all 12 heads (3 per GPU
sequentially), each output matches reference.

**Steps:**
1. Modify head dispatch loop: each GPU thread runs 3 heads sequentially
2. Combine the 12 head outputs into the layer's attention output
3. Verify MAE on full 12-head attention vs reference

### S18: End-to-end 12-layer × 12-head HP-BERT measurement
**Group:** C
**Priority:** P0
**Effort:** 3 hours (incl. SLURM queue time)
**Depends on:** S17
**Acceptance:** First headline number for HP-BERT logged in
`experiments/results/<date>_h100x4_hp-bert/`. Median of 3 trials.

**Steps:**
1. Wrap full BERT-base inference in HP-BERT
2. Submit SLURM job, 3 trials
3. Capture per-layer timing breakdown
4. Compute total full-BERT latency
5. Log to `experiments/RESULTS.md`

### S19: Update paper with HP-BERT row
**Group:** C
**Priority:** P0
**Effort:** 60 min
**Depends on:** S18
**Acceptance:** `paper/main.tex` Table 2 (or syscompare) has an HP-BERT
row with the measured number.

**Steps:**
1. Add row to opt-evolution table
2. Add row to syscompare
3. Update §4.3 narrative with HP-BERT result
4. Recompile, sanity-check rendering

---

# GROUP D — T-MODUP Rescue (parallel to C, optional)

### S20: T-MODUP regression diagnosis
**Group:** D
**Priority:** P1
**Effort:** 2 hours
**Depends on:** none (but only useful if HP-BERT doesn't fully close gap)
**Acceptance:** `docs/T_MODUP_DIAGNOSIS.md` exists with: exact failing
SLURM job ID, exact stack trace, exact chain levels at which beta < n_gpus,
and proposed fix.

**Steps:**
1. SSH to MN5
2. Trigger the failing T-MODUP path with NVTX trace enabled
3. Capture cuda-memcheck output
4. Identify the specific cudaMemcpyAsync call site
5. Document findings

### S21: Implement T-MODUP fix
**Group:** D
**Priority:** P1
**Effort:** 2 hours
**Depends on:** S20
**Acceptance:** A SLURM job runs DKS_ROTATE=1 with T-MODUP active to
completion, MAE ≤ 2.25e-6, bootstrap < 1,800 ms (down from 2,098 ms).

**Steps:**
1. Apply the fix from S20
2. Build on MN5
3. Submit SLURM job
4. Verify correctness and speedup

### S22: Update paper with T-MODUP row
**Group:** D
**Priority:** P1
**Effort:** 30 min
**Depends on:** S21
**Acceptance:** opt-evolution table no longer has "future work" for
T-MODUP; the actual measured value is filled in.

---

# GROUP E — Documentation and Measurements (parallel to C/D)

### S23: Update experiments/RESULTS.md with all MN5 runs from notes/log.md
**Group:** E
**Priority:** P1
**Effort:** 60 min
**Depends on:** none
**Acceptance:** `experiments/RESULTS.md` "Completed Runs" table has at
least 6 rows including all Phase 1, 3, 4a, 4b runs.

**Steps:**
1. Read `notes/log.md` and SLURM output files
2. Add one row per run with: date, hardware, experiment shortname,
   key finding, dir
3. Update plots index if any plots were generated

### S24: Add primer Section 7 — Memory hierarchy and PCIe bottleneck math
**Group:** E
**Priority:** P1
**Effort:** 45 min
**Depends on:** none
**Acceptance:** New section in `docs/HPC_PRIMER.md` covering: H100 HBM
size and bandwidth, host RAM streaming, PCIe Gen5 bandwidth and the
98 GB/bootstrap floor, NVLink internal bandwidth.

### S25: Add primer Section 8 — Why N=65,536 (security argument)
**Group:** E
**Priority:** P2
**Effort:** 30 min
**Depends on:** none
**Acceptance:** New section explaining: λ = 128-bit security, sparse
secret-key with Hamming weight 192, why N=65,536 is the smallest
parameter set meeting both bootstrap depth and security requirements.

### S26: Add primer Section 9 — Numerical accuracy (MAE 2.25e-6)
**Group:** E
**Priority:** P2
**Effort:** 30 min
**Depends on:** none
**Acceptance:** New section explaining: how MAE is computed, what
threshold is acceptable for downstream BERT classification, why our
result is sufficient.

### S27: Benchmark README
**Group:** E
**Priority:** P2
**Effort:** 30 min
**Depends on:** none
**Acceptance:** `src/benchmarks/README.md` exists, lists all 30+
benchmark files with one-line descriptions and current status (active /
deprecated / experimental).

### S28: Annotate dead T-STRAGGLER infrastructure
**Group:** E
**Priority:** P2
**Effort:** 20 min
**Depends on:** none
**Acceptance:** `output_aggregation.cu` and `distributed_context.cuh`
have inline comments at each T-STRAGGLER/T-OVERLAP call site explaining
"inert — see commit 44c849f / docs/PRD §S28".

---

# Dependency Graph (mermaid-readable)

```
S1 (figures) ───┬─→ S9 (fig2/3 included)
                └─→ S10 (DKS fig PDF)

S15 (smoke) ──→ S16 (skeleton) ──→ S17 (12 heads) ──→ S18 (e2e measurement) ──→ S19 (paper update)

S20 (diagnosis) ──→ S21 (fix) ──→ S22 (paper update)

All of A (S1-S10) — independent
All of B (S11-S14) — independent except where noted
All of E (S23-S28) — independent
```

---

# Suggested execution order

## Tonight (after this session)
- S2, S4 (5 min each — easy wins)
- S1 (10 min — unblocks others)
- S6 (30 min — most impactful paper fix)
- S11 (45 min — presentation outline)
- S12 (30 min — memorize one-liners)

## Tomorrow morning before talk
- S3, S5 (paper polish)
- S13 (have primer accessible)
- S14 (money-shot figure if time)
- Final read-through of primer

## Tomorrow after talk → end of week 1
- S15 (Phantom thread-safety smoke test)
- S7, S8 (bibliography cleanup, baseline provenance — if not already done)
- S23 (results log update)
- S24 (primer Section 7)

## Week 2
- S16, S17, S18, S19 (HP-BERT implementation chain)

## Week 3-4
- S20, S21, S22 (T-MODUP rescue, optional)
- S25, S26 (more primer sections)
- S27, S28 (cleanup)

---

# Ralph Loop usage

To run a single slice:
```
/ralph-loop:ralph-loop "Pick the next pending P0 slice from
docs/PRD_PRESENTATION_AND_HP_BERT.md, complete it, and mark it done by
prepending '✅ ' to the slice header."
```

To run a group in parallel (separate terminals):
```
/ralph-loop:ralph-loop "Pick the next pending Group A slice. Skip
slices marked ✅."

/ralph-loop:ralph-loop "Pick the next pending Group B slice."
```

For the long-running implementation work (Group C), run sequentially
because of dependencies:
```
/ralph-loop:ralph-loop "Work through Group C slices in order: S15, S16,
S17, S18, S19. Do not skip ahead. After each, mark ✅ and verify the
acceptance criterion before moving on."
```

---

# Acceptance: presentation-ready

By tomorrow morning, the following should all be true:

- [ ] All Group A P0 slices done (S1, S2, S3, S4, S5, S6)
- [ ] S11 (presentation outline) done
- [ ] S12 (one-liners memorized)
- [ ] User has read MULTIGPU_SCALING_PLAN.md end to end
- [ ] User can answer: "Why does cudaMemcpyAsync from pageable memory block?"
- [ ] User can answer: "Why does AllReduce(SUM) work for DKS?"
- [ ] User can answer: "What's your contribution if Phase 1 alone gets 4.69×?"

If those check, the presentation is defensible.
