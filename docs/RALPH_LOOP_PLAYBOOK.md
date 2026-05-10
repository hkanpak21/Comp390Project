# Ralph Loop Playbook for multiNEXUS PRD

This is the "how to execute the PRD" guide. Pair with
`docs/PRD_PRESENTATION_AND_HP_BERT.md` (which lists the slices).

---

## TL;DR — How ralph-loop works

- You type `/ralph-loop "<prompt>" [options]` in your Claude Code session.
- Claude attempts the task, modifies files, tries to exit.
- A stop hook intercepts and **feeds the same prompt back**.
- Claude sees its previous work in the files (this is the "self-reference").
- Loop continues until Claude outputs `<promise>SOMETHING</promise>` or
  hits `--max-iterations`.

**Mental model:** ralph-loop is a "keep working until done" wrapper.
Best for tasks where Claude needs multiple attempts (build, fail, fix,
build again).

---

## When to use ralph-loop vs single-shot

| Task type | Use | Example |
|---|---|---|
| Atomic 5-15 min text edit | **Single-shot** (just ask Claude normally) | S2: replace `\TODO{}` |
| Multi-file refactor with verification | **ralph-loop, single slice** | S6: reconcile 2.33 vs 2.16 |
| Implementation requiring build+test cycles | **ralph-loop, multi-iteration** | S16: HP-BERT skeleton |
| A whole independent group of slices | **ralph-loop, group prompt** | All of Group A |
| Anything needing MN5 SSH | **Don't loop — you must drive it** | S15, S18, S20, S21 |
| Human design decisions | **Don't loop — interactive instead** | S11 (presentation outline) |

---

## Three execution patterns

### Pattern 1: Single-shot (no loop)

Just ask Claude normally:

```
Do slice S2 from docs/PRD_PRESENTATION_AND_HP_BERT.md
```

What to expect: Claude reads the slice, makes the edit, reports done. ~5 min.

### Pattern 2: Single slice with loop (when verification matters)

```
/ralph-loop "Complete slice S6 from docs/PRD_PRESENTATION_AND_HP_BERT.md.
Read the slice's Acceptance criterion. After making changes, recompile
the paper with 'cd paper && make' and verify the criterion is met.
Output <promise>S6 DONE</promise> when satisfied." --completion-promise "S6 DONE" --max-iterations 5
```

What to expect: Claude makes changes, runs `make`, sees errors, fixes,
retries. Stops at first clean compile + criterion satisfied. ~15-30 min.

### Pattern 3: Group loop (work through many slices)

```
/ralph-loop "Read docs/PRD_PRESENTATION_AND_HP_BERT.md. Find the next
pending Group A slice (not marked with ✅ in the slice header). Complete
its steps and verify its Acceptance. When done, prepend '✅ ' to the
slice header line in the PRD so future iterations skip it. Continue
until ALL Group A slices are ✅. Output <promise>GROUP A DONE</promise>
when no pending Group A slices remain." --completion-promise "GROUP A DONE" --max-iterations 30
```

What to expect: Claude grinds through Group A, marking each ✅ as it
completes. ~2-3 hours unattended. Watch for repeat failures on the same
slice.

---

## Phase-by-phase execution plan

### PHASE 1 — TONIGHT (1 hour, optional)

**Goal:** clear the P0 paper-polish blockers so tomorrow you can
present a clean paper.

Open ONE Claude Code session. Run:

```
/ralph-loop "Read docs/PRD_PRESENTATION_AND_HP_BERT.md and complete the
following slices in order: S2, S1, S6. For each:
  - Read the slice description and Acceptance criterion
  - Execute the Steps
  - Verify the Acceptance is met (recompile paper with 'cd paper && make' if relevant)
  - Mark the slice header with '✅ ' prefix
After S6 is done, output <promise>PHASE 1 DONE</promise>." --completion-promise "PHASE 1 DONE" --max-iterations 10
```

**Expected duration:** ~45-60 min unattended.

**What to watch for:**
- After the first iteration, glance at the file diff. If the changes look
  right, walk away. If the diff looks weird, hit Ctrl+C and `/cancel-ralph`.
- If a single slice eats > 3 iterations, kill the loop and do that slice
  manually — usually means the prompt was ambiguous.

**Then go to bed.** Sleep matters more than another 30 min of fixes.

### PHASE 2 — TOMORROW MORNING (1 hour before talk)

**Goal:** Verify everything compiles, do final polish, rehearse.

**DO NOT use ralph-loop in this window.** Single-shot only — too risky
to have a loop crash 30 min before your talk.

Sequence (single Claude Code session, normal interaction):

1. **Verify paper compiles cleanly.**
   ```
   Compile paper/main.tex and confirm no warnings or errors. Show the
   final PDF page count and any unresolved references.
   ```

2. **Generate the money-shot figure (S14).**
   ```
   Do slice S14 from the PRD. Generate a single SVG showing the four
   parallelism axes with their projected speedups. Use the Sanzo Wada
   palette (cream #F0E8D0, indigo #1B3A55, persimmon #C8723A). Convert
   to PDF and add to paper/.
   ```

3. **Write the presentation outline (S11).**
   ```
   Do slice S11. Write paper/presentation_outline.md with 10 slides
   covering the structure described in the slice. Speaker notes per slide.
   ```

4. **Final read-through.**
   ```
   Read docs/HPC_PRIMER.md sections 1-6 and quote me back each
   "One-liner for the PI" sentence so I can memorize them.
   ```

**Expected total:** 45-60 min.

### PHASE 3 — TOMORROW AFTER TALK (Day 1 of HP-BERT)

**Goal:** kick off Strategy 1 (HP-BERT) implementation.

Open Claude Code in your project directory.

```
/ralph-loop "Complete slice S15 from docs/PRD_PRESENTATION_AND_HP_BERT.md.
This is the Phantom thread-safety smoke test. Steps:
  1. Create src/benchmarks/phantom_threadsafe_smoke.cu following the
     template in the slice
  2. Add it to CMakeLists.txt
  3. Build with 'cd build && make -j4 phantom_threadsafe_smoke'
  4. Run locally if a GPU is accessible; otherwise prepare an MN5 SLURM
     script at scripts/mn5/slurm_phantom_smoke.sh
  5. Mark the slice ✅ in the PRD
Output <promise>S15 DONE</promise> when build succeeds and SLURM script
is ready." --completion-promise "S15 DONE" --max-iterations 15
```

**Expected duration:** ~1-2 hours. Build failures will iterate.

**What to watch for:**
- CMakeLists.txt errors are common on first try
- Phantom header includes — Claude may not know the right paths
- If 5+ iterations fail on the same compile error, kill loop and inspect

**After S15 succeeds, queue up the SLURM job manually:**
```bash
ssh mn5-gpu
sbatch /gpfs/projects/etur02/hkanpak/Comp390Project/scripts/mn5/slurm_phantom_smoke.sh
```
Wait for completion, check output, then proceed to S16 in the next session.

### PHASE 4 — WEEK 1 OF HP-BERT (S16-S19)

**Goal:** ship a measured HP-BERT result.

These slices have **strong sequential dependencies**: S16 → S17 → S18 → S19.
Run them as separate ralph-loops (one slice per loop), with you in the
middle to drive the MN5 measurements.

**Day 2: S16 (skeleton, 1 head per GPU).**
```
/ralph-loop "Complete slice S16 from docs/PRD_PRESENTATION_AND_HP_BERT.md.
Build src/benchmarks/bert_hp_multigpu.cu by:
  1. Copying bert_dks_multigpu.cu
  2. Stripping the DKS rotation calls; keep single-GPU pinned rotations
  3. Wrapping the head loop in std::thread dispatch (one per GPU)
  4. Each thread holds its own PhantomContext, GaloisKeyStore, ciphertext
  5. Add to CMakeLists.txt and build with 'cd build && make -j4 bert_hp_multigpu'
After build succeeds, output <promise>S16 BUILDS</promise>." --completion-promise "S16 BUILDS" --max-iterations 20
```

Then YOU run on MN5:
```bash
ssh mn5-gpu
cd /gpfs/projects/etur02/hkanpak/Comp390Project
rsync from local
sbatch scripts/mn5/slurm_bert_hp_smoke.sh
```
Verify 4 head outputs match reference. If not, debug interactively.

**Day 3: S17 (12 heads on 4 GPUs).**
```
/ralph-loop "Complete slice S17. Modify bert_hp_multigpu.cu so each
GPU's worker thread runs 3 heads sequentially (12 heads total across
4 GPUs). Combine the 12 head outputs into the layer's attention. Build,
prepare SLURM. Output <promise>S17 BUILDS</promise>." --completion-promise "S17 BUILDS" --max-iterations 15
```

Run MN5 measurement; verify MAE.

**Day 4-5: S18 (full BERT-base, 12-layer × 12-head).**
```
/ralph-loop "Complete slice S18. Wrap the full BERT-base inference in
HP-BERT mode. Add 3-trial timing harness. Prepare SLURM script. Output
<promise>S18 BUILDS</promise>." --completion-promise "S18 BUILDS" --max-iterations 15
```

Run on MN5, capture results in experiments/results/.

**Day 6: S19 (paper update).**
Single-shot, no loop:
```
Do slice S19. Update paper/main.tex with the HP-BERT result. Add a row
to the opt-evolution table and to syscompare. Update §4.3 narrative.
Recompile.
```

---

## Monitoring a running loop

While ralph-loop is running:

1. **Glance at the file diff every 5-10 minutes.** If the changes look
   sane, walk away.
2. **If you see the same compile error in iteration 3+**, hit Ctrl+C and
   `/cancel-ralph`. The prompt is probably ambiguous.
3. **If the loop is editing files you didn't expect**, kill it. The slice
   description was probably misleading.
4. **If you need to leave the room**, set `--max-iterations` to a lower
   number (5-10) so it doesn't burn forever.

## Stopping a loop

```
/cancel-ralph
```

Removes the loop state file. Any work already done is preserved (it's in
the git working tree).

---

## Common pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| No `--max-iterations` | Runs forever | Always include it: 5-30 depending on task |
| Vague completion-promise | Loop never exits | Use a specific tag like `S6 DONE` not just `DONE` |
| Loop touches wrong files | Diff looks weird after iter 1 | Cancel, refine prompt with explicit file list |
| Slice scope too broad | Loop oscillates between half-fixes | Split slice into smaller pieces |
| MN5-dependent slice | Loop can't make progress | Don't loop these — drive interactively |

---

## Parallel execution (advanced)

You can run ralph-loops in **separate terminal sessions** for
independent groups. Each terminal needs its own Claude Code session.

**Terminal 1** — Group A paper polish:
```
/ralph-loop "Pick next pending Group A slice from PRD..." ...
```

**Terminal 2** — Group E documentation:
```
/ralph-loop "Pick next pending Group E slice from PRD..." ...
```

**Terminal 3** — Group D T-MODUP (after Phase 3):
```
/ralph-loop "Pick next pending Group D slice from PRD..." ...
```

⚠ **Don't run parallel loops on slices that touch the SAME files**
(e.g., both editing main.tex). Use ✅ marking to coordinate.

---

## What "DONE" looks like for the presentation

After Phase 1 + Phase 2:
- `paper/main.pdf` exists, compiles clean, no red TODO text
- All Group A slices marked ✅ in the PRD
- `paper/presentation_outline.md` exists with 10 slides
- `docs/HPC_PRIMER.md` accessible during talk
- You can recite Q1-Q6 one-liners from memory
- `docs/MULTIGPU_SCALING_PLAN.md` is your "future work" backup if PI presses

After Phase 3 + Phase 4 (week 1):
- `src/benchmarks/bert_hp_multigpu.cu` exists, builds, passes correctness
- HP-BERT measurement logged in `experiments/RESULTS.md`
- Speedup ≥ 2.5× over Phase 4b
- Paper updated with HP-BERT row

That's the path to a defensible PI talk tomorrow + a strong follow-up
result in 1-2 weeks.

---

## One-line cheat sheet

| Goal | Command |
|---|---|
| Tonight, paper polish | `/ralph-loop "Complete S2, S1, S6 from PRD..." --completion-promise "PHASE 1 DONE" --max-iterations 10` |
| Tomorrow morning | (no loop, single-shot only) |
| Day 1 HP-BERT | `/ralph-loop "Complete S15..." --completion-promise "S15 DONE" --max-iterations 15` |
| Day 2 HP-BERT | `/ralph-loop "Complete S16..." --completion-promise "S16 BUILDS" --max-iterations 20` |
| Stop a loop | `/cancel-ralph` |
| Watch progress | Check the file diff in your IDE |
