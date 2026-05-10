# multiNEXUS Multi-GPU Scaling Plan

**Goal:** deliver non-trivial multi-GPU speedups for full BERT-base inference
at N=65,536, comparable to or better than NEXUS's 37.3 s on 4× A100 (which
runs at the smaller, faster N=32,768 with re-encryption).

**Audience:** the work plan we execute through the rest of the semester.
Doubles as the "future work" framing for the PI presentation tomorrow.

---

## 1. Where we are and why the current approach is blocked

Phase 4b champion measurements:

| Quantity | Value | Notes |
|---|---|---|
| Single bootstrap | 2,098 ms | 4× H100, DKS with all 4 GPUs working on ONE bootstrap |
| Per-head per-layer (1 head) | 9,640 ms | 4 bootstraps + attention + LayerNorm + FFN |
| 12-head BERT layer (projection) | 115.7 s | 12 × 9,640 ms — sequential heads, DKS within each |
| Full BERT-base (12 layers × 12 heads) | ~1,388 s ≈ 23 min | 12 × 115.7 s |
| NEXUS reference, full BERT | 37.3 s | 4× A100, N=32,768, head-parallel + re-encryption |

We are **37× slower than NEXUS** on full BERT-base. The DKS-only design
fundamentally cannot close this — the 4 GPUs are serialized into one head
at a time.

The 7% critique is the symptom; the disease is the **parallelism strategy**:
DKS uses all 4 GPUs to do a single bootstrap a little faster, when we could
be using all 4 GPUs to do four heads' worth of bootstraps in parallel. The
math says head-parallelism is ~3.7× faster than DKS at the same hardware.

---

## 2. The four parallelism axes

| Axis | What it parallelizes | Currently exploited? | Theoretical ceiling |
|---|---|---|---|
| **A. Digit-axis (DKS)** | Within ONE bootstrap, across β=36 digits | YES (Phase 4b) | ~G× (limited by mod-up redundancy, AllReduce comm) |
| **B. Head-parallelism** | Across BERT's 12 attention heads | NO | min(G, 12)× — embarrassingly parallel |
| **C. Layer pipelining** | Across BERT's 12 transformer layers | NO | up to 12× in steady-state batched inference |
| **D. Batch parallelism** | Across independent ciphertexts | NO | B× for B ciphertexts |

We have **only attacked axis A** so far, and not even fully (T-MODUP
unlanded). Axes B, C, D are untouched. Axis B is the highest leverage and
the cheapest to implement.

---

## 3. Strategy menu (ranked by impact ÷ effort)

### Strategy 1 — Head-parallel BERT (HP-BERT). HIGHEST PRIORITY.

**Idea.** Replace DKS-within-head with one-head-per-GPU-group. With 4 GPUs
and 12 heads, each GPU runs 3 heads sequentially. Each GPU uses Phase 1
(single-GPU pinned host streaming) for its own key store.

**Implementation.** New benchmark `src/benchmarks/bert_hp_multigpu.cu`,
roughly modeled on existing `bert_dks_multigpu.cu` but with the head loop
distributed across `std::thread`s (one per GPU) instead of within a single
head's bootstrap.

**Memory.** Each GPU's host pins its own copy of the key store: 4 × 62 GB
= 248 GB host RAM. MN5 ACC nodes have 512 GB. Comfortable.

**Math.**
```
   per-head (single-GPU pinned, Phase 1):  ~10,272 ms
       = 4 bootstraps × 2,284 ms + ~1,136 ms other ops
   per-GPU sequential heads (3):           30.8 s
   per-layer (4 GPUs in parallel):         30.8 s
   12 layers:                              370 s ≈ 6.2 min
```

**Speedup vs current Phase 4b:** 1,388 s / 370 s ≈ **3.75×**.
**Speedup vs CPU baseline (50 min):** ≈ **8×**.

**Risks.**
- Thread-safety of Phantom's key store across 4 host-pinned regions
- Per-thread CUDA stream/context initialization
- Memory bandwidth contention on shared PCIe between 4 streaming GPUs

**Implementation cost.** ~2 weeks (1 week build, 3 days debug, 4 days
measure + integrate paper).

---

### Strategy 2 — T-MODUP fix (rescue DKS). MEDIUM PRIORITY.

**Idea.** Fix the zero-sized `cudaMallocAsync` regression so per-GPU mod-up
works. Cuts ~640 ms per bootstrap (~30% of bootstrap time).

**Why now.** Even after switching to HP-BERT, each GPU's individual
bootstraps run with single-GPU pinned (no DKS within head). T-MODUP only
matters if we run a hybrid: HP-BERT × DKS-within-head. That requires
G≥8 (e.g., 4 nodes × 4 GPUs in multi-node, or 2 GPUs per head on single
node).

**Decision:** Defer until HP-BERT is measured. If HP-BERT alone reaches the
NEXUS-comparable ballpark, T-MODUP becomes a paper appendix.

**Implementation cost.** ~2-4 hours of focused debug + 1 SLURM job.

---

### Strategy 3 — LayerPipeline (LP-BERT). MEDIUM PRIORITY, HIGH RISK.

**Idea.** Pipeline 4 layers across 4 GPUs. GPU 0 runs layers 1, 5, 9; GPU 1
runs 2, 6, 10; etc. For batched inference, multiple ciphertexts can flow
through the pipeline.

**Math (single ciphertext):** no win — pipelined sequence is still
serialized end-to-end. Only wins for batched.

**Math (batched, B ≥ G):** steady-state, each ciphertext exits in
~per-layer-time × num-stages, but the **throughput** is per-layer-time per
ciphertext. For BERT: 9,640 ms per layer × B ciphertexts / 4 GPUs.

**Memory.** Each GPU needs the full key store for its 3 layers — same as
HP-BERT (262 GB host pinned per node). Replicated across stages, no
sharing.

**When to consider.** After HP-BERT lands. Useful for inference-server
demonstration, less useful for one-shot latency benchmarks.

**Implementation cost.** ~2-3 weeks.

---

### Strategy 4 — Batch parallelism (B-BERT). LOW PRIORITY, HIGH IMPACT.

**Idea.** Process B independent BERT inferences in parallel. Each GPU
handles B/G inferences. Embarrassingly parallel across requests.

**Why low priority.** This is the inference-server picture, not a
single-query latency story. Less compelling for a research paper, more
compelling for a deployment story.

**Useful when.** If we want a "throughput" headline alongside the latency
headline.

**Implementation cost.** ~1 week (mostly orchestration).

---

### Strategy 5 — Hybrid HP + DKS (HP-DKS-BERT). FUTURE WORK.

**Idea.** Two levels of parallelism: outer head-parallel + inner DKS within
each head. For BERT-base (12 heads) with 16 GPUs (4 nodes × 4 GPUs):
- 4 head-groups (3 heads each)
- DKS within each group's bootstrap (4 GPUs split the digit axis)

**Math (theoretical, requires multi-node):**
```
   per-bootstrap (DKS within 4 GPUs): 2,098 ms (current Phase 4b)
   per-head: 4 × 2,098 + ~1,136 = 9,528 ms
   per-GPU-group: 3 heads sequential = 28.6 s per layer
   12 layers: 343 s ≈ 5.7 min
```

Marginal improvement over HP-BERT-alone (370 s) because mod-up redundancy
is still there. With T-MODUP delivered:
```
   per-bootstrap (DKS + T-MODUP): ~1,486 ms
   per-head: 4 × 1,486 + 1,136 = 7,080 ms
   per-GPU-group: 3 heads sequential = 21.2 s per layer
   12 layers: 254 s ≈ 4.2 min
```

**When to consider.** After HP-BERT and multi-node infrastructure.

**Implementation cost.** ~3-4 weeks (multi-node NCCL config, two-level
thread orchestration, key sharding logic).

---

### Strategy 6 — Multi-node HP (MN-HP-BERT). STRETCH GOAL.

**Idea.** 4 nodes × 4 GPUs = 16 GPUs. With 12 BERT heads, run 12 GPUs (one
per head) leaving 4 GPUs for redundant overhead or batch parallelism.
Or: run on 16 GPUs with replicated layers for pipeline depth.

**When.** End of semester, only if HP-BERT delivers and time permits.

---

## 4. Recommended sequencing

### Phase A (Week 1-2): Implement HP-BERT
- Build `bert_hp_multigpu.cu` and `BertHpRunner` orchestration
- Verify correctness against current Phase 4b output (MAE ≤ 2.25e-6)
- Single SLURM measurement run, 5 trials

### Phase B (Week 3): Measure and integrate
- Full 12-layer × 12-head measurement on 4× H100
- Update `experiments/RESULTS.md` (which is dangerously stale)
- Add a row to `docs/RESULTS_SUMMARY.md` Table 2
- Update `paper/main.tex` headline numbers
- Generate before/after figures

### Phase C (Week 4): T-MODUP rescue (optional, if HP-BERT didn't fully close gap)
- Investigate the zero-sized cudaMallocAsync regression
- Land T-MODUP if fixable in 1 week
- Re-measure; integrate into HP-DKS hybrid if successful

### Phase D (Week 5-6): Multi-node stretch
- Configure NCCL for 16-GPU 4-node setup
- Implement Strategy 5 (HP + DKS hybrid)
- Measure and integrate

### Phase E (Week 7-8): Paper rewrite
- Reframe the contribution narrative around the multi-axis parallelism story
- Replace the 5.04×-over-CPU-baseline with HP-BERT's much stronger numbers
- Position DKS as one axis of several, not THE contribution

---

## 5. Expected outcomes (probability-weighted)

| Outcome | Probability | Impact |
|---|---|---|
| HP-BERT works, ~3.5-4× over Phase 4b, ~5-8× over CPU | 70% | Headline-grade result |
| HP-BERT works but only 2-3× | 20% | Defensible but less impressive |
| HP-BERT crashes / doesn't fit memory | 10% | Fall back to defending current |
| T-MODUP also lands | 30% (independent) | Bonus 30% improvement |
| Multi-node HP + DKS lands | 20% (after above) | Best-case headline |

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Phantom not thread-safe across 4 host pinned regions | Investigate with single-GPU smoke test in week 1 |
| PCIe bandwidth contention from 4 streaming GPUs simultaneously | Profile Phase 1 → see if 4 GPUs streaming saturate PCIe; if so, batch the H→D transfers |
| Per-thread CUDA context initialization | Pre-warm contexts in startup; reuse persistent worker threads |
| HP-BERT memory at 248 GB pinned | Verify MN5 ACC node has > 256 GB free RAM at run time |
| Numerical correctness across head boundary | Implement layer-end output comparison vs Phase 4b reference |

---

## 7. Implications for tomorrow's PI presentation

This plan is your defense against the 7% critique. **Don't apologize for
Phase 4b. Frame it as the foundation:**

> "Phase 4b establishes that DKS works correctly at N=65,536 — the digit
> axis parallelism is sound. The 7% speedup over single-GPU pinned is a
> snapshot at one parallelism axis only. We've identified four orthogonal
> axes — digit-axis, head-axis, layer-axis, and batch-axis — and our
> next phase is head-parallelism, which our analysis projects at
> 3.5-4× over Phase 4b and within range of NEXUS's 37 s headline at the
> larger ring degree we operate at. The current paper documents the
> first axis; the next paper will fold in head-parallelism."

This works because:
1. It owns the limitation honestly
2. It demonstrates strategic thinking (you have a plan, not just hopes)
3. It places the current contribution in a coherent multi-axis story
4. It commits to specific numbers you can deliver in a few weeks

---

## 8. Concrete first-week tasks (start tomorrow afternoon)

After the PI talk, kick off Strategy 1:

1. **Smoke test** — verify Phantom can run two simultaneous bootstraps in
   two different processes on two GPUs without crashing. (½ day)
2. **Skeleton** — copy `bert_dks_multigpu.cu` to `bert_hp_multigpu.cu`,
   strip DKS, replace head loop with per-GPU thread dispatch. (1 day)
3. **Memory** — verify pinned host store works per-GPU; measure pinning
   time and total host RAM consumption. (½ day)
4. **Single-head correctness** — run 1 head on 1 GPU through the new
   skeleton, verify output matches reference. (1 day)
5. **Multi-head correctness** — scale up to 3 heads on 1 GPU sequential,
   then 12 heads across 4 GPUs in parallel. Compare each head's output
   against reference. (2 days)
6. **First measurement** — full 12-layer × 12-head run, 3 trials.
   Update RESULTS.md. (1 day)

Target: working HP-BERT measurement by end of week 1.

---

## 9. The number that matters most

**If HP-BERT delivers ~370 s for full BERT-base at N=65,536:**

> "First multi-GPU implementation of full BERT-base inference at
> N=65,536 with no re-encryption, in 370 s on 4× H100. NEXUS achieves
> 37 s but at N=32,768 with re-encryption (which exposes the secret key
> on the compute server). Our number is the first single-N, no-re-enc
> result at the larger ring degree."

That sentence justifies the project. It is achievable in two weeks.
