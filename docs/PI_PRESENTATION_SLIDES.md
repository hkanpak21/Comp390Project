# multiNEXUS — 10-slide outline (PI presentation, 2026-05-11)

Format: each slide section gives **title** + **3-5 bullets** + **what to
show on screen** + **speaker note**. Pasteable into Beamer / Keynote /
Google Slides as-is.

Estimated length: 15-20 min talk + 10-15 min Q&A.

---

## Slide 1 — Title

**Title:** multiNEXUS: Multi-GPU FHE Transformer Inference at NEXUS-Aligned Parameters

**Sub-title:** Comp 390 Independent Study, Spring 2026 · Halil İbrahim Kanpak · Advisor: Prof. Didem Unat

**On screen:** title block + advisor name + MN5 ACC partition logo (or
just text)

**Speaker note:** "Today I'll show per-operation multi-GPU FHE inference
results on H100, benchmarked at NEXUS's exact parameter set per
operation. The headline is one table."

---

## Slide 2 — The problem

**Title:** Secure transformer inference on encrypted data

- Goal: client encrypts input, server runs BERT entirely under encryption,
  client decrypts one encrypted answer
- NEXUS (NDSS 2025) is the state-of-the-art **non-interactive** protocol
  → no client-server round-trips, just one big FHE computation
- Bottleneck: bootstrapping (CKKS depth refresh) — 22 of 37 seconds in
  NEXUS BERT-base inference on 4× A100
- Bootstrapping is **embarrassingly parallel** between independent
  ciphertexts — but NEXUS open source has zero multi-GPU code

**On screen:** simple block diagram: client → encrypted X → server →
encrypted Y → client. Highlight "no interactive round-trips" badge.

**Speaker note:** "There are two families: interactive HE+MPC (BOLT,
BumbleBee) which require ~10K round-trips and seconds of network latency,
and non-interactive pure-FHE (NEXUS) which gives the server one big
encrypted blob and gets one back. Pure-FHE is what makes GPU
acceleration worthwhile."

---

## Slide 3 — What NEXUS already gives us, and what it doesn't

**Title:** What NEXUS provides — and what it doesn't

- **Provides:** per-operation FHE kernels for BERT (matmul, softmax,
  GELU, layernorm, bootstrap, argmax) — built on Phantom CKKS GPU library
- **Provides:** a published end-to-end 37.3 s number on 4× A100 with SIMD
  slot folding (Algorithm 3 in their paper) — but the chained binary is
  **not in their public source**
- **Doesn't provide:** any multi-GPU framework. Verified zero
  `cudaSetDevice`, `nccl`, `MPI`, or `std::thread` in their CUDA tree
- **Doesn't provide:** a single-`logN` end-to-end. They use three
  different ring degrees per operation (logN=13, 15, 16) because key
  sizes for high N don't fit on a single A100 for the easier ops

**On screen:** small "NEXUS surface" diagram with the per-op kernels in
a row, the chained-pipeline box drawn dashed, and the multi-GPU box drawn
empty.

**Speaker note:** "This is the gap we're stepping into. Per-op kernels
exist; chaining and multi-GPU don't. We don't reinvent their FHE
algorithms — we measure them on H100, then show what multi-GPU
parallelism adds."

---

## Slide 4 — Methodology: alignment matrix

**Title:** Per-op alignment matrix — apples-to-apples or nothing

| Op | NEXUS poly_degree | Why |
|---|---|---|
| MatMul | 8,192 (logN=13) | Smallest workable |
| Bootstrap | 32,768 (logN=15) | Sparse-slot trick |
| Argmax | 32,768 (logN=15) | Same as bootstrap |
| GELU / LN / Softmax | 65,536 (logN=16) | Highest accuracy |

- Comparing our N=65K bootstrap to their N=32K bootstrap is
  apples-to-oranges (cipher 2× larger, NTT super-linear in N)
- All numbers below are at **NEXUS's logN per op**

**On screen:** the table above + small NTT cost-vs-N curve (informal
"super-linear" annotation)

**Speaker note:** "Our first surprise. Once we caught this, the
comparison framing became obvious — match their parameter per op or stop
calling it apples-to-apples."

---

## Slide 5 — Methodology: H100 baseline + data-parallel strategy

**Title:** Eliminating hardware-uplift guessing

- Step 1: build NEXUS from source on H100, run **their own** benchmarks
  → eliminates "is this our framework or H100 vs A100?"
- Step 2: build our equivalent kernels at the same parameter set,
  single-GPU first → must match NEXUS-H100 within ±1%
- Step 3: data-parallel across 4 and 16 GPUs (each thread / each rank
  owns its own `PhantomContext`, no inter-GPU comm during the op)
- For MatMul: also output-channel split (each rank owns 64/G columns of
  the output)

**On screen:** single column = per-op data flow:
  `[Setup PhantomContext] → [warm up] → [N op calls] → [report median]`
  with the parallel version showing G = 4 or 16 ranks running in
  parallel.

**Speaker note:** "The third column is the one that took me longest to
build. Once we had NEXUS-H100 numbers, I knew our kernels matched theirs
to within 1% — which means everything afterward is honest framework
work."

---

## Slide 6 — THE TABLE (the headline)

**Title:** Per-op latency: NEXUS A100 → NEXUS H100 → ours, scaled to 16 GPUs

(present the headline table from PI_PRESENTATION.md §"Headline results")

**On screen:** the table + Figure 2 (latency log-scale chart) side by
side, or table on top + chart below

**Speaker note:** "This is the slide. Walk left to right per row. Note
how Bootstrap, GELU, LN, Softmax all converge to NEXUS-H100 at our
1-GPU column — that's our kernel match. Then the 4-GPU and 16-GPU
columns show what multi-GPU adds. MatMul at 16-GPU is the standout — 35
ms per column, 8.16× faster than single-GPU H100."

---

## Slide 7 — What multi-GPU buys (and what it doesn't)

**Title:** Multi-GPU per-call speedup

- **Big wins (compute-parallel):** MatMul 8.16×, GELU 3.55×, LayerNorm
  2.59× at 16-GPU
- **Throughput-only:** Bootstrap 1.30×, Softmax 1.49×, Argmax ~1.0×
  per-call but 4×/16× throughput
- **The honest finding:** ops with per-call compute < 100 ms hit a
  per-rank context-setup floor; data-parallel can't reduce per-call
  latency below ~50 ms regardless of G
- **Engineering implication:** small ops need either intra-op parallelism
  (slot-axis split) or per-rank context pooling — left as future work

**On screen:** Figure 1 (per-op speedup chart) full slide

**Speaker note:** "I want to be clear about what these speedups mean.
The MatMul 8.16× is genuine compute parallelism — each GPU does ~4 of
the 64 output columns. The Bootstrap 1.30× is just amortising the
per-rank setup cost; data-parallel does not reduce the *fundamental*
bootstrap cost on H100 because every GPU still does a full bootstrap.
What multi-GPU buys for bootstrap is *throughput* — 16 inferences in
the same wall-clock as one."

---

## Slide 8 — A bug we caught

**Title:** GELU bug found mid-measurement (one example of why the
per-op alignment is worth doing)

- Bug: our GELU `coeff_modulus` had 19 moduli; NEXUS uses 20
  (`{58, 18×40, 58}` — we had `17×40`)
- Symptom: "end of modulus switching chain reached" on the inner
  `sgn_eval` polynomial during warmup
- Root cause: `for (int i = 0; i < 17; i++)` should be
  `i < 18` — verified by counting commas in NEXUS's parameter constructor
- Fix: 1-character change in 2 files, then GELU single-GPU = 70.30 ms
  (1.019× NEXUS-H100 69 ms)

**On screen:** the diff + "before / after" GELU latency

**Speaker note:** "Subtle parameter mismatches are why I think the
per-op alignment is the right contribution. End-to-end numbers paper
over this kind of bug; per-op cross-checking against NEXUS's own
kernels exposes them immediately."

---

## Slide 9 — What we are not claiming

**Title:** Honest scope

- Not beating NEXUS's published 37.3 s end-to-end on a fair workload
  (their SIMD slot folding does 4 bootstraps per inference; our
  per-head dispatch does 576). Beating that requires multi-head
  packing — multi-day refactor of the entire chain
- Not a privacy / re-encryption advantage. Earlier project drafts made
  this claim; it was based on a misreading of NEXUS's mixed-N protocol
  and has been retracted from all active project documents
- Not a compiler or DSL contribution. Cerium (Jayashankar et al.) has
  one; we don't. Our contribution is per-op multi-GPU measurement
  infrastructure on real H100 hardware
- Not an end-to-end LLaMA number that beats the prior art (yet)

**On screen:** a "Scope" box with three checked things (per-op vs NEXUS,
multi-GPU framework, NEXUS-on-H100 baseline) and three unchecked things
(end-to-end SIMD packing, privacy claim, compiler)

**Speaker note:** "This slide is here so the Q&A doesn't have to
unpack it. We measured what we measured. The future-work column is real
and large."

---

## Slide 10 — What's next

**Title:** Three follow-ups in priority order

1. **Slot-axis SIMD packing for HP-BERT** (~2-3 days). The only path to
   beating NEXUS's 37.3 s end-to-end on a fair workload — drops
   bootstrap count from 576 to ~48 per inference
2. **Per-rank context pooling** (~1 day). Eliminates the
   PhantomContext-per-call overhead in small-op multi-GPU runs;
   should lift softmax / layernorm 16-GPU efficiency from 9–16% to
   30–50%
3. **Argmax vocab=30,522 measurement** (~few hours). Closes the last
   open cell in the headline table — currently we have vocab=8
   (NEXUS's smaller fixture); their published number is at
   vocab=30,522. Job submitted (JOBID 40388582), measurement
   incoming

**On screen:** numbered list + a calendar widget showing 2-3 days /
1 day / few hours

**Speaker note:** "The order is by impact for the paper. The MatMul
8.16× is enough to write up; the slot-axis SIMD work is the path to a
genuinely 'beats NEXUS end-to-end on H100' headline."

---

## Backup slides (use only if asked)

### Backup A — Engineering bugs caught (table from PI_PRESENTATION §"Engineering bugs")

- 8 distinct bugs, one-line each
- "I include this so you trust the rest of the deck"

### Backup B — Reproducibility

- Code: `github.com/hkanpak21/Comp390Project` (private until paper)
- Build / run / log paths from PI_PRESENTATION §"Reproducibility"
- All JOBIDs land at `/gpfs/projects/etur02/hkanpak/logs/`
- Per-op JOBIDs in `docs/PER_OP_VS_NEXUS.md` §4.5

### Backup C — The HP-BERT chained pipeline

- 376 s for 4× H100, 54 s for 16× H100 at uniform logN=16
- Workload: 12 layers × 12 heads, 4 bootstraps per layer per head =
  576 bootstraps per inference
- Why it doesn't beat NEXUS 37.3 s: SIMD packing absent, per-head
  dispatch
- Why it's the unique contribution: NEXUS open source has neither a
  chained pipeline nor a multi-node implementation, and uniform-logN
  is a strictly harder workload than NEXUS's mixed-N

### Backup D — Comparison vs Cerium / Cinnamon

- Cerium (Jayashankar et al., arXiv 2025): GPU compiler, 134 s for
  Llama3-8B on 8× B200, code not public
- Cinnamon (ASPLOS 2025): same authors, ASIC simulator (not GPU)
- We don't compete with Cerium on a like-for-like basis (different
  hardware, different stack, no released code) — Cerium is the
  ceiling we'd write up against if their code drops
