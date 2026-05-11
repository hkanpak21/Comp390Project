# Section 8 — Discussion

> Status: draft v1
> Slice: WRITE-S8
> Depends-on: WRITE-S6 (skeleton), WRITE-S7 (skeleton); will be refined as those sections firm up

This section surfaces, in one place, the limitations a reviewer or the
PI will reasonably probe. We expand the three disclosures previewed in
§2.4, recap one ceiling already disclosed in §6, summarise what the
per-op bug-audits in Appendix A.6 turned up, and state the threats to
validity. Throughout, the framing is "disclosed openly": none of these
are bugs, and all are out of scope rather than open work the paper
claims to have closed.

## 8.1 What this paper does and does not claim

We claim a *per-operation* multi-GPU typology on H100, with single-GPU
multiNEXUS measurements matching a NEXUS-on-H100 baseline within
$\pm 2\%$ on every operation (§4.5), 4-GPU and 16-GPU latencies for
each of six operations, and a profiling-grounded ceiling explanation
for each (§6); and a *chained end-to-end* BERT inference at uniform
$\log N = 15$ — the pipeline NEXUS's open source does not ship (§7) —
grounded in a 1-head × 2-layer unit measurement plus saturation check,
exercised under head-parallel strong scaling and data-parallel weak
scaling. Both contributions live on real H100 hardware (BSC
MareNostrum 5 ACC partition, §2.5 and §4.3), and every reported number
is reproducible from a SLURM script under `scripts/mn5/` with JOBID +
log-path provenance in Appendix A.

We do *not* claim a wall-clock-fastest end-to-end number on a workload
that apples-to-apples matches NEXUS's published 37.3 s on 4× A100:
NEXUS's 37.3 s depends on slot-axis SIMD packing (their Algorithm 3)
that the open source does not include and that we did not reimplement
this semester (§8.3). The fairest comparison we *can* draw — same
algorithms, same H100 silicon — is per-operation (§4.5, §6), not
end-to-end. We likewise do not claim apples-to-apples argmax at NEXUS's
published vocabulary of 30,522 (§8.2), nor a head-to-head against
Cerium [CITATION_CERIUM] (§8.5).

## 8.2 Multi-cipher argmax gap (limitation)

NEXUS's published end-to-end inference outputs an argmax over the
BERT-base WordPiece vocabulary of 30,522 tokens, padded to a power of
two of $2^{15} = 32{,}768$ slots. At NEXUS's $\log N = 15$ argmax
configuration, the sparse encoding fits 8,192 slots per ciphertext, so
the 32,768-slot vocabulary spans four ciphertexts. NEXUS's open-source
artefact handles the full vocabulary via a *multi-cipher tournament*:
run a per-cipher argmax over each 8,192-slot shard in parallel, then
combine the per-cipher maxima in a final tournament reduction. The
tournament logic is not present in `vendor/nexus/cuda/src/main.cu`'s
argmax test, which exercises only a single-ciphertext vocab = 8 case
(`docs/PER_OP_VS_NEXUS.md`, §4.4 argmax footnote; `paper/sections/04_identifying_nexus.md` §4.5 note $^{\ddagger}$).

Our binary `src/benchmarks/argmax_align_n32k.cu` likewise only handles
single-cipher inputs. It now FATAL-refuses with an explicit
input-validation message when `vocab > sparse_slots = 8{,}192` at
$\log N = 15$, rather than segfaulting on out-of-bounds slot indexing
(this is non-negotiable lesson #10 in `CLAUDE.md`). This is *by
design*: a multi-cipher tournament is a different argmax algorithm —
not a kernel extension — and we chose to refuse cleanly rather than
emit silently wrong results.

The implication for the paper is that the apples-to-apples comparison
with NEXUS at vocab = 30,522 cannot be drawn from our numbers. We
report two argmax cells in §4.5 and §6.3.argmax: vocab = 8 (matching
NEXUS's own bundled test, single-GPU H100 = 848.4 ms versus
NEXUS-on-H100 863 ms, $\Delta = -1.7\%$) and vocab = 8,192 (our maximum
single-cipher comparison, JOBID 40397435 [TODO: confirm cell after job
lands]). The vocab = 30,522 cell against NEXUS's published 2.48 s is
necessarily empty in our comparison table. Implementing the
multi-cipher tournament — split the vocabulary across ciphertexts,
per-cipher argmax in parallel, final tournament — is a roughly
one-day follow-up, flagged in §9 as a future-work item.

## 8.3 No slot-axis SIMD packing for HP-BERT (limitation)

A single BERT-base inference processes 128 input tokens at hidden
dimension 768. CKKS ciphertexts at $\log N = 15$ admit 16,384 plaintext
slots; at $\log N = 16$ they admit 32,768. NEXUS exploits this slot
abundance by packing all 12 attention heads of a single layer into the
slot axis of *one* ciphertext (their Algorithm 3), reducing the
per-layer bootstrap count from $\Theta(\mathrm{heads})$ to $\Theta(1)$.
Because bootstrap dominates the end-to-end critical path at ≈250 ms
per call at $\log N = 15$ (§6.bootstrap), the slot-packing optimisation
is the single largest contributor to NEXUS's published 37.3 s
end-to-end on 4× A100.

Our HP-BERT (`src/benchmarks/bert_hp_multigpu.cu`,
`src/benchmarks/bert_hp_multinode.cu`) does *not* slot-pack across
heads: each head's computations occupy a private ciphertext per slot
bank, leaving the majority of the slot axis idle. This is the right
trade-off for the head-parallel strong-scaling story (one head per
ciphertext maps cleanly to one head per GPU) but it forfeits the
per-layer bootstrap reduction. Without slot packing, no amount of GPU
parallelism beats NEXUS's 37.3 s on a fair workload — adding GPUs
reduces the bootstrap *count per GPU* but not the bootstrap *count
per inference*, and the latter is what 37.3 s reflects.

Adding slot-axis SIMD packing for HP-BERT would compress the per-call
work into fewer ciphertexts and is expected to close most of the gap
to NEXUS's 37.3 s end-to-end, even before adding GPUs. The refactor is
multi-day — it touches the head-to-ciphertext encoding scheme, the
rotation pattern that drives the attention dot-product, and the
bootstrap scheduling — and was not in scope for this semester
(`docs/PI_REPORT.md` "What is left" item 1; `docs/prd/PRD-multiNEXUS-paper.md`
"Out of Scope"). We flag this in §9 as the highest-priority follow-up
item: it is the gating optimisation to a fair-workload wall-clock
comparison.

## 8.4 Per-rank context-setup ceiling for small ops (limitation, profiling-grounded)

The data-parallel-per-op strategy (§5.4) assigns each GPU thread its
own `PhantomContext`, which is what makes the per-call execution
thread-safe under our Phantom modifications (Appendix A, Phantom
modifications #3 and #4). The cost of this design is that each rank
pays a fixed *per-rank context-setup* overhead — roughly 3.7 s for
argmax (`docs/PI_REPORT.md`, "Results" §, argmax footnote) and
commensurate but smaller for bootstrap, GELU, LayerNorm, softmax — that
does *not* amortise across calls when the number of calls per rank is
small.

For operations whose per-call compute is large (MatMul at ≈285 ms
per output column, GELU at ≈70 ms, LayerNorm at ≈45 ms), the
setup overhead is a small fraction of the per-rank wall-clock and the
DP strategy delivers near the expected speedup. For operations whose
per-call compute is small relative to that ceiling (bootstrap at
≈250 ms at $\log N = 15$, softmax at ≈20 ms at $\log N = 16$,
single-cipher argmax in the millisecond range), the setup overhead is
a non-trivial fraction of the per-rank wall-clock, and the 16-GPU
efficiency caps at the 9–22% range reported in §6. The §2.4 preview
called this out, the §6 per-op subsections quantify it under the
"profiling-grounded ceiling" field of the 6-field template, and we
recap it here in one place for the discussion.

The architectural fix is *per-rank context pooling*: share one
`PhantomContext` across multiple ranks with explicit thread
synchronisation, amortising the 3.7 s setup over many calls. The
expected lift is from 9–22% to roughly 30–50% 16-GPU efficiency for
the small ops (`docs/PI_REPORT.md`, "What is left" item 2). It is a
roughly one-day refactor and is also out of scope for this paper;
flagged in §9.

## 8.5 Position vs Cerium and Cinnamon

Two related multi-GPU FHE works frame the comparable-art landscape but
neither admits a head-to-head numeric comparison on H100.

**Cinnamon** [CITATION_CINNAMON] (Jayashankar et al., ASPLOS 2025) is
a Python → ASIC-ISA compiler; its multi-accelerator numbers come from
cycle-accurate architectural simulation, not real-hardware execution.
We draw on Cinnamon's algorithmic decomposition — the $\beta$-digit
key-switch split that motivates DKS in §5.2 — as algorithmic
precedent, but the substrates differ so no direct numeric comparison
is attempted.

**Cerium** [CITATION_CERIUM] (Jayashankar, Chen, Zheng, Skarlatos,
arXiv 2025) is the GPU sibling of Cinnamon. Code is not public as of
2026-05; we cannot reproduce its numbers on H100. Our HP-BERT is
closest in spirit to Cerium's published head-parallel architectural
diagrams, but the comparison is necessarily qualitative. Should Cerium
open-source its artefact, the HP-BERT 16-GPU number in §7.4 becomes a
candidate direct comparison.

Our positioning is therefore:

- We are *not* the fastest open-source GPU FHE BERT artefact at the
  per-operation level — NEXUS is, on the per-op data-parallel-throughput
  cells (§6).
- We *are* the first open-source artefact, on real GPU hardware, to
  ship a chained end-to-end BERT pipeline at uniform $\log N = 15$
  with both head-parallel strong scaling (§7.4) and data-parallel
  weak scaling (§7.5) reported under like-for-like methodology. The
  chained pipeline at uniform $\log N$ is the deliverable NEXUS's
  published open source cannot produce.

## 8.6 What the per-op audits revealed (paper credibility note)

Before paper writing began, we ran four bug-audit slices (BUG-01 through
BUG-04, Appendix A.6) on critical-path code: the six per-op alignment
binaries (BUG-01), the head-parallel BERT binaries (BUG-02), the
`src/multi_gpu/` framework (BUG-03), and the NEXUS evaluator wrappers
in `src/nexus_eval/` (BUG-04). The audits surfaced 48 follow-up FIX
slices across the four lanes [TODO: confirm exact totals once BUG-NN
audit summary tables in Appendix A.6 are finalised; observed severity
counts from the audit files are roughly 6 BLOCKER / 14 HIGH / 14 MEDIUM
/ 14 LOW]. None of the BLOCKER or HIGH findings affect a number
reported in the paper *as-shipped*; two findings do affect the framing
of two specific claims and we disclose both here.

**Bootstrap debug-print synchronisation (BUG-04, HIGH).** The audit
identified roughly seven `fprintf(stderr, …) + cudaDeviceSynchronize()`
debug-print pairs that remain in `src/nexus_eval/bootstrapping/Bootstrapper.cu`
between lines 3043 and 3107 (BUG-04, item "Bootstrapper.cu:3043…3107").
Each pair collapses the H↔D async-prefetch overlap delivered by the
eight prefetch hooks we added in Phase 3, because
`cudaDeviceSynchronize` is a full device barrier. The implication for
the paper's bootstrap critical-path argument (§6.bootstrap) is that
the reported single-GPU H100 bootstrap latency reflects what NEXUS
would achieve *without* our prefetch overlap — the prefetch headroom
is in fact larger than reported. We disclose this openly; removing the
debug prints is a FIX-04 slice slated as the highest-value
single-binary improvement.

**MAE-gate coverage holes (BUG-01, MEDIUM; BUG-02, MEDIUM).** The
audit found that four of the six per-op single-GPU alignment binaries
lack an explicit end-of-run MAE check against the single-GPU reference
output (BUG-01, "MAE-gate coverage" finding) and that `bert_hp_multinode`
likewise lacks an end-of-run MAE gate (BUG-02). The numerical
correctness of the affected binaries is verified by (i) code review of
the underlying NEXUS kernels (which are unmodified) and (ii) the
bootstrap-internal MAE gate that lives inside `Bootstrapper.cu`'s
output validation, but *not* by an end-to-end assertion in each binary
itself. We disclose this; FIX slices to add MAE gates to every
critical-path binary are listed in Appendix A.6.

The paper's per-op latency numbers are not affected by either of these
findings — both are about what is *gated*, not about what is *reported*.
We surface them so a reviewer can see we have looked.

## 8.7 Threats to validity

Three threats to the external validity of the paper's measurements:

- **Single hardware platform.** Every number in §4–§7 comes from the
  ACC partition of BSC MareNostrum 5 (4× H100 64 GB SXM5 per node,
  NVSwitch all-to-all, InfiniBand multi-node). Results on PCIe H100,
  on systems with eight rather than four GPUs per node, or on systems
  without NVSwitch, may differ — particularly for the multi-GPU
  per-call latencies in §6 that depend on the NVSwitch all-to-all
  topology to keep `ncclAllReduce` cost flat.
- **Median-of-3 reporting; not all ops triple-trialled.** The paper
  reports median-of-3 measurements where three independent SLURM runs
  exist. For some op + GPU-count cells we have only single-trial
  numbers so far; §6 marks those cells explicitly as
  `(n=1; not yet repeated)` rather than reporting them as
  median-of-three. The implication for downstream variance bounds is
  that single-trial cells should be read with wider uncertainty than
  median-of-three cells.
- **Saturation tolerance is a tunable parameter.** The Goal-2
  saturation check in §7 verifies that layer-2 timing matches layer-1
  timing within a 5 % relative tolerance, which licenses the
  full-BERT extrapolation by multiplication. The 5 % threshold is
  conservative for chained-pipeline drift in our setup but is not
  derived from a noise model; tightening it (e.g. to 2 %) might fail
  the saturation gate on currently-saturated runs and would force a
  longer unit-measurement methodology. We mark the threshold
  explicitly in §7 so the choice is visible.

A fourth, narrower threat: the §4.5 correctness gate ("multiNEXUS
single-GPU $\equiv$ NEXUS-on-H100 within $\pm 2\%$") is a tolerance,
not a numerical-equality check. We chose $\pm 2\%$ because it covers
measured run-to-run jitter on ACC while staying tight enough to catch
a different underlying algorithm. The MAE gates in Appendix A.6 are
the stricter check we *do* run on the alignment binaries that have
them; their thresholds (typically $\le 10^{-5}$ vs reference) are an
order of magnitude tighter than the timing-based $\pm 2\%$ in §4.5.

## 8.8 Summary

The three architectural limitations disclosed in §2.4 — the
multi-cipher argmax gap (§8.2), the absence of slot-axis SIMD packing
for HP-BERT (§8.3), and the per-rank context-setup ceiling for small
ops (§8.4) — are not bugs. Each is an architectural design point with a
known follow-up; together they form the near-term roadmap §9 picks up.
The audit findings in §8.6 are the analogous transparency item for the
paper's quantitative claims, and §8.7 states the conventional
single-platform / measurement-noise caveats. The discussion does not
change the contributions claimed in §2.2; it calibrates them.
