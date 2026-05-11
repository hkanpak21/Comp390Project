# Section 9 — Conclusion and Future Work

> Status: draft v1
> Slice: WRITE-S9
> Depends-on: WRITE-S2..S8 (depends on the rest of the paper for accuracy of the summary)

## 9.1 Summary

This paper addressed a concrete open problem: NEXUS [CITATION_NEXUS] is
the state-of-the-art non-interactive CKKS [CITATION_CKKS] protocol for
BERT-base inference, but its published artifact (`vendor/nexus/cuda/`)
ships per-operation CUDA kernels without chaining them end-to-end and
without any multi-GPU framework. A direct audit of the vendored tree
finds zero `cudaSetDevice` calls (apart from one hard-coded
`cudaSetDevice(0)` in the Phantom stream constructor [CITATION_PHANTOM]),
zero NCCL calls, zero MPI calls, and zero `std::thread` constructions.
The open problem was therefore not "can we make NEXUS faster" but
"what does multi-GPU even look like for NEXUS, operation by operation,
on real hardware."

Section 4 built NEXUS from source on a single H100 and ran its own
per-operation benchmarks at NEXUS's chosen poly-modulus degrees. This
gave us the *NEXUS-on-H100* column against which every speedup in the
paper is reported, eliminating the A100-to-H100 hardware-normalization
arithmetic. The correctness gate (single-GPU multiNEXUS within $\pm 2\%$
of NEXUS-on-H100 on every operation) licensed the framework
attributions in §6: bootstrap at $\log N = 15$ at $250$ ms vs
$252.8$ ms, LayerNorm at $\log N = 16$ at $45.5$ ms vs $45$ ms,
softmax at $\log N = 16$ at $20$ ms vs $20$ ms, GELU at $\log N = 16$
at $70.30$ ms vs $69$ ms, and argmax at $\log N = 15$, vocab=8 at
$848.4$ ms vs $863$ ms.

Section 5 introduced three multi-GPU strategies and the framework
infrastructure each one rests on. *DKS* (distributed key-switching)
shards key-switch digits across GPUs for a single shared bootstrap;
*HP-BERT* (head-parallel BERT) gives each GPU a subset of attention
heads end-to-end through all 12 layers, with $G$ `std::thread` workers
exchanging activations between heads; and *DP* (data-parallel
per-operation) runs $N/G$ independent op calls per GPU with no
inter-GPU communication during the call. These three strategies are
not interchangeable — each matches a different parallelization regime.

Section 6 placed each of the six NEXUS operations into one of three
typology buckets with profiling-grounded explanations and
profiling-grounded ceilings: *compute-parallel* (MatMul, where
output-channel split is genuine compute parallelism), *transitional*
(GELU and LayerNorm, where per-call compute is large enough to absorb
framework overhead at 4 GPUs), and *data-parallel-throughput*
(bootstrap, softmax, argmax-at-4-GPU, where per-call compute is
comparable to per-rank context-setup so additional GPUs deliver
throughput rather than latency reduction). The headline result —
$8.16\times$ MatMul speedup at 16 GPUs alongside $9$–$22\%$ small-op
16-GPU efficiency — is presented as a typology because no single
aggregate number can honestly summarise both.

Section 7 demonstrated the chained pipeline NEXUS's open source does
not ship. The methodology was a 1-head $\times$ 2-layer unit
measurement on a single GPU at uniform $\log N = 15$, an explicit
saturation check (time per layer 1 matches time per layer 2 within a
stated tolerance), and a multiply-out extrapolation to full BERT. We
then reported the chained pipeline under both parallelization regimes
a reviewer might ask about: HP-BERT strong scaling at 4 and 16 GPUs
(the latency story; $54.27$ s on 16 H100 GPUs at $\log N = 15$, $376$ s
on 4 H100 GPUs at $\log N = 16$) and data-parallel weak scaling at 4
and 16 GPUs (the throughput story). Together they span the two
parallelization regimes a fair reading of the contribution requires.

## 9.2 Goal 1 takeaways

Heterogeneity is the headline. MatMul scales to $8.16\times$ at 16 GPUs
because its parallelization is the disjoint output-channel split,
which has neither inter-GPU communication during the call nor an
amortisable framework overhead. GELU and LayerNorm sit in the middle
($3.55\times$ and $2.56\times$ at 16 GPUs respectively) because their
per-call compute at $\log N = 16$ is large enough to absorb
per-rank context-setup overhead. The small ops — bootstrap, softmax,
and (at 4 GPUs) argmax — are throughput-bound: data-parallel gives
$4\times$ to $16\times$ aggregate throughput as expected, but per-call
latency barely moves because per-rank `PhantomContext` setup time is
comparable to per-call op compute. Bootstrap at $\log N = 15$ stays
near $192$ ms per call at 16 GPUs (down from $250$ ms single-GPU) —
useful for throughput, but not the latency reduction that more GPUs
naively promise.

The per-rank `PhantomContext` setup time is the dominant ceiling for
small ops at 16 GPUs. Each rank currently rebuilds its full Phantom
context object per op-call: this re-allocates NTT tables, regenerates
Galois key tables, and reconstructs internal twiddle factors. The
profiling traces in §6 show this overhead as a flat $\sim 3.7$ s
setup interval per rank on the argmax microbenchmark, and a smaller
but proportionally larger fraction on softmax and bootstrap. Per-rank
context pooling — sharing a single Phantom context across many op
calls within a rank — would lift this efficiency floor; we list it as
future work in §9.4. It is genuinely out of scope for this paper
because Phantom's NTT and rotation key tables were not designed for
thread-safe reuse, and the correctness analysis needed to share them
across calls is multi-day work in its own right.

## 9.3 Goal 2 takeaways

The saturation check at uniform $\log N = 15$ is what makes the
multiply-out extrapolation honest. The 1-head $\times$ 2-layer unit
measurement showed that time-per-layer for layer 2 matches
time-per-layer for layer 1 within the stated tolerance, demonstrating
that the chained pipeline reaches steady-state by the second layer.
Once steady-state is reached, full BERT is $12 \times 12 \times
t_{\text{head,layer}}$ by construction; we do not extrapolate from a
warm-up regime where the modulus chain has not yet stabilised, and we
do not infer the cost of layer 12 from layer 1. The full extrapolated
number lies within the stated tolerance of the directly-measured
HP-BERT pipeline at $\log N = 15$ on 16 GPUs ($54.27$ s, JOBID
40366927).

HP-BERT strong-scaling at 4 GPUs and 16 GPUs covers the latency story
($376$ s $\to 54.27$ s; the four-node 16-GPU measurement is the
tightest single-pipeline number this paper reports). Data-parallel
weak-scaling at 4 GPUs and 16 GPUs covers the throughput story ($G$
concurrent independent inferences in approximately the wall-clock of
a single one, plus framework overhead). Together they span both
parallelization regimes any reviewer would ask about for an end-to-end
result, with the explicit acknowledgement (§8) that neither beats
NEXUS's published $37.3$ s on $4\times$ A100 on a fair workload —
because $37.3$ s depends on slot-axis SIMD packing that NEXUS's open
source does not ship and we did not implement this semester.

## 9.4 Future work (priority order)

1. **Slot-axis SIMD packing for HP-BERT.** This is the gating follow-up
   to the work in this paper. NEXUS's published $37.3$ s end-to-end
   number on $4\times$ A100 depends on packing all 12 attention heads
   into a single ciphertext (Algorithm 3 of [CITATION_NEXUS]),
   reducing the number of bootstraps from $\Theta(\text{heads} \times
   \text{layers})$ to a small constant. Without that packing, no
   amount of GPU parallelism beats $37.3$ s on a fair workload. The
   refactor touches encoding, the rotation pattern, and bootstrap
   scheduling simultaneously; it is multi-day work and was the
   single largest scope item cut from this paper.

2. **Multi-cipher argmax tournament.** Required for an
   apples-to-apples comparison against NEXUS's published $2.48$ s at
   vocab=30,522. Our `argmax_align_n32k` binary handles a single
   sparse-encoded ciphertext, capping the vocabulary it can compare at
   `sparse_slots` $= 8{,}192$ at $\log N = 15$. The tournament path
   (argmax-of-argmaxes across multiple ciphertexts combined in a final
   round) is conceptually straightforward and the per-cipher argmax
   algorithm is already in NEXUS's open source; what is missing is the
   tournament wiring, the cross-cipher selection logic, and the
   correctness gate against a plaintext reference at the full
   vocabulary.

3. **Per-rank context pooling.** Would lift small-op 16-GPU efficiency
   from $9$–$22\%$ to a projected $30$–$50\%$ for softmax, LayerNorm,
   and GELU. Implementation requires a careful thread-safety analysis
   of Phantom's shared NTT tables and rotation key tables (CLAUDE.md
   lesson \#10), and a re-validation that the pooled context produces
   bit-identical results on the relevant correctness gates. We have
   the per-rank profiling traces that show this is the right
   intervention; the work that remains is the safety analysis and the
   gates, not the design.

4. **Fix the high-severity audit findings.** Appendix A.6 catalogues
   6 BLOCKER and 14 HIGH FIX slices from the four BUG-01..04 audits.
   The single largest critical-path win is removing the leftover
   `fprintf` + `cudaDeviceSynchronize()` debug calls in
   `Bootstrapper::bootstrap_sparse_3` (lines 3043–3107), which
   collapse the H$\leftrightarrow$D-overlap the eight prefetch hooks
   were designed to provide. The remaining HIGHs concentrate in three
   classes: missing or weak MAE gates on the align binaries, stream
   /communicator destruction ordering in `MultiGpuContext::destroy()`,
   and a small number of unprofiled rotation-key copy paths. None of
   them invalidate the current paper measurements (the BLOCKERs are
   about future safety, not retroactive correctness), but they are
   the next maintenance pass before any artifact release.

5. **Layer-pipeline parallelism.** Different layers on different GPUs.
   This is a fundamentally different parallelization regime from
   HP-BERT (which is head-axis) and from data-parallel (which is
   inference-axis); the trade-off is pipeline-bubble latency vs
   inter-layer activation transfer, which depends on the modulus chain
   depth at each transfer point. We mention it only as an alternative
   future direction, not as a planned next step — HP-BERT plus
   data-parallel already span the two regimes the chained pipeline
   needs, and slot-axis SIMD packing has higher priority.

6. **HP-LLaMA.** This paper is BERT-only. LLaMA experiments exist in
   the source tree (`src/benchmarks/llama_hp_multigpu.cu`,
   `llama_hp_multinode.cu`), but the algorithmic analogues to BERT's
   per-op structure are different: decoder-only autoregression
   introduces a KV cache that interacts non-trivially with bootstrap
   scheduling, and the per-op typology of §6 would need to be redone
   from scratch. We list it last because the paper's two contributions
   are about the *typology* and the *chained pipeline*, both of which
   are independent of the underlying transformer family but were
   evaluated only on BERT-base.

## 9.5 Closing

The two contributions of this paper — the per-operation multi-GPU
typology against a NEXUS-on-H100 baseline, and the chained end-to-end
BERT pipeline at uniform $\log N = 15$ — were chosen because, in
combination, they answer a question the FHE-on-GPU literature has so
far reported in aggregate: *which operations actually benefit from
more GPUs, and by how much, when chained into a real pipeline on real
hardware?* We hope that the typology in §6 becomes a starting point
for future FHE-on-GPU work that confronts operation-level heterogeneity
directly rather than reporting a single aggregate speedup figure that
hides which of the constituent operations did the work.
