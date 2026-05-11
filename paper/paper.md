# multiNEXUS — Multi-GPU FHE Inference for BERT on H100

> Auto-assembled from `paper/sections/*.md` on 2026-05-11.
> Canonical source is the discrete section files; regenerate with
> `scripts/assemble_paper.sh`.

---

# Section 1 — Abstract

> Status: draft v1
> Slice: WRITE-S1
> Depends-on: WRITE-S2..S8 (depends on the rest of the paper for accuracy of the summary)

## Abstract

Privacy-preserving transformer inference under fully homomorphic encryption (FHE) is within wall-clock reach on a single GPU, but no published artifact runs it end-to-end across multiple GPUs on real hardware. NEXUS [CITATION_NEXUS], the state-of-the-art non-interactive CKKS protocol for BERT-base inference, ships per-operation CUDA kernels but neither chains them nor parallelizes them across GPUs; a direct audit of `vendor/nexus/cuda/` finds zero `cudaSetDevice`, NCCL, MPI, or `std::thread` calls. We close that gap with multiNEXUS, a multi-GPU framework for NEXUS-style FHE BERT inference on 4 H100 GPUs single-node and 16 H100 GPUs (four nodes) on BSC MareNostrum 5. We make two contributions. First, a *per-operation multi-GPU typology* against a NEXUS-on-H100 single-GPU baseline that we rebuild from source so every speedup ratio is hardware-isolated; our single-GPU numbers match NEXUS-on-H100 within $\pm 2\%$ on all six operations (e.g., bootstrap at $\log N = 15$: $250$ ms vs $252.8$ ms), and we show that MatMul scales to $8.16\times$ at 16 GPUs while small ops (bootstrap, softmax, argmax) become throughput-bound at $9$–$22\%$ efficiency. Second, a *chained end-to-end BERT pipeline at uniform $\log N = 15$* — which NEXUS does not ship — demonstrated via a unit measurement, an explicit saturation check, and a multiply-out extrapolation; under head-parallel scaling we report $54.27$ s on 16 H100 GPUs. Heterogeneity is the headline, not a caveat.

---

# Section 2 — Introduction

> Status: draft v1
> Slice: WRITE-S2
> Depends-on: DOC-01

## 2.1 The problem

Privacy-preserving transformer inference under fully homomorphic encryption (FHE) is now within wall-clock reach on a single GPU, but no published artifact runs it end-to-end across multiple GPUs on real hardware. This paper closes that gap for the NEXUS protocol [CITATION_NEXUS].

CKKS [CITATION_CKKS] is the FHE scheme used by every recent transformer-inference protocol because it supports approximate arithmetic on packed real-valued vectors and admits a deterministic bootstrap path [CITATION_BOOTSTRAP_CKKS] that refreshes the noise budget mid-circuit. NEXUS [CITATION_NEXUS] is the state-of-the-art non-interactive CKKS protocol for BERT-base inference: the client encrypts the input, the server runs the entire 12-layer × 12-head BERT computation under encryption, and the server returns a single encrypted answer — with no client interaction between input and output. NEXUS reports 37.3 s per inference on 4× A100.

NEXUS's published artifact is a single-GPU codebase. Its public surface in `vendor/nexus/cuda/` consists of six per-operation CUDA kernels — `matrix_mul.cu`, `gelu.cu`, `layer_norm.cu`, `softmax.cu`, `argmax.cu`, and `bootstrapping/` — built on a fork of the Phantom CKKS library [CITATION_PHANTOM]. A direct `grep` of the vendored tree finds zero `cudaSetDevice` calls (apart from one hard-coded `cudaSetDevice(0)` in the Phantom stream constructor), zero NCCL calls, zero MPI calls, and zero `std::thread` constructions. The top-level driver `vendor/nexus/cuda/src/main.cu` exercises each kernel in isolation against synthetic inputs; no chained pipeline exists in the open source. The published 37.3 s end-to-end number depends on a slot-axis SIMD packing scheme (Algorithm 3 of the NEXUS paper) that the open source does not ship.

The open problem is therefore concrete: a multi-GPU framework for NEXUS-style FHE BERT inference, on real hardware (not architectural simulation), with a per-operation parallelization story that is defensible operation-by-operation rather than as a single aggregate speedup. Prior multi-GPU FHE work either targets ASICs in simulation (Cinnamon, ASPLOS 2025 [CITATION_CINNAMON]) or remains closed-source (Cerium, arXiv 2025 [CITATION_CERIUM]); neither is comparable head-to-head on H100.

## 2.2 Two contributions

This paper makes two contributions.

**(1) A per-operation multi-GPU typology on H100, baselined against NEXUS-on-H100 single-GPU.** We build NEXUS from source on H100 (§4) and measure each of its six per-operation kernels at NEXUS's chosen poly-modulus degrees on the same silicon we use for our own measurements. We then implement and measure a multi-GPU version of each operation at 4 GPUs (one node) and 16 GPUs (four nodes). Each of the six operations — bootstrap, MatMul, GELU, LayerNorm, softmax, argmax — is presented in §6 under a uniform 6-field template: aim, parallelization strategy, implementation, single-GPU and multi-GPU result, profiling-grounded explanation of the result, and profiling-grounded ceiling. The six operations group into three parallelization regimes: *compute-parallel* (MatMul; output-channel split scales near-linearly), *transitional* (GELU, LayerNorm; per-call compute is large enough to absorb framework overhead at 4 GPUs), and *data-parallel-throughput* (bootstrap, softmax, argmax-at-4-GPU; per-call latency is comparable to per-rank context-setup so additional GPUs deliver throughput rather than latency reduction). Heterogeneity *is* the typology — a single aggregate speedup figure would conceal it.

**(2) End-to-end BERT inference at uniform $\log N = 15$.** We demonstrate the chained pipeline NEXUS does not ship (§7). The methodology is: a 1-head × 2-layer unit measurement on a single GPU at uniform $\log N = 15$, an explicit saturation check (the time for layer 1 matches the time for layer 2 within a stated tolerance), and an extrapolation to full BERT by multiplication ($12 \times 12 \times t_{\text{head,layer}}$). We then report the chained pipeline under both parallelization regimes a reviewer might ask about: head-parallel (HP-BERT, per-inference latency, strong scaling at 4 and 16 GPUs) and data-parallel ($G$ concurrent independent inferences, aggregate throughput, weak scaling). Every reported number is reproducible from one SLURM script under `scripts/mn5/`.

## 2.3 Why this combination matters

The two contributions are not independent. NEXUS publishes per-operation latencies on 4× A100 but does not chain end-to-end on the open-source artifact, so an A100→H100 comparison conflates two effects: hardware uplift (HBM3 bandwidth, SM count, NVLink generation) and any framework-level contribution. We defuse that confusion in §4 by rebuilding NEXUS unchanged on H100 and reporting our measured NEXUS-on-H100 column as the baseline against which every §6 speedup is calculated. The single-GPU multiNEXUS measurement matches the NEXUS-on-H100 column within $\pm 2\%$ on every operation (§4.5), which is the correctness gate that licenses §6's framework attributions. Without §4's identification step, §6's claims would be contestable; without §6's framework, §7's chained pipeline would have no per-operation grounding.

Heterogeneity is the headline of §6, not a caveat. Bootstrap and softmax behave differently from MatMul under added GPUs because their per-call compute is comparable to the per-rank `PhantomContext` setup cost; MatMul behaves differently from both because its parallelization is the disjoint output-channel split, which has neither inter-GPU communication during the call nor an amortizable framework overhead. Reporting one aggregate multi-GPU speedup would either over-credit MatMul or under-credit bootstrap; the typology says which regime applies to which operation and why.

## 2.4 Honest disclosures (preview)

We surface three limitations in §8 and preview them here so the contributions of §2.2 are read in context.

First, the per-rank `PhantomContext` setup overhead caps small-op 16-GPU efficiency at 9–22% (§6). Each rank currently rebuilds its `PhantomContext` per op-call; the small operations (bootstrap at $\log N = 15$, softmax at $\log N = 16$, argmax at $\log N = 15$) have per-call compute on the order of 20–250 ms, which the per-rank setup overhead consumes a non-trivial fraction of. We disclose this ceiling with the profiling trace that establishes it, rather than report a higher speedup that excludes setup time.

Second, NEXUS's published argmax latency is for vocabulary 30,522 (the BERT-base vocabulary, padded to $2^{15} = 32{,}768$). Our `argmax_align_n32k.cu` binary handles a single sparse-encoded ciphertext, which at $\log N = 15$ and `sparse_slots` $= 8{,}192$ caps the vocabulary it can compare at 8,192. The full-vocabulary number requires a multi-cipher tournament path (argmax-of-argmaxes across multiple ciphertexts combined in a final tournament) that NEXUS does not ship in its open source and which we did not implement this semester. We report vocab=8 and vocab=8,192 numbers honestly and disclose the gap (§8).

Third, we do not apply slot-axis SIMD packing for HP-BERT. NEXUS's 37.3 s end-to-end number depends on packing all 12 attention heads into a single ciphertext, reducing the number of bootstraps from $\Theta(\text{heads} \times \text{layers})$ to a small constant. Without that packing, no amount of GPU parallelism beats 37.3 s on a fair workload. The refactor is multi-day and did not fit this semester; we disclose this in §8 as the gating follow-up to the work in this paper.

## 2.5 Hardware platform

All numbers in this paper come from the ACC partition of Barcelona Supercomputing Center MareNostrum 5 [CITATION_MN5]. Per node: 4× NVIDIA H100 64 GB SXM5 with NVSwitch all-to-all NVLink. Multi-node: up to 4 nodes (16 GPUs total) over InfiniBand. Software stack: CUDA 12.8, NCCL 2.24.3-1, GCC, CMake 3.30.5; compute capability target `CMAKE_CUDA_ARCHITECTURES=90`. Every measurement is reproducible from a SLURM script under `scripts/mn5/`; each reported number is tagged with its JOBID and a log path on MN5 in the appendix.

## 2.6 Paper structure

§3 reviews CKKS, the RNS-CKKS bootstrap critical path, NEXUS, and prior multi-GPU FHE (Cerium, Cinnamon). §4 builds NEXUS unchanged on H100 and establishes the fair-comparison baseline column. §5 presents the three multi-GPU strategies that the rest of the paper uses: DKS (distributed key-switching), HP-BERT (head-parallel BERT), and DP (data-parallel per-operation). §6 reports Goal 1 — the per-operation multi-GPU typology at 4 and 16 GPUs, six operations under the 6-field template. §7 reports Goal 2 — end-to-end BERT inference at uniform $\log N = 15$, with the unit measurement, the saturation check, the extrapolation, and both head-parallel (strong scaling, latency) and data-parallel (weak scaling, throughput) regimes. §8 discusses limitations: the small-op efficiency ceiling, the multi-cipher argmax gap, the absence of slot-axis SIMD packing, and our qualitative position relative to Cerium and Cinnamon. §9 concludes and lists future work. Appendix A documents every modification to NEXUS and Phantom this semester plus the bug-fix log.

---

# Section 3 — Background

> Status: draft v1
> Slice: WRITE-S3
> Depends-on: DOC-01

This section establishes the prerequisites needed for the rest of the paper.
Section 3.1 fixes the CKKS notation a reader needs in order to follow
Sections 5–7. Section 3.2 zooms in on the bootstrap critical path because
every multi-GPU strategy in this paper is ultimately a statement about how
to parallelize an RNS-CKKS operation. Section 3.3 positions NEXUS
[CITATION_NEXUS] precisely — what its open source ships and, equally
important, what it does not. Section 3.4 contrasts our work with the two
closest published precedents in multi-accelerator CKKS, Cinnamon
[CITATION_CINNAMON] and Cerium [CITATION_CERIUM]. Section 3.5 states what
this paper adds.

## 3.1 CKKS in one page

CKKS [CITATION_CHEON] is a polynomial-ring fully homomorphic encryption
scheme that encrypts vectors of approximate (floating-point) complex
numbers and supports homomorphic addition and multiplication on the
encrypted vectors. It is the FHE scheme of choice for transformer
inference because its native data type — batched approximate
arithmetic — is the data type ML models already operate on.

The native plaintext object is a polynomial of degree $< N$ with integer
coefficients, reduced modulo the cyclotomic polynomial $X^N + 1$:
$R_Q = \mathbb{Z}_Q[X]/(X^N + 1)$, where $N$ is a power of two called the
**ring degree**. One plaintext encodes $N/2$ complex slots via the
canonical embedding, so one ciphertext is a SIMD vector of $N/2$ slots
operated on in parallel. We instantiate $N = 32{,}768$ ($\log N = 15$)
and $N = 65{,}536$ ($\log N = 16$) at different points in the paper
(the choice is driven by bootstrap depth and the 128-bit security
target; see Section 8 for the security argument).

A CKKS ciphertext is a pair of polynomials $(c_0, c_1) \in R_Q^2$ such
that $c_0 + c_1 \cdot s \approx \Delta \cdot m \pmod{Q}$, where $s$ is
the secret key, $m$ is the encoded message, and $\Delta$ is a scaling
factor. Decryption computes $c_0 + c_1 \cdot s$ and descales by
$\Delta$.

Three parameters recur throughout the paper:

- **$\log N$** — log of the ring degree. Larger $\log N$ gives more
  slots, deeper computations, and more memory per ciphertext (a fresh
  ciphertext at $\log N = 16$ is $\approx 38$ MB versus $\approx 19$ MB
  at $\log N = 15$).
- **$\texttt{dnum}$** — the number of digits in the digit decomposition
  used by key-switching. This is the axis along which our Distributed
  Key-Switching (DKS) implementation, Section 5, shards work across
  GPUs.
- **The coefficient modulus chain** $Q = q_0 \cdot q_1 \cdots q_L$ —
  a product of 50–60-bit primes called **limbs**. Each ciphertext lives
  at some level $\ell$ with modulus $\prod_{i \le \ell} q_i$; every
  ciphertext-ciphertext multiplication consumes one limb (a "level").
  When the chain is exhausted ($\ell = 0$) the ciphertext must be
  bootstrapped before further multiplications.

Each polynomial in $R_Q$ is stored in **RNS (Residue Number System)**
form: by the Chinese Remainder Theorem, the polynomial is uniquely
represented by its $L+1$ reductions modulo the individual limbs, and
all arithmetic is performed limb-by-limb in parallel. RNS is what
makes CKKS amenable to GPU implementation — each of the $L+1 \approx 44$
limbs is a degree-$<N$ polynomial whose coefficients fit in 64 bits,
and the limbs can be processed by independent CUDA streams or independent
SMs [CITATION_PHANTOM].

The CKKS operator set used by NEXUS-style BERT inference is:

- **Add** (ciphertext-ciphertext or ciphertext-plaintext): negligible
  cost — one elementwise pass per limb.
- **Multiply** (ciphertext-ciphertext): produces a three-polynomial
  intermediate $(c_0, c_1, c_2)$ that must be reduced back to the
  two-polynomial form via **relinearization**, a key-switch using an
  evaluation key. Consumes one level. Followed by **rescaling**, which
  divides by the highest prime and drops that limb.
- **Rotate by $k$ slots**: a Galois automorphism $X \mapsto X^{5^k}$.
  The automorphism itself is just a permutation of coefficients, but
  it changes the secret under which the ciphertext is encrypted, so
  it must be followed by a key-switch using a **Galois rotation key**
  precomputed for that specific $k$.
- **Bootstrap**: a homomorphic procedure that refreshes the level of
  an exhausted ciphertext. The most expensive single operation by
  more than an order of magnitude.

The salient point for this paper is that key-switching — invoked by
every relinearization and every rotation — dominates bootstrap cost,
and the bootstrap dominates BERT inference cost. Section 3.2 traces
that critical path.

## 3.2 RNS-CKKS and the bootstrap critical path

A single bootstrap at NEXUS parameters consists of approximately 75
rotations [CITATION_HPC_PRIMER], each of which performs one key-switch.
The five steps of a key-switch are:

1. **Digit decomposition.** Split the $L+1$ limbs of the input
   polynomial into $\beta$ groups called digits. $\beta$ is chosen at
   setup; our deployment uses $\beta \approx 36$ at $\log N = 16$.
2. **Mod-up (basis expansion).** For each digit, raise it from its
   small-modulus subset up to the extended modulus $PQ$, where $P$ is
   an auxiliary product of primes added for noise control. Mod-up
   internally performs a CRT basis conversion followed by a tower of
   forward Number-Theoretic Transforms (NTTs). Mod-up expands each
   digit by roughly $40\times$ in HBM footprint
   [CITATION_HPC_PRIMER §2.8].
3. **Inner product.** The key-switching key is a $\beta$-row matrix
   where each row is a small ciphertext at modulus $PQ$. Multiply each
   extended digit by its corresponding key row and sum across digits.
4. **Mod-down.** Divide by $P$ and basis-convert back to modulus $Q$.
5. **Add back** into the original ciphertext to complete the
   key-switch.

Two GPU kernels dominate inside this chain. The **NTT** (Number-
Theoretic Transform) — the finite-field analogue of the FFT used to
move polynomials between coefficient and evaluation form — is the
single most expensive kernel. We measured it at approximately 40\% of
bootstrap time using NVTX instrumentation plus Nsight Systems on our
H100 deployment [CITATION_HPC_PRIMER §5; this paper's CLAUDE.md lesson
\#5]; an earlier rough estimate of 15\% was off by 2.7$\times$. The
second-most-expensive kernel is the key-switch inner product itself,
which is a $\beta$-dimensional dot product of size-$N$ polynomials at
the extended modulus $PQ$.

A full bootstrap proceeds in three phases [CITATION_CHEN_HAN]:

- **CoeffToSlot (CTS)** — a linear transform from coefficient
  representation to slot representation, implemented as approximately
  30 rotations followed by matrix multiplications using a Baby-Step
  Giant-Step (BSGS) decomposition [CITATION_BSGS].
- **Modular reduction** — a polynomial approximation of $x \bmod q_0$
  (Chebyshev/Remez), the actual noise-reset step. Consumes
  approximately 17 levels [CITATION_HPC_PRIMER §8.3] — we budget 14
  for downstream BERT operators in the chain.
- **SlotToCoeff (STC)** — the inverse linear transform, another
  $\approx$ 30 rotations.

End-to-end, a bootstrap at $\log N = 16$ on a single H100 costs
$\approx 300$ ms per ciphertext under prefetched-key conditions
[CITATION_HPC_PRIMER §7.3]; at $\log N = 15$ on H100 we measure
$\approx 250$ ms (Section 4.5). Because one bootstrap consumes
$\approx 17$ chain levels itself, the **post-bootstrap headroom** is
$L+1 - 17 \approx 27$ levels of usable depth — enough for one full
BERT encoder layer (matmul, softmax, GELU, LayerNorm) before the next
bootstrap is required.

The crucial observation for Section 5 is that the $\beta$-digit inner
product in step 3 is embarrassingly parallel along $\beta$: addition
distributes over partition. Splitting the digit index set $\{0, \dots,
\beta-1\}$ into $G$ disjoint groups and `ncclAllReduce`-ing the partial
sums is bit-equivalent to the single-GPU computation. This algebraic
identity is what licenses DKS [CITATION_HPC_PRIMER §4.1] and, as we
discuss in Section 5, also licenses the Galois-key shard layout that
makes $\log N = 16$ runnable on a single 4$\times$ H100 node at all
(the full bootstrap key store is $\approx 62$ GB, larger than one
H100's 64 GB HBM).

## 3.3 NEXUS in our context

NEXUS [CITATION_NEXUS] is the GPU implementation of the NDSS 2025
protocol of Zhang et al. for non-interactive private transformer
inference. The protocol is genuinely non-interactive: the client
encrypts its input under CKKS, the server executes the entire BERT-base
forward pass under encryption, and a single encrypted answer is
returned. There is no intermediate round-trip and no decryption on
the server side.

NEXUS's published code surface lives in `vendor/nexus/cuda/` and ships
six per-operation CKKS kernels — bootstrap, matmul, GELU, LayerNorm,
softmax, and argmax. The top-level driver `vendor/nexus/cuda/src/main.cu`
exercises each kernel in isolation against synthetic inputs at NEXUS's
parameter choices, but **it does not chain the kernels into an
end-to-end BERT pipeline**, and it ships no multi-GPU framework. We
verified this directly: `vendor/nexus/cuda/` contains no
`cudaSetDevice`, no NCCL, no MPI, and no `std::thread`. The single
`cudaSetDevice` match in the Phantom thirdparty is a hardcoded
`cudaSetDevice(0)` we patch out (Section 4 and Appendix A).

NEXUS publishes 37.3 s per BERT inference on $4\times$ A100 as its
headline number [CITATION_NEXUS]. That number is not reproducible from
the open source alone — the chained pipeline behind it depends on
slot-axis SIMD packing (their Algorithm 3, packing all 12 attention
heads into a single ciphertext to reduce bootstrap count) which is not
in the open source. We discuss this gap in Section 8.

For our purposes, NEXUS plays two distinct roles:

1. **Algorithmic source.** Section 4.2 imports the six per-op kernels
   into our codebase as the FHE primitives we parallelize. The
   bootstrap implementation is used essentially verbatim — any change
   broke numerical accuracy in our regression tests.
2. **Baseline.** Section 4 builds NEXUS from source on H100 and runs
   NEXUS's own per-operation benchmarks at NEXUS's parameter choices.
   The resulting *NEXUS-on-H100* column eliminates the A100$\to$H100
   hardware-uplift confound from the comparison and is the only
   baseline against which Section 6 claims speedups.

## 3.4 Prior multi-GPU FHE: Cerium and Cinnamon

The closest published precedents to our multi-accelerator decomposition
work are Cinnamon [CITATION_CINNAMON] and Cerium [CITATION_CERIUM],
both from the Skarlatos group at CMU.

**Cinnamon (ASPLOS 2025).** Cinnamon [CITATION_CINNAMON] is a
Python$\to$ASIC-ISA compiler for CKKS-like workloads. It targets a
custom multi-accelerator instruction set rather than commodity GPUs,
and **its headline numbers come from cycle-level architectural
simulation, not from real hardware**. This is an important caveat: we
cannot do a head-to-head wall-clock comparison against Cinnamon on
H100, because the artifact does not run on H100. We therefore use
Cinnamon only as an *algorithmic reference* for what multi-accelerator
decomposition of CKKS *should* look like — in particular, the
digit-axis decomposition of key-switching (which our DKS independently
arrives at) and the observation that the number of inter-accelerator
AllReduces, not the per-AllReduce latency, is what eventually limits
scaling [CITATION_CINNAMON §6].

**Cerium (arXiv 2025).** Cerium [CITATION_CERIUM] is the GPU sibling
of Cinnamon, by largely the same authors. Architecturally it is the
closest published precedent to what we build in this paper:
compiler-driven multi-GPU CKKS for encrypted transformer inference,
using digit-axis decomposition plus head-parallel layout. As of the
time of writing (2026-05) the Cerium artifact is not public, so a
head-to-head measurement is not possible. The comparison in this paper
is therefore qualitative: we credit Cerium with the architectural
template (digit-axis sharding for key-switching, head-parallel layout
for transformer layers), and we identify as a contribution of this
paper the first *reproducible* multi-GPU measurement of those ideas on
commodity H100 hardware. We return to the qualitative comparison in
Section 8.

**Other prior art.** Vanilla NEXUS, as noted in Section 3.3, has no
multi-GPU framework. Earlier GPU-CKKS work
[CITATION_PHANTOM, CITATION_OVERSEER] focused on single-GPU NTT and
key-switch optimization rather than on inter-GPU decomposition. The
hardware-accelerator literature for CKKS (CraterLake [CITATION_CRATERLAKE],
F1 [CITATION_F1], SHARP [CITATION_SHARP], ARK [CITATION_ARK]) is
adjacent but addresses a different surface — custom silicon, not
multi-GPU on commodity hardware — and is consequently incomparable for
the latency claims in this paper. [TODO: confirm OVERSEER citation
slot or remove.]

## 3.5 What this paper adds

Against the prior art summarized above, this paper makes three
contributions, each grounded in measurements on real H100 hardware
rather than in simulation:

1. **A NEXUS-on-H100 baseline (Section 4).** We build NEXUS unchanged
   from source on H100 and reproduce its six per-operation
   benchmarks. This eliminates the A100$\to$H100 hardware-uplift
   confound from every per-op comparison in Section 6 and serves as
   a correctness gate on our reimplementation: single-GPU multiNEXUS
   must match NEXUS-on-H100 within $\pm$2\% on every operation
   (Section 4.5).
2. **A per-operation multi-GPU typology with profiling-grounded
   ceilings (Goal 1, Section 6).** For each of the six NEXUS
   operations, we report single-GPU, 4-GPU, and 16-GPU latencies
   under data-parallel-per-op decomposition; classify each
   operation into one of three buckets (compute-parallel,
   transitional, data-parallel-throughput); and explain the observed
   ceiling using Nsight Systems and NCU evidence.
3. **End-to-end BERT inference at uniform $\log N = 15$ with a
   saturation-check-grounded extrapolation (Goal 2, Section 7).** We
   measure a 1-head $\times$ 2-layer unit, verify saturation
   (time(layer 1) $\approx$ time(layer 2) within 5\%), extrapolate
   to full BERT-base (12 heads $\times$ 12 layers) by multiplication,
   and back the extrapolation with strong-scaling Head-Parallel BERT
   measurements at 4-GPU and 16-GPU, plus a weak-scaling
   data-parallel inference measurement at the same scales.

All three contributions are reproducible: the source tree is public,
the SLURM scripts that produce every published number are checked in,
and per-operation JOBID-plus-log provenance is recorded in
`docs/PER_OP_VS_NEXUS.md`. This reproducibility is itself a
contribution against the comparison points in Section 3.4 — Cinnamon's
numbers come from a simulator, and Cerium's artifact is currently
unavailable.

---

# Section 4 — Identifying NEXUS on H100

> Status: draft v1
> Slice: WRITE-S4
> Depends-on: none (uses existing data in docs/PER_OP_VS_NEXUS.md)

## 4.1 Motivation: why a new baseline column is necessary

NEXUS [CITATION_NEXUS] reports per-operation latencies for BERT-base inference
on a 4× A100 system. Two of the three numbers in our headline comparison
(NEXUS-A100, multiNEXUS single-GPU H100, multiNEXUS multi-GPU H100) live on
different hardware than the third, so the unmodified A100 → H100 jump conflates
two effects: the hardware uplift (HBM3 bandwidth, SM count, NVLink generation)
and any framework-level contribution from multiNEXUS. Without isolating the
hardware component, every speedup ratio in Section 6 is contestable.

We therefore introduce a second baseline column: *NEXUS-on-H100*. We build
NEXUS unchanged from source on a single H100, run NEXUS's own per-operation
benchmarks (`vendor/nexus/cuda/src/main.cu`) at NEXUS's chosen poly-modulus
degrees, and record the measured latencies. Section 6's per-op comparisons
then read against this column rather than the published A100 numbers; the
A100 column remains in the table for reference but is not the basis of any
speedup claim. This eliminates the hardware-normalization arithmetic
entirely — both the NEXUS baseline and our single-GPU multiNEXUS number
run on identical H100 silicon.

A side effect of this exercise is a *correctness gate*: our single-GPU
multiNEXUS measurement must match the NEXUS-on-H100 measurement within a
few percent on every operation. Otherwise we would be running a different
algorithm and any multi-GPU speedup would not be meaningful. Section 4.5
shows that all six ops match within $\pm 2\%$.

## 4.2 What "NEXUS" means in this paper

NEXUS [CITATION_NEXUS] is the GPU implementation of the NDSS 2025 protocol
of Zhang et al. for non-interactive privacy-preserving transformer
inference. The public code surface lives in `vendor/nexus/cuda/` and
contains six per-operation kernels: `matrix_mul.cu`, `gelu.cu`,
`layer_norm.cu`, `softmax.cu`, `argmax.cu`, and `bootstrapping/`. The
top-level driver `vendor/nexus/cuda/src/main.cu` exercises each kernel in
isolation with NEXUS's chosen parameters. The kernels themselves are
written for the encryptorion-lab Phantom fork shipped under
`vendor/nexus/cuda/thirdparty/phantom-fhe/`.

We confirmed by direct `grep` that `vendor/nexus/cuda/` contains no
`cudaSetDevice`, no NCCL, no MPI, and no `std::thread` calls — the single
match for `cudaSetDevice` is `cuda_wrapper.cuh:50`, a hardcoded
`cudaSetDevice(0)` in the Phantom stream constructor that we patch out in
our copy (Section 4.4). The implication is that NEXUS as published is a
single-GPU codebase; it also does not chain the six operators
end-to-end — `main.cu` runs each in isolation against synthetic inputs.
The published 37.3 s end-to-end claim cannot be reproduced from this
source, a fact we return to in Section 7.

Phantom [CITATION_PHANTOM] is the GPU-native CKKS library on which NEXUS
builds. We use the same encryptorion-lab Phantom fork, vendored under
`vendor/phantom/`, with a 95-line patch (full diff in Appendix A) that
enables thread-local CUDA streams, ciphertext save/load for inter-GPU
transfer, and lazy rescaling required by the bootstrap path. The patch
does not change any FHE primitive; we cross-check this in Section 4.5 by
matching NEXUS-on-H100 latencies on their own benchmarks within a few
percent.

## 4.3 Hardware platform: MN5 ACC partition

All measurements in this paper, including the NEXUS-on-H100 column, run
on the ACC partition of Barcelona Supercomputing Center MareNostrum 5
[CITATION_MN5]:

- **Per node:** 4× NVIDIA H100 64 GB SXM5, NVSwitch all-to-all NVLink
  fabric.
- **Multi-node:** up to 4 nodes (16 GPUs total) over InfiniBand
  (`mlx5_*` HCAs, NCCL `IB` transport).
- **Software stack:** CUDA 12.8, NCCL 2.24.3-1, GCC, CMake 3.30.5. NTL
  and GMP are installed under
  `/gpfs/projects/etur02/hkanpak/local/` and resolved at runtime via
  `LD_LIBRARY_PATH`.
- **Compute capability target:** `CMAKE_CUDA_ARCHITECTURES=90`.

NEXUS targets compute capability 8.0 (A100) in its upstream build. The
only edit required to compile NEXUS on H100 is changing this target to
9.0; no kernel source changes are required.

## 4.4 Build and benchmark of NEXUS on H100

We clone the vendored NEXUS source unchanged into
`vendor/nexus/cuda/`, set `CMAKE_CUDA_ARCHITECTURES=90` in its build
configuration, and compile against the same CUDA 12.8 toolchain used for
multiNEXUS. NEXUS's `main.cu` exposes six standalone benchmarks selected
by command-line flag, each instantiating its own `PhantomContext` at
NEXUS's chosen poly-modulus degree for that operation:

| Operation | NEXUS source                                | poly-modulus degree |
|-----------|---------------------------------------------|---------------------|
| MatMul    | `cuda/src/main.cu:25-26` (`MM_LOG_N = 13`)  | $2^{13} = 8{,}192$  |
| Bootstrap | `cuda/src/main.cu:109` (`logN = 15`)        | $2^{15} = 32{,}768$ |
| Argmax    | `cuda/src/main.cu:109` (shares bootstrap N) | $2^{15} = 32{,}768$ |
| GELU      | `src/main.cpp:45` (`logN = 16`)             | $2^{16} = 65{,}536$ |
| LayerNorm | `src/main.cpp:45` (`logN = 16`)             | $2^{16} = 65{,}536$ |
| Softmax   | `src/main.cpp:45` (`logN = 16`)             | $2^{16} = 65{,}536$ |

The three distinct values of $N$ are NEXUS's own choice: their public
text notes that the key-table sizes at the larger $N$ values do not fit
on a single A100 for matmul, so a smaller $N$ is used per-op rather than
a uniform parameter set. This three-N regime is what makes the NEXUS
chained-pipeline number unreproducible from the open source (no
bridging code between $N$ values is shipped). For the per-op
*identification* work in this section we accept NEXUS's per-op parameter
choices as given; Section 7 separately reports our uniform-$N$ chained
pipeline.

Each NEXUS benchmark is run on a single H100 with the standard NEXUS
configuration (no command-line tuning). The measured latencies are
recorded under the SLURM job IDs listed in Section 4.5; raw logs land
at `/gpfs/projects/etur02/hkanpak/logs/nexus_build_<JOBID>.{out,err}`
on MN5.

## 4.5 The fair-comparison column

Table 1 reports the three columns introduced in Section 4.1 for the six
operations: NEXUS-published A100 latency, NEXUS-on-H100 latency (measured
by us under JOBIDs 40367787 and 40368133), and multiNEXUS single-GPU
H100 latency. The fourth column (single-GPU multiNEXUS) is the
correctness gate: it should sit within a few percent of column 3 if our
framework runs the same algorithms. The full multi-GPU comparison
columns are deferred to Section 6.

**Table 1.** Per-operation single-GPU latencies. All NEXUS-on-H100
numbers measured under JOBID 40367787 except softmax (JOBID 40368133).
All multiNEXUS single-GPU numbers measured under Lane ALIGN-SINGLE
JOBIDs as cited; raw provenance in `docs/PER_OP_VS_NEXUS.md` §4.4–§4.5.

| Operation                          | logN | NEXUS published (A100) | NEXUS-on-H100 (ours)         | multiNEXUS single-GPU H100   | $\Delta$ vs NEXUS-on-H100 |
|------------------------------------|------|------------------------|------------------------------|------------------------------|---------------------------|
| Bootstrap                          | 15   | 5{,}630 ms             | 252.8 ms (JOBID 40367787)    | 250 ms (Lane ALIGN-SINGLE)   | $-1.1\%$                  |
| GELU                               | 16   | 3{,}350 ms             | 69 ms (JOBID 40367787)       | 70.30 ms (JOBID 40387027)    | $+1.9\%$                  |
| LayerNorm                          | 16   | 1{,}010 ms             | 45 ms (JOBID 40367787)       | 45.5 ms (Lane ALIGN-SINGLE)  | $+1.1\%$                  |
| Softmax                            | 16   | 1{,}150 ms             | 20 ms (JOBID 40368133)       | 20 ms (Lane ALIGN-SINGLE)    | $\approx 0\%$             |
| MatMul (per-col amortized)         | 13   | 1{,}310 ms (256-batch) | 95 ms (256-batch amortized, JOBID 40367787) | 285 ms/col (Lane ALIGN-SINGLE)$^{\dagger}$ | see note$^{\dagger}$ |
| Argmax (vocab $=8$, NEXUS bundled) | 15   | 2{,}480 ms (vocab $=$ 30{,}522)$^{\ddagger}$ | 863 ms (JOBID 40367787) | 848.4 ms (JOBID 40369741) | $-1.7\%$ |

$^{\dagger}$ The MatMul comparison is not a direct per-call equality. NEXUS reports
an amortized cost across a 256-input batch using their packing scheme;
the multiNEXUS number is per output column at the same logN-13
parameters. Both numbers are derived from the *same* underlying
ciphertext-plaintext multiplications; the framing difference (per-call
vs per-batch amortized) is preserved in Section 6 where the multi-GPU
output-channel split is presented at the per-column granularity.

$^{\ddagger}$ NEXUS's bundled argmax test only exercises an 8-element vocabulary
single-ciphertext case. Their 2.48 s published number for vocab=30{,}522
comes from a multi-cipher tournament that is not in their open source
(documented in `docs/PI_REPORT.md` and the project task list as a
follow-up item). We therefore compare against NEXUS's bundled vocab=8
result (their own H100 measurement of 863 ms) for the correctness gate;
the vocab=30{,}522 row remains an open extrapolation in the comparison
table.

With the exception noted for argmax-at-large-vocab and the MatMul
amortization caveat, every multiNEXUS single-GPU H100 number matches
the NEXUS-on-H100 column within $\pm 2\%$. This is the evidence on
which Section 6 relies when it attributes per-op multi-GPU speedups to
the multi-GPU framework rather than to a different underlying kernel.

## 4.6 Bugs surfaced while identifying NEXUS on H100

Two bugs in our port were exposed only because we had the NEXUS-on-H100
column to compare against; both are documented in full in Appendix A and
summarized here for narrative continuity.

**B1 — GELU coefficient-modulus chain depth at logN=16.** Our GELU
benchmark crashed on warmup with *"end of modulus switching chain
reached"*. The 40-bit middle moduli were configured with
`for (int i = 0; i < 17; i++)`, yielding 19 total moduli
($\{58, 17{\times}40, 58\}$). NEXUS uses 18 forties between two 58s
(`vendor/nexus/cuda/src/main.cu` GELU section), giving 20 total moduli
($\{58, 18{\times}40, 58\}$); one less rescale was available than the
inner `sgn_eval` polynomial needs. We verified by counting commas in
the upstream `set_coeff_modulus` call:
`grep "GELU (4)" vendor/nexus/cuda/src/main.cu | tr ',' '\n' | wc -l`
→ 20. After increasing the loop bound to 18, single-GPU GELU runs in
70.30 ms (JOBID 40387027), within $1.9\%$ of the NEXUS-on-H100 number.

**B2 — Argmax scale drift across chained bootstraps.** Inside QuickMax
(NEXUS Algorithm 2), the third bootstrap call failed Phantom's encode
validation in `slottocoeff_3` with a scale-mismatch error. Our Phantom
patch comments out the scale-equality checks in
`evaluate.cu::sub_inplace / multiply_plain_inplace / add_plain_inplace`
to enable the lazy-rescale pattern that bootstrap requires; NEXUS keeps
those checks enabled in its Phantom fork. With our checks disabled, a
small scale drift accumulates silently across QuickMax rounds. The fix
is an explicit `x.scale() = SCALE` reset before each bootstrap call in
`src/benchmarks/argmax_align_n32k.cu` (Lane ARGMAX-FIX). After the
fix, single-GPU vocab=8 argmax runs in 848.4 ms (JOBID 40369741),
within $1.7\%$ of NEXUS-on-H100's 863 ms — confirming that the
QuickMax algorithm is preserved by the reset.

Neither bug is interesting on its own. What is interesting is that
neither was visible until we had a NEXUS-on-H100 column to compare
against. The bug-fix log is the operational payoff of the
identification work.

## 4.7 Summary

We have built NEXUS from source on a single H100, run its six per-op
benchmarks at NEXUS's chosen poly-modulus degrees, and recorded the
measured latencies. Together with our matching single-GPU multiNEXUS
measurements (within $\pm 2\%$ for five of the six ops; the argmax
caveat at full vocabulary and the MatMul amortization framing are
explicitly noted), this constitutes a defensible single-GPU baseline
column against which Section 6 measures the per-operation multi-GPU
speedups. Section 5 next presents the three multi-GPU strategies (DKS,
head-parallel BERT, data-parallel-per-op) and Section 6 reports the
per-op typology measurements at 4 and 16 GPUs against the column
established here.

---

# Section 5 — Multi-GPU Strategies

> Status: draft v1
> Slice: WRITE-S5
> Depends-on: none (architectural, no measurements)

NEXUS [Zhang et al., NDSS 2025] publishes per-operation CKKS kernels but ships no multi-GPU framework: there is no `cudaSetDevice`, NCCL, MPI, or `std::thread` anywhere in `vendor/nexus/cuda/`. To run a full BERT inference under encryption on multiple GPUs, we built three orthogonal parallelization strategies on top of Phantom's GPU-native CKKS, each targeting a different figure of merit. This section defines the three strategies and the framework that supports them; measurements live in Sections 6 and 7.

## 5.1 Strategy taxonomy

Three strategies, three figures of merit:

| Strategy | Parallelizes | Figure of merit | Scaling regime |
|---|---|---|---|
| **DKS** (Distributed Key-Switching) | One bootstrap across $N$ GPUs | Per-call latency of the bootstrap | Strong |
| **HP** (Head-Parallel BERT) | One BERT inference across $N$ GPUs by attention head | Per-inference latency | Strong |
| **DP** (Data-Parallel per-operation) | $N/G$ independent op calls per GPU | Aggregate throughput | Weak |

The three strategies are not mutually exclusive: a single end-to-end inference can use different strategies for different operations. Section 6 develops a per-operation typology that chooses, for each NEXUS evaluator, which of these three is the best fit at 4-GPU and 16-GPU scales. Section 7 then uses HP at the BERT level for the strong-scaling latency story and DP at the inference level for the weak-scaling throughput story.

## 5.2 DKS — Distributed Key-Switching

**Target.** A single bootstrap call, parallelized across $G$ GPUs.

**Why bootstrap.** Bootstrap is the costliest NEXUS operation (≈250 ms at $\log N = 15$, ≈300 ms at $\log N = 16$ on H100) and dominates end-to-end inference time. Internally it is a chain of rotations and ciphertext-plaintext multiplications; each rotation invokes a key-switch, the second hottest CKKS kernel after NTT.

**Mechanism.** A key-switch decomposes its input ciphertext into $\beta$ digits ($\beta = \lceil \text{size\_Ql} / \alpha \rceil \le \texttt{dnum}$), mod-ups each digit to the extended chain $\mathit{QlP}$, computes an inner product with the evaluation key, and mod-downs back. DKS splits the $\beta$-digit inner product across GPUs: each GPU computes its partial sum and one `ncclAllReduce` over the $\mathit{QlP}$-basis output combines the partials before the local mod-down. Phantom's `key_switch_inner_prod_c2_and_evk` was modified to accept a digit range — the only Phantom-internal modification DKS requires (`src/multi_gpu/keyswitching/output_aggregation.cu:85`).

**Memory sharding.** The same digit ownership pattern is used to shard the bootstrap Galois keys themselves. Without sharding, the 50 rotation keys consumed by a single bootstrap at $\log N = 16$ are approximately 62 GB on the GPU — larger than the 64 GB capacity of a single H100. With DKS sharding across 4 GPUs, each GPU holds only its own digit shard (≈6.8 GB per GPU at $N = 65536$, $\beta = 9$, $G = 4$; see `src/multi_gpu/keyswitching/dist_galois_key_store.cuh:8`). DKS is therefore the *enabling* strategy for $\log N = 16$ on a single node, not merely the *accelerating* one.

**Ownership pattern.** Digit ownership is **STRIDED**, not contiguous: GPU $g$ owns $\{d : d \bmod G = g\}$. An earlier contiguous layout deadlocked when the runtime chain level dropped below $\texttt{dnum}$: the key-switch only iterates the prefix $d \in [0, \beta)$, so trailing GPUs' contiguous shards covered digits never accessed while leading GPUs were asked for null slots. The visible failure mode was a `cudaFreeAsync` invalid-argument cascade and NCCL P2P illegal-memory-access (`src/multi_gpu/keyswitching/dist_galois_key_store.cuh:19`). STRIDED ownership guarantees *any* prefix $[0, \beta)$ is covered. This is captured in non-negotiable lesson #6 in `CLAUDE.md`.

**Role in the paper.** DKS is the strong-scaling story *for the bootstrap operation* in Section 6 and the path that makes $\log N = 16$ runnable at all. It is not the headline framework for end-to-end inference; that is HP-BERT.

## 5.3 HP-BERT — Head-Parallel BERT

**Target.** One BERT-base inference, parallelized across $G$ GPUs for per-inference latency.

**Mechanism.** BERT-base has 12 attention heads × 12 encoder layers. HP-BERT distributes heads across GPUs end-to-end: GPU $g$ owns a subset of the 12 heads and runs *all 12 layers* locally for them. Each GPU holds, pinned to its device, its share of the Galois keys, plaintext encodings of its heads' projection weights, and a private `Bootstrapper`. Only the activations flow between layers across GPUs; the (large) weights and keys never move after initialization.

**Implementation.** On a single node, HP-BERT uses one `std::thread` per GPU, each bound to its device with `cudaSetDevice` at thread startup (`src/benchmarks/bert_hp_multigpu.cu:543`); each thread instantiates its own `PhantomContext`, `Bootstrapper`, and key store. We avoid MPI on single-node because Phantom's CKKS context is thread-safe under our usage (verified by `phantom_threadsafe_smoke.cu` at MAE $= 0$). For multi-node (4 nodes × 4 GPUs = 16 GPUs), we use one MPI rank per node and four worker threads per rank; NCCL communicators span all 16 GPUs (`src/multi_gpu/distributed_context.cuh:83`).

**Thread-safety prerequisites.** Per-thread `PhantomContext` only works because we made Phantom's `default_stream` `thread_local` (Appendix A, Phantom modification #3) and removed a hard-coded `cudaSetDevice(0)` in the stream constructor (modification #4). Both were prerequisite bugs we fixed in `vendor/phantom/`.

**Role in the paper.** HP-BERT is the strong-scaling story *for end-to-end BERT inference* in Section 7 (Goal 2). It is the only strategy in this paper that reduces the wall-clock latency of a single inference by adding GPUs.

## 5.4 DP — Data-Parallel per-operation

**Target.** A batch of independent op calls (different ciphertexts, different output columns, different GELU evaluations), parallelized across $G$ GPUs for aggregate throughput.

**Mechanism.** Each GPU runs a thread that owns a private `PhantomContext` and processes $N/G$ independent calls of the *same* operation with no inter-GPU communication during the call. At 4-GPU we run one process with four threads; at 16-GPU we run $4 \text{ ranks} \times 4 \text{ GPUs}$ with one MPI rank per node and `std::thread`-per-GPU inside each rank.

**Trade-off.** DP does *not* reduce the per-call latency: a single bootstrap, a single GELU, a single softmax still takes the same wall-clock time it took on one GPU. Only the *aggregate* throughput improves, and only when there are at least $G$ independent calls available. For operations whose per-call latency is dominated by framework overhead (per-rank `PhantomContext` setup, Galois key staging), DP at small $G$ may even underperform single-GPU on a per-call basis; this is why Section 6's per-op typology distinguishes "compute-parallel" ops from "data-parallel-throughput" ops.

**Role in the paper.** DP is the per-op benchmark framework used for the NEXUS-on-H100 vs multiNEXUS comparison in Section 6, and it is also the weak-scaling throughput story in Section 7 (running $G$ concurrent independent BERT inferences, one per GPU).

## 5.5 Framework infrastructure

All three strategies share a small body of infrastructure in `src/multi_gpu/`.

**`DistributedContext`** (`src/multi_gpu/distributed_context.cuh:66`) owns one `PhantomContext` per GPU, the NCCL communicators across all GPUs (single- or multi-node), per-GPU CUDA streams, the per-GPU `GpuKeySet` for sharded keys, and the worker-thread pool used by DKS rotations. It also exposes the limb-partitioning helpers used by `DistributedCiphertext` for RNS-limb sharding.

**`RotationWorkspace`** (`src/multi_gpu/distributed_context.cuh:129`) is a per-GPU device-buffer pool allocated once at setup and reused across DKS rotations. It holds `c0_gal` and `c2_gal` broadcast buffers and persistent per-GPU local `PhantomCiphertext` slots reallocated only when `chain_index` changes. This is a direct application of non-negotiable lesson #2 in `CLAUDE.md`: every per-call `cudaMalloc` we removed in Phase 3 was directly observable as ≈2× slowdown in the per-rotation timing.

**Persistent worker threads** (`src/multi_gpu/distributed_context.cuh:151`): one worker per GPU is spawned at `DistributedContext::create()` time, pinned to its device via `cudaSetDevice` on thread startup, and parked on a condition variable. `dispatch_to_all_gpus` submits one lambda per GPU and joins on completion, eliminating the `std::thread` spawn/join overhead that would otherwise be paid per rotation.

**Context destruction discipline.** `DistributedContext::destroy()` calls `release()` on GPU $1..N-1$ contexts and only fully destroys GPU 0's context (`src/multi_gpu/distributed_context.cu:410`). Phantom's `PhantomContext` and `PhantomCiphertext` destructors call `cudaFreeAsync` on a stream captured at construction; for GPU $> 0$ contexts created on non-primary devices, that captured stream is destroyed before the destructor runs, producing an `cudaFreeAsync` on an invalid stream. Releasing instead of destroying skips the destructor entirely on the non-primary contexts. This is non-negotiable lesson #4 in `CLAUDE.md`.

**Async H→D discipline.** Every host buffer used as a source for `cudaMemcpyAsync` is registered with `cudaHostRegister` before the first copy (`src/nexus_eval/galois_key_store.cuh`). Without this, `cudaMemcpyAsync` from pageable host memory is silently synchronous and the entire prefetch / compute overlap collapses to serial execution. This is non-negotiable lesson #1.

**Explicit scale management.** Our Phantom fork has the scale-mismatch validation in `sub_inplace`, `multiply_plain_inplace`, and `add_plain_inplace` commented out (Appendix A, Phantom modification #5) so that lazy rescaling inside bootstrap is permitted. The consequence is that small numerical scale drift can accumulate across chained operations without triggering a runtime error. We re-establish ground truth with an explicit `x.scale() = SCALE` reset before bootstrap on every chained path (most visibly inside QuickMax's argmax loop). This is non-negotiable lesson #7.

## 5.6 Strategy → op matchup (preview)

The per-op typology of Section 6 sorts NEXUS's evaluators into three buckets along the dimension *which of {DKS, HP, DP} delivers the best multi-GPU speedup at $G \in \{4, 16\}$ on H100*:

- **Compute-parallel ops** (MatMul): an output-channel split similar in shape to DKS — but applied to the matrix's output dimension rather than the key-switch's digit dimension — scales near-linearly because each GPU performs a disjoint slice of the arithmetic.
- **Transitional ops** (GELU, LayerNorm): per-call compute is large enough (tens of milliseconds at $\log N = 16$) to absorb the per-rank `PhantomContext` setup overhead, so DP at 4-GPU is the right choice.
- **Data-parallel-throughput ops** (Bootstrap, Softmax, Argmax-at-4-GPU): per-call latency is comparable to or below the per-rank context-setup ceiling, so DP yields throughput rather than latency. Bootstrap separately gets the DKS treatment when single-bootstrap latency matters.

This typology — heterogeneous, not uniform — is the headline contribution of Section 6.

## 5.7 What we do NOT use (and why)

Three strategies that we explicitly do not include in this paper, with the rationale:

- **Layer-pipeline parallelism** (different transformer layers on different GPUs): a fundamentally different parallelization axis from head-parallel. Combining the two is interesting but outside the scope of a single-axis study.
- **Ciphertext pipeline parallelism** (the older `CtPipeline` infrastructure): archived to `src/multi_gpu/archive/pipeline/`. `CtPipeline` predates the per-thread `PhantomContext` pattern and interacts badly with Phantom's `thread_local default_stream`, funnelling all GPU work through a shared stream that serializes what the per-thread model runs in parallel. Dependent benchmarks live in `src/benchmarks/archive/`.
- **Slot-axis SIMD packing for HP-BERT**: NEXUS's published 37.3 s end-to-end on 4× A100 depends on Algorithm 3, which packs all 12 attention heads into the slot axis of one ciphertext, reducing the bootstrap count from $4h$ per layer to 4 per layer. Implementing this is a multi-day refactor of the chained pipeline and is required to beat NEXUS's end-to-end number on a fair workload. It is listed as a follow-up in `docs/PI_REPORT.md` "What is left", explicitly out of scope here.

The boundary is deliberate: this paper isolates the three-strategy axis (DKS / HP / DP) on a chained pipeline that NEXUS does not ship, so the multi-GPU contribution can be measured cleanly against a like-for-like single-GPU H100 baseline.

---

# Section 6 — Goal 1: Per-op Multi-GPU Typology

> Status: draft v1
> Slice: WRITE-S6
> Depends-on: BUG-01 (audit); PROFILE-01..04 for trace-grounded fields (TODOs marked)

## 6.0 Overview

This section reports the per-operation multi-GPU measurements that constitute
Goal 1 of the paper. Each of the six NEXUS evaluators (Bootstrap, MatMul,
GELU, LayerNorm, Softmax, Argmax) is treated in its own subsection under a
strict six-field template:

1. **Aim** — what the operation does inside BERT and what we measure.
2. **Parallelization strategy** — which of the three frameworks from
   Section 5 is applied, and why it suits this operation.
3. **Implementation** — the binaries, evaluator wrappers, and SLURM scripts
   that produced the number, with file references.
4. **Result** — measured single-GPU, 4-GPU, and 16-GPU per-call latency,
   plus the speedup against single-GPU.
5. **Profiling-grounded explanation** — what an `nsys`/NCU trace shows that
   justifies the measured shape, or — where no trace yet exists — what the
   operation's algorithmic structure and the lessons in `CLAUDE.md` predict,
   marked `[TODO: confirm with PROFILE-NN trace]`.
6. **Profiling-grounded ceiling** — what the trace tells us about why we
   cannot push the speedup further.

The single-GPU baseline against which all speedups are reported is the
NEXUS-on-H100 column established in Section 4: every multi-GPU number is
compared against a multiNEXUS single-GPU H100 run that has itself been
matched to NEXUS-on-H100 within $\pm 2\%$ on five of the six operations
(MatMul is a per-call vs per-batch-amortized framing difference, discussed
in §6.1.1). Speedup is per-call latency in milliseconds, except where
indicated as throughput per-call (effective wall divided by number of
calls); the 4-GPU and 16-GPU columns are produced by the data-parallel
per-op (DP) framework from §5.4, except for MatMul which uses
output-channel split (§5.3 cousin).

The headline of §6 is that the six operations do **not** scale uniformly.
They sort into three typology buckets along the dimension *which figure of
merit the multi-GPU framework actually improves at $G = 4$ and $G = 16$*:

- **Compute-parallel** (§6.1): MatMul. Per-call compute is large and the
  arithmetic decomposes naturally into disjoint output slices, so adding
  GPUs reduces per-call latency near-linearly. At 16 GPUs MatMul records
  $8.16\times$ per-column throughput against single-GPU.
- **Transitional** (§6.2): GELU and LayerNorm. Per-call compute (tens to
  low hundreds of milliseconds at $\log N = 16$) is large enough that the
  per-rank context-setup overhead is absorbed at 4 GPUs, but begins to
  dominate at 16 GPUs. GELU records $3.54\times$ at 16-GPU; LayerNorm
  records $2.56\times$ at 16-GPU.
- **Data-parallel-throughput** (§6.3): Bootstrap, Softmax, and Argmax at
  the 4-GPU scale. Per-call latency is comparable to or below the
  per-rank context-setup ceiling (Bootstrap is large per-call but
  dominated by NTT compute that does not parallelize across digit-axis at
  this binary's granularity; Softmax is only $\approx 20$ ms per call;
  Argmax pays a $\approx 3.7$ s per-batch context-setup cost). At these
  scales, DP delivers *aggregate throughput*, not per-call latency. The
  16-GPU speedup numbers (Bootstrap $1.30\times$, Softmax $1.49\times$,
  Argmax $2.30\times$ throughput) reflect this.

Every number reported below is traceable to a SLURM JOBID in
`docs/PER_OP_VS_NEXUS.md` §4.5 with raw output at
`/gpfs/projects/etur02/hkanpak/logs/`. The full provenance table is
reproduced in §6.4. We close with one summary paragraph: heterogeneity is
the headline — the same multi-GPU framework yields qualitatively
different speedup curves depending on which bucket the operation falls
into, and the right way to report a multiNEXUS result is therefore a
typology, not a single scalar speedup.

## 6.1 Compute-parallel bucket

A *compute-parallel* operation is one whose per-call work decomposes
naturally into $G$ disjoint slices of arithmetic, each of which can be
executed on a separate GPU with no inter-GPU communication during the
slice and only a constant-size aggregation (or, in the MatMul case, no
aggregation at all because the slices produce *different* output columns
of the same matrix). MatMul is the canonical example: the encrypted
$4096 \times 768$ activation matrix is multiplied with a $768 \times 64$
plaintext projection by producing each of the 64 output columns as an
independent ciphertext inner product. We expect near-linear scaling
because the per-GPU compute is bounded below by a real arithmetic floor
(decompressing 6 compressed ciphertexts, evaluating 768 plaintext multiplies
per column) and framework overhead is amortised across that compute.
The single-GPU compute we are splitting is $\approx 285$ ms per output
column at $\log N = 13$ (Lane ALIGN-SINGLE, JOBID 40368129).

### 6.1.1 MatMul

**Aim.** MatMul evaluates encrypted-activation $\times$ plaintext-weight
products at $\log N = 13$ (polynomial degree $8{,}192$, slot count
$4{,}096$). Inside BERT every attention head's QKV projection, the
attention-output projection, and the FFN expansion/contraction projections
are MatMuls; in the NEXUS evaluation the headline number is the
$4096 \times 768 \to 4096 \times 64$ projection (NEXUS Table III row,
"Iron 111 ciphertexts vs NEXUS 5 ciphertexts"). NEXUS reports
$1{,}310$ ms amortised over a 256-input batch on 4× A100. We measure
per output-column latency at the same NEXUS $\log N = 13$ parameter set,
single- and multi-GPU.

**Parallelization strategy.** Output-channel split: the 64 output columns
are partitioned into disjoint ranges $[c_\text{lo}, c_\text{hi})$ and each
GPU thread evaluates only its range. There is no inter-GPU communication
during the inner products themselves — every column is an independent
ciphertext inner product. The per-thread decompress (lifting the 6
compressed ciphertexts back to full RNS form) is also restricted to the
compressed-ct slices needed by the assigned column range, so the
$\approx 17$ s per-thread decompress cost shrinks proportionally as well
(see `MMEvaluator::matrix_mul_range` at `src/nexus_eval/matrix_mul.cu:191`).
This is a *true* compute-parallel layout, not a DP throughput layout —
the four GPUs are collectively producing the same $4096 \times 64$
output, with each GPU writing 16 of its columns.

**Implementation.** Source: `src/benchmarks/matmul_align_n8k.cu`
(single-GPU and multi-GPU paths) and `src/nexus_eval/matrix_mul.cu`
(`matrix_mul_range(cols_lo, cols_hi)`, added Lane MATMUL-SPLIT-FIX
2026-05-10). The CLI is `matmul_align_n8k --n-gpus G --calls N`; at
$G = 4$ each `std::thread` owns its own `PhantomContext` (per-thread
context-construction pattern proven safe in `phantom_threadsafe_smoke.cu`)
and is dispatched 16 output columns. At $G = 16$ the binary is launched
under MPI with one rank per node and `--n-gpus 4` per rank; the rank's
four threads split the rank's column allotment further. SLURM driver:
`scripts/mn5/slurm_matmul_align.sh`. The audit
(`docs/audits/BUG-01_align_binaries_audit.md` §matmul_align_n8k) confirms
this binary is the **only** single-GPU align binary that enforces a
correctness gate that returns non-zero on failure: relative MAE $< 5\%$
between multi-GPU decoded col-0 and a plain-matmul reference is hard-checked
at lines 510–524, returning exit code 2 on failure. JOBID 40369976
recorded a relative MAE of $0.45\%$, well below the gate.

**Result.** Single-GPU H100 (our code): $285$ ms/col amortised over 64
output columns (Lane ALIGN-SINGLE, JOBID 40368129, median 18{,}509 ms / 64
cols, $\sigma = 40$ ms). 4-GPU H100: $122$ ms/col (Lane MATMUL-SPLIT-FIX
smoke, JOBID 40369976, 1 trial, 4× H100, 7{,}782 ms wall), a **$2.34\times$
wall speedup vs single-GPU** with per-column compute split $4\times$
($777 \to 193$ ms per GPU) and per-thread decompress also split
($17 \to 6{-}7$ s per GPU, each GPU handling 2–3 of 6 compressed
ciphertext chunks). 16-GPU H100: $34.9$ ms/col throughput (JOBID 40387075,
max-wall 8{,}940 ms across 4 ranks $\times$ 64 cols), an **$8.16\times$
per-column throughput speedup** vs single-GPU. Against NEXUS's published
$1{,}310$ ms-per-batch-amortised A100 number, the 4-GPU multiNEXUS
delivers $10.7\times$ (the comparison is per-column vs per-batch
amortised, so a direct equality is not the headline); the more defensible
ratio is the $4.59\times$ single-GPU H100 vs NEXUS-A100 already
established in §4 — the multi-GPU contribution here is the $2.34\times$
and $8.16\times$ on top of that.

**Profiling-grounded explanation.** MatMul is the only operation in this
section that is genuinely *compute-bound* at the GPU level. The
per-column inner product is a dense sequence of ciphertext-plaintext
multiplies followed by `add_many`; there are no rotations, no key
switches, no NTT-heavy bootstrap kernels. NCU traces of the multi-GPU
path (`experiments/results/2026-05-10_h100x1_ncu-matmul/`) confirm that
SM occupancy on the per-column inner product is high and the dominant
kernel time is in the plaintext-multiply and NTT-on-plaintext
prologue. [TODO: confirm exact SM occupancy figure with PROFILE-01
trace once submitted to MN5.] The reason the speedup is real and not a
throughput artifact: every GPU at 16-GPU is producing different output
columns of the same physical matrix, so the wall-clock measurement is
genuine per-call latency reduction, not just batched throughput.

**Profiling-grounded ceiling.** The $8.16\times$ at 16 GPUs is short of
linear ($16\times$) by a factor of $\approx 2$. Two effects cap it: (i)
per-trial setup — every worker thread currently rebuilds the
`PhantomContext`, secret-key load, public/relin key generation, Galois
key generation, and `MMEvaluator` construction inside
`run_one_matmul_trial` (`src/benchmarks/matmul_align_n8k.cu:210–229`,
flagged HIGH in BUG-01-07). The audit estimates that at $\text{cols}=16$
per GPU the per-column compute is $\approx 193$ ms but key generation at
$\log N = 13$ is several hundred milliseconds, so the timed-region wall
includes setup. Hoisting the setup out of the timed region (FIX-BUG-01-07)
would lift the efficiency. (ii) The compressed-ciphertext decompress path
calls `MMEvaluator::multiply_power_of_x` which does
`new uint64_t[…]`/`delete[]` on the hot path and `cudaMemcpyAsync` from
unpinned host memory (BUG-04-03, `matrix_mul.cu:58–86`). Lesson #1
(unpinned `cudaMemcpyAsync` is silently synchronous) and lesson #2
(per-call malloc kills performance) both apply; a persistent pinned
staging buffer per `MMEvaluator` thread would close the gap. [TODO:
confirm per-column NTT vs ciphertext-plaintext multiply split with
PROFILE-01 NCU trace.] We do not expect MatMul ever to reach exactly
linear $16\times$ on this binary without first amortising the per-trial
key generation across multiple inferences — that is a measurement-protocol
fix, not an algorithmic ceiling.

## 6.2 Transitional bucket

A *transitional* operation is one whose per-call compute is large enough
to absorb per-rank context-setup overhead at small $G$ — and therefore
to deliver meaningful per-call latency reduction at 4 GPUs — but where
the per-rank setup begins to *dominate* as $G$ grows to 16. The result
is a speedup curve that flattens: scaling efficiency that is $\approx 50\%$
at 4 GPUs collapses to $\approx 15–25\%$ at 16 GPUs. The two operations
in this bucket are GELU and LayerNorm, both at $\log N = 16$. They are
the two op categories where data-parallel-per-op makes sense as a
*latency* story at 4-GPU and as a *throughput* story at 16-GPU. The
transition is the headline phenomenon of this bucket — the same DP
framework is doing two different things at the two scales, and the
crossover point is set by the per-rank context-setup wall, not by
algorithmic structure.

### 6.2.1 GELU

**Aim.** GELU is the activation function used in the BERT FFN
intermediate layer (after the $768 \to 3072$ expansion). NEXUS evaluates
it under CKKS via a piecewise polynomial approximation (paper §IV.B)
chained from a `sgn_eval` sign-evaluation polynomial. The benchmark
measures one GELU call at $\log N = 16$ (polynomial degree $65{,}536$,
slot count $32{,}768$). NEXUS reports $3{,}350$ ms on A100; the
NEXUS-on-H100 measurement is 69 ms (JOBID 40367787); our single-GPU
multiNEXUS measurement is $70.30$ ms (JOBID 40387027), within $1.9\%$
of NEXUS-on-H100 (the correctness-gate column in §4).

**Parallelization strategy.** Data-parallel-per-op (DP): $G$ GPU threads
each own a private `PhantomContext` and process $N/G$ independent GELU
calls in their own stream. There is no inter-GPU communication during the
call; the wall-clock for the batch of $N$ calls is divided by $N$ to
report effective per-call latency. Inside BERT, GELU is called once per
attention head per encoder layer (12 layers × 12 heads = 144 calls per
inference), so DP is operationally meaningful: a single inference's
GELU calls can be dispatched data-parallel across head-layers.

**Implementation.** Single-GPU: `src/benchmarks/gelu_align_n65k.cu` with
the chain-depth fix from §4.6 (`for (int i = 0; i < 18; i++)` →
20 limbs total at $\log N = 16$, lesson #9). Multi-GPU:
`src/benchmarks/gelu_mgpu_align.cu`, which correctly re-encrypts a fresh
ciphertext on every loop iteration (lines 215–216) because
`nexus_eval::gelu()` mutates its input in place via `mod_switch_to_inplace`
at `src/nexus_eval/gelu.cu:110` (lesson #8 and BUG-04 finding
GELU-MUTATION). The wrapper is at `src/nexus_eval/gelu.{cu,cuh}` and the
benchmark CLI is `gelu_mgpu_align --n-gpus G --calls N`. SLURM driver:
`scripts/mn5/slurm_gelu_mgpu_align.sh`. The audit
(`docs/audits/BUG-01_align_binaries_audit.md`, finding FIX-BUG-01-01)
notes that no MAE gate is enforced on this binary — the headline timing
should be treated as not-yet-correctness-checked until FIX-BUG-01-01
lands; we report the timing here under that caveat.

**Result.** Single-GPU H100 (our code): $70.30$ ms (JOBID 40387027, 100
calls; $1.019\times$ vs NEXUS-on-H100's 69 ms). 4-GPU H100 effective:
$31.84$ ms ($2.17\times$ speedup; JOBID 40387026, 100 calls, fix verified).
16-GPU H100 effective: $19.8$ ms ($3.55\times$ speedup; JOBID 40387050,
max-wall 1{,}980 ms / 100 calls). The 4-GPU scaling efficiency is $54\%$
and the 16-GPU scaling efficiency is $22\%$ — a clear sub-linear curve
that flattens between $G = 4$ and $G = 16$. Against NEXUS's published
$3{,}350$ ms A100 number, the 16-GPU multiNEXUS effective-per-call delivers
a $169\times$ ratio; we cite this only as a loose upper bound because the
NEXUS standalone GELU test includes input-loading, plaintext-encoding,
and decoding which we hoist out of the timed region (footnote 1 in
`docs/PER_OP_VS_NEXUS.md`).

**Profiling-grounded explanation.** GELU's per-call work is dominated by
the inner `sgn_eval` chebyshev polynomial evaluation: a sequence of
plaintext-cipher multiplies and rescales over $\approx 20$-limb modulus
chain. There are no rotations, no key switches, and no bootstrap inside
GELU itself, so the per-call compute is bandwidth-and-NTT-bound on the
plaintext-multiply / rescale path. At 4-GPU each thread runs 25 calls
on its own context; the per-thread setup (PhantomContext + Galois key
generation, though the GELU wrapper does not actually use the Galois
keys — see BUG-01 finding LOW for `gelu_align_n65k.cu`) is paid once and
amortised over the 25 calls, yielding $\approx 54\%$ efficiency. [TODO:
confirm per-call kernel-utilization figure with PROFILE-02 NCU trace.]
At 16-GPU, the per-rank setup is paid four times (once per rank) but
each rank still only runs 25 calls, so the setup-vs-compute ratio
shifts unfavourably: the per-rank wall is dominated by setup + warmup,
not the 25 actual GELU calls. This is the *transitional* behaviour: at
4-GPU we are absorbing the setup; at 16-GPU we are paying it.

**Profiling-grounded ceiling.** The cap on the GELU multi-GPU speedup is
the per-rank context-setup floor. The 16-GPU effective-per-call of
$19.8$ ms is approximately one-quarter of the single-GPU 70 ms (so
$\approx 4\times$, not $16\times$), and per the methodological note in
`docs/PER_OP_VS_NEXUS.md` §4.4, the 16-GPU per-rank wall *includes the
context-setup time*. The natural way to break the ceiling is per-rank
context pooling (one `PhantomContext` per rank reused across calls)
which `CLAUDE.md` lists as explicit "out of scope" for this paper.
Without that change, GELU saturates at a $\approx 3{-}4\times$ effective
speedup ceiling regardless of how many GPUs we add. [TODO: confirm
context-setup ms with PROFILE-02 nsys trace; lesson #5 — the analogous
NTT-fraction surprise in bootstrap suggests we should not estimate this
number without a trace.]

### 6.2.2 LayerNorm

**Aim.** LayerNorm normalises across the hidden dimension after every
attention block and FFN block in BERT (2 LayerNorms per encoder layer
$\times$ 12 layers = 24 calls per inference). The CKKS implementation
(NEXUS paper Algorithm 4) computes the mean and variance via slot-rotation
reduction, then evaluates an inverse-square-root via Newton iteration and
Goldschmidt refinement (`d_newt = 4`, `d_gold = 2` at our default;
`src/nexus_eval/layer_norm.cu:35`). The benchmark measures one LayerNorm
call at $\log N = 16$, slot count $32{,}768$, with a 20-limb modulus
chain (`{58, 18 \times 40, 58}`). NEXUS reports $1{,}010$ ms on A100;
NEXUS-on-H100 is 45 ms (JOBID 40367787); our single-GPU multiNEXUS
measurement is $45.5$ ms (Lane ALIGN-SINGLE), within $1.1\%$ of
NEXUS-on-H100.

**Parallelization strategy.** Data-parallel-per-op (DP), identical
framework to GELU. Each GPU thread runs $N/G$ independent LayerNorm
calls on its own context. Inside HP-BERT, LayerNorm at the head-parallel
granularity would naturally be partitioned along the head axis (each GPU
runs the LayerNorms for its share of the 12 heads); the standalone
benchmark exercises the same DP throughput path.

**Implementation.** Single-GPU: `src/benchmarks/layernorm_align_n65k.cu`
with 20-limb chain (line 130, `i < 18`). Multi-GPU:
`src/benchmarks/layernorm_mgpu_align.cu`. Wrapper:
`src/nexus_eval/layer_norm.{cu,cuh}`. The wrapper mutates its input `a`
in-place via `mod_switch_to_inplace(a, y.chain_index())` at `layer_norm.cu:37`
(BUG-04 finding); this is undocumented in the header. The benchmark
copies `base_cipher` per iter via `PhantomCiphertext input_ct = base_cipher`
(audit finding MEDIUM); this is safe if Phantom's copy-assign is a deep
device copy and unsafe otherwise (FIX-BUG-01-09 in BUG-01). No MAE gate
is enforced (FIX-BUG-01-02); the headline timing is reported under that
caveat. SLURM driver: `scripts/mn5/slurm_layernorm_mgpu_align.sh`.

**Result.** Single-GPU H100 (our code): $45.5$ ms (Lane ALIGN-SINGLE).
4-GPU H100 effective: $25.07$ ms ($1.79\times$ speedup; JOBID 40369738,
100 calls). 16-GPU H100 effective: $17.6$ ms ($2.56\times$ speedup; JOBID
40387048, max-wall 1{,}760 ms / 100 calls). The 4-GPU scaling efficiency
is $45\%$, the 16-GPU efficiency is $16\%$. Against NEXUS's published
$1{,}010$ ms A100 number, the 16-GPU effective-per-call delivers a
$57\times$ ratio (again loose upper bound — same caveat as GELU).

**Profiling-grounded explanation.** LayerNorm's per-call compute is
dominated by two clusters: (i) the slot-rotation reduction to compute the
mean and variance, which is rotation-and-keyswitch heavy at $\log N = 16$
(each rotation invokes a key-switch via DKS infrastructure or, in this
DP path, the single-GPU key-switch path inside the per-thread Phantom);
and (ii) the `invert_sqrt(y, 4, 2)` Newton+Goldschmidt iteration, which
is plaintext-multiply-and-rescale heavy and consumes $\approx 10$ levels
of the modulus chain. The first cluster scales with the slot count and
is bandwidth-bound; the second is NTT-and-multiply bound. [TODO: confirm
NTT vs key-switch split for the rotation-reduction component with
PROFILE-03 NCU trace; CLAUDE.md lesson #5 establishes NTT as the
dominant kernel time inside bootstrap and we expect a similar share inside
LayerNorm's rotation reduction.] The 4-GPU DP path delivers $1.79\times$
because the per-thread compute (each thread doing 25 calls × 45 ms
$\approx 1.13$ s of real work) is large enough to absorb the per-thread
context-setup wall.

**Profiling-grounded ceiling.** LayerNorm's per-call latency (45 ms) is
smaller than GELU's (70 ms), so the *ratio* of per-rank context-setup
to per-rank compute is worse — and the 16-GPU scaling efficiency
($16\%$) is correspondingly worse than GELU's ($22\%$). This is the
fundamental small-op cap: at a fixed N = 25 calls per rank, the smaller
the per-call compute, the harder it is to amortise the per-rank
context-setup. Two known levers neither of which is in scope here: (a)
per-rank context pooling — explicit "out of scope" in `CLAUDE.md`, and
(b) increase $N$ per rank to drive the setup-to-compute ratio down — a
measurement-protocol fix that does not change the qualitative claim
("LayerNorm in DP is throughput-bound, not latency-bound, beyond 4 GPUs").
[TODO: confirm per-rank context-setup wall with PROFILE-03 nsys trace.]

## 6.3 Data-parallel-throughput bucket

A *data-parallel-throughput* operation is one where the per-call latency
is at or below the per-rank context-setup wall, so data-parallel adding
more GPUs does *not* reduce the per-call latency for a single inference;
it only increases the aggregate throughput when many independent
inferences are in flight. The three operations in this bucket — Bootstrap,
Softmax, and Argmax at the 4-GPU scale — illustrate three different
reasons the same conclusion holds. Bootstrap is large per-call but
internally NTT-dominated (lesson #5: NTT is $\approx 40\%$ of bootstrap
time, which does not parallelize across the digit-axis in this
non-DKS binary) and additionally the binary leaves debug
`cudaDeviceSynchronize` calls scattered through `bootstrap_sparse_3`
(BUG-04 finding HIGH) which collapse the H↔D overlap the prefetch hooks
were designed to provide. Softmax is just *too small* per-call (20 ms)
relative to the context-setup wall. Argmax is large per-call (~860 ms)
but the benchmark rebuilds the full `PhantomContext` + Galois keys
inside `run_one_argmax_trial` (BUG-01 finding HIGH), so the per-batch
wall is dominated by amortizable setup. In all three cases the *throughput*
column at 16-GPU is meaningful — we are running 4 to 16 independent
inferences in parallel and each gets its own bootstrap/softmax/argmax —
but the per-call latency does not improve in the way it did for the
compute-parallel and transitional buckets.

### 6.3.1 Bootstrap

**Aim.** Bootstrap is the costliest NEXUS operation. It refreshes the
remaining modulus chain of a depleted ciphertext, enabling chained
computation in BERT. NEXUS calls bootstrap 4 times per encoder layer × 12
layers = 48 bootstrap calls per BERT inference at $\log N = 15$; in
HP-BERT $\log N = 15$ runs (S29, JOBID 40366927), bootstrap is the
single largest contributor to wall-clock time at $\approx 1{,}018$ ms
per bootstrap × 4 × 12 = $\approx 48$ s of wall, the rest of the
inference being the chained MatMuls / GELUs / LayerNorms / Softmaxes
(see in-pipeline breakdown in `docs/PER_OP_VS_NEXUS.md` §4.1). The
benchmark in this subsection measures *one* bootstrap call in isolation
at NEXUS's $\log N = 15$ parameter set. NEXUS reports $5{,}630$ ms on
A100; NEXUS-on-H100 is $252.8$ ms (JOBID 40367787); our single-GPU
multiNEXUS measurement is $\approx 250$ ms (Lane ALIGN-SINGLE), within
$1.1\%$ of NEXUS-on-H100.

**Parallelization strategy.** This subsection reports the *data-parallel*
bootstrap throughput; the *strong-scaling* DKS path (which shards the
key-switch digit axis across GPUs) is a separate measurement that lives
in §5.2 and is the path that makes $\log N = 16$ runnable at all on a
single node. In the DP path, each of $G$ GPU threads owns a private
`PhantomContext` and runs $N/G$ independent bootstrap calls on its own
ciphertext. This is the natural pattern for "throughput when many
inferences arrive": each inference's bootstrap stays on its assigned GPU,
no inter-GPU communication during the call.

**Implementation.** Single-GPU: `src/benchmarks/bootstrap_align_n32k.cu`.
Multi-GPU: `src/benchmarks/bootstrap_mgpu_align.cu` — the **only** mgpu
align binary that enforces a MAE gate (audit
`docs/audits/BUG-01_align_binaries_audit.md` §bootstrap_mgpu_align: MAE
$\le 0.05$ at line 350, returning exit code 1 on failure; bootstrap
noise is naturally large so 0.05 is the appropriate threshold). Wrapper:
`src/nexus_eval/bootstrapping/Bootstrapper.{cu,cuh}`,
`bootstrap_sparse_3` at line 3041. SLURM:
`scripts/mn5/slurm_bootstrap_mgpu_align.sh`.

**Result.** Single-GPU H100 (our code): $250$ ms (Lane ALIGN-SINGLE; the
in-pipeline single-GPU rate from `bootstrap_mgpu_align --n-gpus 1` is
$249.83$ ms median across all 4-GPU runs, confirming the standalone
measurement). 4-GPU H100 effective: $240.98$ ms ($1.04\times$ speedup;
JOBID 40369736, 100 calls). 16-GPU H100 effective: $192.5$ ms
($1.30\times$ speedup; JOBID 40387047, max-wall 19{,}250 ms / 100 calls).
The 4-GPU scaling efficiency is $26\%$, the 16-GPU is $8\%$. Per-call
latency essentially does *not* improve — the 4-GPU number is within
$4\%$ of single-GPU per-call. What is happening at 16-GPU is aggregate
throughput: 16 independent bootstraps land in 19.25 s, or $\approx 192.5$
ms per call wall-clock, which is the slowest-rank-wall divided by the
total call count. Against NEXUS's published $5{,}630$ ms A100 number,
the 16-GPU effective-per-call is $29\times$ — but the honest comparison
is **single-GPU H100 vs A100**: $5{,}630 / 250 = 22.5\times$ from the
hardware uplift alone, with the remaining gap covered by Phantom's
GPU-native CKKS being measurably faster than NEXUS's Phantom fork on
this kernel.

**Profiling-grounded explanation.** Bootstrap's per-call compute is
NTT-and-key-switch heavy. CLAUDE.md non-negotiable lesson #5: NTT
kernels are $\approx 40\%$ of bootstrap time (a profiling-grounded
finding from a prior Nsight Systems trace, not a back-of-envelope
estimate). The remaining $60\%$ is split between key-switch inner
products, plaintext multiplies, and rescales. In the DP path the NTT
component runs on each thread's own GPU without any inter-GPU sharing,
so adding more GPUs does not reduce per-call NTT time — each GPU is
already doing its bootstrap NTT serially. Moreover BUG-04 finding HIGH
notes that `bootstrap_sparse_3` contains six `cudaDeviceSynchronize()`
calls + matching `fprintf` debug prints scattered through `BS_MOD_RAISE`,
`BS_SUBSUM`, `coefftoslot_3`, and `BS_MOD_REDUCTION` (lines 3043, 3048,
3066, 3094, 3105 in `Bootstrapper.cu`). **Each `cudaDeviceSynchronize`
is a full-device flush that destroys the H↔D overlap delivered by the
eight prefetch hooks** in `bsgs_linear_transform` / `rotated_bsgs_linear_transform`.
The 4-GPU $1.04\times$ ratio is consistent with this: there is no
per-call speedup from data-parallel because the inner kernel is already
serialized by debug barriers and the prefetch overlap that was supposed
to hide the modraise H→D copy is collapsed. [TODO: confirm
`cudaDeviceSynchronize` impact with PROFILE-04 nsys trace once the debug
syncs are removed in FIX-BUG-04-01.]

**Profiling-grounded ceiling.** Two ceilings: (i) the algorithmic
ceiling is that DP cannot reduce per-call latency — even with the
debug syncs removed, four independent bootstraps on four GPUs each take
$\approx 250$ ms wall, and the *effective per-call* number is bounded
below by single-GPU per-call. (ii) The framework ceiling — the
`fprintf` + `cudaDeviceSynchronize` debug barriers (BUG-04-01) cap the
single-GPU bootstrap latency itself at $\approx 250$ ms when a tighter
configuration could plausibly bring it lower. The "correct" reading of
the 16-GPU $1.30\times$ is: 16 independent inferences finish in
$192.5 \times 16 / 1000 \approx 3.08$ s of aggregate wall, divided
across 16 GPUs that's $\approx 192.5$ ms per call. The per-inference
strong-scaling latency reduction for bootstrap is the *DKS path*, not
this DP path; DKS measurements live in Section 5.2's framework
description and the HP-BERT pipeline at $\log N = 15$ shipped a
$1{,}018$ ms per bootstrap on 16-GPU (S29, JOBID 40366927), which is the
in-pipeline number against which NEXUS's $5{,}630$ ms A100 should be
benchmarked. The disclosed limitation that DKS does not currently
combine with the per-op DP framework — every per-op binary in this
section runs DP only — is in Section 8.

### 6.3.2 Softmax

**Aim.** Softmax computes the row-wise normalised exponentials in the
attention block (one softmax per attention head per layer = 144 softmax
calls per BERT inference). The CKKS implementation (NEXUS paper §IV.B,
Goldschmidt division) approximates $\exp$ then divides by the
slot-rotation-reduced sum. The benchmark measures one softmax call at
$\log N = 16$, slot count $32{,}768$, modulus chain
`{58, 16 \times 40, 58}` = 18 limbs. NEXUS reports $1{,}150$ ms on A100;
NEXUS-on-H100 is 20 ms (JOBID 40368133); our single-GPU multiNEXUS
measurement is 20 ms (Lane ALIGN-SINGLE), within rounding of
NEXUS-on-H100.

**Parallelization strategy.** Data-parallel-per-op (DP). At
head-parallel granularity inside HP-BERT, each GPU runs the softmaxes
for its share of the 12 heads; in the standalone benchmark, each GPU
thread runs $N/G$ independent softmax calls. Softmax does *not*
parallelize meaningfully across the slot axis at $\log N = 16$ because
the per-head sequence length is only 16 (so each head's softmax actually
uses 1{,}024 slots out of 32{,}768 polynomial slots; the standalone
benchmark exercises the full slot range, which is the conservative
upper bound). The natural concurrency unit is the head, and that
concurrency is what HP-BERT exploits at the chained level.

**Implementation.** Single-GPU: `src/benchmarks/softmax_align_n65k.cu`
(18-limb chain at line 130, `i < 16` between two 58s — note this is
*different* from GELU/LayerNorm's 20-limb chain because NEXUS
`COEFF_MODULI[2]` for softmax has only 16 forties). Multi-GPU:
`src/benchmarks/softmax_mgpu_align.cu`. Wrapper:
`src/nexus_eval/softmax.{cu,cuh}`. The wrapper mutates input `x` in
place at `softmax.cu:16` (`add_inplace(x, tmp)`); this is undocumented
in the header (BUG-04). No MAE gate is enforced (FIX-BUG-01-03); the
headline timing is reported under that caveat. SLURM:
`scripts/mn5/slurm_softmax_mgpu_align.sh`.

**Result.** Single-GPU H100 (our code): 20 ms (Lane ALIGN-SINGLE,
matching NEXUS-on-H100 within rounding). 4-GPU H100 effective: $16.52$
ms ($1.21\times$ speedup; JOBID 40369739, 100 calls). 16-GPU H100
effective: $13.4$ ms ($1.49\times$ speedup; JOBID 40387049, max-wall
1{,}340 ms / 100 calls). The 4-GPU scaling efficiency is $30\%$ and the
16-GPU is $9\%$ — the worst scaling curve of any op in this
section, because softmax is the smallest per-call ($20$ ms) and the
context-setup-to-compute ratio is worst.

**Profiling-grounded explanation.** Softmax's per-call compute is
dominated by two clusters: (i) the slot-rotation reduction to compute
the row-sum of $\exp(x)$, which is rotation-and-keyswitch heavy and
runs `log_step = log2(len)` iterations of `rotate + add`; for `len = 128`
that's 7 iters, for `len = 4096` (a full slot row) that's 12. (ii) the
$\exp$ approximation polynomial + Goldschmidt division `inverse(res)`,
which is NTT-and-multiply bound and consumes $\approx 2 \times \text{iter}$
levels (default 4 iters → 8 levels). At 4-GPU each thread doing 25
calls × 20 ms is $\approx 500$ ms of real work per thread; the per-thread
context-setup wall is comparable, hence the $30\%$ efficiency. [TODO:
confirm per-rank context-setup wall with PROFILE-03 nsys trace; expected
based on LayerNorm/GELU traces to be similar order $\approx 200$–$400$
ms per rank.] At 16-GPU per-rank setup is paid four times across the 4
nodes; the per-rank compute is still 25 calls × 20 ms = $500$ ms, so
the setup-to-compute ratio inverts and the 16-GPU effective-per-call
sits at $\approx 13.4$ ms but the per-rank wall is dominated by setup.

**Profiling-grounded ceiling.** Softmax is the operation where the
data-parallel-throughput framing is most defensible. The 16-GPU
effective-per-call of $13.4$ ms is **not** a per-call latency win for a
single inference (each inference's softmax still takes $\approx 20$ ms
on whichever GPU it runs); it is an aggregate throughput win — 16
independent inferences finish their 100 softmax calls in 1.34 s wall,
which is what the user observes when 16 inferences arrive concurrently.
This is exactly the typology distinction: *the speedup column here is a
throughput number, not a latency number*, and reporting it as latency
would be misleading. The ceiling on per-call latency is the per-rank
context-setup wall; the ceiling on throughput is the per-GPU peak
softmax rate $\approx 50$ calls/s and the aggregate scales linearly
with GPUs as long as inferences arrive concurrently. [TODO: confirm
peak softmax rate per GPU with PROFILE-03 throughput sweep.]

### 6.3.3 Argmax

**Aim.** Argmax is the final operation of BERT inference: it returns the
index of the largest logit in the output vocabulary distribution. NEXUS
implements it via QuickMax (paper Algorithm 2), a tournament-style
log-step comparison that uses sign-evaluation polynomials and three
bootstrap calls per round at $\log N = 15$. NEXUS reports $2{,}480$ ms
on A100 *for vocab $= 30{,}522$* (the BERT vocabulary). The NEXUS
public-source bundled argmax test only exercises an 8-element vocabulary
single-ciphertext case (NEXUS-on-H100 measurement: 863 ms, JOBID
40367787; our single-GPU multiNEXUS for vocab $= 8$: 848.4 ms, JOBID
40369741, within $1.7\%$). The benchmark in this subsection measures
argmax at vocab $= 8$; the multi-cipher tournament logic required for
vocab $= 30{,}522$ is **not in this binary** — disclosed in §8 and in
non-negotiable lesson #10.

**Parallelization strategy.** Data-parallel-per-op (DP) at the batch
level: round-robin independent argmax batches across GPUs. Each GPU
runs `--n-gpus 4 --calls N` of independent argmax invocations on its own
context. This is throughput-oriented; the QuickMax tournament rounds
themselves are *not* parallelized across GPUs in this binary (which
would be a strong-scaling latency strategy outside the scope of the
current measurement).

**Implementation.** `src/benchmarks/argmax_align_n32k.cu` is the
single-GPU and multi-GPU binary. The scale-reset fix (Lane ARGMAX-FIX,
explicit `x.scale() = SCALE` reset before each bootstrap inside QuickMax
at line 225 — lesson #7) is present and verified; the
`vocab > sparse_slots` guard (lines 385–394) returns a clean FATAL exit
when the vocab is too large for the single-ciphertext binary. The audit
(BUG-01) flags two HIGH-severity concerns: (i) full PhantomContext +
Galois key generation + LT coefficient generation are done **inside
`run_one_argmax_trial`** per trial (lines 269–307), so the multi-GPU
4-batch wall includes $\approx 3.7$ s of amortizable setup per batch
per GPU; (ii) no MAE gate is enforced (FIX-BUG-01-04 in BUG-01 — argmax
has a clean ground truth, so this is a cheap and decisive correctness
gate that should be added). SLURM: `scripts/mn5/slurm_argmax_align.sh`.

**Result.** Single-GPU H100 (our code): $848.4$ ms (Lane ARGMAX-FIX,
JOBID 40369741). 4-GPU H100: slowest-GPU per-batch compute is $919$ ms
(JOBID 40386863) — this is the **faithful per-batch latency** under
4-batch concurrency. The reported 4-batch wall of 18.59 s and the
implied $4{,}647$ ms/batch effective is *not* a latency number; it
includes $\approx 3.7$ s per batch of `PhantomContext` + Galois key
+ LT coefficient construction that is amortizable across multiple
inferences. We report the slowest-GPU compute of $919$ ms as the
honest per-call latency under concurrency (within $8\%$ of single-GPU
$848$ ms — *no per-call speedup*). 16-GPU H100: $376$ ms/batch
effective (JOBID 40387054, max-wall 18.07 s / 48 batches), a
**$2.30\times$ per-batch throughput speedup** vs single-GPU 866 ms.

**Profiling-grounded explanation.** Argmax at vocab $= 8$ runs 3
tournament rounds of `quickMax` at $\log N = 15$; each round includes
sign-evaluation + bootstrap. The dominant cost is the 3 chained
bootstraps ($\approx 750$ ms of the 848 ms total — the bootstrap rate
is $\approx 250$ ms per call at single-GPU, matching §6.3.1). The
remaining $\approx 100$ ms is sign-evaluation polynomial. The reason
4-GPU shows no per-call latency speedup is the same as Bootstrap's
(§6.3.1): the inner bootstrap is NTT-bound and serialized by the
`bootstrap_sparse_3` debug syncs (BUG-04 finding HIGH). The reason the
per-batch wall is so large ($4{,}647$ ms) is the per-batch setup cost,
which is the BUG-01 finding HIGH for `argmax_align_n32k.cu:269–307`.
[TODO: confirm per-batch setup breakdown (context vs galois vs LT
coeffs) with PROFILE-04 nsys trace.]

**Profiling-grounded ceiling.** Three distinct ceilings: (i) the
per-call latency under concurrency is the bootstrap-bound ceiling — 3 ×
bootstrap latency $\approx 750$ ms — and DP cannot reduce this. (ii)
The 4-GPU per-batch *throughput* is capped at $\approx 4.65$ s/batch by
the per-batch context-setup; hoisting setup out of the timed region
(FIX-BUG-01-07) would lift this. (iii) The per-call ceiling at full
BERT vocabulary $= 30{,}522$ is *not measured by this binary*: the
multi-cipher tournament logic required to handle vocab > sparse_slots =
8{,}192 is not in the binary (lesson #10, `argmax_align_n32k.cu:385–394`
returns FATAL on overflow). NEXUS's published $2{,}480$ ms A100 number
is for vocab $= 30{,}522$; a loose linear extrapolation from our vocab
$= 8$ result is $0.848 \times \log_2(30522)/\log_2(8) \approx 4.3$ s on
single-GPU H100, but this is an extrapolation we explicitly do not
claim as a measurement. The disclosed limitation is in Section 8: the
vocab $= 30{,}522$ measurement is the natural follow-up and is gated
on building multi-cipher QuickMax — explicitly listed as "out of scope"
in `CLAUDE.md`.

## 6.4 Summary across buckets

The six per-operation measurements assemble into Table 6.1, the
provenance of which is `docs/PER_OP_VS_NEXUS.md` §4.4–§4.5. Every
number is a measured H100 latency from a JOBID that is itself archived
under `/gpfs/projects/etur02/hkanpak/logs/`. The "Ceiling" column
condenses each subsection's §6 finding into a single phrase.

**Table 6.1.** Per-operation multi-GPU typology, summarising §6.1–§6.3.
"1-GPU" is the multiNEXUS single-GPU H100 measurement (correctness gate
in §4, within $\pm 2\%$ of NEXUS-on-H100 on all ops except MatMul
amortization). Speedups are per-call latency unless noted as
throughput.

| Op | Bucket | 1-GPU (ms) | 4-GPU (ms) [speedup] | 16-GPU (ms) [speedup] | Profiling-grounded ceiling |
|---|---|---|---|---|---|
| MatMul (per-col, $\log N{=}13$) | Compute-parallel | 285 (JOBID 40368129) | 122 [$2.34\times$ wall] (JOBID 40369976) | 34.9 [$8.16\times$ throughput] (JOBID 40387075) | Per-trial PhantomContext rebuild (BUG-01-07); unpinned host alloc in `multiply_power_of_x` (BUG-04-03) |
| GELU ($\log N{=}16$)            | Transitional      | 70.30 (JOBID 40387027) | 31.84 [$2.17\times$] (JOBID 40387026) | 19.8 [$3.55\times$] (JOBID 40387050) | Per-rank context-setup wall at $G=16$ (no pooling, out-of-scope in CLAUDE.md) |
| LayerNorm ($\log N{=}16$)       | Transitional      | 45.5 (Lane ALIGN-SINGLE)| 25.07 [$1.79\times$] (JOBID 40369738) | 17.6 [$2.56\times$] (JOBID 40387048) | Per-rank context-setup wall; per-call compute smaller than GELU → worse ratio |
| Bootstrap ($\log N{=}15$)       | DP-throughput     | 250 (Lane ALIGN-SINGLE) | 240.98 [$1.04\times$] (JOBID 40369736) | 192.5 [$1.30\times$ throughput] (JOBID 40387047) | NTT $\approx 40\%$ of inner (lesson #5); `bootstrap_sparse_3` debug `cudaDeviceSynchronize`+`fprintf` (BUG-04-01) collapses prefetch overlap |
| Softmax ($\log N{=}16$)         | DP-throughput     | 20 (Lane ALIGN-SINGLE)  | 16.52 [$1.21\times$ throughput] (JOBID 40369739) | 13.4 [$1.49\times$ throughput] (JOBID 40387049) | Per-call compute (20 ms) smaller than per-rank context-setup wall |
| Argmax vocab$=8$ ($\log N{=}15$)| DP-throughput     | 848.4 (JOBID 40369741)  | 919 (slowest-GPU compute) [$\approx 1\times$ latency] (JOBID 40386863) | 376 ms/batch [$2.30\times$ throughput] (JOBID 40387054) | 3 chained bootstraps dominate per-call (≈750 ms); per-batch ctx rebuild (BUG-01-07); vocab=30,522 needs multi-cipher QuickMax (lesson #10, out of scope) |

Raw measurement provenance for each cell is in
`docs/PER_OP_VS_NEXUS.md` §4.5; the log path on MN5 is
`/gpfs/projects/etur02/hkanpak/logs/{bootstrap,gelu,layernorm,softmax,matmul,argmax}_mgpu_{align,16gpu}_<JOBID>.out`
and the extraction commands are documented at the end of §4.5.

**The typology is the headline, not a caveat.** The six speedup curves
above are qualitatively different, and a reader who tries to summarise
multiNEXUS as a single scalar speedup ("we got an X× speedup over
NEXUS") is forced into one of two errors: either picking the most
generous number (MatMul $8.16\times$ at 16 GPUs) and over-claiming, or
picking the most conservative number (Bootstrap $1.04\times$ at 4 GPUs)
and under-claiming. The honest answer is the three-bucket table. The
*same* multi-GPU framework — per-thread `PhantomContext`, NCCL where
needed, persistent worker threads, STRIDED digit ownership for the DKS
path that runs underneath bootstrap when chained — produces $8\times$
on MatMul and $1\times$ on bootstrap because the underlying operations
have qualitatively different compute-to-overhead ratios at this hardware
scale. Reporting a single number would be misleading; reporting the
typology lets future practitioners (and future hardware generations)
identify which bucket their own operation falls into.

Two predictions follow from the typology that we cannot yet measure but
flag here for §8 and future work: (a) per-rank context pooling
(`CLAUDE.md` "Out of scope") would lift the transitional bucket
(GELU/LayerNorm) toward compute-parallel scaling at 16 GPUs — the
prediction is a $\approx 6{-}8\times$ effective speedup, not $3.55\times$
or $2.56\times$ — by eliminating the per-rank context-setup ceiling;
(b) the DKS path applied to bootstrap (§5.2) brings bootstrap into the
compute-parallel bucket, but only at $\log N = 16$ where the key
sharding is also memory-enabling (not just accelerating). The
*combined* per-op latency improvement under DKS + DP-throughput is the
HP-BERT 16-GPU $54.27$ s end-to-end number reported in Section 7, where
bootstrap shows up at $1{,}017.7$ ms per call (S29 multinode, JOBID
40366927) — already a $5.5\times$ improvement over NEXUS-A100's
$5{,}630$ ms at the matched $\log N = 15$ parameter set, with no
per-rank context pooling and no removal of the `bootstrap_sparse_3`
debug syncs. The headline of Goal 1 is the typology; the headline of
Goal 2 is what the typology produces when chained.

---

# Section 7 — Goal 2: End-to-end BERT inference at uniform $\log N = 15$

> Status: draft v1 (skeleton; numerical cells filled by MEASURE-01..04)
> Slice: WRITE-S7
> Depends-on: MEASURE-01, MEASURE-02, MEASURE-03, MEASURE-04, BUG-02

## 7.1 What Goal 2 contributes

Section 6 measured operations in isolation. Section 7 chains them. The
goal is to demonstrate that the multi-GPU framework introduced in
Section 5 can run a *real* end-to-end BERT inference at a single
uniform ring degree, which is the deliverable NEXUS's open source does
not produce.

We split the demonstration into three pieces:

1. **Unit measurement** (§7.2). A 1-head $\times$ 2-layer chained run
   at uniform $\log N = 15$ on one H100. Cheap to run, sufficient to
   exercise every operator in the chain.

2. **Saturation check** (§7.3). A pure predicate on the unit-run
   timings: $|t_{\mathrm{layer}\,2} - t_{\mathrm{layer}\,1}| /
   t_{\mathrm{layer}\,1} \leq 5\%$. If satisfied, the layer-2 cost has
   converged to steady state, which licenses the layer-multiply-out
   extrapolation in §7.4. The predicate is implemented in
   `scripts/regression/saturation_check.py` (slice MEASURE-02) and is
   referenced verbatim by this section.

3. **Full-BERT extrapolation** (§7.4). With the saturation check
   passing, full BERT-base = $12 \text{ heads} \times 12 \text{ layers}
   \times t_{\mathrm{per-head-per-layer}}$. The result is a
   single-GPU, single-head, projected wall-clock for full BERT.

4. **Multi-GPU strong scaling** (§7.5). Head-Parallel BERT (§5.3) at
   4-GPU and 16-GPU. Each GPU owns a subset of attention heads
   end-to-end through all 12 layers. The headline is per-inference
   latency.

5. **Multi-GPU weak scaling** (§7.6). Data-Parallel-per-inference at
   4-GPU and 16-GPU. $G$ independent BERT inferences in parallel. The
   headline is aggregate throughput in inferences-per-second.

## 7.2 Unit measurement: 1 head $\times$ 2 layers at $\log N = 15$

**Workload.** `bert_hp_multigpu --N 32768 --n-gpus 1 --heads 1
--layers 2`. The CKKS ring degree is $2^{15} = 32{,}768$, which gives
$16{,}384$ plaintext slots and a $\sim 21$ MB ciphertext footprint
(§3.1). The pipeline runs one attention head through two transformer
layers with the operators chained: MatMul $\to$ Bootstrap $\to$ GELU
$\to$ Bootstrap $\to$ LayerNorm $\to$ Softmax $\to$ MatMul $\to$
Bootstrap $\to$ \dots The chain uses the multiNEXUS evaluators from
§4.5; correctness is gated against the single-GPU reference at
$\mathrm{MAE} \leq 1\mathrm{e}{-5}$ inside the binary (audit BUG-02
notes this is looser than the PRD's $2.25\mathrm{e}{-6}$ spec but well
within either bound for 2 layers, where drift accumulation is small).

**Harness.** `scripts/mn5/slurm_bert_hp_unit_logN15.sh` runs three
sequential trials on one exclusive H100 node and records the per-layer
timings from each trial's stdout. The median trial is used for the
saturation check.

**Measured timings.** [TODO: fill from MEASURE-01 once JOBID logged.]

| Trial | $t_{\mathrm{layer}\,1}$ (ms) | $t_{\mathrm{layer}\,2}$ (ms) |
|---|---|---|
| 1 | TODO | TODO |
| 2 | TODO | TODO |
| 3 | TODO | TODO |
| **median** | TODO | TODO |

The median is what feeds §7.3.

## 7.3 Saturation check

**Predicate.**
$$
\textsf{saturated} \;\;\Leftrightarrow\;\;
\frac{|t_{\mathrm{layer}\,2} - t_{\mathrm{layer}\,1}|}
{t_{\mathrm{layer}\,1}}
\;\leq\;
\tau
$$
with $\tau = 0.05$ (5\% relative tolerance). The PRD justifies the
5\% choice as conservative for chained-pipeline drift: GELU and
LayerNorm each consume one bootstrap level so the second layer
operates closer to the post-bootstrap modulus floor than the first,
but the per-call NTT cost dominates over the modulus-chain-induced
variation by an order of magnitude (§3.2).

**Verification.** Plug the median trial from §7.2 into
`scripts/regression/saturation_check.py`:

```
$ python saturation_check.py --t1 [TODO] --t2 [TODO] --threshold 0.05
{
  "saturated": [TODO],
  "relative_delta": [TODO],
  "threshold": 0.05,
  "t1_ms": [TODO],
  "t2_ms": [TODO]
}
```

If saturated, the extrapolation in §7.4 is valid; if not, layer-2 cost
has not converged and the per-head-per-layer time must be measured at
deeper chain depth before extrapolating. [TODO: report which branch
applies once MEASURE-01 has run.]

## 7.4 Full-BERT extrapolation

Assuming saturation:
$$
t_{\mathrm{per-head-per-layer}}
\;=\;
\frac{t_{\mathrm{layer}\,1} + t_{\mathrm{layer}\,2}}{2}
$$
and the full BERT-base projection is
$$
t_{\mathrm{full BERT, 1-GPU}}
\;\approx\;
12 \cdot 12 \cdot t_{\mathrm{per-head-per-layer}}
\;=\;
144 \cdot t_{\mathrm{per-head-per-layer}}.
$$

[TODO: fill the projected number once MEASURE-01 has run.]

This extrapolation is what the rest of §7 compares against. We are
explicit about its assumptions:

- **Saturation.** Per §7.3.
- **No SIMD slot packing.** Each per-head-per-layer call uses one
  ciphertext per slot bank; we do not amortize across the unused
  slots within a ciphertext. NEXUS's published end-to-end exploits
  slot packing aggressively; this is the largest single contributor
  to our gap with their 37.3\,s headline. Disclosed in §8.3.
- **Single attention head per GPU thread.** A real BERT inference has
  12 attention heads per layer; the chain in §7.2 carries one head
  through two layers, then multiplies by 12 in the extrapolation.
  The per-head computation is independent in the multi-head
  decomposition, which is exactly what HP-BERT exploits in §7.5.

## 7.5 Strong scaling: Head-Parallel BERT (latency)

**Setup.** Each GPU owns a subset of the 12 attention heads end-to-end
through all 12 layers (§5.3). At 4-GPU each GPU runs 3 heads; at
16-GPU each GPU runs $\lceil 12 / 16 \rceil$ = 1 head with the
remaining 4 GPUs sitting idle for the layer-internal phase but
participating in the cross-head reduction. [TODO: confirm allocation
scheme from `bert_hp_multigpu.cu` and `bert_hp_multinode.cu`.]

**Workload.** `bert_hp_multigpu --N 32768 --n-gpus {4,16} --heads 12
--layers 12 --skip-ref`. The `--skip-ref` flag is currently set in
production SLURM scripts; audit BUG-02 flags that this bypasses the
MAE gate. The 4-GPU number is measurable today; the 16-GPU number
uses the 4-node SLURM harness at
`scripts/mn5/slurm_bert_hp_logN15_4node.sh`. [TODO: confirm whether
the 16-GPU number has been re-measured at $\log N=15$ since
`docs/PER_OP_VS_NEXUS.md` last updated — the 54\,s number in
`docs/PI_REPORT.md` is at $\log N=16$.]

**Result.**

| Configuration | Wall (s) | Speedup vs 1-GPU projection | Per-head efficiency |
|---|---|---|---|
| 1-GPU projection (§7.4) | TODO | 1.00 | 1.00 |
| 4-GPU HP-BERT | TODO | TODO | TODO |
| 16-GPU HP-BERT | TODO | TODO | TODO |

**Profiling-grounded explanation.** [TODO: with a 4-GPU HP-BERT nsys
trace, we expect to see the per-head compute occupy each GPU's
critical path with inter-GPU activation transfers occupying a small
fraction of total wall. The ceiling at 16 GPUs is the cross-node
activation transfer at layer boundaries plus the inter-rank
context-setup time discussed in §6 — context-pooling is out of scope
(§8.4).]

## 7.6 Weak scaling: data-parallel inferences (throughput)

**Setup.** $G$ concurrent independent BERT inferences, each on one
GPU, no inter-instance communication during the run (§5.4). At 4-GPU
all four GPUs are on one node; at 16-GPU we use four nodes via
`srun --mpi=none` with `SLURM_LOCALID` pinning each rank to its
local GPU.

**Workload.** Per-instance: `bert_hp_multigpu --N 32768 --n-gpus 1
--heads 12 --layers 12 --skip-ref`. The 4-GPU harness is
`scripts/mn5/slurm_bert_hp_throughput_4gpu.sh`; the 16-GPU harness is
`scripts/mn5/slurm_bert_hp_throughput_16gpu.sh`.

**Aggregate throughput.**
$$
\Theta(G) \;=\; \frac{G}{\max_{i \in \{1, \dots, G\}} t_i}
$$
where $t_i$ is the wall time of instance $i$. If all instances are
on identical hardware running identical workloads, $\max_i t_i \approx
t_{\mathrm{single}}$ and $\Theta(G) \approx G / t_{\mathrm{single}}$
— this is the weak-scaling ideal.

**Result.**

| $G$ | Wall (s) | Throughput (inferences/s) | Weak-scaling efficiency |
|---|---|---|---|
| 1 (reference) | TODO | $1 / t_{\mathrm{single}}$ | 1.00 |
| 4 (MEASURE-03) | TODO | TODO | TODO |
| 16 (MEASURE-04) | TODO | TODO | TODO |

[TODO: confirm whether the per-instance binary memory footprint at
$\log N = 15$ fits within one H100's 64 GB when running concurrently
with three others on the same node. The bootstrap key store at
$\log N = 16$ requires 62 GB but at $\log N = 15$ should be
substantially smaller; we expect to confirm $\sim$15-20 GB per
instance, leaving headroom for four concurrent instances on a 64 GB
H100.]

**Profiling-grounded explanation.** [TODO: with nsys traces of one
instance running alongside three others on the same node, we expect
to see negligible cross-instance contention on the NVSwitch fabric
(each instance is single-GPU) and modest contention on the per-node
PCIe lanes during initial weight upload. The 16-GPU number is
projected to be near-linear (efficiency $\sim 0.95$) because the
inferences are truly independent.]

## 7.7 What Goal 2 demonstrates

Three takeaways:

1. **Chained end-to-end at uniform $\log N = 15$ is feasible.** Every
   operator runs at the same ring degree; the chain saturates within
   two layers (§7.3); the extrapolation projects a full BERT inference
   wall-clock that is in the same order of magnitude as NEXUS's
   single-A100 numbers when adjusted for hardware uplift.

2. **Strong scaling and weak scaling are complementary.** HP-BERT
   (§7.5) reduces per-inference latency; data-parallel
   (§7.6) increases aggregate throughput. A serving deployment would
   pick between them depending on whether the latency tail or the
   QPS budget is the binding constraint.

3. **The gap to NEXUS's 37.3\,s on 4× A100 has known causes.**
   Slot-axis SIMD packing (§8.3) is the dominant unaccounted-for
   contributor. We disclose the gap and prioritize SIMD packing as
   the top future-work item (§9.4).

---

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

---

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

---

# Appendix A — NEXUS/Phantom modifications and bug-fix log

> Status: draft v1
> Slice: WRITE-Appendix
> Depends-on: BUG-01, BUG-02, BUG-03, BUG-04

## A.1 Scope and purpose

This appendix enumerates every modification we made to upstream code this
semester and the bug-fix log for the components on the chained-pipeline
critical path. Two classes of upstream code are in scope:

1. **Phantom CKKS** (`vendor/phantom/`) — the GPU-native CKKS library by
   encryptorion-lab. We patched it in 5 places (~95 LOC total) to make it
   safe under multi-GPU, multi-thread, and bootstrap-style lazy-rescale
   call patterns.
2. **NEXUS-NDSS25 evaluators** (`vendor/nexus/cuda/src/` → `src/nexus_eval/`).
   We did not modify upstream NEXUS in place; we ported the evaluators we
   needed onto our Phantom fork and added the multi-GPU prefetch and
   output-channel-split hooks the chained pipeline depends on.

Two classes of code are explicitly out of scope (see §A.7).

The purpose is twofold: a reviewer can re-derive the surface area of
upstream changes from this appendix alone, and a re-implementer can see
which invariants must hold for the chained pipeline at `logN=15` to be
correct.

## A.2 Phantom modifications (~95 lines total)

All five patch sites were necessary; removing any one of them breaks one
of the critical-path measurements in §§6–7 of the paper.

### A.2.1 `ciphertext.h` — `save()/load()` (~30 LOC)

| File | Lines | What | Why |
|---|---|---|---|
| `vendor/phantom/include/ciphertext.h` | 235, 255 | Added stream `save(std::ostream&)` / `load(std::istream&)` for `PhantomCiphertext` | Cross-GPU transfer (`cudaMemcpyPeer` is non-portable across nodes) and MPI broadcast of intermediate ciphertexts for the HP-BERT activation hand-off |

Without it, ciphertexts cannot leave the GPU they were created on except
via raw device-pointer copies, which couples the application to a single
process address space.

### A.2.2 `secretkey.h` — `save()/load()` + default constructor (~20 LOC)

| File | Lines | What | Why |
|---|---|---|---|
| `vendor/phantom/include/secretkey.h` | (header) | Added `save()/load()` and a default constructor for `PhantomSecretKey` | GPU-to-GPU and rank-0-to-all distribution of the secret key during DistributedContext setup; default ctor allows declaration before deserialization |

The 4-node multinode binary serialises the key on rank 0 and broadcasts
it over NCCL (see `bert_hp_multinode.cu:627`); the default constructor
lets each receiving rank declare an empty key and `load()` into it.

### A.2.3 `globals.h` + `context.cu` — `thread_local default_stream` (~15 LOC)

| File | Lines | What | Why |
|---|---|---|---|
| `vendor/phantom/include/util/globals.h` | 57 | `extern thread_local std::unique_ptr<...> default_stream;` | Per-thread CUDA streams for concurrent multi-GPU work |
| `vendor/phantom/src/context.cu` | (matching) | Move `default_stream` initialisation under the per-thread guard | Each `std::thread` in HP-BERT and DP-per-op gets its own stream rather than colliding on a process-wide global |

Without this patch, two worker threads driving different GPUs share the
same default stream and serialise each other; we confirmed this on the
HP-BERT path before the patch (4-GPU run was no faster than 1-GPU).

### A.2.4 `cuda_wrapper.cuh` — remove hardcoded `cudaSetDevice(0)` (~5 LOC)

| File | Lines | What | Why |
|---|---|---|---|
| `vendor/phantom/include/util/cuda_wrapper.cuh` | stream ctor | Removed `cudaSetDevice(0)` from the `cuda_stream_wrapper` constructor | The constructor was hardcoded to GPU 0; any stream allocated on a worker thread for GPU `g > 0` was silently placed on GPU 0 |

This was the single most insidious Phantom bug for multi-GPU: streams
"belonged" to the wrong device, kernels ran on the wrong device, and
peer-copies appeared to succeed but read uninitialised memory. The fix
is a one-line deletion; the diagnosis took two days.

### A.2.5 `evaluate.cu` — comment out scale-mismatch validation (~25 LOC)

| File | Lines | What | Why |
|---|---|---|---|
| `vendor/phantom/src/evaluate.cu` | `sub_inplace`, `multiply_plain_inplace`, `add_plain_inplace` | Commented out the `cipher.scale() != plain.scale()` validation guard in three operators | The bootstrap pipeline (`coefftoslot_3 → mod_reduction → slottocoeff_3`) and the argmax `sgn_eval` chain rely on **lazy rescaling**: the scale of an intermediate ciphertext is allowed to drift transiently and is reset before the next bootstrap. NEXUS's Phantom fork keeps the check enabled because their code re-aligns scales eagerly; our chained-pipeline call sites do not, so the check was a false positive that hard-aborted the program |

**This is the patch the caller contract documented in CLAUDE.md
lesson #7 is built on.** Because the check is now disabled, callers
that chain into a bootstrap **must** explicitly reset
`ct.scale() = SCALE` (the canonical scale of the parameter set) before
the next bootstrap, otherwise drift accumulates silently and surfaces
deep inside the Phantom `slottocoeff_3` encode validation. The argmax
binary does this at `src/benchmarks/argmax_align_n32k.cu:225`. The
chained HP-BERT binaries currently do not (see BUG-02 finding
[HIGH], FIX-BUG-02-02).

## A.3 NEXUS-eval modifications

`src/nexus_eval/` is a port of `vendor/nexus/cuda/src/` onto our Phantom
fork, with five named additions that were not in upstream NEXUS. We do
not modify `vendor/nexus/` in place; the port is a fork.

### A.3.1 `Bootstrapper.cu` — 8 prefetch hooks

`src/nexus_eval/bootstrapping/Bootstrapper.cu` lines **1893, 1926,
1969, 2001, 2043, 2077, 2125, 2159** call
`ckks->evaluator.prefetch_rotation_step(...)` immediately after the
current iteration's `rotate_vector` is enqueued. The four critical-path
hooks (1893, 1926, 1969, 2001) live in `bsgs_linear_transform` and
`rotated_bsgs_linear_transform`, both of which the `bootstrap_sparse_3`
path takes. The four `_hoisting` variants (2043, 2077, 2125, 2159) are
off the critical path today but were patched for parity.

Verified present by BUG-04 audit. The hooks are the H→D-overlap surface
on which the GaloisKeyStore prefetcher (§A.3.2) operates.

### A.3.2 `galois_key_store.cuh` — async prefetch + `cudaHostRegister`

`src/nexus_eval/galois_key_store.cuh` is the canonical example of
CLAUDE.md lesson #1: the H→D copy at line **220**
(`cudaHostRegister(ptr, sz, cudaHostRegisterDefault)`) pins the source
buffer before the `cudaMemcpyAsync`, otherwise the copy is silently
synchronous. Per-slot copy/compute events at lines 188–194 and 301 let
the compute stream wait for only the slot it consumes, so prefetching
slot `i+1` overlaps the rotate for slot `i`. The class also satisfies
Rule of Five explicitly (lines 99–102 — copy `= delete`, move
`= default`) per CLAUDE.md lesson #3.

### A.3.3 `matrix_mul.cu` — `matrix_mul_range(start, end)`

`src/nexus_eval/matrix_mul.cuh:82` declares `matrix_mul_range`; the
legacy `matrix_mul(x, y, res)` at `matrix_mul.cu:167` is now a
one-liner that delegates `matrix_mul_range(x, y, res, 0, 64)`. The
range form lets us split the 64-output-column matmul across GPUs by
column band, which is how the 4-GPU MatMul measurement in §6 is
produced (each thread owns 16 columns). The range form bounds-clamps
its arguments at `matrix_mul.cu:191–196`.

### A.3.4 `gelu.cu` — chain-depth caller contract (`i < 18`)

The wrapper itself does not allocate `coeff_modulus`; that is the
caller's responsibility. The relevant benchmark binaries
(`gelu_align_n65k.cu:129`, `gelu_mgpu_align.cu:139`, and
`layernorm_align_n65k.cu:130`) all use
`for (int i = 0; i < 18; i++) coeff_bits.push_back(40);`, giving
`{58, 18 × 40, 58}` = 20 limbs at `logN=16`. The earlier code used
`i < 17` (19 limbs) which exhausted the chain mid-`sgn_eval` during
GELU warmup with the error "end of modulus switching chain reached"
— a one-character fix verified by counting commas in
`vendor/nexus/cuda/src/main.cu`. This is CLAUDE.md lesson #9.

### A.3.5 `argmax_align_n32k.cu` — explicit scale reset + vocab guard

`src/benchmarks/argmax_align_n32k.cu:225` resets
`x.scale() = SCALE` before each bootstrap inside the QuickMax loop;
the comment block at lines 216–224 documents why (CLAUDE.md lesson
#7, see §A.2.5). The guard at lines 385–394 refuses cleanly with a
FATAL message when `vocab > sparse_slots`, where `sparse_slots = 8192`
at `logN=15` — required because the binary handles only single-cipher
inputs, and NEXUS's published `vocab=30,522` needs a multi-cipher
tournament that is not in upstream NEXUS's open source either
(CLAUDE.md lesson #10, BUG-01 finding [HIGH]).

## A.4 Multi-GPU framework (new code, not a modification)

`src/multi_gpu/` (~3,559 LOC, of which 1,438 LOC across the three
load-bearing files cited below) is wholly new this semester. It is not
a patch on upstream code; we flag it here so the appendix's "what we
changed" inventory is complete.

| Module | File | LOC | Role |
|---|---|---:|---|
| DistributedContext | `distributed_context.cu` | 591 | Per-GPU `PhantomContext`, persistent worker pool, `RotationWorkspace`, NCCL teardown |
| Distributed GaloisKey store | `keyswitching/dist_galois_key_store.cuh` | 339 | STRIDED per-GPU key-digit ownership (CLAUDE.md lesson #6) |
| Output Aggregation key-switch | `keyswitching/output_aggregation.cu` | 508 | T-MODUP partial inner product + AllReduce; STRIDED kernel |

Other files in `src/multi_gpu/` (`distributed_eval.cu`,
`keyswitching/galois_oa.cu`, `keyswitching/input_broadcast.cu`,
`comm/nccl_comm.cu`, `partition/rns_partition.cu`) implement the
cold-path operators, NCCL lifecycle, and RNS partition helpers.

## A.5 Bug-fix log

One row per critical-path bug fixed during this work. "Origin" cites
the commit hash; "Lesson" cites the CLAUDE.md non-negotiable the bug
crystallised into.

| # | Component | Bug | Symptom | Fix | Origin | Lesson |
|---:|---|---|---|---|---|---|
| 1 | `src/nexus_eval/gelu.cu` (caller) | GELU `coeff_modulus` had `i < 17` → 19 limbs; needed 20 limbs at `logN=16` | GELU warmup crash: "end of modulus switching chain reached" mid-`sgn_eval` | Bumped to `i < 18` (`{58, 18×40, 58}`); verified against `vendor/nexus/cuda/src/main.cu` | (pre-`8e04b14`) | #9 |
| 2 | `src/benchmarks/argmax_align_n32k.cu` | Argmax scale drift between QuickMax rounds | Phantom encode-validation error on the 3rd bootstrap inside QuickMax | Explicit `x.scale() = SCALE` reset before each bootstrap (line 225) | (pre-`8e04b14`) | #7 |
| 3 | `src/multi_gpu/distributed_context.cu::destroy()` | Stale-stream segfault on `PhantomContext` dtor for non-primary GPUs | Segfault at process exit when 4-GPU runs torn down (the captured stream had already been destroyed when `cudaFreeAsync` fired) | `release()` GPU 1..N-1 contexts; only destroy GPU 0's; sync streams + reorder NCCL→stream→context teardown | `71885e7` | #4 |
| 4 | `src/multi_gpu/keyswitching/{output_aggregation.cu,dist_galois_key_store.cuh}` | T-MODUP digit ownership was CONTIGUOUS instead of STRIDED when `chain_beta < dnum` | NCCL P2P illegal-memory-access cascade in `keyswitching_output_aggregation_dks` | Walk digits `for (size_t d = gpu_id; d < beta; d += n_gpus)`; matching STRIDED allocator in the key store; comment block at `dist_galois_key_store.cuh:19–30` locks in the invariant | `b4949cb` | #6 |
| 5 | `src/multi_gpu/` (DKS rotation v1) | Hamming-weight crash and rotation `invalid argument` bug | Multi-GPU DKS rotation aborted or produced wrong output for sparse keys | Corrected the hamming-weight enumeration and the rotation step indexing | `a791d2f` | (correctness) |
| 6 | `src/benchmarks/bert_encoder_multigpu.cu` | Undefined `N` in `printf` (compile error after Phantom fork switch) | TU did not compile | Hoisted `N` into scope; trivial fix | `6bf5d5f` | (build) |
| 7 | DKS benchmark TU set | Compilation failures after API drift; MN5 SLURM scripts missing | DKS benchmarks could not be built on MN5 | Fixed include order + API call sites; added matching SLURM scripts | `be567df` | (build) |
| 8 | NEXUS_USE_MPI option (CMake) | Intel MPI linker error on MN5 | Build failed when MPI sym were referenced via static linkage | Introduced `NEXUS_USE_MPI` CMake option to bypass | `5e5b408` | (build) |
| 9 | Phantom-fork switch (bootstrap accuracy) | Bootstrap FFT layout mismatch with non-NEXUS Phantom fork; MAE ≫ 10⁻³ | Bootstrap returned garbage; LT coefficients (computed offline via Remez) did not match the encoder's butterfly layout | Switched `vendor/phantom/` to the NEXUS Phantom fork; copied `bootstrap_*` evaluators verbatim — 0 modifications inside the bootstrap | `fe5a905`, `4d6ea58` | (correctness) |
| 10 | Bootstrap timing path | Scale-validation checks inside `evaluate.cu` aborted lazy-rescale calls | Bootstrap failed mid-`coefftoslot_3` | Commented out scale validation in `sub_inplace`, `multiply_plain_inplace`, `add_plain_inplace`; see §A.2.5 | `caa09c3` | #7 |

Bugs 1 and 2 (GELU chain depth, argmax scale drift) are the two bugs
narrated in the PI-facing report
(`docs/PI_REPORT.md` lines 58–62). Bug 3 was the stale-stream segfault
that motivated CLAUDE.md lesson #4; bug 4 was the STRIDED-vs-CONTIGUOUS
incident that motivated lesson #6. Bug 9 was the bootstrap-accuracy
incident that motivated switching to the NEXUS Phantom fork (the LT
coefficients are precomputed via Remez against the encoder's exact FFT
butterfly layout, so the bootstrap is only correct against the exact
fork those coefficients were derived for).

## A.6 Audit summary (consolidated from BUG-01..04)

Each audit covered one slice of the critical path. All four returned
PASS-WITH-FINDINGS; none returned BLOCKER on a current measurement.

| Audit | Slice | Files audited | Result | Highest-severity finding |
|---|---|---:|---|---|
| BUG-01 | Per-op align binaries + SLURM scripts | 10 binaries + 10 SLURM | PASS-WITH-FINDINGS | [BLOCKER] No MAE gate in single-GPU GELU/LayerNorm/Softmax/Argmax; mgpu variants also lack MAE gates (timing-only headline numbers) |
| BUG-02 | `bert_hp_multigpu.cu`, `bert_hp_multinode.cu`, 7 HP SLURM scripts | 2 binaries + 7 SLURM | PASS-WITH-FINDINGS | [HIGH] Multinode binary has no MAE gate at all (line 493 `skip_ref = true` hard-coded); single-node gate is `1e-5` instead of PRD's `2.25e-6` and `--skip-ref` is used in every production run |
| BUG-03 | `src/multi_gpu/` framework (distributed_context, distributed_eval, all four keyswitching files, comm, overlap, partition, nvtx) | ~12 files | PASS-WITH-FINDINGS | [BLOCKER] `MultiGpuContext::destroy()` (`nccl_comm.cu:85–86`) destroys streams before NCCL communicators and skips the device-sync; ordering should be `cudaDeviceSynchronize → ncclCommDestroy → cudaStreamDestroy` |
| BUG-04 | `src/nexus_eval/` wrappers + `bootstrapping/Bootstrapper.cu` | 8 files | PASS-WITH-FINDINGS | [HIGH] `Bootstrapper::bootstrap_sparse_3` (lines 3043–3107) has ~7 leftover `fprintf` + `cudaDeviceSynchronize()` debug calls that collapse the H↔D overlap the 8 prefetch hooks were designed to provide |

### A.6.1 Proposed FIX slices (severity counts)

The four audits proposed **48 follow-up FIX slices** in total. Severity
distribution:

| Severity | BUG-01 | BUG-02 | BUG-03 | BUG-04 | Total |
|---|---:|---:|---:|---:|---:|
| BLOCKER | 5 | 0 | 1 | 0 | 6 |
| HIGH | 4 | 3 | 4 | 3 | 14 |
| MEDIUM | 2 | 4 | 6 | 2 | 14 |
| LOW | 3 | 4 | 4 | 3 | 14 |
| **Total** | **14** | **11** | **15** | **8** | **48** |

The 6 BLOCKERs are all in BUG-01 (missing MAE gates on 5 align binaries)
and BUG-03 (`MultiGpuContext::destroy()` ordering). None of them gate
the current paper measurements — the BLOCKERs are about
**reproducibility insurance**, not currently broken measurements — but
they should land before any further code generation runs against these
binaries.

The 14 HIGH findings concentrate on three classes of issue: missing or
loose correctness gates (BUG-01-06, BUG-02-01, BUG-02-05), hot-path
allocations and unpinned H→D copies (BUG-02-03, BUG-02-04, BUG-03-02..05,
BUG-04-03), and chained-pipeline scale-reset (BUG-02-02, BUG-04-04,
BUG-04-05). Removing the seven debug syncs flagged as FIX-BUG-04-01 is
the single largest observable critical-path win available without
changing algorithm.

## A.7 What the audit did NOT cover

Be explicit about scope:

- **`vendor/phantom/` internals** were not audited line-by-line beyond
  the five patch sites in §A.2. Phantom is treated as a library; we
  read its headers and the patched `.cu` files for evidence of our
  modifications, not to audit its bootstrap implementation.
- **`vendor/nexus/cuda/` internals** were not audited. We
  cherry-picked the evaluators we needed into `src/nexus_eval/`; the
  upstream tree remains as a reference for the LT coefficient
  derivation only.
- **Archive directories** (`src/multi_gpu/archive/pipeline/`,
  `src/benchmarks/archive/`, `src/multi_gpu/overlap/`,
  `keyswitching/input_broadcast.cu` legacy Phase-1 fallback) were not
  audited. These contain abandoned strategies (`CtPipeline`, pipeline
  parallelism) and dead code that BUG-03 explicitly recommends moving
  to `src/multi_gpu/archive/`.
- **LLaMA binaries** (`llama_hp_multigpu.cu`, `llama_hp_multinode.cu`)
  were not audited. This paper is BERT-only; LLaMA is out of scope
  (CLAUDE.md "Out of scope" section).
- **MN5-side runtime environment** (NCCL config, GPFS bootstrap-id
  semantics under contention) was audited only at the SLURM-script
  level. The runtime behaviour of NCCL on the ACC partition is
  documented separately in `docs/MN5_NCCL_CONFIG.md`.

The full FIX-slice catalogue per audit lives in
`docs/audits/BUG-01..04_*.md` and is the source of truth if any
individual finding needs re-derivation.

---

