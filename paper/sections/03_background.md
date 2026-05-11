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
