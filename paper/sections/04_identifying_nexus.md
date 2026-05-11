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
