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
