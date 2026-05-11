# Section 1 — Abstract

> Status: draft v1
> Slice: WRITE-S1
> Depends-on: WRITE-S2..S8 (depends on the rest of the paper for accuracy of the summary)

## Abstract

Privacy-preserving transformer inference under fully homomorphic encryption (FHE) is within wall-clock reach on a single GPU, but no published artifact runs it end-to-end across multiple GPUs on real hardware. NEXUS [CITATION_NEXUS], the state-of-the-art non-interactive CKKS protocol for BERT-base inference, ships per-operation CUDA kernels but neither chains them nor parallelizes them across GPUs; a direct audit of `vendor/nexus/cuda/` finds zero `cudaSetDevice`, NCCL, MPI, or `std::thread` calls. We close that gap with multiNEXUS, a multi-GPU framework for NEXUS-style FHE BERT inference on 4 H100 GPUs single-node and 16 H100 GPUs (four nodes) on BSC MareNostrum 5. We make two contributions. First, a *per-operation multi-GPU typology* against a NEXUS-on-H100 single-GPU baseline that we rebuild from source so every speedup ratio is hardware-isolated; our single-GPU numbers match NEXUS-on-H100 within $\pm 2\%$ on all six operations (e.g., bootstrap at $\log N = 15$: $250$ ms vs $252.8$ ms), and we show that MatMul scales to $8.16\times$ at 16 GPUs while small ops (bootstrap, softmax, argmax) become throughput-bound at $9$–$22\%$ efficiency. Second, a *chained end-to-end BERT pipeline at uniform $\log N = 15$* — which NEXUS does not ship — demonstrated via a unit measurement, an explicit saturation check, and a multiply-out extrapolation; under head-parallel scaling we report $54.27$ s on 16 H100 GPUs. Heterogeneity is the headline, not a caveat.
