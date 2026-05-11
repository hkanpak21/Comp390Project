# multiNEXUS Benchmark Suite

Catalogue of active `.cu` files in `src/benchmarks/`. Use this when picking
which binary to run, when triaging "what does this test cover," or when
deciding whether a stale benchmark can be archived.

## Status legend

- **HEADLINE** — produces a number that lands in the per-op-vs-NEXUS
  comparison table (`docs/PER_OP_VS_NEXUS.md`). Touch with care; re-run on
  every relevant change.
- **ACTIVE** — currently exercised by SLURM scripts under `scripts/mn5/`,
  by the regression harness, or by an in-flight measurement lane.
- **EXPERIMENTAL** — staged for an upcoming measurement; not required for
  the headline numbers; may fail.

Anything previously marked `DEPRECATED` has been moved to
[`archive/`](archive/). Add a row here when you bring something back; do
not promote an archived file silently.

## Quick map

| Want to run … | Use binary |
|---|---|
| Bootstrap microbench at NEXUS logN=15 | `bootstrap_align_n32k` (single-GPU) / `bootstrap_mgpu_align` (data-parallel) |
| GELU microbench at NEXUS logN=16 | `gelu_align_n65k` / `gelu_mgpu_align` |
| LayerNorm microbench at NEXUS logN=16 | `layernorm_align_n65k` / `layernorm_mgpu_align` |
| Softmax microbench at NEXUS logN=16 | `softmax_align_n65k` / `softmax_mgpu_align` |
| MatMul microbench at NEXUS logN=13 | `matmul_align_n8k` |
| Argmax microbench at NEXUS logN=15 | `argmax_align_n32k` |
| Bootstrap kernel-identity (NEXUS workload) | `bootstrap_diagnose` |
| HP-BERT layer w/ DKS bootstrap (logN=16) | `bert_hp_multigpu` (single node) / `bert_hp_multinode` (multi-node) |
| HP-LLaMA decoder layer (logN=16) | `llama_hp_multigpu` (single node) / `llama_hp_multinode` (multi-node) |
| NCCL bandwidth sanity check on a new node | `nccl_bandwidth_test` |
| Phantom thread-safety smoke (HP precondition) | `phantom_threadsafe_smoke` |

---

## Per-op alignment microbenchmarks (the headline table)

Each pair below measures the same NEXUS operation in two configurations:
single-GPU baseline, then data-parallel across 4 H100s. Wall-time goes into
`docs/PER_OP_VS_NEXUS.md` Section 4.

| File | Status | Description |
|---|---|---|
| `bootstrap_align_n32k.cu` | HEADLINE | Single-GPU bootstrap at NEXUS logN=15 (sparse 2^13 slots). Reference number: 250 ms / call vs NEXUS 252.8 ms vendor-on-H100. |
| `bootstrap_mgpu_align.cu` | HEADLINE | Data-parallel bootstrap (N/G calls per GPU). Driven by `slurm_bootstrap_mgpu_align.sh`. |
| `gelu_align_n65k.cu` | HEADLINE | Single-GPU GELU at NEXUS logN=16 (32,768 slots). |
| `gelu_mgpu_align.cu` | HEADLINE | Data-parallel GELU. **Mod-chain-bug fixed 2026-05-10**: re-encrypts a fresh ciphertext per loop iteration (warmup previously depleted base modulus). Driven by `slurm_gelu_mgpu_align.sh`. |
| `layernorm_align_n65k.cu` | HEADLINE | Single-GPU LayerNorm at NEXUS logN=16. |
| `layernorm_mgpu_align.cu` | HEADLINE | Data-parallel LayerNorm. |
| `softmax_align_n65k.cu` | HEADLINE | Single-GPU Softmax at NEXUS logN=16 (128×128 slots). |
| `softmax_mgpu_align.cu` | HEADLINE | Data-parallel Softmax. |
| `matmul_align_n8k.cu` | HEADLINE | Single-GPU MatMul at NEXUS logN=13 (4096×768 × 768×64). Output-channel split via `matrix_mul_range` for the 4-GPU column. |
| `argmax_align_n32k.cu` | HEADLINE | Single-GPU Argmax at NEXUS logN=15. **Phantom-scale-drift fixed 2026-05-10**: explicit `x.scale() = SCALE` reset before bootstrap inside QuickMax. |
| `bootstrap_diagnose.cu` | HEADLINE | Identity check — runs our bootstrap at NEXUS's exact workload to prove kernel-byte-identical (247 ms vs NEXUS 252 ms = 2% drift). |
| `bootstrap_align_pipeline.cu` | EXPERIMENTAL | Pipelined bootstrap variant; not currently in the headline table. |

## HP-BERT / HP-LLaMA (chained pipelines)

Used for in-pipeline per-op cost extraction (NVTX) and as the reference
implementation of the head-parallel framework. Not the per-op-vs-NEXUS
headline binaries.

| File | Status | Description |
|---|---|---|
| `bert_hp_multigpu.cu` | ACTIVE | Single-node head-parallel BERT layer; one head per GPU thread, full DKS bootstrap. Driven by `slurm_bert_hp_*.sh`. |
| `bert_hp_multinode.cu` | ACTIVE | Multi-node HP-BERT (16× H100). Driven by `slurm_bert_hp_logN15_4node.sh`, `slurm_bert_hp_n32768_4node.sh`. |
| `llama_hp_multigpu.cu` | ACTIVE | Head-parallel LLaMA decoder layer (32 heads × 4 GPUs). |
| `llama_hp_multinode.cu` | ACTIVE | Multi-node HP-LLaMA. |
| `llama_layer_baseline.cu` | ACTIVE | Single-GPU LLaMA decoder baseline. |

## DKS BERT / LLaMA (older champion path, kept as reference)

The Phase 4b path that produced the original 2.16×-vs-CPU-baseline
narrative. Superseded as the headline by per-op alignment + HP, but the
binaries still build and serve as DKS reference.

| File | Status | Description |
|---|---|---|
| `bert_dks_multigpu.cu` | ACTIVE | DKS bootstrap + persistent worker threads + persistent RotationWorkspace + async key prefetch. Single-GPU + 4-GPU paths. |
| `bert_dks_multinode.cu` | EXPERIMENTAL | Multi-node DKS BERT encoder. |
| `llama_dks_multigpu.cu` | ACTIVE | DKS variant for the LLaMA decoder layer. |

## Bootstrap / key-switching primitives

| File | Status | Description |
|---|---|---|
| `bootstrap_test.cu` | ACTIVE | Single-GPU bootstrap correctness gate. |
| `bootstrap_test_n65536.cu` | ACTIVE | Single-GPU N=65,536 bootstrap correctness test with memory optimisation. |
| `bootstrap_distkey_test.cu` | ACTIVE | Multi-GPU key-distribution test. |
| `bootstrap_n65536_streaming.cu` | ACTIVE | Single-GPU N=65,536 bootstrap with CPU-side Galois key streaming. The naïve baseline; also exercises Phase 1 cudaHostRegister in isolation. |
| `bootstrapping_bench.cu` | ACTIVE | Isolated bootstrap latency benchmark (mean / min / max / stddev over N iterations). |
| `dist_bootstrap_bench.cu` | ACTIVE | DKS bootstrap rotation-correctness + per-rotation timing sweep. Driven by `slurm_dks_bootstrap.sh`. |
| `rotate_stream_test.cu` | ACTIVE | Minimal repro: N=65,536, key streaming, one rotation. Smallest binary that exercises the prefetch pipeline. |

## Multi-GPU key-switching

| File | Status | Description |
|---|---|---|
| `multi_gpu_keyswitch_test.cu` | ACTIVE | End-to-end correctness validation for Input Broadcast and Output Aggregation. Smoke gate for changes to `src/multi_gpu/keyswitching/`. |
| `dist_bert_layer_bench.cu` | ACTIVE | Single BERT-layer benchmark using distributed FHE. |
| `ks_breakdown_bench.cu` | ACTIVE | Per-stage key-switching micro-benchmark (RNS decomposition, inner product, basis extension). |
| `spmd_keyswitch_bench.cu` | ACTIVE | True SPMD multi-GPU key-switching with persistent threads. |
| `spmd_oa_bench.cu` | ACTIVE | SPMD Output Aggregation benchmark; measures real compute speedup of OA vs Input Broadcast on the same input. |

## NCCL / bandwidth

| File | Status | Description |
|---|---|---|
| `nccl_bandwidth_test.cu` | ACTIVE | Validates NCCL setup, peer access, NVSwitch AllGather / AllReduce / Broadcast bandwidth for ciphertext-sized payloads. Run first on every new node configuration. |

## End-to-end (NEXUS-style chained, multi-GPU)

| File | Status | Description |
|---|---|---|
| `nexus_mgpu_e2e.cu` | EXPERIMENTAL | NEXUS-style chained BERT pipeline distributed across multiple GPUs. Stand-up reference for the e2e measurement; the per-op alignment microbenchmarks are the headline path. |

## BERT ops correctness / smoke

| File | Status | Description |
|---|---|---|
| `bert_ops_test.cu` | ACTIVE | Per-op correctness harness (GELU, Softmax, LayerNorm) on random inputs. Useful for regression after Phantom or `nexus_eval` changes. |
| `phantom_threadsafe_smoke.cu` | ACTIVE | Spawns 2 std::threads on 2 GPUs running simultaneous bootstraps; asserts cross-thread MAE matches single-GPU reference. Gates the HP-BERT track. |
| `matmul_split_smoke.cu` | ACTIVE | 2-GPU plain×ciphertext matmul output-channel split smoke. CKKS N=8,192, L=4. |
| `matmul_qkv_split_smoke.cu` | EXPERIMENTAL | 4-GPU split QKV matmul. Partial PASS; multi-output weight-encode interaction under investigation. |

## Lingering N=65536 streaming-path benchmarks

These predate DKS but still build and run; useful for streaming-baseline
context.

| File | Status | Description |
|---|---|---|
| `bert_encoder_multigpu_n65536.cu` | ACTIVE | Multi-GPU N=65,536 BERT encoder, CPU streaming path. |
| `bert_encoder_multinode_n65536.cu` | ACTIVE | Multi-node version of the above. |
| `llama_layer_multigpu_n65536.cu` | ACTIVE | LLaMA-style decoder layer at N=65,536, CPU streaming path. |
| `llama_layer_multinode_n65536.cu` | ACTIVE | Multi-node version of the above. |

---

## Archive

`archive/` holds binaries that have been superseded or that targeted
abandoned strategies (pipeline-parallel, NEXUS re-encryption framing,
early MatMul prototypes). They build standalone if pulled back into
`src/benchmarks/` and re-added to `CMakeLists.txt`, but they are
intentionally excluded from the build glob. Keep them for provenance;
do not extend.

## Tidying policy

Move a benchmark to `archive/` (rather than delete) when:

1. No active SLURM script under `scripts/mn5/` references it.
2. Its successor binary covers all behaviours it tested.
3. The numbers it produced are preserved in `experiments/results/` or
   `docs/PER_OP_VS_NEXUS.md`.

After moving, drop its name from `CMakeLists.txt`'s benchmark `foreach`.
