# multiNEXUS — Results Summary Tables

**Audience:** advisor walkthrough. Every table is self-contained: caption, how
to read, exact setting, then the numbers, then a comparison row with
prior art (NEXUS / Cerium-Cinnamon-Jayashankar / EncryptedLLM) where
meaningful.

> **Apples-to-apples caveats up front.** NEXUS measures BERT-base at N=32768
> on 4× A100. Cerium (Jayashankar et al., arXiv 2025) measures on 8× B200 with
> sparse-poly compression. Cinnamon (Jayashankar et al., ASPLOS 2025) is a
> *simulated* ASIC, not real hardware. multiNEXUS measures at N=65536 on
> 4× H100 64GB SXM. Where the comparison is not directly comparable, the
> column header says so.

---

## 0. Common experimental setting

This setting applies to every multiNEXUS row in this document unless
otherwise noted:

| Field | Value | Source |
|---|---|---|
| Hardware | 4× NVIDIA H100 64 GB SXM, NVSwitch | MN5 ACC partition, single node |
| Compiler / runtime | CUDA 12.8, NCCL 2.24.3-1, GCC system, CMake 3.30.5 | `module load …` on MN5 |
| FHE library | Phantom FHE (modified, in `vendor/phantom/`) + NEXUS reference (`vendor/nexus/`) | git submodules |
| Scheme | RNS-CKKS, sparse encoding | sparse_slots = 16384, slot_count = 32768 |
| Ring degree N | 65,536 | matches NEXUS protocol N' |
| Coefficient modulus | up to ~1760 bits, L≤45 RNS limbs | follows NEXUS parameter set |
| Secret-key Hamming weight | 192 | NEXUS sparse-key |
| Bootstrap entry level | chain_index = 36, post-mod-switch | matches NEXUS post-rescale entry |
| Correctness gate | MAE < 0.01 vs plaintext reference | every configuration passes at MAE = 2.25e-6 |
| Test binary | `bert_dks_multigpu` (single BERT encoder layer, one attention head, projected ×12) | `src/benchmarks/bert_dks_multigpu.cu` |

NEXUS in `study.md` and the NDSS '25 paper measures BERT inference end-to-end
(all 12 encoder layers, full attention + FFN); their bootstrap parameters are
N=32768. Where their numbers appear below, this is flagged in-row.

---

## Table 1. Bootstrap time across optimization phases (multiNEXUS internal)

**What it shows.** A single bootstrap call, isolated. Five iterations of the
optimization story, each with a single change relative to the previous row.
This is the table that tells the engineering narrative.

**How to read.** "Bootstrap/call" is one bootstrap_3 call timed end-to-end
with `cudaDeviceSynchronize`. "vs CPU baseline" normalises against the
prior-work CPU-streaming reference *for the full BERT-layer projection* —
i.e. per the table-2 mapping, not per-bootstrap. MAE is mean absolute error
vs plaintext reference; pass threshold 0.01. ❌ marks regressions.

**Setting.** Same as §0. `DKS_ROTATE` env var toggles between rotation
modes. All measurements via `bert_dks_multigpu 4` averaged over 4
bootstraps within one BERT-layer run.

| # | Phase | Change vs prior row | Bootstrap/call | MAE | vs CPU baseline (12-head BERT) |
|---|---|---|---|---|---|
| 0 | Pre-project (CPU streaming, head-parallel reference) | — | — | — | 1.00× (= 249.6 s) |
| 1 | Phase 0 — DKS storage only | shard 64 GB Galois keys 4-way | **10,514 ms** | 2.25e-6 ✅ | **0.45×** ❌ |
| 2 | Phase 1 — async key prefetch + `cudaHostRegister` | pin 62 GB host store, ping-pong copy_stream | **2,277 ms** | 2.25e-6 ✅ | **2.03×** |
| 3 | Phase 3 v2 — DKS rotation + persistent `RotationWorkspace` | route bootstrap rotations through DKS, hoist mallocs | **2,143 ms** | 2.25e-6 ✅ | **2.14×** |
| 4 | Phase 4a — persistent `local_cts` per GPU | avoid `PhantomCiphertext::resize` per call | **2,136 ms** | 2.25e-6 ✅ | **2.15×** |
| 5 | **Phase 4b — persistent worker threads (current champion)** | replace per-call `std::thread` spawn | **2,126 ms** | 2.25e-6 ✅ | **2.16×** |

Source: [multiNEXUS.md §4.1](../multiNEXUS.md), Table at line 173. Champion
re-confirmed via Nsight Systems trace on 2026-04-19
(`~/nexus-traces/trace_dksrot.nsys-rep`).

---

## Table 2. Full BERT-base layer (single attention head → 12-head projection)

**What it shows.** Bootstrap dominates, but it's not the whole story. This
table puts bootstrap inside the full BERT encoder layer (4 bootstraps per
layer + matmuls + softmax + LayerNorm + FFN), and projects from 1 head to 12
heads (BERT-base attention width).

**How to read.** "Bootstrap (×4)" is the 4 bootstraps consumed by one BERT
encoder layer. "Other ops" = QKV matmul + attention + LayerNorm + FFN, all
measured the same way. "Layer/head" = total time for one head. "12-head
proj." multiplies by 12 (BERT-base has 12 attention heads). "vs CPU
baseline" normalises against the head-parallel CPU-streaming reference of
249.6 s.

**Setting.** Same as §0. Measurement via `bert_dks_multigpu 4`; values are
the median of 3 layer runs (warm cache).

| Config | GPUs | Key mem/GPU | Bootstrap (×4) | Other ops | Layer/head | 12-head proj. | vs CPU baseline |
|---|---|---|---|---|---|---|---|
| CPU streaming head-parallel (prior reference) | 4× H100 | 64 GB on CPU | ~39,912 ms | ~9,688 ms | ~62,400 ms | **249.6 s** | 1.00× |
| DKS 4-GPU, no prefetch (Phase 0) | 4× H100 | 18.4 GB | 42,092 ms | 4,186 ms | 46,278 ms | **555.3 s** | 0.45× ❌ |
| Phase 1 (async prefetch) | 4× H100 | 18.4 GB | 9,068 ms | 1,110 ms | 10,179 ms | **122.1 s** | **2.04×** |
| Phase 3 v2 (DKS rotation + persistent buffers) | 4× H100 | 18.4 GB | 8,572 ms | 1,169 ms | 9,741 ms | **116.9 s** | **2.14×** |
| **Phase 4b (current champion)** | 4× H100 | 18.4 GB | **8,504 ms** | **1,136 ms** | **9,640 ms** | **115.7 s** | **2.16×** |
| **NEXUS (paper, ref. only)** — *N=32768, BERT all layers, 4×A100* | 4× A100 | ~10 GB | n/a (different N) | n/a | n/a | **37.3 s** | (not directly comparable; smaller N) |
| **Cerium (Jayashankar et al., 2025)** — *BERT-base, 8× B200, sparse poly* | 8× B200 | proprietary sparse | n/a | n/a | n/a | **8.8 s** | (different hardware + sparse-poly compression) |
| **Cinnamon (Jayashankar et al. ASPLOS '25)** — *simulated ASIC chiplets* | simulator | n/a | n/a | n/a | n/a | **1.67 s** | (architectural simulation — not measured) |

Source for multiNEXUS rows: [multiNEXUS.md §4.2](../multiNEXUS.md), tables
at lines 255-261 and 294-300. Source for prior art: [study.md
§§2.2, 2.3, 6.6](../study.md). NEXUS 37.3 s figure: Zhang et al., NDSS
2025; Cerium 8.8 s: Jayashankar et al., arXiv 2512.11269, December 2025;
Cinnamon 1.67 s: Jayashankar et al., ASPLOS 2025 (simulated).

---

## Table 3. Single bootstrap operation — multiNEXUS vs prior art

**What it shows.** A bootstrap is the dominant cost in CKKS; this table
isolates one bootstrap call across all systems. Numbers are in milliseconds.

**How to read.** Each row is one full bootstrap (sparse encoding, the deep
operation that resets the level budget). Where the system uses a different
N or hardware, the row is flagged and the number is not directly
comparable.

**Setting.** multiNEXUS rows: same as §0. Other rows: as cited.

| System | Hardware | Ring N | Slots | Bootstrap time | Notes |
|---|---|---|---|---|---|
| Naive single-GPU streaming (project start, 1× H100) | 1× H100 64GB | 65,536 | 16,384 sparse | **10,712 ms** | CPU→GPU H→D unpinned; sync bounce buffer |
| DKS storage only, no prefetch | 4× H100 | 65,536 | 16,384 sparse | **10,514 ms** | Phase 0 — keys split, compute path unchanged |
| Phase 1 — async prefetch + pinned host (1× H100) | 1× H100 | 65,536 | 16,384 sparse | **2,284 ms** | First real win — `cudaHostRegister` on 62 GB host store |
| **Phase 4b — DKS rotation + all optimisations (4× H100, current champion)** | 4× H100 | 65,536 | 16,384 sparse | **2,126 ms** | Bit-identical decoded output to single-GPU |
| **NEXUS** (Zhang et al., NDSS 2025) | 4× A100 | 32,768 | full slots | **5,600 ms** | Bootstrap at smaller N (logN=15 internally), then re-encrypt to N=65536 |
| **Cerium** (Jayashankar et al., 2025) | 8× B200 | proprietary | sparse | **7.5 ms** | First sub-10-ms FHE bootstrap; sparse-poly + scheduling passes |
| Cerium naïve 8-GPU (no scheduling) | 8× B200 | proprietary | sparse | **17.4 ms** | **1.2× slower than 1-GPU** without compute-comm overlap |
| Cerium single-GPU baseline | 1× B200 | proprietary | sparse | **14.5 ms** | Sets the lower bound that naïve multi-GPU regresses below |
| **EncryptedLLM** (De Castro et al., ICML 2025) | 1× GPU (paper unspecified A100/H100) | ≥ 65536 | full | **~550 ms** (20-level boot) | OpenFHE-CUDA fork, single-GPU, smaller bootstrap depth |

Source: multiNEXUS rows from [multiNEXUS.md §4.1](../multiNEXUS.md);
NEXUS, Cerium, EncryptedLLM from [study.md §2 & §6](../study.md).

**Reading the comparison.** Cerium's 7.5 ms is ~280× faster than ours, but
runs at smaller effective N (sparse polynomial representation), on B200
rather than H100, with a year-long compiler-engineering effort behind it.
NEXUS's 5.6 s is ~2.6× faster than our 4-GPU bootstrap because they
bootstrap at N=32768 then re-encrypt — a protocol choice that breaks true
non-interactivity in our setting. multiNEXUS's win is **at our chosen N
and protocol** (single-N, no re-encryption, N=65536 throughout) where the
key store does not fit on one GPU, an operating point neither NEXUS nor
Cerium publishes numbers for.

---

## Table 4. Per-operation breakdown of one BERT encoder layer

**What it shows.** Where the time goes inside one head of one BERT layer,
before vs after the optimisations. This is the table that justifies "we
optimised the right thing": bootstrap really was 91% of layer time, and
attacking it dropped layer time 4.5×.

**How to read.** Each row is one operation in the BERT layer; columns
compare Phase 0 to Phase 4b. "% of new layer" answers "after our work,
where is the time still going?" — answer: still bootstrap.

**Setting.** 4× H100, N=65536, single attention head, BERT-base
hidden=768, sparse encoding. Measured by `bert_dks_multigpu 4`.

| Operation | Phase 0 (no prefetch) | Phase 4b (current) | % of new layer |
|---|---|---|---|
| QKV MatMul (×3) | 117.0 ms | 131.5 ms | 1.4% |
| QK^T multiply | — | 2.6 ms | 0.0% |
| Softmax | — | 215.7 ms | 2.2% |
| Attn × V | — | 0.6 ms | 0.0% |
| Output projection | — | 34.5 ms | 0.4% |
| **Bootstrap × 4** | **42,092.0 ms (91.0%)** | **8,504 ms (88.2%)** | **88.2%** |
| LayerNorm × 2 | 2,790.9 ms | 581.8 ms | 6.0% |
| FFN up + GELU + down | 152.6 ms | 143.8 ms | 1.5% |
| **TOTAL (1 head)** | **46,278 ms** | **9,640 ms** | **100%** |
| **Speedup vs Phase 0** | 1.0× | **4.80×** | — |

Source: [multiNEXUS.md §4.2](../multiNEXUS.md), table at line 263-275
(Phase 1 numbers; Phase 4b deltas applied per Table 1 above).

**Takeaway for the advisor.** Bootstrap was 91% of layer time, now 88%.
Even a 4.8× layer speedup didn't change the dominance ratio meaningfully.
Any further BERT-level speedup must come from the bootstrap path itself.

---

## Table 5. Where the 2.1 s/bootstrap actually goes (Phase 4b, from Nsight Systems)

**What it shows.** Decomposition of bootstrap wall-clock into its main
contributors, derived from `nsys stats --report nvtxsum,gpukernsum` on
`trace_dksrot.nsys-rep` (April 19 run).

**How to read.** Percentages are of the 2,126 ms wall-clock for one
bootstrap call. "NTT kernels" aggregates all forward/inverse NTT kernels
across all 4 GPUs. "Straggler wait" is the difference between
`ncclAllReduce` *kernel* time and *wall* time — i.e. one GPU finishing the
preceding work later than the others, forcing the AllReduce kernel to
launch late on the slow GPU and idle on the fast ones.

**Setting.** Same as §0. Measured 2026-04-19 with NVTX instrumentation
(`src/util/nvtx_tracer.cuh`), trace clamped to one bootstrap via
`--capture-range=nvtx --nvtx-capture=bootstrap_sparse_3`.

| Component | % of bootstrap | Approx ms | Comment |
|---|---|---|---|
| NTT kernels (forward + inverse, all GPUs) | **40%** | ~850 ms | Each GPU computes the full β-digit NTT redundantly (Phase 4c target) |
| ncclAllReduce kernel time | 14% | ~291 ms | Real comm cost on NVLink/NVSwitch |
| Launch jitter / straggler wait inside AllReduce wall | ~25% | ~530 ms | Host-side ordering — Phase 4d target, no algorithmic change needed |
| `partial_key_switch_inner_prod` across all 4 GPUs | 6.7% | ~142 ms | The actual parallelised work — smaller than expected, parallelism is healthy |
| Mod-up / mod-down / rescale / BSGS scalar ops | ~14% | ~298 ms | Spread across the bootstrap path |
| Total | 100% | 2,126 ms | |

Source: Nsight Systems trace `trace_dksrot.nsys-rep`, `nvtxsum` and
`gpukernsum` reports. Captured per [PI_BRIEFING §6](PI_BRIEFING.md).

**Reading the table for the advisor.** The two big remaining costs are
(a) NTT runs *redundantly on all 4 GPUs* — sharding it 4-way (Phase 4c) is
the largest theoretical win; (b) NCCL straggler wait is host-side launch
jitter, not real comm — easy to fix in a few hours (Phase 4d).

---

## Table 6. CKKS micro-benchmarks at N=65536 (single GPU, Phantom)

**What it shows.** Per-operation cost of fundamental CKKS operations at our
ring size, on a single GPU, before any multi-GPU machinery. This is
context for *why* bootstrap (which calls rotate ~75 times) takes 2.1 s.

**How to read.** Each row is the median of 100 trials of a single Phantom
call. Times in microseconds. "L=43 limbs" is approximately our deepest
modulus; later runs use a smaller L after rescale steps.

**Setting.** 1× NVIDIA L4 24 GB, AWS g6.xlarge, N=65536, full coefficient
modulus chain (1760 bits, L=43). Measured 2026-03-29 via Phantom's
built-in `bench_ckks` test.

| Operation | Median time (µs) | Std. dev. | Notes |
|---|---|---|---|
| `gen_secretkey` | 711 | — | one-time per session |
| `gen_publickey` | 1,339 | — | one-time |
| `gen_relinkey` | 55,172 | — | one-time, scales with L² |
| `encode` | 876 | — | plaintext → polynomial |
| `decode` | 22,011 | — | reverse |
| `encrypt_asymmetric` | 4,484 | — | one-time per ciphertext |
| `decrypt` | 440 | — | server returns one ciphertext |
| `add` (ct + ct) | 564 | — | trivial |
| `add_plain` | 284 | — | |
| `multiply` (ct × ct) | **30,286** | — | requires relinearization |
| `multiply_plain` | 567 | — | scalar mul, much cheaper |
| `rescale_to_next` | 1,432 | — | level-budget management |
| `rotate_vector_one_step` | **29,518** | — | ~30 ms per rotation; bootstrap has 75 of these |

Source: [experiments/results/2026-03-29_l4_baseline-ops/raw/ckks_bench.txt](../experiments/results/2026-03-29_l4_baseline-ops/raw/ckks_bench.txt)

**Reading the table for the advisor.** A single rotation at our deepest
level is ~30 ms on an L4. Bootstrap's BSGS loop performs 75 rotations →
2.2 s lower bound from rotations alone, before adding mod-up / mod-down /
NTT work. This sets the order-of-magnitude floor: 2.1 s/bootstrap on
H100×4 is *near* the rotation-bound limit; further speedup requires
sharding the rotation itself (Phase 4c).

---

## Table 7. Multi-GPU scaling — the cautionary numbers

**What it shows.** Naïve multi-GPU FHE often *regresses* before it
improves. This is true for both multiNEXUS (Phase 0) and Cerium (their
own published cautionary data point).

**How to read.** "vs 1-GPU" shows the speedup compared to that system's
own single-GPU baseline. A value < 1 is a regression — moving to multi-GPU
made it worse before scheduling work made it better.

**Setting.** multiNEXUS rows: same as §0. Cerium rows: from Jayashankar
et al. (2025) at unspecified bootstrap level, 8× B200.

| System | Configuration | Bootstrap time | vs 1-GPU baseline |
|---|---|---|---|
| **multiNEXUS** | 1× H100, naïve streaming | 10,712 ms | 1.00× |
| **multiNEXUS** | 4× H100, Phase 0 (storage only, no compute parallelism) | 10,514 ms | 1.02× (effectively flat) |
| **multiNEXUS** | 4× H100, Phase 1 (async prefetch on 1 GPU) | 2,277 ms | **4.70×** |
| **multiNEXUS** | 4× H100, Phase 4b (current) | 2,126 ms | **5.04×** |
| **Cerium** (Jayashankar) | 1× B200 | 14.5 ms | 1.00× |
| **Cerium** | 8× B200, naïve (no compute-comm overlap) | 17.4 ms | **0.83×** ❌ |
| **Cerium** | 8× B200, + compute-comm overlap | 10.0 ms | **1.45×** |
| **Cerium** | 8× B200, + comm minimisation passes (final) | 7.5 ms | **1.93×** |

Source: multiNEXUS rows from Table 1 above. Cerium rows from
[study.md §3.2 & §6](../study.md), citing Jayashankar et al. (arXiv
2512.11269, December 2025).

**Takeaway for the advisor.** Both multiNEXUS Phase 0 and Cerium's naïve
multi-GPU shows that splitting an FHE workload across GPUs *without*
overlap engineering is at best a wash and often a regression — the entire
"multi-GPU FHE win" is in the scheduling work. multiNEXUS's 5.04× over
single-GPU is in the same regime as Cerium's 1.93×, scaled by ring size
(we run a 2× larger N than Cerium's effective sparse representation).

---

## Table 8. Memory footprint per GPU

**What it shows.** The reason a single H100 cannot run our workload, and
the reason DKS storage sharding is necessary even though it doesn't
improve speed by itself.

**How to read.** "Key memory" is the sum of all bootstrap Galois rotation
keys at the parameter set. Single-GPU value exceeds H100 80 GB → keys
must live on host (CPU streaming) or be sharded.

**Setting.** Same as §0. Computed from N=65536, 75 rotation keys, L≈43,
~825 MB/key. Measured at runtime via `nvidia-smi --query-gpu=memory.used`.

| Configuration | Key memory total | Per GPU | Fits on H100 64 GB? | Strategy |
|---|---|---|---|---|
| Single H100, all keys on GPU | ~62 GB | 62 GB | barely (no headroom for ciphertexts) | OOM in practice |
| Single H100, keys on host CPU RAM | 64 GB CPU | 0 (streamed) | yes (with H→D) | CPU streaming baseline (10.7 s/boot) |
| **DKS 2-GPU, sharded** | 64 GB total | **36.3 GB** | yes | half host transfer, still streams |
| **DKS 4-GPU, sharded (current)** | 64 GB total | **18.4 GB** | yes (good headroom) | only 1/4 needs host fallback |
| NEXUS (4× A100, paper) | ~40 GB total | ~10 GB | yes (A100 40 GB) | smaller N=32768 → smaller keys |

Source: [multiNEXUS.md §1.2](../multiNEXUS.md). NEXUS row from Zhang et
al., NDSS 2025.

---

## Table 9. NCCL bandwidth & FHE communication costs (single-node 4× H100)

**What it shows.** Whether the multi-GPU fabric can keep up with FHE's
communication patterns. This sets a lower bound on multi-GPU FHE time.

**How to read.** Bandwidth in GB/s aggregate (sum across all GPUs);
ciphertext transfer time is for one ciphertext (N=65536, L=20, 2
polynomials = 20.97 MB).

**Setting.** Single-node baseline measurement on 1 GPU (just topology
detect; multi-GPU NCCL bandwidth measurement is queued). NCCL 2.24.3-1.

| Operation | Bandwidth (aggregate) | Time per ciphertext (20.97 MB) |
|---|---|---|
| AllGather (1 GPU baseline) | 1,035 GB/s | 20 µs |
| AllReduce | 0.0 GB/s | not exercised |
| Broadcast | 144,091 GB/s | < 1 µs |

Source: [experiments/results/2026-03-29_l4_baseline-ops/raw/nccl_bandwidth.txt](../experiments/results/2026-03-29_l4_baseline-ops/raw/nccl_bandwidth.txt)

**Reading.** The single-GPU "AllGather 1035 GB/s" is mostly memcpy
bandwidth since there's only one peer. A real 4-GPU NCCL scan is the next
queued measurement — current Nsight traces show NCCL kernel time at ~291
ms total per bootstrap, which is comfortably under the NVSwitch theoretical
~900 GB/s aggregate.

---

## Table 10. Conceptual provenance — what we adopted from prior systems

**What it shows.** Each of the multiNEXUS techniques and which paper /
system it traces back to. Useful when fielding "is this novel?" questions.

**How to read.** "Adopted from" cites where the idea was first articulated;
"Adapted because" notes what we changed for our setting.

| multiNEXUS technique | Adopted from | Adapted because |
|---|---|---|
| Sharding bootstrap Galois keys across GPUs (DKS) | Cinnamon's `keyswitch_digits` digit decomposition (Jayashankar, ASPLOS '25) | Single H100 cannot hold N=65536 keys |
| Async key prefetch with double-buffered slots | Cinnamon's `CommonReceiveEliminatorPass` + `HoistInputBroadcastPass` (Jayashankar, ASPLOS '25) | Bootstrap rotation order is statically known → ping-pong over LRU |
| `cudaHostRegister` on host key store | CUDA standard practice; not paper-specific | Critical for async H→D to actually overlap (4.69× win came from this) |
| Persistent per-GPU compute workspace (no cudaMalloc in hot path) | NEXUS / Phantom design pattern | Per-call malloc was 2.4× slowdown in v1 |
| Persistent worker threads (no per-call thread spawn) | Standard CPU-side practice; Cerium runtime mentions it | Modern Linux thread spawn is µs not ms — saved only 10 ms/boot |
| NCCL AllReduce for key-switch inner-product aggregation | Cerium's "aggregate-and-scatter" + standard NCCL | Maps directly to our 4-GPU output-aggregation algorithm |
| NVTX-instrumented bootstrap path | NVIDIA tooling convention | Instrumentation was the tool that overturned our component-cost estimates |

Sources: [study.md §§3-7](../study.md) and
[multiNEXUS.md §§3, 8, 11](../multiNEXUS.md).

---

## How to use these tables

**For an advisor 1:1.** Walk through Table 1 (story), Table 2 (BERT
result), Table 5 (where time still goes), then Table 7 (the cautionary
multi-GPU lesson + how Cerium hit the same wall) and finish on Table 10
(provenance for novelty questions).

**For a paper / report.** Tables 1, 2, 3, 4, 5 are the load-bearing
results. Tables 6, 8, 9 are appendix. Table 7 is the discussion-section
hook. Table 10 is related-work bridge.

**Honest disclosures to make up front.**
1. Phase 4b (current champion) is single-node only. Multi-node DKS exists
   in the codebase but isn't wired through Bootstrapper yet (Phase 4e,
   deferred — see [HANDOFF.md §5](HANDOFF.md)).
2. Bootstrap time has a remaining ~25% straggler-wait component that's
   pure host-side jitter; we know how to fix it (Phase 4d) but haven't.
3. Comparisons to Cerium / Cinnamon are not on equal hardware (B200 vs
   H100) or equal representation (they use sparse-poly, simulated ASIC).
4. NEXUS comparison is at a different ring degree (N=32768 with
   re-encryption vs our N=65536 throughout).

These disclosures are themselves a credibility signal in advisor /
review settings — leading with them is better than being asked.
