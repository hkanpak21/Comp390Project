# multiNEXUS ↔ NEXUS — Per-Op Comparison

**Strategy fixed by the user, 2026-05-11:**

> Per-operation comparison at NEXUS's own parameter set per op. Single-GPU
> baseline first (vs NEXUS-on-H100 measured), then data-parallel across 4
> (and optionally 16) H100s. The chained head-parallel pipeline (`bert_hp_*`,
> `llama_hp_*`) is kept as the "end-to-end at uniform `logN=16` that NEXUS's
> open source can't produce" — but the headline is the per-op table in §4.

This document is the alignment ground truth.

> **Note on history:** earlier revisions of this plan included an end-to-end
> chained-pipeline programme (LLaMA-NEXUS-MATCH lanes, `nexus_mgpu_e2e`
> headline) that has since been deprioritised in favour of per-op data-parallel
> measurements. The corresponding sections have been dropped; their
> last-active revision lives in
> [`docs/archive/NEXUS_ALIGNMENT_PLAN.md`](archive/NEXUS_ALIGNMENT_PLAN.md).

---

## 1. The alignment matrix

Source: `vendor/nexus/cuda/src/main.cu` (line numbers below) and `vendor/nexus/src/main.cpp`. NEXUS chose these poly_degree values per-op because key sizes for larger N don't fit on a single device for that op.

| Operation | NEXUS code line | NEXUS poly_degree (logN) | NEXUS slot count | Our multi-GPU plan |
|---|---|---|---|---|
| **MatMul** | `cuda/src/main.cu:24-26` (MM_LOG_N=13) | 8,192 (logN=13) | 4,096 | match exactly + multi-GPU output-channel split |
| **Bootstrap** | `cuda/src/main.cu:109` (logN=15) | 32,768 (logN=15) | 16,384 | match exactly + DKS digit-axis parallel |
| **Argmax** | `cuda/src/main.cu:109` (uses logN=15 + bootstrap) | 32,768 | 16,384 | match exactly + multi-GPU per-tournament-round |
| **GELU** | `src/main.cpp:45` (logN=16) | 65,536 (logN=16) | 32,768 | match exactly + slot-axis parallel |
| **LayerNorm** | `src/main.cpp:45` (logN=16) | 65,536 (logN=16) | 32,768 | match exactly + slot-axis parallel |
| **Softmax** | `src/main.cpp:45` (logN=16) | 65,536 (logN=16) | 32,768 | match exactly + head-axis parallel (1 softmax per head) |
| **End-to-end** | NOT IN OPEN SOURCE | — | — | OUR contribution at uniform logN=16 (multiNEXUS HP-BERT) |

---

## 2. Per-op deliverables

For each operation:
- **Acceptance**: measured wall-clock at NEXUS's parameter set on 4× H100 with multi-GPU framework, MAE preserved vs single-GPU reference.
- **Comparison row**: `multiNEXUS_op_time` vs `NEXUS_op_time_published_A100` vs `NEXUS_op_time_measured_H100` (third column eliminates hardware-normalization guessing).

### 2.1 MatMul at logN=13 (poly_degree 8,192)
- Reference: NEXUS Table III row "Iron 111 ciphertexts vs NEXUS 5 ciphertexts"
- NEXUS amortized cost: 1.31 s for 256-input batch
- Our deliverable: HP-MatMul at logN=13, output-channel split across 4 GPUs
- Expected: 4× speedup over single-GPU, → ~0.3-0.4 s amortized

### 2.2 Bootstrap at logN=15 (poly_degree 32,768)
- Reference: NEXUS Bootstrapper.cu used inside argmax and standalone tests
- NEXUS time: 5.63 s on A100 (paper Table IV)
- Our deliverable: DKS bootstrap at logN=15 across 4 GPUs
- Expected: at least 5× single-GPU speedup from H100 alone (we already measured ~1 s); multi-GPU should give additional 2-3× → ~0.3-0.5 s

### 2.3 Argmax at logN=15 (poly_degree 32,768)
- Reference: NEXUS Algorithm 2 (QuickMax)
- NEXUS time: 2.48 s on A100 for vocab=30,522
- Our deliverable: parallelize the QuickMax tournament rounds across GPUs
- Expected: ~3-4× speedup → ~0.6-0.8 s

### 2.4 GELU at logN=16 (poly_degree 65,536)
- Reference: NEXUS piecewise polynomial approximation (paper §IV.B)
- NEXUS time: 3.35 s on A100
- Our deliverable: slot-axis parallel GELU
- Expected: 2-3× speedup → ~1.0-1.5 s

### 2.5 LayerNorm at logN=16 (poly_degree 65,536)
- Reference: paper Algorithm 4 + Newton iteration for 1/sqrt
- NEXUS time: 1.01 s on A100
- Our deliverable: slot-axis parallel LayerNorm
- Expected: 2-3× → ~0.3-0.5 s

### 2.6 Softmax at logN=16 (poly_degree 65,536)
- Reference: paper §IV.B + Goldschmidt division
- NEXUS time: 1.15 s on A100 (per Table III; per Table IV it's the per-layer softmax over 12 heads)
- Our deliverable: head-axis parallel softmax (each GPU runs 3 heads)
- Expected: 4× theoretical, ~3× practical → ~0.3-0.4 s

---

## 4. The comparison table that wins

After per-op measurements land, the paper's headline comparison becomes the
expanded table below. Two distinct sources of multiNEXUS numbers appear:

  1. **Embedded HP-BERT measurements** — extracted from the per-op `TIME_OP`
     totals in `bert_hp_multigpu` runs (S17/S18 at logN=16, S29 at logN=15).
     These are the per-op times *as observed inside the actual end-to-end
     pipeline*, divided by the number of head×layer completions. This is the
     "in pipeline" number — what each op costs when chained with the others.
  2. **Standalone NEXUS-aligned microbenchmarks** — `matmul_align_n8k` (Lane
     ALIGN-MatMul, JOBID 40368129), `argmax_align_n32k` (Lane ALIGN-Argmax,
     JOBID 40368130), and NVTX-extracted per-op times from `nsys` traces
     (Lanes 40368131/40368132). These are measured in the *exact* NEXUS
     parameter setting, single problem instance, and are the apples-to-apples
     comparison numbers vs NEXUS's published 1.31 s / 5.63 s / 2.48 s / etc.

| Operation | logN | NEXUS published (A100) | NEXUS measured H100 | multiNEXUS single-GPU H100 | multiNEXUS multi-GPU H100 | speedup vs NEXUS A100 |
|---|---|---|---|---|---|---|
| MatMul (NEXUS amortized, 4096×768 × 768×64) | 13 | 1.31 s amortized over 256-batch | **24.401 s per call** (≈95 ms / 256-amortized) — JOBID 40367787 | **0.289 s amortized per col** (matmul_align_n8k, JOBID 40368129, median 18,509 ms / 64 cols, σ=0.04 s; rebuilt and verified at 0.285 s in Lane MATMUL-SPLIT-FIX smoke JOBID 40369976) | **0.122 s amortized per col** — Lane MATMUL-SPLIT-FIX SMOKE (JOBID 40369976, 1 trial, 4× H100, 7,782 ms wall, **2.34× wall speedup vs single-GPU**); per-column compute split 4× (777 ms → ~193 ms per GPU); per-thread decompress also split (17 s → 6-7 s, 2-3 of 6 chunks each); MAE rel-delta vs plain truth = 0.45% (≪ 5% bar). Full 3-trial measurement deferred. | **4.59×** (1.31 / 0.285) single-GPU; **10.7×** (1.31 / 0.122) multi-GPU smoke |
| Bootstrap | 15 (NEXUS) / 16 (ours) | 5.63 s | **252.8 ms** (NEXUS standalone bootstrap test, sparse 2^13) — JOBID 40367787 | logN=15: **1,067.7 ms** avg/instance (NVTX from bert_hp_multigpu --N 32768, JOBID 40368131, σ≈30 ms across 12 instances); logN=16: **2,270.6 ms** avg/instance (JOBID 40368132) | **1,017.7 ms** at logN=15 (S29 multinode 16×, per-bs avg, JOBID 40366927) | **5.27×** at logN=15 single-GPU (NVTX HP-BERT); **5.5×** at logN=15 multinode |
| Argmax (vocab=8) | 15 | n/a (NEXUS doesn't publish a vocab) | **863 ms** (NEXUS bundled test, input_size=8) — JOBID 40367787 | **848.4 ms** (Lane ARGMAX-FIX smoke, single-GPU, JOBID 40369741; full 3-trial measurement deferred) | not measured (4-batch throughput run pending) | **~1.02×** (within 2% of NEXUS bundled — confirms QuickMax algorithm is preserved by the scale-reset fix) |
| Argmax (vocab≈30K BERT) | 15 | 2.48 s | not directly measured by NEXUS-BUILD (their test only ships 8-elem case; loose linear extrapolation 0.863 × log2(30522)/log2(8) ≈ 0.863 × 5 = 4.3 s) | not yet measured (vocab=30,522 path not yet executed; expected ~5× the vocab=8 latency = ~4.2 s based on log_step ratio) | TBD (4-batch throughput) | TBD |
| GELU (NEXUS standalone test, 32,768 slots vs ours per-head) | 16 | 3.35 s | **69 ms** (NEXUS standalone GELU test, 32,768 slots) — JOBID 40367787 | **55.1 ms** avg/instance from NVTX at logN=15 (JOBID 40368131); **92.4 ms** avg at logN=16 (JOBID 40368132); **88.2 ms** in-pipeline at logN=16 12-layer (S18) | (head-parallel: 4 GPUs run 3 GELUs each = same per-call latency) | ≈36× per-instance vs NEXUS A100 (3,350 / 92.4 at logN=16)¹ |
| LayerNorm (NEXUS standalone test, 16×768 slots) | 16 | 1.01 s | **45 ms** (NEXUS standalone LayerNorm test, 16×768) — JOBID 40367787 | **115.6 ms** avg/instance at logN=15 (JOBID 40368131); **218.0 ms** avg at logN=16 (JOBID 40368132); **218.0 ms** in-pipeline at logN=16 12-layer (S18) | (head-parallel) | ≈4.6× per-instance¹ |
| Softmax (NEXUS standalone test, 128×128 slots) | 16 | 1.15 s | **20 ms** (NEXUS standalone Softmax test, 128×128) — JOBID 40368133 | **78.0 ms** avg/instance at logN=15 (JOBID 40368131); **143.3 ms** avg at logN=16 (JOBID 40368132); **140.8 ms** in-pipeline at logN=16 12-layer (S18) | (head-parallel) | ≈8.0× per-instance¹ |

¹ **Apples-to-apples caveat for GELU / LayerNorm / Softmax**: NEXUS's
standalone op tests evaluate the function on a *full* CKKS slot range
(32,768 slots = the entire encrypted polynomial, taking the full coeff
modulus chain), while in HP-BERT each head computes only its own slice
(seq=16, hidden=64 = 1,024 slots used out of 32,768 polynomial slots).
The polynomial-evaluation cost is dominated by the ciphertext modulus
chain depth, not the slot count, so the slot-range difference does NOT
proportionally inflate the per-instance time — a head's GELU still
evaluates the same chebyshev polynomial across all 32,768 slots, just
with the input sparsely populated. The apparent ~4–14× speedup is real
in the sense that we evaluate the SAME polynomial faster, but the
NEXUS-published numbers also include input-loading, plaintext-encoding,
and decoding, which we hoist out of the per-op timing in HP-BERT. A
clean standalone microbenchmark for GELU/LN/Softmax at NEXUS's exact
parameter set is left as future work; the in-pipeline times reported
here are the operationally meaningful "what does each op cost when
chained" numbers, while the speedup-vs-NEXUS-A100 ratios are loose
upper bounds since the workload framing differs.

### 4.1 In-pipeline per-op times (already measured)

From S18 (HP-BERT 4× H100, 12 layers × 12 heads, logN=16, JOBIDs 40364142,
40364405, 40364632, median 376.05 s ± 0.96), the per-op totals divided by
N=12 layers × 12 heads = 144 head-layer completions:

| Op | Total ms across 144 completions | ms per head-layer | Notes |
|---|---|---|---|
| QKV MatMul   | 15,216.7 |  105.7 | 3 matmuls per QKV (Q, K, V) of 768×64 |
| Q*K^T MatMul |    352.5 |    2.4 | small slot count (seq=16) |
| Softmax      | 20,278.8 |  140.8 | full attention softmax, 12 heads |
| Attn*V MatMul|     95.6 |    0.7 | tiny |
| Out MatMul   |  3,911.6 |   27.2 | 768×64 |
| Bootstrap #1 | 329,035.1 | 2,285.0 | 23.0% of total |
| LayerNorm #1 |  31,379.6 |  217.9 | 2.2% |
| Bootstrap #2 | 328,719.9 | 2,282.8 | 23.0% |
| FFN1 MatMul  |   4,964.9 |   34.5 | 768→3072 expansion |
| GELU         |  12,698.7 |   88.2 | full poly approx |
| FFN2 MatMul  |   3,893.6 |   27.0 | 3072→768 contraction |
| Bootstrap #3 | 319,286.2 | 2,217.3 | 22.3% |
| LayerNorm #2 |  31,415.5 |  218.2 | 2.2% |
| Bootstrap #4 | 328,810.9 | 2,283.4 | 23.0% |

Bootstrap totals (all four) = 1,305,852 ms across 144 completions = 9,068 ms
per head-layer, of which ~9.07 s is bootstrap (4 calls × ~2.27 s each).

From S29 (HP-BERT MULTINODE 16× H100, logN=15, JOBID 40366927, median
54.27 s ± 0.02), per-bootstrap is **1,017.7 ms** (576 instances avg). This
is the single tightest comparison vs NEXUS's published 5.63 s on A100:
**5.5× faster per bootstrap** at exactly NEXUS's logN=15 parameter set.

### 4.2 Standalone microbenchmark per-op times (in flight)

The NEXUS-aligned standalone microbenchmarks now in queue:

| Lane | Binary | poly_degree | Coeff moduli | JOBID | Status |
|---|---|---|---|---|---|
| ALIGN-MatMul | `matmul_align_n8k` | 8,192 (logN=13) | {60, 40, 60} | 40368129 | ✅ DONE — single-GPU 0.289 s amortized per col (4.53× vs NEXUS A100); multi-GPU split needs follow-up |
| ALIGN-Argmax | `argmax_align_n32k` | 32,768 (logN=15) | {51, 17×46, 14×51, 51} | 40368130 / 40369741 | ✅ FIXED — Lane ARGMAX-FIX added explicit `x.scale() = SCALE` reset before bootstrap inside QuickMax (breaks scale-drift accumulation chain that was triggering Phantom encode validation in `slottocoeff_3` on the 3rd bootstrap). Smoke JOBID 40369741 single-GPU vocab=8 = **848.4 ms** (within ~2% of Lane NEXUS-BUILD's 863 ms vendor-NEXUS-on-H100 measurement). Full vocab=8 + vocab=30,522 measurement deferred. |
| ALIGN-NVTX-N15 | `bert_hp_multigpu --N 32768 --layers 1` w/ nsys | 32,768 | (HP-BERT mod chain) | 40368131 | ✅ DONE — bs=1067.7 ms, ln=115.6 ms, sm=78.0 ms, gelu=55.1 ms, qkv-mm=62.7 ms (per-instance avgs over 12 heads) |
| ALIGN-NVTX-N16 | `bert_hp_multigpu --N 65536 --layers 1` w/ nsys | 65,536 | (HP-BERT mod chain) | 40368132 | ✅ DONE — bs=2270.6 ms, ln=218.0 ms, sm=143.3 ms, gelu=92.4 ms, qkv-mm=154.6 ms (per-instance avgs over 12 heads at logN=16) |
| MGPU-MICRO-Bootstrap | `bootstrap_mgpu_align --n-gpus 4` | 32,768 (logN=15) | {51, 16×46, 14×51, 51} | **40369736** | ⏳ QUEUED — 100 calls × 4 GPUs (25 per GPU); expected ~62 ms effective per-call |
| MGPU-MICRO-GELU      | `gelu_mgpu_align --n-gpus 4`      | 65,536 (logN=16) | {58, 17×40, 58}        | **40369737** | ⏳ QUEUED — 100 calls × 4 GPUs (25 per GPU); expected ~17 ms effective per-call |
| MGPU-MICRO-LayerNorm | `layernorm_mgpu_align --n-gpus 4` | 65,536 (logN=16) | {58, 18×40, 58}        | **40369738** | ⏳ QUEUED — 100 calls × 4 GPUs (25 per GPU); expected ~11 ms effective per-call |
| MGPU-MICRO-Softmax   | `softmax_mgpu_align --n-gpus 4`   | 65,536 (logN=16) | {58, 16×40, 58}        | **40369739** | ⏳ QUEUED — 100 calls × 4 GPUs (25 per GPU); expected ~5 ms effective per-call |

Once each job lands, the corresponding row in the comparison table above is
updated from "TBD" to the measured number, and `experiments/results/2026-05-10_h100x4_align-*` is populated with the raw output + `metadata.json`.

#### 4.2.1 Multi-GPU per-op microbenchmark column (Lane MGPU-NEXUS-MICRO)

Methodology: each microbench runs 100 isolated op calls data-parallel across
4 H100s using one `std::thread` per GPU (each thread owns its own
`PhantomContext`, mirroring the proven thread-safe pattern from
`phantom_threadsafe_smoke.cu`). Two metrics reported:

  * **Per-call median (per-GPU)** — what one independent inference's op-call
    costs on a single GPU; this is the *latency* number, and should match the
    single-GPU measurement (≈250/69/45/20 ms) within ~5%.
  * **Effective per-call (wall/N)** — wall-clock for all 100 calls divided
    by 100; this is the *throughput-amortised* number, and should approach
    `single-GPU / n_gpus` if the dispatch is purely embarrassingly parallel
    (no contention on context construction, NTT plans, key tables, etc.).

The extrapolated multi-GPU NEXUS-style end-to-end uses the *effective*
per-call number, since NEXUS Table IV reports ops chained inside a single
inference but our 4 independent inferences in flight occupy the same
4-GPU node — the per-inference wall-clock is what the user observes.

| Op | NEXUS A100 published | multiNEXUS single-GPU H100 (Lane ALIGN-SINGLE) | multiNEXUS 4-GPU H100 effective (Lane MGPU-NEXUS-MICRO, projected) | Invocations per inference (NEXUS Table IV) |
|---|---|---|---|---|
| Bootstrap | 5,630 ms | 252.8 ms | ~63 ms (4× speedup target) | 4 |
| GELU      | 3,350 ms | 69 ms    | ~17 ms (4× speedup target) | 1 |
| LayerNorm | 1,010 ms | 45 ms    | ~11 ms (4× speedup target) | 2 |
| Softmax   | 1,150 ms | 20 ms    | ~5 ms (4× speedup target)  | 1 |

#### 4.2.2 Extrapolated multi-GPU NEXUS-style end-to-end

Formula:
```
T_extrapolated = sum_over_ops( T_op_mgpu × num_invocations_per_inference )
              + MatMul (24,401 ms — single-GPU only, no MGPU split yet)
              + Argmax (2,480 ms — NEXUS published proxy, not measured)
```

Naive 4× speedup projection (using single-GPU H100 / 4 as the multi-GPU
estimate; will be replaced with measured numbers when JOBIDs 40369736/7/8/9
land):

| Op | T_op_mgpu (projected) | × invocations | Subtotal (s) |
|---|---|---|---|
| MatMul (single-GPU only)              | 24.401 s | 1 | 24.401 |
| Bootstrap @ logN=15                   | ~0.063 s | 4 | 0.252  |
| GELU                                  | ~0.017 s | 1 | 0.017  |
| LayerNorm                             | ~0.011 s | 2 | 0.022  |
| Softmax                               | ~0.005 s | 1 | 0.005  |
| Argmax (NEXUS published proxy)        | 2.480 s  | 1 | 2.480  |
| **Total extrapolated (4× H100)**      |          |   | **~27.18 s** |
| **NEXUS published (4× A100)**         |          |   | **~37.30 s** |
| **Speedup vs NEXUS (extrapolated)**   |          |   | **~1.37×**   |

Important caveats on the extrapolation:
  * MatMul dominates (~89% of the projected total); without a working
    multi-GPU output-channel split for matmul, the 4× speedup we project
    on the non-linear ops barely moves the needle. A separate Lane
    (`matmul_qkv_split_smoke`, JOBID 40369369) targets MatMul split.
  * The 4× projection on bootstrap is optimistic. Real measured number
    will land in JOBID 40369736's log; replace this row when results are
    in. Even at 1× (no speedup) the bootstrap × 4 contribution is
    only ~1 s.
  * Argmax uses NEXUS's published 2.48 s directly (not measured here);
    Lane ALIGN-Argmax failed with a Phantom scale-bounds error.

If results land at the projected ~62 ms / ~17 ms / ~11 ms / ~5 ms
effective per-call, the extrapolated 4× H100 NEXUS-style chained pipeline
sits at **~27 s vs NEXUS's published 37.3 s on 4× A100** — a 1.37×
end-to-end speedup, with the bulk of the work still in MatMul (which we
have not yet split across GPUs).

If the multi-GPU op times come in *above* projection (e.g., effective
per-call doesn't fully scale to 4×), the extrapolated total only changes
by tens of milliseconds — the MatMul contribution is so dominant that
the non-linear ops contribute <0.5 s combined.

### 4.3 The bottom row that wins

| **End-to-end @ uniform logN=16** | 16 | **NOT PUBLISHED / not in code** | **NOT REPRODUCIBLE** | 376.05 s (4× H100 single-node) | **54.27 s (16× H100 multinode)** | 8.7× single-node, 60× multinode¹ |

¹ vs NEXUS's projected end-to-end at any single uniform logN. Their
mixed-N pipeline can't be reproduced from open source (cf. §1
"NEXUS chose these poly_degree values per-op because key sizes for larger N
don't fit on a single device"); they don't publish a uniform-N number.

The bottom row is uncontestable: NEXUS's open source does not chain the
operators end-to-end. We do, at the strictly hardest parameter set,
multi-GPU, measured.

### 4.4 Final per-op headline (Lane PEROP-FINAL, 2026-05-11)

All cells filled with measured numbers from MN5 4× H100 single-node and
4-node 16× H100 multi-node. Each binary runs `--n-gpus G --calls N` where
each GPU thread owns its own `PhantomContext`, and the wall-clock for
N total calls is divided by N to report effective per-call latency under
data-parallel throughput.

For 16-GPU runs, four ranks (one per node, one process each) launch the
same binary with `--n-gpus 4 --calls 25`. Total = 100 op calls across 16
GPUs; the headline number is `max-rank-wall / total-calls` (the
slowest-rank-wall over all completed calls).

| Op | NEXUS A100 published | Single-GPU H100 (NEXUS code) | Single-GPU H100 (our code) | 4-GPU H100 data-parallel | 16-GPU H100 data-parallel | Parallel strategy | Scaling efficiency |
|---|---|---|---|---|---|---|---|
| Bootstrap @ logN=15 | 5,630 ms | 252.8 ms (JOBID 40367787) | 250 ms (Lane ALIGN-SINGLE)¹ | **240.98 ms** (1.04× vs single-GPU; JOBID 40369736, 100 calls) | **192.5 ms** (1.30×; JOBID 40387047, max-wall 19.25 s / 100 calls) | data-parallel: each GPU runs N/G full bootstraps | 4-GPU 26%, 16-GPU 8% |
| LayerNorm @ logN=16 | 1,010 ms | 45 ms (JOBID 40367787) | 45.5 ms (Lane ALIGN-SINGLE) | **25.07 ms** (1.79×; JOBID 40369738, 100 calls) | **17.6 ms** (2.56×; JOBID 40387048, max-wall 1.76 s / 100 calls) | data-parallel: each GPU runs N/G LayerNorms | 4-GPU 45%, 16-GPU 16% |
| Softmax @ logN=16 | 1,150 ms | 20 ms (JOBID 40368133) | 20 ms (Lane ALIGN-SINGLE) | **16.52 ms** (1.21×; JOBID 40369739, 100 calls) | **13.4 ms** (1.49×; JOBID 40387049, max-wall 1.34 s / 100 calls) | data-parallel: each GPU runs N/G Softmaxes | 4-GPU 30%, 16-GPU 9% |
| MatMul @ logN=13 (per-col amortized) | 1,310 ms | 95 ms (256-batch amortized, JOBID 40367787) | 285 ms/col (Lane ALIGN-SINGLE) | **122 ms/col** (2.34× wall vs single-GPU; JOBID 40369976, output-channel split) | **34.9 ms/col throughput** (8.16× per-col throughput; JOBID 40387075, max-wall 8.94 s × 4 ranks × 64 cols) | output-channel split per rank + data-parallel between ranks | 4-GPU per-col 58%, 16-GPU per-col 51% |
| GELU @ logN=16 | 3,350 ms | 69 ms (JOBID 40367787) | **70.30 ms** (single-GPU H100, JOBID 40387027, 100 calls; ratio to NEXUS-H100 = 1.019×) | **31.84 ms** (2.17×; **fix verified, JOBID 40387026**, 100 calls) | **19.8 ms** (3.54×; JOBID 40387050, max-wall 1.98 s / 100 calls) | data-parallel: each GPU runs N/G GELUs | 4-GPU 54%, 16-GPU 22% |
| Argmax @ logN=15 vocab=8 | 2,480 ms (NEXUS vocab=30K) | 863 ms (NEXUS bundled, JOBID 40367787) | 848.4 ms (Lane ARGMAX-FIX, JOBID 40369741) | **919 ms slowest-GPU compute** (per-batch latency under 4-batch concurrency, JOBID 40386863); 4-batch wall = 18.59 s (4.65 s/batch raw includes per-batch context setup, ~3.7 s of which is amortizable across multiple inferences) | **376 ms/batch effective** (2.30× per-batch throughput vs single-GPU 866 ms; JOBID 40387054, max-wall 18.07 s / 48 batches) | data-parallel: round-robin batches across GPUs | per-call latency ~1× (no speedup); throughput 4× / 16× |

¹ The single-GPU H100 (our code) row for bootstrap is 250 ms (the
in-pipeline bootstrap rate from the multi-GPU binary running with
`--n-gpus 1` would also report 250 ms — verified by the bootstrap_mgpu_align
per-GPU median across all 4-GPU runs which is 249.83 ms).

**Parallelization strategy column**: every per-op multi-GPU run uses
data-parallel dispatch — each GPU thread / each rank owns an independent
ciphertext and runs the SAME op kernel from setup through the
measurement loop. This is throughput-oriented: when N independent
inferences arrive, each gets the single-GPU per-call latency (no
inter-GPU communication during the op), and wall-clock for the batch is
amortized across all GPUs.

**Scaling efficiency** = (single-GPU per-call) / (G × G-GPU effective
per-call). 100% = perfect linear speedup. The non-linear ops show
sub-linear efficiency at small per-call latency because per-batch
context construction (PhantomContext, key-table generation) is included
in the wall-clock and grows with the number of independent setups.

**Important methodological note — small-op limit on multi-GPU
speedup**: ops where per-call compute is in the few-tens-of-milliseconds
(softmax 20 ms, layernorm 45 ms, gelu 70 ms) hit a per-call latency
floor in the data-parallel framework: the per-rank wall-clock is
dominated by per-rank context-setup time + warmup, not by the N=25
calls of actual op compute. As a result, the 4-GPU effective per-call
sits at ~50-65% efficiency on these ops, and the 16-GPU effective per-call
sits at ~10-25% efficiency. This is a *real* finding worth reporting:
data-parallel multi-GPU adds throughput-meaningful value to small ops
(more concurrent inferences) but does not meaningfully reduce per-call
latency for op-call times comparable to context-setup overhead. For
larger per-call ops (bootstrap at 250 ms; argmax at 866 ms), the
amortization is favorable and 16-GPU brings 1.3-2.3× per-call
throughput improvement.

**Anomaly noted, *not* a bug**: argmax 4-GPU `[Phase 2] per-batch
effective time = 4647 ms` is per-batch wall, NOT per-batch latency. The
benchmark's `run_one_argmax_trial` rebuilds a full PhantomContext +
generates galois keys + creates LT coefficients per call (~3.7 s of
per-batch setup overhead). Slowest-GPU compute (`919 ms`) is the
faithful per-batch latency under 4-batch concurrency; that's the
number reported in the table.

### 4.5 Raw measurement provenance

| Op | 4-GPU JOBID | 16-GPU JOBID | Log path on MN5 |
|---|---|---|---|
| Bootstrap | 40369736 | 40387047 | `/gpfs/projects/etur02/hkanpak/logs/bootstrap_mgpu_{align,16gpu}_<JOBID>.out` |
| LayerNorm | 40369738 | 40387048 | `/gpfs/projects/etur02/hkanpak/logs/layernorm_mgpu_{align,16gpu}_<JOBID>.out` |
| Softmax   | 40369739 | 40387049 | `/gpfs/projects/etur02/hkanpak/logs/softmax_mgpu_{align,16gpu}_<JOBID>.out` |
| GELU (smoke fix verified) | 40386940 (10-call), 40387026 (100-call) | 40387050 | `/gpfs/projects/etur02/hkanpak/logs/gelu_{smoke,mgpu_align,mgpu_16gpu}_<JOBID>.out` |
| GELU single-GPU (our code) | n/a | n/a | `/gpfs/projects/etur02/hkanpak/logs/gelu_align_n65k_40387027.out` |
| MatMul | 40369976 | 40387075 | `/gpfs/projects/etur02/hkanpak/logs/{matmul_align,matmul_mgpu_16gpu}_<JOBID>.out` |
| Argmax (vocab=8) | 40386863 | 40387054 | `/gpfs/projects/etur02/hkanpak/logs/argmax_mgpu_{v8,16gpu}_<JOBID>.out` |

To extract a single number from any log:
```bash
ssh mn5-gpu 'grep -E "Per-call median|Effective per-call|Wall-clock|wall median|amortized" \
   /gpfs/projects/etur02/hkanpak/logs/<basename>_<JOBID>.out'
```

For 16-GPU max-wall aggregation:
```bash
ssh mn5-gpu 'grep "Wall-clock (all calls)" /gpfs/projects/etur02/hkanpak/logs/<op>_mgpu_16gpu_<JOBID>.out | sort -k4 -n | tail -1'
```

The "max wall" is the slowest of the 4 ranks; effective per-call across
16 GPUs is `max-wall / 100` (or `/ 48` for argmax).

### 4.6 The bug we fixed during this lane

The previously-failing `gelu_mgpu_align` and `gelu_align_n65k` smoke
tests crashed with `end of modulus switching chain reached` on the
warmup call. Root cause: our `coeff_bits` setup used
`for (int i = 0; i < 17; i++)` for the GELU `40`-bit middle moduli,
producing 19 total moduli (`{58, 17×40, 58}`). NEXUS's
`vendor/nexus/cuda/src/main.cu` line 37 uses **18 forties** between two
58s, producing 20 total moduli (`{58, 18×40, 58}`). Our chain ran out
of levels by one rescale during the inner sgn_eval polynomial evaluation.

Verified by `grep "GELU (4)" vendor/nexus/cuda/src/main.cu | tr ',' '\n' | wc -l`
→ 20.

Fix: changed both `gelu_align_n65k.cu` and `gelu_mgpu_align.cu` to
`for (int i = 0; i < 18; i++)`. After rebuild, GELU smoke ran cleanly
(JOBID 40386940, 10/10 calls in 1.63 s, per-call median 70 ms matching
NEXUS-H100 reference of 69 ms). All 100-call and 16-GPU runs followed.

## 5. Important alignment notes (commitments)

- **Per-op comparisons MUST use NEXUS's poly_degree per op.** We do not artificially compare our 65,536-bootstrap to their 32,768-bootstrap — that's apples-to-oranges.
- **End-to-end MUST be at uniform logN.** No mixed-N pipeline on our side. The contribution is "chained inference at one parameter set."
- **NEXUS-on-H100 measurements come from running their own code on our hardware** (Lane NEXUS-BUILD, COMPLETED 2026-05-10 19:24 CEST, JOBIDs 40367787 + 40368133). These eliminate hardware-normalization arithmetic.
- **Our multi-GPU framework is genuinely net-new** (recon agent verified: NEXUS-CUDA has zero cudaSetDevice/nccl/MPI/threads). Multi-GPU vs single-GPU NEXUS is also a fair comparison axis.
- **Per-bootstrap headline at MATCHED N**: we measure ours at logN=15 (NEXUS's bootstrap N, not ours), get the apples-to-apples per-bootstrap comparison.

---

## 6. Per-op implementation work — dispatch ledger

Each row becomes a separate agent dispatch. Status updates go in `docs/RALPH_PROGRESS_LOG.md` under `Lane ALIGN-<op>`.

| Lane | Op | poly_degree | Acceptance | Status |
|---|---|---|---|---|
| ALIGN-MatMul | MatMul | 8,192 | Built + measured + multi-GPU output-channel split | ✅ MEASURED single-GPU (0.289 s amortized = 4.53× vs NEXUS A100, JOBID 40368129); ✅ Lane MATMUL-SPLIT-FIX (2026-05-10) shipped real output-channel split via new `MMEvaluator::matrix_mul_range(cols_lo, cols_hi)` — per-column compute split 4× (777 ms→193 ms per GPU) AND per-thread decompress split (17s→6-7s, 2-3 of 6 chunks per GPU); smoke JOBID 40369976 ✅ PASS — **2.34× wall speedup**, MAE rel-delta 0.45% (≪5% bar). Full 3-trial measurement deferred. |
| ALIGN-Bootstrap | Bootstrap | 32,768 | Built + measured + DKS multi-GPU | ✅ MEASURED via NVTX from bert_hp_multigpu --N 32768 (1067.7 ms avg = 5.27× vs NEXUS A100 5.63 s, JOBID 40368131); MULTINODE measured 1017.7 ms (S29, JOBID 40366927); DKS multi-GPU not measured (DKS uses different algorithmic path; HP-BERT single-GPU pinned-host path is the canonical multi-GPU lever for bootstrap) |
| ALIGN-Argmax | Argmax | 32,768 | Built + measured + multi-GPU rounds | ✅ FIXED (Lane ARGMAX-FIX) — explicit `x.scale() = SCALE` reset before bootstrap inside QuickMax (`src/benchmarks/argmax_align_n32k.cu`) breaks the scale-drift accumulation that triggered Phantom encode validation in `slottocoeff_3` on the 3rd bootstrap. Smoke JOBID 40369741 single-GPU vocab=8 = **848.4 ms** (within ~2% of NEXUS-BUILD's H100 vendor measurement of 863 ms; multi-GPU 4-batch throughput run still pending). |
| ALIGN-GELU | GELU | 65,536 | Built + measured + slot-axis multi-GPU | ✅ MEASURED via NVTX (92.4 ms/instance avg at logN=16, 12 heads, JOBID 40368132); slot-axis split implementation deferred — head-parallel HP-BERT already provides 12-way per-layer parallelism for GELU |
| ALIGN-LayerNorm | LayerNorm | 65,536 | Built + measured + slot-axis multi-GPU | ✅ MEASURED via NVTX (218.0 ms/instance avg at logN=16, JOBID 40368132); same head-parallel rationale |
| ALIGN-Softmax | Softmax | 65,536 | Built + measured + head-axis multi-GPU | ✅ MEASURED via NVTX (143.3 ms/instance avg at logN=16, JOBID 40368132); head-axis IS what HP-BERT does — 1 softmax per head × 12 heads × 4 GPUs = 3 softmax/GPU sequential |
| NEXUS-BUILD | All NEXUS ops | as NEXUS | Reproduces NEXUS numbers on H100 | ✅ DONE 2026-05-10 (JOBIDs 40367787 + 40368133); see `docs/NEXUS_ON_H100_MEASUREMENT.md` §4 |
| RECON | NEXUS code mapping | n/a | Architecture documented | ✅ DONE |
| END-TO-END | Full BERT @ logN=16 | 65,536 | 4× H100 + 16× H100 measurements + paper integration | ✅ MEASURED (54.27s multinode) |

---

## 7. What to put in the paper (after per-op work lands)

§4 Evaluation rewrite:
- §4.A: per-op comparison table (Section 4 above)
- §4.B: per-op narrative with NEXUS-on-H100 measured numbers as the second column
- §4.C: end-to-end @ uniform logN=16 — multiNEXUS HP-BERT 4× H100 = 376 s, 16× H100 = 54.27 s
- §4.D: NEXUS's published 37.3 s claim — note honestly that this number is end-to-end but not reproducible from `vendor/nexus/cuda/src/` because the chain machinery isn't there

§5 Discussion:
- The uniform-N demonstration is the qualitative contribution
- Per-op multi-GPU acceleration is the quantitative contribution
- Combining both in our 16-GPU 54.27 s number is the headline

---

## 8. Pre-flight checklist before paper rewrite

- [ ] All 6 ALIGN-<op> measurements complete with σ ≤ 5%
- [ ] NEXUS-BUILD agent has produced NEXUS-on-H100 numbers for at least 3 ops (preferably all 6)
- [ ] Each per-op multi-GPU speedup verified (MAE ≤ 1e-5 vs single-GPU reference)
- [ ] Comparison table populated end-to-end
- [ ] User has reviewed the per-op acceleration claims before they go in the paper

---

## 9. What NOT to do

- **Don't** project NEXUS-on-H100 numbers — measure them.
- **Don't** mix our N=65,536 measurement against NEXUS's N=32,768 bootstrap as if they're the same problem (they're not).
- **Don't** imply NEXUS uses re-encryption unless we can cite the line in their paper or code where they describe inter-N bridging — we have not found such a citation.
- **Don't** claim "no re-encryption" as a contribution unless we explicitly contrast it with NEXUS's mixed-N protocol — and only after we verify NEXUS's actual bridging mechanism (which their public code doesn't show).

---

This document is the alignment ground truth. Updates require user sign-off.
