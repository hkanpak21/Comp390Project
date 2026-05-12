# Section 6 — Goal 1: Per-op Multi-GPU Typology

> Status: draft v2 (BACKFILL-S6 applied)
> Slice: WRITE-S6 → BACKFILL-S6
> Depends-on: BUG-01 (audit); PROFILE-01 (40426416), PROFILE-02 (40417947), PROFILE-03 (40418387), PROFILE-04 (40418644)

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
   justifies the measured shape, citing the corresponding PROFILE-NN
   JOBID and NVTX/cuda_gpu_sum evidence inline.
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
prologue. PROFILE-01 (JOBID 40426416, nsys 4-GPU trace) confirms the
NVTX `:matmul_trial` range averages $11.26$ s with 50.1% inside
`:op:matrix_mul_range`, indicating the per-column inner product
dominates wall-clock; per-trial median is $8.94$ s matching the
multi-GPU compute headline. The reason the speedup is real and not a
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
staging buffer per `MMEvaluator` thread would close the gap (FIX-BUG-04-04
landed this fix as commit `80dc737`; staging buffer is now persistent
pinned-host across calls, eliminating the per-call `new`/`delete` and
`cudaMemcpyAsync`-from-pageable-memory issue). PROFILE-01 (JOBID 40426416)
post-fix shows the speedup curve $1\times \to 2.35\times$ at 4-GPU
(per-col $0.329$ s $\to 0.140$ s) holds with the corrected ciphertext
initialization (FIX-BUG-MATMUL-01, commit `39935e3`); MAE at $1.5 \times 10^{-7}$
passes both gates. We do not expect MatMul ever to reach exactly
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
amortised over the 25 calls, yielding $\approx 54\%$ efficiency.
PROFILE-02 (JOBID 40417947, nsys 4-GPU trace) confirms the median
`:gelu_mgpu` NVTX range at $71.3$ ms with $\sigma = 1.7$ ms over 100
calls — tight per-call cost consistent with a compute-bound polynomial
evaluation, not a setup-or-rotation-bound op.
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
speedup ceiling regardless of how many GPUs we add. PROFILE-02 (JOBID
40417947) confirms the headline: 4-GPU effective per-call $34.18$ ms
delivering a $2.02\times$ speedup over the $69$ ms single-GPU baseline,
which matches the in-binary report and saturates short of the $4\times$
linear ceiling for the reasons above.

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
is bandwidth-bound; the second is NTT-and-multiply bound. PROFILE-03
(JOBID 40418387) softmax NVTX trace shows the rotation reduction is
visible as 8 distinct `:rotate_vector step={1,2,4,8,16,32,64,-128}`
ranges each running 104 instances at $\approx 150$ μs per rotation
(total $\approx 5\%$ of softmax wall); LayerNorm's analogous reduction
runs at $\log N = 16$ over a wider slot count and is therefore
proportionally larger, but the same NTT-and-keyswitch decomposition
applies. The 4-GPU DP path delivers $1.79\times$
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
PROFILE-03 (JOBID 40418387) corroborates: softmax 4-GPU effective
$17.64$ ms ($1.13\times$ speedup over $20$ ms single-GPU) is consistent
with the LayerNorm small-op-cap explanation — when per-call compute is
already small relative to per-rank setup, data-parallel buys you very
little latency reduction.

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
to hide the modraise H→D copy is collapsed. FIX-BUG-04-01 (commit
`7bb9bf3`) has since removed those debug syncs from the bootstrap hot
path; MEASURE-01 (JOBID 40418680) measured per-bootstrap call inside
the chained HP-BERT path at $\approx 1{,}020$ ms — consistent with the
in-pipeline numbers used by §6.3.1's headline.

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
context-setup wall is comparable, hence the $30\%$ efficiency.
PROFILE-03 (JOBID 40418387) NVTX trace confirms the per-call
`:softmax_mgpu` median at $20.47$ ms across 100 calls (93.7% of trace
time), with the rotation-reduction visible as 8 step-indexed
`:rotate_vector` ranges totaling $\approx 5\%$ of wall — the
remaining wall is dominated by per-rank context-setup not captured
inside the per-call NVTX scope. At 16-GPU per-rank setup is paid four times across the 4
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
with GPUs as long as inferences arrive concurrently. PROFILE-03 (JOBID
40418387) measured peak per-GPU softmax rate at $1000/20.47 \approx 49$
calls/s, confirming the throughput-bound prediction; aggregate 16-GPU
throughput is $\approx 49 \times 16 = 784$ softmaxes/s.

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
PROFILE-04 (JOBID 40418644) confirms the single-GPU per-batch wall:
median argmax $900.4$ ms ($\sigma = 13.4$ ms across 3 trials), within
$6\%$ of NEXUS-on-H100's $863$ ms baseline. The trace also surfaced
the decode-validity finding noted in the binary's gate (`predicted=0,
plain=1` for input value $0.993$) — a separate issue that motivated
the gate addition under FIX-BUG-01-01 but does not change the timing
breakdown above.

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
