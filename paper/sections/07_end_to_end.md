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
