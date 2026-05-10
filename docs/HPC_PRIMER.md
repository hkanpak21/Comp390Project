# multiNEXUS HPC Primer

A teaching companion to the paper. Each section pairs a concept with how it
shows up in our codebase and the pitfall that motivated the work. Written as
a peer-to-peer reference, not a textbook — assume CUDA literacy.

---

## Section 1. Why `cudaMemcpyAsync` from pageable memory is silently synchronous

### The hardware picture

CPU and GPU have separate physical memory subsystems connected by PCIe (or
NVLink on H100 SXM). The unit that drives transfers between them is the
**DMA engine** — a dedicated copy unit on the GPU that reads source bytes
and writes them to a destination address without involving CPU cores. This
is what makes `cudaMemcpyAsync` "async" in principle: the CPU thread fires
the request and returns, and the DMA engine moves bytes in the background.

The DMA engine operates on **physical addresses**, not virtual ones. It has
no view of the OS page table; it just talks to the **IOMMU**, the hardware
unit that translates DMA-side addresses for memory-mapped I/O.

### Pageable memory and why it breaks DMA

A standard `malloc` returns a **pageable** allocation. The virtual address
is stable but the underlying physical pages are managed by the OS:

- Pages can be swapped to disk under memory pressure
- Pages can be migrated between NUMA nodes
- The OS can defragment by moving physical pages

The DMA engine cannot coordinate with the OS scheduler. If a transfer
started reading from physical address X and the OS moved that page
mid-transfer, the result is undefined.

### The bounce buffer fallback

The CUDA runtime works around this by maintaining an internal pinned
**bounce buffer**. When you call `cudaMemcpyAsync(dst, pageable_src, n, ...)`,
the runtime actually does:

1. **Synchronous** CPU-side `memcpy(bounce, pageable_src, n)`
   — your CPU thread is blocked the entire time
2. Async DMA from bounce buffer → GPU HBM
3. Return

For a 1.3 GB rotation key, step 1 alone is ~65 ms of pure CPU blocking.
Any prefetch you build above this is dead in the water — the host thread
never returns control fast enough to issue the next async copy.

### What `cudaHostRegister` changes

`cudaHostRegister(ptr, size, flags)` page-locks an existing allocation:

1. The kernel walks the **page table** for that virtual range
2. Each **page table entry (PTE)** is marked non-evictable — the OS
   swapper will never move or page out those physical frames
3. The pinned physical range is registered with the **IOMMU** so the
   DMA engine has a stable physical address it can read from directly

After registration, the same `cudaMemcpyAsync` call bypasses the bounce
buffer entirely: the DMA engine reads host DRAM → HBM in one pass, the
host thread returns immediately, and the transfer truly overlaps anything
else queued on the stream.

### How we use it in multiNEXUS

The 62 GB host-resident rotation-key store is registered once at startup
in `src/nexus_eval/galois_key_store.cuh`. Every subsequent
`cudaMemcpyAsync` from that buffer goes direct-DMA. Combined with the
double-buffered (two-slot) GPU staging area in `Bootstrapper::bootstrap_3`,
rotation N+1's key transfer overlaps with rotation N's compute kernel.
This was the Phase 1 unlock: 10,712 ms → 2,284 ms (4.69×).

### Pitfall to flag

Pinning is not free. It pins physical RAM until you `cudaHostUnregister`,
so for very large host buffers you can starve the rest of the system. At
62 GB on an MN5 ACC node (typically 512 GB host RAM), the headroom is
fine, but on a smaller node this would matter.

### One-liner for the PI

> "Without pinned memory, `cudaMemcpyAsync` is actually synchronous because
> the DMA engine needs stable physical addresses and the OS can relocate
> pageable pages. `cudaHostRegister` marks the page table entries
> non-evictable and registers the physical range with the IOMMU, so the
> DMA engine reads directly from host DRAM. That single change is the
> entire Phase 1 win."

---

## Section 2. CKKS in 11 minutes — the parts that matter for DKS

### 2.1 What CKKS encrypts

CKKS encrypts vectors of real or complex numbers and lets you do additions
and multiplications on the encrypted vectors. Output is approximate (noise
grows on every op) but accurate enough for ML. This is why it is the FHE
scheme of choice for transformer activations.

### 2.2 The polynomial ring

Native objects are polynomials of degree < N with integer coefficients,
reduced mod the cyclotomic polynomial X^N + 1.

```
   R   = ℤ[X] / (X^N + 1)
   R_Q = ℤ_Q[X] / (X^N + 1)
```

N is a power of two — the **ring degree**. We use **N = 65536**.

One plaintext polynomial encodes N/2 complex slots via the canonical
embedding. One ciphertext = one batched SIMD vector of 32,768 slots.
Operations act on all slots in parallel.

### 2.3 Ciphertext structure

A fresh CKKS ciphertext is a pair (c₀, c₁) ∈ R_Q × R_Q such that

```
   c₀ + c₁ · s ≈ Δ · m   (mod Q)
```

- s is the **secret key** (small polynomial)
- m is the encoded message
- Δ is a scaling factor

Decryption: compute c₀ + c₁ · s, descale by Δ.

### 2.4 The modulus chain — limbs

Q is huge (~1760 bits) and never stored as one big integer. Instead:

```
   Q = q₀ · q₁ · ... · q_L
```

Each q_i is a 50–60 bit prime called a **limb**. We have L+1 ≈ 44 limbs.

By the Chinese Remainder Theorem (CRT), a polynomial in R_Q is uniquely
represented by its L+1 reductions, one per limb. We do all arithmetic
limb-by-limb in parallel. Each limb polynomial has coefficients that fit
in 64 bits. This is the **RNS representation**.

A ciphertext = 2 · (L+1) small polynomials, each of degree < N.

### 2.5 Multiplication consumes a limb (rescaling)

ct · ct inflates noise and scale (Δ → Δ²). After every ct·ct multiply we
**rescale**: divide everything by the highest prime q_ℓ and drop that limb.

The chain shrinks. A ciphertext has "level ℓ" if it lives at modulus
q₀·q₁·...·q_ℓ. When ℓ = 0 you cannot multiply anymore — you need
**bootstrapping** to refresh the level.

### 2.6 Multiplication needs relinearization

ct · ct produces a 3-poly intermediate (c₀, c₁, c₂) with
```
   c₀ + c₁·s + c₂·s² ≈ Δ²·m₁·m₂
```
The extra c₂·s² term is undesirable. **Relinearization** uses a
key-switching key to convert c₂·s² back into a + b·s form, restoring the
clean 2-poly shape.

This is one example of **key-switching**.

### 2.7 Rotations also use key-switching

A rotation by k slots is a Galois automorphism: X → X^(5^k). The ring
algebra works, but the result is encrypted under s(X^(5^k)) instead of
s(X). Apply key-switching again, this time with a **rotation key** (Galois
key) precomputed for that specific rotation amount.

So every rotation = one Galois substitution + one key-switch. Bootstrap
performs ~75 rotations → ~75 key-switches. **Key-switching dominates
bootstrap cost.**

### 2.8 Key-switching, step by step

Five steps. Memorize.

**A. Digit decomposition.** Split the L+1 limbs of the offending poly into
**β groups** called **digits**. β is chosen at setup. Ours is β ≈ 36.

```
   d = (limb₀, limb₁, ..., limb_L)
   →  digit_0, digit_1, ..., digit_(β-1)   each ≈ (L+1)/β consecutive limbs
```

**B. Mod-up (EXPANSION).** For each digit, raise it from its small modulus
subset up to the extended modulus PQ (P = extra auxiliary primes added on
top for noise control).

```
   digit_i  (~19 MB at small modulus)
   →  digit_i_extended  (~780 MB at modulus PQ)
```

CRT basis conversion + tower of forward NTTs. **Mod-up expands ~40× per
digit.**

**C. Inner product.** The key-switching key has β rows; each row is a
small ciphertext (a, b) at modulus PQ. Multiply each digit-extended by its
corresponding key row, sum across digits:

```
   for i = 0..β-1:
       (αᵢ, βᵢ) = digit_i_extended · key_row_i
   (α, β) = ∑ᵢ (αᵢ, βᵢ)        ← the inner product (β-dim dot product)
```

**D. Mod-down.** Divide by P, reduce back to modulus Q. Result is at the
original level.

**E. Add back** into the original ciphertext to complete the key-switch.

### 2.9 Limbs vs digits — never confuse these

| Property        | Limb                          | Digit                                       |
|-----------------|-------------------------------|---------------------------------------------|
| Count           | L + 1 ≈ 44                    | β ≈ 36                                      |
| What it is      | one prime in the modulus chain| group of consecutive limbs                  |
| Purpose         | RNS representation            | Key-switching parallelism axis              |
| Relationship    | atomic                        | each digit holds (L+1)/β consecutive limbs  |

**One-liner:** *"Limbs are the primes in the modulus chain. Digits are
groups of limbs we batch together for key-switching. There are L+1 limbs
and β digits, with β < L+1."*

### 2.10 Where multi-GPU enters — DKS

Step C's β-dimensional inner product is embarrassingly parallel along β.

For G = 4 GPUs and β ≈ 36:

- GPU g owns a contiguous range of ~9 digits
- GPU g stores ONLY those rows of the key-switching key (18 GB vs 62 GB)
- GPU g computes its partial inner product → (α_g, β_g)
- All GPUs `ncclAllReduce(SUM)` over (α_g, β_g) → every GPU has (α, β)

The algebraic identity that makes this correct:

```
   (α, β) = ∑_i (αᵢ, βᵢ) = ∑_g ( ∑_{i ∈ D_g} (αᵢ, βᵢ) )
```

Sums commute with partition. AllReduce sums the four partials. Done.

### 2.11 Why mod-up is the wasteful step (T-MODUP)

Current code: every GPU runs mod-up over **all 36 digits** even though
each only consumes 9 in its inner product. 27 unowned outputs are computed
and discarded → ~850 ms of redundant NTT work per bootstrap.

T-MODUP optimization passes (d_start, d_count) into mod-up so each GPU
computes only its own digits. ~4× reduction in mod-up cost. Blocked by
runtime regression (zero-sized `cudaMallocAsync` at low chain levels when
d_count = 0).

### 2.12 Bootstrapping in one paragraph

When a ciphertext hits level 0, bootstrap to refresh. Three phases:

- **CoeffToSlot (CTS):** linear transform from coefficient → slot
  representation. Implemented as ~30 rotations + matrix mults.
- **Modular reduction:** evaluate a polynomial approximation of x mod q₀
  (sin/cos via Chebyshev or Remez). This is the noise-reset step.
- **SlotToCoeff (STC):** inverse linear transform back to coefficient
  form. Another ~30 rotations.

Total ~75 rotations per bootstrap. Each rotation = one key-switch. That's
why bootstrap takes 2.1 s and dominates BERT layer cost.

---

## Section 3. One DKS rotation, end to end

Setting: G = 4 GPUs, β = 36 digits (~9 per GPU), N = 65,536, current
chain level ℓ. Numbers below are approximate but representative.

### 3.1 t=0 — initial state on each GPU

Every GPU has:
- Its **own 9 rows** of every rotation key in the bootstrap key store
  (~18 GB total per GPU across all 75 rotations)
- A full local copy of the input ciphertext (c₀, c₁) at modulus Q
  (~38 MB — small enough to replicate)
- Workspace buffers (RotationWorkspace, etc., persistent — no per-call
  malloc)

The keys are SHARDED. The ciphertext is REPLICATED.

### 3.2 Step 1 — Galois automorphism (cheap, replicated)

Each GPU applies X → X^(5^k) to (c₀, c₁) locally. Just a permutation of
polynomial coefficients. Microseconds. All 4 GPUs do this identically.

The polynomial that needs key-switching is now `d` (~19 MB at current
chain level, with L+1 ≈ 44 limbs).

### 3.3 Step 2 — Mod-up (the wasteful step)

**Current (Phase 4b):** every GPU computes mod-up for ALL 36 digits.
Output: 36 × ~780 MB = ~28 GB of expanded digit polynomials in HBM. Each
GPU consumes only 9 of them; 27 are computed and discarded.

**T-MODUP target:** every GPU computes mod-up only for ITS 9 digits.
Output: ~7 GB. ~4× less mod-up work.

Mod-up internally:
1. CRT basis conversion from small-modulus subset → full PQ basis
2. Tower of forward NTTs to put result back in evaluation form

This is where the ~850 ms of redundant NTT work per bootstrap lives.

### 3.4 Step 3 — Partial inner product

GPU g loops over its 9 owned digits:

```
for i in owned_range:
    αᵢ = digit_i_extended · key_row_i.a       (poly·poly mod PQ)
    βᵢ = digit_i_extended · key_row_i.b
(α_g, β_g) = ∑ᵢ (αᵢ, βᵢ)
```

Implemented in `partial_key_switch_inner_prod`
(`src/multi_gpu/keyswitching/output_aggregation.cu`). The result
(α_g, β_g) is a 2-poly ciphertext at modulus PQ, ~32 MB on the wire.

Total inner-product cost per bootstrap across all 4 GPUs: ~142 ms.

### 3.5 Step 4 — ncclAllReduce(SUM)

```cpp
ncclAllReduce(
    sendbuf = (α_g, β_g),
    recvbuf = (α, β),
    count   = ~32 MB worth of uint64 limbs,
    op      = ncclSum,
    comm    = ourCommunicator,
    stream  = ourStream
);
```

On the NVSwitch fabric (~900 GB/s aggregate bidirectional), this kernel
runs ~1.77 ms mean per rotation. After AllReduce, every GPU has

```
(α, β) = ∑_g (α_g, β_g) = ∑_i (αᵢ, βᵢ)   for i = 0..35
```

Per bootstrap: 324 AllReduce calls × 1.77 ms = ~573 ms. That's the 27%
bucket in Table 5.

### 3.6 Step 5 — Mod-down (replicated)

Each GPU independently divides (α, β) by P, basis-converts back to
modulus Q, drops auxiliary limbs. Result (α', β') at modulus Q
(~38 MB). All 4 GPUs do this identically — no parallelism opportunity.

### 3.7 Step 6 — Add back

```
c₀_new = c₀_rotated + α'
c₁_new = c₁_rotated + β'
```

Each GPU now has the fully key-switched, rotated ciphertext. Ready for
the next rotation.

### 3.8 The timeline

Current Phase 4b:
```
GPU 0:  [galois][--mod-up all 36 (wasteful)--][IP][AllReduce][mod-down][add]
GPU 1:  [galois][--mod-up all 36 (wasteful)--][IP][AllReduce][mod-down][add]
GPU 2:  [galois][--mod-up all 36 (wasteful)--][IP][AllReduce][mod-down][add]
GPU 3:  [galois][--mod-up all 36 (wasteful)--][IP][AllReduce][mod-down][add]
                                                ↑
                                           all sync here
```

With T-MODUP:
```
GPU g:  [galois][mod-up 9][IP][AllReduce][mod-down][add]
                     ↑
              ~4× shorter
```

### 3.9 Where prefetch fits

The full key store is sharded but lives partly on host. Async H→D
transfer of rotation N+1's key overlaps with rotation N's compute (steps
1–6 above). With pinned memory + double-buffered staging, the per-key
H→D cost (~40 ms) is fully hidden behind the per-rotation compute.

### 3.10 What is sharded vs replicated — a cheat sheet

| Object                          | Sharded across GPUs? | Per-GPU size |
|---------------------------------|----------------------|--------------|
| Rotation keys (full bootstrap)  | YES (β-axis)         | ~18 GB       |
| Input ciphertext (c₀, c₁)       | NO (replicated)      | ~38 MB       |
| Mod-up output (deployed Phase 4b)| NO (all 36 on every GPU) | ~780 MB temp |
| Mod-up output (T-MODUP target)  | YES (9 per GPU)      | ~195 MB temp |
| Partial inner product (α_g, β_g)| YES (per-GPU)        | ~32 MB       |
| Final (α, β) after AllReduce    | NO (replicated by NCCL)| ~32 MB     |
| Mod-down result (α', β')        | NO (replicated)      | ~38 MB       |

### 3.11 Local code vs deployed code — IMPORTANT distinction

The local `output_aggregation.cu` calls `modup_partial(d_start, d_count)` —
this is the T-MODUP code path that computes only owned digits per GPU.
But this version fails at runtime on MN5 (zero-sized cudaMallocAsync at
chain levels where β < n_gpus, despite the std::max(1, ...) guard).

The deployed Phase 4b (which produces the published 107 s number) is
reverted to baseline mod-up over all β digits on every GPU.

So when discussing performance, the relevant numbers come from the
deployed (all-36-digits) version. When discussing the codebase, the
local repo holds the staged (broken) T-MODUP attempt.

### 3.12 Why redundant work on all GPUs is OK for cheap operations

Mod-down runs identically on all 4 GPUs after AllReduce. This is not a
bug — it's a deliberate design choice driven by an HPC tradeoff:

| Design          | Compute time | Comm cost | Total      |
|-----------------|--------------|-----------|------------|
| Replicated mod-down on all 4 | ~10 ms in parallel | 0 | ~10 ms |
| Mod-down on GPU 0 + broadcast | ~10 ms       | ~32 MB / 200 GB/s ≈ 0.2 ms | ~10.2 ms + idle GPUs |

The savings from "one GPU does it" are eaten by the broadcast and the
GPUs 1–3 sit idle. Replicate-when-cheap is the correct call. This is
the standard "owner-computes vs replicate-computes" tradeoff in HPC.

---

## Section 4. Why DKS is mathematically correct

### 4.1 The algebraic identity

Full single-GPU inner product:
```
(α, β) = Σᵢ (αᵢ, βᵢ)            i = 0..β-1
```

DKS partitions {0..β-1} into G disjoint groups D_0, ..., D_{G-1}:
```
(α_g, β_g) = Σᵢ∈D_g (αᵢ, βᵢ)    per-GPU partial
(α, β)     = Σ_g (α_g, β_g)      AllReduce result
```

These are equal:
```
Σ_g ( Σᵢ∈D_g (αᵢ, βᵢ) ) = Σᵢ (αᵢ, βᵢ)     iff partition is disjoint cover
```

The algebraic property is **associativity + commutativity of addition** —
sums distribute over partition. Our contiguous digit assignment
guarantees the partition is a disjoint cover, so the identity holds.

**One-liner:** *"DKS works because the inner product is a sum, and sums
distribute over partition. The disjoint digit assignment plus
AllReduce(SUM) gives the same result as the full single-GPU sum."*

### 4.2 Why ncclSum specifically

If you swap ncclSum for ncclMax, ncclMin, or ncclProd, the result has
no algebraic relationship to the key-switched ciphertext. The math of
key-switching mandates summation across digits — there is no other
reduction that satisfies the partition identity above. SUM is the only
correct choice.

### 4.3 Why mod-reduction is post-AllReduce, not inside it

NCCL has no knowledge of modular arithmetic — it operates on raw
uint64 with sum/prod/max/min. So we have three design options:

| Option | Cost |
|---|---|
| (a) Sum without mod, mod after — current | ~32 MB AllReduce + cheap post-kernel |
| (b) AllGather everything, mod-sum locally | ~128 MB AllGather + 4× redundant compute |
| (c) Custom collective with mod | doesn't exist in NCCL |

We pick (a). It's safe because:

1. Each partial element on one GPU is already in [0, q_j) (Barrett
   reduction inside `partial_key_switch_inner_prod`).
2. NCCL sums G = 4 such values → range [0, 4·q_j).
3. q_j is a 50–60 bit prime → 4·q_j ≤ 62 bits → fits in uint64. No
   overflow.
4. `mod_reduce_after_allreduce` brings everything back to [0, q_j) with
   one elementwise pass.

This breaks if G·q_j overflows uint64. At G=4, 60-bit primes, we have
margin. At G=16 with the same primes, you'd need either smaller q_j or
a multi-precision intermediate.

**One-liner:** *"NCCL is mod-arithmetic-blind, so we sum raw uint64s
across GPUs and apply the modular reduction in a separate post-kernel.
This is safe at G=4 because four 60-bit values sum to under 64 bits.
Cheapest correct design."*

---

## Section 5. NCCL, NVSwitch, and the AllReduce budget

### 5.1 What NVSwitch is

NVSwitch is a dedicated **switch ASIC** on the GPU baseboard. It provides
any-to-any GPU communication at full per-link bandwidth simultaneously.
Each H100 has 18 NVLink-4 links at 50 GB/s bidirectional each, giving
**~900 GB/s aggregate per GPU**. NVSwitch is non-blocking — all 4 GPUs
can be talking simultaneously without contention.

PCIe by contrast is a shared, tree-topology bus where all devices on the
root complex contend for the same bandwidth. PCIe Gen5 x16 is ~128 GB/s
bidirectional but shared across all PCIe peripherals.

### 5.2 Why NVSwitch matters for FHE

| Fabric | Per-GPU bidirectional | 32 MB AllReduce time |
|---|---|---|
| **NVSwitch** (MN5 ACC node) | ~900 GB/s | 1.77 ms (measured) |
| **PCIe Gen5 x16** | ~128 GB/s, **shared** | ~5–10 ms (estimated) |
| **PCIe Gen4 x16** | ~64 GB/s shared | ~10–20 ms |

A workstation with 4× H100 PCIe cards (no NVSwitch baseboard) cannot
reproduce our bootstrap latency. **NVSwitch is a hard prerequisite for
the operating point we publish.**

### 5.3 Ring vs Tree AllReduce

**Ring**:
- G GPUs in a ring
- Phase 1 (Reduce-Scatter, G-1 steps): each GPU passes M/G to its right
  neighbor; after G-1 steps each GPU owns the SUM of one chunk
- Phase 2 (AllGather, G-1 steps): each GPU passes its summed chunk
  around; after G-1 steps everyone has the full result
- Total: 2(G-1) steps, M/G bytes per step
- **Bandwidth-optimal for large messages**

**Tree**:
- GPUs in a binary tree
- Reduce up + Broadcast down
- Total: 2·log(G) steps
- **Latency-optimal for small messages**

**Crossover at ~1 KB.** NCCL picks automatically based on message size,
world size, and topology. For G=4, M=32 MB, **NCCL uses Ring.**

### 5.4 Sanity-check the 1.77 ms

Per-GPU traffic for ring AllReduce, G=4, M=32 MB:
```
chunk size           = M / G = 8 MB
steps                = 2(G-1) = 6
per-GPU send/recv    = 6 × 8 MB = 48 MB each direction
```

Theoretical floor:
```
bandwidth floor  = 48 MB / 450 GB/s ≈ 0.107 ms
latency floor    = 6 steps × ~50 µs ≈ 0.3 ms
combined floor   ≈ 0.4 ms
```

Measured: **1.77 ms**. We're at ~25% of theoretical peak. This is
typical NCCL efficiency — published bus efficiency is usually 60–80% of
theoretical due to chunked transfers, real link efficiency, kernel
launch overhead, and stream synchronization between phases.

### 5.5 Levers to close the gap

| Lever | Saving | Difficulty |
|---|---|---|
| Fewer AllReduces (Cinnamon-style key reuse) | up to 50% (fundamental) | Hard — compiler-level |
| Smaller messages (skip auxiliary limbs) | 10–20% | Medium — Phantom internals |
| T-OVERLAP (hide AllReduce behind next mod-up) | perceived ~100% (latency-hidden) | Medium — event ordering |
| NVIDIA SHARP (in-network reduction) | up to 2× | Easy if available; uncertain on MN5 |

### 5.6 One-liner for the PI

> "We measure 1.77 ms per AllReduce. Theoretical lower bound is ~0.4 ms,
> so we're at ~25% of peak — typical NCCL efficiency at this message
> size and world size. Closing the gap requires reducing the *number* of
> AllReduces (compiler-level, like Cinnamon) or hiding them behind
> compute via stream-event overlap (T-OVERLAP infrastructure, partly
> built)."

---

## Section 6. Streams, events, and the overlap engineering

### 6.1 Foundations

- **CUDA stream**: an **ordered queue of GPU operations**. Operations
  within a stream execute in order; operations on **different streams
  can execute concurrently** on different hardware units (SMs, copy
  engines, NCCL).
- **CUDA event**: a **marker recorded into a stream** — a checkpoint
  that other streams (or the host) can wait on. Used to express
  cross-stream dependencies without blocking the CPU.
- **`cudaStreamWaitEvent(streamA, event)`**: makes **streamA pause**
  until the event fires. The **CPU is NOT blocked** — this is purely
  GPU-side synchronization.

Mantra: **stream = queue. event = marker. CPU never blocks unless you
explicitly call cudaStreamSynchronize / cudaEventSynchronize.**

### 6.2 The Phase 1 prefetch ping-pong

Setup: 62 GB host-resident key store (pinned), 2 small device-side
"slots" each large enough for one rotation key, two streams:
`stream_compute` and `stream_copy`.

For each rotation N:

```
on stream_copy:
    cudaMemcpyAsync(slot[(N+1) mod 2], host_key[N+1], ...)
    cudaEventRecord(key_arrived[N+1], stream_copy)

on stream_compute:
    cudaStreamWaitEvent(stream_compute, key_arrived[N])
    rotation_kernel<<<..., stream_compute>>>(slot[N mod 2], ...)
```

**Why two slots:** with only one slot, the next key can't arrive while
the current rotation is still using the previous key — copy and compute
would serialize on the same buffer. Two slots = classic
producer-consumer double-buffer.

**Why two streams:** operations on the same stream are serialized.
Putting H2D copy on `stream_copy` and the rotation kernel on
`stream_compute` lets them run concurrently on different hardware (the
GPU's DMA copy engine vs the SMs).

**The sync primitive:** `cudaStreamWaitEvent`. The compute stream
pauses on the event the copy stream recorded. CPU never blocks.

### 6.3 T-OVERLAP (the straggler-bucket null result)

[`output_aggregation.cu:149–158`](../src/multi_gpu/keyswitching/output_aggregation.cu#L149-L158)
records `allreduce_done_events[gpu_id]` instead of host-syncing after
NCCL.

**What it was meant to do:** the original code did
`cudaStreamSynchronize(stream)` after `ncclAllReduce`, blocking the CPU
worker thread for ~291 ms (the kernel time of AllReduce). The
optimization removes the host sync so the worker thread can return
immediately and begin launching the next rotation's mod-up kernels on
the GPU. The GPU schedules everything correctly via stream events; the
CPU is free.

**Why it didn't deliver measured speedup:**

1. **Workload structure**: rotations execute sequentially in the
   bootstrap loop because rotation N+1 reads rotation N's output.
   There's no rotation-to-rotation pipelining for the optimization to
   exploit.
2. **The bucket wasn't there**: finer NVTX measurement showed the
   ~530 ms "straggler wait" bucket from the coarse profile was
   actually AllReduce *kernel* time, not host-side jitter. There was
   nothing to overlap.

**What we keep**: the event-based plumbing is correct and harmless. We
report the null result honestly in the paper rather than hiding it.

### 6.4 The PI-credible framing

> "T-OVERLAP is the engineering version of a hypothesis test. We saw a
> ~530 ms bucket that looked like host-side jitter and built event-based
> infrastructure to hide it behind the next rotation's mod-up. Then
> finer measurement showed the bucket was actually the AllReduce kernel
> itself — there was no slack to capture. The plumbing is in the code
> (it's correct), but we don't claim a speedup we didn't measure. This
> is the honest engineering result the paper documents."

---
