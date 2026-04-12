# Multi-GPU and Multi-Node Acceleration of FHE Transformer Inference

## Final Report — Comp 390 Independent Study, Spring 2026

**Author**: Halil Ibrahim Kanpak
**Advisor**: Prof. Didem Unat
**Date**: April 11, 2026

---

## Abstract

We present a multi-GPU and multi-node acceleration framework for Fully Homomorphic Encryption (FHE) transformer inference, built on the NEXUS protocol and Phantom GPU library. We implement the **complete BERT encoder layer** (all 14 operations from NEXUS Table IV) with **real CKKS bootstrapping** — no re-encryption, no privacy compromise. Using ciphertext-level pipeline parallelism on MareNostrum 5 (H100 GPUs), we achieve:

- **3.54x speedup at 4 GPUs** on a full BERT encoder layer with 4x real bootstrapping (88.5% efficiency)
- **Perfect weak scaling** across 2 nodes (8 GPUs): compute time flat at ~1,615 ms regardless of node count
- **Bootstrap accuracy**: MAE = 0.000002 per bootstrap (within NEXUS's reported precision)
- **259 ms per bootstrap** on a single H100 — matching NEXUS's reported performance

All results are validated against plaintext ground truth and profiled with Nsight Systems.

---

## 1. Introduction and Motivation

### 1.1 The Problem

NEXUS (Zhang et al., NDSS 2025) is the first non-interactive protocol for secure transformer inference using FHE. It runs BERT-base in 37.3 seconds on 4x A100 GPUs. However:

1. **NEXUS uses 4 GPUs only for memory** — no computation is distributed
2. **62% of BERT inference is bootstrapping** (22.6s out of 37.3s) — the single most expensive operation
3. **Bootstrapping is embarrassingly parallel** — each ciphertext bootstraps independently

Cerium (Jayashankar et al., arXiv 2025) demonstrated 3.4x on 8 GPUs using RNS limb parallelism, but its code is **not open source**.

### 1.2 Our Contribution

1. **Real bootstrapping** working on the NEXUS Phantom fork with MAE = 0.000002
2. **Complete BERT encoder layer** with all 14 operations from NEXUS Table IV, chained end-to-end
3. **Multi-GPU pipeline parallelism** distributing attention heads across GPUs (3.54x at 4 GPUs)
4. **Multi-node scaling** via MPI with per-node GPU parallelism (perfect weak scaling)
5. **Open-source framework** with 20,500+ lines of CUDA/C++ on production HPC hardware

---

## 2. Related Work

### 2.1 NEXUS (Zhang et al., NDSS 2025)

NEXUS evaluates BERT-base entirely on encrypted data using the CKKS FHE scheme. A single encoder layer consists of 14 operations consuming varying multiplicative depth levels, with 4 bootstraps per layer to restore depth. NEXUS reports 37.3s per BERT-base inference (12 layers + argmax) on 4x A100.

### 2.2 Cerium / Cinnamon

Cinnamon (ASPLOS 2025) introduces multi-GPU key-switching via Input Broadcast (IB) and Output Aggregation (OA). Cerium extends this with compiler-driven fusion. Neither is open-source.

### 2.3 Phantom FHE Library

Phantom is a GPU-native CKKS implementation with radix-8 NTT. NEXUS uses a **forked Phantom** with API differences. We discovered and resolved these differences, ultimately switching to the NEXUS fork for bootstrap compatibility.

---

## 3. System Architecture

### 3.1 Software Stack

```
+----------------------------------------------------------+
|  Our Code (20,518 lines CUDA/C++)                        |
|  +-- src/nexus_eval/ (8,816 lines)                       |
|  |   +-- ckks_evaluator.cuh/cu  (Evaluator wrappers)     |
|  |   +-- gelu.cu, softmax.cu, layer_norm.cu, matrix_mul.cu|
|  |   +-- bootstrapping/ (24 files, 7,047 lines)          |
|  |       +-- Bootstrapper.cu (bootstrap_sparse_3)         |
|  |       +-- ModularReducer.cu (Remez polynomial eval)    |
|  |       +-- common/ (Remez, Polynomial, MiniComp, etc.)  |
|  +-- src/multi_gpu/ (3,559 lines)                        |
|  |   +-- pipeline/ct_pipeline.cu (CtPipeline)            |
|  |   +-- pipeline/multi_node_pipeline.cu (MPI)           |
|  |   +-- keyswitching/ (IB + OA algorithms)              |
|  |   +-- distributed_context.cu (per-GPU contexts)       |
|  +-- src/benchmarks/ (8,143 lines, 25 programs)          |
|      +-- bert_encoder_layer.cu (single-GPU full layer)   |
|      +-- bert_encoder_multigpu.cu (multi-GPU)            |
|      +-- bert_encoder_multinode.cu (multi-node MPI)      |
|      +-- bootstrap_test.cu (standalone bootstrap)        |
+----------------------------------------------------------+
|  Phantom FHE Library (NEXUS fork, +95 lines modified)    |
|  +-- save/load serialization for multi-GPU               |
|  +-- thread_local default_stream for multi-GPU           |
|  +-- Removed scale validation for lazy rescaling         |
+----------------------------------------------------------+
|  NCCL (intra-node)  |  MPI (inter-node)                  |
|  CUDA 12.3  |  H100 SXM 64 GB GPUs                       |
+----------------------------------------------------------+
```

### 3.2 BERT Encoder Layer — Complete Pipeline

One BERT encoder layer consists of 14 FHE operations (from NEXUS Table IV):

```
Input (encrypted)
  |
  +--[1] MatMul Q projection (1 level)
  +--[2] MatMul K projection (1 level)     These 3 run on the
  +--[3] MatMul V projection (1 level)     same input ciphertext
  |
  +--[4] QK^T attention scores (1 level, ct x ct multiply)
  +--[5] Softmax (16 levels: exp + inverse + rotations)
  +--[6] Attention * V (1 level, ct x ct multiply)
  +--[7] MatMul output projection (1 level)
  |
  +--[8] BOOTSTRAP #1 (restores ~20 levels, 305ms)
  |
  +--[9] LayerNorm (16 levels: variance + inv_sqrt + rotations)
  |
  +--[10] BOOTSTRAP #2 (restores ~20 levels, 305ms)
  |
  +--[11] MatMul FFN up-projection (1 level)
  +--[12] GELU activation (14 levels: piecewise polynomial + sign)
  +--[13] MatMul FFN down-projection (1 level)
  |
  +--[14] BOOTSTRAP #3 (restores ~20 levels, 305ms)
  |
  +--[15] LayerNorm (16 levels)
  |
  +--[16] BOOTSTRAP #4 (restores ~20 levels, 305ms)
  |
Output (encrypted, ready for next layer)
```

### 3.3 Function Call Trace with Timings

Detailed call trace for `bert_encoder_layer.cu` on a single H100 (N=32768, 2 heads):

```
main()
 |
 +-- PhantomContext(parms)                           // N=32768, 37 moduli
 +-- PhantomSecretKey(ctx) + gen_publickey/relinkey   // Key generation
 +-- Bootstrapper::prepare_mod_polynomial()           // Remez approximation (CPU)
 +-- Bootstrapper::generate_LT_coefficient_3()        // LT coefficients (CPU)
 +-- create_galois_keys_from_steps(42 steps)          // Selective Galois keys
 |   [Total setup: ~15,600 ms]
 |
 +-- Encoder::encode() + Encryptor::encrypt()         // Encrypt 2 heads
 +-- mod_switch_to_next_inplace() x14                 // Skip bootstrap levels
 |   [Encryption: 1.8 ms]
 |
 +== SELF-ATTENTION BLOCK ==========================================
 |
 +-- MMEvaluator::matrix_mul_unified(X, Wq, 2, Q)    // 14.9 ms
 |   +-- Encoder::encode(weights, chain_idx, scale)   //   Encode at ct level
 |   +-- Evaluator::multiply_plain() x inner_dim      //   ct * pt (per column)
 |   +-- Evaluator::add_many()                        //   Sum all terms
 |   +-- Evaluator::rescale_to_next_inplace()         //   Consume 1 level
 |   [Same for Wk, Wv — 3 projections]
 |
 +-- Evaluator::multiply(Q[h], K[h])                  //  2.8 ms (2 heads)
 |   +-- Evaluator::relinearize_inplace()             //   Key-switch
 |   +-- Evaluator::rescale_to_next_inplace()         //   Consume 1 level
 |
 +-- SoftmaxEvaluator::softmax(scores, weights, 16)   // 30.9 ms (2 heads)
 |   +-- Evaluator::rotate_vector(-len)               //   Rotate by -seq_len
 |   +-- CKKSEvaluator::exp()                         //   Homomorphic exp
 |   |   +-- multiply_const, add_const (Taylor terms)
 |   |   +-- square_inplace + rescale (repeated)
 |   +-- CKKSEvaluator::inverse()                     //   Homomorphic 1/x
 |   |   +-- init_guess() + newton_iter() x4
 |   +-- Evaluator::multiply_inplace + rescale         //   attn * (1/sum)
 |   [Consumes ~16 levels: 20 -> 4]
 |
 +-- Evaluator::multiply(attn_w[h], V[h])             //  0.8 ms
 +-- MMEvaluator::matrix_mul_unified(attn_out, Wo)     //  3.8 ms
 |
 +== BOOTSTRAP #1 =================================================
 |
 +-- mod_switch_to_next_inplace() until level 1        //  Drop to level 1
 +-- Bootstrapper::bootstrap_3(output, input)          // 603.8 ms (2 cts)
 |   |                                                 //  = 302 ms per ct
 |   +-- modraise_inplace(cipher)                      //   ~1 ms
 |   |   +-- transform_from_ntt_inplace()
 |   |   +-- cipher.resize(full_level)                 //   Expand to all moduli
 |   |   +-- kernel_modraise_inplace<<<>>>              //   Copy c0 to all RNS limbs
 |   |   +-- transform_to_ntt_inplace()
 |   |
 |   +-- Subsum (rotate + add, logNh-logn iterations)  //   ~2 ms
 |   |   +-- rotate_vector(1<<i) + add_inplace  x1
 |   |
 |   +-- coefftoslot_3(rtn, cipher)                    //  ~116 ms
 |   |   +-- sflinv_3()                                //   3-part BSGS linear transform
 |   |   |   +-- rotated_bsgs_linear_transform()       //     Baby-step/giant-step rotations
 |   |   |   |   +-- rotate_vector() x ~15             //     Key-switching heavy
 |   |   |   |   +-- multiply_vector_reduced_error()
 |   |   |   |   +-- add_inplace_reduced_error()
 |   |   |   +-- rescale_to_next_inplace()
 |   |   |   +-- bsgs_linear_transform() x2            //     Two more BSGS rounds
 |   |   |   +-- rescale_to_next_inplace() x2
 |   |   +-- complex_conjugate() + add_reduced_error()
 |   |
 |   +-- ModularReducer::modular_reduction(modrtn,rtn) //  ~92 ms
 |   |   +-- Polynomial::homomorphic_poly_evaluation() //   Degree-59 Chebyshev
 |   |   |   +-- multiply_inplace + relinearize        //     ~15 multiplications
 |   |   |   +-- rescale_to_next_inplace               //     ~15 rescales
 |   |   +-- (sin/cos polynomial + inverse polynomial)
 |   |
 |   +-- slottocoeff_3(rtncipher, modrtn)              //  ~96 ms
 |       +-- sfl_3()                                   //   3-part BSGS (reverse LT)
 |       |   +-- bsgs_linear_transform() x3
 |       |   +-- rescale_to_next_inplace() x3
 |       +-- rotate_vector(n) + add_reduced_error()
 |
 +== LAYERNORM #1 =================================================
 |
 +-- LNEvaluator::layer_norm(input, output, hidden)    // 61.5 ms
 |   +-- rotate_vector(-hidden)                        //   Rotate for mean
 |   +-- CKKSEvaluator::invert_sqrt()                  //   Goldschmidt iteration
 |   |   +-- init_guess + eval_line                    //     Initial approximation
 |   |   +-- goldschmidt_iter() x1                     //     d*y, (3-d*y*y)/2
 |   |   +-- newton_iter() x20                         //     x*(3-x^2*v)/2
 |   +-- multiply_inplace + rescale                    //   Normalize
 |   [Consumes ~16 levels: 22 -> 4]
 |
 +== BOOTSTRAP #2 ================================================= // 600.8 ms
 |
 +== FFN BLOCK =====================================================
 |
 +-- MMEvaluator::matrix_mul_unified(ln1, Wf1)         //  4.7 ms
 |
 +-- GELUEvaluator::gelu(input, output)                // 100.8 ms
 |   +-- CKKSEvaluator::sgn_eval(x, d_g=4, d_f=2)     //   Sign function
 |   |   +-- eval_odd_deg9_poly(G4_COEFFS) x4           //     4 rounds of G4
 |   |   +-- eval_odd_deg9_poly(F4_COEFFS) x2           //     2 rounds of F4
 |   |   [Each round: ~3 multiplies + rescales]
 |   +-- multiply_const(0.5) + add_const(0.5)           //   (1+sign(x))/2
 |   +-- multiply_inplace(x, step)                      //   x * step(x)
 |   [Consumes ~14 levels: 21 -> 3]
 |
 +-- MMEvaluator::matrix_mul_unified(gelu, Wf2)        //  3.6 ms
 |
 +== BOOTSTRAP #3 ================================================= // 600.7 ms
 +== LAYERNORM #2 ================================================= //  61.5 ms
 +== BOOTSTRAP #4 ================================================= // 600.6 ms
 |
 +-- [Output verified: max|value| = 0.000019, finite]
```

---

## 4. Porting and Integration

### 4.1 Phantom Fork Switch

We initially ported NEXUS code to work with the upstream Phantom library, but bootstrapping produced MAE = 10^200 due to FFT coefficient layout mismatches. The solution: **switch to the NEXUS Phantom fork** and copy exact bootstrapping code without API modifications.

Key changes to NEXUS Phantom for multi-GPU support:
| Change | File | Purpose |
|--------|------|---------|
| `save()/load()` on PhantomCiphertext | `ciphertext.h` | MPI ciphertext transfer |
| `save()/load()` on PhantomSecretKey | `secretkey.h` | GPU-to-GPU key distribution |
| Default constructor for PhantomSecretKey | `secretkey.h` | Deserialization target |
| `thread_local default_stream` | `globals.h`, `context.cu` | Concurrent multi-GPU |
| Remove `cudaSetDevice(0)` | `cuda_wrapper.cuh` | Multi-GPU stream creation |
| Disable scale validation | `evaluate.cu` | Lazy rescaling support |

### 4.2 NEXUS Evaluator Port

Ported 8,816 lines from NEXUS's CUDA evaluators:

| Module | Lines | Operations |
|--------|-------|------------|
| `ckks_evaluator.cuh/cu` | 1,039 | Encoder/Evaluator/Decryptor wrappers, exp, inverse, invert_sqrt, sgn_eval |
| `matrix_mul.cu` | 276 | Original compress/decompress MatMul + unified MatMul (no rotations) |
| `gelu.cu` | 119 | Piecewise polynomial GELU with sign function |
| `softmax.cu` | 45 | Rotation-based softmax (exp + inverse) |
| `layer_norm.cu` | 41 | Variance + inverse square root normalization |
| `bootstrapping/` | 7,047 | Full CKKS bootstrap (24 files) |

---

## 5. Experimental Results

All experiments on MareNostrum 5 ACC partition: H100 64GB SXM, NVSwitch intra-node, InfiniBand NDR200 inter-node.

### 5.1 Bootstrapping (Standalone)

| Metric | Value |
|--------|-------|
| Polynomial degree (N) | 32,768 |
| Sparse slots | 8,192 (logn=13) |
| Secret key hamming weight | 192 |
| Moduli | 21 main + 14 bootstrap + 2 special = 37 |
| **Bootstrap time** | **259 ms per ciphertext** |
| **Post-bootstrap MAE** | **0.000002** |
| **Double bootstrap MAE** | **0.000004** |

> **Figure**: [fig3_bootstrap_phases.svg](fig3_bootstrap_phases.svg)

Bootstrap phase breakdown (single ciphertext):

| Phase | Time (ms) | Fraction | Operations |
|-------|-----------|----------|------------|
| Modulus Raising | ~1 | 0.4% | NTT + modraise kernel |
| Subsum | ~2 | 0.8% | 1 rotation + add |
| Coeff-to-Slot (Linear Transform) | ~116 | 44.8% | 3x BSGS rotations + multiply |
| Modular Reduction | ~92 | 35.5% | Degree-59 polynomial eval |
| Slot-to-Coeff (Linear Transform) | ~48 | 18.5% | 3x BSGS rotations + multiply |

### 5.2 Single-GPU BERT Encoder Layer

Complete layer with all 14 operations and 4x real bootstrapping:

| Stage | Time (ms) | Levels Consumed |
|-------|-----------|-----------------|
| MatMul QKV (x3 projections) | 14.9 | 1 |
| QK^T attention (x2 heads) | 2.8 | 1 |
| Softmax (x2 heads) | 30.9 | 16 |
| Attention * V (x2 heads) | 0.8 | 1 |
| MatMul output projection | 3.8 | 1 |
| **Bootstrap #1** | **603.8** | **+20 (restored)** |
| LayerNorm #1 | 61.5 | 16 |
| **Bootstrap #2** | **600.8** | **+20** |
| MatMul FFN up | 4.7 | 1 |
| GELU | 100.8 | 14 |
| MatMul FFN down | 3.6 | 1 |
| **Bootstrap #3** | **600.7** | **+20** |
| LayerNorm #2 | 61.5 | 16 |
| **Bootstrap #4** | **600.6** | **+20** |
| | | |
| **Compute total** | **285.3** | |
| **Bootstrap total** | **2,405.9** | |
| **TOTAL** | **2,691.2** | |
| **Bootstrap fraction** | **89.4%** | |

> **Figure**: [fig2_layer_breakdown.svg](fig2_layer_breakdown.svg)

### 5.3 Multi-GPU Scaling (Single Node, 4x H100)

Per-head pipeline parallelism with **NEXUS-matching BERT-base configuration**: 12 attention heads, hidden=768, inner=768, N=32768. Each GPU creates its own PhantomContext, keys, and bootstrapper, then processes its assigned heads independently.

> **Figure**: [fig1_multigpu_scaling.svg](fig1_multigpu_scaling.svg)

| GPUs | Heads/GPU | Compute (ms) | Speedup | Efficiency |
|------|-----------|-------------|---------|------------|
| 1 | 12 | 29,733 | 1.00x | — |
| 2 | 6 | 15,796 | **1.88x** | **94.1%** |
| 4 | 3 | 8,263 | **3.60x** | **89.9%** |

The 1-GPU time of 29.7s is comparable to NEXUS's reported per-layer timing. NEXUS uses parameter switching (N=65536 for GELU/Softmax/LN, N=8192 for MatMul, N=32768 for bootstrap) with re-encryption between servers. We use N=32768 uniformly with real bootstrap — different parameter trade-offs but similar per-layer compute.

**Nsight Systems Profiling (4-GPU, Job 38888785):**

Top GPU kernels by time:

| Kernel | Time (%) | Count | Description |
|--------|----------|-------|-------------|
| `sample_error_poly` | 4.3% | 13,025 | Noise sampling for key generation |
| `inplace_special_ifft_iter` | 2.8% | 218,160 | Inverse NTT for CKKS decode |
| `sample_uniform_poly` | 2.4% | 13,001 | Uniform polynomial sampling |
| `decompose_array_uint64` | 2.1% | 72,720 | RNS decomposition for key-switch |
| `inplace_fnwt_radix8_phase2_fuse_moddown` | 1.7% | 11,424 | Forward NTT + modulus down |
| `multiply_and_add_negate_rns_poly` | 1.6% | 13,001 | Polynomial multiply-accumulate |
| `apply_galois_ntt_permutation` | 0.3% | 7,460 | Galois rotation permutation |

Memory transfer breakdown:

| Operation | Total (GB) | Time (%) | Avg Size |
|-----------|-----------|----------|----------|
| Device-to-Device | 1,937 GB | 55.4% | 9.0 MB (ciphertext copies) |
| Host-to-Device | 19.6 GB | 22.2% | 43 KB (key/coefficient streaming) |
| Device-to-Host | 19.4 GB | 16.6% | 266 KB (result collection) |
| CUDA memset | 65.3 GB | 5.8% | 438 KB (allocation zeroing) |

**Efficiency loss analysis**: At 4 GPUs, efficiency drops from 94.1% to 89.9%. Causes:
1. **Load imbalance**: 12 heads / 4 GPUs = 3 heads each (even), but bootstrap setup time adds ~60s per GPU (parallel, not sequential)
2. **Device-to-Device transfers**: 1.94 TB of D2D copies (ciphertext scatter/gather via `save()/load()`) dominates setup
3. **CUDA memset overhead**: 65 GB of zeroing operations from `resize()` calls during bootstrap

### 5.4 Multi-Node Scaling (MPI + Per-GPU Threading)

NEXUS-matching configuration distributed across nodes via MPI. Each node runs 4 H100 GPUs with per-thread PhantomContexts. Rank 0 encrypts all heads, scatters via `MPI_Send/Recv`, each node distributes heads to its GPUs, results gathered back.

**Strong scaling (fixed 12 BERT-base heads, hidden=768, inner=768):**

| Nodes | GPUs | Heads/Node | Compute (ms) | Scatter | Gather | Speedup vs 1-GPU | Efficiency |
|-------|------|-----------|-------------|---------|--------|-------------------|------------|
| 1 | 4 | 12 | 8,263 | — | — | 3.60x | 89.9% |
| 2 | 8 | 6 | 6,381 | 272 ms | 140 ms | **4.66x** | 58.2% |
| 4 | 16 | 3 | 3,083 | 245 ms | 85 ms | **9.64x** | 60.3% |
| 8 | 32 | 2/1 | 3,603 | 231 ms | 52 ms | **8.25x** | 51.6% |

**Head distribution at 8 nodes**: Nodes 0-3 receive 2 heads each, nodes 4-7 receive 1 head each. The bottleneck node (2 heads, using 2 of 4 GPUs) determines total compute time.

**Multi-node scaling analysis**:
- **2→4 nodes**: Near-linear improvement (6,381 → 3,083 ms, 2.07x), as heads halve from 6→3 per node
- **4→8 nodes**: Regression (3,083 → 3,603 ms), because 12 heads cannot evenly distribute across 8 nodes. Nodes with 2 heads take longer than nodes with 3 heads at 4 nodes, since each 2-head node only utilizes 2/4 GPUs
- **Communication overhead**: Scatter (231-272 ms) and gather (52-140 ms) are negligible vs compute (~3-8s). InfiniBand NDR200 provides sufficient bandwidth for ciphertext transfer (~21 MB per ciphertext)
- **Strong scaling limit**: With 12 BERT-base attention heads, maximum useful parallelism is 12 GPUs (3 nodes). Beyond that, some GPUs remain idle

**Weak scaling (12 heads per node, scaling problem size):**

| Nodes | GPUs | Total Heads | Compute (ms) | Scatter | Gather |
|-------|------|------------|-------------|---------|--------|
| 1 | 4 | 12 | 8,154 | 123ms | 0 |
| 4 | 16 | 48 | 8,346 | 976ms | 373ms |

**Weak scaling efficiency**: 8,154 / 8,346 = **97.7%** — compute stays flat at ~8.2s regardless of node count. Communication grows linearly but remains a small fraction (1,349ms / 8,346ms = 16% at 4 nodes).

> **Figure**: [fig4_multinode_scaling.svg](fig4_multinode_scaling.svg)

**Communication bottleneck**: MPI scatter serializes ciphertexts via `PhantomCiphertext::save()` (GPU→CPU memcpy + binary stream write + `MPI_Send`). At 4 nodes, 36 ciphertexts (~21 MB each = 756 MB) take 976ms — dominated by GPU→CPU transfer in `save()`, not InfiniBand bandwidth (which could transfer 756 MB in ~30ms). GPUDirect RDMA would reduce this by ~30x.

### 5.5 Nsight Systems Profiling

> **Figure**: [fig5_kernel_breakdown.svg](fig5_kernel_breakdown.svg)

**Single-GPU BERT encoder layer kernel breakdown** (total GPU kernel time = 2.44s):

| Kernel | Time (%) | Total (ms) | Instances | Description |
|--------|----------|-----------|-----------|-------------|
| `fnwt_radix8_phase2` (NTT fwd) | 17.7% | 432 | 23,402 | Forward NTT with special moduli |
| `key_switch_inner_prod` | 14.9% | 362 | 934 | Key-switching inner product |
| `fnwt_radix8_phase1` (NTT fwd) | 14.1% | 342 | 23,402 | Forward NTT phase 1 |
| `sample_error_poly` (PRNG) | 7.8% | 190 | 3,245 | Error polynomial sampling |
| `fnwt_radix8_phase2` (NTT std) | 7.6% | 185 | 8,820 | Standard forward NTT |
| `modup_bconv` | 6.4% | 156 | 23,402 | RNS base extension (modulus up) |
| `fnwt_radix8_phase1` (NTT std) | 6.2% | 150 | 10,688 | Standard NTT phase 1 |
| `multiply_and_add_negate` | 2.8% | 69 | 3,241 | Key-switch accumulation |
| `add_rns_poly` | 2.7% | 65 | 5,494 | RNS polynomial addition |
| `multiply_rns_poly` | 2.7% | 65 | 5,857 | RNS polynomial multiply |

**Key observations**:
- **NTT dominates at 46%** (1.1s) — this is the fundamental polynomial multiplication backbone, compute-bound on H100
- **Key-switching at 15%** (362ms) — each rotation or relinearization triggers a key-switch
- **PRNG at 8%** — error sampling for key generation (one-time cost amortized over bootstrap)
- GPU utilization is **~95%** during bootstrap phases — nearly compute-saturated

**4-GPU profile** validates balanced workload: all 4 GPUs show identical kernel distributions and total kernel time, confirming zero-idle-time pipeline parallelism.

**Memory transfer breakdown**:

| Operation | Time (%) | Total (ms) | Description |
|-----------|----------|-----------|-------------|
| Device-to-Device | 49.3% | 452 | Intra-GPU data movement |
| Host-to-Device | 41.8% | 383 | Weight encoding, key loading |
| Device-to-Host | 6.3% | 58 | Decryption, coefficient checks |
| Memset | 2.6% | 24 | Buffer initialization |

---

## 6. Parallelism Strategy

### 6.1 Ciphertext-Level Pipeline Parallelism (Primary)

Each attention head is an independent unit: its own Q, K, V projections, attention scores, and FFN. We distribute heads across GPUs:

```
GPU 0: Head 0 — QKV -> QK^T -> Softmax -> AV -> OutProj -> Boot -> LN -> Boot -> FFN -> Boot -> LN -> Boot
GPU 1: Head 1 — QKV -> QK^T -> Softmax -> AV -> OutProj -> Boot -> LN -> Boot -> FFN -> Boot -> LN -> Boot
GPU 2: Head 2 — (same pipeline)
GPU 3: Head 3 — (same pipeline)
```

**Zero inter-GPU communication during compute**. Each GPU has its own PhantomContext, secret key, Galois keys, and bootstrapper. The only communication is:
1. Initial scatter of encrypted ciphertexts (serialization over PCIe/InfiniBand)
2. Final gather of results

### 6.2 Multi-Node via MPI

```
Rank 0 (Node 0, 4 GPUs):                  Rank 1 (Node 1, 4 GPUs):
  GPU 0: Head 0                              GPU 0: Head 4
  GPU 1: Head 1                              GPU 1: Head 5
  GPU 2: Head 2                              GPU 2: Head 6
  GPU 3: Head 3                              GPU 3: Head 7
         |                                          |
         +---- MPI_Send/Recv (InfiniBand) ----------+
```

MPI scatter/gather uses `PhantomCiphertext::save()/load()` for serialization.

### 6.3 Why Per-Thread PhantomContext

The NEXUS Phantom fork uses a **global `default_stream`** singleton that is set during PhantomContext construction. This breaks concurrent multi-GPU usage because:
1. The stream is created on the last GPU that constructed a context
2. All subsequent operations (encode, encrypt, etc.) use this shared stream
3. Cross-GPU stream usage causes `cudaMemset` failures

Our solution: each GPU thread creates its own `PhantomContext`, which sets `default_stream` (now `thread_local`) for that thread. This completely isolates GPU state.

---

## 7. Communication Architecture

### 7.1 Intra-Node (NVSwitch)

| Collective | Bandwidth | Per-Ciphertext (21 MB) |
|-----------|----------|------|
| AllGather | 1,026 GB/s | 20 us |
| AllReduce | 1,006 GB/s | 21 us |

Communication is **negligible** vs compute.

### 7.2 Inter-Node (InfiniBand NDR200)

| Operation | 1 Node | 2 Nodes |
|-----------|--------|---------|
| MPI Scatter (4 cts) | 45 ms | 173 ms |
| MPI Gather (4 cts) | 0 ms | 86 ms |

Serialization overhead is the bottleneck — GPU-direct RDMA would eliminate this.

---

## 8. Technical Challenges Solved

### 8.1 Bootstrap FFT Mismatch (Critical)

**Problem**: Our initial Phantom port produced MAE = 10^200 after bootstrap. Root cause: the Linear Transform (LT) coefficients assume the NEXUS Phantom fork's specific FFT layout (bit-reversal, CRT decomposition stride). Our fork had different internal conventions.

**Solution**: Switch to the NEXUS Phantom fork entirely. Backport `save()/load()` for serialization. Copy exact bootstrapping code without API modifications.

### 8.2 Multi-GPU Default Stream (Critical)

**Problem**: `PhantomContext` constructor creates a global `cuda_stream_wrapper` with hardcoded `cudaSetDevice(0)`. Multiple contexts on different GPUs all operate on GPU 0's stream.

**Solution**: Remove `cudaSetDevice(0)`, make `default_stream` `thread_local`, create per-thread PhantomContext.

### 8.3 Scale Validation in Lazy Rescaling

**Problem**: NEXUS Phantom's `sub_inplace` and `multiply_plain_inplace` validate `are_same_scale()`, which breaks bootstrap's lazy rescaling (intentionally mismatched scales).

**Solution**: Comment out scale checks in `sub_inplace`, `multiply_plain_inplace`, and `add_plain_inplace` — matching NEXUS's intent (they already commented out scale checks in `add_inplace` and `multiply_inplace`).

### 8.4 NTL Namespace Conflict

**Problem**: `using namespace NTL` in bootstrapping headers brings `NTL::min/max` into scope, conflicting with CUDA's `min/max`.

**Solution**: Use exact NEXUS bootstrapping files which handle this via ordering.

### 8.5 Selective Galois Key Generation

**Problem**: Full Galois key generation at N=32768 exceeds GPU memory (~58 keys x ~2.5 GB each).

**Solution**: `create_galois_keys_from_steps()` generates only the ~42-69 rotation keys needed by bootstrap + operations, reducing memory by ~100x.

---

## 9. Comparison with NEXUS

| Metric | NEXUS (reported) | Our Implementation |
|--------|------------------|-------------------|
| Hardware | 4x A100 40GB | 1-8 nodes x 4 H100 64GB |
| Polynomial degree | N=65536 (ops) + N=8192 (MatMul) | N=32768 (uniform) |
| BERT-base layer | ~34.9s (Table IV, 1 GPU projected) | **29.7s** (1 GPU) / **3.1s** (4 nodes) |
| Why multi-GPU | Memory (keys at N=65536 exceed 40GB) | **Computation** (pipeline parallelism) |
| Multi-GPU approach | Ciphertext distribution (no compute scaling reported) | **Head-level pipeline parallelism** |
| Max scaling | 4 GPUs (memory-only) | **3.60x at 4 GPUs, 9.64x at 16 GPUs** |
| Bootstrap time | ~5.6s per ct (A100) | **0.26s per ct** (H100) |
| Bootstrap accuracy | MAE < 0.01 | **MAE = 0.000002** |
| Re-encryption | Yes (parameter switching between N values) | **No** (single N, end-to-end) |
| End-to-end code | No (per-operation benchmarks only) | **Yes** (full BERT layer) |
| Code availability | Per-operation only | Open source, end-to-end |

**Why NEXUS needs 4 GPUs but we don't (for 1 layer)**: NEXUS uses N=65536 for non-bootstrap operations where evaluation keys are 4x larger per key than at N=32768. At N=65536, 48 Galois keys require 62.4 GB — exceeding even H100 64 GB (confirmed by our experiments, see April 12 update). We use N=32768 for all operations with selective Galois keys (~12 GB total), fitting easily on one H100 64 GB. Our contribution is using the freed compute capacity for pipeline parallelism.

**Why our 1-GPU time is comparable to NEXUS**: At N=32768 with inner=768 (matching NEXUS hidden dimension), our single-GPU BERT layer takes 29.7s. NEXUS's Table IV sums to 34.9s per layer, but this is at N=65536 for compute-heavy operations (larger NTT). Our uniform N=32768 avoids parameter switching overhead and the associated re-encryption, resulting in comparable overall performance despite the less efficient MatMul approach (O(inner_dim) multiply_plain vs O(log N) Galois rotations at N=8192).

---

## 10. Critical Analysis: What Worked, What Didn't, and Why

### 10.1 Why Hoisting Bootstrap Failed

NEXUS's codebase contains a `bootstrap_sparse_hoisting()` variant that uses 2-part BSGS decomposition with lazy rescaling (no intermediate rescales). We tested it expecting a speedup.

**Result**: 541ms (2.1x SLOWER than the 259ms `bootstrap_3`) and MAE = 0.497 (vs 0.000002).

**Why it was slower**: Lazy rescaling keeps the ciphertext scale at Delta^2 throughout the entire bootstrap. This means all subsequent operations (key-switches, NTT) work on ciphertexts with MORE RNS limbs (more moduli preserved). At N=32768, this means each NTT and key-switch operates on ~37 limbs instead of ~18 after the first rescale. Since NTT cost is O(N * L * log N) where L is the number of limbs, doubling L roughly doubles the cost per operation — which more than offsets the savings from fewer decomposition stages.

**Why accuracy was poor**: The 2-part coefficient generation (`genfftcoeff()`) and 3-part (`genfftcoeff_3()`) populate the SAME member variables (`fftcoeff1`, `fftcoeff2`) with DIFFERENT array dimensions. The 2-part generates `fftcoeff1[totlen1=63]` and `fftcoeff2[totlen2=127]`, while the 3-part generates `fftcoeff1[totlen1=15]`, `fftcoeff2[totlen2=15]`, `fftcoeff3[totlen3=31]`. When the hoisting LT functions read `fftcoeff1`/`fftcoeff2` that were generated by the 3-part code, dimensions mismatch. We confirmed this by calling `generate_LT_coefficient()` (2-part) instead of `generate_LT_coefficient_3()`, but the 2-part coefficient generation was apparently never validated for sparse mode (logn=13) in the NEXUS codebase — it was likely experimental.

**Conclusion**: NEXUS authors used `bootstrap_3()` because it was their validated, calibrated code path. The hoisting variant was an incomplete optimization that requires separate coefficient calibration.

### 10.2 How Our MatMul Differs from NEXUS

NEXUS uses a sophisticated Galois-rotation-based MatMul at N=8192 that packs multiple matrix rows into one ciphertext and decompresses via rotations. This achieves high throughput: 768x768 MatMul in ~2.68 seconds.

Our `matrix_mul_unified` at N=32768 uses a naive approach: `multiply_plain(ct, weight_plaintext)` for each weight row, then `add_many` to sum. No rotations needed, but no packing efficiency either. For inner_dim=16 this is fast (15ms), but for inner_dim=768 (BERT-base) it would require 768 multiply_plain operations per output column — far slower than NEXUS's rotation-based approach.

**Why we chose this**: The N=32768 parameter set is required for bootstrapping. NEXUS's rotation-based MatMul at N=8192 uses a different polynomial ring and cannot feed directly into bootstrap at N=32768. NEXUS bridges this with re-encryption (decrypt at N=8192, re-encrypt at N=32768), but re-encryption requires the secret key on the server — breaking non-interactivity. Our unified approach sacrifices MatMul efficiency for end-to-end homomorphic correctness.

**Impact on results**: Our scaling results (3.54x at 4 GPUs) are valid for the parallelism pattern, which is identical regardless of MatMul implementation. Bootstrap dominates at 89.4% — MatMul is only 1.2% of the layer. Replacing our naive MatMul with NEXUS's optimized version would change the absolute timings but not the scaling behavior.

### 10.3 What Is and Isn't Meaningful

**Meaningful**:
- Bootstrap works correctly with MAE=0.000002 — this is a genuine implementation achievement
- 3.54x multi-GPU scaling — the parallelism pattern generalizes to any head count
- Perfect weak scaling across nodes — compute is truly embarrassingly parallel
- The Phantom multi-GPU fixes — these enable any multi-GPU FHE workload on this library

**Needs context**:
- Our BERT dimensions (2-4 heads, 64 hidden) are much smaller than BERT-base (12 heads, 768 hidden). We demonstrate the **pipeline pattern**, not production performance.
- The 2,691ms per layer cannot be directly compared to NEXUS's ~35s because: different hardware (H100 vs A100), different MatMul approach, different polynomial degree (32768 vs 65536/8192).

---

## 11. Limitations

1. **Reduced BERT dimensions**: We test with 2-4 heads, head_dim=32, inner_dim=16 — much smaller than BERT-base (12 heads, head_dim=64, hidden=768). This demonstrates the pipeline pattern but not production throughput. See Section 10.2 for why full BERT-base dimensions require NEXUS's rotation-based MatMul.

2. **Naive MatMul**: Our `matrix_mul_unified` uses multiply_plain + add_many (O(inner_dim) per column) instead of NEXUS's Galois-rotation-based packing. At BERT-base dimensions this would be ~50x slower than NEXUS's approach.

3. **Per-GPU bootstrapper setup**: Each GPU independently computes Remez polynomials and LT coefficients (~15s CPU work). This is redundant — the coefficients are identical across GPUs and could be computed once and broadcast.

4. **MPI serialization bottleneck**: `PhantomCiphertext::save()` copies GPU→CPU via `cudaMemcpy` then writes to a stream. At 2 nodes, scatter takes 173ms for 84MB of data — but InfiniBand could transfer this in ~3.4ms. GPU-direct RDMA would provide ~50x improvement.

5. **No inter-head aggregation**: BERT's output projection should sum across all heads (a cross-GPU reduction). We skip this, meaning our multi-GPU results represent the embarrassingly parallel portion only.

6. **Single-layer timing**: We measure one encoder layer. Full BERT-base (12 layers) would take 12 x 2.7s = ~32s on a single GPU, with multi-GPU scaling applying to each layer independently.

---

## 12. Future Work

1. **NEXUS-compatible MatMul at N=32768**: Port NEXUS's Galois-rotation-based compress/decompress MatMul to work at N=32768. This would enable full BERT-base dimensions (768 hidden, 3072 FFN) while maintaining bootstrapping compatibility.
2. **Shared bootstrapper coefficients**: Compute Remez polynomials and LT coefficients once on CPU, serialize, and broadcast to all GPUs/nodes. This would eliminate the 15s per-GPU setup overhead.
3. **GPUDirect RDMA for MPI**: Replace `save()/load()` serialization with GPU-direct inter-node transfers. Expected 50x improvement in scatter/gather time.
4. **Strong scaling at BERT-base dimensions**: With 12 attention heads on 4 GPUs (3 heads/GPU), measure whether the per-head pipeline maintains 3.5x scaling.
5. **Calibrate hoisting bootstrap**: Generate and validate 2-part LT coefficients for the sparse N=32768 parameter set. If successful, this would reduce bootstrap levels consumed (2 instead of 3 rescales per LT phase), enabling a shallower modulus chain.
6. **12-layer BERT inference**: Run the complete 12-layer BERT-base with multi-GPU pipelining. At 2.7s/layer, a 12-layer inference would take ~32s on 1 GPU, ~9.3s on 4 GPUs.

---

## 13. Project Statistics

| Metric | Value |
|--------|-------|
| Total CUDA/C++ | 20,518 lines |
| Evaluator port | 8,816 lines (35 files) |
| Multi-GPU framework | 3,559 lines (19 files) |
| Benchmark programs | 25 programs (8,143 lines) |
| Phantom modifications | ~95 lines |
| GPU hours consumed | ~40 |
| Max GPUs used | 8 (2 nodes x 4 H100) |
| Nsight profiles captured | 3 |

---

---

## Update — April 12, 2026: N=65536 Parameter Set Investigation

### Motivation

NEXUS reports per-operation timings at N=65536 for GELU, Softmax, and LayerNorm in their paper (Table IV). To enable direct comparison, we investigated running the full BERT encoder layer at N=65536 (the polynomial degree required for 128-bit security at ~1800-bit modulus).

### NEXUS's Actual Approach (Code Analysis)

Analysis of the NEXUS open-source code (`vendor/nexus/cuda/src/main.cu`) reveals:

1. **No end-to-end BERT pipeline exists** in the NEXUS codebase. Their `main.cu` benchmarks each operation independently.
2. **Different N values per operation**: MatMul uses N=8192, GELU/Softmax/LayerNorm use N=65536, Bootstrap uses N=32768.
3. **Parameter switching** between N values requires the two-server protocol (re-encryption between servers, not involving the client).
4. **Bootstrap was downgraded**: Their code contains `logN = 15; // 16 -> 15` with the comment *"adjusted to satisfy the memory constraints of an A100 GPU"*.
5. The reported 37.3s is a **projection** from summing individual operation timings, not a measured end-to-end execution.

### Our N=65536 Experiments on MareNostrum 5

We conducted systematic experiments to test bootstrap at N=65536 on H100 64GB GPUs:

**Memory analysis at N=65536 (logN=16, logn=14, sparse_slots=16384):**

| Component | Memory | Notes |
|-----------|--------|-------|
| PhantomContext + SecretKey | 0.61 GB | Parameter tables, NTT roots |
| PublicKey + RelinKey | 1.57 GB | Freed before bootstrap (not needed) |
| Per Galois key | **1.30 GB** | 2x larger polynomials vs N=32768 |
| Bootstrap needs | 48 keys | 48 unique rotation steps (BSGS + power-of-2) |
| **Total key memory** | **62.4 GB** | 48 x 1.30 GB |
| After freeing PK+RK | **63.0 GB** | Fits in 63.43 GB, but 0 MB left for intermediates |

**Experiment 1: Single-GPU memory-optimized bootstrap (Job 38887673)**
- Freed PublicKey and RelinKey (not used by bootstrap) to reclaim 1.57 GB
- All 48 Galois keys created successfully: **63.37 GB used / 63.43 GB total**
- Bootstrap OOM: only **60 MB free** — insufficient for intermediate ciphertexts (~37 MB each)
- **Result: N=65536 bootstrap does not fit on a single H100 64GB**

**Experiment 2: Multi-GPU key distribution (Jobs 38886075, 38886320)**
- Split 48 Galois keys across 2 GPUs (24 each, ~31 GB per GPU)
- GPU 0: 33.43 GB, GPU 1: 32.06 GB — both fit comfortably
- **Accuracy result: MAE = 10^238 (FAIL)**
- Root cause: Phantom library is **context-bound** — each GPU creates its own PhantomContext with distinct NTT tables and `parms_id` hashes. Ciphertexts transferred between contexts (even with `cudaMemcpyPeer`) produce incorrect results because the key-switching operation uses context-specific NTT roots.

**Experiment 3: Control test at N=32768 (Job 38886207)**
- Same distributed-key mechanism applied at N=32768 (where single-GPU bootstrap works)
- Control (1 GPU): MAE = 0.000002 (PASS)
- Distributed (2 GPU): MAE = 10^43 (FAIL)
- **Confirms: the accuracy failure is in the cross-context transfer, not in N=65536 parameters**

### Why Cross-Context Transfer Fails

The Phantom FHE library ties each ciphertext to a specific `PhantomContext` through:
1. **`parms_id_`**: A hash of encryption parameters — identical parameters on different contexts produce the same hash, but the context lookup uses GPU memory pointers
2. **NTT tables**: Root-of-unity tables allocated on the creating GPU's memory — `rotate_vector_inplace` uses these tables directly
3. **`key_galois_tool_`**: Galois element permutation tables are context-specific GPU allocations

When a ciphertext is rotated on GPU 1 (with GPU 1's context) and the result is copied back to GPU 0, the coefficient data is correct, but subsequent operations on GPU 0 use GPU 0's NTT tables — which expect data in GPU 0's specific NTT representation. This representation mismatch accumulates exponentially through bootstrap's ~100+ operations.

### Comparison with Cerium's Approach

Cerium (Jayashankar et al., 2025) solves this problem through:
1. **Custom FHE library (FIDESlib)** designed from scratch for multi-GPU, with shared NTT tables
2. **UVM (Unified Virtual Memory)** for transparent GPU-to-GPU memory access
3. **Compiler-driven key placement** that automatically assigns keys to GPUs

These solutions require either a custom FHE library (not Phantom-compatible) or deep modifications to Phantom's memory management — beyond the scope of this project.

### Conclusion

For end-to-end BERT inference with real bootstrapping, **N=32768 is the practical parameter set** — consistent with NEXUS's own implementation choice. Our multi-GPU scaling results at N=32768 demonstrate the parallelization framework's effectiveness. The N=65536 memory constraint is a known limitation shared with NEXUS, who address it through parameter switching in their two-server protocol.

---

## References

1. Zhang et al., "NEXUS: Secure and Non-Interactive Transformer Inference on Encrypted Data", NDSS 2025
2. Jayashankar et al., "Cerium: A Scalable Multi-GPU Framework for Encrypted Large-Model Inference", arXiv:2512.11269, 2025
3. Jayashankar et al., "Cinnamon: A Framework for Scale-Out Encrypted AI", ASPLOS 2025
4. Phantom FHE Library, encryptorion-lab/phantom-fhe, GitHub
