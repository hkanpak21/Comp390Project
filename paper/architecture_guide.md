# Architecture Guide: What We Built On Top Of What

This document explains the complete system architecture for Halil's Comp 390 project. Read this to understand every component, what it does, where it came from, and what we modified.

> **Note (2026-05-11):** The Layer-3 `CtPipeline` / `MultiNodePipeline` infrastructure described below has been moved to `src/multi_gpu/archive/pipeline/`. The active multi-GPU paths are the per-thread `PhantomContext` pattern (HP-BERT, HP-LLaMA) and the DKS rotation infrastructure under `src/multi_gpu/keyswitching/`. The pipeline-parallel benchmarks that depended on `CtPipeline` (`bert_e2e_multigpu`, `bert_connected_*`, `bert_multinode_e2e`, etc.) are in `src/benchmarks/archive/`. See `docs/PER_OP_VS_NEXUS.md` for the current measurement strategy.

---

## Layer 0: The Math (CKKS FHE)

**What is it?** CKKS is a Fully Homomorphic Encryption scheme that lets you compute on encrypted floating-point numbers. A plaintext vector of complex numbers (e.g., 16,384 values) is encoded into a polynomial of degree N (e.g., 32,768), encrypted, and can then be added and multiplied without decryption.

**Key concepts:**
- **Ciphertext**: A pair of polynomials (c0, c1) representing encrypted data. Size: ~21 MB at N=32768, 37 moduli.
- **RNS (Residue Number System)**: Large coefficients are represented modulo many smaller primes (37 primes of 46-51 bits). This enables parallel modular arithmetic on GPU.
- **NTT (Number Theoretic Transform)**: Like FFT but over finite fields. Converts polynomials to evaluation form for O(n) multiplication. This is the #1 GPU kernel (~46% of time).
- **Key-switching**: After ct*ct multiply, the result has 3 polynomials; relinearization reduces back to 2 using evaluation keys. This is the #2 GPU kernel (~15% of time).
- **Levels**: Each multiply consumes one "level" (drops one RNS prime). When levels run out, you need bootstrapping.
- **Bootstrapping**: A homomorphic procedure that refreshes levels. It's the most expensive operation (~300ms per ciphertext) and consumes 14 levels itself.

---

## Layer 1: Phantom Library (Not Ours)

**What is it?** A GPU-native implementation of CKKS in CUDA. Source: `vendor/phantom/`

**Who wrote it?** encryptorion-lab (GitHub). NEXUS uses a **fork** of this library with API differences.

**What it provides:**
- `PhantomContext`: Manages encryption parameters, NTT tables, modulus chains
- `PhantomSecretKey/PublicKey/RelinKey/GaloisKey`: Cryptographic keys
- `PhantomCKKSEncoder`: Encode/decode vectors to/from polynomials
- `PhantomCiphertext`: Encrypted data container (GPU memory)
- Evaluate functions: `add_inplace`, `multiply_inplace`, `rotate_vector_inplace`, `relinearize_inplace`, `rescale_to_next_inplace`, `mod_switch_to_next_inplace`, etc.

**What we modified** (95 lines):
1. `ciphertext.h`: Added `save()/load()` — serialize ciphertext to/from streams (GPU↔CPU transfer) for MPI and multi-GPU
2. `secretkey.h`: Added `save()/load()` — serialize secret key for GPU-to-GPU distribution. Added default constructor.
3. `globals.h` + `context.cu`: Made `default_stream` `thread_local` — enables concurrent multi-GPU (each thread gets its own CUDA stream)
4. `cuda_wrapper.cuh`: Removed `cudaSetDevice(0)` from stream constructor — was hardcoded, broke multi-GPU
5. `evaluate.cu`: Commented out scale validation in `sub_inplace`, `multiply_plain_inplace`, `add_plain_inplace` — enables lazy rescaling needed by bootstrap

---

## Layer 2: NEXUS Evaluators (Ported by Us)

**What is it?** The FHE implementations of BERT operations. Source: ported from `vendor/nexus/cuda/src/` to `src/nexus_eval/`

**Who wrote the originals?** Zhang et al. (NEXUS, NDSS 2025). Their code was written for the NEXUS Phantom fork.

**What we ported** (8,816 lines):

### Core Wrapper (`ckks_evaluator.cuh/cu`, 1,039 lines)
Our CKKSEvaluator class wraps Phantom's raw functions into a cleaner API matching NEXUS's calling convention:
- `Encoder`: Wraps `PhantomCKKSEncoder` — handles vector/scalar/complex encoding, resizes to slot_count
- `Evaluator`: Wraps all evaluate functions — adds scale-matching before operations (NEXUS code assumes scales match implicitly)
- `Decryptor`: Wraps `PhantomSecretKey::decrypt` + selective Galois key generation
- `CKKSEvaluator`: The main class holding all components + advanced math (exp, inverse, invert_sqrt, sgn_eval)

### GELU (`gelu.cu`, 119 lines)
Homomorphic GELU activation: `GELU(x) = x * step(x)` where `step(x) = (1 + sign(x))/2`.
The sign function uses composite polynomial approximation (Minimax):
- 4 rounds of G4 polynomial (degree 9, range reduction)
- 2 rounds of F4 polynomial (degree 9, final sign)
Consumes 14 multiplicative levels.

### Softmax (`softmax.cu`, 45 lines)
`softmax(x) = exp(x) / sum(exp(x))` where:
- Rotate by -seq_len, compute exp via Taylor series
- Compute sum via rotation-and-add
- Divide via Newton's method (homomorphic inverse)
Consumes 16 levels.

### LayerNorm (`layer_norm.cu`, 41 lines)
`LN(x) = (x - mean) / sqrt(var)` where:
- Mean: rotate by -hidden_dim + add
- Variance: square(x - mean) + rotation-sum
- Normalize: Goldschmidt iteration for inverse square root
Consumes 16 levels.

### MatMul (`matrix_mul.cu`, 276 lines)
Two implementations:
1. **Original** (N=8192): Uses Galois rotations to compress/decompress packed ciphertexts. Complex but efficient for NEXUS's parameter set.
2. **Unified** (any N): Simple multiply_plain + add_many. No rotations needed. We use this for the bootstrap-compatible N=32768 parameter set.
Consumes 1 level.

### Bootstrapping (`bootstrapping/`, 7,047 lines, 24 files)
The most complex component. Copied directly from NEXUS (any modification broke accuracy):

```
bootstrap_sparse_3(output, input):
  1. modraise_inplace      — Expand from 1 modulus to 37 moduli (copy coefficients)
  2. subsum                — Rotate and add to consolidate sparse slots
  3. coefftoslot_3         — Linear transform: polynomial coefficients → slot values
     Uses 3-part Baby-Step Giant-Step (BSGS) decomposition
     Each part: rotate by baby-steps, multiply by precomputed coefficients, add
  4. modular_reduction     — Evaluate degree-59 Chebyshev polynomial to approximate
                             modular reduction (the core of bootstrapping)
  5. slottocoeff_3         — Linear transform: slot values → polynomial coefficients
     (Reverse of step 3)
```

The LT coefficients are precomputed on CPU using the Remez algorithm (NTL library) and encode the FFT butterfly pattern of the CKKS encoder. **These coefficients must match the exact FFT layout of the Phantom fork** — which is why we had to use the NEXUS fork.

---

## Layer 3: Multi-GPU Framework (Built by Us)

**What is it?** Infrastructure to distribute FHE operations across multiple GPUs and nodes. Source: `src/multi_gpu/` (3,559 lines)

### CtPipeline (`pipeline/ct_pipeline.cu`, 227 lines)
Distributes independent ciphertexts across GPUs via round-robin assignment:
1. `create()`: Creates per-GPU PhantomContext + keys from serialized secret key
2. `scatter()`: Copies ciphertexts to target GPUs via `cudaMemcpyPeer`
3. `execute(fn)` / `execute_full(fn)`: Runs a user function on each GPU's local ciphertexts in parallel threads
4. `gather()`: Copies results back to GPU 0

### Multi-Node Pipeline (`pipeline/multi_node_pipeline.cu`, 134 lines)
MPI wrapper: rank 0 encrypts all ciphertexts, scatter via `MPI_Send/Recv`, each rank runs CtPipeline locally, gather results back.

### Per-Thread Context Pattern (in benchmarks)
For bootstrapping, CtPipeline's shared-context model breaks due to `default_stream`. Our solution: each GPU thread in `bert_encoder_multigpu.cu` creates its own `PhantomContext`, keys, encoder, and `Bootstrapper` instance. This is the cleanest multi-GPU approach.

### Keyswitching Algorithms (`keyswitching/`, ~700 lines)
Two multi-GPU key-switching implementations from the Cinnamon paper:
- **Input Broadcast (IB)**: AllGather full ciphertext, local key-switch, discard redundant
- **Output Aggregation (OA)**: Partition RNS limbs, partial inner products, AllReduce

These achieve only 1.08x due to Amdahl's Law (75% of key-switch is non-distributable `modup`).

---

## Layer 4: Benchmarks (Built by Us)

25 benchmark programs in `src/benchmarks/` (8,143 lines). The key ones:

| Program | What it measures |
|---------|-----------------|
| `bootstrap_test.cu` | Standalone bootstrap accuracy + timing |
| `bert_encoder_layer.cu` | Complete single-GPU BERT layer with 4x bootstrap |
| `bert_encoder_multigpu.cu` | Multi-GPU BERT with per-thread contexts |
| `bert_encoder_multinode.cu` | Multi-node BERT with MPI + per-GPU threading |
| `multi_gpu_keyswitch_test.cu` | IB and OA key-switching correctness |
| `nccl_bandwidth_test.cu` | NCCL collective bandwidth measurement |

---

## Data Flow: From Input to Output

```
User's plaintext vector (e.g., [0.5, -0.3, 0.8, ...])
    |
    v
[1] Encode: FFT-like transform, scale by 2^46, produce polynomial coefficients
    |
    v
[2] Encrypt: Add noise polynomial, produce (c0, c1) pair on GPU
    |                                                              ← All on GPU
    v                                                                from here
[3] Mod-switch past bootstrap levels (drop 14 moduli)
    |
    v
[4] MatMul Q/K/V: multiply_plain(ct, weight_pt) + add_many → Q, K, V ciphertexts
    |
    v
[5] QK^T: multiply(Q, K) → attention scores (ct x ct, needs relinearization)
    |
    v
[6] Softmax: exp(scores) / sum(exp(scores)) — uses 16 levels
    |
    v
[7] Attention * V: multiply(softmax_out, V) → attention output
    |
    v
[8] Output projection: multiply_plain + add_many
    |
    v                                                    Levels exhausted!
[9] BOOTSTRAP: modraise → subsum → coeff2slot → mod_reduction → slot2coeff
    |                                                    Levels restored!
    v
[10] LayerNorm: (x - mean) * invsqrt(var) — uses 16 levels
    |
    v                                                    Levels exhausted!
[11] BOOTSTRAP
    |                                                    Levels restored!
    v
[12] FFN: MatMul up → GELU → MatMul down
    |
    v                                                    Levels exhausted!
[13] BOOTSTRAP
    |
[14] LayerNorm → BOOTSTRAP
    |
    v
[15] Output ciphertext (ready for next encoder layer, or decrypt)
    |
    v
[16] Decrypt + Decode → plaintext result
```

---

## What Came From Where — Summary

| Component | Origin | Our Contribution |
|-----------|--------|-----------------|
| CKKS scheme | Academic (Cheon et al., 2017) | None (math) |
| Phantom library | encryptorion-lab (GitHub) | +95 lines (serialization, multi-GPU) |
| NEXUS evaluators | Zhang et al. (NDSS 2025) | Ported 8,816 lines to our Phantom fork |
| Bootstrapping code | NEXUS (copied exactly) | 0 modifications (accuracy-critical) |
| CtPipeline | **Our design** | 361 lines (pipeline/ct_pipeline.*) |
| Multi-node MPI | **Our design** | 134 lines |
| Per-thread context | **Our design** | Pattern in benchmarks |
| IB/OA key-switching | Cinnamon paper (reimplemented) | ~700 lines |
| All 25 benchmarks | **Our code** | 8,143 lines |
| Phantom multi-GPU fixes | **Our discovery + fix** | Thread-local stream, save/load |
| Bootstrap accuracy fix | **Our debugging** | Identified FFT mismatch, switched forks |

**Bottom line**: We built the parallelization infrastructure (pipeline, MPI, multi-GPU context management) and wired it into the existing NEXUS+Phantom FHE stack. The FHE algorithms themselves (GELU, Softmax, LayerNorm, Bootstrap) are NEXUS's — we ported them and made them work in a multi-GPU/multi-node setting.
