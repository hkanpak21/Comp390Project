# Secure and privacy-preserving LLM inference: a complete research roadmap

**Extending NEXUS to multi-GPU and multi-node execution is technically feasible and would represent a genuine contribution to the field.** No existing work has adapted a non-interactive FHE-based transformer inference protocol for distributed GPU execution using NCCL across HPC-class hardware. NEXUS currently runs on up to four A100 GPUs within a single node and achieves **37.3-second BERT-base inference** with only **164 MB of bandwidth**—orders of magnitude less communication than interactive alternatives. Meanwhile, Cerium has demonstrated that multi-GPU FHE inference can scale to **Llama3-8B in 134 seconds** using compiler-driven parallelization across multiple GPUs. The gap between these two systems—NEXUS's elegant non-interactive protocol and Cerium's multi-GPU infrastructure—defines the opportunity for this project. This report maps every dimension of the research landscape the student will need to navigate: the papers, the cryptographic machinery, the scaling techniques, the software stack, and the hardware platforms.

---

## 1. The academic landscape divides into two competing paradigms

The field of secure transformer inference has converged on two fundamentally different approaches: **interactive protocols** using hybrid HE+MPC, and **non-interactive protocols** using pure FHE. Understanding both is essential because NEXUS's core advantage—eliminating interaction—directly enables the GPU acceleration that makes multi-GPU scaling worthwhile.

### Non-interactive (pure FHE) systems

**NEXUS** (Zhang et al., NDSS 2025) is the first non-interactive protocol for secure transformer inference. The client encrypts input using RNS-CKKS (polynomial degree N'=2^16, **1763-bit ciphertext modulus**, 128-bit security, multiplicative depth L=35, bootstrapping depth K=14) and sends it to the server, which evaluates the entire transformer homomorphically and returns one encrypted result. Key innovations include **SIMD ciphertext compression/decompression** (extending SealPIR's technique to matrix multiplication, reducing m×n ciphertexts to m×n/N' ciphertexts), **SIMD slot folding** for efficient matrix operations, and an efficient **argmax** that reduces O(m) sign operations to O(log m + 1). On four Tesla A100 GPUs, NEXUS completes BERT-base inference in **37.3 seconds** with **164 MB bandwidth**—a **372.5× bandwidth reduction** over BOLT and **53.6×** over BumbleBee. The amortized cost across 256 inputs drops to **1.31 seconds per input**. At AWS pricing, this translates to roughly **$0.05 per token** versus BOLT's $5.44. The implementation uses modified Microsoft SEAL 4.1 for CPU and the **Phantom FHE library** for GPU acceleration. Code is publicly available at `github.com/zju-abclab/NEXUS`.

**Cerium** (Jayashankar et al., CMU/NVIDIA, arXiv 2512.11269, December 2025) is the state-of-the-art multi-GPU FHE framework and the most relevant system for this project. It combines a Python-embedded DSL, an optimizing compiler, and a runtime system to automatically generate fused GPU kernels and parallelize FHE programs across multiple GPUs. Cerium achieves **7.5 ms bootstrapping** (first to break the 10 ms barrier), **8.8 seconds for BERT-Base**, and **134 seconds for Llama3-8B**—the first-ever encrypted inference of an 8B-parameter LLM on GPUs. Its sparse polynomial representation compresses Llama3-8B's weight footprint from **112 TB to 982 GB** (a 119× reduction), and memory-reuse analysis reduces capacity requirements by over **100×**. The system matches the performance of the CraterLake FHE ASIC on commodity hardware. Code has not yet been publicly released as of March 2026.

**EncryptedLLM** (De Castro et al., ICML 2025) provides a GPU-accelerated CKKS implementation as an open-source extension to OpenFHE. It benchmarks an encrypted GPT-2 forward pass with runtimes **over 200× faster** than the CPU baseline and achieves bootstrapping of 20 ciphertext levels in approximately **550 ms** at 128-bit security. While limited to a single GPU and GPT-2, its open-source GPU extensions to OpenFHE may be useful building blocks.

### Interactive (hybrid HE+MPC) systems

**Iron** (Hao et al., NeurIPS 2022) was the first dedicated work on private transformer inference, using hybrid additively homomorphic encryption (BFV) with 2-out-of-2 secret sharing. It achieves 3–14× less communication and 3–11× less runtime than prior CNN-focused systems, but BERT-base still requires approximately **280.99 GB of communication** and **216 minutes** under WAN conditions. Non-linear functions dominate at **76–84% of computation time**, establishing the key bottleneck that subsequent work targets.

**BOLT** (Pang et al., IEEE S&P 2024) improves on Iron with communication-optimized matrix multiplication using a baby-step giant-step strategy that reduces HE rotations by **2.33–9.33×** for BERT-base dimensions. It achieves **4.8–9.5× speedup** over Iron across network settings and **10.91× less communication**. However, it still requires **10,509 interaction rounds** and **59.61 GB bandwidth**, making GPU acceleration difficult due to round-trip latencies. Code is available at `github.com/Clive2312/BOLT`.

**BumbleBee** (Lu et al., NDSS 2025) uses homomorphic automorphism operations for communication-efficient matrix multiplication and was the first 2PC framework evaluated on **LLaMA-7B**, generating one token in approximately **14 minutes** on CPUs. It reduces communication by **80–90%** over prior HE-based protocols. Code is at `github.com/AntCPLab/OpenBumbleBee`.

**PUMA** (Dong et al., 2023/2025) operates in a 3-party MPC setting using replicated secret sharing and was the first MPC framework to evaluate **LLaMA-7B**, achieving approximately **5 minutes per token**. Critically, PUMA works with pre-trained HuggingFace models without retraining, thanks to high-quality polynomial approximations for GeLU and Softmax. Code is at `github.com/AntCPLab/puma_benchmarks`.

**SIGMA** (Gupta et al., PETS 2024) introduces function secret sharing (FSS) for transformer inference and achieves **23 seconds per token** for LLaMA-2-7B and **38 seconds** for LLaMA-2-13B on A100 GPUs—the fastest MPC-based results for large models. It is **11–19× faster** than prior state-of-the-art and supports both CPU and GPU execution.

**SHAFT** (Kei & Chow, NDSS 2025) introduces the first constant-round softmax protocol using ODE-based input clipping, achieving **6.7× faster than BOLT** on LAN and **82% communication reduction**. Code is at `github.com/andeskyl/SHAFT`.

Additional important works include **SecFormer** (ACL Findings 2024), which achieves **3.6× speedup** over PUMA on BERT through combined model design and protocol optimization; **MPCFormer** (ICLR 2023 Spotlight), which introduced knowledge distillation for MPC-friendly approximations with **2.2–5.9× speedups**; **Cheetah** (USENIX Security 2022), which developed rotation-free HE protocols for linear layers; **CipherGPT** (ePrint 2023), the first end-to-end secure GPT inference including a novel top-k sampling protocol; **Primer** (ICCAD 2023), which achieves **90–97.5% latency reduction** through computation merging and tokens-first packing; and **Piranha** (USENIX Security 2022), the first general GPU platform for secret-sharing MPC that achieves **16–48× speedups** over CPU implementations.

### Consolidated performance comparison

| System | Year/Venue | Approach | Largest model | Latency | Communication | Interactive? |
|--------|-----------|----------|---------------|---------|---------------|-------------|
| **NEXUS** | NDSS '25 | Pure FHE (CKKS) | BERT-base | 37.3s (GPU) | 164 MB | **No** |
| **Cerium** | arXiv '25 | Pure FHE (CKKS) + Multi-GPU | Llama3-8B | 134s (GPU) | N/A | **No** |
| **EncryptedLLM** | ICML '25 | Pure FHE (CKKS) | GPT-2 | Minutes (GPU) | N/A | **No** |
| **SIGMA** | PETS '24 | 2PC FSS | LLaMA-2-13B | 38s/tok (GPU) | Moderate | Yes |
| **SHAFT** | NDSS '25 | 2PC SS | BERT/GPT-2 | ~174s WAN | Low | Yes |
| **BOLT** | S&P '24 | Hybrid BFV+MPC | BERT-base | 4.8–9.5× Iron | ~25.74 GB | Yes (10K rounds) |
| **BumbleBee** | NDSS '25 | Hybrid BFV+MPC | LLaMA-7B | ~14 min/tok | 90% < Iron | Yes |
| **PUMA** | S&S '25 | 3PC RSS | LLaMA-7B | ~5 min/tok | Moderate | Yes |
| **Iron** | NeurIPS '22 | Hybrid AHE+SS | BERT-large | ~216 min WAN | ~281 GB | Yes |

---

## 2. Cryptographic primitives and how they map to GPU execution

### FHE schemes: CKKS dominates for ML workloads

Three RLWE-based FHE schemes are used in the literature. **BFV** (Brakerski/Fan-Vercauteren) performs exact integer arithmetic over Z_t, using a fixed ciphertext modulus with noise managed through relinearization. It is used by Iron, BOLT, Cheetah, and BumbleBee for their HE components, primarily because exact arithmetic avoids approximation errors in linear layers that feed into MPC-based non-linear evaluations.

**BGV** (Brakerski-Gentry-Vaikuntanathan) also performs exact integer arithmetic but uses modulus switching for noise management—after each multiplication, the modulus is reduced from Q_l to Q_{l-1}, scaling noise proportionally. HElib implements BGV with bootstrapping, and OpenFHE supports it alongside BFV and CKKS.

**CKKS** (Cheon-Kim-Kim-Song) performs approximate arithmetic over real/complex numbers, treating noise as part of the message with configurable precision. This scheme dominates ML-focused FHE work (NEXUS, Cerium, EncryptedLLM, Orion) for three reasons: it natively encodes real numbers without quantization, its rescaling operation is far cheaper than bootstrapping for managing multiplicative depth, and it provides **N/2 SIMD slots** per ciphertext for maximum throughput. NEXUS uses RNS-CKKS with N'=2^16 (yielding N=2^15 = 32,768 slots), a 1763-bit modulus chain, and 35 multiplicative levels. The scale propagation technique eliminates dominant noise, achieving sufficient precision for BERT-class models.

**Parameter selection** directly impacts performance and security. The Homomorphic Encryption Standard specifies that for **128-bit security**: N=8192 supports max log₂(Q)=218; N=16384 supports 438; N=32768 supports 881; **N=65536 supports 1770**. NEXUS uses N'=65536 with a 1763-bit modulus, right at the security boundary. A single CKKS ciphertext at N=65536 with L=45 RNS limbs occupies approximately **48 MB**. Evaluation keys (relinearization and rotation keys) can total hundreds of GB for deep networks—a critical constraint for GPU memory.

### The four computational bottlenecks in FHE

**Number Theoretic Transform (NTT)** is the finite-field analog of FFT, converting polynomials between coefficient and evaluation representations. Every polynomial multiplication requires forward NTT on both operands, pointwise multiplication, and inverse NTT—each costing O(N log N) modular multiply-adds. For RNS-CKKS with L levels, each polynomial is represented as L+1 residue polynomials, so a single multiplication involves (L+1) NTT/INTT pairs. At N=65536 with L=30, this amounts to roughly **34 million operations** per multiplication. NTT accounts for **over 70% of bootstrapping execution time** and is the primary target for GPU acceleration.

**Key-switching** converts ciphertexts between different encryption keys and is required after every relinearization (post ciphertext-ciphertext multiplication) and every rotation. It involves gadget decomposition into dnum components, modulus raising from Q to P·Q, inner product with key-switching keys, and modulus reduction. Each step requires multiple NTT/INTT operations. Key-switching dominates **70–80% of total FHE execution time** in CKKS bootstrapping, making it the single most important operation to optimize for multi-GPU scaling.

**Bootstrapping** refreshes ciphertext noise by homomorphically evaluating the decryption circuit. In CKKS, it proceeds through ModRaise, CoeffToSlot (homomorphic DFT), EvalMod (Chebyshev polynomial evaluation), and SlotToCoeff (inverse DFT). It consumes 10–15 multiplicative levels itself and requires numerous rotation operations. CPU-based bootstrapping takes hundreds of seconds; Cerium achieves **7.5 ms on GPU**, a transformative improvement.

**Ciphertext matrix multiplication** is the core linear operation. The diagonal method (Halevi-Shoup) requires O(d) rotations for a d×d matrix-vector product. The baby-step giant-step optimization reduces this to O(√d) rotations at the cost of O(√d) additional ciphertext storage. NEXUS's SIMD compression/decompression technique further reduces the number of ciphertexts involved, while BOLT's compact packing eliminates wasted slots.

### GPU acceleration maps these bottlenecks to CUDA parallelism

NTT maps naturally to GPU execution because each butterfly stage contains N independent operations, RNS decomposition gives L independent NTTs, and pointwise operations are embarrassingly parallel. For large N (2^16–2^22), the **4-step NTT** arranges coefficients as an n₁×n₂ matrix, computes row NTTs, applies twiddle factors, transposes, and computes column NTTs. The transpose step is the key bottleneck due to non-coalesced memory access. Recent work achieves **123× speedup** over CPU and **2.37×** over prior GPU NTT implementations through 2D mixed-radix approaches with shared memory caching.

**Tensor Core utilization** for FHE is an active research frontier. **TensorFHE** (HPCA 2023) maps modular polynomial multiplication to 8-bit integer Tensor Core operations, decomposing 64-bit modular arithmetic into multiple 8×8 matrix-multiply-accumulate instructions. **WarpDrive** (HPCA 2025) concurrently utilizes CUDA Cores and Tensor Cores within NTT operations, achieving **73% reduction in instructions** and **86% reduction in pipeline stalls**. The A100 provides **312 TOPS** INT8 Tensor Core throughput versus ~19.5 TFLOPS FP64 on CUDA Cores, making this decomposition worthwhile despite overhead.

Key-switching parallelization leverages RNS-level parallelism (each CRT prime's NTT runs independently on separate CUDA streams), dnum-level parallelism (decomposition components processed in parallel), and pipeline overlap between ModRaise, NTT, inner product, and ModDown stages. The primary challenge is that evaluation keys often exceed GPU memory—at N=32768 with L=30, rotation keys for practical workloads can require **tens to hundreds of GB**. Solutions include streaming keys from pinned host memory using `cudaMemcpyAsync` with overlapped computation.

---

## 3. Multi-GPU scaling requires FHE-specific parallelism strategies

### Cerium's approach is the template for this project

Cerium employs an FHE-domain-specific parallelism strategy combining two orthogonal levels. **Limb-level parallelism** distributes the RNS limbs of each ciphertext across GPUs: for n GPUs, GPU c processes limbs {q_i | i mod n = c}. With N=64K and a 1782-bit modulus decomposed into ~64 limbs using a 28-bit RNS basis, 8 GPUs each handle ~8 limbs. **Program-level parallelism** distributes independent ciphertext operations (e.g., bootstrapping different ciphertexts in parallel) across GPU groups. This is fundamentally different from standard ML tensor or pipeline parallelism.

The critical innovation is in **parallelized key-switching**, which accounts for 70–80% of FHE runtime. Cerium implements two algorithms from Cinnamon (ASPLOS 2025, same research group):

**Input Broadcast**: All limbs of the input polynomial C_Q are broadcast to every GPU at the start of key-switching. Each GPU then independently performs digit splitting, mod-up (base conversion to Q_c ∪ E), multiplication with evaluation keys, and aggregation. Because extension basis limbs are duplicated during mod-up, no further communication is needed for mod-down. This trades compute/storage duplication for **a single broadcast** instead of three communication phases.

**Output Aggregation**: No initial broadcast is needed. The number of digits d equals the number of GPUs n. Each GPU processes its local digit (mod-up, evalkey multiplication, mod-down), and a final **reduce-scatter** operation aggregates and distributes partial sums. This requires **2 aggregation operations** total.

Cerium uses **NCCL** for all inter-GPU communication: `ncclBroadcast` for input broadcast, `ncclAllReduce`/`ncclReduceScatter` + `ncclAllGather` for output aggregation. Each ciphertext transfer moves approximately **32 MB** (2 polynomials × 64 limbs × 256 KB per limb). On NVLink (900 GB/s bidirectional on B200), this takes ~35–70 μs per operation, but the synchronization latency—not raw bandwidth—is the dominant overhead.

### Scaling results reveal the critical role of compiler optimization

Without Cerium's scheduling optimizations, multi-GPU bootstrapping on 8 B200 GPUs is actually **1.2× slower** than single-GPU (17.4 ms vs 14.5 ms)—communication overhead dominates. Cerium's compute-communication overlapping via separate CUDA streams recovers this to **1.45× faster** (10.0 ms), and communication minimization passes achieve a further **1.18× improvement** to **7.5 ms** (1.93× total speedup on 8 GPUs). The compiler achieves a **44% reduction in communicated bytes** over the Cinnamon baseline. These results demonstrate that naive multi-GPU distribution of FHE operations will degrade performance; sophisticated overlapping and communication minimization are mandatory.

For end-to-end inference, scaling is better because larger models provide more program-level parallelism. BERT-Base achieves **9.12× speedup** over prior single-GPU FHE implementations. Llama3-8B was only run in the multi-GPU configuration (it cannot fit on a single GPU), demonstrating **134 seconds** total.

### Communication patterns and CUDA stream overlapping

The relevant NCCL collectives for FHE are:

- **AllReduce**: Aggregates partial evaluation key products across GPUs in output aggregation key-switching
- **ReduceScatter**: Maps directly to Cinnamon's "aggregate and scatter"—more efficient than AllReduce when each GPU only needs its own limb partition
- **AllGather/Broadcast**: Distributes all limbs to all GPUs in input broadcast key-switching

CUDA stream overlapping is implemented by running NCCL operations on dedicated communication streams while FHE kernels (NTT, polynomial multiply, base conversion) execute on compute streams. The compiler analyzes data dependencies and reorders independent compute kernels to fill communication gaps. **CudaGraphs** are used to minimize kernel launch overhead across thousands of FHE kernels by creating reusable computation graphs (creation takes <30s even for Llama3-8B).

### Other multi-GPU FHE work

**Cinnamon** (ASPLOS 2025) is Cerium's predecessor, simulating custom ASIC chiplets. It demonstrated BERT inference in **1.67 seconds** on simulated hardware and achieved a **32× reduction** in required inter-chip bandwidth (16 TB/s → 512 GB/s). **ArctyrEX** (NVIDIA Research, PoPETS 2025) accelerates CGGI/TFHE-based encrypted execution across 8 A100 GPUs with **near-linear scaling** for wide circuits. **REDcuFHE** (NDSS 2023) implements multi-GPU TFHE with a dynamic scheduler distributing gate evaluations. **Al Badawi et al.** (IEEE TPDS 2020) demonstrated multi-GPU BFV on K80/P100 clusters with **1–3 orders of magnitude speedup** over CPU.

---

## 4. The software stack: libraries, profiling, and communication

### FHE libraries span a spectrum from CPU-only to GPU-native

| Library | Language | GPU? | Schemes | Status | Used by |
|---------|----------|------|---------|--------|---------|
| **Microsoft SEAL** | C++17 | No (Intel HEXL) | BFV, BGV, CKKS | Active (v4.1) | NEXUS (CPU path) |
| **OpenFHE** | C++ | Extensions only | BFV, BGV, CKKS, TFHE | Very active (v1.5.0) | EncryptedLLM |
| **Phantom** | CUDA C++ | **Yes (native)** | BFV, BGV, CKKS | Active | **NEXUS (GPU path)** |
| **HEonGPU** | CUDA C++ | **Yes (native)** | BFV, CKKS, TFHE | Active | — |
| **CAT** | CUDA C++ | **Yes (native)** | BFV, BGV, CKKS | Active | Up to 2173× over prior GPU |
| **Lattigo** | Go | No | BFV, BGV, CKKS | Active (v6) | Multiparty HE research |
| **HElib** | C++17 | No | BGV, CKKS | Low activity | Early FHE research |
| **TFHE-rs** | Rust | **Yes (v0.5+)** | TFHE | Very active (v1.5) | Zama ecosystem |
| **Concrete/Concrete ML** | Python/Rust | **Yes (v2.7+)** | TFHE | Very active (v2.10) | HuggingFace demos |
| **TenSEAL** | C++/Python | No | BFV, CKKS (via SEAL) | Moderate | PyTorch integration |
| **cuFHE** | CUDA C++ | **Yes** | TFHE only | Legacy | — |

**For this project, Phantom is the critical library** since NEXUS's GPU implementation depends on it. Phantom (`github.com/encryptorion-lab/phantom-fhe`) provides CUDA-accelerated BFV/BGV/CKKS operations under GPLv3. Build with CMake specifying `CMAKE_CUDA_ARCHITECTURES` (80 for A100, 90 for H100). The NEXUS repository includes a `cuda/` directory with GPU-accelerated inference built on Phantom.

### Profiling tools for identifying FHE bottlenecks

**NVIDIA Nsight Systems** provides system-wide timeline views showing CUDA API calls, kernel launches, memory copies, NVLink bandwidth, and NCCL collective timings. For multi-GPU FHE profiling, launch with: `nsys profile --trace=cuda,nvtx,nccl --output=fhe_profile_%h_%p ./your_fhe_binary`. Use NVTX annotations in code to mark FHE operations (NTT, key-switching, bootstrapping). The timeline view reveals idle gaps between compute and communication, guiding overlap optimization.

**NVIDIA Nsight Compute** provides deep kernel-level analysis including roofline analysis (compute vs. memory bound), occupancy analysis, and source-line correlation. For NTT kernels specifically, check memory bandwidth utilization (NTT is typically memory-bound), L2 cache hit rates for butterfly operations, and warp scheduling efficiency. Launch with: `ncu --set full -k regex:ntt.* -o ntt_profile ./your_fhe_binary`. The recommended workflow is Nsight Systems first (identify hot kernels) → Nsight Compute (deep kernel optimization).

### Communication stack for multi-node execution

**NCCL** (v2.29+) is the primary inter-GPU communication library, supporting AllReduce, AllGather, ReduceScatter, Broadcast, and point-to-point Send/Receive. It auto-detects NVLink, NVSwitch, InfiniBand, and PCIe topologies. For pure CUDA/C++ FHE code (not PyTorch), initialize NCCL directly via `ncclCommInitRank()` after broadcasting a unique ID through MPI.

**On AWS**, NCCL communicates through EFA via the `aws-ofi-nccl` plugin, which maps NCCL transport APIs to libfabric's reliable interface. Key environment variables: `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1` (for GPUDirect RDMA on p4d/p5), `NCCL_PROTO=simple`. The plugin is pre-installed on Deep Learning AMIs.

**On MareNostrum 5**, NCCL communicates through InfiniBand NDR200 via ConnectX-7 adapters. Key variables: `NCCL_IB_HCA=mlx5`, `NCCL_IB_CUDA_SUPPORT=1`, `NCCL_SOCKET_IFNAME=ib0`. MPI (OpenMPI or Intel MPI) is used for process launching via SLURM's `srun`.

### Open-source repositories for immediate use

- **NEXUS**: `github.com/zju-abclab/NEXUS` — C++ with modified SEAL 4.1 + Phantom GPU. The `cuda/` directory contains the GPU implementation.
- **BOLT**: `github.com/Clive2312/BOLT` — Built on EzPC framework
- **BumbleBee**: `github.com/AntCPLab/OpenBumbleBee` — Built on SPU library
- **SHAFT**: `github.com/andeskyl/SHAFT` — Built on CrypTen, Python/PyTorch
- **PUMA**: `github.com/AntCPLab/puma_benchmarks` — MPC benchmarks
- **SIGMA**: `github.com/mpc-msri/EzPC` — FSS-based protocols
- **Phantom FHE**: `github.com/encryptorion-lab/phantom-fhe` — GPU FHE library
- **CrypTen**: `github.com/facebookresearch/CrypTen` — Meta's PyTorch MPC framework
- **Piranha**: `github.com/ucbrise/piranha` — GPU MPC platform

---

## 5. Compute environments: AWS EC2 and MareNostrum 5

### AWS EC2 GPU instances

| Feature | p4d.24xlarge | p4de.24xlarge | p5.48xlarge |
|---------|-------------|---------------|-------------|
| GPUs | 8× A100 **40GB** | 8× A100 **80GB** | 8× H100 **80GB** |
| GPU interconnect | NVSwitch, 600 GB/s/GPU | NVSwitch, 600 GB/s/GPU | NVSwitch, **900 GB/s/GPU** |
| vCPUs / RAM | 96 / 1,152 GiB | 96 / 1,152 GiB | 192 / 2,048 GiB |
| Network (EFA) | 400 Gbps (4×100G) | 400 Gbps (4×100G) | **3,200 Gbps** (EFA v2) |
| On-demand price | ~$32.77/hr | ~$40.97/hr | ~$55.04/hr |
| Spot price | ~$4.41/hr | ~$27.45/hr | ~$10.52/hr |

**For FHE workloads, the p4de or p5 with 80 GB per GPU is strongly recommended.** A single CKKS ciphertext at N=2^16 with L=45 limbs consumes ~48 MB, and evaluation keys can total multiple GB. The 40 GB A100 on p4d may be insufficient for deeply parameterized circuits. The H100's higher HBM bandwidth (**3.35 TB/s** vs A100's 2.0 TB/s) is particularly valuable because FHE operations are memory-bandwidth-bound.

For multi-node clusters, use **AWS ParallelCluster** with SLURM scheduling, EFA-enabled networking, and placement groups for minimal latency. The Deep Learning Base GPU AMI comes pre-installed with CUDA, NCCL, aws-ofi-nccl, OpenMPI, and libfabric/EFA. Run NCCL benchmarks first: expected bus bandwidth is ~560 GB/s intra-node (NVSwitch) and ~40–50 GB/s inter-node (400 Gbps EFA on p4d) or **~300+ GB/s** inter-node on p5 with 3,200 Gbps EFA v2.

### MareNostrum 5 accelerated partition

MareNostrum 5's GPU partition comprises **1,120 nodes** with **4× NVIDIA H100 64GB** per node (4,480 total GPUs), **2× Intel Xeon Platinum 8460Y+ (80 cores/node)**, 512 GB DDR5 RAM, and InfiniBand **NDR200** (200 Gb/s per GPU via ConnectX-7 adapters). The network uses a three-layer **fat-tree** topology with 324 QM9790 switches; islands of 160 nodes have full fat-tree (no contention), with 1:2 contention between islands. Storage provides **248 PB** on GPFS with 1.6 TB/s read throughput.

A critical difference from AWS: MN5 has **4 GPUs per node** versus 8 on AWS, so multi-node communication is needed sooner. However, each GPU has a **direct 200 Gb/s InfiniBand connection** supporting GPUDirect RDMA, and the system is free via EuroHPC allocation.

Example SLURM job script for multi-node FHE:
```bash
#!/bin/bash
#SBATCH --job-name=fhe-multi-gpu
#SBATCH --account=your_project_account
#SBATCH --qos=acc_bsccase
#SBATCH --nodes=4 --ntasks-per-node=4 --cpus-per-task=20
#SBATCH --gres=gpu:4 --time=04:00:00 --exclusive

module load cuda nccl openmpi
export NCCL_IB_HCA=mlx5
export NCCL_IB_CUDA_SUPPORT=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
srun ./your_fhe_application
```

Access is through **EuroHPC Development Access** (continuously open, ~2 weeks approval) for initial testing, scaling to Regular or Extreme Scale access for production runs. Apply via `access.eurohpc-ju.europa.eu`.

### Platform comparison for this project

| Factor | AWS p5 (2 nodes = 16 H100) | MN5 ACC (4 nodes = 16 H100) |
|--------|---------------------------|----------------------------|
| Cost for 24 hours | ~$2,640 on-demand / ~$505 spot | Free (academic allocation) |
| Intra-node GPU bandwidth | 900 GB/s (NVSwitch, 8 GPUs) | Likely NVLink/PCIe (4 GPUs) |
| Inter-node bandwidth | 3,200 Gbps (EFA v2) | 800 Gbps (4×200G NDR200) |
| Allocation timeline | Immediate (credit card) | Weeks–months |
| Best for | Fast iteration, high inter-node BW | Extended experiments, free |

**Recommended strategy**: Develop and debug on AWS using spot instances, then run production experiments on MareNostrum 5 for cost efficiency.

---

## 6. Reported results and benchmarks across the literature

### BERT-base inference latency (the standard benchmark)

| System | Hardware | Total latency | Communication | Notes |
|--------|----------|--------------|---------------|-------|
| **Cerium** | Multi-GPU (B200) | **8.8 s** | N/A (pure FHE) | 9.12× over prior GPU FHE |
| **NEXUS** | 4× A100 GPU | **37.3 s** | 164 MB | Non-interactive, amortized 1.31s |
| **SHAFT** | CPU (32 threads) | ~174 s (WAN) | 82% < BOLT | Constant-round softmax |
| **BOLT** | CPU (32 threads) | ~22–45 min (WAN) | ~25.74 GB | 10,509 rounds |
| **Iron** | CPU | ~216 min (WAN) | ~280.99 GB | Baseline for comparisons |

### Large language model inference

| System | Model | Hardware | Per-token latency | Communication |
|--------|-------|----------|-------------------|---------------|
| **Cerium** | Llama3-8B | Multi-GPU (B200) | **134 s** | N/A (pure FHE) |
| **SIGMA** | LLaMA-2-7B | A100 GPU | **23 s** | Moderate |
| **SIGMA** | LLaMA-2-13B | A100 GPU | **38 s** | Moderate |
| **PUMA** | LLaMA-7B | CPU (3PC) | ~5 min | Moderate |
| **BumbleBee** | LLaMA-7B | CPU (2PC) | ~14 min | 90% < Iron |

### GPU acceleration speedups

Cerium achieves **bootstrapping in 7.5 ms** on multi-GPU versus hundreds of seconds on CPU—approximately **four orders of magnitude improvement**. NEXUS's GPU version is **42.3× faster** than its CPU version. EncryptedLLM reports **150–200×** over CPU OpenFHE. Piranha achieves **16–48×** for MPC training on GPUs. These results consistently show that GPU acceleration transforms FHE from impractical to borderline-feasible for real workloads.

### Memory requirements scale dramatically with model size

Llama3-8B under CKKS FHE has a plaintext-encoded weight footprint of **112 TB** before Cerium's sparse polynomial compression reduces it to **982 GB**. BERT-Base is far more manageable. NEXUS's BERT-base inference consumes approximately **164 MB** of client-server bandwidth. Evaluation keys for deep CKKS circuits with many rotation indices can require **tens to hundreds of GB**—the CAT framework demonstrates that memory pooling can reduce working GPU memory to approximately **6 GB** for complex FHE programs.

### Accuracy under CKKS approximation

NEXUS reports that its polynomial approximations for GeLU, Softmax, LayerNorm, and sign function maintain accuracy comparable to plaintext models on BERT downstream tasks. EncryptedLLM reports "minimal" perplexity degradation on HellaSwag, LAMBADA, and ARC datasets for GPT-2. PUMA explicitly demonstrates that its polynomial approximations allow loading pre-trained HuggingFace models directly without retraining, maintaining accuracy equivalent to plaintext inference. The consensus is that CKKS with carefully designed polynomial approximations introduces negligible accuracy loss for practical ML workloads.

---

## 7. Gaps, open problems, and what makes this project novel

### Current limitations define the opportunity space

**No existing work combines NEXUS's non-interactive protocol with multi-GPU/multi-node execution.** NEXUS runs on up to 4 A100 GPUs within a single node. Cerium achieves multi-GPU scaling but uses its own custom DSL/compiler rather than extending an existing protocol. Bridging this gap—taking NEXUS's proven non-interactive CKKS protocol and distributing it across 8+ GPUs on multiple nodes via NCCL—is an original contribution.

**Multi-node FHE inference remains essentially unexplored.** All current GPU-accelerated FHE work operates within a single node. Extending to multi-node execution via InfiniBand (MN5) or EFA (AWS) introduces new challenges: inter-node bandwidth for ciphertext transfer is **10–100× lower** than intra-node NVSwitch bandwidth, making communication optimization even more critical. No paper has published multi-node FHE inference results.

**The scaling efficiency problem is unsolved.** Cerium's bootstrapping achieves only **24% scaling efficiency** on 8 GPUs (1.93× speedup). Key-switching's cross-limb dependencies create an Amdahl's Law bottleneck. Improving this—perhaps through better overlapping strategies, pipelining across transformer layers, or hybrid parallelism strategies—would be a significant contribution.

**Llama-70B and beyond remain out of reach.** Cerium demonstrated Llama3-8B but nothing larger. The memory and compute requirements scale roughly linearly with parameter count, suggesting Llama-70B would require approximately **8–10× more resources** (multiple nodes with coordinated multi-node FHE execution). This project could provide the infrastructure for future scaling.

### What would be novel about extending NEXUS to multi-GPU/multi-node

Five concrete contributions could emerge from this project:

- **First multi-node non-interactive secure transformer inference**: Distributing NEXUS's CKKS operations across nodes connected via NCCL over InfiniBand/EFA, with empirical scaling results on real HPC hardware
- **NCCL-based ciphertext communication primitives**: Implementing efficient AllReduce, ReduceScatter, and AllGather operations specifically for CKKS ciphertexts with RNS limb-level partitioning, potentially as a reusable library
- **Computation-communication overlap for non-interactive FHE**: Adapting Cerium's overlapping principles to NEXUS's specific operation sequence (which differs from Cerium's compiler-generated programs)
- **Comparative scaling analysis on commodity vs. HPC hardware**: Publishing the first benchmarks comparing multi-GPU FHE scaling on AWS EC2 (NVSwitch + EFA) versus MareNostrum 5 (InfiniBand NDR200)
- **Bootstrapping distribution across nodes**: Implementing and benchmarking input broadcast and output aggregation key-switching algorithms in a multi-node setting where inter-node bandwidth is the bottleneck

### Suggested 12-week project timeline

**Weeks 1–2**: Environment setup and baseline reproduction. Clone NEXUS repository, build CPU and GPU (Phantom) versions. Reproduce BERT-base inference on a single A100. Set up AWS ParallelCluster with p4de/p5 instances. Apply for MareNostrum 5 Development Access.

**Weeks 3–4**: Profiling and bottleneck analysis. Profile NEXUS GPU execution with Nsight Systems and Nsight Compute. Identify the top-5 hotspot kernels (likely NTT, key-switching, bootstrapping components). Measure memory footprint and bandwidth utilization. Run NCCL benchmarks on multi-GPU and multi-node configurations.

**Weeks 5–7**: Implement multi-GPU distribution. Implement RNS limb-level partitioning of CKKS ciphertexts across GPUs. Implement input broadcast key-switching using NCCL Broadcast. Implement output aggregation key-switching using NCCL ReduceScatter. Test on 2, 4, 8 GPUs within a single node.

**Weeks 8–9**: Implement computation-communication overlap. Create separate CUDA streams for compute and communication. Implement kernel reordering to fill communication gaps. Measure speedup versus non-overlapped baseline. Implement CudaGraph capture for repeated operations.

**Weeks 10–11**: Multi-node extension and benchmarking. Extend NCCL communication to multi-node (2+ nodes). Benchmark on AWS p5 (2 nodes = 16 H100s via EFA) and MareNostrum 5 (4 nodes = 16 H100s via InfiniBand). Measure strong scaling (fixed BERT-base, increasing GPUs) and weak scaling (increasing sequence length/batch size). Break down communication vs. compute time.

**Week 12**: Analysis, writing, and documentation. Compile scaling results and generate plots. Compare against NEXUS single-node and Cerium (from paper). Write up findings, identifying scaling bottlenecks and future directions. Open-source the multi-GPU/multi-node extension.

---

## Conclusion: the convergence of FHE and HPC creates a new research frontier

The field has reached an inflection point where GPU-accelerated FHE transforms encrypted transformer inference from hours to seconds. NEXUS proved that non-interactive protocols eliminate the communication overhead that has plagued secure inference for years. Cerium proved that multi-GPU FHE can match custom ASICs. But no one has yet built the bridge: taking a non-interactive protocol designed for elegant single-node GPU execution and distributing it across the multi-node GPU clusters that modern HPC provides.

The key technical insight from this survey is that **naive multi-GPU distribution of FHE operations makes performance worse, not better**—Cerium's results show a 1.2× slowdown without compiler-driven optimization. The critical path runs through key-switching, which accounts for 70–80% of execution time and has cross-limb dependencies that create communication bottlenecks. Success requires implementing the input broadcast and output aggregation algorithms, building sophisticated compute-communication overlapping via CUDA streams, and carefully profiling to ensure that communication never sits on the critical path.

The combination of NEXUS's publicly available codebase (with Phantom GPU acceleration), NCCL's mature multi-GPU communication primitives, and access to AWS p5 instances and MareNostrum 5 makes this project technically executable within 12 weeks. The resulting system would be the first multi-node non-interactive secure transformer inference implementation—a genuine and publishable contribution to a rapidly growing field.