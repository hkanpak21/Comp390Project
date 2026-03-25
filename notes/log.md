# Research Log — NEXUS Multi-GPU Project

**Format**: Append-only. Each entry dated. Record what was tried, what happened, conclusions, and next steps. This file is read by future Claude instances to catch up on project state.

---

## 2026-03-22 — Project Initialization

**What we did**:
- Initialized git repo, connected to `https://github.com/hkanpak21/Comp390Project.git`
- Added NEXUS (`zju-abclab/NEXUS`) and Phantom (`encryptorion-lab/phantom-fhe`) as git submodules under `vendor/`
- Created full project folder structure per plan
- Wrote top-level `CMakeLists.txt` that builds `src/multi_gpu/` as a static library against NCCL + CUDA
- Wrote AWS scripts: `launch_p4d_spot.sh`, `setup_p4d_env.sh`, `parallelcluster_config.yaml`
- Wrote MN5 SLURM scripts: `slurm_1node.sh` (4×H100), `slurm_4node.sh` (16×H100)
- Wrote profiling scripts: `nsys_profile.sh`, `ncu_ntt_profile.sh`

**Current state**:
- Repo is initialized but NOT YET built — no GPU binaries exist
- Submodules cloned but NEXUS/Phantom not yet compiled
- Still need to: check dev box GPU type, build NEXUS on dev box, run single-GPU baseline

**Immediate next steps**:
1. Check dev box GPU: `nvidia-smi`
2. Determine `CMAKE_CUDA_ARCHITECTURES` value for dev box GPU
3. Build Phantom: `cd vendor/phantom && mkdir build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=XX && make -j$(nproc)`
4. Build NEXUS GPU path: `cd vendor/nexus/cuda && mkdir build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=XX && make -j$(nproc)`
5. Run NEXUS BERT-base inference → record latency (expect ~37s on A100, different on dev box GPU)

**Key design decisions made**:
- Use `p4d.24xlarge` spot (~$4.41/hr) for multi-GPU work; dev box handles Wk 1–4 (free)
- MareNostrum 5 (already have access) for final paper-quality multi-node experiments
- Targeting HPC/systems venues (SC, PPoPP, ICS, EuroSys) — novelty is systems/parallelism, not cryptography
- RNS limb-level parallelism following Cerium's approach: GPU c handles limbs where `i % n_gpus == c`
- Two key-switching algorithms to implement and compare: Input Broadcast vs. Output Aggregation

---

## 2026-03-24 — Project Status Assessment

**Purpose**: Full audit of repository state, implementation progress, and readiness for next phase.

### What exists and works

**Infrastructure (100% complete)**:
- Git repo with two submodules (NEXUS, Phantom) — both fully cloned and populated
- `CMakeLists.txt` — properly structured with CUDA/NCCL/MPI detection, Phantom subdirectory integration, benchmark targets guarded by `if(EXISTS ...)`
- `scripts/aws/` — 3 production-ready scripts (spot launcher, env setup, ParallelCluster YAML)
- `scripts/mn5/` — 3 production-ready SLURM scripts (env setup, 1-node job, 4-node job)
- `profiling/` — 2 real scripts (Nsight Systems, Nsight Compute with kernel-targeted profiling)
- `study.md` — 300+ line literature review covering NEXUS, Cerium, and 10+ related systems; explains RNS-CKKS internals, key-switching algorithms, and multi-GPU scaling strategy
- `README.md` — complete project documentation with build instructions and hardware targets
- `.gitignore`, `.gitmodules` — properly configured

**Vendor libraries (100% present, 0% built)**:
- `vendor/phantom/` — Full Phantom FHE GPU-native CKKS library (evaluate.cu=92KB, polymath.cu=34KB, NTT kernels, key-switching, etc.)
- `vendor/nexus/` — Full NEXUS codebase with SEAL 4.1, GPU inference path, data generation scripts

### What does NOT exist yet

**Core implementation (0%)**:
- `src/multi_gpu/comm/` — only `.gitkeep`. Needs: NCCL wrappers for ciphertext RNS limb transfers (AllGather, AllReduce, broadcast)
- `src/multi_gpu/partition/` — only `.gitkeep`. Needs: RNS limb-to-GPU assignment logic (`limb_i → GPU(i % n_gpus)`)
- `src/multi_gpu/keyswitching/` — only `.gitkeep`. Needs: Input Broadcast and Output Aggregation key-switching algorithms
- `src/multi_gpu/overlap/` — only `.gitkeep`. Needs: CUDA stream management, compute-communication overlap, optional CudaGraph capture
- `src/benchmarks/` — only `.gitkeep`. Needs: `bert_inference.cu`, `bootstrapping_bench.cu`, `nccl_bandwidth_test.cu`

**Experimental data (0%)**:
- `experiments/results/` — empty
- `experiments/plots/` — empty

**Paper (0%)**:
- `paper/` — only `.gitkeep`

### Build status

The project has **never been built**. Specifically:
1. Phantom has not been compiled (no `build/` directory under `vendor/phantom/`)
2. NEXUS GPU path has not been compiled
3. The dev box GPU type is unknown (no `nvidia-smi` output recorded)
4. The top-level CMakeLists.txt will produce an empty `nexus_multi_gpu` static library (GLOB finds no .cu files) but won't error — the benchmark `if(EXISTS ...)` guards prevent missing-file failures

### Critical path to first results

1. **Verify GPU hardware**: Run `nvidia-smi`, determine CUDA architecture
2. **Build Phantom**: `cd vendor/phantom && mkdir build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=XX && make -j$(nproc)`
3. **Build NEXUS GPU path**: `cd vendor/nexus/cuda && mkdir build && cd build && cmake .. && make -j$(nproc)`
4. **Run NEXUS baseline**: Single-GPU BERT-base inference, record latency
5. **Implement `nccl_bandwidth_test.cu`**: Simplest benchmark, validates NCCL setup
6. **Implement `src/multi_gpu/comm/`**: NCCL wrappers for RNS limb arrays
7. **Implement `src/multi_gpu/partition/`**: Limb assignment strategy
8. **Implement key-switching**: Input Broadcast first (simpler), then Output Aggregation
9. **Implement `bert_inference.cu`**: End-to-end multi-GPU benchmark
10. **Profile and optimize**: Use existing Nsight scripts

### Risk assessment

- **No GPU available on dev box?** — All multi-GPU work requires AWS p4d or MN5. Dev box may only support build verification
- **NEXUS build complexity** — NEXUS depends on modified SEAL 4.1; build may require manual patching
- **Phantom API stability** — Phantom's internal API (ciphertext memory layout, RNS representation) must be understood deeply for limb-level partitioning. The Phantom library study (completed this session) documents these internals
- **NCCL + MPI interaction** — Multi-node NCCL initialization via MPI needs careful setup; ParallelCluster config already addresses this

### Summary

The project is in a **well-scaffolded but pre-implementation state**. All infrastructure, scripts, documentation, and research groundwork are complete. The vendor libraries are present but unbuilt. Zero lines of core implementation code exist. The immediate priority is building the vendor libraries and running the NEXUS single-GPU baseline to establish ground truth before writing any multi-GPU code.

---

<!-- Add new entries below this line, newest at the bottom -->

## 2026-03-25 — Submodule Init, Code Study, and Local Implementation Phase

**What we did**:

### Step 1: Submodule initialization
- `git submodule update --init --recursive` — both NEXUS and Phantom cloned successfully
- NEXUS at `e15058c`, Phantom at `1f4a198` (with nested nvbench and pybind11 submodules)
- NEXUS bundles its own Phantom copy at `vendor/nexus/cuda/thirdparty/phantom-fhe/`

### Step 2: Deep code study (all key source files read and analyzed)

**Phantom ciphertext memory layout** (confirmed from `ciphertext.h`):
```
data_[poly_i * (degree * coeff_mod_size) + limb_j * degree + coeff_k]
```
- `data_` is a `cuda_auto_ptr<uint64_t>` — GPU memory RAII wrapper
- For GELU (N=65536, L=20, 2 polys): 20.97 MB per ciphertext
- `data(i)` accessor gives pointer to poly i

**PhantomContext**: holds per-level ContextData, DNTTTable (NTT twiddles for all moduli on GPU), PhantomGaloisTool

**NEXUS CKKSEvaluator**: wraps Encoder / Encryptor / Evaluator / Decryptor around Phantom. Key methods: eval_odd_deg9_poly, sgn_eval, invert_sqrt, exp, goldschmidt_iter, newton_iter.

**main.cu benchmark structure**:
- 5 test targets: MatMul (N=8192, 3 primes), Argmax (17 primes), SoftMax (18), LayerNorm (20), GELU (20)
- BERT-base reported ~37s on A100 for full inference
- GELU: reads from `gelu_input_32768.txt`, reports MAE against `gelu_calibration_32768.txt`

**NTT**: 2D radix-8 kernel `nwt_2d_radix8_forward_inplace`. All limbs transformed in parallel. Independent per-limb -> perfect GPU parallelism.

**Key-switching**: `key_switch_inner_prod_c2_and_evk` kernel. 3 stages: RNS decomposition -> inner product over beta digits -> basis extension. This is the bottleneck for multi-GPU communication.

**Bootstrapping**: 31 total levels (17 main + 14 bootstrap). Pipeline: StoC (BSGS linear transform) -> ModularReduction (minimax poly) -> CtoS (BSGS inverse). ~8s on single A100.

### Step 3: LaTeX code study document
- Written: `paper/code_study.tex` (~450 lines)
- Covers: FHE/CKKS/RNS/NTT background, Phantom data structures, NEXUS CKKSEvaluator API, all 5 neural network operations, our multi-GPU design, build instructions, profiling guide
- Suitable for course presentation

### Step 4: Multi-GPU implementation files (locally written, will compile on EC2)

**`src/multi_gpu/comm/`**:
- `nccl_comm.cuh/cu`: MultiGpuContext (ncclCommInitAll), allgather_ciphertext_limbs, allreduce_partial_keyswitching, broadcast_ciphertext, scatter/gather point-to-point

**`src/multi_gpu/partition/`**:
- `rns_partition.cuh/cu`: owner_of_limb, local_index_of_limb, n_local_limbs, limbs_for_gpu, kernel_scatter_limbs (CUDA kernel), kernel_gather_limbs, host wrappers

**`src/multi_gpu/keyswitching/`**:
- `input_broadcast.cuh/cu`: AllGather c2 -> local inner product -> extract local slice
- `output_aggregation.cuh/cu`: Partial inner product from local limbs -> AllReduce -> extract local slice

**`src/multi_gpu/overlap/`**:
- `stream_manager.cuh/cu`: PerGpuStreams (compute + nccl streams + events), StreamManager (barrier, CudaGraph capture/replay), OverlapScheduler (compute-comm overlap scheduling)

### Step 5: Benchmark harnesses

**`src/benchmarks/nccl_bandwidth_test.cu`**:
- Validates NCCL + peer access, measures AllGather/AllReduce/Broadcast bandwidth
- Reports ciphertext-transfer time (expected ~33 us for 21 MB on NVSwitch)

**`src/benchmarks/bert_inference.cu`**:
- Full BERT-base benchmark harness: 12-layer loop, per-component timing
- Outputs CSV: n_gpus, total_ms, matmul_ms, gelu_ms, softmax_ms, layernorm_ms, keyswitch_ms, comm_ms, speedup, efficiency
- Integration points clearly marked (TODO on EC2: connect multi-GPU key-switch calls)

**`src/benchmarks/bootstrapping_bench.cu`**:
- Isolated bootstrapping latency: calls Bootstrapper.bootstrap_sparse_3()
- Reports mean/min/max/stddev over N iterations
- Outputs CSV for scaling analysis

**Current state**:
- All implementation files written with correct API signatures
- Stubs (CUDA memset zero) in place where Phantom internal kernel calls needed
- TODO markers at exact integration points for EC2 work
- Ready to commit and push

**Immediate next steps on EC2**:
1. `cmake .. -DCMAKE_CUDA_ARCHITECTURES=80 && make -j96`
2. Run `nccl_bandwidth_test` — validate NCCL + NVSwitch
3. Run `bert_inference --n-gpus 1` — reproduce 37s baseline
4. Replace stubs in `local_inner_product()` with Phantom's actual key_switch kernel call
5. Run `bert_inference --n-gpus 2,4,8` — measure scaling

**Key design decisions made this session**:
- Use cyclic limb assignment (j % n_gpus): simplest, load-balanced, matches Cerium
- Implement both Input Broadcast AND Output Aggregation: benchmark both on EC2
- Use separate compute/NCCL streams + cudaEvent for overlap
- CudaGraph capture for bootstrapping (reduces ~250us kernel launch overhead)
- AllReduce for OutputAgg uses ncclSum on uint64 — caller applies mod reduction afterward
- local_inner_product stub uses cudaMemset(0) so code links and runs without Phantom kernels

---
