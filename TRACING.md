# Tracing & Timeline Visualisation

This is the "see what's actually happening" doc. Instead of running the benchmark
and guessing where time goes, capture an nsys trace once and read it in the GUI.

## What gets instrumented

NVTX ranges are compiled into the binary unconditionally (overhead ~300 ns per
push/pop). They name the phases that aren't visible from kernel names alone:

| Range | Where | What it measures |
|---|---|---|
| `bootstrap_sparse_3` | [Bootstrapper.cu](src/nexus_eval/bootstrapping/Bootstrapper.cu) | One entire bootstrap call |
| `coefftoslot_3`, `slottocoeff_full_3` | Bootstrapper.cu | The two big sub-phases of bootstrap |
| `sfl_full_3`, `sflinv_full_3` | Bootstrapper.cu | The linear-transform stages that hold the 75 rotations |
| `rotate_inplace step=N`, `rotate_vector step=N` | [ckks_evaluator.cu](src/nexus_eval/ckks_evaluator.cu) | One rotation; step includes the Galois step |
| `prefetch step=N` | ckks_evaluator.cu | Async H→D kick for the next rotation (Phase 1) |
| `ks_prefetch idx=K`, `ks_load idx=K` | [galois_key_store.cuh](src/nexus_eval/galois_key_store.cuh) | Key-store side of the prefetch |
| `dist_rotate_phantom step=N` | [galois_oa.cu](src/multi_gpu/keyswitching/galois_oa.cu) | One distributed rotation (Phase 3) |
| `P1_galois_ntt`, `P2_peer_broadcast`, `P3_dispatch_partialKS`, `P4_writeback` | galois_oa.cu | Inner phases of dist rotation |
| `partialKS gpu=G` | galois_oa.cu | Per-GPU partial key-switch work (shows parallelism) |

All individual CUDA kernels + memcpy + NCCL collectives are captured automatically
by nsys — no annotation needed for those.

## How to run

SLURM job: [scripts/mn5/slurm_trace_nsys.sh](scripts/mn5/slurm_trace_nsys.sh)

```bash
ssh mn5-gpu
cd /gpfs/projects/etur02/hkanpak/Comp390Project
sbatch scripts/mn5/slurm_trace_nsys.sh
```

Runs both `DKS_ROTATE=0` (prefetch path) and `DKS_ROTATE=1` (distributed rotation)
back-to-back, writing two files:

```
traces/trace_prefetch.nsys-rep   (DKS_ROTATE=0)
traces/trace_dksrot.nsys-rep     (DKS_ROTATE=1)
```

Expect ~50–200 MB each. Text summaries (top NVTX ranges, top kernels) also print
to the SLURM log.

## How to view

### Option A — Nsight Systems GUI (recommended)

1. Install [Nsight Systems 2024.6+](https://developer.nvidia.com/nsight-systems) on your Mac.
2. Transfer the trace files:
   ```bash
   scp mn5-gpu:/gpfs/projects/etur02/hkanpak/Comp390Project/traces/*.nsys-rep ~/nexus-traces/
   ```
3. Open in Nsight Systems:
   ```bash
   open -a "Nsight Systems" ~/nexus-traces/trace_dksrot.nsys-rep
   ```

### What to look for in the GUI

**Top of timeline: CPU threads.** You'll see the main thread, plus 4 persistent
worker threads (Phase 4b). NVTX ranges like `bootstrap_sparse_3` sit at top level;
`rotate_inplace step=N` nests inside, `dist_rotate_phantom step=N` nests further.

**Below: per-GPU streams.** For each of 4 GPUs:
- `Default stream`: NTT / key-switching / mod-down kernels
- `copy_stream_` (only in DKS_ROTATE=0): H→D memcpys for async prefetch — should
  run *concurrently* with default-stream kernels; if they're serial, Phase 1
  regressed
- `NCCL comm stream`: AllReduce operations

**Key overlaps to inspect:**

1. **Async prefetch overlap (DKS_ROTATE=0).** Open `trace_prefetch.nsys-rep`,
   zoom into one rotation. The `copy_stream_` H→D block for key N+1 should
   start *while* the default stream is still crunching the rotation-N kernel.
   If you see the H→D block fully *after* the kernel, `cudaHostRegister` didn't
   apply — async is silently sync.

2. **DKS rotation parallelism (DKS_ROTATE=1).** Open `trace_dksrot.nsys-rep`,
   zoom into one `dist_rotate_phantom` range. You should see `partialKS gpu=0..3`
   running simultaneously on 4 different default streams. If only one is busy,
   the dispatcher is serialising (broken).

3. **NCCL AllReduce cost.** Look for NCCL collective ops right after `P3_dispatch_partialKS`.
   Count the bytes (shown in NCCL row tooltip) and compare against NVLink bandwidth
   (~50 GB/s). If AllReduce is > 1 ms for < 50 MB, something's wrong with the comm.

4. **cudaMalloc pauses.** Filter by `cudaMalloc` in the CUDA API row. Before Phase
   3 v2 these happened 8× per rotation; after, only during `ensure_rotation_workspace`
   first call.

### Option B — Text summary only (no GUI needed)

The SLURM log (`logs/trace_<jobid>.out`) includes:
- `nvtxsum`: top NVTX ranges by total time
- `gpukernsum`: top CUDA kernels by total time

Quick answer to "where does the 2.1 s go" without opening the GUI.

### Option C — Re-run for text stats only

```bash
ssh mn5-gpu
module load cuda/12.8
nsys stats --report nvtxsum,gpukernsum,cudaapisum /path/to/trace.nsys-rep
```

## Common gotchas

- **`.nsys-rep` is the report, `.qdrep` is the old format** — nsys 2024 writes `.nsys-rep`.
- **GPU timeline is empty** → your binary was built without `-lineinfo` or nsys couldn't
  attach. Check `nsys profile` ran with sudo privileges or that `NVRM` module supports it.
  On MN5 it works out-of-the-box.
- **NVTX ranges missing** → `-t nvtx` was dropped from the command line, or
  `NEXUS_NO_NVTX` was defined at compile time.
- **Huge trace files (> 500 MB)** → lower the sample rate (`--sample=none`) or
  use `--capture-range=nvtx --nvtx-capture=bootstrap_sparse_3` to only record
  during a single bootstrap call.

## Why this matters

Without traces we were running, timing, guessing. Phase 4a and 4b each delivered
"smaller than expected" wins because our per-component cost estimates were wrong.
A trace would have shown that `std::thread` spawn was microseconds (not milliseconds)
and `PhantomCiphertext::resize` was no-op on same-chain-index calls — saving us two
SLURM round-trips. Future optimisation passes should start with a trace.
