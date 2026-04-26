# Nsight Systems — First-Time Guide for multiNEXUS Traces

This is a "what is this thing and what will I see" guide, written specifically
for your `trace_prefetch.nsys-rep` and `trace_dksrot.nsys-rep` files. Not a
generic Nsight tutorial — every example refers to NVTX ranges and streams that
exist in *your* binaries.

---

## 0. What Nsight Systems is, in one paragraph

Nsight Systems is a **timeline viewer**. You give it a recording file
(`.nsys-rep`) and it shows you, on a horizontal timeline:
- **Rows for CPU threads** — every function call, NVTX range, syscall, and
  pthread event lined up by wall-clock time.
- **Rows for each GPU** — every CUDA kernel launch, memcpy, and NCCL
  collective on each stream, again by wall-clock.
- **Connections between them** — when the host called `cudaMemcpyAsync`, you
  see a line from the CPU thread row down to the GPU stream where the copy
  actually executed.

There is **no aggregate, no benchmark number, no profile-style "% of time"
metric by default**. It's literally a recording of what happened. The value
comes from looking at *overlap*, *idle gaps*, and *which thing waited on which
other thing* — questions that timer printfs cannot answer.

There **is** a stats panel that aggregates per-NVTX-range and per-kernel time
percentages — that's where the numbers in PI_BRIEFING §3c came from. We'll get
to it in §5.

---

## 1. Install on macOS

1. Sign in (or create a free account) at developer.nvidia.com.
2. Download **Nsight Systems 2024.6** or newer:
   https://developer.nvidia.com/nsight-systems/get-started
3. Pick the **macOS Host** package (Universal binary, runs on Apple Silicon).
4. Mount the `.dmg`, drag `Nsight Systems` into `/Applications`.

**Note:** This Mac never runs CUDA itself. The host install is a pure
viewer/analyzer for trace files collected on a Linux+NVIDIA box (your MN5
node). You only need the *Linux target* package on MN5 if `nsys` isn't already
on `$PATH` there — `module load cuda/12.8` ships it.

Check the install:
```bash
ls /Applications/ | grep -i nsight
# Expect: "NVIDIA Nsight Systems 2024.x.app" (or similar)
```

---

## 2. Get the trace files

You have two options. If profiling runs already exist on MN5, just pull them.
Otherwise, generate fresh ones with the recipe in
[PI_BRIEFING.md §6a](PI_BRIEFING.md).

### 2a. Check whether traces already exist on MN5
```bash
ssh mn5-gpu "ls -lh /gpfs/projects/etur02/hkanpak/Comp390Project/traces/ 2>/dev/null"
```

### 2b. Pull them locally
```bash
mkdir -p ~/nexus-traces
rsync -avz --progress \
  mn5-gpu:/gpfs/projects/etur02/hkanpak/Comp390Project/traces/*.nsys-rep \
  ~/nexus-traces/
```

Trace files are typically 50-200 MB each.

### 2c. (Only if needed) Generate fresh traces
Run the `salloc` + `nsys profile` block from
[PI_BRIEFING.md §6a](PI_BRIEFING.md) — that produces both
`trace_prefetch.nsys-rep` and `trace_dksrot.nsys-rep`.

---

## 3. Opening your first trace — the orientation tour

```bash
open -a "Nsight Systems" ~/nexus-traces/trace_dksrot.nsys-rep
```

Wait 5-30 seconds for indexing. Then you'll see roughly this layout:

```
┌─────────────────────────────────────────────────────────────────────┐
│ Project Explorer | Timeline View                                    │
│  trace_dksrot →  │ ┌─────────────────────────────────────────────┐  │
│                  │ │ CPU rows (threads, NVTX, OS runtime)        │  │
│                  │ │   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │  │
│                  │ │ CUDA HW (GPU 0): Default stream | copy | NCCL│ │
│                  │ │ CUDA HW (GPU 1): Default | NCCL              │  │
│                  │ │ CUDA HW (GPU 2): Default | NCCL              │  │
│                  │ │ CUDA HW (GPU 3): Default | NCCL              │  │
│                  │ │ CUDA API (per-thread)                        │  │
│                  │ │ NCCL                                          │  │
│                  │ │ OS runtime libraries                          │  │
│                  │ └─────────────────────────────────────────────┘  │
│                  │ Events View (bottom): table of selected items    │
└─────────────────────────────────────────────────────────────────────┘
```

### What each row means in **your** trace

| Row | Contents in multiNEXUS |
|---|---|
| **Threads → main** | NVTX bands you placed: `bootstrap_sparse_3`, nested inside it `coefftoslot_3`, then `sfl_full_3` containing 75 `rotate_inplace step=N` ranges |
| **Threads → worker_0..3** (Phase 4b) | Persistent worker threads — each runs `partialKS gpu=G` ranges when DKS rotation dispatches |
| **CUDA HW (GPU 0) → Default stream** | NTT kernels, key-switch kernels, mod-down kernels — the actual GPU compute |
| **CUDA HW (GPU 0) → copy_stream_** *(prefetch trace only)* | Async H→D memcpys for next-rotation keys. **This row should overlap with Default stream activity.** |
| **CUDA HW (GPU 0..3) → NCCL stream** *(dks trace only)* | `ncclAllReduce` kernels — the cross-GPU communication |
| **CUDA API** | Every host-side CUDA call (`cudaLaunchKernel`, `cudaMemcpyAsync`, `cudaMalloc`, `cudaStreamSynchronize`, …). Visible *just above* the GPU HW rows so you can see launch latency. |
| **NCCL** | NCCL-level events distinct from the kernel — connects to NCCL HW kernels. |
| **OS runtime** | `pthread_mutex_lock`, `nanosleep`, `read`. Useful for spotting host stalls. |

### First navigation moves

| Action | Shortcut |
|---|---|
| Fit timeline to window | `H` |
| Zoom in/out at cursor | `+` / `-` or scroll-wheel |
| Pan timeline | click+drag |
| Right-click on a range → **Zoom to selection** | jumps to one item's window |
| Hover any band | tooltip with name + duration |
| Click a band, then `F` | follow connections (e.g. CPU launch → GPU kernel) |

---

## 4. The 5 questions to answer in your first 5 minutes

Each of these maps to an exact click-path. Do them in order on
`trace_dksrot.nsys-rep`.

### Q1. Where does the bootstrap actually start?
Press `Ctrl-F` (or `Cmd-F`), type `bootstrap_sparse_3`. The search jumps to
the first NVTX range with that name on the main thread. Right-click →
**Zoom to selected**. You're now looking at one full bootstrap call (~2.1 s
wall in Phase 4b).

### Q2. Are all 4 GPUs actually doing work?
With one bootstrap zoomed in, scan down the GPU rows. You should see solid
colored activity on all four `CUDA HW (GPU N) → Default stream` rows.

- **Healthy:** all 4 GPUs show kernel activity that fills most of the
  timeline (some gaps OK).
- **Broken:** GPU 0 is busy but 1-3 are mostly idle → DKS dispatch isn't
  actually parallelising. (This was the symptom of Phase 3 v1 before the
  persistent workspace fix.)

### Q3. How long is one rotation, and what's it made of?
Inside the bootstrap, find a `dist_rotate_phantom step=N` NVTX band on the
main thread (in the `sfl_full_3` sub-section). Hover for its duration —
typically 25-30 ms. Click into it, then look at the rows underneath:

- `P1_galois_ntt` — single-GPU NTT prep (a few ms, GPU 0 only)
- `P2_peer_broadcast` — GPU 0 → 1,2,3 NVLink memcpy (sub-ms)
- `P3_dispatch_partialKS` — **expect 4 simultaneous bands** (`partialKS gpu=0..3`)
  on 4 different default streams. This is your visual proof of multi-GPU compute.
- NCCL AllReduce kernels right after P3
- `P4_writeback` — back into GPU 0's ciphertext

### Q4. Is the NCCL straggler real?
Click on one `ncclAllReduce` kernel on a GPU NCCL row. Look at:
- **Kernel duration** (in the tooltip / Events View) — typically 3-5 ms.
- **Time between when GPU 0 launched it vs when GPU 3 launched it** — that
  delta is launch jitter; the slowest GPU sets the AllReduce wall-clock.
- The gap *before* the kernel on the fastest GPU's NCCL row — that gap is
  the straggler wait, the Phase 4d target.

### Q5. Are there `cudaMalloc` calls in the hot loop?
In the search box (`Ctrl-F`), type `cudaMalloc`. The CUDA API row will
highlight matches. **Inside `bootstrap_sparse_3`, you should see zero hits.**
If you see them, a workspace regressed (this is one of the highest-leverage
checks — Phase 3 v1's killer was 2,400 mallocs/layer).

---

## 5. The Stats panel — where the actual numbers live

The timeline shows you *what* happened. The Stats panel aggregates *how
much* time everything took.

**Open it:** menu **Project → Generate Report** → select sections:
- **NVTX Range Summary** (`nvtxsum`) — total time per NVTX range, sortable
- **CUDA GPU Kernel Summary** (`gpukernsum`) — total time per kernel name
- **CUDA API Summary** (`cudaapisum`) — host-side API call times
- **NCCL Summary** (`nccl_sum`) — per-collective stats

These tables give you columns like:

```
Time (%) | Total Time | Instances | Avg | Med | Min | Max | StdDev | Name
   40.2  | 854.3ms    |   300     | 2.8ms | 2.7ms | 2.5ms | 4.1ms | 0.2ms | ntt_kernel
   14.1  | 299.7ms    |    75     | 4.0ms | 3.9ms | 3.7ms | 6.2ms | 0.4ms | ncclAllReduce
   ...
```

**This is the table you paste into slides.** The "NTT is 40%" number from
PI_BRIEFING §3c came directly from `gpukernsum` filtered to `*ntt*`.

---

## 6. CLI alternative — `nsys stats` (for slide-friendly text)

You don't need the GUI for the Stats tables. From any terminal with
`module load cuda/12.8`:

```bash
ssh mn5-gpu
module load cuda/12.8
nsys stats --report nvtxsum,gpukernsum,nccl_sum \
  /gpfs/projects/etur02/hkanpak/Comp390Project/traces/trace_dksrot.nsys-rep
```

Outputs plain-text tables you can copy/paste into a doc or email.

For just the top-N kernels:
```bash
nsys stats --report gpukernsum --format csv trace_dksrot.nsys-rep \
  | head -15
```

---

## 7. Comparing the two traces (the most valuable workflow for you)

Open both files in the same Nsight Systems window:
- **File → Open** → `trace_prefetch.nsys-rep`
- **File → Open** → `trace_dksrot.nsys-rep` (opens as second tab)

What to compare side-by-side:

| Question | Where to look |
|---|---|
| Per-bootstrap wall time | `bootstrap_sparse_3` duration (top-thread NVTX), both traces |
| Are GPUs 1-3 used at all? | GPU 1-3 Default-stream rows: empty in prefetch trace, busy in dks trace |
| Where does prefetch save time? | `copy_stream_` row in prefetch trace overlapping `Default` row — this overlap is the entire Phase 1 win |
| Where does DKS spend the time prefetch saved? | NCCL stream activity in dks trace (the new cost) |
| Net winner per bootstrap | `nvtxsum` for `bootstrap_sparse_3` — currently dks wins by ~150 ms |

This side-by-side is what tells the *story* of the project: prefetch
trades H→D bandwidth for kernel time; DKS trades NCCL bandwidth for compute
parallelism. Both work, dks slightly better.

---

## 8. Common first-time gotchas

| Symptom | Cause | Fix |
|---|---|---|
| GPU rows look empty | Section collapsed | Click the disclosure triangle next to "CUDA HW (GPU 0)" |
| No NVTX bands on threads | Trace was captured without `-t nvtx` | Re-run with `--trace=cuda,nvtx,osrt,nccl` |
| Trace file is 5 GB and won't open | No `--capture-range` clamp | Add `--capture-range=nvtx --nvtx-capture=bootstrap_sparse_3` to nsys command |
| Timeline shows ~30 minutes but bootstrap is 2 s | Setup phases dominate | Use Q1 above to zoom into one bootstrap |
| Stream names like `unnamed stream 14` | Phantom doesn't name its streams | Identify by activity: copy_stream is the one with H→D memcpys; NCCL stream has only NCCL kernels |
| GUI very slow on Apple Silicon | Trace is huge | Filter via menu: **Timeline → Filter and Zoom to Selected** |

---

## 9. What to do after your first session

Once you can answer Q1-Q5 above on your own traces, the next steps are:

1. **Verify the headline numbers match the handoff.** `nvtxsum` row for
   `bootstrap_sparse_3` should show ~2,126 ms avg in the Phase 4b dks trace.
   If it doesn't, the trace is from a different phase or the binary changed
   — re-check before quoting numbers to the PI.

2. **Screenshot the side-by-side comparison from §7** — a single image
   showing both bootstraps stacked is the most legible artifact for slides.
   Use the Mac native shortcut `Cmd-Shift-4` to capture a region.

3. **Move to Nsight Compute** ([PI_BRIEFING.md §7a](PI_BRIEFING.md)) — once
   you've identified that NTT is 40% of bootstrap (which nsys tells you), you
   need `ncu` to learn *why* the NTT kernel is slow. That's the input to the
   Phase 4c go/no-go decision.

---

## 10. If something goes wrong

- **`nsys` crashes opening the file:** corrupt trace; re-rsync from MN5,
  check file size matches what `ls -l` shows on the source.
- **GUI says "no GPU activity":** the run probably crashed before any
  bootstrap. Check the SLURM stderr log on MN5 — `cat
  /gpfs/projects/etur02/hkanpak/logs/trace_<jobid>.err`.
- **NVTX ranges are wrong / missing:** confirm the binary you profiled was
  built without `-DNEXUS_NO_NVTX`. Check `nm build/bin/bert_dks_multigpu |
  grep nvtxRangePush` on MN5 — should print non-empty.
