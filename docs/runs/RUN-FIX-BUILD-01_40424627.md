# RUN-FIX-BUILD-01 — clean rebuild of matmul_align_n8k + bert_hp_multigpu

**Slice:** `FIX-BUILD-01`
**JOBID:** `40424627` (submitted 2026-05-12)
**SLURM script:** `scripts/mn5/slurm_rebuild_matmul_bert.sh`
**Walltime budget:** 01:00:00
**Resources:** 1 node, 1 GPU (build-only; cpus-per-task=80 for parallel make)

## Why this rebuild was needed

Two prior rebuild attempts failed with multiple-definition linker errors:

- **40422689** — `rebuild_matmul_40422689.err`
- **40422772** — `rebuild_clean_40422772.err`

Both reported the same symptom:

```
/usr/bin/ld: libnexus_multi_gpu.a(ct_pipeline.cu.o): multiple definition of
  `nexus_multi_gpu::CtPipeline::enable_galois_keys()'; ...first defined here
```

## Root cause

`GLOB_RECURSE src/multi_gpu/*.cu` in `CMakeLists.txt` picked up two byte-identical
copies of `ct_pipeline.cu` on MN5:

1. `src/multi_gpu/archive/pipeline/ct_pipeline.cu` (intended location, archived)
2. `src/multi_gpu/pipeline/ct_pipeline.cu` (stale orphan, no longer in local repo)

Both compiled to `ct_pipeline.cu.o` inside `libnexus_multi_gpu.a`, conflicting at
link time. Same problem for `multi_node_pipeline.cu`.

Why MN5 had the orphan: the rsync sync command in `CLAUDE.md` uses
`--include='*.cu' --include='*.cuh'` without `--delete`, so files removed locally
were never deleted on MN5. Local `CMakeLists.txt` has had
`list(FILTER MULTI_GPU_SRCS EXCLUDE REGEX "src/multi_gpu/archive/")` since the
CtPipeline archival, but the updated `CMakeLists.txt` was not in the rsync
include pattern, so the EXCLUDE filter was never propagated.

## Fix applied prior to job submission

1. `ssh mn5-gpu rm -rf /gpfs/projects/etur02/hkanpak/Comp390Project/src/multi_gpu/pipeline`
2. `scp CMakeLists.txt mn5-gpu:/gpfs/projects/etur02/hkanpak/Comp390Project/CMakeLists.txt`
3. SLURM job (this manifest) removes stale build cache, re-runs `cmake`, then
   `make -j80 matmul_align_n8k bert_hp_multigpu`.

## Sanity checks embedded in the job

The script aborts before `cmake` if either:

- `src/multi_gpu/pipeline/` still exists, or
- `CMakeLists.txt` lacks the `FILTER MULTI_GPU_SRCS EXCLUDE REGEX` line.

## Expected output paths (on MN5)

| File | Path |
|---|---|
| stdout | `/gpfs/projects/etur02/hkanpak/logs/rebuild_mb_40424627.out` |
| stderr | `/gpfs/projects/etur02/hkanpak/logs/rebuild_mb_40424627.err` |
| binary (matmul) | `/gpfs/projects/etur02/hkanpak/Comp390Project/build/bin/matmul_align_n8k` |
| binary (bert)   | `/gpfs/projects/etur02/hkanpak/Comp390Project/build/bin/bert_hp_multigpu` |

## Status checks

```bash
ssh mn5-gpu "squeue -j 40424627 -o '%i %T %r %S %L %D %P'"
ssh mn5-gpu "tail -40 /gpfs/projects/etur02/hkanpak/logs/rebuild_mb_40424627.out"
ssh mn5-gpu "ls -lh /gpfs/projects/etur02/hkanpak/Comp390Project/build/bin/{matmul_align_n8k,bert_hp_multigpu}"
```

## Once COMPLETED

Downstream actions unblocked (each its own slice):

- Resubmit `RUN-PROFILE-01-RETRY` (matmul nsys) — prior attempt 40420523 used
  pre-FIX binary and failed MAE gate.
- Resubmit `RUN-MEASURE-04` (HP-BERT 16-GPU throughput) — prior attempt 40418716
  failed on `/tmp` wrapper path; fix already in
  `scripts/mn5/slurm_bert_hp_throughput_16gpu.sh` (uncommitted).
