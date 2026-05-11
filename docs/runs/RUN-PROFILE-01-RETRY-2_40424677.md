# RUN-PROFILE-01-RETRY-2 — matmul nsys (4× H100) second retry

**Slice:** `RUN-PROFILE-01-RETRY-2`
**JOBID:** `40424677` (submitted 2026-05-12 after FIX-BUILD-01 rebuild)
**SLURM script:** `scripts/mn5/slurm_matmul_mgpu_nsys.sh`
**Binary:** `build/bin/matmul_align_n8k` rebuilt by FIX-BUILD-01 (job 40424627, rc=0)
**Walltime budget:** 00:20:00

## Why a second retry

| Attempt | JOBID | Outcome | Reason |
|---|---|---|---|
| 1 | 40417847 | Failed | Phase 1 absolute MAE gate (4.12e+5 vs 5e-2) — pre-FIX-BUG-01-02 binary |
| 2 (retry 1) | 40420523 | Did not produce results | Binary was rebuilt incrementally but link order was undefined because the orphan ct_pipeline copies on MN5 weren't yet exposed by the failing rebuild jobs. Once a clean rebuild was attempted (40422689, 40422772), it hit the duplicate-symbol error and never produced binaries. |
| 3 (this retry) | 40424677 | Pending | Built against post-FIX-BUILD-01 binary (job 40424627 rc=0). Matrix_mul.cu has the post-FIX-BUG-01-02 source on MN5. |

## Expected output paths (on MN5)

| File | Path |
|---|---|
| stdout | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40424677.out` |
| stderr | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40424677.err` |
| nsys-rep | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40424677.nsys-rep` |
| cuda_gpu_sum | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40424677.cuda_gpu_sum.txt` |
| nvtxsum | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40424677.nvtxsum.txt` |

## Status checks

```bash
ssh mn5-gpu "squeue -j 40424677 -o '%i %T %r %S %L %D %P'"

# Once COMPLETED:
scp mn5-gpu:/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40424677.* \
    experiments/results/2026-05-11_h100x4_matmul-mgpu-nsys/raw/
```

## Expected pass criteria

- Phase 1 absolute MAE gate: < 5e-2
- Phase 2 relative MAE gate: < 5%
- nsys traces produced for BACKFILL-S6 §6 MatMul subsection
