# RUN-PROFILE-01-RETRY — matmul nsys (4× H100) resubmission record

**Reason for resubmission:** JOBID 40417847 failed the MAE gate due to ciphertext initialization bug in matrix_mul.cu (FIX-BUG-01-02). Binary rebuilt with corrected code.

Tracked manifest. The raw `.nsys-rep`, `.cuda_gpu_sum.txt`, `.nvtxsum.txt`,
and stdout/stderr files land under
`experiments/results/2026-05-11_h100x4_matmul-mgpu-nsys/raw/` once the job
COMPLETES; that directory is git-ignored.

| Field | Value |
|---|---|
| Slice | `RUN-PROFILE-01-RETRY` |
| SLURM script | `scripts/mn5/slurm_matmul_mgpu_nsys.sh` |
| Binary | `build/bin/matmul_align_n8k` (rebuilt after FIX-BUG-01-02) |
| Invocation | `--n-gpus 4 --trials 3` (median of 3 trials, under nsys) |
| Host | MareNostrum 5 ACC partition (`acc` / `acc_ehpc`) |
| Submitted | 2026-05-12 (resubmission after fix) |
| JOBID | **40420523** |
| Prior Job | JOBID 40417847 (failed MAE gate) |
| Initial state | PENDING |
| Walltime budget | 00:20:00 |

## Build provenance

`build/bin/matmul_align_n8k` rebuilt on MN5 after fixing FIX-BUG-01-02
(ciphertext initialization in `src/nexus_eval/matrix_mul.cu` lines 309–311).
The prior job (40417847) was built against code with uninitialized ciphertext
passed to `add_many()`, causing garbage output (MAE = 4.2e+5).

Current build: Correct ciphertext initialization where `add_many()` properly
initializes `res_col_ct` before scale adjustment.

Build was incremental (`make -j20 matmul_align_n8k`) on `cuda/12.8 +
cmake/3.30.5 + nccl/2.24.3-1` and clean.

## Expected output paths (on MN5)

| File | Path |
|---|---|
| stdout | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40420523.out` |
| stderr | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40420523.err` |
| nsys-rep | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40420523.nsys-rep` |
| cuda_gpu_sum | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40420523.cuda_gpu_sum.txt` |
| nvtxsum | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40420523.nvtxsum.txt` |

## Status checks

```bash
ssh mn5-gpu "squeue -j 40420523 -o '%i %T %r %S %L %D %P'"

# Once COMPLETED:
scp mn5-gpu:/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40420523.* \
    experiments/results/2026-05-11_h100x4_matmul-mgpu-nsys/raw/
```

## Difference from JOBID 40417847

The binary now has the correct ciphertext initialization logic:
- Old (buggy): `res_col_ct.set_scale()` → `add_many()` (uninitialized ciphertext)
- New (fixed): `add_many()` → `res_col_ct.set_scale()` (properly initialized)

Expected result: Should pass Phase 1 absolute MAE gate (< 5e-2) and Phase 2 relative gate (< 5%).
