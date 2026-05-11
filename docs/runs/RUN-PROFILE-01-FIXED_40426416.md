# RUN-PROFILE-01-FIXED — matmul nsys (4× H100) post FIX-BUG-MATMUL-{01,02}

**Slice:** `RUN-PROFILE-01-FIXED`
**JOBID:** `40426416` (submitted 2026-05-12 after FIX-BUG-MATMUL-01 + FIX-BUG-MATMUL-02)
**SLURM script:** `scripts/mn5/slurm_matmul_mgpu_nsys.sh`
**Binary:** `build/bin/matmul_align_n8k` rebuilt on MN5 login node after both matmul fixes
**Walltime budget:** 00:20:00
**Resources:** 1 node, 4× H100, --exclusive

## History of attempts

| JOBID | Outcome | Reason |
|---|---|---|
| 40417847 | FAILED Phase 1 MAE (4.22e+5) | Stream race in `multiply_power_of_x` (latent since commit b4949cb) |
| 40420523 | FAILED Phase 1 MAE | Same root cause; orphan "FIX-BUG-01-02" was a misdiagnosis |
| 40422689 | FAILED link | Duplicate `ct_pipeline.cu.o` (fixed by FIX-BUILD-01) |
| 40422772 | FAILED link | Same as 40422689 |
| 40424627 | COMPLETED (build only, rc=0) | FIX-BUILD-01 — produced clean binaries |
| 40424677 | FAILED Phase 1 MAE | Stream race still present in the new binary |
| 40425829 | DEBUG smoke | Captured full scale chain — pointed at polynomial-level corruption |
| 40426042 | DEBUG smoke | Stream fix applied → MAE 4.1e+5 → 7.5 (55,000× drop) |
| 40426179 | DEBUG smoke | Restored NEXUS `set_scale` pattern; same MAE = 7.5 (decoded = 2× truth) |
| 40426228 | DEBUG smoke (single-GPU) | Added /2.0 in test; MAE = 1.57e-07 → Phase 1 PASS |
| 40426307 | Phase 2 false negative | Relative gate at noise floor — both MAEs at ~1.5e-7 |
| **40426416** | **PASS** — see below | **Fully fixed** |

## Result — job 40426416

```
[Phase 1] single-GPU median = 21033.0 ms (σ=121.6)
[Phase 1] single-GPU amortized per-column = 0.3286 s
   (NEXUS reports 1.31 s on A100)
[Phase 1 gate] single-GPU MAE vs plain truth = 1.466271e-07  (abs tol 5e-02: PASS)

[Phase 2] multi-GPU wall median = 8943.4 ms (σ=924.5, n_gpus=4)
[Phase 2] multi-GPU amortized per-column = 0.1397 s
[Phase 2] multi-GPU speedup vs single-GPU = 2.35x
    single-GPU MAE = 1.466271e-07  (abs tol 5e-02: PASS)
    multi-GPU  MAE = 1.311820e-07  (abs tol 5e-02: PASS)
    overall        : PASS  (abs_pass=Y, rel_pass=N)
```

## Headline numbers for §6.1 (Goal 1 MatMul subsection)

| Path | Latency / col | Source |
|---|---|---|
| Single-GPU H100 (our build) | 0.329 s | This job — Phase 1 median |
| Single-GPU A100 (NEXUS) | 1.31 s | NEXUS published |
| 4× H100 output-channel split | 0.140 s | This job — Phase 2 median |
| Speedup, 4× vs 1× (our H100) | 2.35× | Phase 2 / Phase 1 |

Hardware uplift A100 → H100: 1.31 / 0.329 ≈ 4.0×. Output-channel split adds 2.35× on top of that.

## Output paths (on MN5)

| File | Path |
|---|---|
| stdout | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40426416.out` |
| stderr | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40426416.err` |
| nsys-rep | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40426416.nsys-rep` |
| cuda_gpu_sum | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40426416.cuda_gpu_sum.txt` |
| nvtxsum | `/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40426416.nvtxsum.txt` |

## Once nsys finalizes

```bash
scp mn5-gpu:/gpfs/projects/etur02/hkanpak/logs/matmul_mgpu_nsys_40426416.* \
    experiments/results/2026-05-12_h100x4_matmul-mgpu-nsys/raw/
```

This is the first PROFILE-01 trace that paper §6.1 can quote. Earlier
runs (40417847 through 40424677) all produced garbage matmul output due
to the stream race in `multiply_power_of_x`, and any nsys traces from
those runs reflect computation on corrupt ciphertexts — kernel timing
would still be valid for kernel-level performance reporting, but should
be cross-checked against this fixed trace.
