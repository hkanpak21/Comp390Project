# RUN-MEASURE-04-RETRY — HP-BERT data-parallel throughput, 16× H100 retry

**Slice:** `RUN-MEASURE-04-RETRY`
**JOBID:** `40424704` (submitted 2026-05-12 after FIX-SLURM-04-01 + FIX-BUILD-01)
**SLURM script:** `scripts/mn5/slurm_bert_hp_throughput_16gpu.sh`
**Binary:** `build/bin/bert_hp_multigpu` rebuilt by FIX-BUILD-01 (job 40424627, rc=0)
**Walltime budget:** see SLURM script header (~12 min wall expected)

## Why a retry

Prior attempt JOBID 40418716 failed with:

```
execve(): /tmp/hp_throughput_wrap_*: No such file or directory
```

Wrapper path was `/tmp/...` — node-local; not visible across compute nodes.
FIX-SLURM-04-01 moved it to `${PROJECT}/build/tmp/hp_throughput_wrap_*` on
the GPFS share.

## Expected output paths (on MN5)

| File | Path |
|---|---|
| stdout | `/gpfs/projects/etur02/hkanpak/logs/bert_hp_throughput_16gpu_40424704.log` (or similar — see script header) |
| stderr | `/gpfs/projects/etur02/hkanpak/logs/bert_hp_throughput_16gpu_40424704.err` |
| wrapper | `/gpfs/projects/etur02/hkanpak/Comp390Project/build/tmp/hp_throughput_wrap_40424704.sh` |

## Status checks

```bash
ssh mn5-gpu "squeue -j 40424704 -o '%i %T %r %S %L %D %P'"
ssh mn5-gpu "ls -lh /gpfs/projects/etur02/hkanpak/Comp390Project/build/tmp/"
ssh mn5-gpu "tail -40 /gpfs/projects/etur02/hkanpak/logs/bert_hp_throughput_16gpu_40424704*.log"

# Once COMPLETED:
scp mn5-gpu:/gpfs/projects/etur02/hkanpak/logs/bert_hp_throughput_16gpu_40424704*.* \
    experiments/results/2026-05-11_h100x16_bert-hp-tput-16gpu/raw/
```

## Once COMPLETED → feeds BACKFILL-S7

Throughput row for the weak-scaling table in §7. With JOBID 40418704 (4-GPU
throughput) also expected to be re-checked under the new binary, both rows of
the weak-scaling sub-table become fillable from RUN-MEASURE-03 and this run.
