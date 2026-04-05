# Experiment Results Index

This file is the running log of every experiment run in this project.
Update it every time you complete a run on EC2 or any other machine.

## Naming Convention

Each run lives in a directory under `experiments/results/`:

```
results/YYYY-MM-DD_<hardware>_<experiment>/
```

### Hardware Shortnames

| Shortname  | Instance / Machine              | GPUs                   |
|------------|---------------------------------|------------------------|
| `l4`       | AWS g6.xlarge                   | 1× NVIDIA L4 (24 GB)   |
| `t4`       | AWS g4dn.2xlarge                | 1× NVIDIA T4 (16 GB)   |
| `a100x1`   | AWS p4d.24xlarge                | 1× A100 40 GB          |
| `a100x2`   | AWS p4d.24xlarge                | 2× A100 40 GB          |
| `a100x4`   | AWS p4d.24xlarge                | 4× A100 40 GB          |
| `a100x8`   | AWS p4d.24xlarge                | 8× A100 40 GB (full)   |
| `h100x4`   | MareNostrum 5, 1 node           | 4× H100 64 GB          |
| `h100x16`  | MareNostrum 5, 4 nodes          | 16× H100 64 GB         |

### Experiment Shortnames

| Shortname          | What It Measures                                              |
|--------------------|---------------------------------------------------------------|
| `build-verify`     | Compilation + smoke test (does it run without crashing?)      |
| `baseline-bert`    | Full BERT-base end-to-end latency on 1 GPU                    |
| `baseline-ops`     | Per-operation latency: MatMul, GELU, SoftMax, LayerNorm, Argmax |
| `profile-nsys`     | Nsight Systems timeline (kernel-level breakdown)              |
| `profile-ncu-ntt`  | Nsight Compute: NTT kernel roofline, occupancy                |
| `profile-ncu-ks`   | Nsight Compute: key-switching kernel roofline                 |
| `scaling-bert`     | BERT-base latency across N GPUs (1,2,4,8)                     |
| `scaling-bootstrap`| Bootstrapping latency across N GPUs                          |
| `algo-compare`     | Input Broadcast vs Output Aggregation key-switching           |
| `nccl-bandwidth`   | Raw NCCL AllGather/AllReduce/Broadcast bandwidth              |
| `level-trace`      | Level budget tracing through BERT layers                      |

## Directory Layout Per Run

```
results/YYYY-MM-DD_<hw>_<exp>/
├── metadata.json       # Machine-readable: instance type, git commit, flags, etc.
├── notes.md            # Human notes: what we ran, what we expected, what we saw
├── raw/                # Unmodified output files (never edit these)
│   ├── *.csv           # Benchmark output CSVs
│   ├── *.nsys-rep      # Nsight Systems report (binary)
│   ├── *.ncu-rep       # Nsight Compute report (binary)
│   └── stdout.txt      # Captured terminal output
└── processed/          # Derived files: parsed CSVs, summary tables
    └── summary.csv
```

Use `experiments/results/template/` as your starting point for each new run:
```bash
cp -r experiments/results/template experiments/results/YYYY-MM-DD_<hw>_<exp>
```

---

## Completed Runs

| Date | Hardware | Experiment | Key Finding | Dir |
|------|----------|------------|-------------|-----|
| 2026-03-29 | L4 (g6.xlarge) | baseline-ops | Phantom CKKS: multiply=30ms, rotate=29.5ms, rescale=1.4ms, add=566µs @N=65536,L=20. NCCL AllGather 21MB=21µs. | `2026-03-29_l4_baseline-ops` |

---

## Plots Index

Plots live in `experiments/plots/YYYY-MM-DD_<description>.pdf`.

| Date | File | What It Shows |
|------|------|---------------|
| *(none yet)* | | |

---

## Notes for Future Claude Instances

- **Always fill in `metadata.json`** before running — it captures the git commit so results are reproducible.
- **Never edit files in `raw/`** — they are the ground truth. Put derived/cleaned files in `processed/`.
- **Add a row to the table above** for every run, even failed ones (note what failed and why).
- **Plots go in `experiments/plots/`** (not inside run directories) because one plot often combines multiple runs.
- To generate plots: `python3 experiments/plot_scaling.py --input experiments/results/ --output experiments/plots/YYYY-MM-DD_bert_scaling.pdf`
