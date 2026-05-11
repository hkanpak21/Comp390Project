# scripts/regression — multiNEXUS regression harness

Slice **I1** (correctness) and the I2-prep performance harness from
`docs/archive/PRD_PRESENTATION_AND_HP_BERT.md`. Designed for nightly CI and
for local sanity checks after each new lane (Group C / D / F / G / H) lands
a binary.

## Files

| File | Purpose |
|---|---|
| `run_correctness.sh` | Runs each benchmark in `build/bin/`, parses MAE from stdout, asserts MAE ≤ threshold (default `2.25e-6`). |
| `run_perf.sh` | Runs each benchmark, measures wall-clock time, flags regressions > 5% vs `baselines.json`. |
| `baselines.json` | Per-binary MAE thresholds and wall-clock baselines. Single source of truth for both scripts. |
| `logs/` | Per-run stdout captures (auto-created). One subdirectory per invocation. Safe to delete. |

## Quick start

```bash
# Correctness gate (default — all known binaries, default thresholds)
./scripts/regression/run_correctness.sh

# Single binary with a tighter threshold
./scripts/regression/run_correctness.sh --binary bert_dks_multigpu --mae-threshold 1e-7

# Performance gate
./scripts/regression/run_perf.sh

# Allow looser regression budget (e.g. shared CI runner)
./scripts/regression/run_perf.sh --regression-pct 10
```

## What the scripts assume

- A built tree at `<repo>/build/bin/` (override with `--build-dir`).
- Each benchmark binary that ships MAE output prints a line containing the
  literal token `MAE` followed by a numeric value (decimal or scientific
  notation). `extract_max_mae` in `run_correctness.sh` keeps the *largest*
  MAE seen across the run, so the strictest measurement gates the harness
  (matches existing `bert_e2e_multigpu.cu` and `bert_ops_test.cu` patterns).
- Missing binaries SKIP rather than FAIL. This is deliberate: the I1 harness
  is intended to ship ahead of `bert_hp_multigpu` (slice S16) and
  `phantom_threadsafe_smoke` (S15), so we want it usable right away on
  whatever subset of binaries already exists.

## Exit codes

Both scripts use:
- `0` — all present binaries passed.
- `1` — at least one present binary failed/regressed.
- `2` — CLI or configuration error.

This matches the slice I1 acceptance criterion: *"Returns nonzero on any
failure. Suitable for CI / nightly."*

## Adding a new benchmark

1. Build the binary into `build/bin/<name>`.
2. Add `<name>` to `DEFAULT_BINARIES=(...)` in both `run_correctness.sh`
   and `run_perf.sh`.
3. Add an MAE entry to `baselines.json` (top-level + inside
   `mae_thresholds`).
4. Capture the wall-clock baseline with `--update-baseline`:
   ```bash
   ./scripts/regression/run_perf.sh --update-baseline --binary <name>
   ```
   then paste the printed entry into the `wall_clock_seconds` block in
   `baselines.json`.

## Updating thresholds

Be deliberate. Both regressions and accuracy drift should be investigated,
not silenced. When the baseline genuinely shifts (new strategy lands, new
hardware, intentional algorithmic change):

1. Capture the new measurement(s).
2. Edit `baselines.json` in a single commit that explains *why* the
   baseline moved — link the slice ID or the perf-surface analysis section
   that justifies the change.
3. Cross-reference the change in `docs/PER_OP_VS_NEXUS.md` so future
   readers understand the lineage.

The non-negotiable correctness floor is **MAE ≤ 2.25e-6** for any
ciphertext-bearing benchmark. That value comes from Phase 4b and is the
threshold cited in the paper. Do not relax it without explicit advisor sign-off.

## Per-binary defaults (current)

| Binary | MAE threshold | Wall-clock baseline (s) |
|---|---:|---:|
| `bert_dks_multigpu` | 2.25e-6 | 115.7 |
| `bert_hp_multigpu` | 2.25e-6 | 370.0 |
| `phantom_threadsafe_smoke` | 1e-5 | 30.0 |

Wall-clock baselines for binaries that do not yet exist are placeholder
projections taken from `docs/archive/MULTIGPU_SCALING_PLAN.md` and
`docs/archive/PRD_PRESENTATION_AND_HP_BERT.md`. Replace with measured
medians of 3 trials once the binary lands on MN5.

## CI integration sketch

```yaml
# .github/workflows/nightly.yml (when CI is set up)
- name: Build
  run: cmake -B build && cmake --build build -j
- name: Correctness
  run: ./scripts/regression/run_correctness.sh
- name: Performance
  run: ./scripts/regression/run_perf.sh --regression-pct 10
```

A non-zero exit from either script fails the job. Per-binary stdout is
preserved under `scripts/regression/logs/` and should be uploaded as a
build artifact for post-mortem.

## Local prerequisites

- `bash` ≥ 4
- `awk` (POSIX)
- `grep`, `sed` (POSIX)
- Optional: `jq` — used when present for cleaner JSON parsing; the scripts
  fall back to a small grep/sed parser if it is missing (so the harness
  works on a bare CI runner with nothing installed).
- Optional: `shellcheck` — run `shellcheck scripts/regression/*.sh` to
  lint before committing.
