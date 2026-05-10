#!/bin/bash
# Submit the full paper-validation benchmark battery on MN5.
# Run from: /gpfs/projects/etur02/hkanpak/Comp390Project
# Order matches docs/RALPH_LOOP_HANDOFF.md §"Validation order".
#
# Usage:
#   bash scripts/mn5/run_paper_battery.sh        # submit everything
#   bash scripts/mn5/run_paper_battery.sh --mae  # submit only MAE check (Step 1)
set -euo pipefail

cd "$(dirname "$0")/../.."
SCRIPTS=scripts/mn5
LOG=/gpfs/projects/etur02/hkanpak/logs

# Pre-flight: required binaries must exist (built via `make -j20 ...`).
# Phantom is ExternalProject without BUILD_ALWAYS, so warn if libPhantom.so
# looks older than the modified vendor source.
check_binary() {
  local bin=$1
  if [[ ! -x build/bin/$bin ]]; then
    echo "ERROR: build/bin/$bin not found. Build first:"
    echo "  cmake --build build --target phantom_ext -- -j20"
    echo "  cd build && make -j20 dist_bootstrap_bench bert_dks_multigpu bert_encoder_multigpu"
    exit 1
  fi
}
check_binary dist_bootstrap_bench
check_binary bert_dks_multigpu
check_binary bert_encoder_multigpu

# Phantom staleness sanity check
if [[ -f build/phantom_build/lib/libPhantom.so ]]; then
  newest_phantom_src=$(find vendor/phantom/src vendor/phantom/include -name '*.cu' -o -name '*.cuh' 2>/dev/null | xargs ls -t 2>/dev/null | head -1)
  if [[ -n "$newest_phantom_src" && "$newest_phantom_src" -nt build/phantom_build/lib/libPhantom.so ]]; then
    echo "⚠ WARNING: $newest_phantom_src is newer than libPhantom.so."
    echo "  T-MODUP / Phantom edits may not be in the linked library. Rebuild Phantom:"
    echo "    cmake --build build --target phantom_ext -- -j20"
    echo "    cd build && make -j20 dist_bootstrap_bench bert_dks_multigpu bert_encoder_multigpu"
    echo ""
    read -r -p "Continue anyway? [y/N] " ans
    [[ "$ans" =~ ^[Yy]$ ]] || exit 1
  fi
fi

submit() {
  local script=$1 label=$2
  local jid
  jid=$(sbatch --parsable "$script")
  printf "  %-30s  job=%s   log=%s/${label}_%s.out\n" "$label" "$jid" "$LOG" "$jid"
  echo "$jid"
}

echo "════════════════════════════════════════════════════════════"
echo "  multiNEXUS paper battery — submitting jobs"
echo "  $(date)"
echo "════════════════════════════════════════════════════════════"

echo ""
echo "Step 1: MAE sanity check (correctness — DO NOT skip)"
mae_jid=$(submit "$SCRIPTS/slurm_dks_bootstrap.sh" "dks_bootstrap")

if [[ "${1:-}" == "--mae" ]]; then
  echo ""
  echo "MAE-only mode. Wait for job $mae_jid to finish, then:"
  echo "  grep -E 'MAE|mae' $LOG/dks_bootstrap_${mae_jid}.out"
  echo "Expected: MAE = 2.25e-6"
  exit 0
fi

echo ""
echo "Step 2-3: Single-layer BERT (5 runs)"
for i in 1 2 3 4 5; do
  submit "$SCRIPTS/slurm_bert_dks.sh" "bert_dks_run${i}" >/dev/null
done

echo ""
echo "Step 4: 12-layer BERT (T-12LAYER-OPT)"
twelve_jid=$(submit "$SCRIPTS/slurm_bert_12layer_dks.sh" "bert_12layer_dks")

echo ""
echo "Step 5: T-NEXUS comparison at N=32768"
nexus_jid=$(submit "$SCRIPTS/slurm_bert_encoder_multigpu.sh" "bert_encoder_n32k")

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All jobs submitted. Track with:"
echo "    squeue -u \$USER"
echo "    tail -f $LOG/dks_bootstrap_${mae_jid}.out"
echo ""
echo "  After Step 1 finishes, IMMEDIATELY verify:"
echo "    grep -E 'MAE' $LOG/dks_bootstrap_${mae_jid}.out"
echo "  If MAE > 1e-3, kill remaining jobs (scancel) — T-MODUP indexing is wrong."
echo ""
echo "  Granular trace (T-TRACE Step 6) is optional — submit manually:"
echo "    sbatch $SCRIPTS/slurm_trace_nsys.sh"
echo "════════════════════════════════════════════════════════════"
