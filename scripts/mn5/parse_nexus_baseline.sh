#!/bin/bash
# Parses NEXUS baseline output (slurm_nexus_baseline.sh) into per-op timings.
# Usage: parse_nexus_baseline.sh /path/to/nexus_baseline_<JOBID>.out

if [ -z "$1" ]; then
  echo "Usage: $0 <nexus_baseline_*.out>"
  exit 1
fi

LOG="$1"
if [ ! -f "$LOG" ]; then
  echo "ERROR: $LOG not found"
  exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo " Parsed per-op timings from $LOG"
echo "═══════════════════════════════════════════════════════════"

for OP in MatMul Argmax SoftMax LayerNorm GELU; do
  echo ""
  echo "[$OP]"
  grep -E "^\[$OP\]" "$LOG" | awk -F'takes:' '{
    n=split($2,a," ");
    val=a[2]; unit=a[3];
    print "  trial " NR ": " val " " unit
    sum += val; n_trials++;
    vals[NR] = val
  }
  END {
    if (n_trials > 0) {
      mean = sum/n_trials
      for (i=1; i<=n_trials; i++) { sd += (vals[i]-mean)^2 }
      sd = (n_trials > 1) ? sqrt(sd/(n_trials-1)) : 0
      printf "  → mean: %.2f, σ: %.2f, n=%d\n", mean, sd, n_trials
    }
  }'
done

echo ""
echo "[Bootstrap]"
grep -E "Bootstrapping took:" "$LOG" | awk -F'took:' '{
  v = $2; gsub(/s.*/, "", v); gsub(/[ \t]/,"",v);
  print "  trial " NR ": " v " s"
  sum += v; n++; vals[NR] = v
}
END {
  if (n > 0) {
    mean = sum/n
    for (i=1; i<=n; i++) sd += (vals[i]-mean)^2
    sd = (n > 1) ? sqrt(sd/(n-1)) : 0
    printf "  → mean: %.4f s, σ: %.4f s, n=%d\n", mean, sd, n
  }
}'

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " MAE / Average Error rows for sanity check:"
echo "═══════════════════════════════════════════════════════════"
grep -E "Mean [Aa]bsolute [Ee]rror|Mean absolute error|Average Error" "$LOG"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " GPU utilization (multi-GPU check):"
echo "═══════════════════════════════════════════════════════════"
GPU_LOG="${LOG%.out}_gpu_util.log"
if [ -f "$GPU_LOG" ]; then
  echo "  source: $GPU_LOG"
  awk -F', *' 'NR>1 && $2 ~ /^[0-9]+$/ && $2+0 > 0 {print "  GPU "$1" util "$2" mem "$3}' "$GPU_LOG" | sort -u | head -40
else
  echo "  (no GPU log found at $GPU_LOG)"
fi
