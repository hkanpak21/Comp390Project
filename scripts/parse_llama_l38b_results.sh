#!/bin/bash
# scripts/parse_llama_l38b_results.sh
#
# Lane LLAMA-NEXUS-MATCH — pulls log files for the 3 LLaMA-3-8B jobs from
# MN5, drops them in experiments/results/<dir>/raw.out, and prints the
# headline number + comparison vs NEXUS 51.84 s.
#
# Usage:  bash scripts/parse_llama_l38b_results.sh
# (no args; jobids are baked in)

set -e

PROJECT=/Users/a90/Documents/COURSES/SPRING2026/Comp390/Comp390Project

declare -A jobs=(
    [40369087]="2026-05-10_h100x1_llama-3-8b_seq8_32layer"
    [40369088]="2026-05-10_h100x4_llama-3-8b_seq8_32layer"
    [40369089]="2026-05-10_h100x16_llama-3-8b_seq8_32layer"
)

declare -A labels=(
    [40369087]="1× H100 (apples-to-apples baseline)"
    [40369088]="4× H100 (DIRECT compare to NEXUS 4× A100 = 51.84 s)"
    [40369089]="16× H100 (unique multi-node)"
)

declare -A name_prefix=(
    [40369087]="llama_l38b_1gpu"
    [40369088]="llama_l38b_4gpu"
    [40369089]="llama_l38b_16gpu"
)

NEXUS_TARGET_S=51.84

echo "═══ Lane LLAMA-NEXUS-MATCH — result summary ═══"
echo ""

for jid in 40369087 40369088 40369089; do
    dir="${PROJECT}/experiments/results/${jobs[$jid]}"
    mkdir -p "$dir"
    log="${dir}/raw.out"
    err="${dir}/raw.err"
    prefix="${name_prefix[$jid]}"

    rsync -az "mn5-gpu:/gpfs/projects/etur02/hkanpak/logs/${prefix}_${jid}.out" \
              "$log" 2>/dev/null || true
    rsync -az "mn5-gpu:/gpfs/projects/etur02/hkanpak/logs/${prefix}_${jid}.err" \
              "$err" 2>/dev/null || true

    if [ ! -s "$log" ]; then
        echo "JOB $jid (${labels[$jid]}): log not yet available"
        echo ""
        continue
    fi

    state=$(ssh mn5-gpu "sacct -X -j $jid -o State --noheader -P 2>/dev/null" 2>/dev/null | tr -d ' ' | head -1)
    elapsed=$(ssh mn5-gpu "sacct -X -j $jid -o Elapsed --noheader -P 2>/dev/null" 2>/dev/null | tr -d ' ' | head -1)

    total_s=$(grep -oE 'Total:[ ]+[0-9.]+ ms = [0-9.]+ s' "$log" | tail -1 | grep -oE '[0-9.]+ s' | head -1 | grep -oE '[0-9.]+')
    compute_s=$(grep -oE 'Compute:[ ]+[0-9.]+ ms' "$log" | tail -1 | grep -oE '[0-9.]+' | awk '{printf "%.2f\n", $1/1000.0}')

    if [ -z "$total_s" ]; then
        # multinode-style: "Compute (max across ranks): X ms = Y s"
        total_s=$(grep -oE 'Compute \(max across ranks\):[ ]+[0-9.]+ ms = [0-9.]+ s' "$log" | tail -1 | grep -oE '= [0-9.]+ s' | grep -oE '[0-9.]+')
        compute_s=$total_s  # for multinode, compute IS the headline
    fi

    echo "JOB $jid (${labels[$jid]}): state=$state, elapsed=$elapsed"
    echo "  Compute time: ${compute_s:-?} s (compute-only)"
    echo "  Total time:   ${total_s:-?} s (compute + setup)"

    if [ -n "$total_s" ] && [ "$jid" == "40369088" ]; then
        ratio=$(echo "scale=3; $total_s / $NEXUS_TARGET_S" | bc)
        echo "  vs NEXUS 4× A100 51.84 s: ratio = $ratio (×$ratio of NEXUS time)"
    fi
    echo ""
done

echo "═══ Done. Raw outputs in experiments/results/2026-05-10_h100x*_llama-3-8b_seq8_32layer/ ═══"
