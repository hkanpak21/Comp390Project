#!/bin/bash
# Lane NCU-PROFILE — Extract human-readable summary + key metric tables from
# an Nsight Compute report file (.ncu-rep).
#
# Usage:
#   ./parse_ncu_summary.sh <report.ncu-rep> [<output_dir>]
#
# Produces, in <output_dir> (default = report's directory):
#   summary.txt         -- ncu --import --print-summary per-kernel (text)
#   metrics.csv         -- raw per-kernel CSV with Compute/Memory/Occupancy/L2/DRAM
#   top_kernels.txt     -- kernels ranked by Duration (top 20)
#
# Designed to run on MN5 head node OR a compute node — `ncu` only needs to
# read the report file; no GPU access required.

set -e

REPORT="$1"
if [ -z "${REPORT}" ]; then
    echo "Usage: $0 <report.ncu-rep> [<output_dir>]" >&2
    exit 2
fi
if [ ! -f "${REPORT}" ]; then
    echo "ERROR: report file not found: ${REPORT}" >&2
    exit 1
fi

OUTDIR="${2:-$(dirname "${REPORT}")}"
mkdir -p "${OUTDIR}"

echo ">>> Parsing ${REPORT}"
echo ">>> Output dir: ${OUTDIR}"

# Make sure ncu is on PATH (helpful when invoked from cron/sbatch)
if ! command -v ncu >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        module load cuda/12.8 2>/dev/null || true
    fi
fi

if ! command -v ncu >/dev/null 2>&1; then
    echo "ERROR: ncu not on PATH. Run 'module load cuda/12.8' first." >&2
    exit 1
fi

echo ""
echo ">>> 1/3  Per-kernel summary (text) -> summary.txt"
ncu --import "${REPORT}" --print-summary per-kernel \
    > "${OUTDIR}/summary.txt" 2>&1 || true

echo ">>> 2/3  Per-kernel metrics (CSV)  -> metrics.csv"
# Pull the headline-roofline metrics that we want for every kernel. Names are
# the canonical NCU metric IDs (--query-metrics for the full list).
METRICS=(
    "gpu__time_duration.sum"                                # Duration
    "sm__throughput.avg.pct_of_peak_sustained_elapsed"      # SM Throughput %
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"  # Memory Throughput %
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"    # DRAM Throughput %
    "dram__bytes.sum.per_second"                            # DRAM B/s
    "sm__warps_active.avg.pct_of_peak_sustained_active"     # Achieved Occupancy
    "lts__t_sector_hit_rate.pct"                            # L2 Cache Hit %
    "smsp__inst_executed.sum"                               # Instructions
    "launch__registers_per_thread"                          # Registers / thread
    "launch__shared_mem_per_block_static"                   # Shared mem / block
)
METRICS_CSV=$(IFS=,; echo "${METRICS[*]}")

ncu --import "${REPORT}" \
    --csv \
    --metrics "${METRICS_CSV}" \
    --page raw \
    > "${OUTDIR}/metrics.csv" 2>&1 || \
ncu --import "${REPORT}" \
    --csv \
    --page details \
    > "${OUTDIR}/metrics.csv" 2>&1 || \
echo "WARN: --metrics extraction failed; falling back to default details" >&2

echo ">>> 3/3  Top kernels by duration -> top_kernels.txt"
# Parse the metrics CSV (columns include Kernel Name + gpu__time_duration.sum)
# and rank by duration. Use Python csv module (kernel names contain commas
# inside `(args)`, so awk -F',' breaks).
python3 - "${OUTDIR}/metrics.csv" "${OUTDIR}/top_kernels.txt" <<'PYEOF' 2>&1 || true
import csv, sys
inp, outp = sys.argv[1], sys.argv[2]
WANT = {
    "Kernel Name":                                                     "kn",
    "gpu__time_duration.sum":                                          "dur_us",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":                "sm",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed":"mem",
    "dram__bytes.sum.per_second":                                      "dram",
    "sm__warps_active.avg.pct_of_peak_sustained_active":               "occ",
    "lts__t_sector_hit_rate.pct":                                      "l2",
    "launch__registers_per_thread":                                    "regs",
}
rows = []
with open(inp) as f:
    rd = csv.reader(f)
    header = next(rd)
    units  = next(rd, None)
    idx = {col: header.index(name) for name, col in WANT.items() if name in header}
    for row in rd:
        if len(row) < 2: continue
        try:
            kn = row[idx["kn"]].split("(", 1)[0]
            d  = float(row[idx["dur_us"]])
            sm = float(row[idx["sm"]]) if "sm" in idx else 0.0
            mem= float(row[idx["mem"]]) if "mem" in idx else 0.0
            dr = float(row[idx["dram"]]) if "dram" in idx else 0.0
            oc = float(row[idx["occ"]]) if "occ" in idx else 0.0
            l2 = float(row[idx["l2"]]) if "l2" in idx else 0.0
            rg = int(float(row[idx["regs"]])) if "regs" in idx else 0
            rows.append((d, sm, mem, dr, oc, l2, rg, kn))
        except (ValueError, KeyError, IndexError):
            continue
rows.sort(key=lambda r: r[0], reverse=True)
with open(outp, "w") as f:
    f.write("# Per-launch metrics ranked by duration (parsed from metrics.csv)\n")
    f.write("# Cols: Dur_us  SM%  Mem%  DRAM_TB/s  Occ%  L2_Hit%  Regs/Th  Kernel\n")
    f.write("\n")
    for r in rows[:40]:
        d, sm, mem, dr, oc, l2, rg, kn = r
        f.write(f"  {d:8.2f}  {sm:5.1f}  {mem:5.1f}  {dr:7.3f}  {oc:5.1f}  {l2:6.1f}  {rg:4d}  {kn}\n")
print(f"Wrote {len(rows[:40])} kernels to {outp}")
PYEOF

echo ""
echo "Done. Files:"
ls -la "${OUTDIR}/summary.txt" "${OUTDIR}/metrics.csv" "${OUTDIR}/top_kernels.txt" 2>&1

echo ""
echo "Quick preview (first 50 lines of summary):"
head -50 "${OUTDIR}/summary.txt" || true
