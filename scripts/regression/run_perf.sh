#!/usr/bin/env bash
# run_perf.sh — multiNEXUS performance regression harness (slice I2 prep).
#
# Runs each benchmark binary under wall-clock measurement and compares against
# the per-binary baseline in baselines.json (key: wall_clock_seconds). Flags
# any binary whose wall-clock time exceeds baseline by more than the
# configurable regression-percent (default 5%).
#
# Like run_correctness.sh, missing binaries SKIP rather than FAIL — multi-lane
# work means binaries arrive incrementally and we want this harness usable
# from week 1.
#
# Usage:
#   ./run_perf.sh
#   ./run_perf.sh --binary bert_dks_multigpu
#   ./run_perf.sh --regression-pct 10
#   ./run_perf.sh --build-dir build_release
#   ./run_perf.sh --update-baseline           # rewrite baselines.json
#   ./run_perf.sh --help

set -euo pipefail

# ── Paths and defaults ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
BASELINES_JSON="${SCRIPT_DIR}/baselines.json"

DEFAULT_BUILD_DIR="${PROJECT_ROOT}/build"
DEFAULT_REGRESSION_PCT="5.0"
DEFAULT_BINARIES=(
    bert_dks_multigpu
    bert_hp_multigpu
    phantom_threadsafe_smoke
)

# Per-run state.
BUILD_DIR="${DEFAULT_BUILD_DIR}"
USER_REGRESSION_PCT=""
SELECTED_BINARY=""
UPDATE_BASELINE=0

# ── Helpers ───────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

multiNEXUS performance regression harness. Runs each benchmark binary under
wall-clock measurement and flags any binary whose run-time exceeds its
baseline (baselines.json -> wall_clock_seconds.<binary>) by more than the
regression-percent threshold (default 5%).

Options:
  --binary <name>             Only run this benchmark binary (default: all known)
  --regression-pct <float>    Override regression threshold percentage
                              (default: from baselines.json -> perf_regression_pct,
                              else 5.0)
  --build-dir <path>          Path to build directory containing bin/
                              (default: \$PROJECT_ROOT/build)
  --update-baseline           Don't compare; print measured times in a form
                              suitable for pasting into baselines.json.
                              Always exits 0.
  -h, --help                  Show this help and exit

Exit codes:
  0  No regressions detected (or --update-baseline mode)
  1  At least one binary regressed beyond the threshold
  2  CLI / configuration error

Examples:
  $(basename "$0")
  $(basename "$0") --binary bert_dks_multigpu --regression-pct 10
  $(basename "$0") --update-baseline

Known binaries (skipped silently if not yet built):
$(printf '  - %s\n' "${DEFAULT_BINARIES[@]}")
EOF
}

log()  { printf '[run_perf] %s\n' "$*"; }
warn() { printf '[run_perf] WARN: %s\n' "$*" >&2; }
die()  { printf '[run_perf] ERROR: %s\n' "$*" >&2; exit 2; }

# Pull a numeric value from baselines.json by JSON-path-ish key. We support
# the two shapes the file uses:
#   wall_clock_seconds.<bin>
#   perf_regression_pct
# jq is preferred when available; otherwise a grep/sed fallback handles the
# simple cases this script needs.
lookup_baseline_seconds() {
    local name="$1"
    local val=""
    if command -v jq >/dev/null 2>&1; then
        val="$(jq -r --arg k "$name" '.wall_clock_seconds[$k] // empty' "${BASELINES_JSON}" 2>/dev/null || true)"
    fi
    if [[ -z "${val}" ]]; then
        # Look for the wall_clock_seconds block, then within it the named entry.
        # Extracts only the first matching number after "name":.
        val="$(awk -v key="\"${name}\"" '
            /"wall_clock_seconds"[[:space:]]*:/ {in_block=1; next}
            in_block && /\}/ {in_block=0}
            in_block && $0 ~ key {
                # Strip everything up through the colon, then take leading number.
                sub(/.*:[[:space:]]*/, "", $0);
                gsub(/[,}].*$/, "", $0);
                print $0;
                exit
            }
        ' "${BASELINES_JSON}" 2>/dev/null || true)"
    fi
    printf '%s' "${val}"
}

lookup_regression_pct() {
    local val=""
    if command -v jq >/dev/null 2>&1; then
        val="$(jq -r '.perf_regression_pct // empty' "${BASELINES_JSON}" 2>/dev/null || true)"
    fi
    if [[ -z "${val}" ]]; then
        val="$(grep -E '"perf_regression_pct"[[:space:]]*:' "${BASELINES_JSON}" 2>/dev/null \
               | head -n1 \
               | sed -E 's/.*:[[:space:]]*([0-9eE.+-]+).*/\1/' || true)"
    fi
    if [[ -z "${val}" ]]; then
        val="${DEFAULT_REGRESSION_PCT}"
    fi
    printf '%s' "${val}"
}

# Wrap `date +%s.%N` portably. macOS `date` lacks %N — fall back to python.
now_seconds() {
    if date +%s.%N 2>/dev/null | grep -qv 'N$'; then
        date +%s.%N
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c 'import time; print(time.time())'
    else
        date +%s
    fi
}

# Float arithmetic via awk (no bc dependency).
fdelta_pct() {
    awk -v m="$1" -v b="$2" 'BEGIN{ if (b+0 == 0) {print "inf"} else {printf "%.2f", ((m-b)/b)*100} }'
}

is_regression() {
    # Returns 0 (true) if measured > baseline * (1 + pct/100)
    awk -v m="$1" -v b="$2" -v p="$3" 'BEGIN{ exit !(m+0 > b * (1 + p/100.0)) }'
}

# ── CLI parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)
            [[ $# -ge 2 ]] || die "--binary requires a value"
            SELECTED_BINARY="$2"
            shift 2
            ;;
        --regression-pct)
            [[ $# -ge 2 ]] || die "--regression-pct requires a value"
            USER_REGRESSION_PCT="$2"
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || die "--build-dir requires a value"
            BUILD_DIR="$2"
            shift 2
            ;;
        --update-baseline)
            UPDATE_BASELINE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1 (use --help)"
            ;;
    esac
done

# ── Pre-flight ────────────────────────────────────────────────────────────────
BIN_DIR="${BUILD_DIR}/bin"
if [[ ! -d "${BIN_DIR}" ]]; then
    warn "Build bin directory does not exist: ${BIN_DIR}"
    warn "Did you run cmake/make? Skipping all binaries — exit 0."
    exit 0
fi

if [[ ! -f "${BASELINES_JSON}" ]]; then
    die "baselines.json not found at ${BASELINES_JSON}"
fi

REGRESSION_PCT="${USER_REGRESSION_PCT:-$(lookup_regression_pct)}"

if [[ -n "${SELECTED_BINARY}" ]]; then
    BINARIES=( "${SELECTED_BINARY}" )
else
    BINARIES=( "${DEFAULT_BINARIES[@]}" )
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs/perf_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

log "Build dir:        ${BUILD_DIR}"
log "Bin dir:          ${BIN_DIR}"
log "Baselines:        ${BASELINES_JSON}"
log "Log dir:          ${LOG_DIR}"
log "Regression pct:   ${REGRESSION_PCT}%"
log "Mode:             $([[ ${UPDATE_BASELINE} -eq 1 ]] && echo update-baseline || echo compare)"
log "Binaries:         ${BINARIES[*]}"
log "----"

# ── Main loop ─────────────────────────────────────────────────────────────────
regress_count=0
ok_count=0
skip_count=0

# Collect measurements for the optional baseline-rewrite output.
declare -a BASELINE_OUT=()

for bin_name in "${BINARIES[@]}"; do
    bin_path="${BIN_DIR}/${bin_name}"
    log_file="${LOG_DIR}/${bin_name}.log"

    if [[ ! -x "${bin_path}" ]]; then
        log "SKIP ${bin_name}    (not present at ${bin_path})"
        skip_count=$((skip_count + 1))
        continue
    fi

    log "RUN  ${bin_name}"
    t0="$(now_seconds)"
    set +e
    "${bin_path}" >"${log_file}" 2>&1
    bin_rc=$?
    set -e
    t1="$(now_seconds)"
    elapsed="$(awk -v a="${t0}" -v b="${t1}" 'BEGIN{printf "%.3f", b-a}')"

    if [[ ${bin_rc} -ne 0 ]]; then
        log "FAIL ${bin_name}    (binary exit code ${bin_rc}, see ${log_file})"
        regress_count=$((regress_count + 1))
        continue
    fi

    BASELINE_OUT+=( "        \"${bin_name}\": ${elapsed}" )

    if [[ ${UPDATE_BASELINE} -eq 1 ]]; then
        log "TIME ${bin_name}    ${elapsed}s (baseline mode — no comparison)"
        ok_count=$((ok_count + 1))
        continue
    fi

    baseline="$(lookup_baseline_seconds "${bin_name}")"
    if [[ -z "${baseline}" ]]; then
        log "WARN ${bin_name}    no baseline in baselines.json — skipping comparison (measured ${elapsed}s)"
        ok_count=$((ok_count + 1))
        continue
    fi

    delta="$(fdelta_pct "${elapsed}" "${baseline}")"
    if is_regression "${elapsed}" "${baseline}" "${REGRESSION_PCT}"; then
        log "REGR ${bin_name}    ${elapsed}s vs baseline ${baseline}s (${delta}%, threshold ${REGRESSION_PCT}%)"
        regress_count=$((regress_count + 1))
    else
        log "OK   ${bin_name}    ${elapsed}s vs baseline ${baseline}s (${delta}%)"
        ok_count=$((ok_count + 1))
    fi
done

log "----"
log "Summary: ${ok_count} OK, ${regress_count} REGRESSED, ${skip_count} SKIP"

if [[ ${UPDATE_BASELINE} -eq 1 ]]; then
    log "Suggested wall_clock_seconds block (paste into baselines.json):"
    printf '    "wall_clock_seconds": {\n'
    n_entries=${#BASELINE_OUT[@]}
    if [[ ${n_entries} -gt 0 ]]; then
        # All entries except the last get a trailing comma; the last one does not.
        if [[ ${n_entries} -gt 1 ]]; then
            for ((i=0; i<n_entries-1; i++)); do
                printf '%s,\n' "${BASELINE_OUT[i]}"
            done
        fi
        printf '%s\n' "${BASELINE_OUT[n_entries-1]}"
    fi
    printf '    }\n'
    exit 0
fi

if [[ ${regress_count} -gt 0 ]]; then
    log "OVERALL: REGRESSED"
    exit 1
fi

log "OVERALL: OK"
exit 0
