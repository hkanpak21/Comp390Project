#!/usr/bin/env bash
# run_correctness.sh — multiNEXUS correctness regression harness (slice I1).
#
# Iterates over a list of benchmark binaries that may exist in build/bin/,
# runs each, parses the worst MAE printed on stdout, and asserts the value
# is below a per-binary threshold (default 2.25e-6 — the multiNEXUS Phase 4b
# correctness floor). Returns nonzero on any failure. Suitable for nightly
# CI use; safe to run locally even when only a subset of binaries exists.
#
# The harness is intentionally permissive: when a binary is not yet built
# (e.g. bert_hp_multigpu before slice S16 lands) we SKIP it rather than
# fail, so other lanes can run the harness as soon as their slice ships.
#
# Per-binary thresholds are read from baselines.json (jq if available, with
# a grep/sed fallback so this script works on bare CI containers too).
#
# Usage:
#   ./run_correctness.sh                          # all known binaries
#   ./run_correctness.sh --binary bert_dks_multigpu
#   ./run_correctness.sh --mae-threshold 1e-5
#   ./run_correctness.sh --build-dir build_local
#   ./run_correctness.sh --help

set -euo pipefail

# ── Paths and defaults ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
BASELINES_JSON="${SCRIPT_DIR}/baselines.json"

DEFAULT_BUILD_DIR="${PROJECT_ROOT}/build"
DEFAULT_MAE_THRESHOLD="2.25e-6"
# Known benchmark binaries we care about. Each is checked with -f before
# being executed; missing binaries SKIP rather than FAIL.
DEFAULT_BINARIES=(
    bert_dks_multigpu
    bert_hp_multigpu
    phantom_threadsafe_smoke
)

# Per-run state.
BUILD_DIR="${DEFAULT_BUILD_DIR}"
USER_THRESHOLD=""
SELECTED_BINARY=""

# ── Helpers ───────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

multiNEXUS correctness regression harness. Runs each benchmark binary that
exists in <build-dir>/bin/, parses the MAE float from stdout, and asserts
MAE <= threshold (default 2.25e-6 — the Phase 4b correctness floor).

Options:
  --binary <name>           Only run this benchmark binary (default: all known)
  --mae-threshold <float>   Override threshold for ALL binaries (default: per-binary
                            from baselines.json, falling back to 2.25e-6)
  --build-dir <path>        Path to the build directory containing bin/
                            (default: \$PROJECT_ROOT/build)
  -h, --help                Show this help and exit

Exit codes:
  0  All present binaries PASSED (skipped binaries do not fail the run)
  1  At least one binary FAILED its MAE assertion
  2  CLI / configuration error

Examples:
  $(basename "$0")
  $(basename "$0") --binary bert_dks_multigpu
  $(basename "$0") --mae-threshold 1e-5 --build-dir build_release

Known binaries (skipped silently if not yet built):
$(printf '  - %s\n' "${DEFAULT_BINARIES[@]}")
EOF
}

log()  { printf '[run_correctness] %s\n' "$*"; }
warn() { printf '[run_correctness] WARN: %s\n' "$*" >&2; }
die()  { printf '[run_correctness] ERROR: %s\n' "$*" >&2; exit 2; }

# Look up the per-binary threshold from baselines.json. Prefers jq when
# available; otherwise falls back to a small grep/sed parser. Returns the
# literal string from JSON so we preserve scientific notation.
lookup_threshold() {
    local name="$1"
    local val=""
    if command -v jq >/dev/null 2>&1; then
        val="$(jq -r --arg k "$name" '.[$k] // empty' "${BASELINES_JSON}" 2>/dev/null || true)"
    fi
    if [[ -z "${val}" ]]; then
        # Grep/sed fallback: pull the first numeric value following "name":
        # tolerates: "name": 2.25e-6 / "name":2.25e-06 / spaces / trailing comma.
        val="$(grep -E "\"${name}\"[[:space:]]*:" "${BASELINES_JSON}" 2>/dev/null \
               | head -n1 \
               | sed -E 's/.*:[[:space:]]*([0-9eE.+-]+).*/\1/' || true)"
    fi
    if [[ -z "${val}" ]]; then
        val="${DEFAULT_MAE_THRESHOLD}"
    fi
    printf '%s' "${val}"
}

# Pull the largest MAE value from a benchmark log. Matches lines like:
#   MAE: 0.000001
#   GELU MAE 1.23e-06
#   MatMul MAE:         0.000123 PASS
# We keep the *maximum* MAE seen so the strictest measurement gates the run.
extract_max_mae() {
    local logfile="$1"
    # Capture the float that follows MAE on each matching line. Handles both
    # decimal and scientific notation; ignores anything not a number.
    grep -Eo 'MAE[^0-9eE+\-]*[+-]?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?' "${logfile}" \
        | sed -E 's/MAE[^0-9eE+\-]*//' \
        | awk 'BEGIN{m=-1} {v=$1+0; if (v>m) m=v} END{ if (m>=0) printf "%.10g", m }'
}

# Compare two floats: returns 0 (pass) iff value <= threshold.
mae_within_threshold() {
    local value="$1"
    local threshold="$2"
    awk -v v="${value}" -v t="${threshold}" 'BEGIN{ exit !(v+0 <= t+0) }'
}

# ── CLI parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)
            [[ $# -ge 2 ]] || die "--binary requires a value"
            SELECTED_BINARY="$2"
            shift 2
            ;;
        --mae-threshold)
            [[ $# -ge 2 ]] || die "--mae-threshold requires a value"
            USER_THRESHOLD="$2"
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || die "--build-dir requires a value"
            BUILD_DIR="$2"
            shift 2
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

if [[ -n "${SELECTED_BINARY}" ]]; then
    BINARIES=( "${SELECTED_BINARY}" )
else
    BINARIES=( "${DEFAULT_BINARIES[@]}" )
fi

# Per-run scratch dir for stdout captures (kept for post-mortem).
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs/correctness_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

log "Build dir:       ${BUILD_DIR}"
log "Bin dir:         ${BIN_DIR}"
log "Baselines:       ${BASELINES_JSON}"
log "Log dir:         ${LOG_DIR}"
log "Default thresh:  ${USER_THRESHOLD:-${DEFAULT_MAE_THRESHOLD} (or per-binary)}"
log "Binaries:        ${BINARIES[*]}"
log "----"

# ── Main loop ─────────────────────────────────────────────────────────────────
fail_count=0
pass_count=0
skip_count=0

for bin_name in "${BINARIES[@]}"; do
    bin_path="${BIN_DIR}/${bin_name}"
    log_file="${LOG_DIR}/${bin_name}.log"

    if [[ ! -x "${bin_path}" ]]; then
        log "SKIP ${bin_name}    (not present at ${bin_path})"
        skip_count=$((skip_count + 1))
        continue
    fi

    # Threshold precedence: --mae-threshold (global) > baselines.json > default.
    if [[ -n "${USER_THRESHOLD}" ]]; then
        threshold="${USER_THRESHOLD}"
    else
        threshold="$(lookup_threshold "${bin_name}")"
    fi

    log "RUN  ${bin_name}     (threshold ${threshold})"
    # Don't crash the harness on a benchmark non-zero exit — capture it as FAIL.
    set +e
    "${bin_path}" >"${log_file}" 2>&1
    bin_rc=$?
    set -e

    if [[ ${bin_rc} -ne 0 ]]; then
        log "FAIL ${bin_name}    (binary exit code ${bin_rc}, see ${log_file})"
        fail_count=$((fail_count + 1))
        continue
    fi

    mae="$(extract_max_mae "${log_file}")"
    if [[ -z "${mae}" ]]; then
        log "FAIL ${bin_name}    (no MAE token found in stdout, see ${log_file})"
        fail_count=$((fail_count + 1))
        continue
    fi

    if mae_within_threshold "${mae}" "${threshold}"; then
        log "PASS ${bin_name}    (max MAE ${mae} <= ${threshold})"
        pass_count=$((pass_count + 1))
    else
        log "FAIL ${bin_name}    (max MAE ${mae} > ${threshold}, see ${log_file})"
        fail_count=$((fail_count + 1))
    fi
done

log "----"
log "Summary: ${pass_count} PASS, ${fail_count} FAIL, ${skip_count} SKIP"

if [[ ${fail_count} -gt 0 ]]; then
    log "OVERALL: FAIL"
    exit 1
fi

log "OVERALL: PASS"
exit 0
