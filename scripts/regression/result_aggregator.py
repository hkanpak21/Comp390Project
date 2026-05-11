#!/usr/bin/env python3
"""Per-op result aggregator for the multiNEXUS paper.

Pure function on file inputs: given a benchmark `.out` / `stdout.txt` log
produced by one of the `*_align_*` or `*_mgpu_align*` binaries, parse the
stdout into a structured per-op record:

    {
      "op": str | None,                          # e.g. "bootstrap", "matmul"
      "gpus": int | None,                        # from --n-gpus invocation line
      "per_call_ms": float | None,               # median per-call (per-GPU)
      "throughput_inferences_per_s": float | None,  # derived: 1000 / effective
      "mae": float | None,                       # worst MAE across threads
      "jobid": str | None,                       # from filename or Job: header
      "log_path": str,                           # absolute path of the log
      "nvtx_breakdown": dict[str, float] | None, # HP-BERT per-op (ms summed)
    }

PRD module sketch 2 (see `docs/prd/PRD-multiNEXUS-paper.md` "Module
Sketches" §2): "Pure function on file inputs ... used by the paper-table
generation. Recommended for tests with checked-in fixture files."

Usage as a library:
    from result_aggregator import parse_benchmark_log
    rec = parse_benchmark_log("experiments/results/.../raw/bootstrap_40369736.out")
    if rec is not None:
        print(rec["per_call_ms"])

Usage as a CLI:
    python result_aggregator.py --log <path>
        # prints one JSON record; exits 0 on success, 1 on malformed log

    python result_aggregator.py --dir <experiments/results/<dirname>>
        # walks raw/ for *.out and stdout.txt, prints a JSON array
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import TypedDict


# Filename heuristics: the binaries under src/benchmarks all match either
#   <op>_align_*        (single-GPU alignment lane)
#   <op>_mgpu_align*    (data-parallel multi-GPU lane)
#   <op>_mgpu_16gpu*    (16-GPU multinode lane)
#   <op>_mgpu_nsys*     (nsys-wrapped multi-GPU lane)
#   bert_hp_*           (HP-BERT chained)
#   bert_dks_*          (DKS chained — reference path only)
# The leading token before the first underscore is the op name we want.
_OP_FROM_NAME_RE = re.compile(
    r"^(?P<op>[a-z0-9]+(?:_[a-z]+)?)(?:_align|_mgpu|_smoke|_diagnose|_v\d|_n\d+k)",
)
# JOBID is always an 8-digit suffix before .out / .nvtxsum.txt / etc.
_JOBID_FROM_NAME_RE = re.compile(r"_(?P<jobid>\d{7,9})(?:\.|$)")

# Stdout-body regexes (case-insensitive where the binary varies wording).
_GPUS_RE = re.compile(r"--n-gpus\s+(?P<n>\d+)")
_JOBID_HEADER_RE = re.compile(r"\bJob:\s*(?P<jobid>\d{7,9})\b")
# MAE patterns. Several phrasings show up across the binaries:
#   "worst MAE = 1.2e-06"                     (per-op smoke)
#   "MAE_post_bootstrap (max across threads): 2.164e-06"  (bootstrap_mgpu)
#   "MAE_first=2.164e-06 OK"                  (per-thread bootstrap_mgpu)
#   "[Phase 2] MAE (GPU 0 thread vs single-GPU ref, 4096 slots) = 5.918e+05"
#   "MAE = 1.5e-05"                           (generic align)
# We try them in priority order: headline "worst" / "post_bootstrap" first
# (these are the paper-quality number), then anything tagged `MAE = ...`,
# then fall back to a per-thread `MAE_first=`.
_FLOAT_RE = r"[\-+]?\d+(?:\.\d+)?(?:[eE][\-+]?\d+)?"
_MAE_PATTERNS = (
    # Headline "worst MAE = X" or "MAE_post_bootstrap ...: X" — paper number.
    re.compile(
        rf"(?:worst\s+MAE|MAE_post_bootstrap)[^\n]*?[:=]\s*(?P<mae>{_FLOAT_RE})"
    ),
    # Generic single-line "MAE = X" or "MAE (...) = X".
    re.compile(rf"\bMAE\b[^=\n]*=\s*(?P<mae>{_FLOAT_RE})"),
    # Per-thread "MAE_first=X".
    re.compile(rf"MAE_first\s*=\s*(?P<mae>{_FLOAT_RE})"),
)
# Per-call: prefer "Per-call median (per-GPU): 263.67 ms" (mgpu binaries);
# fall back to single-GPU "single-GPU median = 18509.2 ms" or
# "single-GPU median = 848.4 ms".
_PER_CALL_MGPU_RE = re.compile(
    r"Per-call\s+median[^:]*:\s*(?P<v>[\d.]+)\s*ms", re.IGNORECASE
)
_PER_CALL_SINGLE_RE = re.compile(
    r"single-GPU\s+median\s*=\s*(?P<v>[\d.]+)\s*ms", re.IGNORECASE
)
_PER_CALL_MEAN_RE = re.compile(
    r"mean\s+per-call:\s*(?P<v>[\d.]+)\s*ms", re.IGNORECASE
)
# Throughput: native phrasing if a binary ever emits "throughput: X
# inferences/s"; otherwise we derive from "Effective per-call (wall/N)".
_THROUGHPUT_NATIVE_RE = re.compile(
    r"throughput:\s*(?P<v>[\d.]+)\s*inferences/s", re.IGNORECASE
)
_EFFECTIVE_PER_CALL_RE = re.compile(
    r"Effective\s+per-call[^:]*:\s*(?P<v>[\d.]+)\s*ms", re.IGNORECASE
)
# HP-BERT op-breakdown table line:
#   "  Bootstrap #1         13018.3 ms    1084.9 ms/head    22.8%"
_NVTX_ROW_RE = re.compile(
    r"^\s{2,}(?P<name>[A-Za-z][A-Za-z0-9 \*#/^]+?)\s{2,}"
    r"(?P<ms>[\d.]+)\s*ms\s+[\d.]+\s*ms/head\s+[\d.]+%"
)


class OpRecord(TypedDict, total=False):
    op: str | None
    gpus: int | None
    per_call_ms: float | None
    throughput_inferences_per_s: float | None
    mae: float | None
    jobid: str | None
    log_path: str
    nvtx_breakdown: dict[str, float] | None


def _extract_op_from_filename(basename: str) -> str | None:
    """Pull the leading op token from a benchmark log filename.

    Examples:
        bootstrap_mgpu_align_40369736.out -> "bootstrap"
        matmul_align_40368129.out         -> "matmul"
        argmax_align_n32k_40369741.out    -> "argmax"
        smoke_40369741.out                -> None (smoke is not an op)
        bert_hp_mnode_n32k_40366927.out   -> "bert" (chained, not a single op)
        layernorm_mgpu_nsys_40371363.out  -> "layernorm"
    """
    stem = basename
    for ext in (".out", ".txt", ".log"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    # Carve off the trailing _<JOBID> if present so the op regex sees the
    # bare benchmark-name prefix.
    stem = re.sub(r"_\d{7,9}$", "", stem)
    head = stem.split("_", 1)[0]
    if head in {"smoke", "stdout"}:
        return None
    return head if re.fullmatch(r"[a-z][a-z0-9]*", head) else None


def _extract_jobid_from_filename(basename: str) -> str | None:
    m = _JOBID_FROM_NAME_RE.search(basename)
    return m.group("jobid") if m else None


def _parse_nvtx_breakdown(text: str) -> dict[str, float] | None:
    """Parse the HP-BERT '── Per-operation timing ──' table, if present."""
    if "Per-operation timing" not in text:
        return None
    breakdown: dict[str, float] = {}
    in_table = False
    for line in text.splitlines():
        if "Per-operation timing" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.strip().startswith("═") or line.strip().startswith("──"):
            if breakdown:
                break
            continue
        m = _NVTX_ROW_RE.match(line)
        if m:
            breakdown[m.group("name").strip()] = float(m.group("ms"))
        elif line.strip() == "":
            if breakdown:
                break
    return breakdown or None


def _coerce_float(text: str, pattern: re.Pattern[str]) -> float | None:
    m = pattern.search(text)
    if not m:
        return None
    try:
        return float(m.group(m.lastindex or 1))
    except (TypeError, ValueError):
        return None


def parse_benchmark_log(log_path: str) -> OpRecord | None:
    """Parse a single benchmark stdout/.out log into a structured record.

    Returns None if the log is unreadable or has no recognizable benchmark
    output (i.e. none of {per-call line, MAE line, n-gpus line, headline
    banner, NVTX op breakdown} is present). Otherwise returns an OpRecord
    where missing fields are explicit None — callers should not assume any
    given field is populated.
    """
    if not os.path.isfile(log_path):
        return None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return None
    if not text.strip():
        return None

    basename = os.path.basename(log_path)
    op = _extract_op_from_filename(basename)
    jobid = _extract_jobid_from_filename(basename)
    if jobid is None:
        m = _JOBID_HEADER_RE.search(text)
        if m:
            jobid = m.group("jobid")

    gpus: int | None = None
    m_gpus = _GPUS_RE.search(text)
    if m_gpus:
        try:
            gpus = int(m_gpus.group("n"))
        except ValueError:
            gpus = None

    # per_call_ms: prefer the multi-GPU phrasing, then single-GPU phrasing,
    # then a generic "mean per-call".
    per_call_ms = _coerce_float(text, _PER_CALL_MGPU_RE)
    if per_call_ms is None:
        per_call_ms = _coerce_float(text, _PER_CALL_SINGLE_RE)
    if per_call_ms is None:
        per_call_ms = _coerce_float(text, _PER_CALL_MEAN_RE)

    # throughput: native first; else derived from effective per-call. The
    # benchmarks emit "Effective per-call (wall/N)" which is the per-GPU
    # wall divided by N total calls — for a data-parallel run, total
    # throughput is the inverse of that.
    throughput = _coerce_float(text, _THROUGHPUT_NATIVE_RE)
    if throughput is None:
        eff = _coerce_float(text, _EFFECTIVE_PER_CALL_RE)
        if eff is not None and eff > 0:
            throughput = 1000.0 / eff

    mae: float | None = None
    for pat in _MAE_PATTERNS:
        mae = _coerce_float(text, pat)
        if mae is not None:
            break

    nvtx = _parse_nvtx_breakdown(text)

    # Sanity gate: if NONE of the structural markers fired, treat the log
    # as malformed and return None. This is what the PRD calls "returns
    # None if the log is malformed". A log with only an op name from the
    # filename but no measurements is not useful for the paper table.
    has_signal = any(
        v is not None
        for v in (per_call_ms, throughput, mae, gpus, nvtx)
    ) or "HEADLINE" in text
    if not has_signal:
        return None

    record: OpRecord = {
        "op": op,
        "gpus": gpus,
        "per_call_ms": per_call_ms,
        "throughput_inferences_per_s": throughput,
        "mae": mae,
        "jobid": jobid,
        "log_path": os.path.abspath(log_path),
        "nvtx_breakdown": nvtx,
    }
    return record


def aggregate_directory(result_dir: str) -> list[OpRecord]:
    """Aggregate every *.out and stdout.txt under <result_dir>/raw/.

    Skips files where parse_benchmark_log returns None (so a directory
    with one malformed log plus several good ones returns just the good
    ones). Order is the alphabetical filename order, which is stable.
    """
    raw_dir = os.path.join(result_dir, "raw")
    if not os.path.isdir(raw_dir):
        return []
    out: list[OpRecord] = []
    for name in sorted(os.listdir(raw_dir)):
        if not (name.endswith(".out") or name == "stdout.txt"):
            continue
        rec = parse_benchmark_log(os.path.join(raw_dir, name))
        if rec is not None:
            out.append(rec)
    return out


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--log", help="path to one benchmark stdout/.out file")
    src.add_argument(
        "--dir",
        help="path to an experiments/results/<dir>; walks raw/ for logs",
    )
    args = p.parse_args(argv)

    if args.log:
        rec = parse_benchmark_log(args.log)
        if rec is None:
            print(
                json.dumps({"error": "malformed or empty log", "log_path": args.log}),
                file=sys.stderr,
            )
            return 1
        print(json.dumps(rec, indent=2))
        return 0

    recs = aggregate_directory(args.dir)
    print(json.dumps(recs, indent=2))
    return 0 if recs else 1


if __name__ == "__main__":
    sys.exit(_cli())
