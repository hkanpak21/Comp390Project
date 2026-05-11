"""Tests for result_aggregator.py.

PRD module sketch 2: pure function on file inputs. Per the PRD's "Testing
Decisions" note, this aggregator is "recommended for tests with checked-in
fixture files" — but since v1 of the parser is heuristic over plain stdout,
we inline three realistic fixture strings here (one bootstrap_mgpu, one
matmul single-GPU + multi-GPU, one HP-BERT NVTX-table) so the test file
is self-contained and reviewable without crawling experiments/results/.

Run from repo root:
    python -m pytest scripts/regression/test_result_aggregator.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from result_aggregator import (  # noqa: E402
    aggregate_directory,
    parse_benchmark_log,
)


# ---------------------------------------------------------------------------
# Inlined fixture log bodies. Each one is a realistic excerpt of what the
# corresponding benchmark binary emits to stdout, trimmed so the test is
# legible. All measurement-bearing lines are preserved verbatim.
# ---------------------------------------------------------------------------

FIXTURE_BOOTSTRAP_MGPU = textwrap.dedent(
    """\
    ════════════════════════════════════════════════════════════
      MGPU-BS-NSYS — bootstrap_mgpu under nsys (4× H100)
      Job: 40371362, Node: as05r3b18
    ════════════════════════════════════════════════════════════
    >>> Running nsys profile bootstrap_mgpu_align --calls 100 --n-gpus 4 ...
    [T0 / GPU 0] setup done, 25 calls scheduled
    [T0 / GPU 0] call 1/25: 270.59 ms

    ────────────── Per-thread results ──────────────
      GPU 0: 25 calls, median=270.59 ms, MAE_first=2.164e-06 OK
      GPU 1: 25 calls, median=257.33 ms, MAE_first=2.153e-06 OK
      GPU 2: 25 calls, median=266.09 ms, MAE_first=2.159e-06 OK
      GPU 3: 25 calls, median=262.00 ms, MAE_first=2.159e-06 OK

    ════════════════════════════════════════════════════════════
      MGPU-MICRO Bootstrap HEADLINE (logN=15, 4 GPUs)
    ════════════════════════════════════════════════════════════
      Total calls done:          100 / 100 (target)
      Wall-clock (all calls):    24.40 s
      Per-call median (per-GPU): 263.67 ms
      Per-call σ:                8.92 ms (3.38% of median)
      Effective per-call (wall/N): 243.99 ms
      ----
      MAE_post_bootstrap (max across threads): 2.164e-06 PASS (<0.05)
    ════════════════════════════════════════════════════════════
    Exit code: 0
    """
)

FIXTURE_MATMUL_ALIGN = textwrap.dedent(
    """\
    ════════════════════════════════════════════════════════════
      ALIGN-MatMul — NEXUS MatMul alignment, logN=13 (N=8,192)
      Job: 40368129, Node: as03r2b23
    ════════════════════════════════════════════════════════════
    >>> Running matmul_align_n8k --n-gpus 4 --trials 3 ...

    [Phase 1] Single-GPU measurement (3 trials on GPU 0)...
      trial 1/3: matmul=18465.1 ms (= 0.2885 s amortized over 64 cols)
      trial 2/3: matmul=18509.2 ms (= 0.2892 s amortized over 64 cols)
      trial 3/3: matmul=18628.3 ms (= 0.2911 s amortized over 64 cols)
    [Phase 1] single-GPU median = 18509.2 ms (σ=84.4)

    [Phase 2] Multi-GPU measurement (3 trials × 4 GPUs)...
    [Phase 2] multi-GPU wall median = 21954.1 ms (σ=591.9, n_gpus=4)
    [Phase 2] MAE (GPU 0 thread vs single-GPU ref, 4096 slots) = 5.918e+05
    [Phase 2] WARN: MAE > 1e-5 threshold (5.918e+05). Multi-GPU result diverges!

    ════════════════════════════════════════════════════════════
      ALIGN-MatMul HEADLINE (logN=13, poly_degree=8,192)
    ════════════════════════════════════════════════════════════
    Exit code: 0
    """
)

FIXTURE_HP_BERT_NVTX = textwrap.dedent(
    """\
    ════════════════════════════════════════════════════════════
      ALIGN-HP-BERT-NVTX — NVTX-scoped HP-BERT, logN=15 (N=32,768)
      Job: 40368131, Node: as05r5b26
    ════════════════════════════════════════════════════════════
    >>> Running nsys profile bert_hp_multigpu --N 32768 --n-gpus 4 --heads 12 --skip-ref ...

    ════════════════════════════════════════════════════════════
      HP-BERT result — 4 GPUs / 12 heads / 1 layer / N=32768
    ════════════════════════════════════════════════════════════
      Setup:    42904.3 ms
      Compute:  21967.3 ms (concurrent across 4 GPUs)
      Total:    64871.6 ms = 64.87 s

    ─── Per-operation timing (summed across 12 head×GPU completions) ───
      QKV MatMul             752.9 ms      62.7 ms/head     1.3%
      Q*K^T MatMul            20.9 ms       1.7 ms/head     0.0%
      Softmax                936.1 ms      78.0 ms/head     1.6%
      Bootstrap #1         13018.3 ms    1084.9 ms/head    22.8%
      LayerNorm #1          1383.3 ms     115.3 ms/head     2.4%
      GELU                   661.3 ms      55.1 ms/head     1.2%

    Exit code: 0
    """
)

FIXTURE_MISSING_FIELDS = textwrap.dedent(
    """\
    ════════════════════════════════════════════════════════════
      MGPU-MICRO Softmax HEADLINE (logN=16, 4 GPUs)
    ════════════════════════════════════════════════════════════
      Per-call median (per-GPU): 16.52 ms
      Effective per-call (wall/N): 18.09 ms
    Exit code: 0
    """
)

# Truly malformed: random unrelated text, no recognizable benchmark signal.
FIXTURE_MALFORMED = textwrap.dedent(
    """\
    Hello, world.
    This is not a benchmark log.
    Some random output without any per-call measurement or MAE line.
    """
)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


# ---------------------------------------------------------------------------
# parse_benchmark_log()
# ---------------------------------------------------------------------------

class TestParseBenchmarkLog:
    def test_bootstrap_mgpu_full_record(self, tmp_path):
        log = _write(tmp_path, "bootstrap_mgpu_align_40371362.out", FIXTURE_BOOTSTRAP_MGPU)
        rec = parse_benchmark_log(str(log))
        assert rec is not None
        assert rec["op"] == "bootstrap"
        assert rec["gpus"] == 4
        assert rec["jobid"] == "40371362"
        assert rec["per_call_ms"] == pytest.approx(263.67)
        # Throughput derived from effective per-call (1000 / 243.99).
        assert rec["throughput_inferences_per_s"] == pytest.approx(1000.0 / 243.99)
        assert rec["mae"] == pytest.approx(2.164e-06)
        assert rec["log_path"].endswith("bootstrap_mgpu_align_40371362.out")
        assert rec["nvtx_breakdown"] is None

    def test_matmul_align_record(self, tmp_path):
        log = _write(tmp_path, "matmul_align_n8k_40368129.out", FIXTURE_MATMUL_ALIGN)
        rec = parse_benchmark_log(str(log))
        assert rec is not None
        assert rec["op"] == "matmul"
        assert rec["gpus"] == 4
        assert rec["jobid"] == "40368129"
        # Single-GPU fallback regex; no Per-call-median line in this binary.
        assert rec["per_call_ms"] == pytest.approx(18509.2)
        # Matmul binary has no Effective-per-call line → no throughput.
        assert rec["throughput_inferences_per_s"] is None
        assert rec["mae"] == pytest.approx(5.918e+05)

    def test_hp_bert_nvtx_breakdown(self, tmp_path):
        log = _write(tmp_path, "hp_bert_align_nvtx_40368131.out", FIXTURE_HP_BERT_NVTX)
        rec = parse_benchmark_log(str(log))
        assert rec is not None
        assert rec["gpus"] == 4
        assert rec["jobid"] == "40368131"
        # No per-call median in HP-BERT chained output — that's expected.
        assert rec["per_call_ms"] is None
        assert rec["mae"] is None
        nvtx = rec["nvtx_breakdown"]
        assert nvtx is not None
        assert nvtx["QKV MatMul"] == pytest.approx(752.9)
        assert nvtx["Bootstrap #1"] == pytest.approx(13018.3)
        assert nvtx["LayerNorm #1"] == pytest.approx(1383.3)
        assert nvtx["GELU"] == pytest.approx(661.3)
        # Q*K^T MatMul has special characters; check it survived the regex.
        assert nvtx["Q*K^T MatMul"] == pytest.approx(20.9)

    def test_missing_fields_become_none(self, tmp_path):
        # Softmax fixture lacks --n-gpus invocation, MAE, and a JOBID in
        # the body. The parser must populate what it can and set the
        # rest to None — never raise.
        log = _write(tmp_path, "softmax_mgpu_align_smoke.out", FIXTURE_MISSING_FIELDS)
        rec = parse_benchmark_log(str(log))
        assert rec is not None
        assert rec["op"] == "softmax"
        assert rec["gpus"] is None
        assert rec["mae"] is None
        assert rec["jobid"] is None
        assert rec["per_call_ms"] == pytest.approx(16.52)
        assert rec["throughput_inferences_per_s"] == pytest.approx(1000.0 / 18.09)
        assert rec["nvtx_breakdown"] is None

    def test_malformed_log_returns_none(self, tmp_path):
        log = _write(tmp_path, "random_99999999.out", FIXTURE_MALFORMED)
        assert parse_benchmark_log(str(log)) is None

    def test_empty_log_returns_none(self, tmp_path):
        log = _write(tmp_path, "empty_99999998.out", "")
        assert parse_benchmark_log(str(log)) is None

    def test_nonexistent_path_returns_none(self):
        assert parse_benchmark_log("/no/such/path/does/not/exist.out") is None

    def test_jobid_falls_back_to_header_when_missing_from_filename(self, tmp_path):
        # stdout.txt has no JOBID in the filename — must be pulled from the
        # "Job: NNN" banner inside the body.
        log = _write(tmp_path, "stdout.txt", FIXTURE_BOOTSTRAP_MGPU)
        rec = parse_benchmark_log(str(log))
        assert rec is not None
        assert rec["jobid"] == "40371362"

    def test_op_token_extraction(self, tmp_path):
        # Filename -> op token extraction is purely lexical; confirm a few
        # representative names. (stdout.txt yields None; smoke_<JOBID>.out
        # also yields None since "smoke" is not an op.)
        cases = {
            "bootstrap_mgpu_align_40369736.out": "bootstrap",
            "matmul_align_n8k_40368129.out": "matmul",
            "argmax_align_n32k_40369741.out": "argmax",
            "layernorm_mgpu_nsys_40371363.out": "layernorm",
            "gelu_mgpu_align_40387026.out": "gelu",
            "softmax_mgpu_align_40369739.out": "softmax",
        }
        for fname, expected in cases.items():
            log = _write(tmp_path, fname, FIXTURE_MISSING_FIELDS)
            rec = parse_benchmark_log(str(log))
            assert rec is not None, fname
            assert rec["op"] == expected, fname

    def test_log_path_is_absolute(self, tmp_path):
        log = _write(tmp_path, "bootstrap_mgpu_40371362.out", FIXTURE_BOOTSTRAP_MGPU)
        rec = parse_benchmark_log(str(log))
        assert rec is not None
        assert os.path.isabs(rec["log_path"])


# ---------------------------------------------------------------------------
# aggregate_directory()
# ---------------------------------------------------------------------------

class TestAggregateDirectory:
    def test_walks_raw_subdir(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "bootstrap_mgpu_align_40371362.out").write_text(FIXTURE_BOOTSTRAP_MGPU)
        (raw / "matmul_align_n8k_40368129.out").write_text(FIXTURE_MATMUL_ALIGN)
        (raw / "stdout.txt").write_text(FIXTURE_HP_BERT_NVTX)
        recs = aggregate_directory(str(tmp_path))
        assert len(recs) == 3
        ops = {r["op"] for r in recs}
        assert ops == {"bootstrap", "matmul", None}  # stdout.txt → op None

    def test_skips_malformed_logs_silently(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "bootstrap_mgpu_align_40371362.out").write_text(FIXTURE_BOOTSTRAP_MGPU)
        (raw / "junk_99999999.out").write_text(FIXTURE_MALFORMED)
        recs = aggregate_directory(str(tmp_path))
        assert len(recs) == 1
        assert recs[0]["op"] == "bootstrap"

    def test_missing_raw_dir_returns_empty(self, tmp_path):
        assert aggregate_directory(str(tmp_path)) == []

    def test_only_picks_out_and_stdout_files(self, tmp_path):
        # Reports / sqlite / nsys-rep siblings must be ignored.
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "bootstrap_mgpu_align_40371362.out").write_text(FIXTURE_BOOTSTRAP_MGPU)
        (raw / "bootstrap_mgpu_align_40371362.nsys-rep").write_text("binary garbage")
        (raw / "bootstrap_mgpu_align_40371362.sqlite").write_text("more garbage")
        (raw / "bootstrap_mgpu_align_40371362.cuda_gpu_sum.txt").write_text("csv data")
        recs = aggregate_directory(str(tmp_path))
        assert len(recs) == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    SCRIPT = Path(__file__).parent / "result_aggregator.py"

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(self.SCRIPT), *args],
            capture_output=True,
            text=True,
        )

    def test_cli_log_returns_0_and_json_record(self, tmp_path):
        log = tmp_path / "bootstrap_mgpu_align_40371362.out"
        log.write_text(FIXTURE_BOOTSTRAP_MGPU)
        r = self._run("--log", str(log))
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        assert payload["op"] == "bootstrap"
        assert payload["gpus"] == 4
        assert payload["per_call_ms"] == pytest.approx(263.67)

    def test_cli_log_returns_1_on_malformed(self, tmp_path):
        log = tmp_path / "junk_99999999.out"
        log.write_text(FIXTURE_MALFORMED)
        r = self._run("--log", str(log))
        assert r.returncode == 1
        # Error goes to stderr as JSON.
        err = json.loads(r.stderr)
        assert "error" in err

    def test_cli_dir_returns_json_array(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "bootstrap_mgpu_align_40371362.out").write_text(FIXTURE_BOOTSTRAP_MGPU)
        (raw / "matmul_align_n8k_40368129.out").write_text(FIXTURE_MATMUL_ALIGN)
        r = self._run("--dir", str(tmp_path))
        assert r.returncode == 0, r.stderr
        payload = json.loads(r.stdout)
        assert isinstance(payload, list)
        assert len(payload) == 2
        ops = {p["op"] for p in payload}
        assert ops == {"bootstrap", "matmul"}

    def test_cli_dir_empty_returns_1(self, tmp_path):
        r = self._run("--dir", str(tmp_path))
        assert r.returncode == 1
        payload = json.loads(r.stdout)
        assert payload == []

    def test_cli_requires_log_or_dir(self):
        r = self._run()
        # argparse exits 2 when mutually-exclusive required group is unset.
        assert r.returncode == 2
