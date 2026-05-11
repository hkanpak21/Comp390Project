"""Tests for saturation_check.py.

Per PRD Testing Decisions section: pure function on timing inputs, easy to
test with synthetic timing inputs. Tests cover the saturation predicate
(within / outside threshold), edge cases, and the CLI exit-code contract.

Run from repo root:
    python -m pytest scripts/regression/test_saturation_check.py -v
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from saturation_check import check_saturation


class TestCheckSaturation:
    def test_exact_match(self):
        r = check_saturation(t1_ms=1000.0, t2_ms=1000.0)
        assert r["saturated"] is True
        assert r["relative_delta"] == 0.0

    def test_within_default_threshold_5pct(self):
        # 4.9% drift; within 5% threshold
        r = check_saturation(t1_ms=1000.0, t2_ms=1049.0)
        assert r["saturated"] is True
        assert abs(r["relative_delta"] - 0.049) < 1e-9

    def test_at_default_threshold_boundary(self):
        # Exactly at 5%; predicate is <=, so saturated
        r = check_saturation(t1_ms=1000.0, t2_ms=1050.0)
        assert r["saturated"] is True
        assert abs(r["relative_delta"] - 0.05) < 1e-9

    def test_above_default_threshold(self):
        # 6% drift; outside 5% threshold
        r = check_saturation(t1_ms=1000.0, t2_ms=1060.0)
        assert r["saturated"] is False
        assert abs(r["relative_delta"] - 0.06) < 1e-9

    def test_layer2_faster_than_layer1_still_uses_abs(self):
        # Layer 2 measured slightly faster (e.g. warmup amortization)
        r = check_saturation(t1_ms=1000.0, t2_ms=970.0)
        assert r["saturated"] is True
        assert abs(r["relative_delta"] - 0.030) < 1e-9

    def test_custom_threshold_tightens(self):
        # 4% drift; outside 3% tightened threshold
        r = check_saturation(t1_ms=1000.0, t2_ms=1040.0, threshold=0.03)
        assert r["saturated"] is False

    def test_custom_threshold_loosens(self):
        # 8% drift; inside 10% loosened threshold
        r = check_saturation(t1_ms=1000.0, t2_ms=1080.0, threshold=0.10)
        assert r["saturated"] is True

    def test_reports_inputs_back(self):
        r = check_saturation(t1_ms=42.5, t2_ms=43.0, threshold=0.05)
        assert r["t1_ms"] == 42.5
        assert r["t2_ms"] == 43.0
        assert r["threshold"] == 0.05

    @pytest.mark.parametrize("t1, t2", [(0.0, 100.0), (-1.0, 100.0), (100.0, 0.0), (100.0, -1.0)])
    def test_rejects_nonpositive_timings(self, t1, t2):
        with pytest.raises(ValueError, match="timings must be positive"):
            check_saturation(t1_ms=t1, t2_ms=t2)

    @pytest.mark.parametrize("th", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_invalid_threshold(self, th):
        with pytest.raises(ValueError, match="threshold must be in"):
            check_saturation(t1_ms=1000.0, t2_ms=1000.0, threshold=th)


class TestCLI:
    SCRIPT = Path(__file__).parent / "saturation_check.py"

    def _run(self, *args):
        return subprocess.run(
            [sys.executable, str(self.SCRIPT), *args],
            capture_output=True,
            text=True,
        )

    def test_cli_saturated_returns_0(self):
        r = self._run("--t1", "1000", "--t2", "1040")
        assert r.returncode == 0
        payload = json.loads(r.stdout)
        assert payload["saturated"] is True

    def test_cli_unsaturated_returns_1(self):
        r = self._run("--t1", "1000", "--t2", "1100")
        assert r.returncode == 1
        payload = json.loads(r.stdout)
        assert payload["saturated"] is False

    def test_cli_invalid_input_returns_2(self):
        r = self._run("--t1", "-1", "--t2", "1000")
        assert r.returncode == 2

    def test_cli_custom_threshold(self):
        r = self._run("--t1", "1000", "--t2", "1080", "--threshold", "0.10")
        assert r.returncode == 0
