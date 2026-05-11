#!/usr/bin/env python3
"""Saturation analyzer for the multiNEXUS Goal 2 unit run.

Given two ordered per-layer timing measurements (e.g. t_layer_1, t_layer_2)
from a 1-head x 2-layer HP-BERT unit run at uniform logN=15, decide whether
the pipeline has reached steady state (i.e. layer 2 has the same per-call
cost as layer 1) so that the extrapolation to full 12-layer BERT is honest.

PRD module sketch 1: pure function on timing inputs. Returns a dict
{saturated, relative_delta, threshold} that the paper's Section 7 cites
verbatim. Default threshold: 5% per docs/prd/PRD-multiNEXUS-paper.md.

Usage as a library:
    from saturation_check import check_saturation
    r = check_saturation(t1_ms=12345.6, t2_ms=12500.0)
    assert r["saturated"]

Usage as a CLI:
    python saturation_check.py --t1 12345.6 --t2 12500.0 --threshold 0.05
    # prints JSON; exits 0 if saturated, 1 if not, 2 on invalid input
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import TypedDict


class SaturationResult(TypedDict):
    saturated: bool
    relative_delta: float
    threshold: float
    t1_ms: float
    t2_ms: float


def check_saturation(t1_ms: float, t2_ms: float, threshold: float = 0.05) -> SaturationResult:
    """Decide whether two layer timings are within `threshold` relative delta.

    The relative delta is |t2 - t1| / t1. Layer 1 carries the per-call
    warmup that does not recur for layer 2 onward; if relative_delta is
    within threshold, the pipeline has saturated and t_per_layer can be
    taken as the mean of the two.
    """
    if t1_ms <= 0 or t2_ms <= 0:
        raise ValueError(f"timings must be positive; got t1={t1_ms}, t2={t2_ms}")
    if not (0 < threshold < 1):
        raise ValueError(f"threshold must be in (0, 1); got {threshold}")
    relative_delta = abs(t2_ms - t1_ms) / t1_ms
    return {
        "saturated": relative_delta <= threshold,
        "relative_delta": relative_delta,
        "threshold": threshold,
        "t1_ms": t1_ms,
        "t2_ms": t2_ms,
    }


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--t1", type=float, required=True, help="layer 1 timing in ms")
    p.add_argument("--t2", type=float, required=True, help="layer 2 timing in ms")
    p.add_argument("--threshold", type=float, default=0.05, help="relative delta tolerance (default 0.05)")
    args = p.parse_args()
    try:
        result = check_saturation(args.t1, args.t2, args.threshold)
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["saturated"] else 1


if __name__ == "__main__":
    sys.exit(_cli())
