"""Tests for slice_tracker.py.

PRD module sketch 3 testing decision: small synthetic slice graphs.
Covers footer parsing (well-formed, tolerant of real-world quirks),
toposort over linear / branching / blocked DAGs, the Mermaid renderer,
and the CLI contract via the --input flag.

Run from repo root:
    python -m pytest scripts/regression/test_slice_tracker.py -v
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from slice_tracker import (  # noqa: E402
    COMMIT_DELIM,
    parse_slice_declarations,
    render_dag,
    toposort,
)


def _make_log(*commits: tuple[str, str, str]) -> str:
    """Build a synthetic git log output in the same format slice_tracker
    expects: hash\\nsubject\\nbody\\n<DELIM>\\n per commit."""
    chunks = []
    for h, subject, body in commits:
        chunks.append(f"{h}\n{subject}\n{body}\n{COMMIT_DELIM}\n")
    return "".join(chunks)


# ---------------------------------------------------------------------------
# parse_slice_declarations
# ---------------------------------------------------------------------------


class TestParseSliceDeclarations:
    def test_single_well_formed_footer(self):
        log = _make_log(
            (
                "deadbeef",
                "BUG-01(benchmarks): audit",
                "Body text.\n\nSlice: BUG-01; Depends-on: none\n",
            )
        )
        decls = parse_slice_declarations(log)
        assert len(decls) == 1
        d = decls[0]
        assert d["commit_hash"] == "deadbeef"
        assert d["subject"] == "BUG-01(benchmarks): audit"
        assert d["slice_id"] == "BUG-01"
        assert d["depends_on"] == []

    def test_comma_separated_deps(self):
        log = _make_log(
            (
                "cafef00d",
                "WRITE-S8(paper): discussion",
                "...\n\nSlice: WRITE-S8; Depends-on: WRITE-S6, WRITE-S7\n",
            )
        )
        decls = parse_slice_declarations(log)
        assert decls[0]["depends_on"] == ["WRITE-S6", "WRITE-S7"]

    def test_skips_commits_without_footer(self):
        log = _make_log(
            ("aaaa111", "chore: nothing", "Just a body, no slice.\n"),
            ("bbbb222", "BUG-02(x): audit", "...\nSlice: BUG-02; Depends-on: none\n"),
            ("cccc333", "merge", "Merge branch 'feature'\n"),
        )
        decls = parse_slice_declarations(log)
        assert [d["slice_id"] for d in decls] == ["BUG-02"]

    def test_parenthetical_commentary_in_deps_is_stripped(self):
        # Real-world example from the multinexus branch.
        log = _make_log(
            (
                "1234567",
                "WRITE-S6(paper): goal 1",
                "body\n\nSlice: WRITE-S6; Depends-on: BUG-01 (audit); PROFILE-01..04 for trace-grounded fields\n",
            )
        )
        decls = parse_slice_declarations(log)
        deps = decls[0]["depends_on"]
        assert "BUG-01" in deps
        # Range expansion: PROFILE-01..04 -> four IDs.
        assert "PROFILE-01" in deps
        assert "PROFILE-02" in deps
        assert "PROFILE-03" in deps
        assert "PROFILE-04" in deps

    def test_range_expansion_with_letter_prefix(self):
        # WRITE-S2..S9 should expand to S2..S9.
        log = _make_log(
            (
                "abcdef0",
                "WRITE-S1(paper): abstract",
                "body\n\nSlice: WRITE-S1; Depends-on: WRITE-S2..S9 (compressed from the rest)\n",
            )
        )
        decls = parse_slice_declarations(log)
        deps = decls[0]["depends_on"]
        assert deps == ["WRITE-S2", "WRITE-S3", "WRITE-S4", "WRITE-S5", "WRITE-S6", "WRITE-S7", "WRITE-S8", "WRITE-S9"]

    def test_none_literal_means_no_deps(self):
        log = _make_log(
            ("0011223", "BUG-04(x): audit", "body\n\nSlice: BUG-04; Depends-on: none\n"),
        )
        assert parse_slice_declarations(log)[0]["depends_on"] == []

    def test_dotted_slice_id_supported(self):
        # WRITE-S6.gelu is canonical per PRD slice map.
        log = _make_log(
            (
                "f00f00f",
                "WRITE-S6.gelu(paper): gelu subsection",
                "...\nSlice: WRITE-S6.gelu; Depends-on: PROFILE-02, BUG-01\n",
            )
        )
        d = parse_slice_declarations(log)[0]
        assert d["slice_id"] == "WRITE-S6.gelu"
        assert d["depends_on"] == ["PROFILE-02", "BUG-01"]

    def test_invalid_slice_id_is_skipped(self):
        log = _make_log(
            ("9999999", "bad", "Slice: not_a_real_id; Depends-on: none\n"),
        )
        assert parse_slice_declarations(log) == []

    def test_self_dependency_is_filtered_out(self):
        log = _make_log(
            ("8888888", "WRITE-S5: bad", "Slice: WRITE-S5; Depends-on: WRITE-S5, BUG-01\n"),
        )
        d = parse_slice_declarations(log)[0]
        assert d["depends_on"] == ["BUG-01"]


# ---------------------------------------------------------------------------
# toposort
# ---------------------------------------------------------------------------


def _decl(sid: str, deps: list[str]) -> dict:
    return {"commit_hash": "x" * 7, "subject": f"{sid}: stub", "slice_id": sid, "depends_on": deps}


class TestToposort:
    def test_linear_chain(self):
        decls = [
            _decl("BUG-01", []),
            _decl("PROFILE-01", ["BUG-01"]),
            _decl("WRITE-S6", ["PROFILE-01"]),
        ]
        ordered, blocked = toposort(decls)
        assert ordered == ["BUG-01", "PROFILE-01", "WRITE-S6"]
        assert blocked == []

    def test_branching_dag(self):
        # BUG-01 feeds PROFILE-01 and PROFILE-02; both feed WRITE-S6.
        decls = [
            _decl("BUG-01", []),
            _decl("PROFILE-01", ["BUG-01"]),
            _decl("PROFILE-02", ["BUG-01"]),
            _decl("WRITE-S6", ["PROFILE-01", "PROFILE-02"]),
        ]
        ordered, blocked = toposort(decls)
        assert blocked == []
        # BUG-01 must come first; WRITE-S6 must come last.
        assert ordered[0] == "BUG-01"
        assert ordered[-1] == "WRITE-S6"
        # Ties broken alphabetically.
        assert ordered.index("PROFILE-01") < ordered.index("PROFILE-02")

    def test_missing_upstream_marks_slice_blocked(self):
        decls = [
            _decl("WRITE-S7", ["MEASURE-01", "MEASURE-02"]),
            _decl("MEASURE-01", []),
            # MEASURE-02 is not declared anywhere -> WRITE-S7 is blocked.
        ]
        ordered, blocked = toposort(decls)
        assert blocked == ["WRITE-S7"]
        # Blocked slices are still ordered (missing deps treated as
        # satisfied so progress reporting is meaningful).
        assert "WRITE-S7" in ordered
        assert ordered.index("MEASURE-01") < ordered.index("WRITE-S7")

    def test_deterministic_alphabetical_tie_break(self):
        # Three independent sources; output must be alphabetical.
        decls = [_decl("BUG-03", []), _decl("BUG-01", []), _decl("BUG-02", [])]
        ordered, _ = toposort(decls)
        assert ordered == ["BUG-01", "BUG-02", "BUG-03"]

    def test_empty_input(self):
        ordered, blocked = toposort([])
        assert ordered == []
        assert blocked == []

    def test_cycle_does_not_raise(self):
        # Pathological: A depends on B, B depends on A. We don't raise;
        # cycle members appear at the tail.
        decls = [_decl("DOC-01", ["DOC-02"]), _decl("DOC-02", ["DOC-01"])]
        ordered, blocked = toposort(decls)
        assert set(ordered) == {"DOC-01", "DOC-02"}
        assert blocked == []


# ---------------------------------------------------------------------------
# render_dag
# ---------------------------------------------------------------------------


class TestRenderDag:
    def test_emits_mermaid_fences(self):
        out = render_dag([_decl("BUG-01", [])])
        assert out.startswith("```mermaid\n")
        assert out.endswith("\n```")
        assert "flowchart TD" in out

    def test_node_and_edge_for_simple_chain(self):
        decls = [_decl("BUG-01", []), _decl("PROFILE-01", ["BUG-01"])]
        out = render_dag(decls)
        # Both nodes declared.
        assert "BUG_01[BUG-01]" in out
        assert "PROFILE_01[PROFILE-01]" in out
        # Edge direction: upstream -> downstream.
        assert "BUG_01 --> PROFILE_01" in out

    def test_dotted_slice_id_becomes_underscored_node(self):
        out = render_dag([_decl("WRITE-S6.gelu", ["BUG-01"])])
        assert "WRITE_S6_gelu[WRITE-S6.gelu]" in out

    def test_missing_upstream_is_styled(self):
        decls = [_decl("WRITE-S7", ["MEASURE-01"])]  # MEASURE-01 absent
        out = render_dag(decls)
        assert "MEASURE_01[MEASURE-01]:::missing" in out
        assert "classDef missing" in out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    SCRIPT = Path(__file__).parent / "slice_tracker.py"

    def _run(self, log_text: str, *extra_args: str) -> subprocess.CompletedProcess:
        # Use --stdin so we don't need to touch a file.
        return subprocess.run(
            [sys.executable, str(self.SCRIPT), "--stdin", *extra_args],
            input=log_text,
            capture_output=True,
            text=True,
        )

    def test_cli_prints_topo_order(self):
        log = _make_log(
            ("aaa1111", "BUG-01(x): audit", "Slice: BUG-01; Depends-on: none\n"),
            ("bbb2222", "PROFILE-01(x): nsys", "Slice: PROFILE-01; Depends-on: BUG-01\n"),
        )
        r = self._run(log)
        assert r.returncode == 0, r.stderr
        # BUG-01 must appear before PROFILE-01.
        assert r.stdout.index("BUG-01") < r.stdout.index("PROFILE-01")
        assert "No blocked slices" in r.stdout

    def test_cli_returns_1_when_blocked(self):
        log = _make_log(
            ("ccc3333", "WRITE-S7: e2e", "Slice: WRITE-S7; Depends-on: MEASURE-01\n"),
        )
        r = self._run(log)
        assert r.returncode == 1
        assert "[BLOCKED]" in r.stdout
        assert "WRITE-S7" in r.stdout
        assert "MEASURE-01" in r.stdout

    def test_cli_empty_history_succeeds(self):
        r = self._run("")
        assert r.returncode == 0
        assert "no Slice: footers" in r.stdout

    def test_cli_graph_flag_emits_mermaid(self):
        log = _make_log(
            ("ddd4444", "BUG-01(x): audit", "Slice: BUG-01; Depends-on: none\n"),
        )
        r = self._run(log, "--graph")
        assert r.returncode == 0
        assert "```mermaid" in r.stdout
        assert "flowchart TD" in r.stdout

    def test_cli_input_file(self, tmp_path):
        log = _make_log(
            ("eee5555", "BUG-01: audit", "Slice: BUG-01; Depends-on: none\n"),
        )
        f = tmp_path / "log.txt"
        f.write_text(log)
        r = subprocess.run(
            [sys.executable, str(self.SCRIPT), "--input", str(f)],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0
        assert "BUG-01" in r.stdout


# ---------------------------------------------------------------------------
# Integration: synthetic but realistic 5-commit log
# ---------------------------------------------------------------------------


class TestSyntheticMultiNexusLog:
    """End-to-end test on a 5-commit log that mirrors the real multinexus
    branch shape (BUG -> PROFILE/MEASURE -> WRITE chain)."""

    @pytest.fixture
    def log(self) -> str:
        return _make_log(
            ("1111111", "BUG-01(b): audit", "...\nSlice: BUG-01; Depends-on: none\n"),
            ("2222222", "BUG-02(b): audit", "...\nSlice: BUG-02; Depends-on: none\n"),
            ("3333333", "PROFILE-01(s): nsys matmul", "...\nSlice: PROFILE-01; Depends-on: BUG-01 (matmul)\n"),
            ("4444444", "MEASURE-01(s): unit run", "...\nSlice: MEASURE-01; Depends-on: BUG-02\n"),
            ("5555555", "WRITE-S6(p): goal 1", "...\nSlice: WRITE-S6; Depends-on: BUG-01, PROFILE-01\n"),
            ("6666666", "WRITE-S7(p): goal 2", "...\nSlice: WRITE-S7; Depends-on: MEASURE-01\n"),
        )

    def test_parses_six_decls(self, log: str):
        assert len(parse_slice_declarations(log)) == 6

    def test_topo_order_is_consistent(self, log: str):
        decls = parse_slice_declarations(log)
        ordered, blocked = toposort(decls)
        # Sources before downstream.
        assert ordered.index("BUG-01") < ordered.index("PROFILE-01")
        assert ordered.index("PROFILE-01") < ordered.index("WRITE-S6")
        assert ordered.index("BUG-02") < ordered.index("MEASURE-01")
        assert ordered.index("MEASURE-01") < ordered.index("WRITE-S7")
        assert blocked == []
