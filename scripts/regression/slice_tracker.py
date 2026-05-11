#!/usr/bin/env python3
"""Vertical-slice dependency tracker for the multiNEXUS paper.

PRD module sketch 3: pure function on commit-message footers. Reads
`git log` output, extracts the `Slice: <id>; Depends-on: <deps>` footer
declared by each vertical-slice commit (per CLAUDE.md "Vertical-slice
work convention"), and emits:

  * a topological order over the declared slices (sources first), and
  * the list of "blocked" slices whose `Depends-on` references a slice
    that does not yet appear in the commit history.

Optionally renders the dependency graph as a GitHub-friendly Mermaid
`flowchart TD` so the user can paste it into a markdown file.

Usage as a library:
    from slice_tracker import parse_slice_declarations, toposort, render_dag
    decls = parse_slice_declarations(git_log_output)
    ordered, blocked = toposort(decls)
    mermaid = render_dag(decls)

Usage as a CLI:
    python slice_tracker.py                       # full branch
    python slice_tracker.py --since 8e04b14       # from a given hash
    python slice_tracker.py --since 8e04b14 --graph
    python slice_tracker.py --input <file>        # read pre-captured log
    cat log.txt | python slice_tracker.py --stdin

Exit codes:
    0   success, no blocked slices
    1   success, but one or more slices have missing upstream deps
    2   parse error or git invocation failed
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from typing import TypedDict


# Sentinel that we tell `git log` to print between commits. Chosen so
# real commit bodies are extremely unlikely to contain it verbatim.
COMMIT_DELIM = "--END-OF-COMMIT--"

GIT_LOG_FORMAT = f"%H%n%s%n%b%n{COMMIT_DELIM}"


class SliceDecl(TypedDict):
    commit_hash: str
    subject: str
    slice_id: str
    depends_on: list[str]


# Canonical slice ID shape: <PHASE>-<TAIL> where PHASE is an uppercase
# token (BUG, PROFILE, MEASURE, WRITE, DOC, APPENDIX, ...) and TAIL is
# something like "01", "S6", "S6.gelu", "Appendix". We accept dots and
# letters in the tail to support WRITE-S6.gelu and WRITE-Appendix.
SLICE_ID_RE = re.compile(r"^[A-Z]+-[A-Za-z0-9.]+$")

# Matches `Slice: <id>; Depends-on: <stuff up to EOL>` anywhere in the
# commit body. Tolerant of leading whitespace and trailing punctuation.
SLICE_FOOTER_RE = re.compile(
    r"^\s*Slice:\s*(?P<sid>[^;]+?)\s*;\s*Depends-on:\s*(?P<deps>.*?)\s*$",
    re.MULTILINE,
)

# Range expander, e.g. "BUG-01..04" -> [BUG-01, BUG-02, BUG-03, BUG-04],
# or "WRITE-S2..S9" -> [WRITE-S2, ..., WRITE-S9]. We only attempt
# expansion when the tail and the after-".." token both look like
# numeric or S-prefixed-numeric counters that share a common prefix.
RANGE_RE = re.compile(
    r"^(?P<phase>[A-Z]+)-(?P<aprefix>[A-Za-z]*)(?P<astart>\d+)\.\.(?P<bprefix>[A-Za-z]*)(?P<bend>\d+)$"
)


def _expand_range(token: str) -> list[str] | None:
    """If token is like BUG-01..04 or WRITE-S2..S9, expand it.

    Returns the list of expanded IDs, or None if the token does not
    look like a range. The width of the numeric portion is preserved
    from the start endpoint (so BUG-01..04 -> BUG-01..BUG-04, two-digit).
    """
    m = RANGE_RE.match(token)
    if not m:
        return None
    phase = m.group("phase")
    aprefix = m.group("aprefix")
    bprefix = m.group("bprefix")
    # End token may omit the letter prefix (BUG-01..04) or repeat it
    # (WRITE-S2..S9). If it includes a prefix, it must match the start.
    if bprefix and bprefix != aprefix:
        return None
    start = int(m.group("astart"))
    end = int(m.group("bend"))
    if end < start:
        return None
    width = len(m.group("astart"))
    return [f"{phase}-{aprefix}{str(n).zfill(width)}" for n in range(start, end + 1)]


def _normalize_dep_token(tok: str) -> list[str]:
    """Turn one raw dependency token into zero-or-more clean slice IDs.

    Strips parenthetical commentary and trailing free text. Returns a
    list because a single token may expand a range into multiple IDs.
    Returns [] for the literal "none" or for anything we cannot parse
    as a slice ID.
    """
    # Drop everything after the first " (" — that's parenthetical
    # commentary like "BUG-01 (audit)".
    paren_idx = tok.find("(")
    if paren_idx >= 0:
        tok = tok[:paren_idx]
    # Drop trailing periods and whitespace.
    tok = tok.strip().rstrip(".").strip()
    if not tok:
        return []
    if tok.lower() == "none":
        return []
    # If the token has trailing free text after the slice ID (e.g.
    # "PROFILE-01..04 for trace-grounded fields"), keep only the first
    # whitespace-delimited word.
    first_word = tok.split()[0].rstrip(",").rstrip(";")
    # Range form?
    expanded = _expand_range(first_word)
    if expanded is not None:
        return expanded
    if SLICE_ID_RE.match(first_word):
        return [first_word]
    return []


def _parse_deps(raw: str) -> list[str]:
    """Split a `Depends-on:` payload into a deduplicated list of IDs.

    Splits on both ',' and ';' (the footer convention is comma-separated
    but real commits mix in semicolons when there's parenthetical text).
    Preserves first-seen order.
    """
    # Strip a single trailing period and trailing whitespace.
    raw = raw.strip().rstrip(".").strip()
    if not raw:
        return []
    # Treat semicolons as list separators too, since real footers use
    # them to delimit grouped sub-clauses.
    tokens = re.split(r"[;,]", raw)
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        for sid in _normalize_dep_token(tok):
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def parse_slice_declarations(git_log_output: str) -> list[SliceDecl]:
    """Parse `git log` output (formatted with GIT_LOG_FORMAT) into decls.

    Each commit produces zero or one SliceDecl: zero if the commit body
    has no `Slice:` footer, one otherwise. The first `Slice:` line wins
    if a commit somehow contains more than one.

    Returns declarations in the order they appear in the input (which,
    for default `git log`, is reverse-chronological — newest first).
    """
    out: list[SliceDecl] = []
    # Split on our delimiter. Each chunk: hash\nsubject\nbody...
    for raw_chunk in git_log_output.split(COMMIT_DELIM):
        chunk = raw_chunk.strip("\n")
        if not chunk.strip():
            continue
        lines = chunk.split("\n")
        if len(lines) < 2:
            continue
        commit_hash = lines[0].strip()
        subject = lines[1].strip()
        body = "\n".join(lines[2:])
        # Hash must look hash-y; skip otherwise (defensive).
        if not re.match(r"^[0-9a-f]{7,40}$", commit_hash):
            continue
        m = SLICE_FOOTER_RE.search(body)
        if m is None:
            continue
        sid_raw = m.group("sid").strip().rstrip(".").strip()
        # Drop any parenthetical commentary attached to the slice ID
        # itself, just in case.
        paren_idx = sid_raw.find("(")
        if paren_idx >= 0:
            sid_raw = sid_raw[:paren_idx].strip()
        if not SLICE_ID_RE.match(sid_raw):
            # Unrecognised slice ID shape — skip rather than fail.
            continue
        deps = _parse_deps(m.group("deps"))
        # A slice never depends on itself; defensive filter.
        deps = [d for d in deps if d != sid_raw]
        out.append(
            {
                "commit_hash": commit_hash,
                "subject": subject,
                "slice_id": sid_raw,
                "depends_on": deps,
            }
        )
    return out


def toposort(declarations: list[SliceDecl]) -> tuple[list[str], list[str]]:
    """Topologically sort declared slices; report blocked ones.

    Returns (ordered_slice_ids, blocked_slice_ids).

    * ordered: sources-first Kahn's-algorithm order, restricted to the
      slices that are actually declared. Among slices with the same
      remaining-in-degree, ties are broken alphabetically so the output
      is deterministic across runs.
    * blocked: slices that name at least one upstream that is missing
      from the declaration set. Blocked slices are still included in
      the ordered list (we treat their missing deps as satisfied, so a
      late-but-blocked slice still gets a position) — listing them
      separately is the contract the CLI exposes to the user.

    If the declared subset has a cycle, the cycle members appear at
    the tail of the ordered list in alphabetical order; we do not raise,
    because a half-typed-in-progress slice graph is a normal state.
    """
    # Deduplicate by slice_id (last declaration wins for the deps list;
    # in practice one slice == one commit, so this is just defensive).
    by_id: dict[str, SliceDecl] = {}
    for d in declarations:
        by_id[d["slice_id"]] = d

    known: set[str] = set(by_id.keys())
    # Effective deps: only those that are themselves declared. Missing
    # deps surface in `blocked` but do not block toposort progress.
    effective_deps: dict[str, list[str]] = {}
    blocked: list[str] = []
    for sid, decl in by_id.items():
        present = [dep for dep in decl["depends_on"] if dep in known]
        missing = [dep for dep in decl["depends_on"] if dep not in known]
        effective_deps[sid] = present
        if missing:
            blocked.append(sid)

    # Kahn's algorithm with deterministic (alphabetical) tie-breaking.
    in_degree: dict[str, int] = {sid: len(deps) for sid, deps in effective_deps.items()}
    # Reverse adjacency: for each dep, which slices wait on it?
    rev: dict[str, list[str]] = {sid: [] for sid in by_id}
    for sid, deps in effective_deps.items():
        for dep in deps:
            rev[dep].append(sid)

    ready = sorted([sid for sid, deg in in_degree.items() if deg == 0])
    ordered: list[str] = []
    while ready:
        sid = ready.pop(0)
        ordered.append(sid)
        for child in rev[sid]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                # Insert in sorted position for determinism.
                # Linear insert is fine for our slice counts (<~100).
                ready.append(child)
                ready.sort()

    if len(ordered) < len(by_id):
        # Cycle: append remaining slices alphabetically so the output
        # is still total. They're still "ordered" relative to their
        # acyclic ancestors; only mutual cycle members are arbitrary.
        remaining = sorted(set(by_id) - set(ordered))
        ordered.extend(remaining)

    # Sort blocked alphabetically for stable display.
    blocked.sort()
    return ordered, blocked


def render_dag(declarations: list[SliceDecl]) -> str:
    """Render the declared slice graph as a Mermaid `flowchart TD`.

    The output is a fenced ```mermaid block so it can be pasted into a
    markdown file and rendered by GitHub. Edges flow upstream → slice
    (i.e. an arrow from BUG-01 to PROFILE-01 means PROFILE-01 depends
    on BUG-01). Missing upstream slices are emitted as nodes too so
    the graph stays connected; they are styled as `:::missing` so the
    reader sees what is not yet committed.
    """
    by_id: dict[str, SliceDecl] = {d["slice_id"]: d for d in declarations}
    known = set(by_id.keys())

    lines: list[str] = ["```mermaid", "flowchart TD"]
    # Declare known slice nodes first, alphabetically.
    for sid in sorted(known):
        lines.append(f"    {_mermaid_id(sid)}[{sid}]")
    # Collect missing upstream slices.
    missing: set[str] = set()
    for sid, decl in by_id.items():
        for dep in decl["depends_on"]:
            if dep not in known:
                missing.add(dep)
    for sid in sorted(missing):
        lines.append(f"    {_mermaid_id(sid)}[{sid}]:::missing")
    # Edges.
    for sid in sorted(known):
        decl = by_id[sid]
        for dep in decl["depends_on"]:
            lines.append(f"    {_mermaid_id(dep)} --> {_mermaid_id(sid)}")
    # Class definition for missing nodes.
    if missing:
        lines.append("    classDef missing stroke-dasharray: 5 5,stroke:#c00,color:#c00;")
    lines.append("```")
    return "\n".join(lines)


def _mermaid_id(slice_id: str) -> str:
    """Mermaid node IDs cannot contain '.' or '-'; replace with '_'."""
    return slice_id.replace("-", "_").replace(".", "_")


def _run_git_log(since: str | None) -> str:
    """Invoke `git log` with our delimited format. Returns stdout text."""
    rev_range = f"{since}..HEAD" if since else "HEAD"
    cmd = ["git", "log", f"--format={GIT_LOG_FORMAT}", rev_range]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"git log failed: {e.stderr.strip() or e}")
    except FileNotFoundError:
        raise SystemExit("git not on PATH; cannot read commit history")
    return proc.stdout


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--since",
        default=None,
        help="git revision to start from (exclusive); default: full HEAD history",
    )
    p.add_argument(
        "--graph",
        action="store_true",
        help="also print the Mermaid flowchart of the dependency graph",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--input",
        type=str,
        default=None,
        help="read a pre-captured git log output from this file instead of running git",
    )
    src.add_argument(
        "--stdin",
        action="store_true",
        help="read pre-captured git log output from stdin",
    )
    args = p.parse_args()

    if args.input is not None:
        try:
            with open(args.input, encoding="utf-8") as f:
                log_text = f.read()
        except OSError as e:
            print(f"error: cannot read --input file: {e}", file=sys.stderr)
            return 2
    elif args.stdin:
        log_text = sys.stdin.read()
    else:
        try:
            log_text = _run_git_log(args.since)
        except SystemExit as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    decls = parse_slice_declarations(log_text)
    if not decls:
        print("(no Slice: footers found in the given range)")
        return 0

    ordered, blocked = toposort(decls)
    print("# Topological order (sources first):")
    for sid in ordered:
        marker = "  [BLOCKED]" if sid in blocked else ""
        print(f"{sid}{marker}")
    print()
    if blocked:
        print(f"# Blocked slices ({len(blocked)}): upstream missing from history")
        by_id = {d["slice_id"]: d for d in decls}
        known = set(by_id.keys())
        for sid in blocked:
            missing = [dep for dep in by_id[sid]["depends_on"] if dep not in known]
            print(f"  - {sid}: missing {', '.join(missing)}")
        print()
    else:
        print("# No blocked slices.\n")

    if args.graph:
        print(render_dag(decls))
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(_cli())
