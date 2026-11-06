"""Tests for the research reorganization tool (manifest, ledger and reference rewriter)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.research_reorg import (
    DEFAULT_ALLOW_NEW,
    DEFAULT_REWRITE_MANIFEST,
    PINNED_PATHS,
    ManifestRow,
    MoveRefusedError,
    Unresolved,
    apply_moves,
    build_ledger,
    check_manifest,
    load_manifest,
    load_rewrite_manifest,
    main,
    moved_csv_text,
    remove_leftover_dirs,
    rewrite_code,
    rewrite_markdown,
    sha256_file,
    verify_final,
    verify_moved,
    verify_rewritten,
)

OLD = "docs/research/"


def _row(current: str, proposed: str, kind: str = "report") -> ManifestRow:
    return ManifestRow.build(current, proposed, kind, "")


def _clean() -> tuple[list[ManifestRow], set[str]]:
    rows = [
        _row(f"{OLD}a.md", "research/allocation/s1/a.md"),
        _row(f"{OLD}b.md", "research/allocation/s1/b.md"),
        _row(f"{OLD}README.md", "research/README.md", "living"),
    ]
    return rows, {r.current_path for r in rows}


def test_check_manifest_passes_clean_fixture() -> None:
    rows, tracked = _clean()
    assert check_manifest(rows, tracked) == []


def test_check_manifest_flags_unmapped_tracked_file() -> None:
    rows, tracked = _clean()
    tracked.add(f"{OLD}new.md")
    problems = check_manifest(rows, tracked)
    assert any("new.md" in p and "not in manifest" in p for p in problems)


def test_check_manifest_flags_duplicate_destination() -> None:
    rows, tracked = _clean()
    rows[1] = _row(f"{OLD}b.md", "research/allocation/s1/a.md")
    problems = check_manifest(rows, tracked)
    assert any("duplicate proposed_path" in p for p in problems)


def test_check_manifest_flags_depth_over_two() -> None:
    rows = [_row(f"{OLD}file.md", "research/a/b/c/file.md")]
    problems = check_manifest(rows, {f"{OLD}file.md"})
    assert any("depth" in p for p in problems)


def test_check_manifest_respects_allow_deeper() -> None:
    deep = "research/audits/root-cause-2026-11-04/lenses/x.md"
    rows = [_row(f"{OLD}x.md", deep)]
    tracked = {f"{OLD}x.md"}
    assert any("depth" in p for p in check_manifest(rows, tracked, allow_deeper=()))
    assert check_manifest(rows, tracked) == []  # the default allows that one prefix
    assert check_manifest(rows, tracked, allow_deeper=("audits/root-cause-2026-11-04",)) == []
    other = [_row(f"{OLD}x.md", "research/audits/other-study/lenses/x.md")]
    assert any("depth" in p for p in check_manifest(other, tracked))
    too_deep = [_row(f"{OLD}x.md", "research/audits/root-cause-2026-11-04/lenses/a/x.md")]
    assert any("depth" in p for p in check_manifest(too_deep, tracked))


def test_check_manifest_flags_unlisted_basename_change() -> None:
    rows = [_row(f"{OLD}x.md", "research/allocation/s1/y.md")]
    problems = check_manifest(rows, {f"{OLD}x.md"})
    assert any("basename" in p for p in problems)


def test_check_manifest_allows_the_three_brief_renames() -> None:
    names = [
        "2026-10-23-century-leverage",
        "2026-09-28-cross-asset-sleeve",
        "2026-09-30-static-fractional-leverage",
    ]
    rows = [
        _row(f"{OLD}briefs/{n}.md", f"research/allocation/s/{n}-brief.md", "brief") for n in names
    ]
    assert check_manifest(rows, {r.current_path for r in rows}) == []


def test_check_manifest_rejects_brief_rename_outside_briefs() -> None:
    n = "2026-10-23-century-leverage"
    rows = [_row(f"{OLD}{n}.md", f"research/allocation/s/{n}-brief.md")]
    assert any("basename" in p for p in check_manifest(rows, {rows[0].current_path}))


def test_check_manifest_flags_bad_prefix_and_outside_source() -> None:
    rows = [_row("docs/other/a.md", "elsewhere/a.md")]
    problems = check_manifest(rows, set())
    assert any("outside docs/research" in p for p in problems)
    assert any("research/" in p and "start" in p for p in problems)


def test_check_manifest_flags_duplicate_source() -> None:
    rows = [
        _row(f"{OLD}a.md", "research/allocation/s1/a.md"),
        _row(f"{OLD}a.md", "research/allocation/s2/a.md"),
    ]
    problems = check_manifest(rows, {f"{OLD}a.md"})
    assert any("duplicate current_path" in p for p in problems)


def test_frozen_flag_rules() -> None:
    for name in ("r.json", "r.csv", "r.py", "r.tgz", "r.txt"):
        assert _row(f"{OLD}{name}", f"research/audits/s/{name}", "result").frozen
    pkg = _row(f"{OLD}artifacts-2026-09-23-web-batch/n.md", "research/allocation/artifacts-x/n.md")
    assert pkg.frozen
    pinned = _row(f"{OLD}briefs/2026-10-21-event-timing-audit.md", "research/audits/e/x.md")
    assert pinned.pinned and pinned.frozen
    plain = _row(f"{OLD}plain.md", "research/allocation/s/plain.md")
    assert not plain.frozen and not plain.pinned


def test_pinned_paths_are_twelve_prefixed() -> None:
    assert len(PINNED_PATHS) == 12
    assert len(set(PINNED_PATHS)) == 12
    assert all(p.startswith(OLD) for p in PINNED_PATHS)


def test_load_manifest_reads_csv_and_computes_flags(tmp_path: Path) -> None:
    csv_path = tmp_path / "m.csv"
    csv_path.write_text(
        "current_path,proposed_path,folder,kind,notes\n"
        f'"{OLD}_x.json","research/audits/s/_x.json","research/audits/s","result","n"\n'
        f'"{OLD}p.md","research/audits/s/p.md","research/audits/s","report",""\n'
    )
    rows = load_manifest(csv_path)
    assert [r.frozen for r in rows] == [True, False]
    assert rows[0].notes == "n"


def test_main_manifest_check_reports_problem(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    csv_path = tmp_path / "m.csv"
    csv_path.write_text(
        "current_path,proposed_path,folder,kind,notes\n"
        '"docs/research/zzz-not-tracked.md","research/a/zzz-not-tracked.md","research/a","report",""\n'
    )
    assert main(["manifest-check", "--manifest", str(csv_path)]) == 1
    assert "PROBLEM" in capsys.readouterr().out


def _write(root: Path, rel: str, text: str | bytes) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        target.write_bytes(text)
    else:
        target.write_text(text, encoding="utf-8")


def _ledger_tree(tmp_path: Path) -> tuple[list[ManifestRow], Path]:
    """Build a small fake repo: research files plus outside files that mention the tree."""
    pinned = PINNED_PATHS[0]
    _write(tmp_path, pinned, "pinned brief\n")
    _write(tmp_path, f"{OLD}plain.md", "# plain\nsee [ok](other.md) and [gone](missing.md#frag)\n")
    _write(tmp_path, f"{OLD}other.md", "# other\n[web](https://x.test/a) [top](#anchor)\n")
    _write(tmp_path, f"{OLD}artifacts-pkg/data.json", '{"a": 1}\n')
    _write(tmp_path, f"{OLD}notes.md", "```\n[fenced](nope.md)\n```\n[mail](mailto:a@b.c)\n")
    _write(tmp_path, "scripts/consumer.py", "A = 'docs/research/x'\nB = 'docs/research/y'\n")
    _write(tmp_path, "docs/guide.md", "see docs/research/plain.md\n")
    _write(tmp_path, "AGENTS.md", "docs/research/RESEARCH_SNAPSHOT.md\n")
    _write(tmp_path, "src/binary.dat", b"\xff\xfe docs/research \x00")
    _write(tmp_path, "scripts/unrelated.py", "nothing here\n")
    _write(tmp_path, "docs/superpowers/plans/2026-11-05-research-reorg-plan.md", "docs/research\n")
    _write(tmp_path, "scripts/research_reorg.py", "docs/research\n")
    _write(tmp_path, "tests/scripts/test_research_reorg.py", "docs/research\n")
    rows = [
        _row(pinned, "research/audits/e/brief.md", "brief"),
        _row(f"{OLD}plain.md", "research/allocation/s/plain.md"),
        _row(f"{OLD}other.md", "research/allocation/s/other.md"),
        _row(f"{OLD}artifacts-pkg/data.json", "research/allocation/artifacts-pkg/data.json"),
        _row(f"{OLD}notes.md", "research/allocation/s/notes.md"),
    ]
    return rows, tmp_path


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    payload = b"abc" * 100_000
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_ledger_has_entry_per_manifest_row_with_digest(tmp_path: Path) -> None:
    rows, root = _ledger_tree(tmp_path)
    ledger = build_ledger(rows, root, "deadbeef")
    assert ledger["source_revision"] == "deadbeef"
    assert [f["current_path"] for f in ledger["files"]] == [r.current_path for r in rows]
    entry = ledger["files"][1]
    assert entry["proposed_path"] == "research/allocation/s/plain.md"
    assert entry["sha256"] == sha256_file(root / entry["current_path"])
    assert entry["size"] == (root / entry["current_path"]).stat().st_size
    assert (entry["frozen"], entry["pinned"]) == (False, False)
    assert ledger["files"][3]["frozen"] is True
    json.dumps(ledger)  # must be serializable as-is


def test_ledger_records_pinned_flag_for_twelve_paths(tmp_path: Path) -> None:
    rows = []
    for rel in PINNED_PATHS:
        _write(tmp_path, rel, rel)
        rows.append(_row(rel, f"research/audits/e/{Path(rel).name}"))
    _write(tmp_path, f"{OLD}plain.md", "x")
    rows.append(_row(f"{OLD}plain.md", "research/allocation/s/plain.md"))
    ledger = build_ledger(rows, tmp_path, "abc")
    pinned = [f["current_path"] for f in ledger["files"] if f["pinned"]]
    assert sorted(pinned) == sorted(PINNED_PATHS)
    assert len(pinned) == 12
    assert all(f["frozen"] for f in ledger["files"] if f["pinned"])


def test_ledger_lists_preexisting_dead_relative_links(tmp_path: Path) -> None:
    rows, root = _ledger_tree(tmp_path)
    dead = build_ledger(rows, root, "abc")["existing_dead_links"]
    assert dead == [{"file": f"{OLD}plain.md", "line": 2, "link": "missing.md#frag"}]


def test_ledger_counts_inbound_references_outside_research(tmp_path: Path) -> None:
    rows, root = _ledger_tree(tmp_path)
    inbound = build_ledger(rows, root, "abc")["inbound_references"]
    assert inbound == {"AGENTS.md": 1, "docs/guide.md": 1, "scripts/consumer.py": 2}


def test_ledger_git_head_via_cli_is_deterministic(tmp_path: Path) -> None:
    rows, root = _ledger_tree(tmp_path)
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "current_path,proposed_path,folder,kind,notes\n"
        + "".join(f'"{r.current_path}","{r.proposed_path}","","{r.kind}",""\n' for r in rows)
    )
    out = tmp_path / "ledger.json"
    argv = ["ledger", "--manifest", str(manifest), "--out", str(out), "--root", str(root)]
    assert main(argv) == 0
    first = out.read_text()
    assert main(argv) == 0
    assert out.read_text() == first
    assert first.endswith("}\n") and '\n  "files"' in first
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert json.loads(first)["source_revision"] == head


def test_inbound_scan_includes_root_readme(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "a docs/research b docs/research c docs/research\n")
    rows, root = _ledger_tree(tmp_path)
    inbound = build_ledger(rows, root, "abc")["inbound_references"]
    assert inbound["README.md"] == 3


def test_inbound_scan_skips_workspace_directories(tmp_path: Path) -> None:
    _write(tmp_path, ".claude/skills/foo-workspace/run/out.md", "docs/research\n")
    _write(tmp_path, ".claude/skills/foo/SKILL.md", "docs/research\n")
    rows, root = _ledger_tree(tmp_path)
    inbound = build_ledger(rows, root, "abc")["inbound_references"]
    assert ".claude/skills/foo/SKILL.md" in inbound
    assert not any("-workspace" in key for key in inbound)


# --- Reference rewriter (Task 4) -------------------------------------------------------------

A_OLD = f"{OLD}2026-01-01-a.md"
A_NEW = "research/allocation/a-study/2026-01-01-a.md"
B_OLD = f"{OLD}briefs/2026-01-01-b.md"
B_NEW = "research/audits/b-study/2026-01-01-b.md"
MAPPING = {
    A_OLD: A_NEW,
    f"{OLD}_a_results.json": "research/allocation/a-study/_a_results.json",
    B_OLD: B_NEW,
    f"{OLD}RESEARCH_SNAPSHOT.md": "research/RESEARCH_SNAPSHOT.md",
}
FIXED = "docs/next_steps.md"


def _reasons(unresolved: list[Unresolved]) -> list[str]:
    return [u.reason for u in unresolved]


def test_backticked_path_rewritten_and_line_suffix_kept() -> None:
    text = (
        "See `docs/research/2026-01-01-a.md:12-20` and `docs/research/_a_results.json:5`\r\n"
        "and `docs/research/RESEARCH_SNAPSHOT.md` (snapshot).\n"
    )
    out, unresolved = rewrite_markdown(text, FIXED, FIXED, MAPPING)
    assert out == (
        f"See `{A_NEW}:12-20` and `research/allocation/a-study/_a_results.json:5`\r\n"
        "and `research/RESEARCH_SNAPSHOT.md` (snapshot).\n"
    )
    assert unresolved == []


def test_relative_link_outside_tree_recomputed_from_new_parent() -> None:
    text = "[r](../roadmap.md) and [s](../superpowers/specs/x.md#sec) and [t](../../README.md)\n"
    out, unresolved = rewrite_markdown(text, A_OLD, A_NEW, MAPPING)
    assert out == (
        "[r](../../../docs/roadmap.md) and [s](../../../docs/superpowers/specs/x.md#sec)"
        " and [t](../../../README.md)\n"
    )
    assert unresolved == []
    # A file that stays put but links into the moved tree.
    out, unresolved = rewrite_markdown(
        "[a](research/2026-01-01-a.md) [b](research/briefs/2026-01-01-b.md)\n",
        FIXED,
        FIXED,
        MAPPING,
    )
    assert out == f"[a](../{A_NEW}) [b](../{B_NEW})\n"
    assert unresolved == []
    out, _ = rewrite_markdown(
        "[s](docs/research/RESEARCH_SNAPSHOT.md)\n", "README.md", "README.md", MAPPING
    )
    assert out == "[s](research/RESEARCH_SNAPSHOT.md)\n"


def test_relative_link_inside_tree_follows_mapping() -> None:
    text = "[b](briefs/2026-01-01-b.md) [j](_a_results.json) [s](RESEARCH_SNAPSHOT.md#roster)\n"
    out, unresolved = rewrite_markdown(text, A_OLD, A_NEW, MAPPING)
    assert out == (
        "[b](../../audits/b-study/2026-01-01-b.md) [j](_a_results.json)"
        " [s](../../RESEARCH_SNAPSHOT.md#roster)\n"
    )
    assert unresolved == []
    # From inside briefs/ back up to a sibling report.
    out, _ = rewrite_markdown("[a](../2026-01-01-a.md)\n", B_OLD, B_NEW, MAPPING)
    assert out == "[a](../../allocation/a-study/2026-01-01-a.md)\n"


def test_directory_and_glob_citations_are_unresolved() -> None:
    text = (
        "Briefs live in `docs/research/briefs/` and lenses in `docs/research/_lens*.json`.\n"
        "Plain prose docs/research/2026-01-01-a.md and `docs/research/_gone.json`.\n"
        "Template `docs/research/<date>-<slug>.md`, tree docs/research and [d](research/briefs/).\n"
    )
    out, unresolved = rewrite_markdown(text, FIXED, FIXED, MAPPING)
    assert out == text
    got = sorted((u.line, u.reason) for u in unresolved)
    assert got == [
        (1, "directory citation"),
        (1, "glob citation"),
        (2, "not in manifest"),
        (2, "prose mention"),
        (3, "directory citation"),
        (3, "directory citation"),
        (3, "glob citation"),
    ]
    assert all(u.file == "" for u in unresolved)
    code = 'A = "docs/research/briefs/"\nB = glob("docs/research/_lens*.json")\n'
    out, unresolved = rewrite_code(code, MAPPING)
    assert out == code
    assert _reasons(unresolved) == ["directory citation", "glob citation"]


def test_bare_filename_and_url_untouched() -> None:
    text = (
        "Read 2026-01-01-a.md or RESEARCH_SNAPSHOT.md first.\n"
        "[gh](https://github.com/o/r/blob/main/docs/research/2026-01-01-a.md) "
        "[m](mailto:a@b.c) [top](#anchor)\n"
    )
    out, unresolved = rewrite_markdown(text, A_OLD, A_NEW, MAPPING)
    assert out == text
    # The URL still mentions the old tree: reported, never rewritten.
    assert [(u.line, u.reason) for u in unresolved] == [(2, "prose mention")]


def test_split_path_construction_is_flagged() -> None:
    code = (
        'OUT = REPO_ROOT / "docs" / "research" / "_a_results.json"\n'
        'P = Path("docs") / "research"\n'
        'J = os.path.join("docs", "research", "x.json")\n'
        'F = f"docs/research/_audit_{day}.json"\n'
        'G = ROOT / "docs/research" / name\n'
        'H = Path(__file__).parents[2] / "research"\n'
        'K = REPO_ROOT / "docs/research/_a_results.json"\n'
    )
    out, unresolved = rewrite_code(code, MAPPING)
    lines = out.split("\n")
    assert lines[:6] == code.split("\n")[:6]
    assert lines[6] == 'K = REPO_ROOT / "research/allocation/a-study/_a_results.json"'
    got = sorted((u.line, u.reason) for u in unresolved)
    assert got == [
        (1, "split path construction"),
        (2, "split path construction"),
        (3, "split path construction"),
        (4, "format string"),
        (5, "base path"),
        (6, "parent arithmetic"),
    ]


def test_fenced_code_untouched() -> None:
    text = (
        "```bash\n"
        "cat docs/research/2026-01-01-a.md\n"
        "~~~\n"
        "[x](../roadmap.md) `docs/research/_a_results.json`\n"
        "```\n"
        "after `docs/research/_a_results.json`\n"
        "~~~~\n"
        "`docs/research/RESEARCH_SNAPSHOT.md`\n"
        "~~~~\n"
    )
    out, unresolved = rewrite_markdown(text, A_OLD, A_NEW, MAPPING)
    assert out == text.replace(
        "after `docs/research/_a_results.json`",
        "after `research/allocation/a-study/_a_results.json`",
    )
    # Fenced mentions are listed for review (lines 2, 4 and 8) but never modified.
    assert [(u.line, u.reason) for u in unresolved] == [
        (2, "fenced code"),
        (4, "fenced code"),
        (8, "fenced code"),
    ]


def test_link_fragment_and_title_preserved() -> None:
    text = "[r](../roadmap.md#status \"Road map\") ![img](briefs/2026-01-01-b.md 'x')\n"
    out, unresolved = rewrite_markdown(text, A_OLD, A_NEW, MAPPING)
    assert out == (
        '[r](../../../docs/roadmap.md#status "Road map")'
        " ![img](../../audits/b-study/2026-01-01-b.md 'x')\n"
    )
    assert unresolved == []


def test_longest_token_not_partially_replaced() -> None:
    code = (
        'A = "docs/research/_a_results.json.bak"\n'
        'B = "docs/research/_a_results.jsonl"\n'
        "# see docs/research/_a_results.json.\n"
        "# docs/research/RESEARCH_SNAPSHOT.md:21-29 is the source\n"
    )
    out, unresolved = rewrite_code(code, MAPPING)
    lines = out.split("\n")
    assert lines[0:2] == code.split("\n")[0:2]
    assert lines[2] == "# see research/allocation/a-study/_a_results.json."
    assert lines[3] == "# research/RESEARCH_SNAPSHOT.md:21-29 is the source"
    assert sorted((u.line, u.reason) for u in unresolved) == [
        (1, "not in manifest"),
        (2, "not in manifest"),
    ]


# --- rewrite CLI on a throwaway git repository ------------------------------------------------

PINNED = PINNED_PATHS[0]
PINNED_NEW = "research/audits/event/2026-10-21-event-timing-audit.md"
LINKY = "See `docs/research/2026-01-01-a.md` and [r](../roadmap.md).\n"


def _git_tree(tmp_path: Path) -> Path:
    """Build a tiny git repo with research files, consumers and the reorg's own files."""
    root = tmp_path / "repo"
    _write(root, A_OLD, "# A\n[b](briefs/2026-01-01-b.md) [r](../roadmap.md)\n")
    _write(root, B_OLD, "# B\n[a](../2026-01-01-a.md)\n")
    _write(root, f"{OLD}_a_results.json", '{"path": "docs/research/2026-01-01-a.md"}\n')
    _write(root, PINNED, "# pinned\n[a](../2026-01-01-a.md) `docs/research/2026-01-01-a.md`\n")
    _write(root, "docs/roadmap.md", "# Roadmap\n[a](research/2026-01-01-a.md)\n")
    _write(root, "docs/plain.md", "nothing to see\n")
    _write(root, "README.md", "[s](docs/research/briefs/2026-01-01-b.md)\n")
    _write(root, "AGENTS.md", "check `docs/research/2026-01-01-a.md` first\n")
    _write(root, "scripts/driver.py", 'OUT = ROOT / "docs/research/_a_results.json"\n')
    _write(root, "scripts/notes.md", "code-scope markdown: docs/research/2026-01-01-a.md\n")
    _write(root, ".claude/skills/x/SKILL.md", "read docs/research/2026-01-01-a.md\n")
    _write(root, "src/pkg/mod.py", "X = 1\n")
    _write(root, "scripts/research_reorg.py", 'A = "docs/research/2026-01-01-a.md"\n')
    _write(root, "tests/scripts/test_research_reorg.py", 'A = "docs/research/2026-01-01-a.md"\n')
    for rel in (
        "docs/superpowers/plans/2026-11-05-research-reorganization.md",
        "docs/superpowers/plans/2026-11-05-research-reorg-recipe.md",
        "docs/superpowers/specs/2026-11-05-research-reorganization-and-consolidation-design.md",
    ):
        _write(root, rel, LINKY.replace("../roadmap.md", "../../roadmap.md"))
    _write(root, "docs/keep-out.md", LINKY)
    (root / "CLAUDE.md").symlink_to("AGENTS.md")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    # Scratch the git index never sees: must be ignored by every scope.
    _write(root, ".claude/skills/x-workspace/out.md", "docs/research/2026-01-01-a.md\n")
    _write(root, "docs/untracked.md", "docs/research/2026-01-01-a.md\n")
    manifest = tmp_path / "manifest.csv"
    rows = [
        (A_OLD, A_NEW),
        (B_OLD, B_NEW),
        (f"{OLD}_a_results.json", "research/allocation/a-study/_a_results.json"),
        (PINNED, PINNED_NEW),
    ]
    manifest.write_text(
        "current_path,proposed_path,folder,kind,notes\n"
        + "".join(f'"{c}","{p}","","report",""\n' for c, p in rows)
    )
    return root


def _run(root: Path, scope: str, *extra: str) -> dict:  # type: ignore[type-arg]
    report = root.parent / f"report-{scope}.json"
    argv = [
        "rewrite",
        "--manifest",
        str(root.parent / "manifest.csv"),
        "--scope",
        scope,
        "--report",
        str(report),
        "--root",
        str(root),
        *extra,
    ]
    assert main(argv) == 0
    text = report.read_text()
    assert text.endswith("}\n")
    return json.loads(text)


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink() and ".git" not in p.relative_to(root).parts
    }


def test_frozen_file_is_never_written(tmp_path: Path) -> None:
    root = _git_tree(tmp_path)
    frozen = [root / PINNED, root / f"{OLD}_a_results.json"]
    past = 1_000_000_000
    for path in frozen:
        os.utime(path, (past, past))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in frozen}
    report = _run(root, "moved-markdown")
    assert report["skipped_frozen"] == 2
    assert {c["file"] for c in report["changed"]} == {A_OLD, B_OLD}
    for path in frozen:
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before[path]
    for scope in ("fixed-markdown", "code"):
        _run(root, scope)
    for path in frozen:
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before[path]


def test_report_lists_every_changed_file(tmp_path: Path) -> None:
    root = _git_tree(tmp_path)
    for scope, expected in (
        ("moved-markdown", {A_OLD: 2, B_OLD: 1}),
        ("fixed-markdown", {"docs/roadmap.md": 1, "README.md": 1, "docs/keep-out.md": 1}),
        (
            "code",
            {
                "AGENTS.md": 1,
                "scripts/driver.py": 1,
                "scripts/notes.md": 1,
                ".claude/skills/x/SKILL.md": 1,
            },
        ),
    ):
        before = _snapshot(root)
        report = _run(root, scope)
        assert report["scope"] == scope and report["dry_run"] is False
        changed = {c["file"]: c["replacements"] for c in report["changed"]}
        assert changed == expected
        after = _snapshot(root)
        assert {k for k in after if after[k] != before[k]} == set(expected)
        assert report["changed"] == sorted(report["changed"], key=lambda c: c["file"])
    assert (root / "docs/roadmap.md").read_text() == f"# Roadmap\n[a](../{A_NEW})\n"
    assert (root / A_OLD).read_text() == (
        "# A\n[b](../../audits/b-study/2026-01-01-b.md) [r](../../../docs/roadmap.md)\n"
    )


def test_always_excluded_files_not_processed(tmp_path: Path) -> None:
    root = _git_tree(tmp_path)
    before = _snapshot(root)
    fixed = _run(root, "fixed-markdown", "--exclude", "docs/keep-out.md")
    # The CLAUDE.md symlink would write through to AGENTS.md: never followed.
    assert (root / "AGENTS.md").read_bytes() == before["AGENTS.md"]
    assert "CLAUDE.md" in fixed["excluded"]
    code = _run(root, "code")
    after = _snapshot(root)
    always = [
        "docs/superpowers/plans/2026-11-05-research-reorg-recipe.md",
        "docs/superpowers/plans/2026-11-05-research-reorganization.md",
        "docs/superpowers/specs/2026-11-05-research-reorganization-and-consolidation-design.md",
    ]
    for rel in [
        *always,
        "docs/keep-out.md",
        "scripts/research_reorg.py",
        "tests/scripts/test_research_reorg.py",
    ]:
        assert after[rel] == before[rel], rel
    assert set(always) | {"docs/keep-out.md"} <= set(fixed["excluded"])
    assert {"scripts/research_reorg.py", "tests/scripts/test_research_reorg.py"} <= set(
        code["excluded"]
    )
    # Untracked and git-ignored scratch is never in a scope.
    assert after[".claude/skills/x-workspace/out.md"] == before[".claude/skills/x-workspace/out.md"]
    assert after["docs/untracked.md"] == before["docs/untracked.md"]
    assert fixed["status"] == code["status"] == "ok"
    reported = {u["file"] for r in (fixed, code) for u in r["unresolved"]}
    assert not reported & (set(always) | {"docs/keep-out.md", "scripts/research_reorg.py"})


def test_recorded_hash_plan_is_never_rewritten(tmp_path: Path) -> None:
    """The plan file restored to its recorded sha256 must stay byte-identical in every scope."""
    root = _git_tree(tmp_path)
    plan = "docs/superpowers/plans/2026-10-22-event-evidence-resolution.md"
    _write(
        root, plan, "see `docs/research/2026-01-01-a.md` and [a](../../research/2026-01-01-a.md)\n"
    )
    subprocess.run(["git", "add", plan], cwd=root, check=True)
    before = (root / plan).read_bytes()
    reports = [_run(root, scope, "--dry-run") for scope in ("fixed-markdown", "code")]
    assert plan in reports[0]["excluded"]
    for report in reports:
        assert plan not in {c["file"] for c in report["changed"]}
        assert plan not in {u["file"] for u in report["unresolved"]}
    _run(root, "fixed-markdown")
    assert (root / plan).read_bytes() == before


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    root = _git_tree(tmp_path)
    before = _snapshot(root)
    for scope in ("moved-markdown", "fixed-markdown", "code"):
        report = _run(root, scope, "--dry-run")
        assert report["dry_run"] is True
        assert report["changed"], scope
    assert _snapshot(root) == before


def test_scope_moved_markdown_reads_proposed_path_when_present_else_current(
    tmp_path: Path,
) -> None:
    root = _git_tree(tmp_path)
    # Simulate a partial move: A already sits at its new home, B has not moved.
    (root / A_NEW).parent.mkdir(parents=True)
    (root / A_OLD).rename(root / A_NEW)
    report = _run(root, "moved-markdown")
    assert {c["file"] for c in report["changed"]} == {A_NEW, B_OLD}
    assert not (root / A_OLD).exists()
    assert (root / A_NEW).read_text() == (
        "# A\n[b](../../audits/b-study/2026-01-01-b.md) [r](../../../docs/roadmap.md)\n"
    )
    assert (root / B_OLD).read_text() == "# B\n[a](../../allocation/a-study/2026-01-01-a.md)\n"


def test_moved_markdown_rerun_after_move_is_stable(tmp_path: Path) -> None:
    root = _git_tree(tmp_path)
    for row_old, row_new in ((A_OLD, A_NEW), (B_OLD, B_NEW)):
        (root / row_new).parent.mkdir(parents=True, exist_ok=True)
        (root / row_old).rename(root / row_new)
    _write(root, "research/RESEARCH_SNAPSHOT.md", "x\n")
    _run(root, "moved-markdown")
    first = _snapshot(root)
    second = _run(root, "moved-markdown")
    assert second["changed"] == []
    assert _snapshot(root) == first


# --- Fix round 1 (review I-1, rulings R7a-R7e) ----------------------------------------------


def test_multi_range_line_suffix_kept() -> None:
    text = (
        "See `docs/research/2026-01-01-a.md:27-42, 360-403` and "
        "`docs/research/_a_results.json:5,9-11,  20`.\n"
    )
    out, unresolved = rewrite_markdown(text, FIXED, FIXED, MAPPING)
    assert out == (
        f"See `{A_NEW}:27-42, 360-403` and "
        "`research/allocation/a-study/_a_results.json:5,9-11,  20`.\n"
    )
    assert unresolved == []


def test_mapped_token_followed_by_brace_or_percent_is_flagged_not_changed() -> None:
    code = (
        'A = f"docs/research/_a_results.json{suffix}"\n'
        'B = "docs/research/_a_results.json%s" % ext\n'
        'C = "docs/research/2026-01-01-a.md{x}".format(x=1)\n'
    )
    out, unresolved = rewrite_code(code, MAPPING)
    assert out == code
    assert [(u.line, u.reason) for u in unresolved] == [
        (1, "format string"),
        (2, "format string"),
        (3, "format string"),
    ]


def test_complete_path_in_fstring_is_rewritten() -> None:
    code = 'msg = f"brief docs/research/briefs/2026-01-01-b.md §6, value {v}"\n'
    out, unresolved = rewrite_code(code, MAPPING)
    assert out == f'msg = f"brief {B_NEW} §6, value {{v}}"\n'
    assert unresolved == []


def test_directory_link_left_unchanged_and_reported() -> None:
    # Moved file: unmapped old-tree targets keep their raw text (no new literal appears).
    text = "[briefs](briefs/) [gone](_gone.json) [tree](./)\n"
    out, unresolved = rewrite_markdown(text, A_OLD, A_NEW, MAPPING)
    assert out == text
    assert [u.reason for u in unresolved] == [
        "directory citation",
        "not in manifest",
        "directory citation",
    ]
    assert OLD not in out


def test_link_with_query_string_handled() -> None:
    text = "[a](briefs/2026-01-01-b.md?plain=1#L3) [r](../roadmap.md?x=1) [g](_gone.md?x=1)\n"
    out, unresolved = rewrite_markdown(text, A_OLD, A_NEW, MAPPING)
    assert out == (
        "[a](../../audits/b-study/2026-01-01-b.md?plain=1#L3) "
        "[r](../../../docs/roadmap.md?x=1) [g](_gone.md?x=1)\n"
    )
    assert [u.reason for u in unresolved] == ["not in manifest"]


def test_fenced_mention_reported_not_modified() -> None:
    text = "```bash\ncat docs/research/_a_results.json\n```\n~~~\n`docs/research/briefs/`\n~~~\n"
    for old_file, new_file in ((FIXED, FIXED), (A_OLD, A_NEW)):
        out, unresolved = rewrite_markdown(text, old_file, new_file, MAPPING)
        assert out == text
        assert [(u.line, u.reason) for u in unresolved] == [(2, "fenced code"), (5, "fenced code")]


def test_reference_style_definition_rewritten() -> None:
    text = (
        '[snap]: RESEARCH_SNAPSHOT.md#roster "Snapshot"\n'
        "  [road]: <../roadmap.md>\n"
        "[^1]: a footnote, not a link: docs/research/2026-01-01-a.md\n"
        "[dir]: briefs/\n"
    )
    out, unresolved = rewrite_markdown(text, A_OLD, A_NEW, MAPPING)
    assert out.split("\n")[:4] == [
        '[snap]: ../../RESEARCH_SNAPSHOT.md#roster "Snapshot"',
        "  [road]: <../../../docs/roadmap.md>",
        "[^1]: a footnote, not a link: docs/research/2026-01-01-a.md",
        "[dir]: briefs/",
    ]
    assert [(u.line, u.reason) for u in unresolved] == [
        (3, "prose mention"),
        (4, "directory citation"),
    ]


def test_parenthesized_title_and_parens_in_path_handled() -> None:
    text = "[r](../roadmap.md (Road map)) [w](../notes(1).md) [b](briefs/2026-01-01-b.md (t))\n"
    out, unresolved = rewrite_markdown(text, A_OLD, A_NEW, MAPPING)
    assert out == (
        "[r](../../../docs/roadmap.md (Road map)) [w](../../../docs/notes(1).md)"
        " [b](../../audits/b-study/2026-01-01-b.md (t))\n"
    )
    assert unresolved == []


def test_unparseable_link_form_is_reported() -> None:
    # Two levels of parentheses: INLINE_LINK cannot parse it, so it must be reported.
    text = "[x](../a((b)).md) and [y](docs/research/a b.md)\n"
    out, unresolved = rewrite_markdown(text, A_OLD, A_NEW, MAPPING)
    assert out == text
    reasons = [u.reason for u in unresolved]
    assert reasons.count("unsupported link form") == 2
    out, unresolved = rewrite_markdown("[y](research/x y.md)\n", FIXED, FIXED, MAPPING)
    assert [u.reason for u in unresolved] == ["unsupported link form"]
    # A file that does not move: an unparseable link outside the old tree is harmless.
    out, unresolved = rewrite_markdown("[z](../a((b)).md)\n", FIXED, FIXED, MAPPING)
    assert unresolved == []


def test_preflight_failure_writes_nothing_and_still_writes_report(tmp_path: Path) -> None:
    root = _git_tree(tmp_path)
    _write(root, "docs/binary.md", b"\xff\xfe docs/research/2026-01-01-a.md\n")
    _write(root, "docs/locked.md", "`docs/research/2026-01-01-a.md`\n")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    (root / "docs/locked.md").chmod(0)
    before = _snapshot_readable(root)
    report_path = tmp_path / "report.json"
    argv = [
        "rewrite",
        "--manifest",
        str(tmp_path / "manifest.csv"),
        "--scope",
        "fixed-markdown",
        "--report",
        str(report_path),
        "--root",
        str(root),
    ]
    try:
        assert main(argv) == 1
        report = json.loads(report_path.read_text())
        assert report["status"] == "failed"
        assert "docs/binary.md" in report["error"] and "docs/locked.md" in report["error"]
        assert report["skipped_undecodable"] == ["docs/binary.md"]
        assert [p.split(":")[0] for p in report["skipped_unreadable"]] == ["docs/locked.md"]
        assert report["written"] == []
        assert report["changed"]  # other files would have changed, yet none was written
        assert _snapshot_readable(root) == before
        # A failure before any rewrite (bad manifest) still produces a report.
        argv[2] = str(tmp_path / "no-such-manifest.csv")
        assert main(argv) == 1
        report = json.loads(report_path.read_text())
        assert report["status"] == "failed" and "FileNotFoundError" in report["error"]
    finally:
        (root / "docs/locked.md").chmod(0o644)


def _snapshot_readable(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and os.access(p, os.R_OK)
        and ".git" not in p.relative_to(root).parts
    }


def test_write_is_atomic_and_preserves_mode(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import scripts.research_reorg as reorg

    root = _git_tree(tmp_path)
    target = root / "docs/roadmap.md"
    target.chmod(0o640)
    report = _run(root, "fixed-markdown")
    assert report["status"] == "ok" and "docs/roadmap.md" in report["written"]
    assert (target.stat().st_mode & 0o777) == 0o640
    assert target.read_text() == f"# Roadmap\n[a](../{A_NEW})\n"
    assert not list(root.rglob("*.tmp"))
    # A failing replace leaves the original bytes and no temporary file behind.
    _write(root, "docs/roadmap.md", "# Roadmap\n[a](research/2026-01-01-a.md)\n")
    original = target.read_bytes()

    real_replace = reorg.os.replace

    def _boom(src: str, dst: object) -> None:
        if str(dst).endswith("roadmap.md"):
            raise OSError("disk full")
        real_replace(src, dst)

    monkeypatch.setattr(reorg.os, "replace", _boom)
    report_path = tmp_path / "r.json"
    argv = ["rewrite", "--manifest", str(tmp_path / "manifest.csv"), "--scope"]
    argv += ["fixed-markdown", "--report", str(report_path), "--root", str(root)]
    assert main(argv) == 1
    failed = json.loads(report_path.read_text())
    assert failed["status"] == "failed" and "disk full" in failed["error"]
    assert "docs/roadmap.md" not in failed["written"]
    assert target.read_bytes() == original
    assert not list(root.rglob("*.tmp"))


# --- move and verify stages on a throwaway git repository -------------------------------------

BRIEF_OLD = f"{OLD}briefs/2026-10-23-century-leverage.md"
BRIEF_NEW = "research/allocation/century-leverage/2026-10-23-century-leverage-brief.md"
JSON_OLD = f"{OLD}_century_results.json"
JSON_NEW = "research/allocation/century-leverage/_century_results.json"
REPORT_OLD = f"{OLD}2026-10-23-century-leverage.md"
REPORT_NEW = "research/allocation/century-leverage/2026-10-23-century-leverage.md"
GIT_ID = ["-c", "user.name=Test", "-c", "user.email=test@example.com"]


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *GIT_ID, *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def _move_repo(tmp_path: Path) -> tuple[Path, list[ManifestRow], dict]:  # type: ignore[type-arg]
    """Build and commit a small repo with research files; return root, rows and ledger."""
    root = tmp_path / "repo"
    _write(root, BRIEF_OLD, "# brief\nregistered before the run\n")
    _write(root, JSON_OLD, '{"sharpe": 0.8, "path": "docs/research/x.md"}\n')
    _write(root, REPORT_OLD, "# report\nNO-GO\n")
    _write(root, "docs/other.md", "outside the tree\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    rows = [
        _row(JSON_OLD, JSON_NEW, "result"),
        _row(REPORT_OLD, REPORT_NEW),
        _row(BRIEF_OLD, BRIEF_NEW, "brief"),
    ]
    ledger = build_ledger(rows, root, "test-head")
    return root, rows, ledger


def _tree_state(root: Path) -> tuple[dict[str, bytes], str]:
    files = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }
    return files, _git(root, "status", "--porcelain")


def test_apply_moves_preserves_history_and_content(tmp_path: Path) -> None:
    root, rows, _ = _move_repo(tmp_path)
    before = {r.current_path: (root / r.current_path).read_bytes() for r in rows}
    moved = apply_moves(rows, root, dry_run=False)
    assert moved == [(r.current_path, r.proposed_path) for r in rows]
    _git(root, "commit", "-q", "-m", "move")
    for row in rows:
        assert (root / row.proposed_path).read_bytes() == before[row.current_path]
        assert not (root / row.current_path).exists()
    status = _git(root, "diff", "-M", "--name-status", "HEAD~1", "HEAD").splitlines()
    assert sorted(status) == sorted(f"R100\t{r.current_path}\t{r.proposed_path}" for r in rows)
    log = _git(root, "log", "--follow", "--oneline", "--", BRIEF_NEW).splitlines()
    assert len(log) == 2


def test_apply_moves_refuses_existing_destination(tmp_path: Path) -> None:
    root, rows, _ = _move_repo(tmp_path)
    _write(root, REPORT_NEW, "already here\n")
    before = _tree_state(root)
    with pytest.raises(MoveRefusedError, match="destination exists"):
        apply_moves(rows, root, dry_run=False)
    assert _tree_state(root) == before
    assert (root / REPORT_NEW).read_text() == "already here\n"


def test_apply_moves_refuses_missing_source_before_moving_anything(tmp_path: Path) -> None:
    root, rows, _ = _move_repo(tmp_path)
    _git(root, "rm", "-q", BRIEF_OLD)  # the last row: earlier rows must not move either
    before = _tree_state(root)
    with pytest.raises(MoveRefusedError, match="source missing"):
        apply_moves(rows, root, dry_run=False)
    assert _tree_state(root) == before
    assert (root / JSON_OLD).exists()


def test_apply_moves_dry_run_changes_nothing(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    root, rows, _ = _move_repo(tmp_path)
    before = _tree_state(root)
    plan = apply_moves(rows, root, dry_run=True)
    assert plan == [(r.current_path, r.proposed_path) for r in rows]
    assert _tree_state(root) == before
    assert not (root / "research").exists()
    assert BRIEF_NEW in capsys.readouterr().out


def test_apply_moves_skips_rows_that_do_not_move(tmp_path: Path) -> None:
    root, rows, _ = _move_repo(tmp_path)
    rows = [*rows, _row("docs/other.md", "docs/other.md")]
    moved = apply_moves(rows, root, dry_run=False)
    assert ("docs/other.md", "docs/other.md") not in moved
    assert len(moved) == 3


def test_verify_passes_after_clean_move(tmp_path: Path) -> None:
    root, rows, ledger = _move_repo(tmp_path)
    apply_moves(rows, root, dry_run=False)
    assert verify_moved(rows, ledger, root) == []


def test_verify_fails_when_frozen_file_bytes_change(tmp_path: Path) -> None:
    root, rows, ledger = _move_repo(tmp_path)
    apply_moves(rows, root, dry_run=False)
    # A re-serialization that keeps the meaning but not the bytes must still fail.
    (root / JSON_NEW).write_text('{"sharpe":0.8,"path":"docs/research/x.md"}\n')
    problems = verify_moved(rows, ledger, root)
    assert any("sha256" in p and JSON_NEW in p for p in problems)


def test_verify_fails_when_unfrozen_markdown_bytes_change(tmp_path: Path) -> None:
    root, rows, ledger = _move_repo(tmp_path)
    apply_moves(rows, root, dry_run=False)
    (root / REPORT_NEW).write_text("# report\nGO\n")
    assert any(REPORT_NEW in p for p in verify_moved(rows, ledger, root))


def test_verify_flags_leftover_tracked_file_under_old_root(tmp_path: Path) -> None:
    root, rows, ledger = _move_repo(tmp_path)
    apply_moves(rows, root, dry_run=False)
    _write(root, f"{OLD}stray.md", "left behind\n")
    _git(root, "add", f"{OLD}stray.md")
    problems = verify_moved(rows, ledger, root)
    assert any("tracked" in p and f"{OLD}stray.md" in p for p in problems)


def test_verify_flags_untracked_leftover_but_ignores_pycache(tmp_path: Path) -> None:
    root, rows, ledger = _move_repo(tmp_path)
    apply_moves(rows, root, dry_run=False)
    _write(root, f"{OLD}__pycache__/x.cpython-312.pyc", b"\x00")
    assert verify_moved(rows, ledger, root) == []
    _write(root, f"{OLD}notes.txt", "scratch\n")
    problems = verify_moved(rows, ledger, root)
    assert any("untracked" in p and f"{OLD}notes.txt" in p for p in problems)


def test_verify_flags_missing_destination_and_extra_file(tmp_path: Path) -> None:
    root, rows, ledger = _move_repo(tmp_path)
    apply_moves(rows, root, dry_run=False)
    (root / REPORT_NEW).unlink()
    _write(root, "research/allocation/century-leverage/extra.md", "new\n")
    problems = verify_moved(rows, ledger, root)
    assert any("missing destination" in p and REPORT_NEW in p for p in problems)
    assert any("extra.md" in p for p in problems)


def test_verify_flags_destination_too_deep(tmp_path: Path) -> None:
    root, rows, ledger = _move_repo(tmp_path)
    deep = "research/allocation/century-leverage/sub/_century_results.json"
    rows = [_row(JSON_OLD, deep, "result"), *rows[1:]]
    ledger = build_ledger(rows, root, "test-head")
    apply_moves(rows, root, dry_run=False)
    assert any("depth" in p and deep in p for p in verify_moved(rows, ledger, root))


def test_remove_leftover_dirs_only_removes_empty_and_pycache(tmp_path: Path) -> None:
    root, rows, _ = _move_repo(tmp_path)
    apply_moves(rows, root, dry_run=False)
    _write(root, f"{OLD}__pycache__/x.cpython-312.pyc", b"\x00")
    (root / OLD / "empty" / "nested").mkdir(parents=True)
    removed = remove_leftover_dirs(root)
    assert not (root / OLD).exists()
    assert f"{OLD}__pycache__" in removed
    _write(root, f"{OLD}keep.md", "real\n")
    remove_leftover_dirs(root)
    assert (root / OLD / "keep.md").exists()


def test_main_move_and_verify_cli(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    root, rows, ledger = _move_repo(tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "current_path,proposed_path,folder,kind,notes\n"
        + "".join(f'"{r.current_path}","{r.proposed_path}","","{r.kind}",""\n' for r in rows)
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    common = ["--manifest", str(manifest), "--root", str(root)]
    assert main(["move", *common, "--dry-run"]) == 0
    assert not (root / "research").exists()
    verify = ["verify", "--stage", "moved", *common, "--ledger", str(ledger_path)]
    assert main(verify) == 1
    assert main(["move", *common]) == 0
    assert main(verify) == 0
    assert "OK" in capsys.readouterr().out
    final = ["verify", "--stage", "final", *common, "--ledger", str(ledger_path)]
    assert main(final) == 1  # the rewrite manifest is required for the final gate
    assert "--rewrite-manifest is required" in capsys.readouterr().out


# --- verify --stage rewritten ------------------------------------------------------------------

PINNED_OLD = sorted(PINNED_PATHS)[0]
PINNED_NEW = "research/audits/pinned/" + PINNED_OLD.rsplit("/", 1)[-1]


def _rewritten_repo(tmp_path: Path) -> tuple[Path, list[ManifestRow], dict]:  # type: ignore[type-arg]
    """Moved repo whose rows also include a pinned markdown file."""
    root = tmp_path / "repo"
    _write(root, BRIEF_OLD, "# brief\nregistered before the run\n")
    _write(root, JSON_OLD, '{"sharpe": 0.8}\n')
    _write(root, REPORT_OLD, "# report\nsee docs/research/x.md\n")
    _write(root, PINNED_OLD, "# pinned\nsee docs/research/x.md\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    rows = [
        _row(JSON_OLD, JSON_NEW, "result"),
        _row(REPORT_OLD, REPORT_NEW),
        _row(BRIEF_OLD, BRIEF_NEW, "brief"),
        _row(PINNED_OLD, PINNED_NEW),
    ]
    ledger = build_ledger(rows, root, "test-head")
    apply_moves(rows, root, dry_run=False)
    return root, rows, ledger


def test_verify_rewritten_passes_when_bytes_still_match_ledger(tmp_path: Path) -> None:
    root, rows, ledger = _rewritten_repo(tmp_path)
    assert verify_rewritten(rows, ledger, root, set()) == []


def test_verify_rewritten_flags_unlisted_change(tmp_path: Path) -> None:
    root, rows, ledger = _rewritten_repo(tmp_path)
    (root / REPORT_NEW).write_text("# report\nsee research/x.md\n")
    problems = verify_rewritten(rows, ledger, root, set())
    assert any("unlisted change" in p and REPORT_NEW in p for p in problems)


def test_verify_rewritten_allows_change_listed_in_rewrite_manifest(tmp_path: Path) -> None:
    root, rows, ledger = _rewritten_repo(tmp_path)
    (root / REPORT_NEW).write_text("# report\nsee research/x.md\n")
    assert verify_rewritten(rows, ledger, root, {REPORT_NEW}) == []


def test_verify_rewritten_flags_frozen_change_even_when_listed(tmp_path: Path) -> None:
    root, rows, ledger = _rewritten_repo(tmp_path)
    (root / JSON_NEW).write_text('{"sharpe":0.8}\n')
    problems = verify_rewritten(rows, ledger, root, {JSON_NEW})
    assert any("frozen" in p and JSON_NEW in p for p in problems)


def test_verify_rewritten_flags_pinned_change_even_when_listed(tmp_path: Path) -> None:
    root, rows, ledger = _rewritten_repo(tmp_path)
    (root / PINNED_NEW).write_text("# pinned\nsee research/x.md\n")
    problems = verify_rewritten(rows, ledger, root, {PINNED_NEW})
    assert any("frozen" in p and PINNED_NEW in p for p in problems)


def test_verify_rewritten_flags_listed_path_that_must_not_change(tmp_path: Path) -> None:
    root, rows, ledger = _rewritten_repo(tmp_path)
    problems = verify_rewritten(rows, ledger, root, {JSON_NEW, PINNED_NEW, "research/ghost.md"})
    assert any("cannot be rewritten" in p and JSON_NEW in p for p in problems)
    assert any("cannot be rewritten" in p and PINNED_NEW in p for p in problems)
    assert any("not in manifest" in p and "research/ghost.md" in p for p in problems)


def test_verify_rewritten_flags_missing_and_extra_files(tmp_path: Path) -> None:
    root, rows, ledger = _rewritten_repo(tmp_path)
    (root / BRIEF_NEW).unlink()
    _write(root, "research/allocation/century-leverage/extra.md", "new\n")
    problems = verify_rewritten(rows, ledger, root, set())
    assert any("missing destination" in p and BRIEF_NEW in p for p in problems)
    assert any("extra.md" in p for p in problems)


def test_load_rewrite_manifest_accepts_files_list_and_report_shape(tmp_path: Path) -> None:
    files = tmp_path / "files.json"
    files.write_text(json.dumps({"files": ["research/a.md", "research/b.md"]}))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"changed": [{"file": "research/c.md", "replacements": 2}]}))
    assert load_rewrite_manifest(files) == {"research/a.md", "research/b.md"}
    assert load_rewrite_manifest(report) == {"research/c.md"}


def test_main_verify_rewritten_cli(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    root, rows, ledger = _rewritten_repo(tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "current_path,proposed_path,folder,kind,notes\n"
        + "".join(f'"{r.current_path}","{r.proposed_path}","","{r.kind}",""\n' for r in rows)
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    listing = tmp_path / "rewrite.json"
    listing.write_text(json.dumps({"files": []}))
    argv = ["verify", "--stage", "rewritten", "--manifest", str(manifest), "--root", str(root)]
    argv += ["--ledger", str(ledger_path), "--rewrite-manifest", str(listing)]
    assert main(argv) == 0
    (root / REPORT_NEW).write_text("# report\nsee research/x.md\n")
    assert main(argv) == 1
    assert f"PROBLEM unlisted change: {REPORT_NEW}" in capsys.readouterr().out
    listing.write_text(json.dumps({"files": [REPORT_NEW]}))
    assert main(argv) == 0
    assert main(argv[:-2]) == 1  # --rewrite-manifest is required for this stage


# --- verify --stage final ----------------------------------------------------------------------


def _final_repo(tmp_path: Path) -> tuple[Path, list[ManifestRow], dict]:  # type: ignore[type-arg]
    """Rewritten-stage repo with the emptied docs/research folders removed, as `move` does."""
    root, rows, ledger = _rewritten_repo(tmp_path)
    remove_leftover_dirs(root)
    return root, rows, ledger


def _final(root: Path, rows: list[ManifestRow], ledger: dict, **kw) -> list[str]:  # type: ignore[type-arg,no-untyped-def]
    """Run verify_final on a one-pinned-file fixture (production expects twelve)."""
    kw.setdefault("expected_pinned", 1)
    return verify_final(rows, ledger, root, set(), **kw)


def test_verify_final_passes_clean_tree_and_allows_new_files(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    assert _final(root, rows, ledger) == []
    _write(root, "research/README.md", "# index\n")
    _git(root, "add", "research/README.md")
    assert _final(root, rows, ledger) == []  # default allow-list
    _write(root, "research/audits/new-study/notes.md", "x\n")
    _git(root, "add", "research/audits/new-study/notes.md")
    problems = _final(root, rows, ledger)
    assert any("unexpected tracked file" in p and "new-study/notes.md" in p for p in problems)
    assert _final(root, rows, ledger, allow_new=("README.md", "audits/new-study/*")) == []


def test_verify_final_flags_docs_research_leftover(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    _write(root, f"{OLD}stray.md", "left behind\n")
    _git(root, "add", f"{OLD}stray.md")
    problems = _final(root, rows, ledger)
    assert any("tracked" in p and f"{OLD}stray.md" in p for p in problems)
    assert any(p == "docs/research still exists" for p in problems)


def test_verify_final_flags_empty_docs_research_directory(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    (root / OLD / "empty").mkdir(parents=True)
    assert "docs/research still exists" in _final(root, rows, ledger)


def test_verify_final_flags_missing_destination(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    _git(root, "rm", "-q", "-f", BRIEF_NEW)
    problems = _final(root, rows, ledger)
    assert any("missing destination" in p and BRIEF_NEW in p for p in problems)
    assert any("destination not tracked" in p and BRIEF_NEW in p for p in problems)


def test_verify_final_flags_duplicate_proposed_path(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    dup = [*rows, _row(f"{OLD}other.md", REPORT_NEW)]
    problems = _final(root, dup, ledger)
    assert any("duplicate proposed path" in p and REPORT_NEW in p for p in problems)


def test_verify_final_flags_tracked_file_too_deep(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    deep = "research/allocation/century-leverage/sub/notes.md"
    _write(root, deep, "x\n")
    _git(root, "add", deep)
    problems = _final(root, rows, ledger, allow_new=("*",))
    assert any("deeper than" in p and deep in p for p in problems)


def test_verify_final_allows_one_extra_level_under_allow_deeper_prefix(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    deep = "research/audits/root-cause-2026-11-04/lenses/x.md"
    _write(root, deep, "x\n")
    _git(root, "add", deep)
    patterns = ("audits/root-cause-2026-11-04/lenses/*",)
    problems = _final(root, rows, ledger, allow_new=patterns, allow_deeper=())
    assert any("deeper than" in p and deep in p for p in problems)
    prefix = ("audits/root-cause-2026-11-04",)
    assert _final(root, rows, ledger, allow_new=patterns, allow_deeper=prefix) == []
    assert _final(root, rows, ledger, allow_new=patterns) == []  # default prefix
    other = "research/audits/other-study/lenses/y.md"
    _write(root, other, "y\n")
    _git(root, "add", other)
    problems = _final(root, rows, ledger, allow_new=("*",))
    assert any("deeper than" in p and other in p for p in problems)


def test_verify_final_still_rejects_two_extra_levels(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    deep = "research/audits/root-cause-2026-11-04/lenses/a/x.md"
    _write(root, deep, "x\n")
    _git(root, "add", deep)
    problems = _final(
        root, rows, ledger, allow_new=("*",), allow_deeper=("audits/root-cause-2026-11-04",)
    )
    assert any("deeper than" in p and deep in p for p in problems)


def test_verify_final_default_allow_new_lists_reading_documents() -> None:
    base = "audits/root-cause-2026-11-04"
    assert set(DEFAULT_ALLOW_NEW) >= {
        "README.md",
        "MOVED.csv",
        f"{base}/README.md",
        f"{base}/reference-sheet.md",
        f"{base}/findings/2026-11-05-findings-plain-english.md",
        f"{base}/reviews/2026-11-05-direction-reviews-consolidated.md",
    }
    # The pre-move home of the consolidated review was dropped once the sub-folder move landed.
    assert f"{base}/2026-11-05-direction-reviews-consolidated.md" not in DEFAULT_ALLOW_NEW


def test_main_verify_final_allow_deeper_cli(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    root, rows, ledger = _final_repo(tmp_path)
    deep = "research/allocation/century-leverage/sub/x.md"
    _write(root, deep, "x\n")
    _git(root, "add", deep)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "current_path,proposed_path,folder,kind,notes\n"
        + "".join(f'"{r.current_path}","{r.proposed_path}","","{r.kind}",""\n' for r in rows)
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    listing = tmp_path / "rewrite.json"
    listing.write_text(json.dumps({"files": []}))
    argv = ["verify", "--stage", "final", "--manifest", str(manifest), "--root", str(root)]
    argv += ["--ledger", str(ledger_path), "--rewrite-manifest", str(listing)]
    argv += ["--allow-new", "allocation/century-leverage/sub/*"]
    assert main(argv) == 1
    assert "tracked file deeper than" in capsys.readouterr().out
    main([*argv, "--allow-deeper", "allocation/century-leverage"])
    out = capsys.readouterr().out
    assert "deeper than" not in out


def test_verify_final_accepts_artifact_package_as_study_folder(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    pkg = "research/crypto/artifacts-2026-10-25-crypto-web-batch/_run.json"
    _write(root, pkg, "{}\n")
    _git(root, "add", pkg)
    assert _final(root, rows, ledger, allow_new=("crypto/artifacts-*/*",)) == []


def test_verify_final_flags_pinned_digest_change(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    (root / PINNED_NEW).write_text("# pinned\nsee research/x.md\n")
    problems = verify_final(rows, ledger, root, {PINNED_NEW}, expected_pinned=1)
    assert any("pinned digest differs" in p and PINNED_NEW in p for p in problems)


def test_verify_final_flags_wrong_pinned_count(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    problems = verify_final(rows, ledger, root, set())  # production expects twelve
    assert any("1 pinned rows in the manifest, expected 12" in p for p in problems)


def test_verify_final_flags_unlisted_change_and_untracked_extra(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    (root / REPORT_NEW).write_text("# report\nsee research/x.md\n")
    _write(root, "research/allocation/century-leverage/extra.md", "new\n")  # untracked
    problems = _final(root, rows, ledger)
    assert any("unlisted change" in p and REPORT_NEW in p for p in problems)
    assert any("unexpected file" in p and "extra.md" in p for p in problems)


def test_main_verify_final_cli(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    root, rows, ledger = _final_repo(tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "current_path,proposed_path,folder,kind,notes\n"
        + "".join(f'"{r.current_path}","{r.proposed_path}","","{r.kind}",""\n' for r in rows)
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    listing = tmp_path / "rewrite.json"
    listing.write_text(json.dumps({"files": []}))
    argv = ["verify", "--stage", "final", "--manifest", str(manifest), "--root", str(root)]
    argv += ["--ledger", str(ledger_path), "--rewrite-manifest", str(listing)]
    assert main(argv) == 1  # the fixture has one pinned row, production expects twelve
    assert "expected 12" in capsys.readouterr().out


def test_verify_final_flags_destination_on_disk_but_untracked(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    _git(root, "rm", "-q", "--cached", BRIEF_NEW)
    problems = _final(root, rows, ledger)
    assert any("destination not tracked" in p and BRIEF_NEW in p for p in problems)
    assert not any("missing destination" in p for p in problems)


def test_verify_final_flags_ledger_entry_not_marked_pinned(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    for entry in ledger["files"]:
        if entry["current_path"] == PINNED_OLD:
            entry["pinned"] = False
    problems = _final(root, rows, ledger)
    assert any("ledger does not mark this file pinned" in p and PINNED_OLD in p for p in problems)


def test_verify_final_require_new_flags_missing_and_untracked(tmp_path: Path) -> None:
    root, rows, ledger = _final_repo(tmp_path)
    required = ("README.md", "MOVED.csv")
    problems = _final(root, rows, ledger, require_new=required)
    assert any("required new file missing" in p and "README.md" in p for p in problems)
    assert any("required new file missing" in p and "MOVED.csv" in p for p in problems)
    _write(root, "research/README.md", "# index\n")  # on disk, not added to git
    problems = _final(root, rows, ledger, require_new=required)
    assert any("required new file not tracked" in p and "README.md" in p for p in problems)
    _git(root, "add", "research/README.md")
    _write(root, "research/MOVED.csv", "a,b\n")
    _git(root, "add", "research/MOVED.csv")
    assert _final(root, rows, ledger, require_new=required) == []


def test_main_verify_final_require_new_cli(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    root, rows, ledger = _final_repo(tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "current_path,proposed_path,folder,kind,notes\n"
        + "".join(f'"{r.current_path}","{r.proposed_path}","","{r.kind}",""\n' for r in rows)
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    listing = tmp_path / "rewrite.json"
    listing.write_text(json.dumps({"files": []}))
    argv = ["verify", "--stage", "final", "--manifest", str(manifest), "--root", str(root)]
    argv += ["--ledger", str(ledger_path), "--rewrite-manifest", str(listing)]
    # Production expects twelve pinned rows, so this fixture always fails that one check;
    # the assertion is on the require-new problem lines only.
    assert main([*argv, "--require-new", "README.md"]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM required new file missing: research/README.md" in out


def test_main_verify_moved_failure_notes_existing_rewrite_manifest(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    root, rows, ledger = _rewritten_repo(tmp_path)
    (root / REPORT_NEW).write_text("# report\nsee research/x.md\n")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "current_path,proposed_path,folder,kind,notes\n"
        + "".join(f'"{r.current_path}","{r.proposed_path}","","{r.kind}",""\n' for r in rows)
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger))
    argv = ["verify", "--stage", "moved", "--manifest", str(manifest), "--root", str(root)]
    argv += ["--ledger", str(ledger_path)]
    assert main(argv) == 1
    assert "NOTE" not in capsys.readouterr().out
    _write(root, DEFAULT_REWRITE_MANIFEST, '{"files": []}\n')
    assert main(argv) == 1  # the exit code does not change
    out = capsys.readouterr().out
    assert out.count("NOTE ") == 1
    assert "only valid before the rewrite" in out


def test_moved_csv_covers_every_manifest_row_and_marks_three_renames(
    tmp_path: Path,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    rows = [
        _row(f"{OLD}z.md", "research/allocation/s1/z.md"),
        _row(
            f"{OLD}briefs/2026-10-23-century-leverage.md",
            "research/allocation/c/2026-10-23-century-leverage-brief.md",
            "brief",
        ),
        _row(
            f"{OLD}briefs/2026-09-28-cross-asset-sleeve.md",
            "research/allocation/x/2026-09-28-cross-asset-sleeve-brief.md",
            "brief",
        ),
        _row(
            f"{OLD}briefs/2026-09-30-static-fractional-leverage.md",
            "research/allocation/s/2026-09-30-static-fractional-leverage-brief.md",
            "brief",
        ),
        _row(f"{OLD}a, odd.md", "research/allocation/s1/a, odd.md"),
    ]
    text = moved_csv_text(rows)
    parsed = list(csv.DictReader(io.StringIO(text)))
    assert list(parsed[0]) == ["old_path", "new_path", "renamed"]
    assert len(parsed) == len(rows)
    assert {r["old_path"] for r in parsed} == {r.current_path for r in rows}
    assert [r["old_path"] for r in parsed] == sorted(r["old_path"] for r in parsed)
    assert sum(r["renamed"] == "yes" for r in parsed) == 3
    assert all(r["renamed"] in ("yes", "no") for r in parsed)
    odd = next(r for r in parsed if r["old_path"].endswith("a, odd.md"))
    assert odd["new_path"] == "research/allocation/s1/a, odd.md" and odd["renamed"] == "no"
    assert moved_csv_text(list(reversed(rows))) == text

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "current_path,proposed_path,folder,kind,notes\n"
        + "".join(f'"{r.current_path}","{r.proposed_path}","","{r.kind}",""\n' for r in rows)
    )
    out = tmp_path / "MOVED.csv"
    assert main(["moved-csv", "--manifest", str(manifest), "--out", str(out)]) == 0
    first = out.read_bytes()
    assert first.decode("utf-8") == text
    assert main(["moved-csv", "--manifest", str(manifest), "--out", str(out)]) == 0
    assert out.read_bytes() == first
    assert "5 rows" in capsys.readouterr().out
