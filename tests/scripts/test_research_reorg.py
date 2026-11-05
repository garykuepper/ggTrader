"""Tests for the research reorganization tool (manifest, ledger and reference rewriter)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from scripts.research_reorg import (
    PINNED_PATHS,
    ManifestRow,
    Unresolved,
    build_ledger,
    check_manifest,
    load_manifest,
    main,
    rewrite_code,
    rewrite_markdown,
    sha256_file,
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
    assert unresolved == []


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
    _write(root, "docs/binary.md", b"\xff\xfe docs/research/2026-01-01-a.md\n")
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
    assert fixed["skipped_undecodable"] == ["docs/binary.md"]
    reported = {u["file"] for r in (fixed, code) for u in r["unresolved"]}
    assert not reported & (set(always) | {"docs/keep-out.md", "scripts/research_reorg.py"})


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
