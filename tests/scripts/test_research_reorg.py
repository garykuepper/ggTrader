"""Tests for the research reorganization tool (manifest checking and the provenance ledger)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from scripts.research_reorg import (
    PINNED_PATHS,
    ManifestRow,
    build_ledger,
    check_manifest,
    load_manifest,
    main,
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
