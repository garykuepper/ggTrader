"""Tests for the research reorganization tool (manifest loading and checking)."""

from __future__ import annotations

from pathlib import Path

from scripts.research_reorg import (
    PINNED_PATHS,
    ManifestRow,
    check_manifest,
    load_manifest,
    main,
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
