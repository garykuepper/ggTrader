"""Tests for the investigation lens report checker."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_lens_report import check_report

pytestmark = pytest.mark.tooling

SECTIONS = ["Falsifiers", "Steelman", "Questions and answers", "Verification", "Done checklist"]


def _report(skip: str | None = None, empty: str | None = None, extra: str = "") -> str:
    parts = ["# Lens report\n"]
    for name in SECTIONS:
        if name == skip:
            continue
        body = "" if name == empty else "Some content.\n"
        parts.append(f"## {name}\n\n{body}\n")
    parts.append(extra)
    return "\n".join(parts)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "report.md"
    path.write_text(text)
    return path


def test_passes_complete_report(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "next_steps.md").write_text("x")
    report = _write(tmp_path, _report(extra="See `docs/next_steps.md` for context.\n"))
    assert check_report(report, tmp_path) == []


def test_fails_missing_steelman_heading(tmp_path: Path) -> None:
    report = _write(tmp_path, _report(skip="Steelman"))
    problems = check_report(report, tmp_path)
    assert any("Steelman" in p for p in problems)


def test_fails_empty_section(tmp_path: Path) -> None:
    report = _write(tmp_path, _report(empty="Verification"))
    problems = check_report(report, tmp_path)
    assert any("Verification" in p and "empty" in p.lower() for p in problems)


def test_fails_nonexistent_cited_path(tmp_path: Path) -> None:
    report = _write(tmp_path, _report(extra="Cites `src/ggTrader/nope/missing.py` here.\n"))
    problems = check_report(report, tmp_path)
    assert any("src/ggTrader/nope/missing.py" in p for p in problems)


def test_fails_banned_phrase_retire_the_sleeve(tmp_path: Path) -> None:
    report = _write(tmp_path, _report(extra="We should Retire The Sleeve soon.\n"))
    problems = check_report(report, tmp_path)
    assert any("retire the sleeve" in p.lower() for p in problems)


def test_banned_phrase_in_blockquote_is_ignored(tmp_path: Path) -> None:
    report = _write(
        tmp_path,
        _report(
            extra="> Owner: do not recommend to retire the sleeve.\n  > shut down the project\n"
        ),
    )
    assert check_report(report, tmp_path) == []


def test_ignores_paths_in_urls(tmp_path: Path) -> None:
    report = _write(
        tmp_path,
        _report(extra="See `https://example.com/docs/missing.md` and `docs/*.md` globs.\n"),
    )
    assert check_report(report, tmp_path) == []


def test_paths_only_skips_heading_rules(tmp_path: Path) -> None:
    report = _write(tmp_path, "# Synthesis\n\nNo required headings, cites `docs/missing.md`.\n")
    assert check_report(report, tmp_path, paths_only=False) != []
    problems = check_report(report, tmp_path, paths_only=True)
    assert all("heading" not in p.lower() for p in problems)
    assert any("docs/missing.md" in p for p in problems)


def test_research_path_that_exists_passes(tmp_path: Path) -> None:
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "x.md").write_text("x")
    report = _write(tmp_path, _report(extra="See `research/x.md` for the study.\n"))
    assert check_report(report, tmp_path) == []


def test_missing_research_path_fails_and_moved_csv_is_not_consulted(tmp_path: Path) -> None:
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "real.md").write_text("x")
    # A move map naming the missing path must not rescue a stale citation.
    (tmp_path / "research" / "MOVED.csv").write_text(
        "old_path,new_path\nresearch/missing.md,research/real.md\n"
    )
    report = _write(tmp_path, _report(extra="Cites `research/missing.md` here.\n"))
    problems = check_report(report, tmp_path)
    assert any("research/missing.md" in p for p in problems)
