"""Tests for the reading-edition lint."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_reading_edition import check_file, load_sheet_terms, main

EM_DASH = "—"
TERMS = {"DSR", "NO-GO"}


def _write(tmp_path: Path, text: str, name: str = "doc.md") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _check(
    tmp_path: Path,
    text: str,
    terms: set[str] | None = None,
    sources: dict[str, str] | None = None,
    max_words: int | None = None,
) -> list[str]:
    path = _write(tmp_path, text)
    return check_file(
        path, TERMS if terms is None else terms, sources or {}, max_words, root=tmp_path
    )


def test_flags_em_dash_and_semicolon_outside_code(tmp_path: Path) -> None:
    problems = _check(tmp_path, f"Fine line.\nBad {EM_DASH} dash here.\nBad; semicolon here.\n")
    assert any(":2:" in p and "em dash" in p for p in problems)
    assert any(":3:" in p and "semicolon" in p for p in problems)
    assert not any(":1:" in p for p in problems)


def test_allows_em_dash_inside_code_span(tmp_path: Path) -> None:
    text = f"Quote `a {EM_DASH} b; c` here.\n\n```\nfenced {EM_DASH} text; more\n```\n"
    assert _check(tmp_path, text) == []


def test_flags_lens_number_in_prose(tmp_path: Path) -> None:
    text = "See Lens 4 and Phase 1 and Part A and Gate B.\nAlso B0 and B1 and P2-04 appear.\n"
    problems = _check(tmp_path, text)
    details = " ".join(problems)
    for token in ("Lens 4", "Phase 1", "Part A", "Gate B", "B0", "B1", "P2-04"):
        assert token in details
    # Ordinary words that merely start like a label are not flagged.
    assert _check(tmp_path, "Part of the plan. Gate Check passes. A partial bar.\n") == []


def test_allows_lens_number_in_file_name_column(tmp_path: Path) -> None:
    table = (
        "| Name | File |\n|---|---|\n| Cleanliness check | 2026-11-04-lens1-Lens 1-Phase 2 |\n"
        "| Other | Lens 2 |\n"
    )
    assert _check(tmp_path, table) == []
    bad = "| Name | Note |\n|---|---|\n| Cleanliness check | Lens 1 |\n"
    assert any("Lens 1" in p for p in _check(tmp_path, bad))


def test_flags_undefined_abbreviation(tmp_path: Path) -> None:
    problems = _check(tmp_path, "The XYZ rule and NO-GO verdicts. Also TOTALLY ok.\n")
    assert any("XYZ" in p for p in problems)
    assert any("TOTALLY" in p for p in problems)
    assert not any("NO-GO" in p for p in problems)
    # Tokens inside code spans, paths and URLs are not checked.
    clean = "Use `XYZ` and research/ABC/file.md and https://example.com/QRS here.\n"
    assert _check(tmp_path, clean) == []
    # Domain-style names such as Binance.US are names, not abbreviations.
    assert _check(tmp_path, "Trades on Binance.US daily.\n") == []


def test_allows_abbreviation_defined_in_sheet(tmp_path: Path) -> None:
    assert _check(tmp_path, "A DSR gate and the UTC clock in a JSON file.\n") == []
    assert any("DSR" in p for p in _check(tmp_path, "A DSR gate.\n", terms=set()))


def test_flags_number_absent_from_sources(tmp_path: Path) -> None:
    sources = {"s.md": "The Sharpe was 0.59 and the tape had 12,345 rows and 14.19% gain."}
    problems = _check(tmp_path, "Sharpe 0.58, 54,321 rows and 14.20%.\n", sources=sources)
    for token in ("0.58", "54321", "14.20%"):
        assert any(token in p for p in problems), token


def test_allows_number_present_in_sources(tmp_path: Path) -> None:
    sources = {"s.md": "Sharpe 0.59, 12,345 rows, +14.19% gain, ratio 0.8 and 26% less."}
    text = "Sharpe 0.59 with 12345 rows, 14.19% gain, 0.80 ratio and 26% less, 3 tests, 42.\n"
    assert _check(tmp_path, text, sources=sources) == []
    # A bare integer in the source does not vouch for the same integer with a percent sign.
    problems = _check(tmp_path, "About 26% fell.\n", sources={"s.md": "26 tests"})
    assert any("26%" in p for p in problems)


def test_dates_years_and_line_refs_are_exempt_from_number_rule(tmp_path: Path) -> None:
    text = (
        "Dated 2026-11-04, spanning 2021 to 2026, see research/x.md:1234 and `a.py:99.5` "
        "plus (file.md:123) and 12.5 inside `0.99`.\n"
    )
    problems = _check(tmp_path, text)
    assert [p for p in problems if "number" in p] == [problems[0]]
    assert "12.5" in problems[0]


def test_flags_missing_backticked_path(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "real.md").write_text("x")
    text = "See `docs/real.md:10` and `docs/missing.md` today.\n"
    problems = _check(tmp_path, text)
    assert len(problems) == 1
    assert "docs/missing.md" in problems[0] and ":1:" in problems[0]


def test_word_cap(tmp_path: Path) -> None:
    body = "one two three four five.\n"
    table = "| a | b c d |\n|---|---|\n| e f g | h i j |\n"
    text = f"---\ntitle: front matter words here\n---\n{body}\n{table}\n```\ncode words here\n```\n"
    assert _check(tmp_path, text, max_words=5) == []
    problems = _check(tmp_path, text, max_words=4)
    assert len(problems) == 1 and "word" in problems[0] and "5" in problems[0]


def test_load_sheet_terms_reads_first_column(tmp_path: Path) -> None:
    sheet = _write(
        tmp_path,
        "| Term | Meaning | Source |\n|---|---|---|\n"
        "| DSR | Deflated Sharpe ratio, uses OTHERABBR here | x |\n"
        "| pp, pp/yr | points | x |\n"
        "| `A23` | study | x |\n"
        "| S&P | index | x |\n"
        "| MaxDD | drawdown | x |\n"
        "| GO or NO-GO | labels | x |\n"
        "| ST / LT | gains | x |\n",
        "sheet.md",
    )
    terms = load_sheet_terms(sheet)
    assert {"DSR", "pp", "pp/yr", "yr", "A23", "S&P", "MaxDD", "NO-GO", "GO", "ST", "LT"} <= terms
    assert "OTHERABBR" not in terms and "Deflated" not in terms


def test_cli_exit_codes_and_help(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sheet = _write(tmp_path, "| Term | M |\n|---|---|\n| DSR | x |\n", "sheet.md")
    source = _write(tmp_path, "Value 1.25 here.\n", "source.md")
    good = _write(tmp_path, "A DSR of 1.25.\n", "good.md")
    bad = _write(tmp_path, "A DSR of 9.99; oops.\n", "bad.md")
    base = ["--sheet", str(sheet), "--sources", str(source), "--root", str(tmp_path)]
    assert main([str(good), *base]) == 0
    assert capsys.readouterr().out == ""
    assert main([str(bad), *base]) == 1
    out = capsys.readouterr().out
    assert "bad.md:1:" in out and "9.99" in out and "semicolon" in out
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    assert "--sheet" in capsys.readouterr().out
