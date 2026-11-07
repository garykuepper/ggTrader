"""Tests for the reading-edition lint."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from scripts.check_reading_edition import check_file, load_sheet_terms, main

pytestmark = pytest.mark.tooling

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
    **options: Any,
) -> list[str]:
    path = _write(tmp_path, text)
    return check_file(
        path, TERMS if terms is None else terms, sources or {}, max_words, root=tmp_path, **options
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


def test_number_must_not_trace_to_the_checked_file_itself(tmp_path: Path) -> None:
    path = _write(tmp_path, "An invented 7.777 figure.\n")
    own = {str(path): path.read_text(encoding="utf-8")}
    problems = check_file(path, TERMS, own, root=tmp_path)
    assert any("7.777" in p for p in problems)
    other = {**own, "src.md": "The real figure is 7.777."}
    assert check_file(path, TERMS, other, root=tmp_path) == []


@pytest.mark.parametrize(
    "text",
    ["lens 6 says", "phase 2 ran", "Lenses 2 and 6", "Gates A and B", "Lens-6 here", "Parts B"],
)
def test_label_rule_is_case_insensitive_and_covers_plurals(tmp_path: Path, text: str) -> None:
    problems = _check(tmp_path, f"{text}.\n")
    assert any("numbered-label" in p for p in problems)


def test_label_message_is_full_and_actionable(tmp_path: Path) -> None:
    problems = _check(tmp_path, "See Lens 10 now.\n")
    assert len(problems) == 1
    assert "`Lens 10`" in problems[0] and "use the reference-sheet name" in problems[0]
    assert "code span" in problems[0]


@pytest.mark.parametrize("text", ["part a deal", "Part of it", "Gate Check", "A partial bar"])
def test_label_rule_ignores_ordinary_words(tmp_path: Path, text: str) -> None:
    assert _check(tmp_path, f"{text}.\n") == []


def test_plural_abbreviations_are_checked_by_stem(tmp_path: Path) -> None:
    problems = _check(tmp_path, "QQQs and ETFs fell.\n")
    assert any("`QQQ`" in p for p in problems)
    assert any("`ETF`" in p for p in problems)
    assert _check(tmp_path, "Many NO-GOs, As, MaxDD and A23 appear.\n") == []


def test_year_exemption_does_not_swallow_money_or_counts(tmp_path: Path) -> None:
    problems = _check(tmp_path, "It cost $2,000 and $1,950 across 1,994 rows or $2026.\n")
    flagged = " ".join(problems)
    for token in ("2000", "1950", "1994", "2026"):
        assert token in flagged, token
    assert _check(tmp_path, "Between 1994 and 2026 only.\n") == []


def test_semicolon_inside_link_target_is_not_flagged(tmp_path: Path) -> None:
    assert _check(tmp_path, "See [x](http://a/q;x) now.\n") == []
    assert any("semicolon" in p for p in _check(tmp_path, "See [x](http://a/q) now; ok.\n"))


def test_fence_closes_only_on_same_char_and_length(tmp_path: Path) -> None:
    text = "````\n```\nin; fence\n```\n````\nafter; fence\n"
    problems = _check(tmp_path, text)
    assert [p.split(":")[1] for p in problems] == ["6"]


@pytest.mark.parametrize("header", ["Files", "File Name", "file"])
def test_file_column_header_variants_and_escaped_pipes(tmp_path: Path, header: str) -> None:
    table = f"| Name | {header} |\n|---|---|\n| a \\| b | Lens 1 |\n"
    assert _check(tmp_path, table) == []


def test_word_cap_ignores_link_targets_and_html_comments(tmp_path: Path) -> None:
    text = "[word](http://very/long/path/with-many/parts) here.\n<!-- hidden words\nmore -->\n"
    assert _check(tmp_path, text, max_words=2) == []
    assert len(_check(tmp_path, text, max_words=1)) == 1


def test_placeholder_paths_are_skipped(tmp_path: Path) -> None:
    assert _check(tmp_path, "Write `research/<category>/<study>/report.md` first.\n") == []


def test_cli_default_root_is_repo_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sheet = _write(tmp_path, "| Term | M |\n|---|---|\n| DSR | x |\n", "sheet.md")
    doc = _write(tmp_path, "See `scripts/check_reading_edition.py` here.\n", "doc.md")
    monkeypatch.chdir(tmp_path)
    assert main([str(doc), "--sheet", str(sheet)]) == 0


def test_cli_missing_inputs_fail_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = _write(tmp_path, "| Term | M |\n|---|---|\n| DSR | x |\n", "sheet.md")
    doc = _write(tmp_path, "Plain text.\n", "doc.md")
    assert main([str(doc), "--sheet", str(tmp_path / "nope.md")]) == 1
    assert capsys.readouterr().out.startswith("FAIL ")
    assert main([str(doc), "--sheet", str(sheet), "--sources", str(tmp_path / "nope.md")]) == 1
    assert capsys.readouterr().out.startswith("FAIL ")
    assert main([str(tmp_path / "nope.md"), "--sheet", str(sheet)]) == 1
    assert capsys.readouterr().out.startswith("FAIL ")


def test_us_is_ignored_and_hyphenated_tokens_check_each_part(tmp_path: Path) -> None:
    assert _check(tmp_path, "The US market.\n") == []
    assert _check(tmp_path, "The SPY-IEF pair.\n", terms={"SPY", "IEF"}) == []
    problems = _check(tmp_path, "The SPY-IEF pair.\n", terms={"SPY"})
    assert len(problems) == 1 and "`IEF`" in problems[0]


def _words(count: int) -> str:
    """Return `count` plain lowercase words that trip no other rule."""
    return " ".join(["plain"] * count)


def _wrapped(count: int, per_line: int = 12, indent: str = "") -> str:
    """Return `count` words wrapped over several lines."""
    words = ["plain"] * count
    return "\n".join(indent + " ".join(words[i : i + per_line]) for i in range(0, count, per_line))


def test_flags_paragraph_over_limit(tmp_path: Path) -> None:
    assert _check(tmp_path, f"{_words(70)}.\n") == []
    problems = _check(tmp_path, f"{_words(71)}.\n")
    assert len(problems) == 1
    assert ":1:" in problems[0] and "long-paragraph" in problems[0]
    assert "71" in problems[0] and "70" in problems[0]
    # A paragraph wrapped over several lines is one paragraph, reported at its first line.
    wrapped = _check(tmp_path, f"# Title\n\n{_wrapped(71)}\n")
    assert len(wrapped) == 1 and ":3:" in wrapped[0] and "long-paragraph" in wrapped[0]
    # Two short paragraphs split by a blank line never add up.
    assert _check(tmp_path, f"{_words(40)}.\n\n{_words(40)}.\n") == []
    # Code spans and link targets are not words.
    link = "[a](http://very/long/path/with-many/parts) "
    assert _check(tmp_path, f"{_words(69)} {link}\n") == []


def test_allows_long_table_and_list_of_short_items(tmp_path: Path) -> None:
    table = "| a | b |\n|---|---|\n" + "".join(
        f"| {_words(10)} | {_words(10)} |\n" for _ in range(8)
    )
    items = "".join(f"- {_words(5)}\n" for _ in range(30))
    numbered = "".join(f"{n}. {_words(5)}\n" for n in range(1, 21))
    fenced = f"```\n{_words(200)}\n```\n"
    comment = f"<!-- {_words(100)}\n{_words(100)} -->\n"
    front = f"---\ntitle: {_words(100)}\n---\n"
    text = f"{front}# {_words(90)}\n\n{table}\n{items}\n{numbered}\n{fenced}{comment}"
    assert _check(tmp_path, text) == []
    # A fenced block between two text blocks splits them into two paragraphs.
    split = f"{_words(40)}\n```\ncode\n```\n{_words(40)}\n"
    assert _check(tmp_path, split) == []


def test_flags_long_list_item(tmp_path: Path) -> None:
    assert _check(tmp_path, f"- {_words(50)}\n") == []
    problems = _check(tmp_path, f"- {_words(51)}\n")
    assert len(problems) == 1 and "long-paragraph" in problems[0] and "list item" in problems[0]
    assert ":1:" in problems[0]
    # Wrapped continuation lines count towards the item, and the item may be numbered or nested.
    first = "- short item\n"
    wrapped = f"{first}1. {_wrapped(26)}\n   {_wrapped(26)}\n  - {_words(5)}\n"
    problems = _check(tmp_path, wrapped)
    assert len(problems) == 1 and ":2:" in problems[0] and "list item" in problems[0]


def test_flags_long_blockquote_callout(tmp_path: Path) -> None:
    quote = "\n".join(f"> {line}" for line in _wrapped(70).split("\n"))
    assert _check(tmp_path, quote + "\n") == []
    long_quote = "\n".join(f"> {line}" for line in _wrapped(71).split("\n"))
    problems = _check(tmp_path, long_quote + "\n")
    assert len(problems) == 1 and "blockquote" in problems[0] and "71" in problems[0]
    # Two callouts split by a blank line are counted one by one.
    two = f"> {_words(40)}\n\n> {_words(40)}\n"
    assert _check(tmp_path, two) == []
    # A bare `>` line keeps one callout together.
    joined = f"> {_words(40)}\n>\n> {_words(40)}\n"
    assert len(_check(tmp_path, joined)) == 1


def test_flags_image_without_alt_text(tmp_path: Path) -> None:
    (tmp_path / "fig.png").write_bytes(b"x")
    for empty in ("![](fig.png)", "![   ](fig.png)", '![](fig.png "A title")'):
        problems = _check(tmp_path, f"{empty}\n")
        assert len(problems) == 1, empty
        assert ":1:" in problems[0] and "image" in problems[0] and "alt" in problems[0]


def test_flags_missing_image_file(tmp_path: Path) -> None:
    problems = _check(tmp_path, "Intro.\n\n![A chart](figures/missing.png)\n")
    assert len(problems) == 1
    assert ":3:" in problems[0] and "image" in problems[0] and "figures/missing.png" in problems[0]
    # The path resolves against the file's own directory, not the working directory.
    sub = tmp_path / "docs"
    sub.mkdir()
    (tmp_path / "fig.png").write_bytes(b"x")
    nested = sub / "doc.md"
    nested.write_text("![A chart](fig.png)\n", encoding="utf-8")
    assert any("fig.png" in p for p in check_file(nested, TERMS, {}, root=tmp_path))
    nested.write_text("![A chart](../fig.png)\n", encoding="utf-8")
    assert check_file(nested, TERMS, {}, root=tmp_path) == []


def test_flags_remote_image(tmp_path: Path) -> None:
    for url in ("http://example.com/a.png", "https://example.com/a.png"):
        problems = _check(tmp_path, f"![A chart]({url})\n")
        assert len(problems) == 1
        assert "image: remote image" in problems[0]


def test_allows_existing_image_with_alt(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "gap.png").write_bytes(b"x")
    text = (
        "![Bars of the yearly gap](figures/gap.png)\n\n"
        '![With a title](figures/gap.png "Gap")\n\n'
        "A link [not an image](figures/missing.png) and `![](code.png)` are fine.\n"
        "```\n![](fenced.png)\n```\n"
    )
    assert _check(tmp_path, text) == []


def test_no_scan_rules_flag_disables_rules_7_and_8(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = f"{_words(90)}.\n\n- {_words(80)}\n\n![](gone.png)\n"
    # A long paragraph, a long list item, and an image with no alt text and no file.
    assert len(_check(tmp_path, text)) == 4
    assert _check(tmp_path, text, scan_rules=False) == []
    sheet = _write(tmp_path, "| Term | M |\n|---|---|\n| DSR | x |\n", "sheet.md")
    doc = _write(tmp_path, text, "long.md")
    base = [str(doc), "--sheet", str(sheet), "--root", str(tmp_path)]
    assert main(base) == 1
    assert "long-paragraph" in capsys.readouterr().out
    assert main([*base, "--no-scan-rules"]) == 0
    assert capsys.readouterr().out == ""


def test_cli_max_paragraph_words_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sheet = _write(tmp_path, "| Term | M |\n|---|---|\n| DSR | x |\n", "sheet.md")
    doc = _write(tmp_path, f"{_words(30)}.\n", "doc.md")
    base = [str(doc), "--sheet", str(sheet), "--root", str(tmp_path)]
    assert main(base) == 0
    assert main([*base, "--max-paragraph-words", "30"]) == 0
    assert main([*base, "--max-paragraph-words", "29"]) == 1
    out = capsys.readouterr().out
    assert "long-paragraph" in out and "29" in out
    with pytest.raises(SystemExit):
        main(["--help"])
    help_text = capsys.readouterr().out
    assert "--max-paragraph-words" in help_text and "--no-scan-rules" in help_text


def _bullets(count: int, words: int, prefix: str = "> ") -> str:
    return "".join(f"{prefix}- {_words(words)}\n" for _ in range(count))


def test_callout_with_short_bullets_is_allowed(tmp_path: Path) -> None:
    text = f"> **The short version**\n{_bullets(6, 15)}"
    assert _check(tmp_path, text) == []
    # Header prose over the callout limit is still flagged, bullets or not.
    long_header = f"> {_words(71)}\n{_bullets(2, 5)}"
    problems = _check(tmp_path, long_header)
    assert len(problems) == 1 and "blockquote callout" in problems[0] and ":1:" in problems[0]


def test_long_quoted_list_item_is_flagged(tmp_path: Path) -> None:
    assert _check(tmp_path, f"> head\n> - {_words(50)}\n") == []
    problems = _check(tmp_path, f"> head\n> - {_words(51)}\n")
    assert len(problems) == 1 and "list item" in problems[0] and ":2:" in problems[0]
    # A wrapped continuation line belongs to the item.
    wrapped = f"> head\n> - {_words(30)}\n>   {_words(30)}\n"
    problems = _check(tmp_path, wrapped)
    assert len(problems) == 1 and "list item" in problems[0] and ":2:" in problems[0]
    # A bare `>` ends the item but keeps the callout; later prose is not part of the item.
    split = f"> - {_words(30)}\n>\n> {_words(30)}\n"
    assert _check(tmp_path, split) == []


def test_code_span_counts_as_one_word_in_scan_rules(tmp_path: Path) -> None:
    spans = " ".join(f"`item{i}`" for i in range(71))
    problems = _check(tmp_path, f"{spans}\n")
    assert len(problems) == 1 and "long-paragraph" in problems[0] and "71" in problems[0]
    assert _check(tmp_path, f"`{_words(200)}`\n") == []
    # The word cap keeps its own counting, where code spans are free.
    assert _check(tmp_path, f"{spans}\n", max_words=1, scan_rules=False) == []


def test_task_list_marker_and_year_start_are_not_words(tmp_path: Path) -> None:
    assert _check(tmp_path, f"- [x] {_words(50)}\n- [ ] {_words(50)}\n") == []
    # A wrapped line that begins with a year continues the paragraph.
    text = f"{_words(40)}\n2026. {_words(40)}\n"
    problems = _check(tmp_path, text)
    assert len(problems) == 1 and "paragraph has" in problems[0]


def test_flags_reference_style_and_html_images(tmp_path: Path) -> None:
    for text in ("![Chart][fig]\n", '<img src="a.png" alt="x">\n'):
        problems = _check(tmp_path, text)
        assert len(problems) == 1 and "image: use an inline ![alt](path) image" in problems[0]


def test_data_uri_image_and_parentheses_in_names(tmp_path: Path) -> None:
    problems = _check(tmp_path, "![A chart](data:image/png;base64,AAAA)\n")
    assert any("image: data URI image" in p for p in problems)
    assert not any("not found" in p for p in problems)
    (tmp_path / "fig(1).png").write_bytes(b"x")
    assert _check(tmp_path, "![A chart](fig(1).png)\n") == []


@pytest.mark.parametrize("value", ["0", "-5", "x"])
def test_cli_max_paragraph_words_must_be_positive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], value: str
) -> None:
    sheet = _write(tmp_path, "| Term | M |\n|---|---|\n| DSR | x |\n", "sheet.md")
    doc = _write(tmp_path, "Plain text.\n", "doc.md")
    with pytest.raises(SystemExit) as info:
        main([str(doc), "--sheet", str(sheet), "--max-paragraph-words", value])
    assert info.value.code == 2
    assert "--max-paragraph-words" in capsys.readouterr().err


def test_problems_are_sorted_by_line_number(tmp_path: Path) -> None:
    text = f"{_words(80)}\n\n![](gone.png)\n\nBad; semicolon.\n"
    problems = _check(tmp_path, text, max_words=10)
    numbers = [int(re.search(r"\.md:(\d+): ", p).group(1)) for p in problems]  # type: ignore[union-attr]
    assert len(problems) >= 4
    assert numbers == sorted(numbers)


def test_flags_table_cell_over_limit(tmp_path: Path) -> None:
    ok = f"| a | b |\n|---|---|\n| {_words(40)} | x |\n"
    assert _check(tmp_path, ok) == []
    long_row = f"| a | b |\n|---|---|\n| x | {_words(41)} |\n"
    problems = _check(tmp_path, long_row)
    assert len(problems) == 1 and ":3:" in problems[0]
    assert "table-cell" in problems[0] and "41" in problems[0] and "40" in problems[0]
    header = f"| {_words(41)} | b |\n|---|---|\n| x | y |\n"
    assert any(":1:" in p and "table-cell" in p for p in _check(tmp_path, header))
    assert _check(tmp_path, long_row, scan_rules=False) == []


def test_allows_table_cells_under_limit(tmp_path: Path) -> None:
    cells = " | ".join(_words(39) for _ in range(5))
    table = f"| a | b | c | d | e |\n|---|---|---|---|---|\n| {cells} |\n"
    assert _check(tmp_path, table) == []
    # A code span counts as one word, and an escaped pipe does not split a cell.
    span = f"| a | b |\n|---|---|\n| `{_words(100)}` {_words(30)} a \\| b | x |\n"
    assert _check(tmp_path, span) == []
