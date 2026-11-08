"""Tests for the root-cause chart script: traced numbers, specs, rendering, CLI."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from ggTrader.utils.study import load_driver, study_dir

pytestmark = pytest.mark.study

CODE = study_dir(__file__) / "code"
rcf = load_driver(CODE / "root_cause_figures.py")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
EXPECTED_NAMES = {"gap-vs-measured", "verdict-trust", "paper-shortfall", "options-at-a-glance"}


def _source_line(source: str) -> str:
    path, _, number = source.rpartition(":")
    assert path and number.isdigit(), f"bad source {source!r}"
    lines = (rcf.REPO_ROOT / path).read_text(encoding="utf-8").splitlines()
    assert 1 <= int(number) <= len(lines), f"{source} is out of range"
    return lines[int(number) - 1]


NUMBER = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)(?!\d|\.\d)")
YEAR_SPAN = re.compile(r"\d{4}-\d{4}")


def _appears(value: float, line: str) -> bool:
    """True when `line` states `value` with its sign (a zero matches either sign)."""
    for match in NUMBER.finditer(line):
        if float(match.group(1)) != abs(value):
            continue
        negative = line[: match.start()].endswith(("-", "\u2212"))
        if value == 0 or negative == (value < 0):
            return True
    return False


def _point_problems(point: rcf.Point) -> list[str]:
    """Why a point is not traceable to its cited line, or an empty list."""
    line = _source_line(point.source)
    matches = [
        m
        for m in NUMBER.finditer(line)
        if float(m.group(1)) == abs(point.value)
        and (point.value == 0 or line[: m.start()].endswith(("-", "\u2212")) == (point.value < 0))
    ]
    if not matches:
        return [f"value {point.value} not stated with its sign at {point.source}"]
    problems = []
    for span in YEAR_SPAN.findall(point.label):
        if span not in line:
            problems.append(f"window {span} in the label is not on {point.source}")
            continue
        # The window that follows the number must be the labelled one, so a number cannot be
        # paired with another window on the same line.
        following = [YEAR_SPAN.search(line, m.end()) for m in matches]
        if span not in {f.group(0) for f in following if f}:
            problems.append(f"{point.value} is not followed by {span} at {point.source}")
    return problems


def test_every_point_value_appears_at_its_source_line() -> None:
    checked = 0
    for spec in rcf.FIGURES.values():
        for point in spec.points:
            problems = _point_problems(point)
            assert not problems, f"{spec.name}: {point.label}: {problems} {_line_of(point)!r}"
            checked += 1
    assert checked == 23


def _line_of(point: rcf.Point) -> str:
    return _source_line(point.source)


def test_every_year_span_in_a_label_appears_on_its_source_line() -> None:
    spans = 0
    for spec in rcf.FIGURES.values():
        for point in spec.points:
            for span in YEAR_SPAN.findall(point.label):
                assert span in _source_line(point.source), (point.label, point.source)
                spans += 1
    assert spans >= 6


def _point(label: str) -> rcf.Point:
    return next(p for p in rcf.FIGURES["gap-vs-measured"].points if p.label == label)


def _mutated(point: rcf.Point, **changes: object) -> rcf.Point:
    return replace(point, **changes)


def test_mutation_sign_flip_on_interval_low_is_caught() -> None:
    point = _point("momentum interval low")
    assert not _point_problems(point)
    assert _point_problems(_mutated(point, value=7.07))


def test_mutation_sign_flip_on_mnst_is_caught() -> None:
    point = next(p for p in rcf.FIGURES["paper-shortfall"].points if p.value == -0.85)
    assert not _point_problems(point)
    assert _point_problems(_mutated(point, value=0.85))


def test_mutation_other_number_on_the_same_line_is_caught_by_the_window() -> None:
    # CEILING:38 states 2.34 and 0.77 together, but its 0.77 window (1927-2026) is on the next line.
    point = _point("gap 2x needs, 1927-2026")
    assert not _point_problems(point)
    on_shared_line = _mutated(point, value=2.34, source=f"{rcf.CEILING}:38")
    assert any("1927-2026" in problem for problem in _point_problems(on_shared_line))
    wrong_value = _mutated(point, value=2.34)
    assert _point_problems(wrong_value)


def test_mutation_number_paired_with_another_window_on_its_line_is_caught() -> None:
    point = _point("gap 2x needs, 2006-2026")
    assert not _point_problems(point)
    # SYNTHESIS:67 holds 9.57 (2021-2026) and 3.54 (2006-2026): swapping the values must fail.
    assert _point_problems(_mutated(point, value=9.57))
    other = _point("gap 2x needs, 2021-2026")
    assert not _point_problems(other)
    assert _point_problems(_mutated(other, label="gap 2x needs, 2006-2026"))


def test_mutation_wrong_window_is_caught() -> None:
    point = _point("gap 2x needs, 2006-2026")
    assert _point_problems(_mutated(point, label="gap 2x needs, 1994-2026"))


def test_figure_specs_have_alt_caption_and_unique_names() -> None:
    assert set(rcf.FIGURES) == EXPECTED_NAMES
    for key, spec in rcf.FIGURES.items():
        assert spec.name == key
        assert len(spec.title.split()) >= 4
        assert len(spec.alt.split()) >= 8
        assert len(spec.caption.split()) >= 8
        assert "Source" in spec.caption
    assert len({spec.title for spec in rcf.FIGURES.values()}) == len(rcf.FIGURES)


def test_point_sources_are_repo_relative_with_dates_kept() -> None:
    for spec in rcf.FIGURES.values():
        for point in spec.points:
            path = point.source.rpartition(":")[0]
            assert path.startswith("research/audits/root-cause-2026-11-04/")
            assert re.search(r"/2026-11-0\d-", path)
            assert (rcf.REPO_ROOT / path).is_file()


def test_option_texts_come_from_cited_review_lines() -> None:
    words = lambda text: {w for w in re.findall(r"[a-z0-9.+\-]+", text.lower()) if len(w) > 3}  # noqa: E731
    assert [block.option for block in rcf.OPTIONS] == [1, 2, 3]
    for block in rcf.OPTIONS:
        for text, source in (
            (block.question, block.question_source),
            (block.step, block.step_source),
            (block.stop, block.stop_source),
        ):
            assert 6 <= len(text.split()) <= 14, text
            line = _source_line(source).lower()
            missing = {w for w in words(text) if w not in line}
            assert not missing, f"{text!r} has words not at {source}: {missing}"
        assert block.title.lower().split()[0] in _source_line(block.title_source).lower()


def test_gap_figure_has_required_values() -> None:
    values = {p.label: p.value for p in rcf.FIGURES["gap-vs-measured"].points}
    assert 9.57 in values.values() and 3.54 in values.values() and 0.77 in values.values()
    assert 4.15 in values.values() and 4.37 in values.values()
    assert -7.07 in values.values() and 16.05 in values.values()
    assert -0.62 in values.values() and 0.83 in values.values()
    assert {9.54, -84.8, -55.3, 1.64} <= set(values.values())


def test_render_writes_png_files(tmp_path: Path) -> None:
    for name in rcf.FIGURES:
        out = rcf.render(name, tmp_path)
        assert out == tmp_path / f"{name}.png"
        data = out.read_bytes()
        assert data.startswith(PNG_MAGIC)
        assert 10_000 < len(data) <= 250_000, f"{name}: {len(data)} bytes"
        width, height = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        assert width <= 1600 and height <= 900


def test_render_rejects_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        rcf.render("nope", tmp_path)


def test_cli_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        rcf.main(["--help"])
    assert info.value.code == 0
    assert "--only" in capsys.readouterr().out


def test_cli_only_writes_one_file(tmp_path: Path) -> None:
    assert rcf.main(["--out", str(tmp_path), "--only", "verdict-trust"]) == 0
    assert [p.name for p in tmp_path.iterdir()] == ["verdict-trust.png"]


def test_default_output_directory_is_the_figures_folder() -> None:
    assert rcf.DEFAULT_OUT_DIR == "research/audits/root-cause-2026-11-04/figures"
