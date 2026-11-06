"""Tests for the root-cause chart script: traced numbers, specs, rendering, CLI."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts import root_cause_figures as rcf
from scripts.root_cause_figures import FIGURES, OPTIONS, REPO_ROOT, main, render

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
EXPECTED_NAMES = {"gap-vs-measured", "verdict-trust", "paper-shortfall", "options-at-a-glance"}


def _source_line(source: str) -> str:
    path, _, number = source.rpartition(":")
    assert path and number.isdigit(), f"bad source {source!r}"
    lines = (REPO_ROOT / path).read_text(encoding="utf-8").splitlines()
    assert 1 <= int(number) <= len(lines), f"{source} is out of range"
    return lines[int(number) - 1]


def _forms(value: float) -> list[str]:
    """Text forms of a value once signs and a leading plus are normalised away."""
    magnitude = abs(value)
    text = f"{magnitude:.2f}"
    forms = [text]
    if text.endswith("0"):  # 3.50 may be written 3.5
        forms.append(text[:-1])
    if magnitude == int(magnitude):
        forms.append(str(int(magnitude)))
    return forms


def _appears(value: float, line: str) -> bool:
    for form in _forms(value):
        if re.search(rf"(?<![\d.]){re.escape(form)}(?![\d]|\.\d)", line):
            return True
    return False


def test_every_point_value_appears_at_its_source_line() -> None:
    checked = 0
    for spec in FIGURES.values():
        for point in spec.points:
            line = _source_line(point.source)
            assert _appears(point.value, line), (
                f"{spec.name}: {point.label} = {point.value} not found at {point.source}: {line!r}"
            )
            checked += 1
    assert checked >= 15


def test_figure_specs_have_alt_caption_and_unique_names() -> None:
    assert set(FIGURES) == EXPECTED_NAMES
    for key, spec in FIGURES.items():
        assert spec.name == key
        assert len(spec.title.split()) >= 4
        assert len(spec.alt.split()) >= 8
        assert len(spec.caption.split()) >= 8
        assert "Source" in spec.caption
    assert len({spec.title for spec in FIGURES.values()}) == len(FIGURES)


def test_point_sources_are_repo_relative_with_dates_kept() -> None:
    for spec in FIGURES.values():
        for point in spec.points:
            path = point.source.rpartition(":")[0]
            assert path.startswith("research/audits/root-cause-2026-11-04/")
            assert re.search(r"/2026-11-0\d-", path)
            assert (REPO_ROOT / path).is_file()


def test_option_texts_come_from_cited_review_lines() -> None:
    words = lambda text: {w for w in re.findall(r"[a-z0-9.+\-]+", text.lower()) if len(w) > 3}  # noqa: E731
    assert [block.option for block in OPTIONS] == [1, 2, 3]
    for block in OPTIONS:
        for text, source in (
            (block.question, block.question_source),
            (block.step, block.step_source),
            (block.stop, block.stop_source),
        ):
            assert 6 <= len(text.split()) <= 12, text
            line = _source_line(source).lower()
            missing = {w for w in words(text) if w not in line}
            assert not missing, f"{text!r} has words not at {source}: {missing}"
        assert block.title.lower().split()[0] in _source_line(block.title_source).lower()


def test_gap_figure_has_required_values() -> None:
    values = {p.label: p.value for p in FIGURES["gap-vs-measured"].points}
    assert 9.57 in values.values() and 3.54 in values.values() and 0.77 in values.values()
    assert 4.15 in values.values() and 4.37 in values.values()
    assert -7.07 in values.values() and 16.05 in values.values()
    assert -0.62 in values.values() and 0.83 in values.values()


def test_render_writes_png_files(tmp_path: Path) -> None:
    for name in FIGURES:
        out = render(name, tmp_path)
        assert out == tmp_path / f"{name}.png"
        data = out.read_bytes()
        assert data.startswith(PNG_MAGIC)
        assert 10_000 < len(data) <= 250_000, f"{name}: {len(data)} bytes"
        width, height = int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
        assert width <= 1600 and height <= 900


def test_render_rejects_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        render("nope", tmp_path)


def test_cli_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    assert "--only" in capsys.readouterr().out


def test_cli_only_writes_one_file(tmp_path: Path) -> None:
    assert main(["--out", str(tmp_path), "--only", "verdict-trust"]) == 0
    assert [p.name for p in tmp_path.iterdir()] == ["verdict-trust.png"]


def test_default_output_directory_is_the_figures_folder() -> None:
    assert rcf.DEFAULT_OUT_DIR == "research/audits/root-cause-2026-11-04/figures"
