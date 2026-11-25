"""The five engine files copied from the first test: only the named import lines differ."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ggTrader.utils.study import study_dir

pytestmark = pytest.mark.study

STUDY = study_dir(__file__)
CODE = STUDY / "code"
FIRST_TEST_CODE = STUDY.parent / "dead-stock-first-test" / "code"
RECORD = json.loads((CODE / "frozen_copies.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_record_lists_exactly_the_five_engine_files() -> None:
    assert sorted(RECORD) == [
        "bars_reader.py",
        "dead_stock_first_test.py",
        "ledger_run.py",
        "replay.py",
        "universe_frames.py",
    ]
    assert {entry["copy"] for entry in RECORD.values()} == {f"fp_{name}" for name in RECORD}


@pytest.mark.parametrize("name", sorted(RECORD))
def test_copies_match_recorded_hashes(name: str) -> None:
    assert _sha256(CODE / RECORD[name]["copy"]) == RECORD[name]["copy_sha256"]


@pytest.mark.parametrize("name", sorted(RECORD))
def test_copy_is_the_original_with_only_the_named_substitutions(name: str) -> None:
    original = FIRST_TEST_CODE / name
    if not original.is_file():
        pytest.fail(
            "the first test's folder is missing: the copies cannot be checked against the originals"
        )
    entry = RECORD[name]
    assert _sha256(original) == entry["original_sha256"]
    text = original.read_text(encoding="utf-8")
    for old, new in entry["substitutions"]:
        assert text.count(old) == 1
        text = text.replace(old, new)
    assert (CODE / entry["copy"]).read_text(encoding="utf-8") == text


def test_every_substitution_is_a_sibling_import_line() -> None:
    for entry in RECORD.values():
        for old, new in entry["substitutions"]:
            assert old.startswith("from ") and new == old.replace("from ", "from fp_", 1)
