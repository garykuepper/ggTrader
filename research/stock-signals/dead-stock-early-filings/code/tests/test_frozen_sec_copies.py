"""The SEC client copied from dead-stock-data-integrity must remain byte-identical."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ggTrader.utils.study import study_dir

pytestmark = pytest.mark.study

STUDY = study_dir(__file__)
CODE = STUDY / "code"
DATA_INTEGRITY_CODE = STUDY.parent / "dead-stock-data-integrity" / "code"
RECORD = json.loads((CODE / "frozen_sec_copies.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_record_lists_only_sec_client_with_no_substitutions() -> None:
    assert list(RECORD) == ["sec_client.py"]
    entry = RECORD["sec_client.py"]
    assert entry["copy"] == "ef_sec_client.py"
    assert entry["substitutions"] == []
    assert entry["original_sha256"] == entry["copy_sha256"]


@pytest.mark.parametrize("name", sorted(RECORD))
def test_sec_copy_matches_recorded_hashes_and_original_bytes(name: str) -> None:
    entry = RECORD[name]
    copy_path = CODE / entry["copy"]
    original_path = DATA_INTEGRITY_CODE / name
    if not original_path.is_file():
        pytest.fail(
            "the dead-stock-data-integrity folder is missing: "
            "the SEC client copy cannot be checked against the original"
        )
    assert _sha256(copy_path) == entry["copy_sha256"]
    assert _sha256(original_path) == entry["original_sha256"]
    assert copy_path.read_bytes() == original_path.read_bytes()
