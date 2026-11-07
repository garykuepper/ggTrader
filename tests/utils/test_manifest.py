"""Hash records and frozen-input verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ggTrader.utils.manifest import (
    FrozenInputError,
    canonical_hash,
    file_record,
    sha256_file,
    verify_frozen,
)


def test_file_record_matches_hashlib(tmp_path: Path) -> None:
    f = tmp_path / "a.csv"
    f.write_bytes(b"date,close\n2024-01-02,1\n")
    record = file_record(f)
    assert record == {
        "path": str(f),
        "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        "bytes": f.stat().st_size,
    }
    assert sha256_file(f) == record["sha256"]


def test_canonical_hash_ignores_key_order_and_sees_values() -> None:
    assert canonical_hash({"a": 1, "b": [1, 2]}) == canonical_hash({"b": [1, 2], "a": 1})
    assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})


def test_canonical_hash_is_the_sha256_of_compact_sorted_json() -> None:
    want = hashlib.sha256(b'{"a":1,"b":[1,2]}').hexdigest()
    assert canonical_hash({"b": [1, 2], "a": 1}) == want


def test_verify_frozen_passes_when_every_hash_matches(tmp_path: Path) -> None:
    (tmp_path / "x.csv").write_bytes(b"1")
    verify_frozen({"x.csv": hashlib.sha256(b"1").hexdigest()}, tmp_path)


def test_verify_frozen_names_every_missing_and_changed_file(tmp_path: Path) -> None:
    (tmp_path / "x.csv").write_bytes(b"1")
    expected = {"x.csv": hashlib.sha256(b"2").hexdigest(), "gone.csv": "0" * 64}
    with pytest.raises(FrozenInputError) as err:
        verify_frozen(expected, tmp_path)
    assert "x.csv" in str(err.value) and "gone.csv" in str(err.value)
