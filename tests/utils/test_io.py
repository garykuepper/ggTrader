"""Atomic writes and JSON-safe output."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.utils import io, results
from ggTrader.utils.json_safe import json_safe


def test_atomic_write_bytes_creates_parents_and_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "x.bin"
    assert io.atomic_write_bytes(target, b"\x00\x01") == target
    assert target.read_bytes() == b"\x00\x01"
    assert [p.name for p in target.parent.iterdir()] == ["x.bin"]


def test_atomic_write_replaces_an_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "x.txt"
    io.atomic_write_text(target, "old")
    io.atomic_write_text(target, "new")
    assert target.read_text() == "new"


def test_a_failed_write_keeps_the_old_file_and_removes_the_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "x.txt"
    io.atomic_write_text(target, "old")

    def boom(*_args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(io.os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        io.atomic_write_text(target, "new")
    assert target.read_text() == "old"
    assert [p.name for p in tmp_path.iterdir()] == ["x.txt"]


def test_json_safe_converts_numpy_pandas_and_non_finite_values() -> None:
    cleaned = io.json_safe(
        {
            "nan": math.nan,
            "inf": math.inf,
            "np": np.float64(1.5),
            "ts": pd.Timestamp("2024-01-02"),
            "nested": [(np.int64(3), math.nan)],
            1: "int key",
        }
    )
    assert cleaned == {
        "nan": None,
        "inf": None,
        "np": 1.5,
        "ts": "2024-01-02T00:00:00",
        "nested": [[3, None]],
        "1": "int key",
    }
    json.dumps(cleaned, allow_nan=False)


def test_results_keeps_clean_as_an_alias_of_json_safe() -> None:
    assert results._clean is io.json_safe is json_safe


def test_write_json_sanitizes_by_default(tmp_path: Path) -> None:
    out = io.write_json(tmp_path / "r.json", {"x": math.nan, "y": np.float64(2.0)})
    assert json.loads(out.read_text()) == {"x": None, "y": 2.0}
    assert out.read_text().endswith("}\n")


def test_write_json_strict_refuses_nan_and_leaves_no_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        io.write_json(tmp_path / "r.json", {"x": math.nan}, strict=True)
    assert list(tmp_path.iterdir()) == []


def test_write_json_strict_writes_clean_payloads_unchanged(tmp_path: Path) -> None:
    out = io.write_json(tmp_path / "r.json", {"a": [1, 2.5, "z"]}, strict=True)
    assert json.loads(out.read_text()) == {"a": [1, 2.5, "z"]}
