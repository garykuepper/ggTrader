"""Snapshot reader: hash verification against MANIFEST.json, refusal on mismatch."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.data import snapshot as snap

REAL_DIR = Path(snap.DEFAULT_SNAPSHOT_DIR)
REAL_PARQUET = REAL_DIR / "ohlcv_20261106.parquet"
needs_real = pytest.mark.skipif(
    not REAL_PARQUET.exists(),
    reason="frozen tape data/snapshots/ohlcv_20261106.parquet is absent (git-ignored; rebuild "
    "with scripts/make_tape_snapshot.py)",
)


def _long_frame() -> pd.DataFrame:
    """Two symbols over three weekdays, in the stored long layout."""
    stamps = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    rows = []
    for sym, base in (("AAA", 10.0), ("BBB", 20.0)):
        for i, ts in enumerate(stamps):
            px = base + i
            rows.append((ts, sym, px, px + 1, px - 1, px + 0.5, 1000.0 + i))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", *snap.FIELDS])


def _write_fixture(tmp_path: Path, flat: pd.DataFrame | None = None, **overrides) -> Path:
    """Write a tiny snapshot plus manifest in ``tmp_path`` and return the parquet path."""
    flat = _long_frame() if flat is None else flat
    path = tmp_path / "ohlcv_test.parquet"
    flat.to_parquet(path, index=False)
    entry = {
        "path": "data/snapshots/ohlcv_test.parquet",
        "sha256": snap.sha256_file(path),
        "content_sha256": snap.content_sha256(flat),
        "rows": len(flat),
        "symbols": flat["symbol"].nunique(),
        "date_min": str(flat["timestamp"].min().date()),
        "date_max": str(flat["timestamp"].max().date()),
    }
    entry.update(overrides)
    (tmp_path / snap.MANIFEST_NAME).write_text(json.dumps({"snapshots": [entry]}))
    return path


def test_load_verified_roundtrip(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    flat, info = snap.load_snapshot_long(path)
    assert len(flat) == 6
    assert info.sha256 == snap.sha256_file(path)
    assert info.rows == 6
    assert flat["timestamp"].dt.tz is None  # stored stamps are naive UTC


def test_hash_mismatch_is_refused(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    # Overwrite the file with different (still valid) content: the manifest hash no longer fits.
    other = _long_frame()
    other.loc[0, "close"] = 999.0
    other.to_parquet(path, index=False)
    with pytest.raises(snap.SnapshotIntegrityError, match="sha256"):
        snap.load_snapshot_long(path)


def test_row_count_mismatch_is_refused(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, rows=7)
    with pytest.raises(snap.SnapshotIntegrityError, match="rows"):
        snap.load_snapshot_long(path)


def test_date_range_mismatch_is_refused(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path, date_max="2024-01-05")
    with pytest.raises(snap.SnapshotIntegrityError, match="date"):
        snap.load_snapshot_long(path)


def test_missing_manifest_entry_is_refused(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    (tmp_path / snap.MANIFEST_NAME).write_text(json.dumps({"snapshots": []}))
    with pytest.raises(snap.SnapshotIntegrityError, match="no manifest entry"):
        snap.load_snapshot_long(path)


def test_missing_manifest_file_is_refused(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    (tmp_path / snap.MANIFEST_NAME).unlink()
    with pytest.raises(FileNotFoundError):
        snap.load_snapshot_long(path)


def test_symbol_filter_happens_after_verification(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    flat, info = snap.load_snapshot_long(path, symbols=["BBB"])
    assert set(flat["symbol"]) == {"BBB"}
    assert info.rows == 6  # the manifest figure describes the whole file, not the filter


def test_deep_check_detects_content_change_with_same_bytes_hash(tmp_path: Path) -> None:
    flat = _long_frame()
    path = _write_fixture(tmp_path, flat, content_sha256="0" * 64)
    snap.load_snapshot_long(path)  # shallow check passes
    with pytest.raises(snap.SnapshotIntegrityError, match="content_sha256"):
        snap.load_snapshot_long(path, deep=True)


def test_default_path_uses_last_manifest_entry(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    assert snap.default_snapshot_path(tmp_path).name == "ohlcv_test.parquet"


def test_default_path_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_fixture(tmp_path)
    monkeypatch.setenv("GGT_SNAPSHOT", str(path))
    assert snap.default_snapshot_path() == path


@needs_real
def test_real_snapshot_verifies_against_manifest() -> None:
    flat, info = snap.load_snapshot_long(REAL_PARQUET, deep=True)
    manifest = json.loads((REAL_DIR / snap.MANIFEST_NAME).read_text())["snapshots"][0]
    assert info.sha256 == manifest["sha256"]
    assert info.content_sha256 == manifest["content_sha256"]
    assert len(flat) == manifest["rows"] == 1_110_874
    assert flat["symbol"].nunique() == manifest["symbols"]


@needs_real
def test_real_snapshot_corrupted_copy_is_refused(tmp_path: Path) -> None:
    """A copy with one flipped byte (and the real manifest beside it) must be refused."""
    copy = tmp_path / REAL_PARQUET.name
    shutil.copyfile(REAL_PARQUET, copy)
    shutil.copyfile(REAL_DIR / snap.MANIFEST_NAME, tmp_path / snap.MANIFEST_NAME)
    raw = bytearray(copy.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    copy.write_bytes(bytes(raw))
    with pytest.raises(snap.SnapshotIntegrityError, match="sha256"):
        snap.load_snapshot_long(copy)


def test_content_sha256_is_order_independent() -> None:
    flat = _long_frame()
    shuffled = flat.sample(frac=1.0, random_state=1).reset_index(drop=True)
    assert snap.content_sha256(flat) == snap.content_sha256(shuffled)
    changed = flat.copy()
    changed.loc[0, "volume"] = np.float64(1.0)
    assert snap.content_sha256(flat) != snap.content_sha256(changed)
