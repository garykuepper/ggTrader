"""Tests for the batched close reader and the release spans reader (no database)."""

from __future__ import annotations

from types import SimpleNamespace

import bars_reader
import pandas as pd
import pytest

pytestmark = pytest.mark.study

DAYS = pd.DatetimeIndex(pd.bdate_range("2020-01-02", "2020-01-08"), tz="UTC")


def fake_bars(symbols: list[str], offset: float) -> SimpleNamespace:
    """A Bars-like object with every field frame, ended-span rows and a release record."""
    frame = pd.DataFrame(
        {s: offset + i + pd.RangeIndex(len(DAYS)).to_numpy() for i, s in enumerate(symbols)},
        index=DAYS,
    )
    spans = pd.DataFrame(
        {"symbol": symbols[:1], "last_date": [pd.Timestamp("2020-01-06")], "has_successor": [False]}
    )
    return SimpleNamespace(
        close=frame,
        open=frame * 2,
        high=frame * 3,
        low=frame * 4,
        volume=frame * 5,
        spans=spans,
        provenance={"data_release": {"release_id": 1}, "bars_sha256": "x"},
    )


def test_read_close_batched_keeps_only_close_and_concatenates(monkeypatch):
    calls = []

    def fake_load_bars(symbols, start, end, *, source, release_id):
        calls.append((list(symbols), start, end, source, release_id))
        return fake_bars(sorted(symbols), offset=10.0 * len(calls))

    monkeypatch.setattr(bars_reader, "load_bars", fake_load_bars)
    provenance: dict = {}
    close, spans = bars_reader.read_close_batched(
        ["C", "A", "B"],
        "2020-01-02",
        "2020-01-08",
        release_id=1,
        batch=2,
        provenance_out=provenance,
    )
    assert [c[0] for c in calls] == [["A", "B"], ["C"]]
    assert all(c[3:] == ("combined", 1) for c in calls)
    assert list(close.columns) == ["A", "B", "C"]
    assert len(close) == len(DAYS)
    assert list(spans["symbol"]) == ["A", "C"]
    assert provenance["data_release"] == {"release_id": 1}
    assert provenance["n_batches"] == 2


def test_read_close_batched_empty_symbols_returns_empty_frames():
    close, spans = bars_reader.read_close_batched([], "2020-01-02", "2020-01-08", release_id=1)
    assert close.empty and spans.empty


def test_read_close_batched_rejects_bad_batch():
    with pytest.raises(ValueError):
        bars_reader.read_close_batched(["A"], "2020-01-02", "2020-01-08", release_id=1, batch=0)


def test_batches_cover_all_columns_once():
    items = [f"S{i}" for i in range(23)]
    for size in (1, 5, 23, 100):
        parts = bars_reader.symbol_batches(items, size)
        flat = [s for part in parts for s in part]
        assert flat == sorted(set(items))
        assert all(len(part) <= size for part in parts)


def test_read_release_spans_selects_by_release(monkeypatch):
    seen = {}

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params):
            seen["sql"], seen["params"] = sql, params

        def fetchall(self):
            return [("A", 1, 1, "2000-01-03", "2001-02-03", "placeholder", None)]

    conn = SimpleNamespace(cursor=lambda: Cursor())
    spans = bars_reader.read_release_spans(conn, 7)
    assert seen["params"] == (7,)
    assert "market.stock_spans" in seen["sql"]
    assert list(spans.columns) == bars_reader.SPAN_COLUMNS
    assert spans.loc[0, "possible_double"] == False  # noqa: E712 (None filled to False)
    assert pd.api.types.is_datetime64_any_dtype(spans["first_date"])
