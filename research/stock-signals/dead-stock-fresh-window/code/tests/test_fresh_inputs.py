"""Tests for fresh_inputs: registered windows, input reading and drift helpers."""

from __future__ import annotations

from typing import Any

import fp_bars_reader as bars_reader
import fp_dead_stock_first_test as study
import fp_universe_frames as universe_frames
import fresh_inputs
import pandas as pd
import pytest

import ggTrader.data.bars as bars_module

pytestmark = pytest.mark.study


class _StubCursor:
    """Minimal read-only cursor double for audit queries."""

    def __init__(self, conn: _StubConn) -> None:
        self._conn = conn

    def __enter__(self) -> _StubCursor:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple[Any, ...]) -> None:
        self._conn.queries.append((sql, params))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._conn.rows


class _StubConn:
    """Minimal read-only connection double that records close calls and queries."""

    def __init__(self, rows: list[tuple[Any, ...]] | None = None) -> None:
        self.rows = rows or []
        self.queries: list[tuple[str, tuple[Any, ...]]] = []
        self.closed = False

    def cursor(self) -> _StubCursor:
        return _StubCursor(self)

    def close(self) -> None:
        self.closed = True


def test_windows_are_registered_values() -> None:
    assert fresh_inputs.WINDOWS == {
        "anchor": ("2020-01-02", "2024-12-31"),
        "fresh": ("2000-01-03", "2004-12-31"),
    }
    assert fresh_inputs.SPY_FIRST_DAY == "2000-01-03"


def test_fresh_window_not_before_spy_first_day() -> None:
    assert fresh_inputs.WINDOWS["fresh"][0] >= fresh_inputs.SPY_FIRST_DAY


def test_unknown_window_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_connect() -> Any:
        raise AssertionError("connect_readonly must not be called for an unknown window")

    monkeypatch.setattr(bars_module, "connect_readonly", fail_connect)
    with pytest.raises(ValueError, match="dev"):
        fresh_inputs.read_inputs("dev", 1, 1)


def test_spy_missing_before_start_is_refused() -> None:
    stock_days = pd.bdate_range("2000-01-03", "2004-12-31")
    spy_days = pd.bdate_range("2000-01-12", "2004-12-31")
    close = pd.DataFrame({"AAA": 25.0}, index=stock_days)
    spy_close = pd.Series(100.0, index=spy_days, name="SPY")
    with pytest.raises(ValueError, match="starts late"):
        study.validate_window_inputs(close, spy_close, "2000-01-03", "2004-12-31")


def test_read_inputs_returns_expected_keys_and_uses_universe_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conns: list[_StubConn] = []

    def fake_connect() -> _StubConn:
        conn = _StubConn([(9, pd.Timestamp("2004-06-01"))])
        conns.append(conn)
        return conn

    release = pd.DataFrame(
        {
            "symbol": ["LINK", "PERMNO-9-2", "OTHER"],
            "permno": [9, 9, 4],
            "spell": [1, 2, 1],
            "kind": ["linked_yahoo", "spell", "placeholder"],
            "first_date": pd.to_datetime(["1998-01-02", "2003-01-02", "1998-01-02"]),
            "last_date": pd.to_datetime(["2002-05-01", "2004-06-01", "2024-12-31"]),
        }
    )
    seen_fields_call: dict[str, Any] = {}
    seen_universes: list[str] = []
    seen_reads: list[tuple[list[str], str, str, int]] = []

    def fake_fields(conn: Any, crsp_load: int, start: str, end: str) -> pd.DataFrame:
        seen_fields_call.update({"crsp_load": crsp_load, "start": start, "end": end})
        return pd.DataFrame()

    def fake_eligibility(
        _fields: pd.DataFrame,
        spans: pd.DataFrame,
        *,
        universe: str,
        daily_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        seen_universes.append(universe)
        frame = pd.DataFrame(False, index=daily_index, columns=spans["symbol"].tolist())
        if "LINK" in frame.columns:
            frame["LINK"] = True
        return frame

    def fake_close(
        symbols: list[str],
        start: str,
        end: str,
        *,
        release_id: int,
        provenance_out: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        seen_reads.append((list(symbols), start, end, release_id))
        index = pd.DatetimeIndex(pd.bdate_range(start, end), tz="UTC")
        if provenance_out is not None:
            provenance_out.update({"drift": False, "drift_symbols": [], "n_batches": 1})
        spans = pd.DataFrame({"symbol": [], "last_date": [], "has_successor": []})
        return pd.DataFrame(10.0, index=index, columns=list(symbols)), spans

    monkeypatch.setattr(bars_module, "connect_readonly", fake_connect)
    monkeypatch.setattr(bars_reader, "read_release_spans", lambda _c, _r: release)
    monkeypatch.setattr(universe_frames, "month_end_fields", fake_fields)
    monkeypatch.setattr(universe_frames, "eligibility", fake_eligibility)
    monkeypatch.setattr(bars_reader, "read_close_batched", fake_close)

    out = fresh_inputs.read_inputs("fresh", release_id=1, crsp_load=1)

    assert set(out) == {
        "close",
        "spy_close",
        "eligible",
        "span_rows",
        "release_spans",
        "false_death_audit",
        "data_provenance",
    }
    assert seen_universes == ["A", "A"]
    assert seen_fields_call == {"crsp_load": 1, "start": "1999-12-01", "end": "2004-12-31"}
    expected_read_start = (
        pd.Timestamp("2000-01-03") - pd.Timedelta(days=study.WARMUP_CALENDAR_DAYS)
    ).strftime("%Y-%m-%d")
    assert seen_reads == [
        (["LINK"], expected_read_start, "2004-12-31", 1),
        (["SPY"], expected_read_start, "2004-12-31", 1),
    ]
    assert list(out["eligible"].columns) == ["LINK"]
    assert out["eligible"].index.tz is None
    assert out["eligible"].index[0] >= pd.Timestamp("2000-01-03")
    assert out["data_provenance"]["n_candidate_columns"] == 1
    assert out["false_death_audit"]["n_suspect"] == 0
    assert out["false_death_audit"]["n_informational"] == 1
    assert len(conns) == 2 and all(conn.closed for conn in conns)


def test_read_inputs_raises_when_spy_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _StubConn()
    release = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "permno": [1],
            "spell": [1],
            "kind": ["placeholder"],
            "first_date": pd.to_datetime(["1998-01-02"]),
            "last_date": pd.to_datetime(["2024-12-31"]),
        }
    )

    def fake_close(
        symbols: list[str],
        start: str,
        end: str,
        *,
        release_id: int,
        provenance_out: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        index = pd.bdate_range(start, end)
        if symbols == ["SPY"]:
            return pd.DataFrame(index=index), pd.DataFrame()
        return pd.DataFrame(10.0, index=index, columns=list(symbols)), pd.DataFrame()

    monkeypatch.setattr(bars_module, "connect_readonly", lambda: conn)
    monkeypatch.setattr(bars_reader, "read_release_spans", lambda _c, _r: release)
    monkeypatch.setattr(universe_frames, "month_end_fields", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(
        universe_frames,
        "eligibility",
        lambda _f, spans, *, universe, daily_index: pd.DataFrame(
            True, index=daily_index, columns=spans["symbol"].tolist()
        ),
    )
    monkeypatch.setattr(bars_reader, "read_close_batched", fake_close)

    with pytest.raises(SystemExit, match="SPY has no bars"):
        fresh_inputs.read_inputs("anchor", release_id=1, crsp_load=1)


def test_read_drift_queries_release_and_closes_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _StubConn()
    seen: list[tuple[Any, int]] = []

    def fake_linked_drift(c: Any, release_id: int) -> list[str]:
        seen.append((c, release_id))
        return ["BBB", "AAA"]

    monkeypatch.setattr(bars_module, "connect_readonly", lambda: conn)
    monkeypatch.setattr(bars_reader, "read_linked_drift", fake_linked_drift)

    assert fresh_inputs.read_drift(1) == ["BBB", "AAA"]
    assert seen == [(conn, 1)]
    assert conn.closed is True


def test_drift_in_collects_sorted_symbols() -> None:
    assert fresh_inputs.drift_in({}) == []
    assert (
        fresh_inputs.drift_in(
            {
                "data_provenance": {
                    "candidates": {"drift": False, "drift_symbols": []},
                    "spy": {"drift": False, "drift_symbols": []},
                    "n_candidate_columns": 10,
                }
            }
        )
        == []
    )
    assert fresh_inputs.drift_in(
        {
            "data_provenance": {
                "candidates": {"drift": True, "drift_symbols": ["ZZZ", "AAA"]},
                "spy": {"drift": True, "drift_symbols": ["SPY", "AAA"]},
                "n_candidate_columns": 10,
            }
        }
    ) == ["AAA", "SPY", "ZZZ"]
    assert fresh_inputs.drift_in(
        {"data_provenance": {"candidates": {"drift": True, "drift_symbols": []}}}
    ) == ["(drift reported without symbols)"]
