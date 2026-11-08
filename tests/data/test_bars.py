"""load_bars: per-field frames, assertions, missing symbols, read-only DB source."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ggTrader.data import bars as bars_mod
from ggTrader.data import snapshot as snap
from ggTrader.data.bars import BarsValidationError, load_bars, to_legacy_frame
from ggTrader.data.membership import FixtureMembershipSource

REAL_PARQUET = Path(snap.DEFAULT_SNAPSHOT_DIR) / "ohlcv_20261106.parquet"
needs_real = pytest.mark.skipif(
    not REAL_PARQUET.exists(),
    reason="frozen tape data/snapshots/ohlcv_20261106.parquet is absent (git-ignored; rebuild "
    "with scripts/make_tape_snapshot.py)",
)

#: The S&P 500 list for universe requests (the frozen vintage-1 fixture).
SOURCE = FixtureMembershipSource("sp500_v1_2026-06-02")

DAYS = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])


def _long(symbols: dict[str, list[pd.Timestamp]] | None = None) -> pd.DataFrame:
    """Long rows; each symbol is given the list of days it has bars on."""
    symbols = symbols or {"AAA": list(DAYS), "BBB": list(DAYS[:3]), "CCC": list(DAYS[1:])}
    rows = []
    for sym, days in symbols.items():
        for i, ts in enumerate(days):
            px = 10.0 + len(sym) + i
            rows.append((ts, sym, px, px + 1, px - 1, px + 0.5, 100.0 + i))
    return pd.DataFrame(rows, columns=["timestamp", "symbol", *snap.FIELDS])


@pytest.fixture
def patched_snapshot(monkeypatch: pytest.MonkeyPatch):
    """Route ``load_bars(source='snapshot')`` to an in-memory long frame."""

    def _install(flat: pd.DataFrame) -> None:
        info = snap.SnapshotInfo(
            path=Path("fake.parquet"),
            sha256="a" * 64,
            content_sha256=None,
            rows=len(flat),
            symbols=int(flat["symbol"].nunique()),
            date_min=str(flat["timestamp"].min().date()),
            date_max=str(flat["timestamp"].max().date()),
            db_session_timezone="America/Los_Angeles",
            deep_checked=False,
        )

        def fake_load(path=None, symbols=None, *, deep=False):
            out = flat if symbols is None else flat[flat["symbol"].isin(set(symbols))]
            return out.reset_index(drop=True), info

        monkeypatch.setattr(bars_mod.snapshot, "load_snapshot_long", fake_load)

    return _install


def test_per_field_frames_shape_and_dtype(patched_snapshot) -> None:
    patched_snapshot(_long())
    b = load_bars(["AAA", "BBB", "CCC"], "2024-01-01", "2024-01-31")
    for field in ("close", "open", "high", "low", "volume"):
        frame = getattr(b, field)
        assert list(frame.columns) == ["AAA", "BBB", "CCC"]
        assert frame.index.is_unique and frame.index.is_monotonic_increasing
        assert str(frame.index.tz) == "UTC"
        assert all(dt == np.float64 for dt in frame.dtypes)
    assert b.close.shape == (5, 3)
    assert np.isnan(b.close.loc["2024-01-08", "BBB"])  # BBB has no bar after 01-04
    assert b.missing_symbols == ()


def test_last_valid_bar_per_symbol(patched_snapshot) -> None:
    patched_snapshot(_long())
    b = load_bars(["AAA", "BBB", "CCC"], "2024-01-01", "2024-01-31")
    assert b.last_valid_bar["AAA"] == pd.Timestamp("2024-01-08", tz="UTC")
    assert b.last_valid_bar["BBB"] == pd.Timestamp("2024-01-04", tz="UTC")
    assert b.last_valid_bar["CCC"] == pd.Timestamp("2024-01-08", tz="UTC")


def test_missing_symbols_reported(patched_snapshot) -> None:
    patched_snapshot(_long())
    b = load_bars(["AAA", "ZZZ", "YYY"], "2024-01-01", "2024-01-31")
    assert b.missing_symbols == ("YYY", "ZZZ")
    assert list(b.close.columns) == ["AAA"]
    assert set(b.last_valid_bar.index) == {"AAA"}


def test_symbol_normalisation_and_dedupe(patched_snapshot) -> None:
    flat = _long({"BRK-B": list(DAYS), "AAA": list(DAYS)})
    patched_snapshot(flat)
    b = load_bars(["brk.b", "BRK-B", "AAA"], "2024-01-01", "2024-01-31")
    assert b.requested == ("AAA", "BRK-B")
    assert list(b.close.columns) == ["AAA", "BRK-B"]


def test_date_bounds_are_inclusive(patched_snapshot) -> None:
    patched_snapshot(_long())
    b = load_bars(["AAA"], "2024-01-03", "2024-01-05")
    assert [d.date().isoformat() for d in b.close.index] == [
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
    ]


def test_open_ended_window(patched_snapshot) -> None:
    patched_snapshot(_long())
    b = load_bars(["AAA"])
    assert len(b.close) == 5


def test_rejects_duplicate_rows(patched_snapshot) -> None:
    flat = _long()
    patched_snapshot(pd.concat([flat, flat.iloc[[0]]], ignore_index=True))
    with pytest.raises(BarsValidationError, match="duplicate"):
        load_bars(["AAA"], "2024-01-01", "2024-01-31")


def test_rejects_non_midnight_stamps(patched_snapshot) -> None:
    flat = _long()
    flat.loc[0, "timestamp"] = flat.loc[0, "timestamp"] + pd.Timedelta(hours=16)
    patched_snapshot(flat)
    with pytest.raises(BarsValidationError, match="midnight"):
        load_bars(["AAA"], "2024-01-01", "2024-01-31")


def test_rejects_weekend_stamps(patched_snapshot) -> None:
    flat = _long()
    flat.loc[0, "timestamp"] = pd.Timestamp("2024-01-06")  # Saturday
    patched_snapshot(flat)
    with pytest.raises(BarsValidationError, match="weekend"):
        load_bars(["AAA"], "2024-01-01", "2024-01-31")


def test_validation_applies_to_rows_outside_the_window_too(patched_snapshot) -> None:
    flat = _long()
    flat.loc[0, "timestamp"] = pd.Timestamp("2023-06-03")  # Saturday, before the window
    patched_snapshot(flat)
    with pytest.raises(BarsValidationError, match="weekend"):
        load_bars(["AAA"], "2024-01-01", "2024-01-31")


def test_universe_and_span_validation(patched_snapshot) -> None:
    patched_snapshot(_long())
    with pytest.raises(ValueError, match="symbols or universe"):
        load_bars(None, "2024-01-01", "2024-01-31")
    with pytest.raises(ValueError, match="not both"):
        load_bars(["AAA"], universe="sp500")
    with pytest.raises(ValueError, match="universe_span"):
        load_bars(universe="sp500")
    with pytest.raises(ValueError, match="membership"):  # sp500 needs an explicit source
        load_bars(universe="sp500", universe_span=("2024-01-02", "2024-01-08"))
    with pytest.raises(ValueError, match="source"):
        load_bars(["AAA"], source="yfinance")


def test_universe_selection_uses_pit_union(patched_snapshot, monkeypatch) -> None:
    patched_snapshot(_long())
    seen = {}

    def fake_union(universe, start, end, membership=None):
        seen["args"] = (universe, start, end)
        seen["membership"] = membership
        return ["AAA", "ZZZ"]

    monkeypatch.setattr(bars_mod, "universe_all_between", fake_union)
    b = load_bars(
        universe="sp500",
        universe_span=("2024-01-02", "2024-01-08"),
        extra_symbols=["CCC"],
        start="2024-01-01",
        end="2024-01-31",
        membership=SOURCE,
    )
    assert seen["args"][0] == "sp500"
    assert seen["membership"] is SOURCE  # the chosen source is passed down, not re-selected
    assert b.requested == ("AAA", "CCC", "ZZZ")
    assert b.missing_symbols == ("ZZZ",)


@pytest.mark.parametrize(
    "span",
    [
        ("2024-01-02", "2024-01-08"),
        (pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-08", tz="UTC")),
        (pd.Timestamp("2024-01-02 00:00"), pd.Timestamp("2024-01-08 00:00")),
        (
            pd.Timestamp("2024-01-01 16:00", tz="US/Pacific"),  # = 2024-01-02 00:00 UTC
            pd.Timestamp("2024-01-07 16:00", tz="US/Pacific"),
        ),
    ],
    ids=["strings", "tz_utc", "naive", "tz_pacific_converted"],
)
def test_universe_span_accepts_aware_naive_and_string_bounds(
    patched_snapshot, monkeypatch, span
) -> None:
    """The lab passes tz-aware timestamps everywhere; the span must be converted, not rejected."""
    patched_snapshot(_long())
    seen = {}

    def fake_union(universe, start, end, membership=None):
        seen["span"] = (start, end)
        return ["AAA"]

    monkeypatch.setattr(bars_mod, "universe_all_between", fake_union)
    load_bars(
        universe="sp500",
        universe_span=span,
        start="2024-01-01",
        end="2024-01-31",
        membership=SOURCE,
    )
    start, end = seen["span"]
    assert (start, end) == (
        pd.Timestamp("2024-01-02", tz="UTC"),
        pd.Timestamp("2024-01-08", tz="UTC"),
    )


@pytest.mark.parametrize(
    ("end", "expected"),
    [
        ("2026-04-30", "2026-04-29"),  # Thursday -> Wednesday (P0: --eval-end 04-30 ends OOS 04-29)
        ("2026-05-04", "2026-05-01"),  # Monday -> the Friday before
        ("2026-05-02", "2026-05-01"),  # Saturday -> Friday
        ("2026-05-03", "2026-05-01"),  # Sunday -> Friday
        (pd.Timestamp("2026-04-30", tz="UTC"), "2026-04-29"),
        (pd.Timestamp("2026-04-30 17:00"), "2026-04-29"),  # time of day is dropped
    ],
)
def test_legacy_exclusive_end_is_the_previous_weekday(end, expected) -> None:
    got = bars_mod.legacy_exclusive_end(end)
    assert got == pd.Timestamp(expected) and got.tzinfo is None


def test_legacy_exclusive_end_reproduces_the_legacy_window(patched_snapshot) -> None:
    """Legacy excluded the bar stamped on ``end``; inclusive load_bars needs the previous day."""
    patched_snapshot(_long())
    inclusive = load_bars(["AAA"], "2024-01-02", "2024-01-05")
    assert inclusive.close.index[-1] == pd.Timestamp("2024-01-05", tz="UTC")
    legacy = load_bars(["AAA"], "2024-01-02", bars_mod.legacy_exclusive_end("2024-01-05"))
    assert legacy.close.index[-1] == pd.Timestamp("2024-01-04", tz="UTC")


@needs_real
def test_legacy_exclusive_end_matches_the_p0_oos_last_on_the_real_tape() -> None:
    """P0's 17-fold reference: --eval-end 2026-04-30 ended the last OOS fold on 2026-04-29."""
    import json

    ref = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "research/programme/vectorbt-engine/legacy-references"
            / "_legacy_reference_core17_ensemble_sp500.json"
        ).read_text()
    )
    oos_last = pd.Timestamp(ref["oos_last"]).tz_localize(None)
    b = load_bars(["SPY"], "2021-01-31", bars_mod.legacy_exclusive_end("2026-04-30"))
    assert b.close.index[-1].tz_localize(None) == oos_last


def test_legacy_frame_layout(patched_snapshot) -> None:
    patched_snapshot(_long())
    b = load_bars(["AAA", "BBB"], "2024-01-01", "2024-01-31")
    legacy = to_legacy_frame(b)
    assert list(legacy.columns.names) == ["symbol", "field"]
    assert list(legacy.columns.get_level_values(1)[:5]) == [
        "close",
        "high",
        "low",
        "open",
        "volume",
    ]
    assert legacy.columns.is_monotonic_increasing
    assert legacy.index.equals(b.close.index)
    pd.testing.assert_series_equal(legacy[("AAA", "close")], b.close["AAA"], check_names=False)
    assert b.legacy_frame().equals(legacy)


def test_summary_flags_early_ending_and_last_bar_coverage(patched_snapshot) -> None:
    patched_snapshot(_long())
    b = load_bars(["AAA", "BBB", "CCC"], "2024-01-01", "2024-01-31")
    assert b.last_bar_coverage() == pytest.approx(2 / 3)
    assert b.early_ending(min_bars=1) == ["BBB"]
    assert b.early_ending(min_bars=3) == []
    summary = b.summary(early_min_bars=1)
    assert summary["n_symbols"] == 3
    assert summary["early_ending"] == {"BBB": "2024-01-04"}
    assert summary["missing_symbols"] == []


# ---------------------------------------------------------------- db source: read-only only


class _FakeCursor:
    def __init__(self, log: list[str], rows: list[tuple], answer: str = "on") -> None:
        self.log = log
        self.rows = rows
        self.answer = answer

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.log.append(sql)
        self._last = sql

    def fetchone(self) -> tuple:
        return (self.answer,)

    def fetchall(self) -> list[tuple]:
        return self.rows


class _FakeConn:
    def __init__(self, rows: list[tuple]) -> None:
        self.log: list[str] = []
        self.rows = rows
        self.answer = "on"
        self.session: dict[str, Any] = {}
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.log, self.rows, self.answer)

    def set_session(self, **kwargs: Any) -> None:
        self.session = kwargs

    def close(self) -> None:
        self.closed = True


def _db_rows() -> list[tuple]:
    flat = _long()
    return [tuple(r) for r in flat.itertuples(index=False, name=None)]


def test_db_source_opens_read_only_and_only_selects(monkeypatch: pytest.MonkeyPatch) -> None:
    conn = _FakeConn(_db_rows())
    captured: dict[str, Any] = {}

    import psycopg2

    def fake_connect(dsn: str, **kwargs: Any) -> _FakeConn:
        captured["dsn"] = dsn
        captured["kwargs"] = kwargs
        return conn

    monkeypatch.setattr(psycopg2, "connect", fake_connect)
    monkeypatch.setattr(
        "ggTrader.utils.config.get_db_connection_string",
        lambda: "postgresql+psycopg2://u:p@localhost/db",
    )
    b = load_bars(["AAA", "BBB", "CCC"], "2024-01-01", "2024-01-31", source="db")

    assert "default_transaction_read_only=on" in captured["kwargs"]["options"]
    assert captured["dsn"].startswith("postgresql://")
    assert conn.session == {"readonly": True, "autocommit": True}
    assert conn.closed
    statements = [s.strip().split()[0].upper() for s in conn.log]
    assert set(statements) <= {"SELECT", "SHOW"}
    assert "SELECT" in statements
    assert b.close.shape == (5, 3)
    assert b.provenance["source"] == "db"


def test_db_source_refuses_a_session_that_is_not_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conn = _FakeConn(_db_rows())
    conn.answer = "off"
    import psycopg2

    monkeypatch.setattr(psycopg2, "connect", lambda dsn, **kw: conn)
    monkeypatch.setattr(
        "ggTrader.utils.config.get_db_connection_string", lambda: "postgresql://x/y"
    )
    with pytest.raises(RuntimeError, match="read-only"):
        load_bars(["AAA"], "2024-01-01", "2024-01-31", source="db")
    assert conn.closed
    assert not any(s.strip().upper().startswith("SELECT TIMESTAMP") for s in conn.log)


WRITE_SQL = re.compile(
    r"\b(insert\s+into|create\s+(table|index|schema|extension|view|database)|update\s+\w+\s+set"
    r"|delete\s+from|drop\s+(table|index|schema|view|database)|alter\s+table|truncate)\b",
    re.I,
)
BARE_WORDS = re.compile(r"\b(insert|create)\b", re.I)


@pytest.mark.parametrize("module", ["bars.py", "snapshot.py"])
def test_no_write_path_in_data_modules(module: str) -> None:
    """Static guard: no SQL write or DDL statement, and not even the words INSERT/CREATE."""
    text = (Path(bars_mod.__file__).parent / module).read_text()
    assert WRITE_SQL.findall(text) == []
    assert BARE_WORDS.findall(text) == []


def test_only_sql_statement_is_a_select() -> None:
    assert bars_mod.SELECT_SQL.lstrip().upper().startswith("SELECT")
    assert ";" not in bars_mod.SELECT_SQL.strip().rstrip(";")
    assert "default_transaction_read_only=on" in bars_mod.READ_ONLY_OPTIONS


def test_no_network_imports() -> None:
    text = (Path(bars_mod.__file__).parent / "bars.py").read_text()
    for banned in ("yfinance", "requests", "urllib", "tiingo", "alpaca"):
        assert not re.search(rf"^\s*(import|from)\s+{banned}", text, re.M | re.I), banned


# ---------------------------------------------------------------- real tape


def _script_module(name: str):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@needs_real
def test_real_tape_matches_p0_legacy_reader_bit_for_bit() -> None:
    """to_legacy_frame(load_bars) == P0's load_snapshot_wide (the frame the legacy loader gave).

    P0's reader excludes the bar stamped on ``end`` through the DB session-zone rule, so the
    equal window is its end=04-30 against ours end=04-29 (inclusive).
    """
    p0 = _script_module("make_tape_snapshot")
    syms = ["AAPL", "MSFT", "SPY", "SIVB", "PARA", "BRK-B", "NOSUCH"]
    expected = p0.load_snapshot_wide(
        REAL_PARQUET, symbols=syms, start="2021-01-31", end="2026-04-30"
    )
    got = to_legacy_frame(load_bars(syms, "2021-01-31", "2026-04-29"))
    pd.testing.assert_frame_equal(got, expected, check_exact=True)


@needs_real
def test_real_tape_universe_load_and_missing_symbols() -> None:
    b = load_bars(
        universe="sp500",
        universe_span=("2021-01-31", "2026-04-30"),
        start="2019-03-01",
        end="2026-10-30",
        extra_symbols=["SPY", "TLT", "GLD", "PDBC", "BIL"],
        membership=SOURCE,
    )
    # Manifest: 617 requested, 12 not in the database; the extras are all present.
    assert len(b.requested) == 617
    assert len(b.missing_symbols) == 12
    assert {"ANTM", "FRC", "WLTW"} <= set(b.missing_symbols)
    assert b.close.shape == (1929, 605)
    assert b.close.index.is_unique
    assert b.last_valid_bar.notna().all()
    # Manifest sanity block: last bar coverage was 554 of 605.
    assert round(b.last_bar_coverage() * 605) == 554
