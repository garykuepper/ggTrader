"""Fast offline tests of ``scripts/backfill_missing_members.py``: planning, SQL shape, safety."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "backfill_missing_members.py"


def _load():
    spec = importlib.util.spec_from_file_location("backfill_missing_members", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bm = _load()
THROUGH = date(2026, 11, 5)


def bars(days, close=10.0, tz=None, volume=1000.0):
    index = pd.DatetimeIndex(pd.to_datetime(days), tz=tz)
    n = len(index)
    return pd.DataFrame(
        {
            "Open": [close] * n,
            "High": [close + 1] * n,
            "Low": [close - 1] * n,
            "Close": [close + i * 0.1 for i in range(n)],
            "Volume": [volume] * n,
        },
        index=index,
    )


STORED = (date(2026, 9, 21), {"open": 9.0, "high": 11.0, "low": 8.0, "close": 10.0, "volume": 5.0})


def test_clean_frame_makes_naive_utc_dates_drops_no_close_and_future_bars():
    frame = bars(["2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"], tz="UTC")
    frame.loc[frame.index[1], "Close"] = float("nan")
    out = bm.clean_frame(frame, THROUGH)
    assert out.index.tz is None and list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert [d.date().isoformat() for d in out.index] == ["2026-11-03", "2026-11-05"]


def test_a_symbol_with_no_bars_gets_its_whole_history():
    plan = bm.plan_symbol(
        "VYLR", None, bars(["2026-10-01", "2026-10-02"]), through=THROUGH, refresh_last=False
    )
    assert plan.n_insert == 2 and plan.n_refresh == 0 and plan.last_stored is None


def test_a_stale_symbol_gets_only_the_bars_after_its_last_stored_one():
    frame = bars(["2026-09-18", "2026-09-21", "2026-09-22", "2026-11-05", "2026-11-09"])
    plan = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=False)
    assert [d.date().isoformat() for d in plan.inserts.index] == ["2026-09-22", "2026-11-05"]
    assert plan.n_refresh == 0


def test_the_last_bar_is_refreshed_only_when_asked_and_only_beyond_the_tolerance():
    frame = bars(["2026-09-21", "2026-09-22"], close=10.5)  # fresh close 10.5 vs stored 10.0
    off = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=False)
    on = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=True)
    assert off.n_refresh == 0 and on.n_refresh == 1 and "differs" in on.note
    same = bars(["2026-09-21", "2026-09-22"], close=10.0)
    assert bm.plan_symbol("P", STORED, same, through=THROUGH, refresh_last=True).n_refresh == 0


def test_no_data_plans_nothing():
    plan = bm.plan_symbol("ZZZ", STORED, pd.DataFrame(), through=THROUGH, refresh_last=True)
    assert plan.n_insert == 0 and plan.n_refresh == 0 and plan.note == "no data"


def test_db_rows_carry_naive_midnight_timestamps_and_the_project_conventions():
    plan = bm.plan_symbol(
        "BE", None, bars(["2026-11-03"], tz="UTC"), through=THROUGH, refresh_last=False
    )
    (row,) = bm.rows_for_db("BE", plan.inserts)
    assert row[0] == datetime(2026, 11, 3) and row[0].tzinfo is None
    assert row[1:4] == ("BE", "1d", "yfinance") and row[-1] == 0 and row[0].weekday() < 5


def test_symbols_are_validated():
    assert bm.parse_symbols("be, FERG,be") == ["BE", "FERG"]
    for bad in ("", "x;drop table ohlcv", "TOOLONGTICKER", "a b"):
        with pytest.raises(argparse.ArgumentTypeError):
            bm.parse_symbols(bad)


class _Cursor:
    def __init__(self, log):
        self.log, self.rowcount = log, 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.log.append(sql)

    def mogrify(self, template, row):
        return b"(x)"


class _Conn:
    def __init__(self):
        self.log, self.commits = [], 0

    def cursor(self):
        return _Cursor(self.log)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def test_the_write_is_insert_do_nothing_plus_a_keyed_update_and_never_a_delete(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "psycopg2.extras.execute_values", lambda cur, sql, rows, page_size: seen.append(sql)
    )
    frame = bars(["2026-09-21", "2026-09-22"], close=10.5)
    plan = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=True)
    conn = _Conn()
    assert bm.commit_plan(conn, plan) == (1, 1) and conn.commits == 1
    text = " ".join(seen + conn.log).upper()
    assert "ON CONFLICT (TIMESTAMP, SYMBOL, INTERVAL, VENUE) DO NOTHING" in text
    assert "DO UPDATE" not in text and "DELETE" not in text and "TRUNCATE" not in text
    update = next(s for s in conn.log if s.startswith("UPDATE"))
    assert "WHERE timestamp = %s AND symbol = %s AND interval = %s AND venue = %s" in update


def test_the_backup_lists_every_inserted_key_and_the_old_row(tmp_path):
    frame = bars(["2026-09-21", "2026-09-22"], close=10.5)
    plan = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=True)
    path = bm.write_backup([plan], {"P": STORED}, tmp_path, "20261109T000000Z")
    lines = path.read_text().splitlines()
    assert lines[0].startswith("kind,symbol,timestamp")
    assert "insert,P,2026-09-22" in lines and any(
        line.startswith("old_row,P,2026-09-21,9.0") for line in lines
    )
    with pytest.raises(FileExistsError):
        bm.write_backup([plan], {"P": STORED}, tmp_path, "20261109T000000Z")  # never overwritten
    assert bm.write_backup([], {}, tmp_path, "x") is None


def test_main_refuses_today_or_later_and_a_dry_run_writes_nothing(monkeypatch, capsys):
    assert bm.main(["--symbols", "P", "--through", "2999-01-01"]) == 2

    class _Reader:
        def cursor(self):
            return _Cursor([])

        def close(self):
            pass

    calls = {"connect": 0}

    def fake_connect(*args, **kwargs):
        calls["connect"] += 1
        assert kwargs.get("read_only") is True  # a dry run may only open a read-only session
        return _Reader()

    monkeypatch.setattr("ggTrader.data.universe_db.connect", fake_connect)
    monkeypatch.setattr(bm, "read_last_bars", lambda cur, symbols: {})
    monkeypatch.setattr(bm, "fetch_history", lambda s, start, through: {"P": bars(["2026-11-03"])})
    assert bm.main(["--symbols", "P", "--through", "2026-11-05"]) == 0
    out = capsys.readouterr().out
    assert "dry run: nothing written" in out and calls["connect"] == 1


def test_the_script_has_help():
    import subprocess

    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0 and "--commit" in done.stdout
