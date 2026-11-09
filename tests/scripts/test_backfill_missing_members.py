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


STORED = (
    date(2026, 9, 21),
    {"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1000.0},
)


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
    frame = bars(["2026-09-21", "2026-09-22"], close=10.2)  # fresh close 10.2 vs stored 10.0 (2%)
    off = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=False)
    on = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=True)
    assert off.n_refresh == 0 and on.n_refresh == 1 and "differs" in on.note
    same = bars(["2026-09-21", "2026-09-22"], close=10.0)  # the stored bar, field for field
    assert bm.plan_symbol("P", STORED, same, through=THROUGH, refresh_last=True).n_refresh == 0


def test_no_data_plans_nothing():
    plan = bm.plan_symbol("ZZZ", STORED, pd.DataFrame(), through=THROUGH, refresh_last=True)
    assert (
        plan.n_insert == 0
        and plan.n_refresh == 0
        and plan.failed
        and plan.note.startswith("no data")
    )


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
    def __init__(self, log, rowcount=1):
        self.log, self.rowcount = log, rowcount

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.log.append((sql, params))


class _Conn:
    def __init__(self, rowcount=1):
        self.log, self.commits, self.rollbacks, self.rowcount = [], 0, 0, rowcount

    def cursor(self):
        return _Cursor(self.log, self.rowcount)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _fake_execute_values(seen):
    def fake(cur, sql, rows, page_size, fetch=False):
        seen.append((sql, rows))
        return [(r[0],) for r in rows] if fetch else None

    return fake


def test_the_write_is_insert_do_nothing_plus_a_keyed_update_and_never_a_delete(monkeypatch):
    seen = []
    monkeypatch.setattr("psycopg2.extras.execute_values", _fake_execute_values(seen))
    frame = bars(["2026-09-21", "2026-09-22"], close=10.2)
    plan = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=True)
    conn = _Conn()
    keys, refreshed = bm.commit_plan(conn, plan)
    assert keys == [datetime(2026, 9, 22)] and refreshed == 1 and conn.commits == 1
    text = " ".join([sql for sql, _ in seen] + [sql for sql, _ in conn.log]).upper()
    assert "ON CONFLICT (TIMESTAMP, SYMBOL, INTERVAL, VENUE) DO NOTHING RETURNING TIMESTAMP" in text
    assert "DO UPDATE" not in text and "DELETE" not in text and "TRUNCATE" not in text
    sql, params = conn.log[0]
    assert "WHERE timestamp = %s AND symbol = %s AND interval = %s AND venue = %s" in sql
    # the bound values: the new bar, then the key of the OLD last bar of THIS symbol
    assert params[5:] == (datetime(2026, 9, 21), "P", "1d", "yfinance")
    assert params[3] == pytest.approx(10.2) and all(isinstance(v, float) for v in params[:5])


def test_a_refresh_that_changes_other_than_one_row_rolls_the_symbol_back(monkeypatch):
    monkeypatch.setattr("psycopg2.extras.execute_values", _fake_execute_values([]))
    plan = bm.plan_symbol(
        "P",
        STORED,
        bars(["2026-09-21", "2026-09-22"], close=10.2),
        through=THROUGH,
        refresh_last=True,
    )
    conn = _Conn(rowcount=0)
    with pytest.raises(RuntimeError):
        bm.commit_plan(conn, plan)
    assert conn.rollbacks == 1 and conn.commits == 0


def test_only_the_keys_the_database_returned_count_as_inserted(monkeypatch):
    def skipping(cur, sql, rows, page_size, fetch=False):
        return [(r[0],) for r in rows[1:]]  # a concurrent writer already had the first key

    monkeypatch.setattr("psycopg2.extras.execute_values", skipping)
    plan = bm.plan_symbol(
        "BE",
        None,
        bars(["2026-11-02", "2026-11-03", "2026-11-04"]),
        through=THROUGH,
        refresh_last=False,
    )
    keys, _ = bm.commit_plan(_Conn(), plan)
    assert keys == [datetime(2026, 11, 3), datetime(2026, 11, 4)]


def test_the_inserted_file_is_appended_after_each_commit(tmp_path):
    path = tmp_path / "inserted.csv"
    bm.append_inserted(path, "BE", [datetime(2026, 11, 3)])
    bm.append_inserted(path, "P", [datetime(2026, 9, 22), datetime(2026, 9, 23)])
    assert path.read_text().splitlines() == [
        "inserted,BE,2026-11-03",
        "inserted,P,2026-09-22",
        "inserted,P,2026-09-23",
    ]


def test_a_fresh_bar_with_missing_fields_is_never_used_for_a_refresh():
    frame = bars(["2026-09-21", "2026-09-22"], close=10.2)
    frame.loc[frame.index[0], "Open"] = float("nan")
    plan = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=True)
    assert plan.n_refresh == 0 and "missing fields" in plan.note


def test_a_volume_only_difference_also_triggers_a_refresh():
    stored = STORED
    frame = bars(["2026-09-21"], close=10.0, volume=2000.0)  # prices equal, volume doubled
    plan = bm.plan_symbol("P", stored, frame, through=THROUGH, refresh_last=True)
    assert plan.n_refresh == 1


def test_a_stored_null_field_is_always_refreshed_from_a_complete_bar():
    stored = (STORED[0], {**STORED[1], "high": None})
    frame = bars(["2026-09-21"], close=10.0)
    assert bm.plan_symbol("P", stored, frame, through=THROUGH, refresh_last=True).n_refresh == 1


def test_weekend_rows_are_dropped():
    out = bm.clean_frame(bars(["2026-11-01", "2026-11-02"]), THROUGH)  # a Sunday and a Monday
    assert [d.date().isoformat() for d in out.index] == ["2026-11-02"]


def test_the_backup_lists_every_inserted_key_and_the_old_row(tmp_path):
    frame = bars(["2026-09-21", "2026-09-22"], close=10.2)
    plan = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=True)
    path = bm.write_backup([plan], {"P": STORED}, tmp_path, "20261109T000000Z")
    lines = path.read_text().splitlines()
    assert lines[0].startswith("kind,symbol,timestamp")
    assert "planned,P,2026-09-22" in lines and any(
        line.startswith("old_row,P,2026-09-21,10.0") for line in lines
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
    monkeypatch.setattr(bm, "read_recent_bars", lambda cur, symbols: {})
    monkeypatch.setattr(bm, "fetch_history", lambda s, start, through: {"P": bars(["2026-11-03"])})
    assert bm.main(["--symbols", "P", "--through", "2026-11-05"]) == 0
    out = capsys.readouterr().out
    assert "dry run: nothing written" in out and calls["connect"] == 1


def test_a_symbol_with_no_usable_download_makes_the_run_fail_even_as_a_dry_run(monkeypatch, capsys):
    class _Reader:
        def cursor(self):
            return _Cursor([])

        def close(self):
            pass

    monkeypatch.setattr("ggTrader.data.universe_db.connect", lambda *a, **k: _Reader())
    monkeypatch.setattr(bm, "read_recent_bars", lambda cur, symbols: {})
    monkeypatch.setattr(bm, "fetch_history", lambda s, start, through: {"P": pd.DataFrame()})
    assert bm.main(["--symbols", "P", "--through", "2026-11-05"]) == 1
    assert "no usable data for: P" in capsys.readouterr().err


def test_a_prior_bar_that_disagrees_means_a_different_series_and_the_symbol_is_refused():
    frame = bars(["2026-09-18", "2026-09-21", "2026-09-22"], close=10.0)
    prior = ((date(2026, 9, 18), {"close": 8.0}),)  # the download says 10.0: 25% apart
    plan = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=False, prior=prior)
    assert plan.failed and plan.n_insert == 0 and "not the same series" in plan.note
    ok = ((date(2026, 9, 18), {"close": 10.0}),)
    assert not bm.plan_symbol(
        "P", STORED, frame, through=THROUGH, refresh_last=False, prior=ok
    ).failed


def test_a_last_bar_off_by_more_than_five_percent_is_refused_not_refreshed():
    frame = bars(["2026-09-21", "2026-09-22"], close=21.0)  # stored 10.0: a 2-for-1 look
    plan = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=True)
    assert plan.failed and plan.n_refresh == 0 and plan.n_insert == 0 and "refused" in plan.note


def test_today_is_refused_as_the_end_date(capsys):
    from datetime import datetime, timezone

    today = datetime.now(timezone.utc).date().isoformat()
    assert bm.main(["--symbols", "P", "--through", today]) == 2


def test_a_commit_run_writes_the_backup_before_any_database_write(monkeypatch, tmp_path):
    order = []

    class _Reader:
        def cursor(self):
            return _Cursor([])

        def close(self):
            pass

    class _Writer:
        def close(self):
            pass

    monkeypatch.setattr(
        "ggTrader.data.universe_db.connect",
        lambda *a, **k: _Reader() if k.get("read_only") else _Writer(),
    )
    monkeypatch.setattr(bm, "read_recent_bars", lambda cur, symbols: {})
    monkeypatch.setattr(bm, "fetch_history", lambda s, start, through: {"BE": bars(["2026-11-03"])})

    def fake_commit(conn, plan):
        order.append(("commit", sorted(p.name for p in tmp_path.iterdir())))
        return [datetime(2026, 11, 3)], 0

    monkeypatch.setattr(bm, "commit_plan", fake_commit)
    code = bm.main(
        ["--symbols", "BE", "--through", "2026-11-05", "--commit", "--backup-dir", str(tmp_path)]
    )
    assert code == 0 and order and any(n.endswith(".csv") for n in order[0][1])
    inserted = next(p for p in tmp_path.iterdir() if p.name.endswith("_inserted.csv"))
    assert inserted.read_text().strip() == "inserted,BE,2026-11-03"


def test_the_five_percent_guard_applies_even_without_refresh_last():
    frame = bars(["2026-09-21", "2026-09-22"], close=21.0)  # stored 10.0
    plan = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=False)
    assert plan.failed and plan.n_insert == 0 and "not the same series" in plan.note


def test_a_download_that_overlaps_no_stored_bar_cannot_be_verified_and_is_refused():
    frame = bars(["2026-09-22", "2026-09-23"], close=10.0)  # nothing on or before the last stored
    plan = bm.plan_symbol("P", STORED, frame, through=THROUGH, refresh_last=True)
    assert plan.failed and plan.n_insert == 0 and "cannot be verified" in plan.note


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("Close", -1.0, "non-positive close"),
        ("Close", float("inf"), "non-finite"),
        ("Volume", -5.0, "negative volume"),
        ("Open", -2.0, "negative open"),
        ("High", 1.0, "high below low"),  # High 1.0 < Low 9.0
    ],
)
def test_invalid_downloaded_values_refuse_the_whole_symbol(field, value, reason):
    frame = bars(["2026-11-03", "2026-11-04"])
    frame.loc[frame.index[1], field] = value
    plan = bm.plan_symbol("BE", None, frame, through=THROUGH, refresh_last=False)
    assert plan.failed and plan.n_insert == 0 and reason in plan.note


def test_zero_volume_is_valid():
    frame = bars(["2026-11-03"], volume=0.0)
    plan = bm.plan_symbol("FERG", None, frame, through=THROUGH, refresh_last=False)
    assert not plan.failed and plan.n_insert == 1


def _commit_main(monkeypatch, tmp_path, append=None):
    order = []

    class _Reader:
        def cursor(self):
            return _Cursor([])

        def close(self):
            pass

    class _Writer:
        def close(self):
            pass

    monkeypatch.setattr(
        "ggTrader.data.universe_db.connect",
        lambda *a, **k: _Reader() if k.get("read_only") else _Writer(),
    )
    monkeypatch.setattr(bm, "read_recent_bars", lambda cur, symbols: {})
    monkeypatch.setattr(bm, "fetch_history", lambda s, start, through: {"BE": bars(["2026-11-03"])})
    real_backup = bm.write_backup

    def backup(*a, **k):
        order.append("backup")
        return real_backup(*a, **k)

    def commit(conn, plan):
        order.append("commit")
        return [datetime(2026, 11, 3)], 0

    def journal(path, symbol, stamps):
        order.append("append")
        if append is not None:
            raise append

    monkeypatch.setattr(bm, "write_backup", backup)
    monkeypatch.setattr(bm, "commit_plan", commit)
    monkeypatch.setattr(bm, "append_inserted", journal)
    code = bm.main(
        ["--symbols", "BE", "--through", "2026-11-05", "--commit", "--backup-dir", str(tmp_path)]
    )
    return code, order


def test_the_order_is_backup_then_commit_then_journal(monkeypatch, tmp_path):
    code, order = _commit_main(monkeypatch, tmp_path)
    assert code == 0 and order == ["backup", "commit", "append"]


def test_a_journal_failure_after_the_commit_is_reported_as_committed(monkeypatch, tmp_path, capsys):
    code, order = _commit_main(monkeypatch, tmp_path, append=OSError("disk full"))
    err = capsys.readouterr().err
    assert code == 1 and order == ["backup", "commit", "append"]
    assert "COMMITTED" in err and "planned file" in err


def test_the_journal_is_synced_to_disk(monkeypatch, tmp_path):
    synced = []
    monkeypatch.setattr(bm.os, "fsync", lambda fd: synced.append(fd))
    bm.append_inserted(tmp_path / "j.csv", "BE", [datetime(2026, 11, 3)])
    assert len(synced) == 1


def test_the_script_has_help():
    import subprocess

    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0 and "--commit" in done.stdout
