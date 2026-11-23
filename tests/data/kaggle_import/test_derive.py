"""Tests for rescaling CRSP spell prices."""

import numpy as np
import pandas as pd
import pytest

from ggTrader.data.kaggle_import.derive import build_spell_closes


def test_close_anchored_and_returns():
    rows = pd.DataFrame(
        {
            "permno": [1, 1, 1, 1],
            "date": pd.date_range("2020-01-01", periods=4),
            "spell": [0, 0, 1, 1],
            "prc": [10.0, -11.0, 20.0, 22.0],
            "chained_return": [np.nan, 0.1, np.nan, 0.1],
        }
    )
    result = build_spell_closes(rows)
    assert result.close.tolist() == pytest.approx([10, 11, 20, 22])
    assert result.groupby(
        ["permno", "spell"]
    ).close.pct_change().dropna().tolist() == pytest.approx([0.1, 0.1], abs=1e-12)
    assert result.midpoint_flag.tolist() == [False, True, False, False]
    assert result.final_flag.tolist() == [False, False, False, True]


def test_total_loss_raises():
    rows = pd.DataFrame(
        {
            "permno": [1, 1],
            "date": pd.date_range("2020-01-01", periods=2),
            "spell": [0, 0],
            "prc": [10.0, 1.0],
            "chained_return": [np.nan, -1.0],
        }
    )
    with pytest.raises(ValueError, match="1.*2020-01-02"):
        build_spell_closes(rows)


@pytest.mark.integration
@pytest.mark.db
def test_derive_in_scratch_database(tmp_path):
    from contextlib import closing
    from datetime import datetime

    import psycopg2

    from ggTrader.data.kaggle_import.crsp import load_crsp
    from ggTrader.data.kaggle_import.ddl import apply_ddl
    from ggTrader.data.kaggle_import.derive import derive_bars
    from ggTrader.data.kaggle_import.loadlog import AlreadyLoaded, LoadFailed, fail_load
    from tests.data.kaggle_import.test_crsp_load import make_file, row
    from tests.data.scratch_db import scratch_database

    dates = [datetime(1999, 1, 4), datetime(1999, 1, 5)]
    rows = []
    for permno in (1, 2, 3):
        for day in dates:
            item = row(permno, day)
            item.update(prc=11.0 if day == dates[1] else 10.0, ret=0.1)
            rows.append(item)
    path, sums = make_file(tmp_path, rows, row_group_size=6)
    with scratch_database() as url, closing(psycopg2.connect(url)) as conn:
        apply_ddl(conn)
        source = load_crsp(conn, path, sums, backup_dir=None, db_free_path=tmp_path, margin_bytes=0)
        calendar = pd.DatetimeIndex(dates)
        running = load_crsp(
            conn,
            path,
            sums,
            backup_dir=None,
            db_free_path=tmp_path,
            row_groups=[0],
            parser_suffix="running",
            margin_bytes=0,
        )
        with pytest.raises(LoadFailed):
            derive_bars(conn, running, calendar, backup_dir=None, db_free_path=tmp_path)
        fail_load(conn, running, "test failure")
        with pytest.raises(LoadFailed):
            derive_bars(conn, running, calendar, backup_dir=None, db_free_path=tmp_path)
        with pytest.raises(LoadFailed):
            derive_bars(conn, source + 100, calendar, backup_dir=None, db_free_path=tmp_path)
        derived = derive_bars(
            conn,
            source,
            calendar,
            backup_dir=None,
            db_free_path=tmp_path,
            rows_per_chunk=2,
            margin_bytes=0,
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT close, final_flag FROM market.daily_bars "
                "WHERE load_id=%s ORDER BY permno, date",
                (derived,),
            )
            assert cur.fetchall() == [(10.0, False), (11.0, True)] * 3
            cur.execute("SELECT status, checks FROM ref.loads WHERE load_id=%s", (derived,))
            status, checks = cur.fetchone()
            assert status == "complete" and checks["source_rows"] == 6
        with pytest.raises(AlreadyLoaded):
            derive_bars(
                conn,
                source,
                calendar,
                backup_dir=None,
                db_free_path=tmp_path,
                margin_bytes=0,
                rows_per_chunk=2,
            )


def test_split_day_does_not_change_returns():
    from pathlib import Path

    from ggTrader.data.kaggle_import.spells import assign_spells

    fixtures = Path(__file__).parent / "fixtures"
    rows = pd.read_csv(fixtures / "crsp_cases.csv", parse_dates=["date"])
    rows = rows[(rows.permno == 14593) & rows.date.between("1987-06-04", "1987-06-24")]
    calendar = pd.DatetimeIndex(pd.read_csv(fixtures / "crsp_calendar.csv")["date"])
    bars = build_spell_closes(assign_spells(rows, calendar))
    assert bars.spell.nunique() == 1
    np.testing.assert_allclose(
        bars.close.pct_change().iloc[1:], bars.chained_return.iloc[1:], atol=1e-12
    )


def test_missing_price_raises():
    rows = pd.DataFrame(
        {
            "permno": [1],
            "date": pd.to_datetime(["2020-01-01"]),
            "spell": [0],
            "prc": [np.nan],
            "chained_return": [np.nan],
        }
    )
    with pytest.raises(ValueError, match="Missing price"):
        build_spell_closes(rows)


def test_cli_requires_yes(monkeypatch, capsys):
    import argparse

    import psycopg2

    from ggTrader.cli.cmd_db import register_db_parser, run_db

    def reject_connection(*args, **kwargs):
        raise AssertionError("dry run must not connect")

    monkeypatch.setattr(psycopg2, "connect", reject_connection)
    monkeypatch.setenv("TEST_IMPORT_URL", "postgresql://localhost/scratch")
    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(
        [
            "db",
            "import",
            "derive-bars",
            "--crsp-load",
            "1",
            "--calendar-file",
            "x",
            "--no-backup-check",
            "--db-free-path",
            "/",
            "--db-url-env",
            "TEST_IMPORT_URL",
        ]
    )
    assert run_db(args) is None
    assert "pass --yes" in capsys.readouterr().out


@pytest.mark.integration
@pytest.mark.db
def test_derived_retention_counts_running_and_complete(tmp_path, monkeypatch):
    from contextlib import closing
    from datetime import datetime

    import psycopg2

    import ggTrader.data.kaggle_import.derive as derive_module
    from ggTrader.data.kaggle_import.crsp import load_crsp
    from ggTrader.data.kaggle_import.ddl import apply_ddl
    from ggTrader.data.kaggle_import.derive import derive_bars
    from ggTrader.data.kaggle_import.loadlog import RetentionLimit
    from tests.data.kaggle_import.test_crsp_load import make_file, row
    from tests.data.scratch_db import scratch_database

    date = datetime(1999, 1, 4)
    calendar = pd.DatetimeIndex([date])
    with scratch_database() as url, closing(psycopg2.connect(url)) as conn:
        apply_ddl(conn)
        sources = []
        for number in (1, 2, 3):
            item = row(number, date)
            item["ret"] = 0.1
            path, sums = make_file(tmp_path, [item])
            sources.append(
                load_crsp(
                    conn,
                    path,
                    sums,
                    backup_dir=None,
                    db_free_path=tmp_path,
                    year=1999 if number == 3 else None,
                    margin_bytes=0,
                )
            )
        kwargs = dict(backup_dir=None, db_free_path=tmp_path, margin_bytes=0)
        derive_bars(conn, sources[0], calendar, **kwargs)

        def interrupted(*args, **kwargs):
            raise psycopg2.OperationalError("simulated interruption")

        with monkeypatch.context() as patch:
            patch.setattr(derive_module, "commit_chunk", interrupted)
            with pytest.raises(psycopg2.OperationalError):
                derive_bars(conn, sources[1], calendar, **kwargs)
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT load_id FROM ref.loads WHERE dataset='crsp-derived-bars' "
                "AND file_path=%s AND status='running'",
                (f"load:{sources[1]}",),
            )
            running = cur.fetchone()[0]
        with pytest.raises(RetentionLimit):
            derive_bars(conn, sources[2], calendar, **kwargs)
        assert derive_bars(conn, sources[1], calendar, **kwargs) == running
        with pytest.raises(RetentionLimit):
            derive_bars(conn, sources[2], calendar, **kwargs)


@pytest.mark.integration
@pytest.mark.db
def test_copy_round_trip_is_exact_for_close(tmp_path):
    from contextlib import closing
    from datetime import datetime

    import psycopg2

    from ggTrader.data.kaggle_import.crsp import load_crsp
    from ggTrader.data.kaggle_import.ddl import apply_ddl
    from ggTrader.data.kaggle_import.derive import derive_bars
    from ggTrader.data.kaggle_import.spells import assign_spells
    from tests.data.kaggle_import.test_crsp_load import make_file, row
    from tests.data.scratch_db import scratch_database

    dates = [datetime(1999, 1, day) for day in (4, 5, 6, 7)]
    rows = []
    for date, ret in zip(dates, [None, 1 / 3, -3 / 7, 1 / 7], strict=True):
        item = row(1, date)
        item.update(prc=0.1234567890123456, ret=ret)
        rows.append(item)
    path, sums = make_file(tmp_path, rows, row_group_size=len(rows))
    calendar = pd.DatetimeIndex(dates)
    expected = build_spell_closes(assign_spells(pd.DataFrame(rows), calendar)).close.tolist()
    with scratch_database() as url, closing(psycopg2.connect(url)) as conn:
        apply_ddl(conn)
        source = load_crsp(conn, path, sums, backup_dir=None, db_free_path=tmp_path, margin_bytes=0)
        derived = derive_bars(
            conn, source, calendar, backup_dir=None, db_free_path=tmp_path, margin_bytes=0
        )
        with conn.cursor() as cur:
            cur.execute(
                "SELECT close FROM market.daily_bars WHERE load_id=%s ORDER BY permno, date",
                (derived,),
            )
            assert [value for (value,) in cur.fetchall()] == expected


@pytest.mark.integration
@pytest.mark.db
def test_null_returns_and_computation_failure(tmp_path, monkeypatch):
    from contextlib import closing
    from datetime import datetime

    import psycopg2

    from ggTrader.data.kaggle_import import derive
    from ggTrader.data.kaggle_import.crsp import load_crsp
    from ggTrader.data.kaggle_import.ddl import apply_ddl
    from tests.data.kaggle_import.test_crsp_load import make_file, row
    from tests.data.scratch_db import scratch_database

    dates = [datetime(1999, 1, day) for day in (4, 5, 6)]
    items = [row(1, day) for day in dates]
    for item in items:
        item.update(prc=10.0, ret=None)
    path, sums = make_file(tmp_path, items)
    with scratch_database() as url, closing(psycopg2.connect(url)) as conn:
        apply_ddl(conn)
        source = load_crsp(conn, path, sums, backup_dir=None, db_free_path=tmp_path, margin_bytes=0)
        kwargs = dict(backup_dir=None, db_free_path=tmp_path, margin_bytes=0)

        def fail(_rows):
            raise RuntimeError("computation exploded")

        original = derive.build_spell_closes
        monkeypatch.setattr(derive, "build_spell_closes", fail)
        with pytest.raises(RuntimeError, match="computation exploded"):
            derive.derive_bars(conn, source, pd.DatetimeIndex(dates), **kwargs)
        with conn.cursor() as cur:
            cur.execute("SELECT status, checks FROM ref.loads WHERE dataset='crsp-derived-bars'")
            status, checks = cur.fetchone()
            assert status == "failed" and "computation exploded" in checks["failure"]
        monkeypatch.setattr(derive, "build_spell_closes", original)
        derived = derive.derive_bars(conn, source, pd.DatetimeIndex(dates), **kwargs)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT spell FROM market.daily_bars WHERE load_id=%s ORDER BY date", (derived,)
            )
            assert cur.fetchall() == [(0,), (1,), (2,)]
