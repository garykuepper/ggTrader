"""Read-only closing checks on a synthetic scratch release."""

import os
from urllib.parse import urlparse, urlunparse

import psycopg2
import pytest

from scripts.kaggle_closing_checks import connect_target, run_checks
from tests.data.kaggle_import.test_release import _inputs
from tests.data.kaggle_import.test_release import conn as db_fixture

conn = db_fixture
pytestmark = [pytest.mark.integration, pytest.mark.db]


def test_read_only_and_report(conn, tmp_path, monkeypatch):
    from ggTrader.data.kaggle_import.release import build_release

    db, load = conn
    # The shared tiny fixture uses weekend dates; move them to trading days here.
    with db.cursor() as cur:
        cur.execute("UPDATE public.ohlcv SET timestamp=timestamp + interval '1 day'")
        cur.execute(
            "UPDATE market.daily_bars SET date='1999-12-31' "
            "WHERE load_id=%s AND permno=1 AND date='2000-01-01'",
            (load,),
        )
        cur.execute(
            "UPDATE market.daily_bars SET date='2000-01-03' "
            "WHERE load_id=%s AND permno=1 AND date='2000-01-02'",
            (load,),
        )
    db.commit()
    links, settings = _inputs()
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    with db.cursor() as cur:
        cur.execute("SET LOCAL ggt.release_build = 'on'")
        for number in range(92000, 92020):
            cur.execute(
                "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) "
                "VALUES (%s,%s,1,'1999-12-31',10),(%s,%s,1,'2000-01-03',11)",
                (load, number, load, number),
            )
            cur.execute(
                "INSERT INTO market.stock_spans(release_id,symbol,permno,spell,kind,"
                "first_date,last_date) "
                "VALUES (%s,%s,%s,1,'placeholder','1999-12-31','2000-01-03')",
                (rid, f"PERMNO-{number}", number),
            )
    db.commit()
    original = urlparse(os.environ["GGT_TEST_DB_URL"])
    url = urlunparse(original._replace(path="/" + db.info.dbname))
    readonly = connect_target(url)
    try:
        with pytest.raises(psycopg2.Error), readonly.cursor() as cur:
            cur.execute("INSERT INTO ref.loads(status) VALUES ('running')")
        readonly.rollback()
    finally:
        readonly.close()
    report = run_checks(url, rid, tmp_path)
    assert report["release_id"] == rid
    checks = {item["id"]: item for item in report["checks"]}
    assert len(checks) == 13
    assert checks["check_01"]["status"] in ("pass", "skipped")
    assert checks["check_06"]["measured"]["database_bytes"] > 0
    assert checks["check_06"]["measured"]["free_bytes"] > 0
    assert checks["check_07"]["status"] == "pass"
    assert all(
        "name" in item and "measured" in item and "tolerance" in item and "elapsed_seconds" in item
        for item in checks.values()
    )
    import json
    import sys

    from scripts import kaggle_closing_checks as closing

    monkeypatch.setenv("CLOSING_SCRATCH_URL", url)
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "checks",
            "--db-url-env",
            "CLOSING_SCRATCH_URL",
            "--release-id",
            str(rid),
            "--data-volume",
            str(tmp_path),
            "--out",
            str(out),
        ],
    )
    assert closing.main() == 1  # Incomplete or failing checks prevent a successful exit.
    saved = json.loads(out.read_text())
    assert saved["release_id"] == rid and len(saved["checks"]) == 13


def test_row_counts_reconcile_database_and_parquet(conn, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ggTrader.data.kaggle_import.release import build_release
    from scripts.kaggle_closing_checks import _check_counts

    db, existing = conn
    path = tmp_path / "source.parquet"
    pq.write_table(pa.table({"permno": [1, 2]}), path)
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO ref.loads(dataset,file_path,status,rows_parsed, "
            "rows_accepted,rows_quarantined) "
            "VALUES ('crsp-daily',%s,'complete',2,2,0) RETURNING load_id",
            (str(path),),
        )
        raw = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO kaggle_crsp.daily(load_id,permno,date) "
            "VALUES (%s,1,'2000-01-01'),(%s,2,'2000-01-01')",
            (raw, raw),
        )
        cur.execute(
            "INSERT INTO ref.load_chunks(load_id,chunk_key,rows) VALUES (%s,'rg-000',2)", (raw,)
        )
        cur.execute(
            "INSERT INTO ref.loads(dataset,file_path,status,parser_version,checks) "
            "VALUES ('crsp-derived-bars',%s,'complete','derive-1',"
            "'{\"source_rows\":2}') RETURNING load_id",
            (f"load:{raw}",),
        )
        derived = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) "
            "SELECT %s,permno,spell,date,close FROM market.daily_bars WHERE load_id=%s",
            (derived, existing),
        )
        cur.execute(
            "INSERT INTO ref.loads(dataset,status,rows_parsed,rows_accepted,rows_quarantined) "
            "VALUES ('mcgill-fiam','complete',1,1,0) RETURNING load_id"
        )
        mcgill = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO kaggle_chars.mcgill(load_id,permno,date) VALUES (%s,1,'2000-01-01')",
            (mcgill,),
        )
        cur.execute(
            "INSERT INTO ref.load_chunks(load_id,chunk_key,rows) VALUES (%s,'batch-000',1)",
            (mcgill,),
        )
    db.commit()
    links, settings = _inputs()
    rid = build_release(
        db, crsp_bars_load=derived, mcgill_load=mcgill, links=links, settings=settings
    )
    status, values, _ = _check_counts(db, rid)
    assert status == "pass" and values["crsp"]["parquet_rows"] == 2
    assert values["mcgill"]["database_rows"] == 1


def test_returns_comparisons_on_synthetic_database(conn):
    from scripts.kaggle_closing_checks import _mcgill_comparison, _stock_comparison

    db, derived = conn
    with db.cursor() as cur:
        cur.execute("INSERT INTO ref.releases(status) VALUES ('published') RETURNING release_id")
        rid = cur.fetchone()[0]
        cur.execute("SET LOCAL ggt.release_build = 'on'")
        cur.execute(
            "INSERT INTO ref.release_loads(release_id,load_id,role) VALUES (%s,%s,'crsp-bars')",
            (rid, derived),
        )
        cur.execute(
            "INSERT INTO ref.loads(dataset,status) VALUES "
            "('crsp-daily','complete'),('mcgill-fiam','complete') RETURNING load_id"
        )
        raw, mcgill = [row[0] for row in cur.fetchall()]
        days = ["2005-01-31", "2005-02-28", "2005-03-31", "2005-04-29"]
        prices = [100.0, 110.0, 99.0, 118.8]
        returns = [0.0, 0.1, -0.1, 0.2]
        for day, price, ret in zip(days, prices, returns, strict=True):
            cur.execute(
                "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) "
                "VALUES (%s,10,1,%s,%s)",
                (derived, day, price),
            )
            cur.execute(
                "INSERT INTO public.ohlcv(timestamp,symbol,interval,venue,close) "
                "VALUES (%s,'EXAMPLE','1d','yfinance',%s)",
                (day, price),
            )
            cur.execute(
                "INSERT INTO kaggle_crsp.daily(load_id,permno,date,ret) VALUES (%s,10,%s,%s)",
                (raw, day, ret),
            )
            cur.execute(
                "INSERT INTO kaggle_chars.mcgill(load_id,permno,date,ret_eom,stock_exret,rf) "
                "VALUES (%s,10,%s,%s,%s,0.01)",
                (mcgill, day, day, ret - 0.01),
            )
    db.commit()
    status, measured, _ = _stock_comparison(db, rid, {"EXAMPLE": 10})
    assert status == "pass" and measured["EXAMPLE"]["count"] == 3
    assert measured["EXAMPLE"]["mean_abs_points"] == pytest.approx(0, abs=1e-9)
    status, measured, _ = _mcgill_comparison(db, raw, mcgill, count=1)
    assert status == "pass" and measured["count"] == 4
    assert measured["correlation"] == pytest.approx(1.0)
    with db.cursor() as cur:
        cur.execute(
            "UPDATE public.ohlcv SET close=50 WHERE symbol='EXAMPLE' AND timestamp='2005-03-31'"
        )
    db.commit()
    assert _stock_comparison(db, rid, {"EXAMPLE": 10})[0] == "fail"


def test_yahoo_key_comparison_detects_changed_combined_value(conn, monkeypatch):
    from ggTrader.data.kaggle_import.release import build_release
    from scripts import kaggle_closing_checks as closing

    db, load = conn
    links, settings = _inputs()
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    with db.cursor() as cur:
        cur.execute("DELETE FROM public.ohlcv WHERE timestamp='2000-01-02'")
    db.commit()
    original = urlparse(os.environ["GGT_TEST_DB_URL"])
    url = urlunparse(original._replace(path="/" + db.info.dbname))
    # Use an ordinary Yahoo-only name so no splice anchor is required.
    with db.cursor() as cur:
        cur.execute(
            "INSERT INTO public.ohlcv(timestamp,symbol,interval,venue,close) "
            "VALUES ('2000-01-03','EXAMPLE','1d','yfinance',10)"
        )
    db.commit()
    status, measured, _ = closing._check_yahoo_keys(db, url, rid, ["EXAMPLE"])
    assert status == "pass" and measured["compared_rows"] == 1
    real_load = closing._load

    def changed(url, names, source, release_id=None, **kwargs):
        bars = real_load(url, names, source, release_id, **kwargs)
        if source == "combined":
            bars.close.iloc[0, 0] = 11
        return bars

    monkeypatch.setattr(closing, "_load", changed)
    assert closing._check_yahoo_keys(db, url, rid, ["EXAMPLE"])[0] == "fail"


def test_named_audit_finds_unflagged_duplicate(conn):
    from ggTrader.data.kaggle_import.release import build_release
    from scripts.kaggle_closing_checks import _check_named_stocks

    db, load = conn
    with db.cursor() as cur:
        cur.execute("UPDATE public.ohlcv SET timestamp=timestamp + interval '1 day'")
        cur.execute(
            "UPDATE market.daily_bars SET date='1999-12-31' "
            "WHERE load_id=%s AND permno=1 AND date='2000-01-01'",
            (load,),
        )
        cur.execute(
            "UPDATE market.daily_bars SET date='2000-01-03' "
            "WHERE load_id=%s AND permno=1 AND date='2000-01-02'",
            (load,),
        )
    db.commit()
    links, settings = _inputs()
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
    with db.cursor() as cur:
        cur.execute("SET LOCAL ggt.release_build = 'on'")
        cur.execute(
            "INSERT INTO market.stock_spans(release_id,symbol,permno,spell,kind,"
            "first_date,last_date,possible_double) "
            "VALUES (%s,'PERMNO-1',1,1,'placeholder','2000-01-03','2000-01-03',false)",
            (rid,),
        )
    db.commit()
    original = urlparse(os.environ["GGT_TEST_DB_URL"])
    url = urlunparse(original._replace(path="/" + db.info.dbname))
    status, measured, _ = _check_named_stocks(db, url, rid)
    assert status == "fail" and measured["unflagged_duplicate_days"] >= 1
