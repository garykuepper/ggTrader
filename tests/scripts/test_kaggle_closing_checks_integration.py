"""Read-only closing checks on a synthetic scratch release."""

import os
from urllib.parse import urlparse, urlunparse

import psycopg2
import pytest

from ggTrader.data.kaggle_import.guards import checked_pg_url
from scripts.kaggle_closing_checks import connect_target, run_checks
from tests.data.kaggle_import.test_release import _inputs
from tests.data.kaggle_import.test_release import conn as db_fixture

conn = db_fixture
pytestmark = [pytest.mark.integration, pytest.mark.db]


def test_read_only_and_report(conn, tmp_path, monkeypatch, capsys):
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

    def add_controls(cur, rid):
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

    rid = build_release(
        db,
        crsp_bars_load=load,
        mcgill_load=None,
        links=links,
        settings=settings,
        before_publish=add_controls,
    )
    original = urlparse(checked_pg_url(os.environ["GGT_TEST_DB_URL"]))
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
    assert report["statement_timeout_ms"] == 900_000
    progress = capsys.readouterr().err
    for number in range(1, 14):
        assert f"check_{number:02}" in progress
    assert "Ten-stock monthly returns" in progress and "McGill and CRSP returns" in progress
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
        cur.execute(
            "INSERT INTO ref.loads(dataset,status) VALUES "
            "('crsp-daily','complete'),('mcgill-fiam','complete') RETURNING load_id"
        )
        raw, mcgill = [row[0] for row in cur.fetchall()]
        cur.execute(
            "INSERT INTO ref.loads(dataset,status,file_path) "
            "VALUES ('crsp-derived-bars','complete',%s) RETURNING load_id",
            (f"load:{raw}",),
        )
        derived = cur.fetchone()[0]
        cur.execute("INSERT INTO ref.releases(status) VALUES ('building') RETURNING release_id")
        rid = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO ref.release_loads(release_id,load_id,role) "
            "VALUES (%s,%s,'crsp-bars'),(%s,%s,'mcgill')",
            (rid, derived, rid, mcgill),
        )
        cur.execute("UPDATE ref.releases SET status='published' WHERE release_id=%s", (rid,))
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
                "INSERT INTO kaggle_chars.mcgill(load_id,permno,date,ret_eom,stock_exret,rf,"
                "stock_ticker) "
                "VALUES (%s,10,%s,%s,%s,0.01,'EXAMPLE')",
                (mcgill, day, day, ret - 0.01),
            )
    db.commit()
    status, measured, _ = _stock_comparison(db, rid, {"EXAMPLE": 10})
    assert status == "pass" and measured["EXAMPLE"]["count"] == 3
    assert measured["EXAMPLE"]["raw_rows"] == 4
    assert measured["EXAMPLE"]["mcgill_dated_tickers"][0]["permno"] == 10
    assert measured["EXAMPLE"]["mean_abs_points"] == pytest.approx(0, abs=1e-9)
    status, measured, _ = _mcgill_comparison(db, raw, mcgill, count=1)
    assert status == "pass" and measured["count"] == 4
    assert measured["correlation"] == pytest.approx(1.0)
    with db.cursor() as cur:
        cur.execute(
            "UPDATE kaggle_chars.mcgill SET stock_exret=0.9 WHERE load_id=%s AND date='2005-03-31'",
            (mcgill,),
        )
    db.commit()
    assert _mcgill_comparison(db, raw, mcgill, count=1)[0] == "fail"
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
    original = urlparse(checked_pg_url(os.environ["GGT_TEST_DB_URL"]))
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

    def add_duplicate(cur, rid):
        cur.execute(
            "INSERT INTO market.stock_spans(release_id,symbol,permno,spell,kind,"
            "first_date,last_date,possible_double) "
            "VALUES (%s,'PERMNO-1',1,1,'placeholder','2000-01-03','2000-01-03',false)",
            (rid,),
        )

    rid = build_release(
        db,
        crsp_bars_load=load,
        mcgill_load=None,
        links=links,
        settings=settings,
        before_publish=add_duplicate,
    )
    original = urlparse(checked_pg_url(os.environ["GGT_TEST_DB_URL"]))
    url = urlunparse(original._replace(path="/" + db.info.dbname))
    status, measured, _ = _check_named_stocks(db, url, rid)
    assert status == "fail" and measured["crsp_backed_overlapping_stock_days"] >= 1


@pytest.fixture
def healthy_release(conn, tmp_path, monkeypatch, request):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ggTrader.data.kaggle_import.release import build_release

    db, _ = conn
    path = tmp_path / "healthy.parquet"
    # AAPL has three rows, the two 10128 spells have one each, and 20 controls have two each.
    pq.write_table(pa.table({"row": list(range(45))}), path)
    with db.cursor() as cur:
        cur.execute("DELETE FROM public.ohlcv")
        cur.execute(
            "INSERT INTO ref.loads(dataset,file_path,status,rows_parsed, "
            "rows_accepted,rows_quarantined) "
            "VALUES ('crsp-daily',%s,'complete',45,45,0) RETURNING load_id",
            (str(path),),
        )
        raw = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO ref.loads(dataset,file_path,status,parser_version,checks) "
            "VALUES ('crsp-derived-bars',%s,'complete','derive-1',"
            "'{\"source_rows\":45}') RETURNING load_id",
            (f"load:{raw}",),
        )
        derived = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO ref.loads(dataset,status,rows_parsed,rows_accepted,rows_quarantined) "
            "VALUES ('mcgill-fiam','complete',1,1,0) RETURNING load_id"
        )
        mcgill = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO ref.load_chunks(load_id,chunk_key,rows) "
            "VALUES (%s,'rg',45),(%s,'batch',1)",
            (raw, mcgill),
        )
        cur.execute(
            "INSERT INTO kaggle_chars.mcgill(load_id,permno,date) VALUES (%s,14593,'2000-01-03')",
            (mcgill,),
        )
        rows = [
            (14593, 1, "1980-12-15", 10.0),
            (14593, 1, "1999-12-31", 10.0),
            (14593, 1, "2000-01-03", 10.0),
            (10128, 1, "1990-01-02", 3.0),
            (10128, 2, "1999-12-31", 4.0),
        ]
        rows += [
            (number, 1, day, 10.0)
            for number in range(93000, 93020)
            for day in ("1999-12-31", "2000-01-03")
        ]
        cur.executemany(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close,abs_prc) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            [(derived, permno, spell, day, close, close) for permno, spell, day, close in rows],
        )
        cur.executemany(
            "INSERT INTO kaggle_crsp.daily(load_id,permno,date,ret) VALUES (%s,%s,%s,0)",
            [(raw, permno, day) for permno, _, day, _ in rows],
        )
        cur.executemany(
            "INSERT INTO public.ohlcv(timestamp,symbol,interval,venue,close) "
            "VALUES (%s,'AAPL','1d','yfinance',%s)",
            [("2000-01-03", 10.0), ("2000-01-04", 11.0), ("2000-01-05", 12.0)],
        )
    db.commit()
    links, settings = _inputs()
    links.loc[0, ["symbol", "permno"]] = ["AAPL", 14593]

    def add_negative_control(cur, rid):
        if getattr(getattr(request.node, "callspec", None), "params", {}).get("check") == "05":
            cur.execute(
                "INSERT INTO market.stock_spans(release_id,symbol,permno,spell,kind,"
                "first_date,last_date,possible_double) "
                "VALUES (%s,'PERMNO-14593',14593,1,'placeholder',"
                "'1980-12-15','1999-12-31',false)",
                (rid,),
            )

    rid = build_release(
        db,
        crsp_bars_load=derived,
        mcgill_load=mcgill,
        links=links,
        settings=settings,
        before_publish=add_negative_control,
    )
    source = urlparse(checked_pg_url(os.environ["GGT_TEST_DB_URL"]))
    url = urlunparse(source._replace(path="/" + db.info.dbname))
    monkeypatch.setenv("HEALTHY_SCRATCH_URL", url)
    return db, rid, url


def _run_healthy_cli(healthy_release, tmp_path, monkeypatch, *options):
    import json
    import sys
    from pathlib import Path

    from scripts import kaggle_closing_checks as closing

    db, rid, _ = healthy_release
    out = tmp_path / "healthy-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "checks",
            "--db-url-env",
            "HEALTHY_SCRATCH_URL",
            "--release-id",
            str(rid),
            "--data-volume",
            str(Path(__file__).resolve().parents[2]),
            "--out",
            str(out),
            *options,
        ],
    )
    code = closing.main()
    return code, {item["id"]: item for item in json.loads(out.read_text())["checks"]}


def test_healthy_synthetic_release_requires_explicit_skip(healthy_release, tmp_path, monkeypatch):
    code, checks = _run_healthy_cli(
        healthy_release, tmp_path, monkeypatch, "--allow-synthetic-skips"
    )
    assert code == 0
    assert (
        checks["check_01"]["status"]
        == checks["check_05"]["status"]
        == checks["check_07"]["status"]
        == "pass"
    )
    assert all(
        item["status"] in ("pass", "covered_by_tests")
        or (
            item["status"] == "skipped"
            and item["reason"] in ("synthetic data", "name not in release", "no baseline given")
        )
        for item in checks.values()
    )
    code, _ = _run_healthy_cli(healthy_release, tmp_path, monkeypatch)
    assert code != 0


@pytest.mark.parametrize("check", ["01", "04", "05", "07"])
def test_healthy_release_negative_controls(healthy_release, tmp_path, monkeypatch, check):
    from dataclasses import replace

    import ggTrader.data.bars as bars_module
    from scripts import kaggle_closing_checks as closing

    db, rid, url = healthy_release
    options = ["--allow-synthetic-skips"]
    if check == "01":
        with db.cursor() as cur:
            cur.execute("DELETE FROM kaggle_crsp.daily WHERE permno=14593 AND date='1980-12-15'")
        db.commit()
    elif check == "04":
        readonly = closing.connect_target(url)
        baseline = tmp_path / "baseline.json"
        try:
            closing._baseline(readonly, url, save=baseline)
        finally:
            readonly.close()
        with db.cursor() as cur:
            cur.execute(
                "UPDATE public.ohlcv SET close=13 WHERE symbol='AAPL' AND timestamp='2000-01-04'"
            )
        db.commit()
        options += ["--compare-baseline", str(baseline)]
    elif check == "05":
        pass  # The duplicate was included before publication by the fixture.
    else:
        original = closing._load

        def timezone_fault(*args, **kwargs):
            result = original(*args, **kwargs)
            if os.environ.get("PGTZ") == "America/Los_Angeles":
                close = result.close.copy()
                close.index = close.index + __import__("pandas").Timedelta(hours=1)
                frames = {
                    name: (close if name == "close" else result.frame(name))
                    for name in bars_module.BAR_FIELDS
                }
                result = replace(
                    result,
                    close=close,
                    provenance={
                        **result.provenance,
                        "bars_sha256": bars_module.bars_sha256(frames),
                    },
                )
            return result

        monkeypatch.setattr(closing, "_load", timezone_fault)
    code, checks = _run_healthy_cli(healthy_release, tmp_path, monkeypatch, *options)
    assert code != 0 and checks[f"check_{check}"]["status"] == "fail"


@pytest.mark.parametrize("include_invalid", [False, True])
def test_identity_audit_classifies_candidates_on_synthetic_release(conn, include_invalid):
    import pandas as pd

    from ggTrader.data.kaggle_import.release import build_release
    from scripts.kaggle_closing_checks import _check_named_stocks

    db, load = conn
    with db.cursor() as cur:
        cur.execute("UPDATE public.ohlcv SET timestamp=timestamp + interval '1 day'")
        cur.execute(
            "UPDATE market.daily_bars SET date='1999-12-31' WHERE load_id=%s "
            "AND permno=1 AND date='2000-01-01'",
            (load,),
        )
        cur.execute(
            "UPDATE market.daily_bars SET date='2000-01-03' WHERE load_id=%s "
            "AND permno=1 AND date='2000-01-02'",
            (load,),
        )
        cur.execute(
            "INSERT INTO public.ohlcv(timestamp,symbol,interval,venue,close) VALUES "
            "('2000-01-03','ACGL','1d','yfinance',8),"
            "('2000-01-03','CMVT','1d','yfinance',8)"
        )
    db.commit()
    links, settings = _inputs()
    links = pd.concat(
        [
            links,
            pd.DataFrame(
                [
                    dict(symbol="ACGL", permno=1, status="unlinked"),
                    dict(symbol="CMVT", permno=2, status="unlinked"),
                ]
            ),
        ],
        ignore_index=True,
    )

    def add_bad_rows(cur, rid):
        if not include_invalid:
            return
        cur.execute(
            "INSERT INTO market.daily_bars(load_id,permno,spell,date,close) VALUES "
            "(%s,3,1,'1999-01-04',10)",
            (load,),
        )
        cur.execute(
            "INSERT INTO market.stock_spans(release_id,symbol,permno,spell,kind,"
            "first_date,last_date,possible_double) VALUES "
            "(%s,'BAD',3,NULL,'yahoo','2000-01-03','2000-01-03',true),"
            "(%s,'PERMNO-3',3,1,'placeholder','1999-01-04','1999-01-04',false),"
            "(%s,'PERMNO-1',1,1,'placeholder','2000-01-03','2000-01-03',true)",
            (rid,) * 3,
        )

    rid = build_release(
        db,
        crsp_bars_load=load,
        mcgill_load=None,
        links=links,
        settings=settings,
        before_publish=add_bad_rows,
    )
    original = urlparse(checked_pg_url(os.environ["GGT_TEST_DB_URL"]))
    url = urlunparse(original._replace(path="/" + db.info.dbname))
    status, measured, _ = _check_named_stocks(db, url, rid)
    assert measured["yahoo_candidates_matched_to_different_linked_company"] == 1
    assert measured["different_company_candidate_examples"][0]["yahoo_symbol"] == "ACGL"
    assert measured["yahoo_candidates_missing_double_flags"] == int(include_invalid)
    assert measured["yahoo_candidate_and_crsp_pairs_checked"] == 1 + int(include_invalid)
    assert measured["yahoo_candidate_and_crsp_pairs_both_flagged"] == 1
    assert measured["crsp_backed_overlapping_stock_days"] == (2 if include_invalid else 0)
    if include_invalid:
        assert status == "fail"
        assert measured["yahoo_candidate_flag_exceptions"][0]["yahoo_symbol"] == "BAD"
