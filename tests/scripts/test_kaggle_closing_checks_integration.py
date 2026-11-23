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
    links, settings = _inputs()
    rid = build_release(db, crsp_bars_load=load, mcgill_load=None, links=links, settings=settings)
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
