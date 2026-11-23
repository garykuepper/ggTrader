"""Derived bars are pinned to every input that determines their output."""

import pandas as pd
import psycopg2
import pytest

from ggTrader.data.kaggle_import.ddl import apply_ddl
from ggTrader.data.kaggle_import.derive import derive_bars
from ggTrader.data.kaggle_import.loadlog import AlreadyLoaded, LoadFailed
from tests.data.scratch_db import scratch_database

pytestmark = [pytest.mark.integration, pytest.mark.db]


@pytest.fixture
def inputs(tmp_path):
    with scratch_database() as url, psycopg2.connect(url) as conn:
        apply_ddl(conn)
        ids = []
        with conn.cursor() as cur:
            for parser in ("parser-1", "parser-2"):
                cur.execute(
                    "INSERT INTO ref.loads(dataset,status,file_sha256,parser_version) "
                    "VALUES ('crsp-daily','running',%s,%s) RETURNING load_id",
                    ("a" * 64, parser),
                )
                source = cur.fetchone()[0]
                ids.append(source)
                cur.execute(
                    "INSERT INTO kaggle_crsp.daily(load_id,permno,date,prc,ret,shrout) "
                    "VALUES (%s,1,'2020-01-02',10,0,1),(%s,1,'2020-01-03',11,0.1,1)",
                    (source, source),
                )
                cur.execute("UPDATE ref.loads SET status='complete' WHERE load_id=%s", (source,))
        conn.commit()
        yield conn, ids, pd.to_datetime(["2020-01-02", "2020-01-03"]), tmp_path


def derive(inputs, source, calendar, **kwargs):
    conn, _, _, path = inputs
    return derive_bars(
        conn, source, calendar, db_free_path=path, backup_dir=None, margin_bytes=0, **kwargs
    )


def test_corrected_parser_gets_distinct_derived_load(inputs):
    _, (first, second), calendar, _ = inputs
    assert derive(inputs, first, calendar) != derive(inputs, second, calendar)


def test_changed_calendar_gets_distinct_derived_load(inputs):
    _, (source, _), calendar, _ = inputs
    assert derive(inputs, source, calendar) != derive(
        inputs, source, calendar.union(pd.to_datetime(["2020-01-06"]))
    )


def test_same_identity_still_reports_already_loaded(inputs):
    conn, (source, _), calendar, _ = inputs
    derived = derive(inputs, source, calendar)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT file_sha256,content_sha256,checks FROM ref.loads WHERE load_id=%s", (derived,)
        )
        identity_sha, raw_sha, checks = cur.fetchone()
    identity = checks["derivation_identity"]
    assert identity_sha != raw_sha == "a" * 64
    assert identity["parent_load_id"] == source
    assert identity["parent_file_sha256"] == raw_sha
    assert identity["parent_parser_version"] == "parser-1"
    assert len(identity["calendar_sha256"]) == 64
    assert identity["derivation_version"] == "derive-1"
    assert identity["chunk_layout"]["rows_per_chunk"] == 1_000_000
    with pytest.raises(AlreadyLoaded):
        derive(inputs, source, calendar)


def test_resume_with_changed_chunk_layout_is_refused(inputs, monkeypatch):
    from ggTrader.data.kaggle_import import derive as module

    conn, (source, _), calendar, _ = inputs

    def interruption(*args, **kwargs):
        raise psycopg2.OperationalError("simulated disconnect")

    original = module.commit_chunk
    monkeypatch.setattr(module, "commit_chunk", interruption)
    with pytest.raises(psycopg2.OperationalError):
        derive(inputs, source, calendar, rows_per_chunk=1)
    conn.rollback()
    monkeypatch.setattr(module, "commit_chunk", original)
    with pytest.raises(LoadFailed, match="(?i)(identity|layout|chunk)"):
        derive(inputs, source, calendar, rows_per_chunk=2)
