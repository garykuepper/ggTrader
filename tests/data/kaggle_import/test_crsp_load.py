"""CRSP import tests using synthetic data and a scratch database."""

import hashlib
from contextlib import closing
from datetime import datetime

import psycopg2
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ggTrader.data.kaggle_import.crsp import compress_kaggle_chunks, load_crsp
from ggTrader.data.kaggle_import.ddl import apply_ddl
from ggTrader.data.kaggle_import.guards import HashMismatch
from ggTrader.data.kaggle_import.loadlog import LoadFailed
from tests.data.scratch_db import scratch_database

pytestmark = [pytest.mark.integration, pytest.mark.db]


@pytest.fixture
def conn():
    with scratch_database() as url, closing(psycopg2.connect(url)) as connection:
        apply_ddl(connection)
        yield connection


def make_file(tmp_path, rows, row_group_size=1):
    schema = pa.schema(
        [
            (k, t)
            for k, t in [
                ("permno", pa.int64()),
                ("date", pa.timestamp("ns")),
                ("shrcd", pa.float64()),
                ("exchcd", pa.float64()),
                ("prc", pa.float64()),
                ("ret", pa.float64()),
                ("shrout", pa.float64()),
                ("vwretd", pa.float64()),
                ("year", pa.int32()),
                ("month", pa.int32()),
            ]
        ]
    )
    path = tmp_path / "x.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, row_group_size=row_group_size)
    sums = tmp_path / "sums"
    sums.write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    return path, sums


def row(permno, stamp, shrcd=10.0):
    return dict(
        permno=permno,
        date=stamp,
        shrcd=shrcd,
        exchcd=1.0,
        prc=-5.0,
        ret=float("nan"),
        shrout=10.0,
        vwretd=0.1,
        year=stamp.year,
        month=stamp.month,
    )


def test_resume_quarantine_and_compression(conn, tmp_path):
    path, sums = make_file(tmp_path, [row(1, datetime(1999, 1, 1)), row(2, datetime(1900, 1, 1))])
    first = load_crsp(conn, path, sums, backup_dir=None, row_groups=[0], margin_bytes=0)
    assert load_crsp(conn, path, sums, backup_dir=None, margin_bytes=0) == first
    with conn.cursor() as cur:
        cur.execute("SELECT prc, shrcd, ret FROM kaggle_crsp.daily WHERE load_id=%s", (first,))
        assert cur.fetchone() == (-5.0, 10, None)
        cur.execute(
            "SELECT rows_parsed, rows_quarantined, checks FROM ref.loads WHERE load_id=%s", (first,)
        )
        parsed, quarantined, checks = cur.fetchone()
        assert (parsed, quarantined, checks["nulled_values"]) == (2, 1, 1)
    assert compress_kaggle_chunks(conn) == 1
    load_crsp(conn, path, sums, backup_dir=None, parser_suffix="v2", margin_bytes=0)
    assert compress_kaggle_chunks(conn) > 0
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM kaggle_crsp.daily")
        assert cur.fetchone()[0] == 2
        cur.execute(
            "SELECT count(*) FROM _timescaledb_catalog.chunk c JOIN "
            "_timescaledb_catalog.hypertable h ON c.hypertable_id=h.id "
            "WHERE h.schema_name='kaggle_crsp' AND h.table_name='daily' "
            "AND (c.status & 8) <> 0"
        )
        assert cur.fetchone()[0] == 0
    assert compress_kaggle_chunks(conn) == 0


def test_pilot_and_full_are_distinct(conn, tmp_path):
    path, sums = make_file(tmp_path, [row(1, datetime(1999, 1, 1))])
    pilot = load_crsp(conn, path, sums, backup_dir=None, year=1999, margin_bytes=0)
    full = load_crsp(conn, path, sums, backup_dir=None, margin_bytes=0)
    assert pilot != full


def test_fractional_and_hash_and_space(conn, tmp_path, monkeypatch):
    path, sums = make_file(tmp_path, [row(1, datetime(1999, 1, 1), 10.5)])
    with pytest.raises(LoadFailed, match="permno 1"):
        load_crsp(conn, path, sums, backup_dir=None, margin_bytes=0)
    sums.write_text("0  x.parquet\n")
    with pytest.raises(HashMismatch):
        load_crsp(conn, path, sums, backup_dir=None, margin_bytes=0)
    sums.write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  x.parquet\n")
    monkeypatch.setattr("ggTrader.data.kaggle_import.crsp.free_space_ok", lambda *a, **k: False)
    with pytest.raises(LoadFailed, match="space"):
        load_crsp(conn, path, sums, backup_dir=None, margin_bytes=0)


def test_non_midnight_rolls_back_chunk(conn, tmp_path):
    path, sums = make_file(tmp_path, [row(1, datetime(1999, 1, 1, 12))])
    with pytest.raises(LoadFailed, match="Non-midnight"):
        load_crsp(conn, path, sums, backup_dir=None, margin_bytes=0)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM kaggle_crsp.daily")
        assert cur.fetchone()[0] == 0


def test_large_group_pilot_and_full(conn, tmp_path):
    count = 60_000
    rows = [row(i, datetime(1999, 1, 1), 11.0 if i % 2 else None) for i in range(count)]
    schema_path, sums = make_file(tmp_path, rows, row_group_size=count)
    pilot = load_crsp(conn, schema_path, sums, backup_dir=None, year=1999, margin_bytes=0)
    full = load_crsp(conn, schema_path, sums, backup_dir=None, margin_bytes=0)
    with conn.cursor() as cur:
        for load_id in (pilot, full):
            cur.execute(
                "SELECT count(*), sum(prc), count(shrcd), min(shrcd), max(shrcd) "
                "FROM kaggle_crsp.daily WHERE load_id=%s",
                (load_id,),
            )
            assert cur.fetchone() == (count, -5.0 * count, count // 2, 11, 11)


def test_pilot_filters_other_years_before_validation(conn, tmp_path):
    path, sums = make_file(
        tmp_path,
        [row(1, datetime(1999, 1, 1)), row(2, datetime(2000, 1, 1), 10.5)],
        row_group_size=2,
    )
    load_id = load_crsp(conn, path, sums, backup_dir=None, year=1999, margin_bytes=0)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM kaggle_crsp.daily WHERE load_id=%s", (load_id,))
        assert cur.fetchone()[0] == 1


def test_two_pilot_versions(conn, tmp_path):
    path, sums = make_file(tmp_path, [row(1, datetime(1999, 1, 1))])
    first = load_crsp(conn, path, sums, backup_dir=None, year=1999, margin_bytes=0)
    second = load_crsp(
        conn, path, sums, backup_dir=None, year=1999, parser_suffix="v2", margin_bytes=0
    )
    assert first != second
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM ref.loads WHERE load_id IN (%s, %s)", (first, second))
        assert cur.fetchall() == [("complete",), ("complete",)]


def test_fatal_data_marks_failed(conn, tmp_path):
    path, sums = make_file(tmp_path, [row(1, datetime(1999, 1, 1), 10.5)])
    with pytest.raises(LoadFailed):
        load_crsp(conn, path, sums, backup_dir=None, margin_bytes=0)
    with conn.cursor() as cur:
        cur.execute("SELECT status, checks->>'failure' FROM ref.loads")
        status, reason = cur.fetchone()
        assert status == "failed" and "Invalid shrcd" in reason


def test_reconciliation_failure_marks_failed(conn, tmp_path, monkeypatch):
    import ggTrader.data.kaggle_import.crsp as crsp

    path, sums = make_file(tmp_path, [row(1, datetime(1999, 1, 1))])
    original = crsp._write_chunk

    def wrong_count(*args, **kwargs):
        original(*args, **kwargs)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ref.loads SET rows_parsed=rows_parsed+1 WHERE load_id=%s", (args[1],)
            )
        conn.commit()

    monkeypatch.setattr(crsp, "_write_chunk", wrong_count)
    with pytest.raises(LoadFailed, match="row count"):
        load_crsp(conn, path, sums, backup_dir=None, margin_bytes=0)
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM ref.loads")
        assert cur.fetchone()[0] == "failed"


def test_recompression_reclaims_space(tmp_path):
    count = 60_000
    path, sums = make_file(
        tmp_path, [row(i, datetime(1999, 1, 1)) for i in range(count)], row_group_size=count
    )
    sizes = []
    for reclaim in (False, True):
        with scratch_database() as url, closing(psycopg2.connect(url)) as connection:
            apply_ddl(connection)
            load_crsp(connection, path, sums, backup_dir=None, year=1999, margin_bytes=0)
            assert compress_kaggle_chunks(connection, reclaim=reclaim) == 1
            load_crsp(
                connection,
                path,
                sums,
                backup_dir=None,
                year=1999,
                parser_suffix="v2",
                margin_bytes=0,
            )
            assert compress_kaggle_chunks(connection, reclaim=reclaim) == 1
            with connection.cursor() as cur:
                cur.execute(
                    "SELECT hypertable_size('kaggle_crsp.daily'), count(*) FROM kaggle_crsp.daily"
                )
                size, rows = cur.fetchone()
                assert rows == 2 * count
                sizes.append(size)
    assert sizes[1] < sizes[0]


def test_double_values_survive_copy_exactly(conn, tmp_path):
    values = [0.1234567890123456, 1 / 3, 123456789.12345679, 2.2250738585072014e-308, 1e-10]
    rows = []
    for i, value in enumerate(values):
        item = row(2000000000 + i, datetime(1999, 1, 1))
        item.update(prc=value, ret=value, shrout=3.4e7, vwretd=0.0004785)
        rows.append(item)
    special = row(2000000005, datetime(1999, 1, 1))
    special.update(prc=-18.25, ret=None, shrout=0.0, vwretd=0.0)
    rows.append(special)
    path, sums = make_file(tmp_path, rows, row_group_size=len(rows))
    load_id = load_crsp(conn, path, sums, backup_dir=None, margin_bytes=0)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT permno, prc, ret, shrout, vwretd FROM kaggle_crsp.daily "
            "WHERE load_id=%s ORDER BY permno",
            (load_id,),
        )
        assert cur.fetchall() == [
            (2000000000 + i, value, value, 3.4e7, 0.0004785) for i, value in enumerate(values)
        ] + [(2000000005, -18.25, None, 0.0, 0.0)]
