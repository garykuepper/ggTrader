"""McGill import tests against a scratch database."""

import csv
import hashlib

import psycopg2
import pytest

from ggTrader.data.kaggle_import import mcgill
from ggTrader.data.kaggle_import.ddl import MCGILL_COLUMNS, apply_ddl
from ggTrader.data.kaggle_import.guards import HashMismatch
from ggTrader.data.kaggle_import.loadlog import LoadFailed
from tests.data.scratch_db import scratch_database

pytestmark = [pytest.mark.integration, pytest.mark.db]


@pytest.fixture
def conn(monkeypatch):
    monkeypatch.setattr(mcgill, "free_space_ok", lambda *a, **k: True)
    with scratch_database() as url, psycopg2.connect(url) as connection:
        apply_ddl(connection)
        yield connection


def make_file(tmp_path, permno="123.0"):
    path = tmp_path / "sample.csv"
    names = [name for name, _ in MCGILL_COLUMNS]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for i in range(10):
            writer.writerow(
                {
                    "date": f"2020-01-{i + 1:02d}",
                    "ret_eom": "2020-01-31",
                    "permno": permno,
                    "prc": "NaN",
                }
            )
    sums = tmp_path / "sums"
    sums.write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    return path, sums


def test_declared_types_not_inferred(conn, tmp_path):
    path, sums = make_file(tmp_path, "123.5")
    with pytest.raises(LoadFailed, match="row 2"):
        mcgill.load_mcgill(conn, path, sums, backup_dir=None)


def test_batch_resume(conn, tmp_path, monkeypatch):
    path, sums = make_file(tmp_path)
    original = mcgill.commit_chunk
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("interrupt")
        return original(*args, **kwargs)

    monkeypatch.setattr(mcgill, "commit_chunk", interrupt)
    with pytest.raises(RuntimeError):
        mcgill.load_mcgill(conn, path, sums, backup_dir=None, batch_rows=4)
    monkeypatch.setattr(mcgill, "commit_chunk", original)
    load = mcgill.load_mcgill(conn, path, sums, backup_dir=None, batch_rows=4)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*), min(permno), count(stock_ticker), count(prc) "
            "FROM kaggle_chars.mcgill WHERE load_id=%s",
            (load,),
        )
        assert cur.fetchone() == (10, 123, 0, 0)
        cur.execute("SELECT status, rows_parsed, checks FROM ref.loads WHERE load_id=%s", (load,))
        status, count, checks = cur.fetchone()
        assert (status, count, checks["nulled_values"]) == ("complete", 10, 10)
        cur.execute("SELECT count(*) FROM ref.load_chunks WHERE load_id=%s", (load,))
        assert cur.fetchone()[0] == 3


def test_hash_mismatch_refuses(conn, tmp_path):
    path, sums = make_file(tmp_path)
    sums.write_text("0  sample.csv\n")
    with pytest.raises(HashMismatch):
        mcgill.load_mcgill(conn, path, sums, backup_dir=None)
