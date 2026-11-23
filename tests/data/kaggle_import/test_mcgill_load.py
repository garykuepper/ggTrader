"""McGill import tests against a scratch database."""

import csv
import hashlib
from datetime import date

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
        for _ in range(10):
            writer.writerow(
                {
                    "date": "20000131",
                    "ret_eom": "20000131",
                    "size_port": "B",
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
        mcgill.load_mcgill(conn, path, sums, backup_dir=None, db_free_path=tmp_path)


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
        mcgill.load_mcgill(conn, path, sums, backup_dir=None, db_free_path=tmp_path, batch_rows=4)
    monkeypatch.setattr(mcgill, "commit_chunk", original)
    load = mcgill.load_mcgill(
        conn, path, sums, backup_dir=None, db_free_path=tmp_path, batch_rows=4
    )
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
        mcgill.load_mcgill(conn, path, sums, backup_dir=None, db_free_path=tmp_path)


def test_resume_changed_batch_size_refused(conn, tmp_path, monkeypatch):
    path, sums = make_file(tmp_path)
    original = mcgill.commit_chunk

    def interrupt(*args, **kwargs):
        if args[2] == "batch-001":
            raise RuntimeError("interrupt")
        return original(*args, **kwargs)

    monkeypatch.setattr(mcgill, "commit_chunk", interrupt)
    with pytest.raises(RuntimeError):
        mcgill.load_mcgill(conn, path, sums, backup_dir=None, db_free_path=tmp_path, batch_rows=4)
    monkeypatch.setattr(mcgill, "commit_chunk", original)
    with pytest.raises(LoadFailed, match="batch size or file changed"):
        mcgill.load_mcgill(conn, path, sums, backup_dir=None, db_free_path=tmp_path, batch_rows=5)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM kaggle_chars.mcgill")
        assert cur.fetchone()[0] == 4


def test_large_integer_refused(conn, tmp_path):
    path, sums = make_file(tmp_path, "9007199254740993")
    with pytest.raises(LoadFailed, match="row 2"):
        mcgill.load_mcgill(conn, path, sums, backup_dir=None, db_free_path=tmp_path)


def test_null_ticker_rows_kept(conn, tmp_path):
    path, sums = make_file(tmp_path)
    load = mcgill.load_mcgill(conn, path, sums, backup_dir=None, db_free_path=tmp_path)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM kaggle_chars.mcgill WHERE load_id=%s "
            "AND stock_ticker IS NULL AND cusip IS NULL",
            (load,),
        )
        assert cur.fetchone()[0] == 10
        cur.execute("SELECT rows_accepted FROM ref.loads WHERE load_id=%s", (load,))
        assert cur.fetchone()[0] == 10


def test_row_count_reconciles_and_load_completes(conn, tmp_path):
    path, sums = make_file(tmp_path)
    load = mcgill.load_mcgill(
        conn, path, sums, backup_dir=None, db_free_path=tmp_path, batch_rows=4
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status, rows_parsed, rows_accepted, checks FROM ref.loads WHERE load_id=%s",
            (load,),
        )
        status, parsed, accepted, checks = cur.fetchone()
        assert (status, parsed, accepted) == ("complete", 10, 10)
        assert checks["expected_chunks"] == ["batch-000", "batch-001", "batch-002"]


def test_non_finite_double_becomes_null_and_is_counted(conn, tmp_path):
    path, sums = make_file(tmp_path)
    contents = path.read_text().replace("NaN", "", 9)
    path.write_text(contents)
    sums.write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    load = mcgill.load_mcgill(conn, path, sums, backup_dir=None, db_free_path=tmp_path)
    with conn.cursor() as cur:
        cur.execute("SELECT count(prc) FROM kaggle_chars.mcgill WHERE load_id=%s", (load,))
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT checks->>'nulled_values' FROM ref.loads WHERE load_id=%s", (load,))
        assert cur.fetchone()[0] == "1"


def test_cli_requires_yes(conn, tmp_path, monkeypatch, capsys):
    import argparse

    from ggTrader.cli.cmd_db import run_db
    from ggTrader.data import universe_db

    path, sums = make_file(tmp_path)

    class ConnectionProxy:
        def __getattr__(self, name):
            return getattr(conn, name)

        def close(self):
            pass

    monkeypatch.setattr(universe_db, "connect", lambda url: ConnectionProxy())
    monkeypatch.setenv("TEST_IMPORT_URL", "postgresql://unused/db")
    args = argparse.Namespace(
        db_command="import",
        import_command="mcgill",
        file=path,
        sums=sums,
        backup_dir=None,
        db_free_path=tmp_path,
        db_url_env="TEST_IMPORT_URL",
        use_configured_db=False,
        yes=False,
        no_backup_check=False,
    )
    run_db(args)
    assert "Would import" in capsys.readouterr().out
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ref.loads")
        assert cur.fetchone()[0] == 0
    conn.commit()
    args.yes = True
    run_db(args)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM ref.loads WHERE status='complete'")
        assert cur.fetchone()[0] == 1


@pytest.mark.parametrize("column", ["date", "ret_eom"])
def test_invalid_date_refused(conn, tmp_path, column):
    path, sums = make_file(tmp_path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows[0][column] = "20001341"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[name for name, _ in MCGILL_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
    sums.write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    with pytest.raises(LoadFailed, match=f"Invalid {column} at row 2"):
        mcgill.load_mcgill(conn, path, sums, backup_dir=None, db_free_path=tmp_path)


def test_real_date_and_size_port_format(conn, tmp_path):
    path, sums = make_file(tmp_path)
    load = mcgill.load_mcgill(conn, path, sums, backup_dir=None, db_free_path=tmp_path)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT date, ret_eom, size_port FROM kaggle_chars.mcgill WHERE load_id=%s LIMIT 1",
            (load,),
        )
        assert cur.fetchone() == (date(2000, 1, 31), date(2000, 1, 31), "B")


def test_reconciliation_failure_marks_failed(conn, tmp_path, monkeypatch):
    path, sums = make_file(tmp_path)
    original = mcgill.commit_chunk

    def wrong_count(*args, **kwargs):
        result = original(*args, **kwargs)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE ref.loads SET rows_parsed=rows_parsed+1 WHERE load_id=%s", (args[1],)
            )
        conn.commit()
        return result

    monkeypatch.setattr(mcgill, "commit_chunk", wrong_count)
    with pytest.raises(LoadFailed, match="row count"):
        mcgill.load_mcgill(conn, path, sums, backup_dir=None, db_free_path=tmp_path)
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM ref.loads")
        assert cur.fetchone()[0] == "failed"


def test_double_values_survive_copy_exactly(conn, tmp_path):
    values = [0.1234567890123456, 1 / 3, 123456789.12345679, 2.2250738585072014e-308, 1e-10]
    path = tmp_path / "precision.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[name for name, _ in MCGILL_COLUMNS])
        writer.writeheader()
        for i, value in enumerate(values):
            writer.writerow(
                {
                    "date": "20000131",
                    "ret_eom": "20000131",
                    "size_port": "B",
                    "permno": 2000000000 + i,
                    "prc": repr(value),
                    "rf": repr(value),
                    "market_equity": repr(value),
                }
            )
        writer.writerow(
            {
                "date": "20000131",
                "ret_eom": "20000131",
                "size_port": "B",
                "permno": 2000000005,
                "prc": "-18.25",
                "rf": "",
                "market_equity": "0",
            }
        )
    sums = tmp_path / "sums"
    sums.write_text(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    load_id = mcgill.load_mcgill(conn, path, sums, backup_dir=None, db_free_path=tmp_path)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT permno, prc, rf, market_equity FROM kaggle_chars.mcgill "
            "WHERE load_id=%s ORDER BY permno",
            (load_id,),
        )
        assert cur.fetchall() == [
            (2000000000 + i, value, value, value) for i, value in enumerate(values)
        ] + [(2000000005, -18.25, None, 0.0)]


def test_space_uses_database_path(tmp_path, monkeypatch):
    from types import SimpleNamespace

    path, sums = make_file(tmp_path)
    seen = []
    monkeypatch.setattr(
        "ggTrader.data.kaggle_import.guards.shutil.disk_usage",
        lambda p: (seen.append(p), SimpleNamespace(free=0))[1],
    )
    target = tmp_path / "database"
    with pytest.raises(LoadFailed, match="Insufficient"):
        mcgill.load_mcgill(None, path, sums, backup_dir=None, db_free_path=target)
    assert seen == [target]


def test_space_failure_between_batches_is_resumable(conn, tmp_path, monkeypatch):
    path, sums = make_file(tmp_path)
    calls = iter([True, True, False])
    monkeypatch.setattr(mcgill, "free_space_ok", lambda *a, **kw: next(calls))
    with pytest.raises(LoadFailed, match="Insufficient"):
        mcgill.load_mcgill(conn, path, sums, backup_dir=None, db_free_path=tmp_path, batch_rows=4)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM kaggle_chars.mcgill")
        assert cur.fetchone()[0] == 4
    monkeypatch.setattr(mcgill, "free_space_ok", lambda *a, **kw: True)
    load = mcgill.load_mcgill(
        conn, path, sums, backup_dir=None, db_free_path=tmp_path, batch_rows=4
    )
    with conn.cursor() as cur:
        cur.execute("SELECT status, rows_parsed FROM ref.loads WHERE load_id=%s", (load,))
        assert cur.fetchone() == ("complete", 10)
