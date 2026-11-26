"""Tests for quarterly SEC indexes."""

from datetime import date
from pathlib import Path
from zipfile import ZipFile

import pytest

from ggTrader.data.sec_import import index

HEADER = "header\nCIK|Company Name|Form Type|Date Filed|Filename\n-------\n"


def test_parse_valid_lines() -> None:
    rows, bad = index.parse_master_idx(HEADER + "123|Café|10-K|2005-04-01|file\n", "2005Q2")
    assert rows == [("2005Q2", 123, "Café", "10-K", date(2005, 4, 1), "file")]
    assert not bad


@pytest.mark.parametrize(
    "line",
    [
        "1|Co|10-K|bad|file",
        "1|Co",
        "1|Extra|pipe|10-K|2005-01-01|file",
        "x|Co|10-K|2005-01-01|file",
        "1||10-K|2005-01-01|file",
    ],
)
def test_bad_lines_are_quarantined(line: str) -> None:
    rows, bad = index.parse_master_idx(HEADER + line, "2005Q2")
    assert not rows
    assert len(bad) == 1 and bad[0][1]["line"] == line and bad[0][2]


def test_missing_separator() -> None:
    with pytest.raises(ValueError, match="2005Q2"):
        index.parse_master_idx("header", "2005Q2")


def test_quarter_of_urls() -> None:
    assert index.quarter_of("https://example.org/full-index/2005/QTR2/master.zip") == "2005Q2"
    with pytest.raises(ValueError):
        index.quarter_of("https://example.org/full-index/2005/QTR5/master.zip")


def test_read_master_zip_reads_member(tmp_path: Path) -> None:
    path = tmp_path / "x.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("master.idx", "Café".encode("latin-1"))
    assert index.read_master_zip(path) == "Café"


def test_copy_escape() -> None:
    assert index._copy_escape('a\tb\nc\\d"|é') == 'a\\tb\\nc\\\\d"|é'


def test_load_index_skips_quarters_not_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    from datetime import datetime, timezone

    from ggTrader.data.sec_import.cache import CacheEntry

    path = tmp_path / "body"
    with ZipFile(path, "w") as archive:
        archive.writestr("master.idx", HEADER + "1|Co|10-K|2005-01-01|file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def entry(q: int) -> CacheEntry:
        return CacheEntry(
            str(q),
            f"https://x/full-index/2005/QTR{q}/master.zip",
            digest,
            datetime.now(timezone.utc),
            200,
            path.stat().st_size,
            path,
            "quarterly_index",
        )

    monkeypatch.setattr(index, "backup_fresh", lambda _: None)
    monkeypatch.setattr(index, "free_space_ok", lambda *a, **k: True)
    monkeypatch.setattr(index, "begin_load", lambda *a, **k: 1)
    monkeypatch.setattr(index, "record_files", lambda *a, **k: None)
    keys = []
    monkeypatch.setattr(index, "commit_chunk", lambda conn, lid, key, *a, **k: keys.append(key))
    monkeypatch.setattr(index, "complete_load", lambda *a, **k: None)
    index.load_index(
        None,
        [entry(1), entry(2)],
        backup_dir=tmp_path,
        db_free_path=tmp_path,
        only_quarters=["2005Q2"],
    )
    assert keys == ["2005Q2"]


@pytest.mark.integration
@pytest.mark.db
def test_load_index_writes_rows_quarantine_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    from datetime import datetime, timezone

    import psycopg2

    from ggTrader.data.kaggle_import.ddl import apply_ddl as apply_kaggle_ddl
    from ggTrader.data.sec_import.cache import CacheEntry
    from ggTrader.data.sec_import.ddl import apply_ddl
    from tests.data.scratch_db import scratch_database

    path = tmp_path / "body"
    with ZipFile(path, "w") as archive:
        archive.writestr("master.idx", HEADER + "1|Café|10-K|2005-01-01|file\n2|Bad|10-K|bad|file")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = CacheEntry(
        "x",
        "https://x/full-index/2005/QTR1/master.zip",
        digest,
        datetime.now(timezone.utc),
        200,
        path.stat().st_size,
        path,
        "quarterly_index",
    )
    monkeypatch.setattr(index, "backup_fresh", lambda _: None)
    monkeypatch.setattr(index, "free_space_ok", lambda *a, **k: True)
    original = index.complete_load

    def interrupted(*args: object, **kwargs: object) -> None:
        raise RuntimeError("interrupted")

    with scratch_database() as url, psycopg2.connect(url) as conn:
        apply_kaggle_ddl(conn)
        apply_ddl(conn)
        monkeypatch.setattr(index, "complete_load", interrupted)
        with pytest.raises(RuntimeError, match="interrupted"):
            index.load_index(conn, [entry], backup_dir=tmp_path, db_free_path=tmp_path)
        monkeypatch.setattr(index, "complete_load", original)
        load_id = index.load_index(conn, [entry], backup_dir=tmp_path, db_free_path=tmp_path)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT cik, company_name, date_filed FROM sec.index_entries WHERE load_id=%s",
                (load_id,),
            )
            assert cur.fetchall() == [(1, "Café", date(2005, 1, 1))]
            cur.execute(
                "SELECT rows_parsed, rows_accepted, rows_quarantined "
                "FROM ref.loads WHERE load_id=%s",
                (load_id,),
            )
            assert cur.fetchone() == (2, 1, 1)
            cur.execute("SELECT count(*) FROM ref.quarantine WHERE load_id=%s", (load_id,))
            assert cur.fetchone() == (1,)
