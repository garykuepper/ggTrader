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
        "1|Co||2005-01-01|file",
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
        archive.writestr(
            "master.idx",
            (
                HEADER
                + "1|Café|10-K|2005-01-01|file\n"
                + "3|A\tB\\C\\N D|10-K|2005-01-02|file\n"
                + "4||10-K|2005-01-03|file\n"
                + "2|Bad|10-K|bad|file"
            ).encode("latin-1"),
        )
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
                "SELECT cik, company_name, date_filed FROM sec.index_entries "
                "WHERE load_id=%s ORDER BY cik",
                (load_id,),
            )
            assert cur.fetchall() == [
                (1, "Café", date(2005, 1, 1)),
                (3, "A\tB\\C\\N D", date(2005, 1, 2)),
                (4, "", date(2005, 1, 3)),
            ]
            cur.execute(
                "SELECT rows_parsed, rows_accepted, rows_quarantined "
                "FROM ref.loads WHERE load_id=%s",
                (load_id,),
            )
            assert cur.fetchone() == (4, 3, 1)
            cur.execute("SELECT count(*) FROM ref.quarantine WHERE load_id=%s", (load_id,))
            assert cur.fetchone() == (1,)


def test_parse_keeps_blank_company_name_and_quarantines_blank_form() -> None:
    text = (
        "x\n-----\n1||10-K|2016-02-26|edgar/data/1/a.txt\n2|B CO||2016-02-26|edgar/data/2/b.txt\n"
    )
    rows, bad = index.parse_master_idx(text, "2016Q1")
    assert rows == [("2016Q1", 1, "", "10-K", date(2016, 2, 26), "edgar/data/1/a.txt")]
    assert [(ref, reason) for ref, _, reason in bad] == [("4", "empty CIK, form, date or filename")]


def test_copy_escape_handles_carriage_return_and_null_marker() -> None:
    assert index._copy_escape("a\rb") == "a\\rb"
    assert index._copy_escape("\\N") == "\\\\N"


def test_next_line_character_inside_a_company_name_does_not_split_the_row() -> None:
    text = "x\n-----\n1|A\x85B CO|10-K|2016-02-26|edgar/data/1/a.txt\r\n"
    rows, bad = index.parse_master_idx(text, "2016Q1")
    assert [row[2] for row in rows] == ["A\x85B CO"] and not bad


def test_nul_character_is_quarantined_and_kept_visible() -> None:
    text = "x\n-----\n1|A\x00B|10-K|2016-02-26|edgar/data/1/a.txt\n"
    rows, bad = index.parse_master_idx(text, "2016Q1")
    assert not rows
    assert bad[0][1]["line"] == "1|A\\x00B|10-K|2016-02-26|edgar/data/1/a.txt"
    assert "NUL" in bad[0][2]


def test_requesting_a_quarter_that_is_not_there_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ggTrader.data.kaggle_import.loadlog import LoadFailed

    monkeypatch.setattr(index, "backup_fresh", lambda _: None)
    with pytest.raises(LoadFailed, match="2005Q2"):
        index.load_index(
            None, [], backup_dir=tmp_path, db_free_path=tmp_path, only_quarters=["2005Q2"]
        )


def _entry_for(tmp_path: Path, quarter: str, status: int = 200):
    import hashlib
    from datetime import datetime, timezone

    from ggTrader.data.sec_import.cache import CacheEntry

    path = tmp_path / f"{quarter}.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("master.idx", HEADER + "1|Co|10-K|2005-01-01|file")
    return CacheEntry(
        quarter,
        f"https://x/full-index/{quarter[:4]}/QTR{quarter[-1]}/master.zip",
        hashlib.sha256(path.read_bytes()).hexdigest(),
        datetime.now(timezone.utc),
        status,
        path.stat().st_size,
        path,
        "quarterly_index",
    )


def test_a_cached_error_response_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ggTrader.data.kaggle_import.loadlog import LoadFailed

    monkeypatch.setattr(index, "backup_fresh", lambda _: None)
    with pytest.raises(LoadFailed, match="status 404"):
        index.load_index(
            None,
            [_entry_for(tmp_path, "2005Q2", status=404)],
            backup_dir=tmp_path,
            db_free_path=tmp_path,
        )


def test_free_space_is_checked_again_before_each_quarter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from ggTrader.data.kaggle_import.loadlog import LoadFailed

    calls = []

    def space(path, needed, **kwargs):
        calls.append(needed)
        return len(calls) < 3  # fine at the start and for the first quarter, then full

    monkeypatch.setattr(index, "backup_fresh", lambda _: None)
    monkeypatch.setattr(index, "free_space_ok", space)
    monkeypatch.setattr(index, "begin_load", lambda *a, **k: 1)
    monkeypatch.setattr(index, "record_files", lambda *a, **k: None)
    monkeypatch.setattr(index, "commit_chunk", lambda *a, **k: True)
    entries = [_entry_for(tmp_path, "2005Q1"), _entry_for(tmp_path, "2005Q2")]
    with pytest.raises(LoadFailed, match="before 2005Q2"):
        index.load_index(None, entries, backup_dir=tmp_path, db_free_path=tmp_path)
    assert len(calls) == 3 and calls[0] == index.EXPANSION * sum(e.bytes for e in entries)
