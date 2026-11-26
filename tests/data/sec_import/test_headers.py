"""Tests for SEC filing headers."""

import hashlib
from datetime import date, datetime, timezone

import pytest

from ggTrader.data.sec_import.cache import CacheEntry
from ggTrader.data.sec_import.headers import (
    HeaderIncomplete,
    load_headers,
    parse_header,
    read_header_text,
)

HEADER = """<SEC-HEADER>file
ACCESSION NUMBER: 123
CONFORMED SUBMISSION TYPE: 8-K
FILED AS OF DATE: 20001211
ITEM INFORMATION: Item 1
ITEM INFORMATION:
FILER:
 CENTRAL INDEX KEY: 00042
</SEC-HEADER>
"""


def test_parse_single_company_8k_with_items():
    row = parse_header(HEADER)[0]
    assert (row.accession, row.cik, row.form_type, row.items) == ("123", 42, "8-K", ("Item 1",))


def test_parse_two_companies_gives_two_rows_same_accession():
    rows = parse_header(
        HEADER.replace(
            "</SEC-HEADER>", " CENTRAL INDEX KEY: 43\n CENTRAL INDEX KEY: 42\n</SEC-HEADER>"
        )
    )
    assert [(r.accession, r.cik) for r in rows] == [("123", 42), ("123", 43)]


def test_parse_filed_date_formats_yyyymmdd():
    assert parse_header(HEADER)[0].filed_date == date(2000, 12, 11)


def test_truncated_header_raises():
    with pytest.raises(HeaderIncomplete):
        parse_header(HEADER.replace("</SEC-HEADER>", ""))


def test_quoted_closing_tag_inside_text_is_not_the_end():
    with pytest.raises(HeaderIncomplete):
        parse_header(HEADER.replace("</SEC-HEADER>", '"</SEC-HEADER>"'))


def test_missing_cik_raises():
    with pytest.raises(HeaderIncomplete):
        parse_header(HEADER.replace(" CENTRAL INDEX KEY: 00042", ""))


def test_read_header_text_stops_at_closing_line(tmp_path):
    path = tmp_path / "body"
    path.write_bytes((HEADER + "body").encode("latin-1"))
    assert read_header_text(path) == HEADER


def test_load_headers_quarantines_incomplete_header(tmp_path, monkeypatch):
    from ggTrader.data.sec_import import headers

    path = tmp_path / "body"
    data = b"incomplete"
    path.write_bytes(data)
    entry = CacheEntry(
        "a",
        "url",
        hashlib.sha256(data).hexdigest(),
        datetime.now(timezone.utc),
        200,
        len(data),
        path,
        "filing_text",
    )
    monkeypatch.setattr(headers, "backup_fresh", lambda *_: None)
    monkeypatch.setattr(headers, "free_space_ok", lambda *a, **k: True)
    monkeypatch.setattr(headers, "begin_load", lambda *a, **k: 1)
    monkeypatch.setattr(headers, "record_files", lambda *a, **k: None)
    monkeypatch.setattr(headers, "complete_load", lambda *a, **k: None)
    calls = []
    monkeypatch.setattr(headers, "commit_chunk", lambda *a, **k: calls.append((a, k)))
    assert load_headers(None, [entry], backup_dir=tmp_path, db_free_path=tmp_path) == 1
    assert calls[0][0][3] == 0
    assert calls[0][1]["quarantine"][0][0] == str(path)


@pytest.mark.integration
@pytest.mark.db
def test_load_headers_writes_rows_and_quarantines(tmp_path, monkeypatch):
    import psycopg2

    from ggTrader.data.kaggle_import.ddl import apply_ddl as apply_kaggle_ddl
    from ggTrader.data.sec_import import headers
    from ggTrader.data.sec_import.ddl import apply_ddl
    from tests.data.scratch_db import scratch_database

    entries = []
    for i, body in enumerate((HEADER.replace("ITEM INFORMATION: Item 1", ""), "incomplete")):
        path = tmp_path / str(i)
        data = body.encode("latin-1")
        path.write_bytes(data)
        entries.append(
            CacheEntry(
                str(i),
                str(i),
                hashlib.sha256(data).hexdigest(),
                datetime.now(timezone.utc),
                200,
                len(data),
                path,
                "filing_text",
            )
        )
    monkeypatch.setattr(headers, "backup_fresh", lambda _: None)
    monkeypatch.setattr(headers, "free_space_ok", lambda *a, **k: True)
    with scratch_database() as url, psycopg2.connect(url) as conn:
        apply_kaggle_ddl(conn)
        apply_ddl(conn)
        load_id = load_headers(conn, entries, backup_dir=tmp_path, db_free_path=tmp_path)
        with conn.cursor() as cur:
            cur.execute("SELECT cik, items FROM sec.filing_headers WHERE load_id=%s", (load_id,))
            assert cur.fetchall() == [(42, [])]
            cur.execute(
                "SELECT rows_parsed, rows_accepted, rows_quarantined "
                "FROM ref.loads WHERE load_id=%s",
                (load_id,),
            )
            assert cur.fetchone() == (2, 1, 1)
            cur.execute("SELECT row_ref FROM ref.quarantine WHERE load_id=%s", (load_id,))
            assert cur.fetchone() == (str(entries[1].body_path),)
