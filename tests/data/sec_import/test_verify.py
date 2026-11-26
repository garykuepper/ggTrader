"""Test the database integrity checks without a configured database."""

import argparse
import hashlib
import json
from collections import deque
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import psycopg2
import psycopg2.extras
import pytest
from psycopg2.extensions import connection

from ggTrader.cli.cmd_db import register_db_parser, run_db
from ggTrader.data.kaggle_import.ddl import apply_ddl as apply_kaggle_ddl
from ggTrader.data.sec_import.ddl import apply_ddl as apply_sec_ddl
from tests.data.scratch_db import scratch_database

HEADER_ACCESSION = "0000000001-05-000001"
INDEX_ACCESSION = "0000000002-05-000002"
MISSING_ACCESSION = "0000000003-05-000003"
LAST_DATE = date(2005, 6, 1)


class FakeConnection:
    """Return canned rows and reject unexpected database commands."""

    def __init__(self, *responses: tuple[str, list[tuple] | Exception]) -> None:
        self.responses = deque(responses)
        self.rows: list[tuple] = []
        self.params: list[tuple] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.readonly = False

    def cursor(self) -> "FakeConnection":
        return self

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, query: str, params: tuple = ()) -> None:
        assert self.responses, f"Unexpected SQL: {query}"
        fragment, rows = self.responses.popleft()
        assert fragment in " ".join(query.split())
        self.params.append(params)
        if isinstance(rows, Exception):
            raise rows
        self.rows = rows

    def fetchall(self) -> list[tuple]:
        return self.rows

    def fetchone(self) -> tuple:
        [row] = self.rows
        return row

    def rollback(self) -> None:
        self.rollbacks += 1

    def commit(self) -> None:
        self.commits += 1

    def set_session(self, *, readonly: bool) -> None:
        self.readonly = readonly

    def close(self) -> None:
        self.closed = True


def ordinary_responses() -> list[tuple[str, list[tuple] | Exception]]:
    """A completed database with no imported rows, including no labels."""
    return [
        ("sum(c.rows)", []),
        ("FROM ref.loads", []),
        ("FROM ref.load_files", []),
        ("FROM ref.loads", []),
        ("ref.load_coverage", []),
        ("market.stock_spans", []),
        ("count(*) FROM sec.death_labels", [(0,)]),
        ("unnest", []),
    ]


def test_cli_help_and_no_write_without_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ggTrader.data import universe_db

    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    base = ["db", "verify-data"]
    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args([*base, "--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--deep", "--record", "--out", "--db-url-env", "--use-configured-db"):
        assert option in help_text
    assert "scripts/run_capped.sh" in help_text
    with pytest.raises(SystemExit) as no_target:
        parser.parse_args(base)
    assert no_target.value.code == 2

    conn = FakeConnection(("current_database()", [("scratch",)]), *ordinary_responses())
    monkeypatch.setenv("VERIFY_TEST_URL", "postgresql://localhost/scratch")
    monkeypatch.setattr(universe_db, "connect", lambda _url: conn)
    output = tmp_path / "report.json"
    run_db(parser.parse_args([*base, "--db-url-env", "VERIFY_TEST_URL", "--out", str(output)]))

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert set(report["checks"]) == {
        "load_row_counts_match_log",
        "load_files_exist_with_recorded_size",
        "no_incomplete_loads",
        "every_complete_load_has_one_coverage_row",
        "labels_match_stock_spells_in_release",
        "cited_accessions_exist",
    }
    assert "6 of 6 checks passed" in capsys.readouterr().out
    assert not conn.responses
    assert conn.commits == 0
    assert conn.readonly  # without --record the session cannot write
    assert conn.closed


def test_cli_records_failed_deep_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ggTrader.data import universe_db

    parser = argparse.ArgumentParser()
    register_db_parser(parser.add_subparsers(dest="command"))
    responses = ordinary_responses()
    responses[0] = ("sum(c.rows)", [(1, 100, 99)])
    conn = FakeConnection(
        ("current_database()", [("scratch",)]),
        *responses,
        ("FROM ref.load_files", []),
        ("FROM ref.loads", []),
        ("INSERT INTO ref.integrity_checks", [(42,)]),
    )
    monkeypatch.setenv("VERIFY_TEST_URL", "postgresql://localhost/scratch")
    monkeypatch.setattr(universe_db, "connect", lambda _url: conn)
    args = parser.parse_args(
        ["db", "verify-data", "--db-url-env", "VERIFY_TEST_URL", "--deep", "--record"]
    )
    with pytest.raises(SystemExit) as exc:
        run_db(args)

    assert exc.value.code == 1
    assert not conn.responses
    mode, ok, payload = conn.params[-1]
    assert (mode, ok) == ("deep", False)
    assert payload.adapted["checks"]["load_row_counts_match_log"]["ok"] is False
    assert conn.commits == 1
    assert not conn.readonly  # --record needs to write its one row
    assert conn.closed
    output = capsys.readouterr().out
    assert "heavy" in output.lower()
    assert "7 of 8 checks passed" in output


@pytest.mark.parametrize(("logged", "expected_ok"), [(10, True), (9, False)])
def test_row_count_check_uses_chunk_log(logged: int, expected_ok: bool) -> None:
    from ggTrader.data.sec_import.verify import load_row_counts_match_log

    result = load_row_counts_match_log(FakeConnection(("sum(c.rows)", [(3, 10, logged)])))
    assert result["ok"] is expected_ok
    if not expected_ok:
        assert "load 3" in result["detail"]


def test_detects_changed_file_size(tmp_path: Path) -> None:
    from ggTrader.data.sec_import.verify import load_files_exist_with_recorded_size

    path = tmp_path / "source"
    path.write_bytes(b"12345")
    for size in (5, 4):
        conn = FakeConnection(
            ("FROM ref.loads", [(5, "sec-cache", "sec-cache", "complete", 1, True)]),
            ("FROM ref.load_files", [(5, str(path), "digest", size)]),
        )
        result = load_files_exist_with_recorded_size(conn)
        assert result["ok"] is (size == 5)
        if size == 4:
            assert "5 bytes" in result["detail"]
    path.unlink()
    conn = FakeConnection(
        ("FROM ref.loads", [(5, "sec-cache", "sec-cache", "complete", 1, True)]),
        ("FROM ref.load_files", [(5, str(path), "digest", 5)]),
    )
    assert not load_files_exist_with_recorded_size(conn)["ok"]


@pytest.mark.parametrize(
    ("parent_status", "parent_rows", "expected_ok"),
    [("complete", 1, True), ("running", 1, False), ("complete", 2, False), (None, 0, False)],
)
def test_derived_load_checked_against_parent(
    tmp_path: Path, parent_status: str | None, parent_rows: int, expected_ok: bool
) -> None:
    from ggTrader.data.sec_import.verify import load_files_exist_with_recorded_size

    path = tmp_path / "parent"
    path.touch()
    loads = [(2, "crsp-derived-bars", "load:1", "complete", 1, True)]
    if parent_status:
        loads.insert(0, (1, "crsp-daily-pilot", str(path), parent_status, parent_rows, True))
    report = load_files_exist_with_recorded_size(
        FakeConnection(("FROM ref.loads", loads), ("FROM ref.load_files", []))
    )
    assert report["ok"] is expected_ok
    assert "parent" in report["detail"]
    if expected_ok:
        assert "no recorded size or fingerprint" in report["detail"]


@pytest.mark.parametrize("running", [[], [(3, "running")]])
def test_incomplete_check_distinguishes_statuses(running: list[tuple]) -> None:
    from ggTrader.data.sec_import.verify import no_incomplete_loads

    report = no_incomplete_loads(
        FakeConnection(("FROM ref.loads", [(1, "failed"), (2, "withdrawn"), *running]))
    )
    assert report["ok"] is (not running)
    assert "failed" in report["detail"]
    assert "withdrawn" in report["detail"]


@pytest.mark.parametrize("missing", [[], [(3, 0)]])
def test_coverage_check_reports_missing_rows(missing: list[tuple]) -> None:
    from ggTrader.data.sec_import.verify import every_complete_load_has_one_coverage_row

    report = every_complete_load_has_one_coverage_row(
        FakeConnection(
            ("ref.load_coverage", [(load, count, "sec-index") for load, count in missing])
        )
    )
    assert report["ok"] is (not missing)
    if missing:
        assert "load 3" in report["detail"]


@pytest.mark.parametrize("mismatch", [[], [(2, 6, 12345, LAST_DATE, 55)]])
def test_label_check_reports_wrong_stock_spell(mismatch: list[tuple]) -> None:
    from ggTrader.data.sec_import.verify import labels_match_stock_spells_in_release

    report = labels_match_stock_spells_in_release(
        FakeConnection(
            ("market.stock_spans", mismatch), ("count(*) FROM sec.death_labels", [(60,)])
        )
    )
    assert report["ok"] is (not mismatch)
    if mismatch:
        assert all(text in report["detail"] for text in ("12345", "2005-06-01", "55"))


def test_accessions_search_index_once_for_header_misses() -> None:
    from ggTrader.data.sec_import.verify import cited_accessions_exist

    conn = FakeConnection(
        ("unnest", [(HEADER_ACCESSION,), (INDEX_ACCESSION,), (MISSING_ACCESSION,)]),
        ("FROM sec.filing_headers", [(HEADER_ACCESSION,)]),
        ("substring(e.filename", [(INDEX_ACCESSION,)]),
    )
    result = cited_accessions_exist(conn)
    assert result["ok"] is False
    assert MISSING_ACCESSION in result["detail"]
    assert INDEX_ACCESSION not in result["detail"]
    assert conn.params[-1] == ([INDEX_ACCESSION, MISSING_ACCESSION],)
    assert not conn.responses


def test_accessions_skip_index_when_headers_cover_every_citation() -> None:
    from ggTrader.data.sec_import.verify import cited_accessions_exist

    conn = FakeConnection(
        ("unnest", [(HEADER_ACCESSION,)]),
        ("FROM sec.filing_headers", [(HEADER_ACCESSION,)]),
    )
    assert cited_accessions_exist(conn)["ok"]
    assert not conn.responses


def test_accessions_report_first_fifty_and_total() -> None:
    from ggTrader.data.sec_import.verify import cited_accessions_exist

    citations = [f"0000000001-05-{number:06}" for number in range(51)]
    report = cited_accessions_exist(
        FakeConnection(
            ("unnest", [(citation,) for citation in reversed(citations)]),
            ("FROM sec.filing_headers", []),
            ("substring(e.filename", []),
        )
    )
    assert not report["ok"]
    assert all(citation in report["detail"] for citation in citations[:50])
    assert citations[50] not in report["detail"]
    assert "51 total" in report["detail"]


def test_deep_detects_changed_file_bytes(tmp_path: Path) -> None:
    from ggTrader.data.sec_import.verify import load_files_rehash

    path = tmp_path / "source"
    digest = hashlib.sha256(b"abc").hexdigest()
    for content in (b"abc", b"abd"):
        path.write_bytes(content)
        report = load_files_rehash(
            FakeConnection(("FROM ref.load_files", [(5, str(path), digest, 3)]))
        )
        assert report["ok"] is (content == b"abc")
        if content == b"abd":
            assert str(path) in report["detail"]


def test_deep_detects_altered_stored_row() -> None:
    from ggTrader.data.sec_import.verify import content_sha256_matches_stored_rows

    digest = hashlib.sha256(b"a:abc\nz:xyz").hexdigest()
    for content in ("abc", "bad"):
        conn = FakeConnection(
            ("FROM ref.loads", [(1, "sec-cache", 2, digest)]),
            ("count(*) FROM sec.fetched_files", [(2,)]),
            ("url_hash, content_sha256", [("z", "xyz"), ("a", content)]),
        )
        report = content_sha256_matches_stored_rows(conn)
        assert report["ok"] is (content == "abc")
        if content == "bad":
            assert "load 1" in report["detail"]


@pytest.mark.parametrize("actual", [1, 0])
@pytest.mark.parametrize(
    ("dataset", "table"),
    [
        ("crsp-daily", "kaggle_crsp.daily"),
        ("crsp-daily-pilot", "kaggle_crsp.daily"),
        ("crsp-derived-bars", "market.daily_bars"),
        ("mcgill-fiam", "kaggle_chars.mcgill"),
        ("sec-headers", "sec.filing_headers"),
        ("sec-labels:study", "sec.death_labels"),
    ],
)
def test_deep_counts_stored_rows_by_load(dataset: str, table: str, actual: int) -> None:
    from ggTrader.data.sec_import.verify import content_sha256_matches_stored_rows

    responses = [
        ("FROM ref.loads", [(1, dataset, 1, "digest")]),
        (f"count(*) FROM {table} WHERE load_id", [(actual,)]),
    ]
    if dataset.startswith("sec-labels"):
        responses.append(("source_row_sha256", []))
    conn = FakeConnection(*responses)
    report = content_sha256_matches_stored_rows(conn)
    assert report["ok"] is (actual == 1)
    assert table in report["detail"]
    assert "load 1" in report["detail"]
    assert "load_files_rehash" in report["detail"]
    assert conn.params[-1] == (1,)


@pytest.mark.parametrize("actual_quarter", ["2005Q2", "2005Q3"])
def test_deep_checks_each_index_quarter_even_if_total_matches(actual_quarter: str) -> None:
    from ggTrader.data.sec_import.verify import content_sha256_matches_stored_rows

    conn = FakeConnection(
        ("FROM ref.loads", [(8, "sec-index", 1, "digest")]),
        ("FROM sec.index_entries", [(actual_quarter, 1)]),
        ("FROM ref.load_chunks", [("2005Q2", 1)]),
    )
    report = content_sha256_matches_stored_rows(conn)
    assert report["ok"] is (actual_quarter == "2005Q2")
    assert "sec.index_entries" in report["detail"]
    if actual_quarter == "2005Q3":
        assert all(quarter in report["detail"] for quarter in ("2005Q2", "2005Q3"))


def test_run_checks_reports_error_and_continues() -> None:
    from ggTrader.data.sec_import.verify import run_checks

    responses = ordinary_responses()
    responses[0] = ("sum(c.rows)", ValueError("broken postgresql://user:fake@localhost/scratch"))
    conn = FakeConnection(*responses)
    report = run_checks(conn)
    assert not report["ok"]
    assert "ValueError" in report["checks"]["load_row_counts_match_log"]["detail"]
    assert "fake@" not in report["checks"]["load_row_counts_match_log"]["detail"]
    assert report["checks"]["cited_accessions_exist"]["detail"] == "no labels loaded"
    assert conn.rollbacks == 2  # one after the failed check, one when the run ends
    assert conn.commits == 0
    assert not conn.responses


@pytest.fixture
def verify_db() -> Iterator[connection]:
    """Apply both schemas in a disposable database."""
    with scratch_database() as url, psycopg2.connect(url) as conn:
        apply_kaggle_ddl(conn)
        apply_sec_ddl(conn)
        yield conn


def stage_load(
    conn: connection, dataset: str, source: Path, *, chunk: str = "one", coverage: bool = True
) -> int:
    """Stage one accepted row and the source bookkeeping that its loader would write."""
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO ref.loads (dataset, file_path, content_sha256, rows_accepted, status) "
            "VALUES (%s, %s, %s, 1, 'running') RETURNING load_id",
            (dataset, str(source), digest),
        )
        load_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO ref.load_chunks (load_id, chunk_key, rows) VALUES (%s, %s, 1)",
            (load_id, chunk),
        )
        if dataset.startswith("sec-"):
            cur.execute(
                "INSERT INTO ref.load_files (load_id, path, sha256, bytes) VALUES (%s, %s, %s, %s)",
                (load_id, str(source), digest, source.stat().st_size),
            )
        if coverage:
            cur.execute(
                "INSERT INTO ref.load_coverage (load_id, coverage, note) "
                "VALUES (%s, 'not_a_stock_source', 'scratch fixture')",
                (load_id,),
            )
        cur.execute("UPDATE ref.loads SET status='complete' WHERE load_id=%s", (load_id,))
    conn.commit()
    return load_id


@pytest.mark.integration
@pytest.mark.db
def test_ordinary_run_passes_on_clean_database(tmp_path: Path, verify_db: connection) -> None:
    from ggTrader.data.sec_import.verify import run_checks

    path = tmp_path / "source"
    path.write_text("fixture", encoding="utf-8")
    labels = stage_load(verify_db, "sec-labels:study", path)
    headers = stage_load(verify_db, "sec-headers", path)
    index = stage_load(verify_db, "sec-index", path, chunk="2005Q2")
    with verify_db.cursor() as cur:
        cur.execute("INSERT INTO ref.releases (status) VALUES ('building') RETURNING release_id")
        release = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO market.stock_spans (release_id, symbol, permno, last_date) "
            "VALUES (%s, 'OLD', 12345, %s)",
            (release, LAST_DATE),
        )
        cur.execute("UPDATE ref.releases SET status='published' WHERE release_id=%s", (release,))
        fields = {"permno": "12345"}
        fingerprint = hashlib.sha256(
            json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        cur.execute(
            "INSERT INTO sec.death_labels (load_id, study, method_version, permno, last_date, "
            "release_id, accessions, source_fields, source_row_sha256) "
            "VALUES (%s, 'study', 'v1', 12345, %s, %s, %s, %s, %s)",
            (
                labels,
                LAST_DATE,
                release,
                [HEADER_ACCESSION, INDEX_ACCESSION],
                psycopg2.extras.Json(fields),
                fingerprint,
            ),
        )
        cur.execute(
            "INSERT INTO sec.filing_headers (load_id, accession, cik) VALUES (%s, %s, 1)",
            (headers, HEADER_ACCESSION),
        )
        cur.execute(
            "INSERT INTO sec.index_entries (load_id, quarter, filename) VALUES (%s, '2005Q2', %s)",
            (index, f"edgar/data/2/{INDEX_ACCESSION}.txt"),
        )
    verify_db.commit()

    ordinary = run_checks(verify_db)
    assert ordinary["ok"], ordinary
    deep = run_checks(verify_db, deep=True)
    assert deep["ok"], deep

    with verify_db.cursor() as cur:
        cur.execute("UPDATE sec.index_entries SET quarter='2005Q3' WHERE load_id=%s", (index,))
    verify_db.commit()
    deep = run_checks(verify_db, deep=True)
    assert not deep["checks"]["content_sha256_matches_stored_rows"]["ok"]


@pytest.mark.integration
@pytest.mark.db
def test_older_market_data_load_without_coverage_is_allowed(
    tmp_path: Path, verify_db: connection
) -> None:
    from ggTrader.data.sec_import.verify import run_checks

    path = tmp_path / "legacy"
    path.touch()
    stage_load(verify_db, "crsp-daily-pilot", path, coverage=False)
    result = run_checks(verify_db)["checks"]["every_complete_load_has_one_coverage_row"]
    assert result["ok"]
    assert "older market-data loads may have none" in result["detail"]


@pytest.mark.integration
@pytest.mark.db
def test_detects_label_with_wrong_last_date(tmp_path: Path, verify_db: connection) -> None:
    from ggTrader.data.sec_import.verify import run_checks

    path = tmp_path / "labels"
    path.touch()
    load_id = stage_load(verify_db, "sec-labels:study", path)
    with verify_db.cursor() as cur:
        cur.execute("INSERT INTO ref.releases (status) VALUES ('building') RETURNING release_id")
        release = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO market.stock_spans (release_id, symbol, permno, last_date) "
            "VALUES (%s, 'OLD', 12345, %s)",
            (release, LAST_DATE),
        )
        cur.execute(
            "INSERT INTO sec.death_labels "
            "(load_id, study, method_version, permno, last_date, release_id, accessions) "
            "VALUES (%s, 'study', 'v1', 12345, '2005-06-02', %s, %s)",
            (load_id, release, [MISSING_ACCESSION]),
        )
    verify_db.commit()
    checks = run_checks(verify_db)["checks"]
    assert not checks["labels_match_stock_spells_in_release"]["ok"]
    assert "2005-06-02" in checks["labels_match_stock_spells_in_release"]["detail"]
    assert not checks["cited_accessions_exist"]["ok"]
    assert MISSING_ACCESSION in checks["cited_accessions_exist"]["detail"]


@pytest.mark.integration
@pytest.mark.db
def test_record_inserts_one_row_and_nothing_else_changes(
    tmp_path: Path, verify_db: connection
) -> None:
    from ggTrader.data.sec_import.verify import record, run_checks

    path = tmp_path / "legacy"
    path.touch()
    stage_load(verify_db, "mcgill-fiam", path)
    with verify_db.cursor() as cur:
        cur.execute("SELECT * FROM ref.loads")
        original_loads = cur.fetchall()
    report = run_checks(verify_db)
    check_id = record(verify_db, report, "ordinary")
    with verify_db.cursor() as cur:
        cur.execute("SELECT check_id, mode, ok, report FROM ref.integrity_checks")
        assert cur.fetchall() == [(check_id, "ordinary", True, report)]
        cur.execute("SELECT * FROM ref.loads")
        assert cur.fetchall() == original_loads
    assert run_checks(verify_db) == report
