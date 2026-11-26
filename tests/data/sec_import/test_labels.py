"""Tests for the frozen SEC study label importers."""

import csv
import hashlib
import json
import os
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import psycopg2
import pytest
from psycopg2.extensions import connection

from ggTrader.data.kaggle_import.ddl import apply_ddl as apply_kaggle_ddl
from ggTrader.data.kaggle_import.loadlog import LoadFailed
from ggTrader.data.sec_import import labels
from ggTrader.data.sec_import.ddl import apply_ddl
from ggTrader.utils.paths import REPO_ROOT
from tests.data.scratch_db import scratch_database

EARLY = {
    "column": "PERMNO-12345",
    "permno": "12345",
    "symbol": "OLD",
    "last_trading_day": "2004-06-01",
    "company_name": "Old Company",
    "name_source": "historical",
    "exchange": "NYSE",
    "rank_band": "",
    "held_by_rule": "",
    "label": "no_event_found",
    "confidence": "",
    "evidence": "",
    "suggested_loss": "",
}
PAYOUT = {
    "permno": "12345",
    "exit_date": "2004-06-01",
    "evidence_class": "acquired_or_merged",
    "match_confidence": "",
    "loss_note": "Acquisition recorded",
    "suggested_loss": "12.5",
    "form25_accession": "25-1",
    "event_accessions": " 8K-1 ; ;25-1",
    "deal_form_accessions": "; DEAL-1 ;8K-1;",
}
LAST_DATE = date(2004, 6, 1)


def write_early(path: Path, *rows: dict[str, str]) -> Path:
    """Write early labels with the study's CSV columns."""
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(EARLY))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_payout(path: Path, *rows: dict[str, str]) -> Path:
    """Write payouts inside the study's result object."""
    path.write_text(
        json.dumps({"phase": "payouts", "rows": rows, "coverage": {}}), encoding="utf-8"
    )
    return path


def test_read_early_labels_maps_columns_and_empty_loss_to_none(tmp_path: Path) -> None:
    path = write_early(tmp_path / "early.csv", EARLY)
    assert labels.read_early_labels(path) == [
        labels.LabelRow(
            study="dead-stock-early-filings",
            method_version="early-v1-1",
            permno=12345,
            last_date=LAST_DATE,
            label="no_event_found",
            confidence=None,
            evidence_text=None,
            suggested_loss=None,
            accessions=(),
            source_fields=EARLY,
            source_row_sha256=hashlib.sha256(
                json.dumps(EARLY, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        )
    ]


def test_read_early_labels_keeps_evidence_verbatim(tmp_path: Path) -> None:
    evidence = "  Café, first line\r\nsecond line  "
    path = write_early(tmp_path / "early.csv", {**EARLY, "evidence": evidence})
    assert [row.evidence_text for row in labels.read_early_labels(path)] == [evidence]


def test_read_payout_labels_splits_accessions_on_semicolon(tmp_path: Path) -> None:
    path = write_payout(tmp_path / "payout.json", {**PAYOUT, "match_confidence": "high"})
    assert labels.read_payout_labels(path) == [
        labels.LabelRow(
            study="dead-stock-data-integrity",
            method_version="payouts-v2",
            permno=12345,
            last_date=LAST_DATE,
            label="acquired_or_merged",
            confidence="high",
            evidence_text="Acquisition recorded",
            suggested_loss=12.5,
            accessions=("25-1", "8K-1", "DEAL-1"),
            source_fields={**PAYOUT, "match_confidence": "high"},
            source_row_sha256=hashlib.sha256(
                json.dumps(
                    {**PAYOUT, "match_confidence": "high"}, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        )
    ]


def test_read_payout_labels_empty_confidence_is_none(tmp_path: Path) -> None:
    raw = {
        **PAYOUT,
        "suggested_loss": "",
        "loss_note": "",
        "form25_accession": "",
        "event_accessions": "",
        "deal_form_accessions": "",
    }
    path = write_payout(tmp_path / "payout.json", raw)
    [row] = labels.read_payout_labels(path)
    assert (row.confidence, row.evidence_text, row.suggested_loss, row.accessions) == (
        None,
        None,
        None,
        (),
    )


def test_row_hash_is_stable_for_key_order(tmp_path: Path) -> None:
    first = write_payout(tmp_path / "first.json", PAYOUT)
    second = write_payout(tmp_path / "second.json", dict(reversed(list(PAYOUT.items()))))
    assert first.read_bytes() != second.read_bytes()
    assert labels.read_payout_labels(first) == labels.read_payout_labels(second)
    assert len(labels.read_payout_labels(first)) == 1


def load(conn: connection, path: Path, rows: list[labels.LabelRow]) -> int:
    """Load a fixture with its directory as the stand-in for the database drive."""
    return labels.load_labels(
        conn,
        rows,
        source_file=path,
        backup_dir=path.parent,
        db_free_path=path.parent,
        margin_bytes=0,
    )


def test_load_labels_refuses_duplicate_key_in_one_file(tmp_path: Path) -> None:
    path = write_early(tmp_path / "early.csv", EARLY, EARLY)
    conn = MagicMock(spec=connection)
    with pytest.raises(labels.LabelConflict, match="early-v1-1.*12345.*2004-06-01"):
        load(conn, path, labels.read_early_labels(path))
    conn.cursor.assert_not_called()


def test_load_labels_refuses_file_changed_since_read(tmp_path: Path) -> None:
    path = write_early(tmp_path / "early.csv", EARLY)
    rows = labels.read_early_labels(path)
    write_early(path, {**EARLY, "label": "bankruptcy"})
    conn = MagicMock(spec=connection)
    with pytest.raises(LoadFailed, match="do not match"):
        load(conn, path, rows)
    conn.cursor.assert_not_called()


@pytest.fixture
def label_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[connection]:
    """Create a stock list with two spells for one PERMNO and one for another."""
    with scratch_database() as url, psycopg2.connect(url) as conn:
        apply_kaggle_ddl(conn)
        apply_ddl(conn)
        with conn.cursor() as cur:
            cur.execute("INSERT INTO ref.releases (created_at, status) VALUES (now(), 'building')")
            cur.execute(
                "INSERT INTO market.stock_spans (release_id, symbol, permno, spell, "
                "first_date, last_date, linked, kind) VALUES "
                "(1, 'OLD-1', 12345, 1, '2000-01-01', '2001-01-01', false, 'spell'), "
                "(1, 'OLD-2', 12345, 2, '2002-01-01', '2004-06-01', false, 'spell'), "
                "(1, 'OLD-3', 23456, 1, '2002-01-01', '2004-06-01', false, 'spell')"
            )
            cur.execute("UPDATE ref.releases SET status='published' WHERE release_id=1")
        conn.commit()
        monkeypatch.setattr(labels, "backup_fresh", lambda _: None)
        yield conn


@pytest.mark.integration
@pytest.mark.db
def test_unmatched_against_stock_list_reports_wrong_last_date(
    tmp_path: Path, label_db: connection
) -> None:
    earlier = {**EARLY, "last_trading_day": "2001-01-01"}
    wrong = {**EARLY, "last_trading_day": "2004-06-02"}
    path = write_early(tmp_path / "early.csv", EARLY, earlier, wrong)
    rows = labels.read_early_labels(path)
    assert labels.unmatched_against_stock_list(label_db, rows, 1) == [rows[2]]
    assert labels.unmatched_against_stock_list(label_db, rows, 2) == rows


@pytest.mark.integration
@pytest.mark.db
def test_load_labels_refuses_whole_load_on_mismatch(tmp_path: Path, label_db: connection) -> None:
    wrong = {**EARLY, "last_trading_day": "2004-06-02"}
    path = write_early(tmp_path / "early.csv", EARLY, wrong)
    rows = labels.read_early_labels(path)
    with pytest.raises(labels.LabelMismatch) as exc:
        load(label_db, path, rows)
    assert exc.value.unmatched == [rows[1]]
    with label_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM ref.loads")
        assert cur.fetchone() == (0,)
        cur.execute("SELECT count(*) FROM sec.death_labels")
        assert cur.fetchone() == (0,)


@pytest.mark.integration
@pytest.mark.db
def test_load_labels_conflict_on_changed_row_and_skip_on_identical(
    tmp_path: Path, label_db: connection
) -> None:
    path = write_early(tmp_path / "early.csv", EARLY)
    load_id = load(label_db, path, labels.read_early_labels(path))
    assert load(label_db, path, labels.read_early_labels(path)) == load_id
    write_early(path, {**EARLY, "label": "bankruptcy"})
    with pytest.raises(labels.LabelConflict, match="dead-stock-early-filings.*12345.*2004-06-01"):
        load(label_db, path, labels.read_early_labels(path))
    with label_db.cursor() as cur:
        cur.execute("SELECT load_id, status FROM ref.loads")
        assert cur.fetchall() == [(load_id, "complete")]
        cur.execute("SELECT load_id, label FROM sec.death_labels")
        assert cur.fetchall() == [(load_id, EARLY["label"])]


@pytest.mark.integration
@pytest.mark.db
def test_load_labels_appends_only_new_keys(tmp_path: Path, label_db: connection) -> None:
    path = write_early(tmp_path / "early.csv", EARLY)
    first = load(label_db, path, labels.read_early_labels(path))
    new = {**EARLY, "column": "PERMNO-23456", "permno": "23456"}
    write_early(path, EARLY, new)
    second = load(label_db, path, labels.read_early_labels(path))
    assert second != first
    assert load(label_db, path, labels.read_early_labels(path)) == second
    with label_db.cursor() as cur:
        cur.execute("SELECT load_id, permno FROM sec.death_labels ORDER BY load_id")
        assert cur.fetchall() == [(first, 12345), (second, 23456)]
        cur.execute(
            "SELECT rows_parsed, rows_accepted, rows_quarantined, checks->>'rows_skipped' "
            "FROM ref.loads WHERE load_id=%s",
            (second,),
        )
        assert cur.fetchone() == (1, 1, 0, "1")


@pytest.mark.integration
@pytest.mark.db
def test_load_labels_keeps_two_studies_separate_with_provenance(
    tmp_path: Path, label_db: connection
) -> None:
    early = write_early(tmp_path / "early.csv", EARLY)
    payout = write_payout(tmp_path / "payout.json", PAYOUT)
    first = load(label_db, early, labels.read_early_labels(early))
    payout_rows = labels.read_payout_labels(payout)
    second = load(label_db, payout, payout_rows)
    assert second != first
    [row] = payout_rows
    with label_db.cursor() as cur:
        cur.execute(
            "SELECT l.dataset, c.coverage, c.note FROM ref.loads l "
            "JOIN ref.load_coverage c USING (load_id) ORDER BY load_id"
        )
        assert cur.fetchall() == [
            (
                f"sec-labels:{study}",
                "not_a_stock_source",
                f"labels from {study}, copied as stored",
            )
            for study in ("dead-stock-early-filings", "dead-stock-data-integrity")
        ]
        cur.execute(
            "SELECT release_id, confidence, evidence_text, suggested_loss, accessions, "
            "source_file, source_file_sha256, source_row_sha256, source_fields "
            "FROM sec.death_labels WHERE load_id=%s",
            (second,),
        )
        source_path = os.path.relpath(payout, REPO_ROOT)
        digest = hashlib.sha256(payout.read_bytes()).hexdigest()
        assert cur.fetchone() == (
            1,
            None,
            PAYOUT["loss_note"],
            12.5,
            list(row.accessions),
            source_path,
            digest,
            row.source_row_sha256,
            PAYOUT,
        )
        cur.execute("SELECT path, sha256, bytes FROM ref.load_files WHERE load_id=%s", (second,))
        assert cur.fetchone() == (source_path, digest, payout.stat().st_size)


@pytest.mark.integration
@pytest.mark.db
def test_load_labels_quarantines_unparseable_rows_and_resumes(
    tmp_path: Path, label_db: connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    damaged = {**EARLY, "permno": "bad"}
    path = write_early(tmp_path / "early.csv", EARLY, damaged)
    rows = labels.read_early_labels(path)

    def interrupt(*args: object, **kwargs: object) -> None:
        raise RuntimeError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(labels, "complete_load", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            load(label_db, path, rows)
    load_id = load(label_db, path, rows)
    with label_db.cursor() as cur:
        cur.execute(
            "SELECT rows_parsed, rows_accepted, rows_quarantined, status "
            "FROM ref.loads WHERE load_id=%s",
            (load_id,),
        )
        assert cur.fetchone() == (2, 1, 1, "complete")
        cur.execute("SELECT raw, reason FROM ref.quarantine WHERE load_id=%s", (load_id,))
        assert cur.fetchall() == [(damaged, "invalid PERMNO or date")]


def test_bom_non_finite_loss_and_nul_are_handled_on_the_early_csv(tmp_path: Path) -> None:
    path = tmp_path / "early.csv"
    write_early(path, EARLY, {**EARLY, "permno": "2", "suggested_loss": "nan"})
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    nul = {**EARLY, "permno": "3", "evidence": "a\x00b"}
    with path.open("a", encoding="utf-8", newline="") as output:
        csv.DictWriter(output, fieldnames=list(EARLY)).writerow(nul)
    rows, bad = labels._parse_file(labels.EARLY_STUDY, path.read_bytes())
    assert [row.permno for row in rows] == [12345]
    assert "column" in rows[0].source_fields  # no BOM glued to the first header
    assert [reason for _, _, reason in bad] == [
        "suggested_loss is not a finite number",
        "NUL character in row",
    ]
    assert bad[1][1]["evidence"] == "a\\x00b"


@pytest.mark.parametrize("permno", ["1_0", " 5", "+7", "-1"])
def test_odd_permno_text_is_quarantined(tmp_path: Path, permno: str) -> None:
    path = write_early(tmp_path / "early.csv", {**EARLY, "permno": permno})
    rows, bad = labels._parse_file(labels.EARLY_STUDY, path.read_bytes())
    assert not rows and len(bad) == 1


@pytest.mark.integration
@pytest.mark.db
def test_same_labels_in_different_bytes_do_not_start_a_new_load(
    tmp_path: Path, label_db: connection
) -> None:
    path = write_early(tmp_path / "early.csv", EARLY)
    first = load(label_db, path, labels.read_early_labels(path))
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(reversed(list(EARLY))))
        writer.writeheader()
        writer.writerow(EARLY)
    assert load(label_db, path, labels.read_early_labels(path)) == first
    with label_db.cursor() as cur:
        cur.execute("SELECT count(*) FROM ref.loads")
        assert cur.fetchone() == (1,)


@pytest.mark.integration
@pytest.mark.db
def test_chunks_mix_rows_and_quarantine_and_resume_between_chunks(
    tmp_path: Path, label_db: connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = {**EARLY, "column": "PERMNO-23456", "permno": "23456"}
    broken = {**EARLY, "permno": "bad"}
    path = write_early(tmp_path / "early.csv", EARLY, other, broken)
    rows = labels.read_early_labels(path)
    monkeypatch.setattr(labels, "CHUNK_SIZE", 2)
    real_commit = labels.commit_chunk
    calls: list[str] = []

    def crash_on_second(conn: object, load_id: int, key: str, *args: object, **kw: object) -> bool:
        calls.append(key)
        if len(calls) == 2:
            raise RuntimeError("crash between chunks")
        return real_commit(conn, load_id, key, *args, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(labels, "commit_chunk", crash_on_second)
        with pytest.raises(RuntimeError, match="crash"):
            load(label_db, path, rows)
    load_id = load(label_db, path, rows)
    with label_db.cursor() as cur:
        cur.execute("SELECT load_id, status FROM ref.loads")
        assert cur.fetchall() == [(load_id, "complete")]
        cur.execute("SELECT chunk_key, rows FROM ref.load_chunks ORDER BY chunk_key")
        assert cur.fetchall() == [("labels-0000", 2), ("labels-0001", 0)]
        cur.execute("SELECT count(*) FROM sec.death_labels")
        assert cur.fetchone() == (2,)
        cur.execute("SELECT count(*) FROM ref.quarantine WHERE load_id=%s", (load_id,))
        assert cur.fetchone() == (1,)


@pytest.mark.integration
@pytest.mark.db
def test_a_pending_load_of_another_file_must_be_resumed_first(
    tmp_path: Path, label_db: connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = write_early(tmp_path / "first.csv", EARLY)

    def interrupt(*args: object, **kwargs: object) -> None:
        raise RuntimeError("interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(labels, "complete_load", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            load(label_db, first, labels.read_early_labels(first))
    other = {**EARLY, "column": "PERMNO-23456", "permno": "23456"}
    second = write_early(tmp_path / "second.csv", EARLY, other)
    with pytest.raises(LoadFailed, match="Resume pending load"):
        load(label_db, second, labels.read_early_labels(second))


@pytest.mark.integration
@pytest.mark.db
def test_a_label_for_another_release_is_a_conflict(tmp_path: Path, label_db: connection) -> None:
    path = write_early(tmp_path / "early.csv", EARLY)
    rows = labels.read_early_labels(path)
    load(label_db, path, rows)
    with label_db.cursor() as cur:
        cur.execute("INSERT INTO ref.releases (created_at, status) VALUES (now(), 'building')")
        cur.execute(
            "INSERT INTO market.stock_spans (release_id, symbol, permno, spell, first_date, "
            "last_date, linked, kind) VALUES (2, 'OLD-9', 12345, 1, '2002-01-01', '2004-06-01', "
            "false, 'spell')"
        )
    label_db.commit()
    with pytest.raises(labels.LabelConflict):
        labels.load_labels(
            label_db,
            rows,
            source_file=path,
            release_id=2,
            backup_dir=path.parent,
            db_free_path=path.parent,
            margin_bytes=0,
        )


@pytest.mark.integration
@pytest.mark.db
def test_a_third_file_version_hits_the_retention_limit(
    tmp_path: Path, label_db: connection
) -> None:
    from ggTrader.data.kaggle_import.loadlog import RetentionLimit

    second_key = {**EARLY, "last_trading_day": "2001-01-01"}
    third_key = {**EARLY, "column": "PERMNO-23456", "permno": "23456"}
    path = write_early(tmp_path / "early.csv", EARLY)
    load(label_db, path, labels.read_early_labels(path))
    write_early(path, EARLY, second_key)
    load(label_db, path, labels.read_early_labels(path))
    write_early(path, EARLY, second_key, third_key)
    with pytest.raises(RetentionLimit):
        load(label_db, path, labels.read_early_labels(path))
