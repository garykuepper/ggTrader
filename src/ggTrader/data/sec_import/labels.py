"""Read and load frozen SEC study labels."""

import csv
import hashlib
import io
import json
import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from os.path import relpath
from pathlib import Path

from psycopg2.extensions import connection, cursor
from psycopg2.extras import Json

from ggTrader.data.kaggle_import.guards import backup_fresh, free_space_ok
from ggTrader.data.kaggle_import.loadlog import (
    LoadFailed,
    begin_load,
    commit_chunk,
    complete_load,
    record_files,
)
from ggTrader.utils.paths import REPO_ROOT

EARLY_STUDY = "dead-stock-early-filings"
PAYOUT_STUDY = "dead-stock-data-integrity"
ACCESSION_FIELDS = ("form25_accession", "event_accessions", "deal_form_accessions")
PARSER_VERSION = "sec-labels-1"
CHUNK_SIZE = 200
LOGGER = logging.getLogger(__name__)

LabelKey = tuple[str, str, int, date]
QuarantinedRow = tuple[str, dict, str]


@dataclass(frozen=True)
class LabelRow:
    """One frozen label with its original fields and their hash."""

    study: str
    method_version: str
    permno: int
    last_date: date
    label: str
    confidence: str | None
    evidence_text: str | None
    suggested_loss: float | None
    accessions: tuple[str, ...]
    source_fields: dict
    source_row_sha256: str

    @property
    def key(self) -> LabelKey:
        """Identify a label within its study and method."""
        return self.study, self.method_version, self.permno, self.last_date


class LabelMismatch(Exception):
    """One or more label rows have no matching stock spell."""

    def __init__(self, unmatched: list[LabelRow]) -> None:
        self.unmatched = unmatched
        super().__init__(f"{len(unmatched)} label rows do not match the stock list")


class LabelConflict(Exception):
    """A label key is repeated in a file or conflicts with a completed load."""

    def __init__(self, key: LabelKey, reason: str) -> None:
        self.key = key
        study, method, permno, last_date = key
        super().__init__(f"{reason}: {study}, {method}, PERMNO {permno}, {last_date}")


def _parse_row(study: str, raw: dict[str, str]) -> LabelRow:
    """Map one source row, preserving optional fields and evidence text."""
    if study == EARLY_STUDY:
        method = "early-v1-1"
        last_date = raw["last_trading_day"]
        label = raw["label"]
        confidence = raw.get("confidence")
        evidence = raw.get("evidence")
        accessions: tuple[str, ...] = ()
    else:
        method = "payouts-v2"
        last_date = raw["exit_date"]
        label = raw["evidence_class"]
        confidence = raw.get("match_confidence")
        evidence = raw.get("loss_note")
        parts = (
            part.strip() for field in ACCESSION_FIELDS for part in raw.get(field, "").split(";")
        )
        accessions = tuple(dict.fromkeys(filter(None, parts)))

    permno = int(raw["permno"])
    if not -(2**63) <= permno < 2**63 or not label:
        raise ValueError("invalid PERMNO or empty label")
    loss = raw.get("suggested_loss")
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return LabelRow(
        study,
        method,
        permno,
        date.fromisoformat(last_date),
        label,
        confidence or None,
        evidence or None,
        float(loss) if loss else None,
        accessions,
        raw,
        digest,
    )


def _parse_file(study: str, content: bytes) -> tuple[list[LabelRow], list[QuarantinedRow]]:
    """Decode the frozen file, keeping invalid rows for the load log."""
    if study == EARLY_STUDY:
        raw_rows = csv.DictReader(io.StringIO(content.decode("utf-8"), newline=""), strict=True)
    else:
        payload = json.loads(content)
        if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
            raise ValueError("Expected a payout result object with a rows list")
        raw_rows = payload["rows"]

    rows = []
    bad = []
    for number, raw in enumerate(raw_rows, start=1):
        try:
            if not isinstance(raw, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in raw.items()
            ):
                raise ValueError("expected a row with text columns and values")
            rows.append(_parse_row(study, raw))
        except (KeyError, ValueError) as exc:
            if isinstance(raw, dict):
                fields = raw if None not in raw else {"columns": list(raw.items())}
            else:
                fields = {"row": raw}
            bad.append((str(number), fields, str(exc)))
    return rows, bad


def _read_labels(study: str, path: Path) -> list[LabelRow]:
    """Read usable rows and report parse failures that the loader will quarantine."""
    try:
        rows, bad = _parse_file(study, path.read_bytes())
    except (OSError, ValueError, csv.Error) as exc:
        raise LoadFailed(f"Cannot read labels from {path}: {exc}") from exc
    if bad:
        LOGGER.warning("%s has %d label rows that could not be parsed", path, len(bad))
    return rows


def read_early_labels(csv_path: Path) -> list[LabelRow]:
    """Read the early filings labels from the study's CSV."""
    return _read_labels(EARLY_STUDY, csv_path)


def read_payout_labels(json_path: Path) -> list[LabelRow]:
    """Read the payout labels from the study's JSON result object."""
    return _read_labels(PAYOUT_STUDY, json_path)


def unmatched_against_stock_list(
    conn: connection, rows: Sequence[LabelRow], release_id: int
) -> list[LabelRow]:
    """Find labels without an exact PERMNO and end-date match in any spell."""
    if not rows:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT permno, last_date FROM market.stock_spans "
            "WHERE release_id=%s AND permno=ANY(%s)",
            (release_id, sorted({row.permno for row in rows})),
        )
        spans = set(cur.fetchall())
    return [row for row in rows if (row.permno, row.last_date) not in spans]


@contextmanager
def _lock_dataset(conn: connection, dataset: str) -> Iterator[None]:
    """Keep preflight and publication serialized across chunk commits."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (dataset,))
    try:
        conn.commit()
        yield
    except BaseException:
        conn.rollback()
        raise
    finally:
        if not conn.closed:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (dataset,))
            conn.commit()


def _new_rows(
    conn: connection, rows: Sequence[LabelRow], dataset: str, release_id: int
) -> tuple[list[LabelRow], int]:
    """Reject conflicts and return the new rows and latest matching load ID."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT study, method_version, permno, last_date, source_row_sha256, "
            "release_id, load_id FROM sec.death_labels JOIN ref.loads USING (load_id) "
            "WHERE dataset=%s AND status='complete'",
            (dataset,),
        )
        prior: dict[LabelKey, list[tuple[str, int, int]]] = {}
        for study, method, permno, last_date, digest, release, load_id in cur.fetchall():
            prior.setdefault((study, method, permno, last_date), []).append(
                (digest, release, load_id)
            )
    new = []
    latest = 0
    for row in rows:
        matches = prior.get(row.key, [])
        if not matches:
            new.append(row)
        for digest, release, load_id in matches:
            if digest != row.source_row_sha256 or release != release_id:
                raise LabelConflict(row.key, "Label differs from its completed load")
            latest = max(latest, load_id)
    return new, latest


def _commit_labels(
    conn: connection,
    load_id: int,
    rows: list[LabelRow],
    bad: list[QuarantinedRow],
    keys: list[str],
    release_id: int,
    source_path: str,
    digest: str,
) -> None:
    """Write label rows and raw quarantined rows in resumable chunks."""
    for start, key in enumerate(keys):
        offset = start * CHUNK_SIZE
        end = offset + CHUNK_SIZE
        chunk = rows[offset:end]

        def write(cur: cursor, batch: list[LabelRow] = chunk) -> int:
            cur.executemany(
                "INSERT INTO sec.death_labels (load_id, study, method_version, permno, "
                "last_date, release_id, label, confidence, evidence_text, suggested_loss, "
                "accessions, source_file, source_file_sha256, source_row_sha256, source_fields) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [
                    (
                        load_id,
                        *row.key,
                        release_id,
                        row.label,
                        row.confidence,
                        row.evidence_text,
                        row.suggested_loss,
                        list(row.accessions),
                        source_path,
                        digest,
                        row.source_row_sha256,
                        Json(row.source_fields),
                    )
                    for row in batch
                ],
            )
            return len(batch)

        # Put valid rows first, then quarantine rows, capping the combined chunk size.
        bad_start = max(0, offset - len(rows))
        bad_end = max(0, end - len(rows))
        commit_chunk(conn, load_id, key, len(chunk), write, quarantine=bad[bad_start:bad_end])


def _existing_load(
    conn: connection, dataset: str, digest: str, source_path: str, identity: dict, *, write: bool
) -> int | None:
    """Find an exact completed source and reject a different pending load."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT load_id, status, file_sha256, file_path, parser_version, "
            "checks->'derivation_identity' FROM ref.loads "
            "WHERE dataset=%s AND status IN ('running','complete')",
            (dataset,),
        )
        completed = None
        pending = None
        for load_id, status, sha, path, parser, previous in cur.fetchall():
            if status == "running" and (sha, path, parser, previous) != (
                digest,
                source_path,
                PARSER_VERSION,
                identity,
            ):
                pending = load_id
            if (
                status == "complete"
                and sha == digest
                and parser == PARSER_VERSION
                and previous is not None
                and previous.get("release_id") == identity["release_id"]
            ):
                completed = load_id
        if write and completed is None and pending is not None:
            raise LoadFailed(f"Resume pending load {pending} for {dataset} first")
        return completed


def load_labels(
    conn: connection,
    rows: Sequence[LabelRow],
    *,
    source_file: Path,
    release_id: int = 1,
    backup_dir: Path,
    db_free_path: Path,
    margin_bytes: int | None = None,
) -> int:
    """Publish a label file after checking its rows against the selected release.

    Return the existing ID for an identical file. If another file supplies only
    previously loaded labels, return the latest matching ID without a new load.
    """
    source_path = relpath(source_file.resolve(), REPO_ROOT)
    study = (
        rows[0].study
        if rows
        else {".csv": EARLY_STUDY, ".json": PAYOUT_STUDY}.get(source_file.suffix)
    )
    if study not in (EARLY_STUDY, PAYOUT_STUDY):
        raise LoadFailed(f"Unknown study for {source_path}")
    try:
        content = source_file.read_bytes()
        verified, bad = _parse_file(study, content)
    except (OSError, ValueError, csv.Error) as exc:
        raise LoadFailed(f"Cannot read labels from {source_path}: {exc}") from exc
    if list(rows) != verified:
        raise LoadFailed(f"Supplied labels do not match {source_path}")
    if not (verified or bad):
        raise LoadFailed(f"No label rows in {source_path}")
    keys_seen = set()
    for row in verified:
        if row.key in keys_seen:
            raise LabelConflict(row.key, f"Duplicate label key in {source_path}")
        keys_seen.add(row.key)

    backup_fresh(backup_dir)
    options = {} if margin_bytes is None else {"margin_bytes": margin_bytes}
    if not free_space_ok(db_free_path, len(content), **options):
        raise LoadFailed("Insufficient free space on database drive")
    dataset = f"sec-labels:{study}"
    digest = hashlib.sha256(content).hexdigest()
    identity = {"release_id": release_id, "chunk_size": CHUNK_SIZE}

    with _lock_dataset(conn, dataset):
        unmatched = unmatched_against_stock_list(conn, verified, release_id)
        if unmatched:
            raise LabelMismatch(unmatched)
        new, latest = _new_rows(conn, verified, dataset, release_id)
        completed = _existing_load(
            conn, dataset, digest, source_path, identity, write=bool(new or bad)
        )
        if completed is not None:
            return completed
        if not new and not bad:
            return latest

        keys = [
            f"labels-{index:04d}"
            for index, _ in enumerate(range(0, len(new) + len(bad), CHUNK_SIZE))
        ]
        load_id = begin_load(
            conn,
            dataset=dataset,
            kaggle_slug=dataset,
            file_path=Path(source_path),
            file_sha256=digest,
            parser_version=PARSER_VERSION,
            content_sha256=digest,
            expected_chunks=keys,
            derivation_identity=identity,
        )
        record_files(conn, load_id, [(source_path, digest, len(content))])
        _commit_labels(conn, load_id, new, bad, keys, release_id, source_path, digest)
        complete_load(
            conn,
            load_id,
            {"rows_skipped": len(verified) - len(new)},
            expected_chunks=keys,
            coverage=("not_a_stock_source", f"labels from {study}, copied as stored"),
        )
        return load_id
