"""Transactional bookkeeping for resumable imports."""

import getpass
from collections.abc import Callable, Sequence
from pathlib import Path

from psycopg2.extensions import connection, cursor
from psycopg2.extras import Json

from ggTrader.data.universe_db import git_sha


class LoadFailed(Exception):
    """A load cannot proceed safely."""


class AlreadyLoaded(LoadFailed):
    """The same file and parser have already completed."""


class RetentionLimit(LoadFailed):
    """A third retained load needs authorization."""


def begin_load(  # noqa: PLR0913 — public import contract
    conn: connection,
    *,
    dataset: str,
    kaggle_slug: str,
    file_path: Path,
    file_sha256: str,
    parser_version: str,
    content_sha256: str,
    expected_chunks: Sequence[str] | None = None,
    allow_third: bool = False,
    slug_version: str | None = None,
    download_date: str | None = None,
    derivation_identity: dict | None = None,
) -> int:
    """Start or resume a load, respecting retained versions."""
    try:
        with conn.cursor() as cur:
            cur.execute("LOCK TABLE ref.loads IN SHARE ROW EXCLUSIVE MODE")
            if derivation_identity is not None:
                cur.execute(
                    "SELECT checks FROM ref.loads WHERE dataset=%s AND file_path=%s "
                    "AND status='running'",
                    (dataset, str(file_path)),
                )
                for (checks,) in cur.fetchall():
                    if (checks or {}).get("derivation_identity") != derivation_identity:
                        raise LoadFailed("Running derivation identity or chunk layout differs")
            cur.execute(
                "SELECT load_id, status"
                " FROM ref.loads WHERE file_sha256=%s AND parser_version=%s"
                " AND status IN ('running', 'complete') ORDER BY status",
                (file_sha256, parser_version),
            )
            matches = cur.fetchall()
            if any(status == "complete" for _, status in matches):
                raise AlreadyLoaded(f"{file_sha256} with parser {parser_version} already loaded")
            if matches:
                result = matches[0][0]
            else:
                cur.execute(
                    "SELECT count(*) FROM ref.loads WHERE dataset=%s "
                    "AND status IN ('running','complete')",
                    (dataset,),
                )
                if cur.fetchone()[0] >= 2 and not allow_third:
                    raise RetentionLimit(f"{dataset} already has two retained loads")
                cur.execute("SELECT current_user")
                loaded_by = f"{getpass.getuser()}@{cur.fetchone()[0]}"
                cur.execute(
                    "INSERT INTO ref.loads(dataset,kaggle_slug,file_path,file_sha256,"
                    "parser_version,"
                    "content_sha256,rows_parsed,rows_accepted,rows_quarantined,"
                    "started_at,status,checks,loader_git_sha,loaded_by) "
                    "VALUES (%s,%s,%s,%s,%s,%s,0,0,0,now(),'running',%s,%s,%s) RETURNING load_id",
                    (
                        dataset,
                        kaggle_slug,
                        str(file_path),
                        file_sha256,
                        parser_version,
                        content_sha256,
                        Json(
                            {
                                "expected_chunks": list(expected_chunks or []),
                                "slug_version": slug_version,
                                "download_date": download_date,
                                **(
                                    {"derivation_identity": derivation_identity}
                                    if derivation_identity is not None
                                    else {}
                                ),
                            }
                        ),
                        git_sha(),
                        loaded_by,
                    ),
                )
                result = cur.fetchone()[0]
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise


def _running(cur: cursor, load_id: int) -> tuple:
    cur.execute(
        "SELECT rows_parsed, rows_accepted, rows_quarantined, checks"
        " FROM ref.loads WHERE load_id=%s AND status='running' FOR UPDATE",
        (load_id,),
    )
    row = cur.fetchone()
    if row is None:
        raise LoadFailed(f"Load {load_id} is not running")
    return row


def fail_load(conn: connection, load_id: int, reason: str) -> None:
    """Mark a running load terminal while preserving its checks."""
    try:
        with conn.cursor() as cur:
            _, _, _, checks = _running(cur, load_id)
            cur.execute(
                "UPDATE ref.loads SET status='failed', checks=%s WHERE load_id=%s",
                (Json({**(checks or {}), "failure": reason}), load_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def commit_chunk(
    conn: connection,
    load_id: int,
    chunk_key: str,
    rows: int,
    writer: Callable[[cursor], int],
    quarantine: Sequence[tuple[str, dict, str]] = (),
) -> bool:
    """Commit data and chunk marker together; retries do not call writer."""
    try:
        with conn.cursor() as cur:
            _running(cur, load_id)
            cur.execute(
                "SELECT 1 FROM ref.load_chunks WHERE load_id=%s AND chunk_key=%s",
                (load_id, chunk_key),
            )
            if cur.fetchone():
                conn.commit()
                return False
            if rows < 0 or writer(cur) != rows:
                raise LoadFailed(f"Chunk {chunk_key} wrote an unexpected row count")
            cur.executemany(
                "INSERT INTO ref.quarantine VALUES (%s,%s,%s,%s,%s)",
                [(load_id, chunk_key, ref, Json(raw), reason) for ref, raw, reason in quarantine],
            )
            cur.execute(
                "INSERT INTO ref.load_chunks VALUES (%s,%s,%s,now())", (load_id, chunk_key, rows)
            )
            cur.execute(
                "UPDATE ref.loads SET rows_accepted=rows_accepted+%s, "
                "rows_quarantined=rows_quarantined+%s, rows_parsed=rows_parsed+%s "
                "WHERE load_id=%s",
                (rows, len(quarantine), rows + len(quarantine), load_id),
            )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise


def record_files(conn: connection, load_id: int, files: Sequence[tuple[str, str, int]]) -> None:
    """Register source files, retaining the first record for each load and path."""
    try:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO ref.load_files (load_id, path, sha256, bytes) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (load_id, path) DO NOTHING",
                [(load_id, path, sha256, size) for path, sha256, size in files],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def complete_load(
    conn: connection,
    load_id: int,
    checks: dict,
    *,
    expected_chunks: Sequence[str] | None = None,
    coverage: tuple[str, str] | None = None,
) -> None:
    """Complete only a reconciled load with all expected chunks."""
    try:
        with conn.cursor() as cur:
            parsed, accepted, quarantined, stored = _running(cur, load_id)
            expected = (
                expected_chunks
                if expected_chunks is not None
                else (stored or {}).get("expected_chunks", [])
            )
            cur.execute("SELECT chunk_key FROM ref.load_chunks WHERE load_id=%s", (load_id,))
            missing = set(expected) - {row[0] for row in cur.fetchall()}
            if missing:
                raise LoadFailed(f"Missing chunks: {', '.join(sorted(missing))}")
            if parsed != accepted + quarantined:
                raise LoadFailed("Parsed rows do not reconcile")
            combined = {**(stored or {}), **checks}
            if coverage is not None:
                cur.execute(
                    "INSERT INTO ref.load_coverage (load_id, coverage, note) VALUES (%s, %s, %s)",
                    (load_id, *coverage),
                )
            cur.execute(
                "UPDATE ref.loads SET status='complete', completed_at=now(), checks=%s"
                " WHERE load_id=%s",
                (Json(combined), load_id),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def can_delete_load(conn: connection, load_id: int) -> bool:
    """A published release protects its loads."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT NOT EXISTS (SELECT 1 FROM ref.release_loads l"
            " JOIN ref.releases r USING (release_id) WHERE l.load_id=%s AND r.status='published')",
            (load_id,),
        )
        return cur.fetchone()[0]
