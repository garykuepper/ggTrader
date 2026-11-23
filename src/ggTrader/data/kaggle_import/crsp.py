"""Import the delivered CRSP parquet in resumable row groups."""

import csv
import io
import math
from collections.abc import Sequence
from datetime import date, time
from pathlib import Path

import pyarrow.parquet as pq
from psycopg2.extensions import connection, cursor

from ggTrader.data.kaggle_import.guards import backup_fresh, free_space_ok, verify_sha256
from ggTrader.data.kaggle_import.loadlog import LoadFailed, begin_load, commit_chunk, complete_load

PARSER_VERSION = "crsp-1"
COLUMNS = ("permno", "date", "shrcd", "exchcd", "prc", "ret", "shrout", "vwretd", "year", "month")
FLOATS = ("prc", "ret", "shrout", "vwretd")


def _convert(row: dict, row_ref: str):
    stamp = row["date"]
    if stamp is None or stamp.time() != time.min:
        raise LoadFailed(f"Non-midnight date for permno {row['permno']}: {stamp}")
    if not date(1968, 1, 1) <= stamp.date() <= date(2024, 12, 31):
        raw = {k: (v.isoformat() if k == "date" else v) for k, v in row.items()}
        # PostgreSQL JSON does not accept non-finite numbers.
        raw = {
            k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in raw.items()
        }
        return None, (row_ref, raw, "date outside 1968-2024"), 0
    for key in ("shrcd", "exchcd"):
        value = row[key]
        if value is not None:
            if not math.isfinite(value) or not value.is_integer():
                raise LoadFailed(f"Invalid {key} for permno {row['permno']} date {stamp}")
            row[key] = int(value)
    count = 0
    for key in FLOATS:
        value = row[key]
        if value is not None and not math.isfinite(value):
            row[key] = None
            count += 1
    row["date"] = stamp.date().isoformat()
    return row, None, count


def _write_chunk(  # noqa: PLR0913 — chunk inputs
    conn: connection,
    load_id: int,
    key: str,
    groups: Sequence[int],
    parquet: pq.ParquetFile,
    year: int | None,
    free_path: Path,
    margin: int,
):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM ref.load_chunks WHERE load_id=%s AND chunk_key=%s", (load_id, key)
        )
        if cur.fetchone():
            conn.rollback()
            return
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    quarantine = []
    accepted = nulled = 0
    for group in groups:
        size = parquet.metadata.row_group(group).total_byte_size * 12
        if not free_space_ok(free_path, needed_bytes=size, margin_bytes=margin):
            raise LoadFailed("Insufficient free space for chunk")
        table = parquet.read_row_group(group, columns=list(COLUMNS))
        for index, row in enumerate(table.to_pylist()):
            if year is not None and row["year"] != year:
                continue
            converted, rejected, count = _convert(row, f"rg-{group:03d}:{index}")
            if rejected:
                quarantine.append(rejected)
            else:
                writer.writerow([load_id, *(converted[column] for column in COLUMNS)])
                accepted += 1
                nulled += count
    buffer.seek(0)

    def copy(cur: cursor) -> int:
        cur.copy_expert("COPY kaggle_crsp.daily FROM STDIN WITH (FORMAT CSV, NULL '')", buffer)
        cur.execute(
            "UPDATE ref.loads SET checks=jsonb_set(checks, '{nulled_values}', "
            "to_jsonb(COALESCE((checks->>'nulled_values')::bigint, 0) + %s)) WHERE load_id=%s",
            (nulled, load_id),
        )
        return accepted

    commit_chunk(conn, load_id, key, accepted, copy, quarantine=quarantine)


def load_crsp(  # noqa: PLR0913 — public loader contract
    conn: connection,
    parquet_path: Path,
    sums_file: Path,
    *,
    backup_dir: Path | None,
    row_groups: Sequence[int] | None = None,
    year: int | None = None,
    margin_bytes: int = 40 * 1024**3,
    db_free_path: Path | None = None,
) -> int:
    """Verify and load a file, or a resumable subset of its row groups."""
    if year is not None and row_groups is not None:
        raise ValueError("year and row_groups cannot be combined")
    if backup_dir is not None:
        backup_fresh(backup_dir)
    digest = verify_sha256(parquet_path, sums_file)
    free_path = db_free_path or parquet_path.parent
    if not free_space_ok(
        free_path, needed_bytes=parquet_path.stat().st_size * 12, margin_bytes=margin_bytes
    ):
        raise LoadFailed("Insufficient free space for load")
    parquet = pq.ParquetFile(parquet_path)
    all_groups = list(range(parquet.num_row_groups))
    groups = all_groups if row_groups is None else list(row_groups)
    if len(set(groups)) != len(groups) or any(i not in all_groups for i in groups):
        raise ValueError("Invalid or repeated row group")
    expected = [f"year-{year:04d}"] if year is not None else [f"rg-{i:03d}" for i in all_groups]
    load_id = begin_load(
        conn,
        dataset="crsp-daily-pilot" if year is not None else "crsp-daily",
        kaggle_slug="crsp-kaggle",
        file_path=parquet_path,
        file_sha256=digest,
        parser_version=f"{PARSER_VERSION}:pilot-year-{year:04d}"
        if year is not None
        else PARSER_VERSION,
        content_sha256=digest,
        expected_chunks=expected,
    )
    if year is not None:
        _write_chunk(conn, load_id, expected[0], groups, parquet, year, free_path, margin_bytes)
    else:
        for group in groups:
            _write_chunk(
                conn, load_id, f"rg-{group:03d}", [group], parquet, None, free_path, margin_bytes
            )
    if row_groups is None and year is None:
        with conn.cursor() as cur:
            cur.execute("SELECT rows_parsed FROM ref.loads WHERE load_id=%s", (load_id,))
            if cur.fetchone()[0] != parquet.metadata.num_rows:
                conn.rollback()
                raise LoadFailed("Parsed row count differs from Parquet metadata")
    if row_groups is None:
        complete_load(conn, load_id, {}, expected_chunks=expected)
    return load_id


def compress_kaggle_chunks(conn: connection, table: str = "kaggle_crsp.daily") -> int:
    """Compress uncompressed chunks and return their count."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT compress_chunk(c, if_not_compressed => true) FROM show_chunks(%s::regclass) c "
            "JOIN timescaledb_information.chunks i ON i.chunk_schema = split_part(c::text, '.', 1) "
            "AND i.chunk_name = split_part(c::text, '.', 2) WHERE NOT i.is_compressed",
            (table,),
        )
        count = len(cur.fetchall())
    conn.commit()
    return count
