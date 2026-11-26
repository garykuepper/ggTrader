"""Import the delivered CRSP parquet in resumable row groups."""

import io
import math
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from psycopg2 import sql
from psycopg2.extensions import connection, cursor

from ggTrader.data.kaggle_import.guards import backup_fresh, free_space_ok, verify_sha256
from ggTrader.data.kaggle_import.loadlog import (
    LoadFailed,
    begin_load,
    commit_chunk,
    complete_load,
    fail_load,
)

PARSER_VERSION = "crsp-1"
#: Coverage label written with every completed CRSP load (same wording as the schema backfill).
CRSP_COVERAGE = ("includes_dead_stocks", "CRSP; no delisting returns")
COLUMNS = ("permno", "date", "shrcd", "exchcd", "prc", "ret", "shrout", "vwretd", "year", "month")
FLOATS = ("prc", "ret", "shrout", "vwretd")


def _first_error(table: pa.Table, mask: pa.Array, message: str) -> None:
    indices = pc.indices_nonzero(pc.fill_null(mask, False))
    if len(indices):
        index = indices[0].as_py()
        raise LoadFailed(
            f"{message} for permno {table['permno'][index].as_py()} "
            f"date {table['date'][index].as_py()}"
        )


def _prepare(table: pa.Table, group: int, indices: pa.Array, load_id: int):
    stamps = table["date"]
    _first_error(
        table,
        pc.or_(
            pc.is_null(stamps),
            pc.fill_null(pc.not_equal(stamps, pc.floor_temporal(stamps, unit="day")), False),
        ),
        "Non-midnight",
    )
    days = pc.cast(stamps, pa.date32())
    outside = pc.or_(pc.less(days, date(1968, 1, 1)), pc.greater(days, date(2024, 12, 31)))
    rejected_indices = pc.indices_nonzero(outside)
    rejected = table.take(rejected_indices).to_pylist()
    original_indices = pc.take(indices, rejected_indices).to_pylist()
    quarantine = []
    for row, index in zip(rejected, original_indices, strict=True):
        raw = {k: (v.isoformat() if k == "date" else v) for k, v in row.items()}
        raw = {
            k: (str(v) if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in raw.items()
        }
        quarantine.append((f"rg-{group:03d}:{index}", raw, "date outside 1968-2024"))
    table = table.filter(pc.invert(outside))
    columns = {name: table[name] for name in COLUMNS}
    for name in ("shrcd", "exchcd"):
        values = columns[name]
        invalid = pc.and_(
            pc.is_valid(values),
            pc.or_(pc.invert(pc.is_finite(values)), pc.not_equal(values, pc.trunc(values))),
        )
        _first_error(table, invalid, f"Invalid {name}")
        columns[name] = pc.cast(values, pa.int64())
    nulled = 0
    for name in FLOATS:
        values = columns[name]
        invalid = pc.and_(pc.is_valid(values), pc.invert(pc.is_finite(values)))
        nulled += pc.sum(pc.cast(pc.fill_null(invalid, False), pa.int64())).as_py() or 0
        columns[name] = pc.if_else(invalid, pa.scalar(None, type=values.type), values)
    columns["date"] = pc.cast(columns["date"], pa.date32())
    output = pa.table({"load_id": pa.repeat(load_id, table.num_rows), **columns})
    return output, quarantine, nulled


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
    buffer = io.BytesIO()
    quarantine = []
    accepted = nulled = 0
    for group in groups:
        size = parquet.metadata.row_group(group).num_rows * BYTES_PER_ROW
        if not free_space_ok(free_path, needed_bytes=size, margin_bytes=margin):
            raise LoadFailed("Insufficient free space for chunk")
        table = parquet.read_row_group(group, columns=list(COLUMNS))
        indices = pa.array(range(table.num_rows), type=pa.int64())
        if year is not None:
            mask = pc.fill_null(pc.equal(table["year"], year), False)
            indices = indices.filter(mask)
            table = table.filter(mask)
        try:
            output, rejected, count = _prepare(table, group, indices, load_id)
        except LoadFailed as exc:
            fail_load(conn, load_id, str(exc))
            raise
        quarantine.extend(rejected)
        nulled += count
        accepted += output.num_rows
        if output.num_rows:
            pacsv.write_csv(output, buffer, write_options=pacsv.WriteOptions(include_header=False))
        del table, output, indices
    buffer.seek(0)

    def copy(cur: cursor) -> int:
        cur.copy_expert(
            f"COPY kaggle_crsp.daily (load_id, {', '.join(COLUMNS)}) "
            "FROM STDIN WITH (FORMAT CSV, NULL '')",
            buffer,
        )
        cur.execute(
            "UPDATE ref.loads SET checks=jsonb_set(checks, '{nulled_values}', "
            "to_jsonb(COALESCE((checks->>'nulled_values')::bigint, 0) + %s)) WHERE load_id=%s",
            (nulled, load_id),
        )
        return accepted

    try:
        commit_chunk(conn, load_id, key, accepted, copy, quarantine=quarantine)
    finally:
        buffer.close()


# Pilot measured 149 uncompressed bytes per row; round up for the estimate.
BYTES_PER_ROW = 160


def load_crsp(  # noqa: PLR0913 — public loader contract
    conn: connection,
    parquet_path: Path,
    sums_file: Path,
    *,
    backup_dir: Path | None,
    row_groups: Sequence[int] | None = None,
    year: int | None = None,
    margin_bytes: int = 40 * 1024**3,
    db_free_path: Path,
    parser_suffix: str | None = None,
) -> int:
    """Verify and load a file, or a resumable subset of its row groups."""
    if year is not None and row_groups is not None:
        raise ValueError("year and row_groups cannot be combined")
    if backup_dir is not None:
        backup_fresh(backup_dir)
    digest = verify_sha256(parquet_path, sums_file)
    free_path = db_free_path
    parquet = pq.ParquetFile(parquet_path)
    if not free_space_ok(
        free_path, needed_bytes=parquet.metadata.num_rows * BYTES_PER_ROW, margin_bytes=margin_bytes
    ):
        raise LoadFailed("Insufficient free space for load")
    all_groups = list(range(parquet.num_row_groups))
    groups = all_groups if row_groups is None else list(row_groups)
    if len(set(groups)) != len(groups) or any(i not in all_groups for i in groups):
        raise ValueError("Invalid or repeated row group")
    expected = [f"year-{year:04d}"] if year is not None else [f"rg-{i:03d}" for i in all_groups]
    parser_version = (
        f"{PARSER_VERSION}:pilot-year-{year:04d}" if year is not None else PARSER_VERSION
    )
    if parser_suffix is not None:
        parser_version += f":{parser_suffix}"
    load_id = begin_load(
        conn,
        dataset="crsp-daily-pilot" if year is not None else "crsp-daily",
        kaggle_slug="crsp-kaggle",
        file_path=parquet_path,
        file_sha256=digest,
        parser_version=parser_version,
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
                reason = "Parsed row count differs from Parquet metadata"
                fail_load(conn, load_id, reason)
                raise LoadFailed(reason)
    if row_groups is None:
        complete_load(conn, load_id, {}, expected_chunks=expected, coverage=CRSP_COVERAGE)
    return load_id


def compress_kaggle_chunks(
    conn: connection, table: str = "kaggle_crsp.daily", *, reclaim: bool = True
) -> int:
    """Compress eligible chunks and optionally reclaim their dead space."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT i.id, c::text FROM show_chunks(%s::regclass) c "
            "JOIN _timescaledb_catalog.chunk i ON "
            "i.schema_name = split_part(c::text, '.', 1) "
            "AND i.table_name = split_part(c::text, '.', 2) "
            "WHERE (i.status & 1) = 0 OR (i.status & 8) <> 0",
            (table,),
        )
        chunks = cur.fetchall()
        for _, chunk in chunks:
            cur.execute("SELECT compress_chunk(%s::regclass, if_not_compressed => true)", (chunk,))
        if reclaim and chunks:
            ids = [chunk[0] for chunk in chunks]
            cur.execute(
                "SELECT c.schema_name, c.table_name FROM _timescaledb_catalog.chunk c "
                "WHERE c.id = ANY(%s) OR c.id IN "
                "(SELECT compressed_chunk_id FROM _timescaledb_catalog.chunk WHERE id = ANY(%s))",
                (ids, ids),
            )
            relations = cur.fetchall()
    conn.commit()
    if reclaim and chunks:
        previous_autocommit = conn.autocommit
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                for schema, relation in relations:
                    cur.execute(
                        sql.SQL("VACUUM FULL {}.{}").format(
                            sql.Identifier(schema), sql.Identifier(relation)
                        )
                    )
        finally:
            conn.rollback()
            conn.autocommit = previous_autocommit
    return len(chunks)
