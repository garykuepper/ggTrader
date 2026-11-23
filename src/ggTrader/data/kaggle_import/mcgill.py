"""Import McGill characteristics as resumable raw batches."""

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as csv
from psycopg2.extensions import connection, cursor

from ggTrader.data.kaggle_import.ddl import MCGILL_COLUMNS
from ggTrader.data.kaggle_import.guards import backup_fresh, free_space_ok, verify_sha256
from ggTrader.data.kaggle_import.loadlog import (
    LoadFailed,
    begin_load,
    commit_chunk,
    complete_load,
    fail_load,
)

PARSER_VERSION = "mcgill-1"
_TYPES = {
    "date": pa.string(),
    "text": pa.string(),
    "bigint": pa.float64(),
    "integer": pa.float64(),
    "double precision": pa.float64(),
}


def _batches(path: Path, batch_rows: int):
    options = csv.ConvertOptions(
        column_types={name: _TYPES[kind] for name, kind in MCGILL_COLUMNS},
        strings_can_be_null=True,
        null_values=[""],
    )
    reader = csv.open_csv(path, convert_options=options)
    parts = []
    size = 0
    for batch in reader:
        offset = 0
        while offset < batch.num_rows:
            take = min(batch_rows - size, batch.num_rows - offset)
            parts.append(batch.slice(offset, take))
            size += take
            offset += take
            if size == batch_rows:
                yield pa.Table.from_batches(parts)
                parts, size = [], 0
    if size:
        yield pa.Table.from_batches(parts)


def _prepare(table: pa.Table, load_id: int, offset: int):
    columns = {}
    nulled = 0
    for name, kind in MCGILL_COLUMNS:
        values = table[name]
        if kind == "date":
            parsed = pc.strptime(values, format="%Y%m%d", unit="s", error_is_null=True)
            invalid = pc.and_(
                pc.is_valid(values),
                pc.or_(
                    pc.is_null(parsed),
                    pc.fill_null(pc.not_equal(pc.strftime(parsed, format="%Y%m%d"), values), False),
                ),
            )
            indices = pc.indices_nonzero(invalid)
            if len(indices):
                raise LoadFailed(f"Invalid {name} at row {offset + indices[0].as_py() + 2}")
            values = pc.cast(parsed, pa.date32())
        elif kind in ("bigint", "integer"):
            # Float parsing can round 2**53 + 1 down to 2**53.
            invalid = pc.fill_null(
                pc.and_(
                    pc.is_valid(values),
                    pc.or_(
                        pc.or_(
                            pc.invert(pc.is_finite(values)), pc.not_equal(values, pc.trunc(values))
                        ),
                        pc.greater_equal(pc.abs(values), 2**53),
                    ),
                ),
                False,
            )
            indices = pc.indices_nonzero(invalid)
            if len(indices):
                raise LoadFailed(f"Invalid {name} at row {offset + indices[0].as_py() + 2}")
            try:
                values = pc.cast(values, pa.int64() if kind == "bigint" else pa.int32())
            except pa.ArrowInvalid as exc:
                raise LoadFailed(f"Invalid {name} in batch starting at row {offset + 2}") from exc
        elif kind == "double precision":
            invalid = pc.and_(pc.is_valid(values), pc.invert(pc.is_finite(values)))
            nulled += pc.sum(pc.cast(pc.fill_null(invalid, False), pa.int64())).as_py() or 0
            values = pc.if_else(invalid, pa.scalar(None, type=pa.float64()), values)
        columns[name] = values
    return pa.table({"load_id": pa.repeat(load_id, table.num_rows), **columns}), nulled


# Conservative per-row estimate for the wide characteristics table.
BYTES_PER_ROW = 4096


def load_mcgill(
    conn: connection,
    csv_path: Path,
    sums_file: Path,
    *,
    backup_dir: Path | None,
    db_free_path: Path,
    batch_rows: int = 50_000,
) -> int:
    """Verify, count and import the raw file without exposing it to signals."""
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    if backup_dir is not None:
        backup_fresh(backup_dir)
    digest = verify_sha256(csv_path, sums_file)
    count = sum(batch.num_rows for batch in _batches(csv_path, batch_rows))
    if not free_space_ok(
        db_free_path, needed_bytes=count * BYTES_PER_ROW, margin_bytes=40 * 1024**3
    ):
        raise LoadFailed("Insufficient free space for load")
    expected = [f"batch-{i:03d}" for i in range((count + batch_rows - 1) // batch_rows)]
    load_id = begin_load(
        conn,
        dataset="mcgill-fiam",
        kaggle_slug="mcgill-fiam",
        file_path=csv_path,
        file_sha256=digest,
        parser_version=PARSER_VERSION,
        content_sha256=digest,
        expected_chunks=expected,
    )
    with conn.cursor() as cur:
        cur.execute("SELECT checks FROM ref.loads WHERE load_id=%s FOR UPDATE", (load_id,))
        checks = cur.fetchone()[0]
        cur.execute(
            "SELECT rows FROM ref.load_chunks WHERE load_id=%s AND chunk_key='batch-000'",
            (load_id,),
        )
        first = cur.fetchone()
        if (
            checks.get("expected_chunks") != expected
            or ("batch_rows" in checks and checks["batch_rows"] != batch_rows)
            or (first and len(expected) > 1 and first[0] != batch_rows)
        ):
            conn.rollback()
            raise LoadFailed("batch size or file changed since the load began")
        cur.execute(
            "UPDATE ref.loads SET checks=jsonb_set(checks, '{batch_rows}', "
            "to_jsonb(%s::int)) WHERE load_id=%s",
            (batch_rows, load_id),
        )
    conn.commit()
    for index, table in enumerate(_batches(csv_path, batch_rows)):
        if not free_space_ok(db_free_path, table.num_rows * BYTES_PER_ROW):
            raise LoadFailed("Insufficient free space for batch")
        try:
            output, nulled = _prepare(table, load_id, index * batch_rows)
        except LoadFailed as exc:
            fail_load(conn, load_id, str(exc))
            raise
        buffer = io.BytesIO()
        csv.write_csv(output, buffer, write_options=csv.WriteOptions(include_header=False))
        buffer.seek(0)

        def write(
            cur: cursor,
            buffer: io.BytesIO = buffer,
            nulled: int = nulled,
            rows: int = output.num_rows,
        ):
            cur.copy_expert(
                "COPY kaggle_chars.mcgill FROM STDIN WITH (FORMAT CSV, NULL '')", buffer
            )
            cur.execute(
                "UPDATE ref.loads SET checks=jsonb_set(checks, '{nulled_values}', "
                "to_jsonb(COALESCE((checks->>'nulled_values')::bigint, 0) + %s)) "
                "WHERE load_id=%s",
                (nulled, load_id),
            )
            return rows

        commit_chunk(conn, load_id, f"batch-{index:03d}", output.num_rows, write)
    with conn.cursor() as cur:
        cur.execute("SELECT rows_parsed FROM ref.loads WHERE load_id=%s", (load_id,))
        if cur.fetchone()[0] != count:
            conn.rollback()
            reason = "Parsed row count differs from counting pass"
            fail_load(conn, load_id, reason)
            raise LoadFailed(reason)
    complete_load(conn, load_id, {}, expected_chunks=expected)
    return load_id
