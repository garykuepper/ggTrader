"""Build return-chained bars from a completed CRSP load."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
from psycopg2 import InterfaceError, OperationalError
from psycopg2.extensions import connection, cursor

from ggTrader.data.kaggle_import.guards import backup_fresh, free_space_ok
from ggTrader.data.kaggle_import.loadlog import (
    LoadFailed,
    begin_load,
    commit_chunk,
    complete_load,
    fail_load,
)
from ggTrader.data.kaggle_import.spells import assign_spells

# Measured 124 bytes per row uncompressed in the 2026-11-22 rehearsal, rounded up.
BYTES_PER_ROW = 140
#: Coverage label for derived bars (same wording as the schema backfill).
DERIVED_COVERAGE = ("includes_dead_stocks", "derived from the CRSP load")


def build_spell_closes(spelled: pd.DataFrame) -> pd.DataFrame:
    """Anchor each spell at its final observed price and chain returns backwards."""
    result = spelled.copy()
    result["abs_prc"] = result.prc.abs()
    if result.abs_prc.isna().any():
        raise ValueError("Missing price in spelled rows")
    bad = result.chained_return.le(-1)
    if bad.any():
        row = result.loc[bad].iloc[0]
        raise ValueError(f"Invalid return for permno {row.permno} date {row.date}")
    groups = [result.permno, result.spell]
    logs = np.log1p(result.chained_return.fillna(0))
    cumulative = logs.groupby(groups, sort=False).cumsum()
    total = cumulative.groupby(groups, sort=False).transform("last")
    anchor = result.abs_prc.groupby(groups, sort=False).transform("last")
    result["close"] = anchor * np.exp(cumulative - total)
    result["midpoint_flag"] = result.prc.lt(0)
    result["final_flag"] = ~result.permno.duplicated(keep="last")
    return result


def derive_bars(  # noqa: PLR0912, PLR0913, PLR0915 — loader contract and sequential chunk workflow
    conn: connection,
    crsp_load_id: int,
    calendar: pd.DatetimeIndex,
    *,
    db_free_path: Path,
    backup_dir: Path | None,
    rows_per_chunk: int = 1_000_000,
    margin_bytes: int = 40 * 1024**3,
) -> int:
    """Derive resumable stock-aligned chunks from a completed source."""
    if rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    if backup_dir is not None:
        backup_fresh(backup_dir)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT file_sha256, parser_version FROM ref.loads WHERE load_id=%s "
            "AND dataset IN ('crsp-daily','crsp-daily-pilot') AND status='complete'",
            (crsp_load_id,),
        )
        source = cur.fetchone()
        if source is None:
            raise LoadFailed(f"Source load {crsp_load_id} is not complete")
        cur.execute(
            "SELECT permno, count(*) FROM kaggle_crsp.daily WHERE load_id=%s "
            "GROUP BY permno ORDER BY permno",
            (crsp_load_id,),
        )
        counts = cur.fetchall()
    chunks = []
    for permno, count in counts:
        if not chunks or chunks[-1][2] >= rows_per_chunk:
            chunks.append([permno, permno, count])
        else:
            chunks[-1][1] = permno
            chunks[-1][2] += count
    keys = [f"pm-{first:06d}" for first, _, _ in chunks]
    calendar_hash = hashlib.sha256(
        "\n".join(pd.Timestamp(day).isoformat() for day in calendar).encode("utf-8")
    ).hexdigest()
    identity = {
        "parent_load_id": crsp_load_id,
        "parent_file_sha256": source[0],
        "parent_parser_version": source[1],
        "calendar_sha256": calendar_hash,
        "derivation_version": "derive-1",
        "chunk_layout": {"algorithm": "permno-ranges-v1", "rows_per_chunk": rows_per_chunk},
    }
    identity_sha = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    load_id = begin_load(
        conn,
        dataset="crsp-derived-bars",
        kaggle_slug="derived",
        file_path=Path(f"load:{crsp_load_id}"),
        # The derived file identity is a hash of the inputs, not the raw file's hash.
        file_sha256=identity_sha,
        parser_version="derive-1",
        content_sha256=source[0],
        expected_chunks=keys,
        derivation_identity=identity,
    )
    try:
        for (first, last, count), key in zip(chunks, keys, strict=True):
            if not free_space_ok(db_free_path, count * BYTES_PER_ROW, margin_bytes):
                raise LoadFailed("Insufficient free space for chunk")
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM ref.load_chunks WHERE load_id=%s AND chunk_key=%s",
                    (load_id, key),
                )
                if cur.fetchone():
                    continue
                cur.execute(
                    "SELECT permno, date, prc, ret, shrout FROM kaggle_crsp.daily "
                    "WHERE load_id=%s AND permno BETWEEN %s AND %s ORDER BY permno, date",
                    (crsp_load_id, first, last),
                )
                rows = pd.DataFrame(
                    cur.fetchall(), columns=["permno", "date", "prc", "ret", "shrout"]
                )
            rows[["prc", "ret", "shrout"]] = rows[["prc", "ret", "shrout"]].astype("float64")
            result = build_spell_closes(assign_spells(rows, calendar))
            result.insert(0, "load_id", load_id)
            columns = [
                "load_id",
                "permno",
                "spell",
                "date",
                "close",
                "abs_prc",
                "midpoint_flag",
                "gap_return_approx",
                "final_flag",
            ]
            buffer = io.BytesIO()
            pacsv.write_csv(
                pa.Table.from_pandas(result[columns], preserve_index=False),
                buffer,
                write_options=pacsv.WriteOptions(include_header=False),
            )
            buffer.seek(0)

            size = len(result)

            def copy(
                cur: cursor,
                buffer: io.BytesIO = buffer,
                columns: list[str] = columns,
                size: int = size,
            ) -> int:  # noqa: PLR0913 — binds chunk values
                cur.copy_expert(
                    f"COPY market.daily_bars ({', '.join(columns)}) "
                    "FROM STDIN WITH (FORMAT CSV, NULL '')",
                    buffer,
                )
                return size

            commit_chunk(conn, load_id, key, count, copy)
            buffer.close()
        with conn.cursor() as cur:
            cur.execute("SELECT rows_accepted FROM ref.loads WHERE load_id=%s", (load_id,))
            written = cur.fetchone()[0]
        source_rows = sum(count for _, count in counts)
        if written != source_rows:
            raise LoadFailed("Derived row count differs from source")
        complete_load(conn, load_id, {"source_rows": source_rows}, coverage=DERIVED_COVERAGE)
    except (InterfaceError, OperationalError):
        raise
    except Exception as exc:
        try:
            conn.rollback()
            fail_load(conn, load_id, str(exc))
        except Exception:  # noqa: S110 — preserve the original failure
            pass
        raise
    return load_id
