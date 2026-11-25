"""Read release spans and close prices of a pinned combined release, in column batches.

``load_bars`` returns five field frames for every symbol at once; for the dead-stock test only
``close`` is needed. ``read_close_batched`` asks for a few hundred symbols at a time, keeps the
``close`` frame and the ended-series rows, and drops the rest before the next batch, so the
largest object alive is one batch of five fields plus the growing close frame.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from ggTrader.data.bars import load_bars

# Columns of the release spans read straight from ``market.stock_spans``. ``has_successor`` and
# ``total_loss`` are not stored there (``load_bars`` derives them), so they are absent here.
SPAN_COLUMNS = ["symbol", "permno", "spell", "first_date", "last_date", "kind", "possible_double"]
SPANS_SQL = (
    "SELECT symbol, permno, spell, first_date, last_date, kind, possible_double "
    "FROM market.stock_spans WHERE release_id = %s ORDER BY symbol"
)
DEFAULT_BATCH = 500
# Which linked Yahoo columns changed since the release was published (a metadata check that
# reads only the first few Yahoo closes next to each splice, never the window's prices).
DRIFT_SQL = (
    "SELECT symbol FROM market._combined_linked(%s::bigint, %s::text[]) WHERE drift ORDER BY symbol"
)
LINKED_SQL = (
    "SELECT symbol FROM market.stock_spans "
    "WHERE release_id = %s AND kind = 'linked_yahoo' ORDER BY symbol"
)


def read_release_spans(conn: Any, release_id: int) -> pd.DataFrame:
    """Every series of one published release, with the dates and kind that eligibility needs.

    Args:
        conn: An open read-only connection (``ggTrader.data.bars.connect_readonly``).
        release_id: The release to read; never "newest".

    Returns:
        One row per release column, columns ``SPAN_COLUMNS``; dates are datetimes and a missing
        ``possible_double`` is False.
    """
    with conn.cursor() as cur:
        cur.execute(SPANS_SQL, (int(release_id),))
        rows = cur.fetchall()
    frame = pd.DataFrame(rows, columns=SPAN_COLUMNS)
    for name in ("first_date", "last_date"):
        frame[name] = pd.to_datetime(frame[name])
    frame["possible_double"] = frame["possible_double"].fillna(False).astype(bool)
    return frame


def read_linked_drift(conn: Any, release_id: int) -> list[str]:
    """Linked Yahoo columns whose Yahoo rows no longer match the pinned release (drift).

    One metadata read: the release's linked columns, then the release function's own drift
    flag for them. It never reads the window's prices.

    Returns:
        The drifted symbols, sorted (empty when none).
    """
    with conn.cursor() as cur:
        cur.execute(LINKED_SQL, (int(release_id),))
        linked = [row[0] for row in cur.fetchall()]
    if not linked:
        return []
    with conn.cursor() as cur:
        cur.execute(DRIFT_SQL, (int(release_id), linked))
        return sorted(row[0] for row in cur.fetchall())


def symbol_batches(symbols: Sequence[str], batch: int) -> list[list[str]]:
    """Split the sorted, de-duplicated symbols into consecutive lists of at most ``batch``."""
    if batch < 1:
        raise ValueError(f"batch must be at least 1, got {batch}")
    ordered = sorted(set(symbols))
    return [ordered[i : i + batch] for i in range(0, len(ordered), batch)]


def read_close_batched(
    symbols: Sequence[str],
    start: str,
    end: str,
    *,
    release_id: int,
    batch: int = DEFAULT_BATCH,
    provenance_out: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``close`` for the symbols from a pinned release, one column batch at a time.

    Args:
        symbols: Release column names to read (duplicates ignored).
        start: First date, ``YYYY-MM-DD``.
        end: Last date, ``YYYY-MM-DD``.
        release_id: The published release to read (always given, never "newest").
        batch: Symbols per ``load_bars`` call.
        provenance_out: If given, filled with ``data_release`` (from the first batch),
            ``bars_sha256`` per batch (list), ``n_batches``, and ``drift`` (any batch reported
            drift: the live Yahoo rows no longer match the pinned release) with the
            ``drift_symbols`` seen (each batch's list is already cut at 50 by ``load_bars``).

    Returns:
        ``(close, spans)``: ``close`` is dates by symbols with sorted columns (only symbols with
        a bar in the window); ``spans`` is the ended-series rows of ``Bars.spans`` for the
        symbols, concatenated.

    Raises:
        ValueError: ``batch`` below 1.
    """
    parts = symbol_batches(symbols, batch)
    closes: list[pd.DataFrame] = []
    spans: list[pd.DataFrame] = []
    hashes: list[str | None] = []
    drift = False
    drift_symbols: set[str] = set()
    data_release = None
    for part in parts:
        bars = load_bars(part, start, end, source="combined", release_id=release_id)
        closes.append(bars.close)
        if bars.spans is not None:
            spans.append(bars.spans)
        hashes.append(bars.provenance.get("bars_sha256"))
        data_release = data_release or bars.provenance.get("data_release")
        drift = drift or bool(bars.provenance.get("drift", False))
        drift_symbols.update(bars.provenance.get("drift_symbols") or [])
        del bars  # frees open, high, low and volume before the next batch
    if provenance_out is not None:
        provenance_out.update(
            {
                "data_release": data_release,
                "bars_sha256": hashes,
                "n_batches": len(parts),
                "drift": drift,
                "drift_symbols": sorted(drift_symbols),
            }
        )
    close = pd.concat(closes, axis=1) if closes else pd.DataFrame()
    spans_frame = pd.concat(spans, ignore_index=True) if spans else pd.DataFrame()
    if not close.empty:
        close = close.sort_index().reindex(sorted(close.columns), axis=1)
    return close, spans_frame
