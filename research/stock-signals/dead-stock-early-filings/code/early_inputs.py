"""Window constants, read-only input loading and drift checks for dead-stock-early-filings."""

from __future__ import annotations

from typing import Any, Literal

import ef_bars_reader as bars_reader
import ef_dead_stock_first_test as study
import ef_universe_frames as universe_frames
import pandas as pd

WINDOWS: dict[str, tuple[str, str]] = {
    "early": ("2000-01-03", "2004-12-31"),
}
SPY_FIRST_DAY = "2000-01-03"
SPY = "SPY"
UNIVERSE: Literal["A"] = "A"


def read_inputs(window: str, release_id: int, crsp_load: int) -> dict[str, Any]:
    """Read the release spans, universe A candidate closes and SPY from the read-only database.

    Args:
        window: Registered window name (``"early"``).
        release_id: Published combined release id, pinned by the caller.
        crsp_load: Raw CRSP daily load id used for point-in-time eligibility.

    Returns:
        Dictionary with keys ``close``, ``spy_close``, ``eligible``, ``span_rows``,
        ``release_spans`` and ``data_provenance``.

    Raises:
        ValueError: Unknown window name (checked before opening any database connection).
        SystemExit: SPY has no bars in the pinned release for the requested read span.
    """
    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}; expected one of {sorted(WINDOWS)}")

    from ggTrader.data.bars import connect_readonly

    start, end = WINDOWS[window]
    read_start = (pd.Timestamp(start) - pd.Timedelta(days=study.WARMUP_CALENDAR_DAYS)).strftime(
        "%Y-%m-%d"
    )
    # Eligibility looks at the previous month-end, so read fields from one month before.
    fields_start = (pd.Timestamp(start) - pd.DateOffset(months=1)).replace(day=1)
    conn = connect_readonly()
    try:
        release_spans = bars_reader.read_release_spans(conn, release_id)
        fields = universe_frames.month_end_fields(
            conn, crsp_load, fields_start.strftime("%Y-%m-%d"), end
        )
    finally:
        conn.close()
    days = pd.bdate_range(start, end)
    ever = universe_frames.eligibility(fields, release_spans, universe=UNIVERSE, daily_index=days)
    candidates = sorted(ever.columns[ever.to_numpy().any(axis=0)])
    del ever
    provenance: dict[str, Any] = {}
    close, span_rows = bars_reader.read_close_batched(
        candidates, read_start, end, release_id=release_id, provenance_out=provenance
    )
    spy_provenance: dict[str, Any] = {}
    spy_frame, _ = bars_reader.read_close_batched(
        [SPY], read_start, end, release_id=release_id, provenance_out=spy_provenance
    )
    if SPY not in spy_frame.columns:
        raise SystemExit(f"SPY has no bars in release {release_id} for {read_start}..{end}")
    wanted = release_spans[release_spans["symbol"].isin(candidates)]
    window_days = _naive(close.index)
    window_days = window_days[window_days >= pd.Timestamp(start)]
    eligible = universe_frames.eligibility(
        fields, wanted, universe=UNIVERSE, daily_index=window_days
    ).reindex(columns=close.columns, fill_value=False)
    return {
        "close": close,
        "spy_close": spy_frame[SPY],
        "eligible": eligible,
        "span_rows": span_rows,
        "release_spans": release_spans,
        "data_provenance": {
            "candidates": provenance,
            "spy": spy_provenance,
            "n_candidate_columns": len(candidates),
        },
    }


def read_drift(release_id: int) -> list[str]:
    """Drifted linked Yahoo columns of the pinned release (metadata only), before any claim."""
    from ggTrader.data.bars import connect_readonly

    conn = connect_readonly()
    try:
        return bars_reader.read_linked_drift(conn, release_id)
    finally:
        conn.close()


def drift_in(inputs: dict[str, Any]) -> list[str]:
    """Symbols the close reads reported as drifted (empty when none)."""
    found: set[str] = set()
    for part in (inputs.get("data_provenance") or {}).values():
        if isinstance(part, dict) and part.get("drift"):
            found.update(part.get("drift_symbols") or ["(drift reported without symbols)"])
    return sorted(found)


def _naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Drop timezone information from a DatetimeIndex while keeping UTC wall times."""
    return index.tz_convert(None) if index.tz is not None else index
