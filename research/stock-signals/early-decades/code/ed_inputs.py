"""Read the previously used first-test window for the regression."""

import argparse
from typing import Any

import pandas as pd

from ggTrader.utils.study import bootstrap

bootstrap(__file__)

import ed_bars_reader as bars_reader  # noqa: E402
import ed_dead_stock_first_test as study  # noqa: E402
import ed_universe_frames as universe_frames  # noqa: E402

SPY = "SPY"


def read_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Read the release spans, the candidate closes and SPY from the read-only database.

    Two kinds of span rows are kept apart. ``release_spans`` is one row per release column
    (symbol, permno, kind, first and last date): eligibility and the one-company-one-column
    check read it. ``span_rows`` is what the close reader returns for the columns it read:
    the series that END inside the window (symbol, last_date, has_successor, total_loss), which
    the replay reads to find dead stocks and series that continue in a later CRSP spell.
    """
    from ggTrader.data.bars import connect_readonly

    start, end = study.WINDOWS[args.window]
    read_start = (pd.Timestamp(start) - pd.Timedelta(days=study.WARMUP_CALENDAR_DAYS)).strftime(
        "%Y-%m-%d"
    )
    # Eligibility looks at the previous month-end, so read fields from one month before.
    fields_start = (pd.Timestamp(start) - pd.DateOffset(months=1)).replace(day=1)
    conn = connect_readonly()
    try:
        release_spans = bars_reader.read_release_spans(conn, args.release_id)
        fields = universe_frames.month_end_fields(
            conn, args.crsp_load, fields_start.strftime("%Y-%m-%d"), end
        )
    finally:
        conn.close()
    days = pd.bdate_range(start, end)
    ever = universe_frames.eligibility(
        fields, release_spans, universe=args.universe, daily_index=days
    )
    candidates = sorted(ever.columns[ever.to_numpy().any(axis=0)])
    del ever
    provenance: dict[str, Any] = {}
    close, span_rows = bars_reader.read_close_batched(
        candidates, read_start, end, release_id=args.release_id, provenance_out=provenance
    )
    spy_provenance: dict[str, Any] = {}
    spy_frame, _ = bars_reader.read_close_batched(
        [SPY], read_start, end, release_id=args.release_id, provenance_out=spy_provenance
    )
    if SPY not in spy_frame.columns:
        raise SystemExit(f"SPY has no bars in release {args.release_id} for {read_start}..{end}")
    # Same rule on the real trading days. Only the candidate columns are needed: a winner among
    # one company's columns is always a candidate, so dropping the others changes nothing.
    wanted = release_spans[release_spans["symbol"].isin(candidates)]
    window_days = _naive(close.index)
    window_days = window_days[window_days >= pd.Timestamp(start)]
    eligible = universe_frames.eligibility(
        fields, wanted, universe=args.universe, daily_index=window_days
    ).reindex(columns=close.columns, fill_value=False)
    conn = connect_readonly()
    try:
        # The full release spans, as in --audit-only: "has a later spell" must see spells that
        # never qualify for the universe, so both entry points classify identically.
        audit = study.audit_false_deaths(
            conn,
            release_spans,
            crsp_load=args.crsp_load,
            window_start=start,
            window_end=end,
            crsp_end=study.CRSP_END,
        )
    finally:
        conn.close()
    return {
        "false_death_audit": audit,
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


def _naive(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(None) if index.tz is not None else index
