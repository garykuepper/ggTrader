"""Run the registered early-window grid using the first-test ledger."""

from typing import Any

import pandas as pd
import pandas_market_calendars as market_calendars
from ed_arms import plan_arm_trades
from ed_dead_stock_first_test import (
    LOOKBACK,
    SKIP,
    START_CASH,
    TOP_N,
    _cell,
    _normalise_spans,
    drop_file_end_rows,
)
from ed_ledger_run import TaxParams, run_book
from ed_replay import exit_close, span_end_dates

COSTS = (0.00183, 0.005)
LOSSES = (0.0, 0.5, 1.0)


def simulate_arm(
    arm: str,
    inputs: dict[str, Any],
    start: str,
    end: str,
    *,
    top_n: int = TOP_N,
    lookback: int = LOOKBACK,
    skip: int = SKIP,
) -> list[dict[str, Any]]:
    """Run all six registered cells with shared target decisions."""
    close = inputs["close"]
    spans, censored = drop_file_end_rows(_normalise_spans(inputs["span_rows"]), end)
    if censored:
        raise ValueError("unexpected series censored at the end of the CRSP file")
    steps = plan_arm_trades(
        arm,
        close,
        inputs["eligible"],
        spans,
        top_n=top_n,
        lookback=lookback,
        skip=skip,
        start=start,
        end=end,
    )
    ends = span_end_dates(close, spans)
    flagged = spans.drop_duplicates("symbol").set_index("symbol")["has_successor"]
    successors = frozenset(s for s in ends.index if bool(flagged.get(s, False)))
    cells = []
    for loss in LOSSES:
        prices = exit_close(close, spans, loss)
        for cost in COSTS:
            book = run_book(
                close,
                steps,
                ends,
                prices,
                params=TaxParams(0.24, 0.15),
                one_way_cost=cost,
                start=start,
                end=end,
                successors=successors,
                start_cash=START_CASH,
            )
            cells.append(
                {
                    "arm": arm,
                    "cost": cost,
                    "dead_stock_loss": loss,
                    "tax_short": 0.24,
                    "tax_long": 0.15,
                    **_cell(book),
                }
            )
    return cells


def validate_calendar(index: pd.DatetimeIndex, comparison: pd.Series, start: str, end: str) -> None:
    """Require identical trading dates in the stock tape and comparison."""
    dates = pd.DatetimeIndex(index)
    if dates.tz is not None:
        dates = dates.tz_convert(None)
    dates = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    expected = market_calendars.get_calendar("NYSE").valid_days(start, end).tz_localize(None)
    observed = pd.DatetimeIndex(comparison.index)
    if dates.has_duplicates or not dates.equals(expected) or not observed.equals(expected):
        missing = expected.difference(observed)
        extra = observed.difference(expected)
        raise ValueError(
            f"calendar mismatch: {len(missing)} comparison dates missing, "
            f"{len(extra)} extra, stock tape matches: {dates.equals(expected)}"
        )
