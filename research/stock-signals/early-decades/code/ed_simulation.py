"""Run the registered early-window grid using the first-test ledger."""

from dataclasses import asdict
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
    coverage_block,
    drop_file_end_rows,
    holdings_audit,
    permno_double_holds,
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
    costs: tuple[float, ...] = COSTS,
    taxes: tuple[tuple[str, float, float], ...] = (("low", 0.24, 0.15),),
) -> list[dict[str, Any]]:
    """Run all six registered cells with shared target decisions."""
    close = inputs["close"]
    spans, censored = drop_file_end_rows(_normalise_spans(inputs["span_rows"]), end)
    if censored and pd.Timestamp(end) < pd.Timestamp("2024-12-31"):
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
    days = pd.DatetimeIndex(close.index)
    if days.tz is not None:
        days = days.tz_convert(None)
    days = days[(days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))]
    month_ends = list(pd.Series(days, index=days).groupby(days.to_period("M")).last())
    release_spans = inputs.get("release_spans")
    kinds = None
    if release_spans is not None and "kind" in release_spans:
        kinds = dict(zip(release_spans["symbol"], release_spans["kind"], strict=False))
    cells = []
    for loss in LOSSES:
        prices = exit_close(close, spans, loss)
        for cost in costs:
            for tax_name, short_rate, long_rate in taxes:
                book = run_book(
                    close,
                    steps,
                    ends,
                    prices,
                    params=TaxParams(short_rate, long_rate),
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
                        "tax": tax_name,
                        "tax_short": short_rate,
                        "tax_long": long_rate,
                        "coverage": coverage_block(
                            inputs["eligible"], close, ends, successors, book.trades, start, end
                        ),
                        "one_permno_one_column": permno_double_holds(book.trades, release_spans),
                        "holdings_audit": holdings_audit(
                            book.trades, month_ends, ends, successors, kinds
                        ),
                        "trades": [asdict(trade) for trade in book.trades],
                        "realized_gains": {
                            "short_term": sum(
                                float(event.payload["gain"])
                                for event in book.events
                                if event.kind == "sell" and event.payload["term"] == "ST"
                            ),
                            "long_term": sum(
                                float(event.payload["gain"])
                                for event in book.events
                                if event.kind == "sell" and event.payload["term"] == "LT"
                            ),
                        },
                        "account_value_daily": [
                            [str(stamp.date()), float(value)]
                            for stamp, value in book.account_value.items()
                        ],
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


def period_metrics(curve: pd.Series, start: str, end: str) -> dict[str, float | None]:
    """Measure a continuous account from the close before a period through its last close."""
    import math

    from ggTrader.engine import stats

    series = curve.copy()
    series.index = pd.DatetimeIndex(series.index).tz_localize(None)
    selected = series.loc[start:end]
    prior = series.loc[series.index < pd.Timestamp(start)]
    if selected.empty:
        raise ValueError("no account values in the requested period")
    path = pd.concat([prior.iloc[-1:], selected]) if not prior.empty else selected
    if path.isna().any() or path.iloc[0] <= 0:
        raise ValueError("missing values or nonpositive opening account value")
    growth = float(path.iloc[-1] / path.iloc[0])
    days = (path.index[-1] - path.index[0]).days
    returns = path.pct_change().dropna()
    sharpe = float(stats.sharpe(returns)) if len(returns) and (path > 0).all() else float("nan")
    drawdown = float((path / path.cummax() - 1).min())
    return {
        "growth": growth,
        "cagr": growth ** (365.25 / days) - 1 if days and growth >= 0 else None,
        "max_drawdown": drawdown if math.isfinite(drawdown) else None,
        "sharpe": sharpe if math.isfinite(sharpe) else None,
    }


def loss_monotone(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Identify cases where a larger death loss increases final account value."""
    violations = []
    for cost, tax in sorted({(cell["cost"], cell["tax"]) for cell in cells}):
        selected = sorted(
            (cell for cell in cells if cell["cost"] == cost and cell["tax"] == tax),
            key=lambda cell: cell["dead_stock_loss"],
        )
        for lower, higher in zip(selected, selected[1:]):
            if higher["after_tax_final"] > lower["after_tax_final"] * (1 + 1e-9):
                violations.append(
                    {
                        "cost": cost,
                        "tax": tax,
                        "lower_loss": lower["dead_stock_loss"],
                        "higher_loss": higher["dead_stock_loss"],
                        "lower_final": lower["after_tax_final"],
                        "higher_final": higher["after_tax_final"],
                    }
                )
    return violations


def comparison_metrics(returns: pd.Series, start: str, end: str) -> dict[str, float | None]:
    """Measure a cost-free comparison with tax on a sale at the period end."""
    selected = returns.loc[start:end]
    if selected.empty:
        raise ValueError("no comparison returns in the requested period")
    earlier = returns.loc[returns.index < pd.Timestamp(start)]
    if earlier.empty:
        curve = (1.0 + selected).cumprod()
    else:
        baseline = pd.Series([1.0], index=earlier.index[-1:])
        curve = pd.concat([baseline, (1.0 + selected).cumprod()])
    curve.iloc[-1] -= max(float(curve.iloc[-1]) - 1.0, 0.0) * 0.15
    return period_metrics(curve, start, end)
