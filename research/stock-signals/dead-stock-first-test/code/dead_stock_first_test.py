"""Study logic for the dead-stock first test: the fixed grid, the sanity checks, the guard.

Everything here is pure (frames in, a plain dict out) except ``resolve_output``, which only
looks at a path. ``run.py`` wires the database reads and writes the JSON.

The fixed grid (the brief's registered values, never tuned): dead-stock loss 1.0, 0.5, 0.0
(the share of its last price a holder loses when a stock stops trading), one-way cost 6.4,
18.3 and 40 basis points of traded value (a basis point is 0.01 percent) and two tax sets.
That is 18 strategy cells per window and universe, plus 6 SPY cells (SPY does not depend on
the loss). The planner does not depend on the loss either, so it runs once.

Turnover is the yearly traded dollars (every buy and sell except the final liquidation, which
only closes the experiment) divided by that year's average pre-tax equity.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ledger_run import BookResult, TaxParams, run_book, run_spy
from replay import (
    DEAD_EXIT_REASON,
    LIQUIDATION_REASON,
    TargetStep,
    exit_close,
    plan_trades,
    span_end_dates,
)

from ggTrader.lab.wfo.report import curve_summary

WINDOWS: dict[str, tuple[str, str]] = {
    "dev": ("2005-01-03", "2019-12-31"),
    "holdout": ("2020-01-02", "2024-12-31"),
}
CRSP_END = "2024-12-31"
WARMUP_CALENDAR_DAYS = 500  # closes are read this far before the window for 252 + 21 days
TOP_N = 50
LOOKBACK = 252
SKIP = 21
START_CASH = 100_000.0
LOSSES = (1.0, 0.5, 0.0)
COSTS = (0.00064, 0.00183, 0.0040)  # one way: 6.4, 18.3 and 40 basis points
TAXES: dict[str, tuple[float, float]] = {"low": (0.24, 0.15), "high": (0.35, 0.20)}
REFERENCE = (1.0, 0.00183, "low")  # the cell that coverage and the death count are read from
SPY_TOLERANCE = 1e-3
MONOTONE_TOLERANCE = 1e-9  # relative; a rise above this as the loss grows is reported
HELD_QTY = 1e-9  # shares below this are float dust, not a holding
RESULTS_FOLDER = Path("results") / "dead_stock_first_test"
SPAN_ROW_COLUMNS = ["symbol", "last_date", "has_successor", "total_loss"]


def _bps(cost: float) -> str:
    return f"{cost * 1e4:g}"


def cell_key(loss: float, cost: float, tax: str) -> str:
    """The dictionary key of one strategy cell."""
    return f"loss{loss:g}_cost{_bps(cost)}bps_tax{tax}"


def spy_key(cost: float, tax: str) -> str:
    """The dictionary key of one SPY cell."""
    return f"cost{_bps(cost)}bps_tax{tax}"


def resolve_output(window: str, universe: str, out: Path | None, results_dir: Path) -> Path:
    """Where this run writes, refusing a second holdout run.

    The holdout always writes ``holdout-<universe>.json`` in ``results_dir`` (``out`` is
    ignored) and stops with ``SystemExit`` naming the file when it already exists. A development
    run may be repeated.
    """
    if window == "holdout":
        target = results_dir / f"holdout-{universe}.json"
        if target.exists():
            raise SystemExit(
                f"refusing to run: the holdout was already run and {target} exists. "
                "The holdout is looked at once; the owner must say so before it is repeated."
            )
        return target
    return out if out is not None else results_dir / f"dev-{universe}.json"


def _naive_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.tz_convert(None) if index.tz is not None else index


def _normalise_spans(span_rows: pd.DataFrame) -> pd.DataFrame:
    """Ended-series rows with every column the replay reads (an empty read has none)."""
    frame = span_rows.copy()
    if "symbol" not in frame.columns:
        return pd.DataFrame({name: pd.Series(dtype=object) for name in SPAN_ROW_COLUMNS})
    for name in ("has_successor", "total_loss"):
        if name not in frame.columns:
            frame[name] = False
        frame[name] = frame[name].fillna(False).astype(bool)
    return frame


def yearly_turnover(result: BookResult) -> dict[str, float]:
    """Traded dollars per calendar year over that year's average pre-tax equity."""
    traded: dict[int, float] = {}
    for trade in result.trades:
        if trade.reason == LIQUIDATION_REASON:
            continue
        traded[trade.date.year] = traded.get(trade.date.year, 0.0) + abs(trade.qty * trade.price)
    equity = result.pretax_equity
    mean_equity = equity.groupby(equity.index.year).mean()
    return {
        str(int(year)): float(traded.get(int(year), 0.0) / level) if level > 0 else math.nan
        for year, level in mean_equity.items()
    }


def _cell(result: BookResult) -> dict[str, Any]:
    """The reported fields of one finished book."""
    curve = curve_summary(result.pretax_equity)
    turnover = yearly_turnover(result)
    return {
        "after_tax_final": result.after_tax_final,
        "cagr_pretax": result.cagr_pretax,
        "cagr_after_tax": result.cagr_after_tax,
        "tax_paid": result.tax_paid,
        "cost_paid": result.cost_paid,
        "unpaid_tax_left": result.unpaid_tax_left,
        "n_dead_exits": result.n_dead_exits,
        "dead_exit_writeoff": result.dead_exit_writeoff,
        "skipped_buys": result.skipped_buys,
        "sharpe": curve["sharpe"],
        "max_drawdown_pct": curve["max_drawdown_pct"],
        "turnover_by_year": turnover,
        "mean_yearly_turnover": float(np.nanmean(list(turnover.values()))) if turnover else 0.0,
        "n_trades": len(result.trades),
        "chain_verified": result.chain_verified,
        "min_cash": result.min_cash,
    }


def permno_double_holds(trades: list, release_spans: pd.DataFrame | None) -> dict[str, Any]:
    """Days on which one PERMNO (one company) was held in two columns at the same time.

    Args:
        trades: A book's fills, in order.
        release_spans: Release spans with ``symbol`` and ``permno``; ``None`` skips the check
            and says so (``checked`` is False).

    Returns:
        ``checked`` and the ``violations`` (first day, permno, symbols), never hidden.
    """
    if release_spans is None or not {"symbol", "permno"}.issubset(release_spans.columns):
        return {"checked": False, "violations": []}
    permno = pd.to_numeric(release_spans.drop_duplicates("symbol").set_index("symbol")["permno"])
    qty: dict[str, float] = {}
    seen: set[tuple[int, tuple[str, ...]]] = set()
    violations: list[dict[str, Any]] = []

    def check(day: pd.Timestamp) -> None:
        by_company: dict[int, list[str]] = {}
        for symbol, held in qty.items():
            company = permno.get(symbol)
            if held > HELD_QTY and company is not None and np.isfinite(company):
                by_company.setdefault(int(company), []).append(symbol)
        for company, symbols in by_company.items():
            names = tuple(sorted(symbols))
            if len(names) > 1 and (company, names) not in seen:
                seen.add((company, names))
                violations.append(
                    {"first_day": str(day.date()), "permno": company, "symbols": list(names)}
                )

    previous = None
    for trade in trades:
        if previous is not None and trade.date != previous:
            check(previous)
        previous = trade.date
        qty[trade.symbol] = qty.get(trade.symbol, 0.0) + trade.qty
    if previous is not None:
        check(previous)
    return {"checked": True, "violations": violations}


def coverage_block(
    eligible: pd.DataFrame,
    close: pd.DataFrame,
    ends: pd.Series,
    successors: frozenset[str],
    trades: list,
    start: str,
    end: str,
) -> dict[str, dict[str, Any]]:
    """Per calendar year: how many stocks were eligible, died, and were held when they died."""
    days = _naive_index(eligible.index)
    inside = (days >= pd.Timestamp(start)) & (days <= pd.Timestamp(end))
    window = eligible.set_axis(days, axis=0).loc[inside].reindex(columns=close.columns)
    window = window.fillna(False).astype(bool)
    counts = window.sum(axis=1)
    month_end = counts.groupby(counts.index.to_period("M")).last()
    per_month = month_end.groupby(month_end.index.year).mean()
    ever = set(window.columns[window.to_numpy().any(axis=0)])
    stamps = _naive_index(close.index)
    years: dict[str, dict[str, Any]] = {
        str(int(y)): {
            "eligible_per_month_mean": float(per_month[y]),
            "ever_eligible_dead": 0,
            "ever_eligible_bought_out": 0,
            "held_at_death": 0,
            "held_at_buyout": 0,
            "dollars_written_off": 0.0,
        }
        for y in per_month.index
    }
    low, high = pd.Timestamp(start), pd.Timestamp(end)
    for symbol, last in ends.items():
        when = pd.Timestamp(last)
        when = when.tz_convert(None) if when.tzinfo else when
        key = str(when.year)
        if symbol in ever and low <= when <= high and key in years:
            field = "ever_eligible_bought_out" if symbol in successors else "ever_eligible_dead"
            years[key][field] += 1
    for trade in trades:
        if trade.reason != DEAD_EXIT_REASON or str(trade.date.year) not in years:
            continue
        year = years[str(trade.date.year)]
        if trade.symbol in successors:
            year["held_at_buyout"] += 1
            continue
        year["held_at_death"] += 1
        row = stamps.get_loc(trade.date)
        last_close = float(close.iat[row, close.columns.get_loc(trade.symbol)])
        year["dollars_written_off"] += abs(trade.qty) * (last_close - trade.price)
    return years


def spy_zero_cost_check(spy_close: pd.Series, start: str, end: str) -> dict[str, Any]:
    """SPY through the book at zero cost and no tax against SPY's own close ratio."""
    result = run_spy(spy_close, params=TaxParams(0.0, 0.0), one_way_cost=0.0, start=start, end=end)
    stamps = _naive_index(spy_close.index)
    prices = spy_close.to_numpy(dtype=float)[
        (stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))
    ]
    own = float(prices[-1] / prices[0])
    book = result.after_tax_final / START_CASH
    difference = book / own - 1.0
    return {
        "book_growth": book,
        "close_ratio": own,
        "relative_difference": difference,
        "within_tolerance": bool(abs(difference) <= SPY_TOLERANCE),
        "tolerance": SPY_TOLERANCE,
    }


def loss_monotone_check(strategy: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """After-tax final value must not rise as the loss rises (0.0 to 0.5 to 1.0)."""
    violations = []
    for cost in COSTS:
        for tax in TAXES:
            finals = [strategy[cell_key(loss, cost, tax)]["after_tax_final"] for loss in LOSSES]
            for lower_loss, higher_loss, a, b in zip(
                LOSSES[::-1], LOSSES[::-1][1:], finals[::-1], finals[::-1][1:], strict=False
            ):
                if b > a * (1.0 + MONOTONE_TOLERANCE):
                    violations.append(
                        {
                            "cost": cost,
                            "tax": tax,
                            "lower_loss": lower_loss,
                            "higher_loss": higher_loss,
                            "final_at_lower_loss": a,
                            "final_at_higher_loss": b,
                        }
                    )
    return {"violations": violations}


def run_study(
    close: pd.DataFrame,
    spy_close: pd.Series,
    eligible: pd.DataFrame,
    span_rows: pd.DataFrame,
    *,
    start: str,
    end: str,
    release_spans: pd.DataFrame | None = None,
    start_cash: float = START_CASH,
) -> dict[str, Any]:
    """Run the fixed grid over one window and return every cell, the coverage and the checks.

    Args:
        close: Daily closes, trading days x release columns, with warm-up history.
        spy_close: SPY daily closes.
        eligible: Boolean eligibility (``universe_frames.eligibility``).
        span_rows: The ended-series rows the reader returns (``symbol``, ``last_date``,
            ``has_successor``, ``total_loss``): what the replay needs to find dead stocks.
        start: First day of the window.
        end: Last day of the window.
        release_spans: The release's span rows with ``permno`` (for the one-company-one-column
            check); ``None`` skips that check and says so.
        start_cash: Opening cash.

    Returns:
        ``strategy`` (18 cells), ``spy`` (6 cells), ``coverage`` and ``sanity``.
    """
    spans = _normalise_spans(span_rows)
    steps: list[TargetStep] = plan_trades(  # the plan does not depend on the loss: once
        close,
        eligible,
        spans,
        top_n=TOP_N,
        lookback=LOOKBACK,
        skip=SKIP,
        dead_loss=LOSSES[0],
        start=start,
        end=end,
    )
    ends = span_end_dates(close, spans)
    flagged = spans.drop_duplicates("symbol").set_index("symbol")["has_successor"]
    successors = frozenset(s for s in ends.index if bool(flagged.get(s, False)))
    taxes = {name: TaxParams(*rates) for name, rates in TAXES.items()}
    strategy: dict[str, dict[str, Any]] = {}
    spy: dict[str, dict[str, Any]] = {}
    for cost in COSTS:
        for tax, params in taxes.items():
            book = run_spy(
                spy_close,
                params=params,
                one_way_cost=cost,
                start=start,
                end=end,
                start_cash=start_cash,
            )
            spy[spy_key(cost, tax)] = {"one_way_cost": cost, "tax": tax, **_cell(book)}
    reference_trades: list = []
    doubles: list[dict[str, Any]] = []
    checked = False
    for loss in LOSSES:
        prices = exit_close(close, spans, loss)
        for cost in COSTS:
            for tax, params in taxes.items():
                book = run_book(
                    close,
                    steps,
                    ends,
                    prices,
                    params=params,
                    one_way_cost=cost,
                    start=start,
                    end=end,
                    successors=successors,
                    start_cash=start_cash,
                )
                comparison = spy[spy_key(cost, tax)]["cagr_after_tax"]
                strategy[cell_key(loss, cost, tax)] = {
                    "loss": loss,
                    "one_way_cost": cost,
                    "tax": tax,
                    **_cell(book),
                    "gap_cagr_after_tax_pp": (book.cagr_after_tax - comparison) * 100.0,
                }
                check = permno_double_holds(book.trades, release_spans)
                checked = checked or check["checked"]
                doubles += [
                    {"loss": loss, "cost": cost, "tax": tax, **v} for v in check["violations"]
                ]
                if (loss, cost, tax) == REFERENCE:
                    reference_trades = book.trades
        del prices
    held_at_death = sum(
        1 for t in reference_trades if t.reason == DEAD_EXIT_REASON and t.symbol not in successors
    )
    sanity = {
        "spy_zero_cost_growth": spy_zero_cost_check(spy_close, start, end),
        "loss_monotone": loss_monotone_check(strategy),
        "held_at_death": held_at_death,
        "no_dead_stock_held": held_at_death == 0,
        "one_permno_one_column": {"checked": checked, "violations": doubles},
        "reference_cell": cell_key(*REFERENCE),
    }
    return {
        "strategy": strategy,
        "spy": spy,
        "coverage": coverage_block(eligible, close, ends, successors, reference_trades, start, end),
        "sanity": sanity,
        "n_rebalance_steps": len(steps),
        "n_columns": int(close.shape[1]),
        "n_ended_series_in_window": int(len(ends)),
    }
