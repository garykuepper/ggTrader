"""Frozen-history parity checks for the A23 backtest library against the live advisory helper.

Both tests replay the real-SSO frozen inputs (``data/external/a23_inputs``) through the
backtest simulator and compare it with ``ggTrader.lab.a23_rebalance``; they live next to the
library they import (``a23_backtest``), not with the pure advisory tests in ``tests/lab``.
"""

from __future__ import annotations

import a23_backtest as a23
import pandas as pd
import pytest

from ggTrader.lab.a23_rebalance import Book, _trade_to, check_rebalance
from ggTrader.utils.study import repo_root

pytestmark = pytest.mark.study

FROZEN_INPUTS = repo_root(__file__) / "data/external/a23_inputs"


def test_frozen_a23_shadow_rebalance_dates_and_lots_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay the real-SSO frozen path and compare every next-close A23 lot sale."""
    close, dist, _ = a23.load_prices(0.0075, frozen_inputs=FROZEN_INPUTS)
    close = close.loc["2006-06-21":"2026-09-29"]
    dist = dist.loc[close.index]
    dates = close.index
    month_ends = set(close.resample("ME").last().index.map(lambda t: dates[dates <= t][-1]))
    reference: list[tuple[str, list[tuple[str, str, float]]]] = []
    original_trade = a23._trade_to

    def record_trade(book: Book, targets: dict, px: pd.Series, day: pd.Timestamp) -> None:
        before = {
            symbol: [(lot, lot.qty) for lot in book.lots[symbol]] for symbol in ("SPY", "SSO")
        }
        original_trade(book, targets, px, day)
        sold = [
            (symbol, str(lot.date.date()), lot.basis)
            for symbol in ("SPY", "SSO")
            for lot, qty in sorted(
                before[symbol],
                key=lambda pair: (
                    pair[0].date + pd.DateOffset(years=1) >= day,
                    -pair[0].basis,
                ),
            )
            if qty - lot.qty > 1e-12
        ]
        reference.append((str(day.date()), sold))

    monkeypatch.setattr(a23, "_trade_to", record_trade)
    a23.run(close, dist, "2006-06-21", "2026-09-29", 0.30, 0.08, 0.24, 0.15, False)

    book = Book(0.24, 0.15, False)
    targets = {"SPY": 0.70, "SSO": 0.30, "BIL": 0.0}
    for symbol, weight in targets.items():
        book.buy(symbol, 100_000 * weight, close.iloc[0][symbol], dates[0])
    book.cash = 0.0
    shadow: list[tuple[str, list[tuple[str, str, float]]]] = []
    last_year = dates[0].year
    for i, day in enumerate(dates):
        px = close.loc[day]
        for symbol in targets:
            if dist.at[day, symbol] > 0:
                book.distributions(symbol, dist.at[day, symbol], px[symbol], day)
        if day.year != last_year:
            book.year_end()
            last_year = day.year
        if day.month == 4 and day.day >= 15 and book.tax_due > 0:
            book.pay_tax(px, day)
        if day not in month_ends or i == len(dates) - 1:
            continue
        next_day = dates[i + 1]
        plan = check_rebalance(
            {s: book.lots[s] for s in ("SPY", "SSO")},
            {s: px[s] for s in ("SPY", "SSO")},
            day,
            next_day,
            True,
            execution_prices={s: close.at[next_day, s] for s in ("SPY", "SSO")},
        )
        if plan["status"] == "rebalance proposed":
            shadow.append(
                (
                    str(next_day.date()),
                    [
                        (s["symbol"], s["acquired_date"], s["cost_basis_per_share"])
                        for s in plan["sales"]
                    ],
                )
            )
            _trade_to(book, targets, close.loc[next_day], next_day)
    assert [day for day, _ in shadow] == [day for day, _ in reference]
    for (_, expected), (_, actual) in zip(reference, shadow, strict=True):
        assert len(actual) == len(expected)
        for (a_symbol, a_day, a_basis), (e_symbol, e_day, e_basis) in zip(
            actual, expected, strict=True
        ):
            assert (a_symbol, a_day) == (e_symbol, e_day)
            assert a_basis == pytest.approx(e_basis)


def test_frozen_marks_match_sold_and_bought_shares() -> None:
    close, _, _ = a23.load_prices(0.0075, frozen_inputs=FROZEN_INPUTS)
    marks = close.loc["2026-09-29"]
    day = pd.Timestamp("2026-09-30")
    book = Book(0.24, 0.15, False)
    for symbol, dollars in (("SPY", 80_000), ("SSO", 20_000)):
        book.buy(symbol, dollars, marks[symbol], pd.Timestamp("2024-01-02"))
    book.cash = 0.0
    before = book.qty("SSO")
    sold_lot = book.lots["SPY"][0]
    initial_qty = sold_lot.qty
    plan = check_rebalance(
        {symbol: book.lots[symbol] for symbol in ("SPY", "SSO")},
        {symbol: marks[symbol] for symbol in ("SPY", "SSO")},
        pd.Timestamp("2026-09-29"),
        day,
        True,
    )
    _trade_to(
        book,
        {"SPY": 0.70, "SSO": 0.30, "BIL": 0.0},
        marks,
        day,
        skip_buy_sold=True,
    )
    assert sum(sale["shares"] for sale in plan["sales"]) == pytest.approx(
        initial_qty - sold_lot.qty
    )
    assert len(plan["buys"]) == 1
    assert plan["buys"][0]["symbol"] == "SSO"
    assert plan["buys"][0]["shares"] == pytest.approx(book.qty("SSO") - before)
