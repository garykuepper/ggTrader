"""Behavior and frozen-history parity checks for the advisory A23 helper."""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest

from ggTrader.lab.a23_rebalance import Book, Lot, _trade_to, check_rebalance
from scripts import static_leverage_aftertax as a23


def check(
    spy: list[Lot],
    sso: list[Lot],
    *,
    cash: float = 0,
    new_cash: float = 0,
    month_end: bool = True,
) -> dict:
    """Check a small book at fixed, equal SPY and SSO prices."""
    return check_rebalance(
        {"SPY": spy, "SSO": sso},
        {"SPY": 100.0, "SSO": 100.0},
        pd.Timestamp("2026-09-30"),
        pd.Timestamp("2026-10-01"),
        month_end,
        cash=cash,
        new_cash=new_cash,
    )


def lot(shares: float, basis: float = 100.0, acquired: str = "2024-01-02") -> Lot:
    """Make a lot with a per-share basis."""
    return Lot(pd.Timestamp(acquired), shares, basis)


@pytest.mark.parametrize("sso_shares", [0.22, 0.38])
def test_band_edges_are_inclusive(sso_shares: float) -> None:
    plan = check([lot(1 - sso_shares)], [lot(sso_shares)])
    assert plan["status"] == "within band"
    assert not plan["sales"] and not plan["buys"]


def test_cash_restores_target_without_a_sale() -> None:
    plan = check([lot(0.8)], [lot(0.2)], new_cash=20)
    assert plan["status"] == "rebalance proposed"
    assert plan["sales"] == []
    assert len(plan["buys"]) == 1
    assert plan["buys"][0]["symbol"] == "SSO"
    assert plan["buys"][0]["estimated_cost"] == pytest.approx(14.285714, rel=1e-5)
    assert plan["leftover_cash"] == pytest.approx(20 - 14.285714, rel=1e-5)
    sso = 20 + plan["buys"][0]["shares"] * 100
    assert sso / (80 + sso) == pytest.approx(0.30, abs=0.0001)
    assert f"{sso / (80 + sso):.2%}" == "30.00%"


@pytest.mark.parametrize("new_cash", [2.50, 2.60])
def test_partial_cash_still_restores_target_with_one_buy(new_cash: float) -> None:
    plan = check([lot(0.8)], [lot(0.2)], new_cash=new_cash)
    assert len(plan["sales"]) == len(plan["buys"]) == 1
    assert plan["sales"][0]["symbol"] == "SPY"
    assert plan["buys"][0]["symbol"] == "SSO"
    spy = 80 - plan["sales"][0]["shares"] * 100
    sso = 20 + plan["buys"][0]["shares"] * 100
    assert sso / (spy + sso) == pytest.approx(0.30, abs=0.0001)
    assert f"{sso / (spy + sso):.2%}" == "30.00%"


def test_cash_threshold_uses_execution_marks() -> None:
    plan = check_rebalance(
        {"SPY": [lot(0.8)], "SSO": [lot(0.2)]},
        {"SPY": 100.0, "SSO": 100.0},
        pd.Timestamp("2026-09-30"),
        pd.Timestamp("2026-10-01"),
        True,
        new_cash=30,
        execution_prices={"SPY": 120.0, "SSO": 80.0},
    )
    assert plan["sales"] == []
    assert len(plan["buys"]) == 1
    sso = 16 + plan["buys"][0]["shares"] * 80
    assert f"{sso / (96 + sso):.2%}" == "30.00%"
    assert plan["leftover_cash"] == pytest.approx(30 - 25.142857, abs=0.01)


def test_next_session_one_year_boundary() -> None:
    plan = check(
        [lot(0.4)],
        [lot(0.1, acquired="2025-09-30"), lot(0.5, acquired="2025-10-01")],
    )
    assert [(sale["acquired_date"], sale["term"]) for sale in plan["sales"]] == [
        ("2025-09-30", "LT"),
        ("2025-10-01", "ST"),
    ]


def test_insufficient_cash_sells_highest_basis_long_term_first() -> None:
    plan = check(
        [lot(0.5)],
        [lot(0.1, 80), lot(0.1, 95), lot(0.3, 99, "2026-06-01")],
        new_cash=1,
    )
    assert plan["sales"]
    assert plan["sales"][0]["cost_basis_per_share"] == 95
    assert plan["sales"][0]["term"] == "LT"
    assert not plan["st_fallback"]
    assert sum(b["estimated_cost"] for b in plan["buys"]) > 1


def test_short_term_fallback_is_loud_and_taxed_at_st_rate() -> None:
    plan = check([lot(0.4)], [lot(0.01, 80), lot(0.59, 90, "2026-06-01")])
    assert plan["st_fallback"]
    assert any("SHORT-TERM LOTS REQUIRED" in warning for warning in plan["warnings"])
    st_sale = next(s for s in plan["sales"] if s["term"] == "ST")
    assert st_sale["estimated_tax"] == pytest.approx(st_sale["estimated_gain"] * 0.24)


def test_loss_sale_warns_for_recent_cross_leg_buy_and_plan_buy() -> None:
    plan = check(
        [lot(0.39), lot(0.01, acquired="2026-09-20")],
        [lot(0.59, 120), lot(0.01, 120, "2026-09-20")],
    )
    assert plan["sales"][0]["estimated_gain"] < 0
    assert any("SSO (same symbol)" in w for w in plan["warnings"])
    assert any("SPY (potential substantially identical)" in w for w in plan["warnings"])


def test_loss_warning_does_not_match_sold_lot_or_repeat_symbol() -> None:
    plan = check(
        [lot(0.4)],
        [lot(0.3, 120, "2026-09-20"), lot(0.3, 130, "2026-09-20")],
    )
    assert len([w for w in plan["warnings"] if "SSO (same symbol)" in w]) == 0
    assert len([w for w in plan["warnings"] if "purchases/reinvestments" in w]) == 1
    assert "BEFORE or AFTER" in plan["warnings"][0]


def test_no_trade_above_minimum_has_no_proposed_status() -> None:
    plan = check([lot(0.04)], [lot(0.01)])
    assert plan["status"] != "rebalance proposed"
    assert not plan["sales"] and not plan["buys"]


def test_off_cycle_never_proposes_a_trade() -> None:
    plan = check([lot(0.1)], [lot(0.9)], new_cash=1000, month_end=False)
    assert plan["status"] == "no action until month-end"
    assert plan["sso_weight"] == pytest.approx(0.9)
    assert not plan["sales"] and not plan["buys"]


def test_advisor_has_no_drawdown_or_trend_input() -> None:
    source = inspect.getsource(check_rebalance).lower()
    assert "drawdown" not in source
    assert "trend" not in source


def test_frozen_a23_shadow_rebalance_dates_and_lots_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay the real-SSO frozen path and compare every next-close A23 lot sale."""
    close, dist, _ = a23.load_prices(0.0075, frozen_inputs=Path("data/external/a23_inputs"))
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
    close, _, _ = a23.load_prices(0.0075, frozen_inputs=Path("data/external/a23_inputs"))
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
