"""Behavior checks for the advisory A23 helper (frozen parity: research/allocation/_lib/tests)."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from ggTrader.lab.a23_rebalance import (
    WASH_LOSS_MIN_DOLLARS,
    Lot,
    check_rebalance,
)


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


def test_sub_cent_friction_loss_does_not_warn() -> None:
    """A flat-price sale loses only the 1 bp fill friction: no wash-sale warning."""
    plan = check([lot(0.8)], [lot(0.2)], new_cash=2.5)
    assert plan["sales"] and -WASH_LOSS_MIN_DOLLARS < plan["sales"][0]["estimated_gain"] < 0
    assert not any("WASH-SALE" in w for w in plan["warnings"])
