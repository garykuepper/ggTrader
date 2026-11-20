"""Tests for the portfolio read and filtering step."""

import logging
from dataclasses import FrozenInstanceError, replace
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ggTrader.paper.config import TraderConfig
from ggTrader.paper.models import RunContext, StrategySignals
from ggTrader.paper.portfolio import PortfolioView


@pytest.fixture
def portfolio(monkeypatch):
    events = []
    broker = Mock()
    positions = {
        "HELD": {"qty": 2, "market_value": 100, "cost_basis": 100},
        "SWEEP": {"qty": 1},
        "FROZEN": {"qty": 3},
    }
    broker.get_account.side_effect = lambda: events.append("account") or {"portfolio_value": 500}
    broker.get_positions.side_effect = lambda: events.append("positions") or positions
    broker.get_untradable.side_effect = lambda symbols: events.append("untradable") or ["BLOCKED"]
    notifier = Mock()
    notifier.send.side_effect = lambda message: events.append(message)
    corporate_actions = SimpleNamespace(
        split_corrections=Mock(side_effect=lambda *args: events.append("splits") or {"HELD": 2}),
        missed_exits=Mock(side_effect=lambda *args: events.append("missed") or ["Z", "A"]),
        accrue_dividends=Mock(side_effect=lambda *args: events.append("dividends") or {"total": 1}),
    )
    cfg = replace(
        TraderConfig.from_env(dry_run=True, read_only=True),
        sweep_enabled=True,
        sweep_symbol="SWEEP",
        untradable=frozenset({"FROZEN"}),
    )
    ctx = RunContext(date(2026, 11, 19), cfg, frozenset())
    signals = StrategySignals(
        (("FROZEN", "core"), ("BLOCKED", "core"), ("BUY", "core")),
        ("Z",),
        {"HELD": "2026-11-17"},
        "2026-11-18",
        {},
        1.0,
    )

    def load():
        return PortfolioView.load(
            broker=broker,
            notifier=notifier,
            corporate_actions=corporate_actions,
            ctx=ctx,
            signals=signals,
        )

    return events, broker, notifier, corporate_actions, positions, ctx, load


def test_load_reads_in_todays_order(portfolio):
    events, _, _, _, _, _, load = portfolio
    load()
    assert events[:3] == ["account", "positions", "untradable"]
    assert "Untradable holdings" in events[3]
    assert "Untradable buy candidates" in events[4]
    assert events[5:] == ["splits", "missed", "dividends"]


def test_account_failure_halts_with_the_old_notice(portfolio, caplog):
    _, broker, notifier, _, _, _, load = portfolio
    broker.get_account.side_effect = ValueError("bad")
    with pytest.raises(ValueError, match="bad"):
        load()
    notifier.send.assert_called_once_with(
        "<b>🛑 HALTED:</b> Account data unavailable; no orders submitted"
    )
    broker.get_positions.assert_not_called()
    assert "Account read failed (ValueError); stopping before orders" in caplog.text


def test_position_failure_halts_even_if_the_notice_fails(portfolio, caplog):
    _, broker, notifier, corporate_actions, _, _, load = portfolio
    broker.get_positions.side_effect = ValueError("positions")
    notifier.send.side_effect = RuntimeError("notice")
    with pytest.raises(ValueError, match="positions"):
        load()
    assert "Position read failed (ValueError); stopping before orders" in caplog.text
    assert "Could not send position read failure notice" in caplog.text
    broker.get_untradable.assert_not_called()
    corporate_actions.split_corrections.assert_not_called()
    corporate_actions.missed_exits.assert_not_called()
    corporate_actions.accrue_dividends.assert_not_called()


def test_sweep_position_is_invisible_to_strategy_positions(portfolio):
    view = portfolio[-1]()
    assert "SWEEP" in view.positions and "SWEEP" not in view.strategy_positions


def test_untradable_holdings_are_frozen_and_their_buy_signal_dropped(portfolio):
    view = portfolio[-1]()
    assert "FROZEN" not in view.strategy_positions
    assert view.buy_candidates == (("BUY", "core"),)


def test_blocked_buy_candidates_are_announced(portfolio):
    events, *_, load = portfolio
    load()
    assert "BLOCKED" in events[4] and "FROZEN" not in events[4]


def test_orders_use_uncorrected_positions_and_reports_use_corrected(portfolio):
    view = portfolio[-1]()
    assert view.strategy_positions["HELD"]["qty"] == 2
    assert view.strategy_positions["HELD"]["market_value"] == 100
    assert view.corrected_positions["HELD"]["market_value"] == 200
    assert set(view.corrected_positions) == {"HELD"}


def test_portfolio_routes_raw_and_filtered_positions_to_their_collaborators(portfolio):
    _, broker, _, corporate_actions, positions, _, load = portfolio
    view = load()
    broker.get_untradable.assert_called_once_with(["BLOCKED", "BUY", "HELD"])
    corporate_actions.split_corrections.assert_called_once_with(
        positions, date(2026, 11, 5), date(2026, 11, 19)
    )
    corporate_actions.accrue_dividends.assert_called_once_with(positions)
    corporate_actions.missed_exits.assert_called_once_with(
        {"HELD": positions["HELD"]}, {"HELD": "2026-11-17"}, {"Z"}
    )
    assert set(view.strategy_positions) == set(view.corrected_positions) == {"HELD"}


def test_missing_portfolio_value_fails_before_downstream_reads_and_writes(portfolio):
    _, broker, _, corporate_actions, _, _, load = portfolio
    broker.get_account.side_effect = lambda: {}
    with pytest.raises(KeyError, match="portfolio_value"):
        load()
    broker.get_positions.assert_called_once()
    broker.get_untradable.assert_not_called()
    corporate_actions.split_corrections.assert_not_called()
    corporate_actions.missed_exits.assert_not_called()
    corporate_actions.accrue_dividends.assert_not_called()


def test_load_logs_filtering_corrections_and_missed_exits(portfolio, caplog):
    with caplog.at_level(logging.INFO, logger="ggTrader.paper.trader"):
        portfolio[-1]()
    for message in (
        "Untradable holdings excluded from strategy: ['FROZEN']",
        "Untradable buy candidates skipped: ['BLOCKED']",
        "Unapplied broker split corrections: {'HELD': 2}",
        "Catching up missed exits: ['Z', 'A']",
    ):
        assert message in caplog.text


def test_missed_exits_are_merged_into_the_sells_sorted(portfolio):
    assert portfolio[-1]().sells == ("A", "Z")


def test_the_view_is_frozen_and_shares_no_mutable_state_with_the_broker_reply(portfolio):
    _, broker, _, corporate_actions, positions, _, load = portfolio
    account = {"portfolio_value": 500}
    corrections = {"HELD": 2}
    dividends = {"total": 1, "items": ["original"]}
    broker.get_account.side_effect = lambda: account
    corporate_actions.split_corrections.side_effect = lambda *_: corrections
    corporate_actions.accrue_dividends.side_effect = lambda *_: dividends
    view = load()
    positions["HELD"]["qty"] = 42
    account["portfolio_value"] = 999
    corrections["HELD"] = 5
    dividends["items"].append("later")
    assert view.positions["HELD"]["qty"] == 2
    assert view.strategy_positions["HELD"]["qty"] == 2
    assert view.corrected_positions["HELD"]["market_value"] == 200
    assert view.account["portfolio_value"] == 500
    assert view.split_corrections["HELD"] == 2
    assert view.dividends["items"] == ["original"]
    with pytest.raises(FrozenInstanceError):
        view.sells = ()
    with pytest.raises((TypeError, FrozenInstanceError)):
        view.positions["HELD"]["qty"] = 3


def test_corrected_view_copies_positions_without_a_split(portfolio):
    _, _, _, _, positions, _, load = portfolio
    positions["PLAIN"] = {"qty": 4, "market_value": 400}
    view = load()
    positions["PLAIN"]["qty"] = 99
    assert view.corrected_positions["PLAIN"]["qty"] == 4


def test_load_uses_the_per_run_config_not_the_environment(portfolio, monkeypatch):
    monkeypatch.setenv("CASH_SWEEP_ENABLED", "false")
    view = portfolio[-1]()
    assert view.sweep_on and "SWEEP" not in view.strategy_positions
