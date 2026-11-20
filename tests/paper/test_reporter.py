"""Focused checks for the run reporter."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ggTrader.paper.models import ExecutionBook, SubmittedOrder
from ggTrader.paper.order_flow import FlowOutcome
from ggTrader.paper.reporter import RunReporter


@pytest.fixture
def rig():
    broker, notifier, ledger = MagicMock(), MagicMock(), MagicMock()
    broker.get_account.return_value = {"portfolio_value": 110.0, "cash": 10.0}
    broker.get_positions.return_value = {}
    ledger.get_earliest_snapshot.return_value = 100.0
    reporter = RunReporter(
        broker=broker,
        notifier=notifier,
        ledger=ledger,
        risk=SimpleNamespace(cfg=SimpleNamespace(max_positions=10)),
        skipped_writes=[],
        dry_run=False,
    )
    ctx = SimpleNamespace(today_et=date(2026, 1, 2))
    view = SimpleNamespace(
        split_corrections={},
        dividends={"total": 0.0, "new_accruals": []},
        untradable_seen=frozenset(),
    )
    signals = SimpleNamespace(weights={"core": 1.0}, scale=1.0)
    book = ExecutionBook()
    outcome = FlowOutcome(book, (), frozenset())
    return reporter, ctx, view, signals, outcome


def test_one_trade_alert_per_order_with_filled_or_intended_amount(rig):
    reporter, ctx, view, signals, outcome = rig
    outcome.book.orders.extend(
        [SubmittedOrder("a", "BUY", "A", 80, None), SubmittedOrder("b", "BUY", "B", 90, None)]
    )
    outcome.book.filled["a"] = {"status": "filled", "filled_qty": 2, "filled_avg_price": 30}
    reporter.finish(ctx, view, signals, outcome, None)
    calls = reporter.notifier.trade_alert.call_args_list
    assert len(calls) == 2
    assert calls[0].args == ("BUY", "A", 60, "a")
    assert calls[1].args == ("BUY", "B", 90, "b")
    assert calls[0].kwargs == {"qty": 2, "price": 30, "status": "filled"}
    assert calls[1].kwargs == {"qty": None, "price": None, "status": "pending"}


@pytest.mark.parametrize(
    "dry_run, text",
    [(True, "dry run; no orders submitted"), (False, "orders may have been submitted")],
)
@pytest.mark.parametrize("read", ["get_account", "get_positions"])
def test_post_trade_read_failure_text_depends_on_dry_or_live(rig, dry_run, text, read):
    reporter, ctx, view, signals, outcome = rig
    reporter.dry_run = dry_run
    getattr(reporter.broker, read).side_effect = ValueError("unavailable")
    with pytest.raises(ValueError):
        reporter.finish(ctx, view, signals, outcome, None)
    assert text in reporter.notifier.send.call_args.args[0]
    reporter.notifier.daily_summary.assert_not_called()
    reporter.ledger.log_snapshot.assert_not_called()


def test_invalid_post_trade_value_is_reported_and_no_snapshot_is_written(rig):
    reporter, ctx, view, signals, outcome = rig
    reporter.broker.get_account.return_value["portfolio_value"] = float("nan")
    result = reporter.finish(ctx, view, signals, outcome, None)
    assert "Invalid post-trade portfolio value; snapshot not saved" in result.errors
    reporter.ledger.log_snapshot.assert_not_called()


def test_summary_errors_exclude_snapshot_failure_but_result_includes_it(rig):
    reporter, ctx, view, signals, outcome = rig
    reporter.ledger.log_snapshot.side_effect = ValueError("unavailable")
    result = reporter.finish(ctx, view, signals, outcome, 90.0)
    kwargs = reporter.notifier.daily_summary.call_args.kwargs
    assert kwargs["errors"] == []
    assert result.errors == ["Daily snapshot could not be saved"]
    assert set(kwargs) == {
        "total_pnl",
        "unrealized_pnl",
        "split_corrections",
        "booked_equity",
        "dividend_total",
        "new_dividend_accruals",
        "errors",
        "starting_capital",
    }
    assert kwargs["starting_capital"] == 100.0
    assert kwargs["total_pnl"] == 10.0
    assert kwargs["booked_equity"] is None
    assert kwargs["dividend_total"] == 0.0
    assert result.buys is outcome.book.buys and result.sells is outcome.book.sells


def test_pnl_is_unavailable_if_any_unrealized_value_is_invalid(rig):
    reporter, ctx, view, signals, outcome = rig
    reporter.broker.get_positions.return_value = {"A": {"unrealized_pl": float("nan")}}
    reporter.finish(ctx, view, signals, outcome, None)
    assert reporter.notifier.daily_summary.call_args.kwargs["unrealized_pnl"] is None


def test_starting_snapshot_failure_keeps_the_summary_and_attempts_the_snapshot(rig):
    reporter, ctx, view, signals, outcome = rig
    reporter.ledger.get_earliest_snapshot.side_effect = ValueError("unavailable")
    result = reporter.finish(ctx, view, signals, outcome, None)
    assert "Starting snapshot unavailable; total P&L unavailable" in result.errors
    assert reporter.notifier.daily_summary.call_args.kwargs["starting_capital"] is None
    reporter.ledger.log_snapshot.assert_called_once()


def test_previous_value_guards_invalid_and_failed_reads(rig):
    reporter, *_ = rig
    reporter.ledger.get_latest_snapshot.return_value = float("nan")
    assert reporter.previous_value(date(2026, 1, 2)) is None
    reporter.ledger.get_latest_snapshot.side_effect = ValueError("unavailable")
    assert reporter.previous_value(date(2026, 1, 2)) is None
