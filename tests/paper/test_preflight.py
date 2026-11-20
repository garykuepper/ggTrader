"""Tests for the paper run preflight gates."""

from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from ggTrader.paper.gates import SignalGateError
from ggTrader.paper.preflight import Halted, PreflightGate, PreflightResult


@pytest.fixture
def setup_gate():
    broker = Mock()
    broker.get_clock.return_value = {"is_open": True, "next_close": None}
    notifier = Mock()
    bundle = {
        "sleeves": {"core": {"buys": [], "sells": [], "as_of": "2026-11-19"}},
        "weights": {"core": 1.0},
        "scale": 1.0,
        "fallback_used": False,
    }
    hooks = SimpleNamespace(
        generate_signals=Mock(return_value=bundle),
        membership_source=Mock(return_value="source"),
        now_pt=Mock(
            return_value=datetime(2026, 11, 19, 12, 45, tzinfo=ZoneInfo("America/Los_Angeles"))
        ),
        today_et=Mock(return_value=date(2026, 11, 19)),
        refresh_tape=Mock(return_value=["SPY"]),
    )
    skipped = []

    def make(*, dry_run=True, read_only=True):
        return PreflightGate(
            broker=broker,
            notifier=notifier,
            hooks=hooks,
            membership="db",
            read_only=read_only,
            dry_run=dry_run,
            skipped_writes=skipped,
        )

    return broker, notifier, hooks, bundle, skipped, make


def test_market_closed_returns_halted_with_the_old_notice_and_result(setup_gate):
    broker, notifier, hooks, _, _, make = setup_gate
    broker.get_clock.return_value = {"is_open": False}
    result = make().check()
    assert isinstance(result, Halted) and result.reason == "market_closed"
    assert result.result.as_dict() == {"buys": [], "sells": [], "errors": []}
    notifier.send.assert_called_once_with("<b>ℹ️ Market closed</b> — no trading this run.")
    hooks.generate_signals.assert_not_called()


def test_membership_source_failure_is_inside_the_signal_try(setup_gate):
    _, notifier, hooks, _, _, make = setup_gate
    error = RuntimeError("membership failed")
    hooks.membership_source.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        make().check()
    assert caught.value is error
    notifier.send.assert_called_once_with(
        "Paper trading failed: signal generation error\nmembership failed"
    )
    hooks.generate_signals.assert_not_called()


def test_clock_error_fails_open_and_skips_read_only_write(setup_gate):
    broker, _, hooks, _, skipped, make = setup_gate
    broker.get_clock.side_effect = RuntimeError("clock")
    assert isinstance(make().check(), PreflightResult)
    assert skipped == ["refresh_benchmark_tape"]
    hooks.refresh_tape.assert_not_called()


def test_signal_gate_error_notifies_and_reraises(setup_gate):
    _, notifier, hooks, _, _, make = setup_gate
    hooks.generate_signals.side_effect = SignalGateError("code", "bad")
    with pytest.raises(SignalGateError):
        make().check()
    notifier.send.assert_called_once_with("<b>🛑 Signal gate — no orders placed:</b>\n[code] bad")


def test_late_live_halts_and_dry_continues(setup_gate):
    _, notifier, hooks, _, _, make = setup_gate
    hooks.now_pt.return_value = datetime(
        2026, 11, 19, 13, 0, tzinfo=ZoneInfo("America/Los_Angeles")
    )
    halted = make(dry_run=False, read_only=False).check()
    assert isinstance(halted, Halted) and halted.reason == "late_run"
    assert halted.result.as_dict()["gate"] == "late_run"
    hooks.refresh_tape.assert_not_called()
    assert isinstance(make().check(), PreflightResult)
    assert any("DRY RUN late-run notice" in call.args[0] for call in notifier.send.call_args_list)


@pytest.mark.parametrize(
    "as_of,description",
    [(None, "unparseable"), ("2026-11-20", "1d in the future"), ("2026-11-10", "9d old")],
)
def test_stale_dates_halt_after_fallback_notice(setup_gate, as_of, description):
    _, notifier, _, bundle, _, make = setup_gate
    bundle["fallback_used"] = True
    bundle["sleeves"]["core"]["as_of"] = as_of
    result = make().check()
    assert isinstance(result, Halted) and result.reason == "stale_signals"
    messages = [call.args[0] for call in notifier.send.call_args_list]
    assert "Overlay fallback" in messages[0]
    assert description in messages[1]


def test_clock_is_read_after_signals_and_today_after_late_gate(setup_gate):
    broker, _, hooks, _, _, make = setup_gate
    events = []
    broker.get_clock.side_effect = lambda: events.append("clock") or {"is_open": True}
    hooks.membership_source.side_effect = lambda _: events.append("membership") or "source"
    bundle = hooks.generate_signals.return_value
    hooks.generate_signals.side_effect = lambda **_: events.append("signals") or bundle
    hooks.now_pt.side_effect = lambda: (
        events.append("now") or datetime(2026, 11, 19, 12, tzinfo=ZoneInfo("America/Los_Angeles"))
    )
    hooks.today_et.side_effect = lambda: events.append("today") or date(2026, 11, 19)
    make().check()
    assert events == ["clock", "membership", "signals", "now", "today"]


def test_empty_sleeves_raise_stop_iteration_through_check(setup_gate):
    _, _, _, bundle, _, make = setup_gate
    bundle["sleeves"] = {}
    with pytest.raises(StopIteration):
        make().check()


def test_live_refreshes_and_warns_when_nothing_refreshed(setup_gate, caplog):
    _, _, hooks, _, _, make = setup_gate
    hooks.refresh_tape.return_value = []
    assert isinstance(make(dry_run=False, read_only=False).check(), PreflightResult)
    assert "benchmark tape refresh returned no symbols" in caplog.text


def test_signal_error_notifies_and_reraises(setup_gate):
    _, notifier, hooks, _, _, make = setup_gate
    error = ValueError("signal failed")
    hooks.generate_signals.side_effect = error
    with pytest.raises(ValueError) as caught:
        make().check()
    assert caught.value is error
    notifier.send.assert_called_once_with(
        "Paper trading failed: signal generation error\nsignal failed"
    )
