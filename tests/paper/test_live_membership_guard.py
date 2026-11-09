"""A live run must read the latest database vintage, so the freshness gate cannot be skipped."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ggTrader.paper.gates import SignalGateError
from ggTrader.paper.signal_runner import require_live_membership_spec
from ggTrader.paper.trader import PaperTrader, run_paper_trading


@pytest.mark.parametrize("spec", ["fixture:sp500_v1_2026-06-02", "db:3", None])
def test_a_live_run_on_a_fixture_a_pinned_vintage_or_nothing_is_refused(spec, monkeypatch):
    monkeypatch.setenv("GGT_MEMBERSHIP", "fixture:sp500_v1_2026-06-02")
    with pytest.raises(SignalGateError) as caught:
        require_live_membership_spec(spec, dry_run=False)
    assert caught.value.code == "membership_not_live"


def test_the_environment_is_used_when_no_spec_is_passed(monkeypatch):
    monkeypatch.setenv("GGT_MEMBERSHIP", "db")
    require_live_membership_spec(None, dry_run=False)
    monkeypatch.setenv("GGT_MEMBERSHIP", "db:2")
    with pytest.raises(SignalGateError):
        require_live_membership_spec(None, dry_run=False)


def test_a_live_run_on_the_latest_vintage_and_any_dry_run_are_allowed(monkeypatch):
    monkeypatch.delenv("GGT_MEMBERSHIP", raising=False)
    require_live_membership_spec("db", dry_run=False)
    for spec in ("fixture:sp500_v1_2026-06-02", "db:3", None):
        require_live_membership_spec(spec, dry_run=True)


def test_the_trader_refuses_a_live_fixture_before_it_touches_the_broker():
    broker, notifier = MagicMock(), MagicMock()
    with pytest.raises(SignalGateError, match="membership_not_live"):
        PaperTrader(broker, notifier, dry_run=False, membership="fixture:sp500_v1_2026-06-02")
    assert broker.method_calls == [] and notifier.method_calls == []
    PaperTrader(broker, notifier, dry_run=True, membership="fixture:sp500_v1_2026-06-02")


def test_run_paper_trading_refuses_before_any_broker_object_exists():
    with patch("ggTrader.paper.trader.AlpacaBroker") as broker_class:
        with pytest.raises(SignalGateError, match="membership_not_live"):
            run_paper_trading(dry_run=False, membership="fixture:sp500_v1_2026-06-02")
        with pytest.raises(SignalGateError, match="membership_not_live"):
            run_paper_trading(dry_run=False, membership=None)
    broker_class.assert_not_called()
