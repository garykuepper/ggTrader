"""Late binding of the trader's collaborators."""

from unittest.mock import MagicMock

import pytest

from ggTrader.paper import trader


@pytest.mark.parametrize(
    ("field", "target", "args", "kwargs"),
    [
        ("today_et", "_today_et", (), {}),
        ("now_pt", "_now_pt", (), {}),
        ("new_uuid", "uuid4", (), {}),
        (
            "generate_signals",
            "generate_core_signals",
            (),
            {"persist_tape": False, "membership": "fixture"},
        ),
        ("membership_source", "live_membership_source", (None,), {}),
        ("refresh_tape", "refresh_benchmark_tape", (), {}),
        ("find_split_applied", "find_split_applied_symbols", ({}, [], {}, None), {}),
    ],
)
def test_hooks_resolve_trader_namespace_at_call_time(monkeypatch, field, target, args, kwargs):
    hooks = trader.PaperTrader.__new__(trader.PaperTrader)._hooks()
    replacement = MagicMock(return_value="patched")
    monkeypatch.setattr(trader, target, replacement)
    assert getattr(hooks, field)(*args, **kwargs) == "patched"
    replacement.assert_called_once_with(*args, **kwargs)


def test_clock_hooks_are_late_bound(monkeypatch):
    hooks = trader.PaperTrader.__new__(trader.PaperTrader)._hooks()
    clock = MagicMock()
    monkeypatch.setattr(trader, "datetime", clock)
    monkeypatch.setattr(trader, "time", clock)
    assert hooks.utcnow() is clock.now.return_value
    clock.now.assert_called_once_with(trader.timezone.utc)
    assert hooks.clock_time() is clock.time.return_value
    hooks.sleep(2)
    clock.sleep.assert_called_once_with(2)


def test_seams_resolve_instance_attributes_at_call_time():
    instance = trader.PaperTrader.__new__(trader.PaperTrader)
    seams = instance._seams()
    instance._submit_order = lambda *args: "patched"
    assert seams.submit_order("BUY", "ABC", 1, None) == "patched"


def test_default_hooks_read_nothing_at_construction(monkeypatch):
    replacement = MagicMock(side_effect=AssertionError("called during construction"))
    for name in (
        "_today_et",
        "_now_pt",
        "uuid4",
        "generate_core_signals",
        "live_membership_source",
        "refresh_benchmark_tape",
        "find_split_applied_symbols",
    ):
        monkeypatch.setattr(trader, name, replacement)
    monkeypatch.setattr(trader, "datetime", MagicMock(now=replacement))
    monkeypatch.setattr(trader, "time", MagicMock(time=replacement, sleep=replacement))
    trader.PaperTrader.__new__(trader.PaperTrader)._hooks()


@pytest.mark.parametrize(
    ("field", "target", "args"),
    [
        ("poll_orders", "_poll_orders", ([],)),
        ("reconcile_intents", "_reconcile_order_intents", ()),
        ("missed_exits", "_missed_exits", ({}, {}, set())),
        ("split_corrections", "_compute_split_corrections", ({}, None, None)),
        ("accrue_dividends", "_accrue_dividends", ({},)),
    ],
)
def test_other_seams_resolve_instance_attributes_at_call_time(field, target, args):
    instance = trader.PaperTrader.__new__(trader.PaperTrader)
    seams = instance._seams()
    replacement = MagicMock(return_value="patched")
    setattr(instance, target, replacement)
    assert getattr(seams, field)(*args) == "patched"
    replacement.assert_called_once_with(*args)
