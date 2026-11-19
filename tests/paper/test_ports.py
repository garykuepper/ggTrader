"""Tests of the Broker and Notifier ports and the typed records they carry."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from ggTrader.paper.alpaca_broker import AlpacaBroker
from ggTrader.paper.models import Position
from ggTrader.paper.notifier import DryRunNotifier, TelegramNotifier
from ggTrader.paper.ports import Broker, Notifier
from tests.paper.transcript.harness import FakeBroker, RecordingNotifier


def _public_methods(cls: type) -> dict[str, inspect.Signature]:
    return {
        name: inspect.signature(member)
        for name, member in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    }


def _shape(sig: inspect.Signature) -> list[tuple[str, object, object]]:
    """Parameter names, kinds and defaults; annotations are left out on purpose."""
    return [(p.name, p.kind, p.default) for p in sig.parameters.values()]


def _protocol_methods(proto: type) -> dict[str, inspect.Signature]:
    return {
        name: inspect.signature(member)
        for name, member in vars(proto).items()
        if inspect.isfunction(member) and not name.startswith("_")
    }


def test_broker_protocol_lists_the_eleven_public_methods() -> None:
    assert sorted(_protocol_methods(Broker)) == sorted(
        [
            "get_account",
            "get_positions",
            "get_untradable",
            "submit_buy",
            "submit_sell",
            "get_order",
            "get_order_by_client_id",
            "get_clock",
            "get_split_evidence",
            "get_split_corrections",
            "get_dividend_corrections",
        ]
    )


def test_alpaca_broker_satisfies_broker_protocol() -> None:
    # AlpacaBroker is not instantiated: its constructor needs credentials.
    proto = _protocol_methods(Broker)
    real = _public_methods(AlpacaBroker)
    assert set(real) == set(proto)
    for name, sig in proto.items():
        assert _shape(sig) == _shape(real[name]), name


def test_fake_broker_of_the_transcript_harness_satisfies_broker_protocol() -> None:
    proto = _protocol_methods(Broker)
    fake = _public_methods(FakeBroker)
    assert set(fake) == set(proto)
    for name, sig in proto.items():
        assert _shape(sig) == _shape(fake[name]), name


def test_notifier_protocols_match_telegram_and_dry_run_notifier() -> None:
    proto = _protocol_methods(Notifier)
    assert sorted(proto) == ["daily_summary", "send", "trade_alert"]
    telegram = _public_methods(TelegramNotifier)
    assert set(telegram) == set(proto)
    for name, sig in proto.items():
        assert _shape(sig) == _shape(telegram[name]), name

    # DryRunNotifier ends its two alerts in catch-alls; compare the parameters it names and check
    # that it accepts the protocol's call shape.
    dry = _public_methods(DryRunNotifier)
    assert set(dry) == set(proto)
    assert _shape(dry["send"]) == _shape(proto["send"])
    for name in ("trade_alert", "daily_summary"):
        named = [
            p.name
            for p in dry[name].parameters.values()
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY)
        ]
        assert named == [p.name for p in proto[name].parameters.values()][: len(named)], name
        # Positional only as far as the parameters it names (its `**_` catch-all takes the rest
        # by keyword, as the trader passes them); every name by keyword also binds.
        names_in_proto = list(proto[name].parameters)
        dry[name].bind(
            *[object()] * len(named), **{n: object() for n in names_in_proto[len(named) :]}
        )
        dry[name].bind(**{n: object() for n in names_in_proto})


def _bind_full_call(target: inspect.Signature, proto: inspect.Signature) -> None:
    """Bind every protocol parameter, positionally and then by keyword, to `target`."""
    names = list(proto.parameters)
    target.bind(*[object()] * len(names))
    target.bind(**{name: object() for name in names})


@pytest.mark.parametrize("name", ["trade_alert", "daily_summary"])
def test_recording_notifier_wrappers_accept_the_telegram_call_shape(name: str) -> None:
    # The wrappers are variadic recorders: argument acceptance is all that can be checked.
    telegram = inspect.signature(getattr(TelegramNotifier, name))
    wrapper = inspect.signature(getattr(RecordingNotifier, name))
    _bind_full_call(wrapper, telegram)
    assert _shape(inspect.signature(RecordingNotifier.send)) == _shape(
        inspect.signature(TelegramNotifier.send)
    )


def test_position_keys_match_alpaca_broker() -> None:
    with_stub = AlpacaBroker.__new__(AlpacaBroker)
    with_stub._client = MagicMock()
    with_stub._client.get_all_positions.return_value = [
        MagicMock(
            symbol="AAPL",
            qty="10",
            market_value="1500",
            current_price="150",
            avg_entry_price="145",
            unrealized_pl="50",
            unrealized_plpc="0.0345",
            change_today="0.012",
            cost_basis="1450",
        )
    ]
    keys = set(with_stub.get_positions()["AAPL"])
    assert keys == set(Position.__annotations__)


def test_protocols_are_runtime_checkable_by_method_presence() -> None:
    assert isinstance(TelegramNotifier(), Notifier)
    assert isinstance(DryRunNotifier(object()), Notifier)
    assert not isinstance(object(), Broker)
