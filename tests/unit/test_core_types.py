"""Phase 0 core-types tests: frozen models, Decimal/float discipline, UTC discipline, enum JSON."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from ggTrader.core import (
    AssetClass,
    Direction,
    Fill,
    Instrument,
    Order,
    Signal,
    Venue,
)


def _btc() -> Instrument:
    return Instrument(
        symbol="BTC-USD",
        asset_class=AssetClass.CRYPTO_SPOT,
        venue=Venue.KRAKEN_SPOT,
        quote_currency="USD",
        base_currency="BTC",
        tick_size=Decimal("0.01"),
        min_order_size=Decimal("0.0001"),
        maker_fee_bps=Decimal("16"),
        taker_fee_bps=Decimal("26"),
        calendar_id="crypto_24_7",
    )


def test_instrument_is_frozen():
    inst = _btc()
    with pytest.raises(ValidationError):
        inst.symbol = "ETH-USD"  # type: ignore[misc]


def test_instrument_rejects_float_tick_size():
    with pytest.raises(ValidationError):
        Instrument(
            symbol="BTC-USD",
            asset_class=AssetClass.CRYPTO_SPOT,
            venue=Venue.KRAKEN_SPOT,
            quote_currency="USD",
            base_currency="BTC",
            tick_size=0.01,  # type: ignore[arg-type]  # float not allowed under strict mode
            min_order_size=Decimal("0.0001"),
            maker_fee_bps=Decimal("16"),
            taker_fee_bps=Decimal("26"),
            calendar_id="crypto_24_7",
        )


def test_signal_rejects_naive_datetime():
    with pytest.raises(ValidationError):
        Signal(
            ts=datetime(2026, 1, 1, 0, 0, 0),
            instrument=_btc(),
            direction=Direction.LONG,
            strategy_id="s1",
        )


def test_signal_accepts_utc_datetime():
    sig = Signal(
        ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        instrument=_btc(),
        direction=Direction.LONG,
        strategy_id="s1",
    )
    assert sig.direction is Direction.LONG
    assert sig.confidence == Decimal("1.0")


def test_signal_normalizes_non_utc_tz_to_utc():
    est = timezone(timedelta(hours=-5))
    sig = Signal(
        ts=datetime(2026, 1, 1, 12, 0, tzinfo=est),
        instrument=_btc(),
        direction=Direction.LONG,
        strategy_id="s1",
    )
    assert sig.ts.tzinfo is not None
    assert sig.ts.utcoffset() == timedelta(0)


def test_order_rejects_naive_datetime():
    with pytest.raises(ValidationError):
        Order(
            ts=datetime(2026, 1, 1),
            instrument=_btc(),
            side=Direction.LONG,
            quantity=Decimal("0.1"),
            order_type="market",
            client_order_id="abc",
        )


def test_fill_rejects_naive_datetime():
    with pytest.raises(ValidationError):
        Fill(
            ts=datetime(2026, 1, 1),
            order_id="o1",
            instrument=_btc(),
            side=Direction.LONG,
            quantity=Decimal("0.1"),
            price=Decimal("50000"),
            fee=Decimal("0.5"),
            fee_currency="USD",
        )


def test_enum_json_roundtrip():
    payload = {
        "asset_class": AssetClass.CRYPTO_PERP.value,
        "venue": Venue.KRAKEN_FUTURES.value,
        "direction": Direction.SHORT.value,
    }
    assert json.loads(json.dumps(payload)) == payload
    assert AssetClass(payload["asset_class"]) is AssetClass.CRYPTO_PERP
    assert Venue(payload["venue"]) is Venue.KRAKEN_FUTURES
    assert Direction(payload["direction"]) is Direction.SHORT
