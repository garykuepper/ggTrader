"""Phase 1 Broker tests: Protocol structural conformance + KrakenSpotBroker shim."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import ccxt
import pytest

from ggTrader.core import (
    AssetClass,
    Balance,
    Direction,
    Fill,
    Instrument,
    Order,
    Ticker,
    Venue,
)
from ggTrader.execution import Broker, KrakenSpotBroker
from ggTrader.execution.errors import (
    InsufficientFunds,
    OrderNotFound,
    OrderRejected,
    VenueUnavailable,
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


def _make_order(order_type: str = "market", limit_price: Decimal | None = None) -> Order:
    return Order(
        ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        instrument=_btc(),
        side=Direction.LONG,
        quantity=Decimal("0.1"),
        order_type=order_type,
        limit_price=limit_price,
        client_order_id="test-1",
    )


def _broker_with_mock() -> tuple[KrakenSpotBroker, MagicMock]:
    mock = MagicMock(spec=ccxt.Exchange)
    return KrakenSpotBroker(exchange=mock), mock


# ---- Protocol conformance ---------------------------------------------------


def test_kraken_spot_broker_satisfies_broker_protocol():
    broker, _ = _broker_with_mock()
    assert isinstance(broker, Broker)


def test_broker_protocol_is_runtime_checkable():
    class FakeBroker:
        venue = Venue.KRAKEN_SPOT

        def submit_order(self, order: Order) -> str:
            return "x"

        def cancel_order(self, order_id: str, instrument: Instrument) -> bool:
            return True

        def fetch_order(self, order_id, instrument):
            return None

        def fetch_open_orders(self, instrument=None):
            return []

        def fetch_fills(self, instrument=None, since=None, limit=100):
            return []

        def fetch_balance(self):
            return {}

        def fetch_ticker(self, instrument):
            return Ticker(
                ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
                instrument=instrument,
                last=Decimal("1"),
            )

    assert isinstance(FakeBroker(), Broker)


# ---- submit_order -----------------------------------------------------------


def test_submit_market_buy_calls_ccxt_create_order():
    broker, mock = _broker_with_mock()
    mock.create_order.return_value = {"id": "kraken-123"}

    order_id = broker.submit_order(_make_order(order_type="market"))

    assert order_id == "kraken-123"
    mock.create_order.assert_called_once_with("BTC/USD", "market", "buy", 0.1)


def test_submit_limit_sell_passes_price():
    broker, mock = _broker_with_mock()
    mock.create_order.return_value = {"id": "k-456"}

    order = Order(
        ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        instrument=_btc(),
        side=Direction.SHORT,
        quantity=Decimal("0.5"),
        order_type="limit",
        limit_price=Decimal("55000"),
        client_order_id="t-2",
    )
    broker.submit_order(order)

    mock.create_order.assert_called_once_with("BTC/USD", "limit", "sell", 0.5, 55000.0)


def test_submit_limit_without_price_raises():
    broker, _ = _broker_with_mock()
    with pytest.raises(OrderRejected):
        broker.submit_order(_make_order(order_type="limit"))


def test_submit_insufficient_funds_translates_to_domain_error():
    broker, mock = _broker_with_mock()
    mock.create_order.side_effect = ccxt.InsufficientFunds("nope")
    with pytest.raises(InsufficientFunds):
        broker.submit_order(_make_order())


def test_submit_network_error_translates_to_venue_unavailable():
    broker, mock = _broker_with_mock()
    mock.create_order.side_effect = ccxt.NetworkError("flaky")
    with pytest.raises(VenueUnavailable):
        broker.submit_order(_make_order())


def test_submit_exchange_error_translates_to_order_rejected():
    broker, mock = _broker_with_mock()
    mock.create_order.side_effect = ccxt.ExchangeError("bad price")
    with pytest.raises(OrderRejected):
        broker.submit_order(_make_order())


# ---- cancel / fetch_order ---------------------------------------------------


def test_cancel_order_calls_ccxt():
    broker, mock = _broker_with_mock()
    assert broker.cancel_order("oid", _btc()) is True
    mock.cancel_order.assert_called_once_with("oid", "BTC/USD")


def test_cancel_unknown_order_raises_order_not_found():
    broker, mock = _broker_with_mock()
    mock.cancel_order.side_effect = ccxt.OrderNotFound("gone")
    with pytest.raises(OrderNotFound):
        broker.cancel_order("oid", _btc())


def test_fetch_order_returns_none_when_not_found():
    broker, mock = _broker_with_mock()
    mock.fetch_order.side_effect = ccxt.OrderNotFound("gone")
    assert broker.fetch_order("oid", _btc()) is None


def test_fetch_order_parses_to_domain_order():
    broker, mock = _broker_with_mock()
    mock.fetch_order.return_value = {
        "id": "k-99",
        "timestamp": 1735689600000,
        "side": "sell",
        "amount": "0.2",
        "type": "limit",
        "price": "60000",
        "timeInForce": "GTC",
    }
    result = broker.fetch_order("k-99", _btc())
    assert result is not None
    assert result.side is Direction.SHORT
    assert result.quantity == Decimal("0.2")
    assert result.order_type == "limit"
    assert result.limit_price == Decimal("60000")


# ---- balance / ticker / fills ----------------------------------------------


def test_fetch_balance_translates_ccxt_dict():
    broker, mock = _broker_with_mock()
    mock.fetch_balance.return_value = {
        "USD": {"free": "1000.5", "used": "0", "total": "1000.5"},
        "BTC": {"free": "0.1", "used": "0.05", "total": "0.15"},
        "info": {"raw": "kraken-passthrough"},
        "free": {"USD": "1000.5"},
    }
    balances = broker.fetch_balance()
    assert set(balances.keys()) == {"USD", "BTC"}
    assert isinstance(balances["USD"], Balance)
    assert balances["BTC"].total == Decimal("0.15")
    assert balances["BTC"].free + balances["BTC"].used == balances["BTC"].total


def test_fetch_ticker_returns_populated_domain_ticker():
    broker, mock = _broker_with_mock()
    mock.fetch_ticker.return_value = {
        "timestamp": 1735689600000,
        "bid": "50100",
        "ask": "50110",
        "last": "50105",
        "baseVolume": "1234.5",
    }
    ticker = broker.fetch_ticker(_btc())
    assert ticker.bid == Decimal("50100")
    assert ticker.ask == Decimal("50110")
    assert ticker.last == Decimal("50105")
    assert ticker.volume_24h == Decimal("1234.5")


def test_fetch_fills_translates_trades():
    broker, mock = _broker_with_mock()
    mock.fetch_my_trades.return_value = [
        {
            "timestamp": 1735689600000,
            "order": "k-99",
            "side": "buy",
            "amount": "0.1",
            "price": "50000",
            "fee": {"cost": "0.5", "currency": "USD"},
        }
    ]
    fills = broker.fetch_fills(instrument=_btc())
    assert len(fills) == 1
    assert isinstance(fills[0], Fill)
    assert fills[0].side is Direction.LONG
    assert fills[0].fee == Decimal("0.5")
    assert fills[0].fee_currency == "USD"


# ---- symbol mapping ---------------------------------------------------------


def test_venue_specific_id_preferred_over_canonical():
    broker, mock = _broker_with_mock()
    mock.fetch_ticker.return_value = {"last": "1", "timestamp": 0}
    inst = Instrument(
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
        venue_specific_id="XXBTZUSD",
    )
    broker.fetch_ticker(inst)
    mock.fetch_ticker.assert_called_once_with("XXBTZUSD")
