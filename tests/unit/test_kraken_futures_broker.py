"""KrakenFuturesBroker mirrors KrakenSpotBroker behavior; spot-check the parts
that differ (symbol mapping for dated contracts) plus Protocol conformance.
The bulk of error-translation behavior is covered by the spot tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import ccxt

from ggTrader.core import AssetClass, Instrument, Venue
from ggTrader.execution import Broker, KrakenFuturesBroker


def _dated_future() -> Instrument:
    return Instrument(
        symbol="BTC-USD-260626",
        asset_class=AssetClass.CRYPTO_DATED_FUTURE,
        venue=Venue.KRAKEN_FUTURES,
        base_currency="BTC",
        quote_currency="USD",
        tick_size=Decimal("0.5"),
        min_order_size=Decimal("0.0001"),
        maker_fee_bps=Decimal("2"),
        taker_fee_bps=Decimal("5"),
        calendar_id="crypto_24_7",
        expiry="260626",
        venue_specific_id="BTC/USD:USD-260626",
    )


def test_kraken_futures_broker_satisfies_broker_protocol():
    broker = KrakenFuturesBroker(exchange=MagicMock(spec=ccxt.Exchange))
    assert isinstance(broker, Broker)
    assert broker.venue is Venue.KRAKEN_FUTURES


def test_dated_future_symbol_uses_venue_specific_id():
    mock = MagicMock(spec=ccxt.Exchange)
    mock.fetch_ticker.return_value = {"last": "50000", "timestamp": 0}
    broker = KrakenFuturesBroker(exchange=mock)
    broker.fetch_ticker(_dated_future())
    mock.fetch_ticker.assert_called_once_with("BTC/USD:USD-260626")


def test_dated_future_symbol_derives_from_expiry_when_no_venue_id():
    mock = MagicMock(spec=ccxt.Exchange)
    mock.fetch_ticker.return_value = {"last": "50000", "timestamp": 0}
    inst = Instrument(
        symbol="BTC-USD-260626",
        asset_class=AssetClass.CRYPTO_DATED_FUTURE,
        venue=Venue.KRAKEN_FUTURES,
        base_currency="BTC",
        quote_currency="USD",
        tick_size=Decimal("0.5"),
        min_order_size=Decimal("0.0001"),
        maker_fee_bps=Decimal("2"),
        taker_fee_bps=Decimal("5"),
        calendar_id="crypto_24_7",
        expiry="260626",
    )
    KrakenFuturesBroker(exchange=mock).fetch_ticker(inst)
    mock.fetch_ticker.assert_called_once_with("BTC/USD:USD-260626")
