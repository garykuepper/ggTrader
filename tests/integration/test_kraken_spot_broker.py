"""Integration tests: KrakenSpotBroker against real Kraken public endpoints.

No auth, no orders — strictly read-only ticker fetches that verify the adapter
round-trips real venue payloads into domain types. Skipped if Kraken is
unreachable so offline CI doesn't block.
"""

from __future__ import annotations

from decimal import Decimal

import ccxt
import pytest

from ggTrader.core import AssetClass, Instrument, Venue
from ggTrader.execution import KrakenSpotBroker

pytestmark = pytest.mark.integration


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


def test_fetch_ticker_btc_usd_returns_sane_values():
    broker = KrakenSpotBroker(exchange=ccxt.kraken({"enableRateLimit": True}))
    try:
        ticker = broker.fetch_ticker(_btc())
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Kraken unreachable: {exc}")

    assert ticker.last > Decimal("0")
    if ticker.bid is not None and ticker.ask is not None:
        assert ticker.bid <= ticker.ask
