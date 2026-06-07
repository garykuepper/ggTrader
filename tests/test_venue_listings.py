import pytest
from ggTrader.data.core.venue_listings import fetch_venue_listings

from ggTrader.data.core import venue_listings

# base/quote/active/spot mirror the ccxt market structure
FAKE_MARKETS = {
    "BTC/USD": {"base": "BTC", "quote": "USD", "active": True, "spot": True},
    "XXBT/USD": {
        "base": "XXBT",
        "quote": "USD",
        "active": True,
        "spot": True,
    },  # maps to BTC -> dedup
    "ETH/USD": {"base": "ETH", "quote": "USD", "active": True, "spot": True},
    "SOL/USDT": {
        "base": "SOL",
        "quote": "USDT",
        "active": True,
        "spot": True,
    },  # non-USD quote -> drop
    "OLD/USD": {"base": "OLD", "quote": "USD", "active": False, "spot": True},  # inactive -> drop
    "BTC/USD:USD": {"base": "BTC", "quote": "USD", "active": True, "spot": False},  # perp -> drop
    "USDT/USD": {
        "base": "USDT",
        "quote": "USD",
        "active": True,
        "spot": True,
    },  # stable base -> drop
}


class _FakeExchange:
    def load_markets(self):
        return FAKE_MARKETS


@pytest.fixture
def fake_kraken(monkeypatch):
    monkeypatch.setitem(venue_listings.SUPPORTED_VENUES, "kraken", _FakeExchange)


def test_fetch_keeps_only_active_usd_spot_nonstable(fake_kraken):
    listings = fetch_venue_listings("kraken")
    symbols = [e["symbol"] for e in listings]
    # BTC (deduped from BTC/USD + XXBT/USD) and ETH only; sorted
    assert symbols == ["BTC", "ETH"]


def test_fetch_normalizes_and_keeps_first_ccxt_symbol(fake_kraken):
    listings = fetch_venue_listings("kraken")
    btc = next(e for e in listings if e["symbol"] == "BTC")
    assert btc["ccxt_symbol"] == "BTC/USD"
    assert btc["base"] == "BTC"
    assert btc["quote"] == "USD"


def test_fetch_unsupported_venue_raises():
    with pytest.raises(ValueError):
        fetch_venue_listings("coinbase")
