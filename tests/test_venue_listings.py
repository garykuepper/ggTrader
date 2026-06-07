import importlib.util
import json
from pathlib import Path as _Path

import pytest

from ggTrader.data.core import venue_listings
from ggTrader.data.core.venue_listings import (
    fetch_venue_listings,
    filter_to_listed,
    load_venue_listing_symbols,
    write_venue_listings,
)

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
    "BTC3L/USD": {
        "base": "BTC3L.x",
        "quote": "USD",
        "active": True,
        "spot": True,
    },  # dot in base (synthetic/leveraged) -> drop
}


class _FakeExchange:
    def __init__(self, *args, **kwargs):
        # ccxt exchange classes accept a config dict; ignore it in the fake.
        pass

    def load_markets(self):
        return FAKE_MARKETS


@pytest.fixture
def fake_kraken(monkeypatch):
    monkeypatch.setitem(venue_listings.SUPPORTED_VENUES, "kraken", _FakeExchange)


def test_fetch_keeps_only_active_usd_spot_nonstable(fake_kraken):
    listings = fetch_venue_listings("kraken")
    symbols = [e["symbol"] for e in listings]
    # BTC (deduped from BTC/USD + XXBT/USD) and ETH only; sorted. SOL (USDT quote),
    # OLD (inactive), BTC/USD:USD (perp), USDT (stable), BTC3L (dot base) all dropped.
    assert symbols == ["BTC", "ETH"]


def test_fetch_venue_is_case_insensitive(fake_kraken):
    assert [e["symbol"] for e in fetch_venue_listings("KRAKEN")] == ["BTC", "ETH"]


def test_fetch_normalizes_and_keeps_first_ccxt_symbol(fake_kraken):
    listings = fetch_venue_listings("kraken")
    btc = next(e for e in listings if e["symbol"] == "BTC")
    assert btc["ccxt_symbol"] == "BTC/USD"
    assert btc["base"] == "BTC"
    assert btc["quote"] == "USD"


def test_fetch_unsupported_venue_raises():
    with pytest.raises(ValueError):
        fetch_venue_listings("coinbase")


def test_filter_to_listed_drops_unlisted():
    candidates = [{"symbol": "BTC"}, {"symbol": "FOO"}, {"symbol": "ETH"}]
    kept = filter_to_listed(candidates, {"BTC", "ETH"})
    assert [c["symbol"] for c in kept] == ["BTC", "ETH"]


def test_load_symbols_returns_set(tmp_path):
    (tmp_path / "kraken_listings.json").write_text(
        json.dumps({"listings": [{"symbol": "BTC"}, {"symbol": "ETH"}]})
    )
    symbols = load_venue_listing_symbols("kraken", listings_dir=str(tmp_path))
    assert symbols == {"BTC", "ETH"}


def test_load_symbols_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_venue_listing_symbols("kraken", listings_dir=str(tmp_path))


def test_write_creates_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(
        venue_listings,
        "fetch_venue_listings",
        lambda v: [{"symbol": "BTC", "ccxt_symbol": "BTC/USD", "base": "BTC", "quote": "USD"}],
    )
    out = write_venue_listings("binanceus", listings_dir=str(tmp_path))
    payload = json.loads(out.read_text())
    assert payload["venue"] == "binanceus"
    assert payload["count"] == 1
    assert payload["listings"][0]["symbol"] == "BTC"
    assert "updated_at" in payload


def test_write_empty_preserves_existing(monkeypatch, tmp_path):
    existing = tmp_path / "kraken_listings.json"
    existing.write_text('{"sentinel": true}')
    monkeypatch.setattr(venue_listings, "fetch_venue_listings", lambda v: [])
    with pytest.raises(RuntimeError):
        write_venue_listings("kraken", listings_dir=str(tmp_path))
    # existing good snapshot must be untouched
    assert existing.read_text() == '{"sentinel": true}'


def _load_ranker():
    path = _Path(__file__).resolve().parent.parent / "scripts" / "update_universe_ccxt.py"
    spec = importlib.util.spec_from_file_location("update_universe_ccxt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeRankExchange:
    id = "kraken"

    def load_markets(self):
        return {}

    def fetch_tickers(self, symbols=None):
        return {
            "BTC/USD": {"quoteVolume": 1000, "last": 1, "baseVolume": 1000},
            "FOO/USD": {"quoteVolume": 900, "last": 1, "baseVolume": 900},
            "ETH/USD": {"quoteVolume": 800, "last": 1, "baseVolume": 800},
        }


def test_ranker_drops_unlisted_before_topn(monkeypatch, tmp_path):
    ranker = _load_ranker()
    monkeypatch.setattr(ranker.ccxt, "kraken", lambda: _FakeRankExchange())

    # snapshot lists BTC and ETH but NOT FOO
    (tmp_path / "kraken_listings.json").write_text(
        json.dumps({"listings": [{"symbol": "BTC"}, {"symbol": "ETH"}]})
    )

    out_path = tmp_path / "out.json"
    ranker.generate_ccxt_universe(
        limit=10,
        output_path=str(out_path),
        window="24h",
        venue="kraken",
        min_volume=0.0,
        listings_dir=str(tmp_path),
    )

    results = json.loads(out_path.read_text())
    symbols = {r["symbol"] for r in results}
    assert "FOO" not in symbols
    assert {"BTC", "ETH"} <= symbols
