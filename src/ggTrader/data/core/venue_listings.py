"""Per-venue tradeable-coin availability snapshots (Layer 1 of the universe pipeline).

OHLCV price/volume is effectively venue-agnostic, but *which* coins are listed
differs by venue. This module captures a current snapshot of the active USD spot
pairs on a venue so the mover ranker can intersect its candidates against a
tracked, version-controlled availability set. See
docs/superpowers/specs/2026-06-06-per-venue-availability-registry-design.md.
"""

from __future__ import annotations

import ccxt

from ggTrader.data.core.constants import STABLE_BASES, SYMBOL_MAPPING

# Exchanges we support generating availability snapshots for.
SUPPORTED_VENUES = {"kraken": ccxt.kraken, "binanceus": ccxt.binanceus}

DEFAULT_LISTINGS_DIR = "data/universe"


def fetch_venue_listings(venue: str) -> list[dict]:
    """Fetch the active USD spot listings for a venue.

    Args:
        venue: Exchange name; one of ``SUPPORTED_VENUES`` (case-insensitive).

    Returns:
        Sorted, de-duplicated list of ``{symbol, ccxt_symbol, base, quote}`` dicts,
        where ``symbol`` is the normalized base (e.g. ``"BTC"``). One entry per
        normalized base; the first ccxt market id encountered wins.

    Raises:
        ValueError: If ``venue`` is not supported.
    """
    venue = venue.lower()
    exchange_cls = SUPPORTED_VENUES.get(venue)
    if exchange_cls is None:
        raise ValueError(
            f"Unsupported venue: {venue!r} (expected one of {sorted(SUPPORTED_VENUES)})"
        )
    exchange = exchange_cls()
    markets = exchange.load_markets()

    listings: list[dict] = []
    for ccxt_symbol, market in markets.items():
        if not (market.get("active") and market.get("spot")):
            continue
        if market.get("quote") != "USD":
            continue
        base = market.get("base") or ""
        if "." in base or ":" in base:
            continue
        std_base = SYMBOL_MAPPING.get(base, base)
        if std_base in STABLE_BASES:
            continue
        listings.append(
            {
                "symbol": std_base,
                "ccxt_symbol": ccxt_symbol,
                "base": std_base,
                "quote": "USD",
            }
        )

    listings.sort(key=lambda e: e["symbol"])

    seen: set[str] = set()
    deduped: list[dict] = []
    for entry in listings:
        if entry["symbol"] in seen:
            continue
        seen.add(entry["symbol"])
        deduped.append(entry)
    return deduped
