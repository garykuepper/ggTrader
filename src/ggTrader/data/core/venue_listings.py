"""Per-venue tradeable-coin availability snapshots (Layer 1 of the universe pipeline).

OHLCV price/volume is effectively venue-agnostic, but *which* coins are listed
differs by venue. This module captures a current snapshot of the active USD spot
pairs on a venue so the mover ranker can intersect its candidates against a
tracked, version-controlled availability set. See
docs/superpowers/specs/2026-06-06-per-venue-availability-registry-design.md.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import ccxt

from ggTrader.data.core.constants import STABLE_BASES, SYMBOL_MAPPING

# Exchanges we support generating availability snapshots for.
SUPPORTED_VENUES = {"kraken": ccxt.kraken, "binanceus": ccxt.binanceus}

# Output directory for snapshots, relative to the process working directory
# (callers run from the repo root, matching the rest of the data/ tooling).
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
    # enableRateLimit matches every other ccxt caller in the project; load_markets
    # can fan out to multiple HTTP requests on large venues.
    exchange = exchange_cls({"enableRateLimit": True})
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
        # SYMBOL_MAPPING is Kraken-specific (XXBT->BTC etc); a harmless no-op for
        # other venues, which already use standard bases.
        std_base = SYMBOL_MAPPING.get(base, base)
        if std_base in STABLE_BASES:
            continue
        listings.append(
            {
                "symbol": std_base,
                "ccxt_symbol": ccxt_symbol,
                "base": base,  # raw pre-normalization base, for audit/debug
                "quote": "USD",
            }
        )

    listings.sort(key=lambda e: e["symbol"])

    seen: set[str] = set()
    deduped: list[dict] = []
    for entry in listings:
        # Stable sort preserves dict iteration order, so the first market id seen
        # for a normalized symbol wins.
        if entry["symbol"] in seen:
            continue
        seen.add(entry["symbol"])
        deduped.append(entry)
    return deduped


def filter_to_listed(
    candidates: list[dict], listed_symbols: set[str], key: str = "symbol"
) -> list[dict]:
    """Keep only candidates whose ``key`` value is in ``listed_symbols``.

    Pure function; preserves input order.
    """
    return [c for c in candidates if c[key] in listed_symbols]


def load_venue_listing_symbols(venue: str, listings_dir: str = DEFAULT_LISTINGS_DIR) -> set[str]:
    """Load the set of available ``symbol`` values from a venue's snapshot.

    Raises:
        FileNotFoundError: If the snapshot does not exist (with a hint to run the
            ``update_venue_listings`` command). The ranker must fail loud rather
            than silently skip the availability filter.
    """
    venue = venue.lower()
    path = Path(listings_dir) / f"{venue}_listings.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No availability snapshot at {path}. Run: "
            f"python scripts/update_venue_listings.py --venue {venue}"
        )
    with open(path) as f:
        payload = json.load(f)
    return {entry["symbol"] for entry in payload.get("listings", [])}


def write_venue_listings(venue: str, listings_dir: str = DEFAULT_LISTINGS_DIR) -> Path:
    """Fetch and atomically write a venue's availability snapshot.

    Writes to a temp file then ``os.replace`` so a partial/failed write never
    clobbers an existing good snapshot. Refuses to write an empty listing set
    (treated as a fetch failure) to avoid wiping a valid snapshot.

    Returns:
        Path to the written ``{venue}_listings.json``.

    Raises:
        RuntimeError: If the fetched listing set is empty.
    """
    venue = venue.lower()
    listings = fetch_venue_listings(venue)
    if not listings:
        raise RuntimeError(
            f"Refusing to write empty listings for {venue!r}; "
            "aborting to preserve any existing snapshot."
        )

    out_dir = Path(listings_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{venue}_listings.json"

    payload = {
        "venue": venue,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "count": len(listings),
        "listings": listings,
    }

    tmp_path = out_path.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return out_path
