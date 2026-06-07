"""Script to generate the top traded USD pairs on Kraken using live CCXT data."""

import argparse
import json
import os

import ccxt

from ggTrader.data.core.constants import STABLE_BASES, SYMBOL_MAPPING
from ggTrader.data.core.venue_listings import (
    DEFAULT_LISTINGS_DIR,
    filter_to_listed,
    load_venue_listing_symbols,
)

SYMBOL_TO_NAME = {
    "AAVE": "Aave",
    "ADA": "Cardano",
    "AKT": "Akash Network",
    "ALGO": "Algorand",
    "ARB": "Arbitrum",
    "ATOM": "Cosmos",
    "AVAX": "Avalanche",
    "BCH": "Bitcoin Cash",
    "BONK": "Bonk",
    "BTC": "Bitcoin",
    "CFG": "Centrifuge",
    "CRV": "Curve DAO Token",
    "DOGE": "Dogecoin",
    "DOT": "Polkadot",
    "ENA": "Ethena",
    "ETH": "Ethereum",
    "EWT": "Energy Web Token",
    "FET": "Fetch.ai",
    "FIL": "Filecoin",
    "FLR": "Flare",
    "FTM": "Fantom",
    "GALA": "Gala",
    "GRT": "The Graph",
    "ICP": "Internet Computer",
    "IMX": "Immutable",
    "INJ": "Injective",
    "KAS": "Kaspa",
    "KSM": "Kusama",
    "LDO": "Lido DAO",
    "LINK": "Chainlink",
    "LTC": "Litecoin",
    "LUNA": "Terra",
    "MATIC": "Polygon",
    "NEAR": "Near Protocol",
    "OCEAN": "Ocean Protocol",
    "ONDO": "Ondo",
    "PEPE": "Pepe",
    "QNT": "Quant",
    "RENDER": "Render Token",
    "SAND": "The Sandbox",
    "SCRT": "Secret",
    "SEI": "Sei",
    "SGB": "Songbird",
    "SHIB": "Shiba Inu",
    "SOL": "Solana",
    "SPX": "SPX6900",
    "STX": "Stacks",
    "SUI": "Sui",
    "TAO": "Bittensor",
    "TIA": "Celestia",
    "TRX": "TRON",
    "UNI": "Uniswap",
    "WIF": "dogwifhat",
    "XCN": "Onyxcoin",
    "XLM": "Stellar",
    "XMR": "Monero",
    "XRP": "XRP",
    "ZEC": "Zcash",
}


def generate_ccxt_universe(
    limit: int = 50,
    output_path: str = "data/top_50_ccxt_volume.json",
    window: str = "24h",
    venue: str | None = None,
    min_volume: float = 0.0,
    listings_dir: str = DEFAULT_LISTINGS_DIR,
):
    """Fetch tickers and select top volume USD pairs, optionally using historical windows.

    Venue is picked from (in order): the ``venue`` arg, the ``EXCHANGE`` env var,
    then "kraken" as a fallback. The selected venue is the exchange we'll trade
    on, so top-N is filtered to its actual listing set — no more selecting coins
    that turn out to be unlisted at deploy time.

    ``min_volume`` is a hard floor (in USD, on the ranking window's volume) applied
    *before* the ``limit`` cap. On thin venues like Binance.US only ~11 USD pairs
    clear a $50K 30-day-avg floor, so the floor — not the count — defines the
    tradeable universe and keeps dead <$2K/day markets out. ``limit`` remains a
    safety cap. ``min_volume=0`` preserves the legacy fixed top-N behavior.
    """
    venue = (venue or os.getenv("EXCHANGE") or "kraken").lower()
    exchange_cls = {"kraken": ccxt.kraken, "binanceus": ccxt.binanceus}.get(venue)
    if exchange_cls is None:
        raise ValueError(f"Unsupported venue: {venue!r} (expected 'kraken' or 'binanceus')")
    exchange = exchange_cls()
    print(f"Fetching live tickers from {exchange.id} (window: {window})...")

    # Load markets to get asset names
    exchange.load_markets()
    tickers = exchange.fetch_tickers()

    candidates = []

    for symbol, ticker_info in tickers.items():
        # Only look at USD quoting pairs
        if not symbol.endswith("/USD") or "/" not in symbol:
            continue

        base_asset = symbol.split("/")[0]
        if base_asset in STABLE_BASES:
            continue

        # Skip index or future weird tickers if they appear
        if "." in base_asset or ":" in base_asset:
            continue

        # Standardize symbol using SYMBOL_MAPPING
        standard_base = SYMBOL_MAPPING.get(base_asset, base_asset)

        # Using quoteVolume (volume in USD) for initial sort
        quote_vol_24h = ticker_info.get("quoteVolume", 0)
        if quote_vol_24h is None or quote_vol_24h == 0:
            base_vol = ticker_info.get("baseVolume", 0)
            last_price = ticker_info.get("last", 0)
            quote_vol_24h = (base_vol * last_price) if (base_vol and last_price) else 0

        full_name = SYMBOL_TO_NAME.get(standard_base, standard_base)

        candidates.append(
            {
                "symbol": standard_base,
                "kraken_symbol": standard_base,
                "volume_24h": quote_vol_24h,
                "ccxt_symbol": symbol,
                "name": full_name,
            }
        )

    # Layer-1 availability intersection: keep only coins present in the venue's
    # committed listings snapshot, BEFORE the volume floor / top-N cut. Fails loud
    # if the snapshot is missing (no silent fall-through to an unfiltered universe).
    listed_symbols = load_venue_listing_symbols(venue, listings_dir=listings_dir)
    before = len(candidates)
    candidates = filter_to_listed(candidates, listed_symbols)
    print(
        f"Availability filter ({venue}): {len(candidates)}/{before} USD candidates "
        f"are in the listings snapshot."
    )

    # Sort by 24h volume for the initial filter
    candidates.sort(key=lambda x: x["volume_24h"], reverse=True)

    # If window is > 24h, fetch historical volume for the top 100 candidates
    if window in ["7d", "30d"]:
        days = 7 if window == "7d" else 30
        print(f"Refining ranking using {window} volume for top 100 candidates...")

        refined_candidates = candidates[:100]
        final_rankable = []

        for i, c in enumerate(refined_candidates):
            try:
                # Fetch daily OHLCV for the last N days
                ohlcv = exchange.fetch_ohlcv(c["ccxt_symbol"], timeframe="1d", limit=days)
                if not ohlcv:
                    final_rankable.append({**c, "volume_window": c["volume_24h"]})
                    continue

                # Mean daily (volume * close) over the period — a true average-daily-USD
                # figure so --min-volume is an intuitive per-day floor regardless of window
                # (and so the stored value matches its `average_notional_volume` label).
                # ohlcv: [timestamp, open, high, low, close, volume]
                daily = [candle[5] * candle[4] for candle in ohlcv]
                avg_vol = (sum(daily) / len(daily)) if daily else c["volume_24h"]
                final_rankable.append({**c, "volume_window": avg_vol})
                print(
                    f"  [{i + 1}/100] {c['symbol']:8s} | {window} avg/day: ${avg_vol:,.0f}",
                    end="\r",
                )
            except Exception:
                # Fallback to 24h if history fails
                final_rankable.append({**c, "volume_window": c["volume_24h"]})

        candidates = final_rankable
        candidates.sort(key=lambda x: x["volume_window"], reverse=True)
        vol_key = "volume_window"
    else:
        vol_key = "volume_24h"

    # Apply the volume floor (on avg-daily USD volume for 7d/30d, else 24h) before the
    # count cap. The floor defines the tradeable universe on thin venues; limit is a cap.
    if min_volume > 0:
        passed = [c for c in candidates if c[vol_key] >= min_volume]
        print(
            f"\nVolume floor ${min_volume:,.0f}/day ({window}): "
            f"{len(passed)}/{len(candidates)} USD pairs passed on {exchange.id}."
        )
        candidates = passed

    top_candidates = candidates[:limit]

    results = []
    print(f"\nTop {len(top_candidates)} pairs by {window} USD volume on {exchange.id}:")
    for i, c in enumerate(top_candidates, 1):
        display_vol = c[vol_key]
        print(f"  {i:02d}. {c['kraken_symbol']:10s} (${display_vol:,.0f})")
        results.append(
            {
                "rank": i,
                "symbol": c["symbol"],
                "kraken_symbol": c["kraken_symbol"],
                "frequency": 1,
                "average_notional_volume": display_vol,
                "name": c["name"],
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved {len(results)} symbols in detailed format to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate live Top N asset universe via CCXT.")
    parser.add_argument("--limit", type=int, default=50, help="Number of assets to select.")
    parser.add_argument(
        "--out", type=str, default="data/top_50_ccxt_volume.json", help="Output JSON path."
    )
    parser.add_argument(
        "--window", type=str, default="30d", choices=["24h", "7d", "30d"], help="Volume window."
    )
    parser.add_argument(
        "--venue",
        type=str,
        default=None,
        choices=["kraken", "binanceus"],
        help="Exchange to query (default: $EXCHANGE env var, else 'kraken').",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=0.0,
        help="Minimum window USD volume floor; coins below it are dropped before the "
        "--limit cap. 0 (default) keeps legacy fixed top-N behavior.",
    )

    args = parser.parse_args()
    generate_ccxt_universe(
        limit=args.limit,
        output_path=args.out,
        window=args.window,
        venue=args.venue,
        min_volume=args.min_volume,
    )


if __name__ == "__main__":
    main()
