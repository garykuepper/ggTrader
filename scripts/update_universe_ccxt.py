"""Script to generate the top traded USD pairs on Kraken using live CCXT data."""

import argparse
import json
import os

import ccxt

from ggTrader.data.core.constants import STABLE_BASES, SYMBOL_MAPPING

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
    limit: int = 50, output_path: str = "data/top_50_ccxt_volume.json", window: str = "24h"
):
    """Fetch tickers and select top volume USD pairs, optionally using historical windows."""
    exchange = ccxt.kraken()
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

                # Sum of (volume * close) for the period
                # ohlcv: [timestamp, open, high, low, close, volume]
                total_vol = sum(candle[5] * candle[4] for candle in ohlcv)
                final_rankable.append({**c, "volume_window": total_vol})
                print(
                    f"  [{i + 1}/100] {c['symbol']:8s} | {window} Vol: ${total_vol:,.0f}", end="\r"
                )
            except Exception:
                # Fallback to 24h if history fails
                final_rankable.append({**c, "volume_window": c["volume_24h"]})

        candidates = final_rankable
        candidates.sort(key=lambda x: x["volume_window"], reverse=True)
        vol_key = "volume_window"
    else:
        vol_key = "volume_24h"

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

    args = parser.parse_args()
    generate_ccxt_universe(limit=args.limit, output_path=args.out, window=args.window)


if __name__ == "__main__":
    main()
