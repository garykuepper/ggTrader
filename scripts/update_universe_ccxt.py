"""Script to generate the top traded USD pairs on Kraken using live CCXT data."""

import argparse
import json
import os
import sys
from pathlib import Path

import ccxt

# Add src to path just in case we need project utilities later
sys.path.append(os.path.join(os.getcwd(), "src"))

def generate_ccxt_universe(limit: int = 50, output_path: str = "data/top_50_ccxt_volume.json"):
    """Fetch live tickers and select top volume USD pairs."""
    exchange = ccxt.kraken()
    print("Fetching tickers from Kraken...")
    tickers = exchange.fetch_tickers()
    
    # Exclude stablecoins and fiat
    blacklist = {
        "USDT", "USDC", "DAI", "FDUSD", "TUSD", "BUSD", "PYUSD", "USDD", 
        "EUR", "GBP", "JPY", "CAD", "CHF", "AUD"
    }
    
    candidates = []
    
    for symbol, ticker_info in tickers.items():
        # Only look at USD quoting pairs
        if not symbol.endswith("/USD"):
            continue
            
        base_asset = symbol.split("/")[0]
        if base_asset in blacklist:
            continue
            
        # Skip index or future weird tickers if they appear
        if "." in base_asset or ":" in base_asset:
            continue
            
        # Using quoteVolume (volume in USD)
        quote_vol = ticker_info.get("quoteVolume", 0)
        if quote_vol is None or quote_vol == 0:
            # Fallback to calculating from baseVolume and last price
            base_vol = ticker_info.get("baseVolume", 0)
            last_price = ticker_info.get("last", 0)
            if base_vol and last_price:
                quote_vol = base_vol * last_price
            else:
                quote_vol = 0
            
        candidates.append({
            "ccxt_symbol": symbol,
            "project_symbol": symbol.replace("/", "-"),
            "volume_24h": quote_vol
        })
        
    # Sort by 24h USD volume descending
    candidates.sort(key=lambda x: x["volume_24h"], reverse=True)
    top_candidates = candidates[:limit]
    
    symbols_list = [c["project_symbol"] for c in top_candidates]
    print(f"\nTop {limit} pairs by 24h USD volume on {exchange.id}:")
    for i, c in enumerate(top_candidates, 1):
        print(f"  {i:02d}. {c['project_symbol']:10s} (${c['volume_24h']:,.0f})")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(symbols_list, f, indent=4)
        
    print(f"\nSaved {len(symbols_list)} symbols to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate live Top N asset universe via CCXT.")
    parser.add_argument("--limit", type=int, default=50, help="Number of assets to select.")
    parser.add_argument("--out", type=str, default="data/top_50_ccxt_volume.json", help="Output JSON path.")
    
    args = parser.parse_args()
    generate_ccxt_universe(limit=args.limit, output_path=args.out)

if __name__ == "__main__":
    main()
