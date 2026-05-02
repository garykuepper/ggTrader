"""Script to generate the top traded S&P 500 stocks using yfinance data."""

import argparse
import json
import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from ggTrader.data.core.stock_constants import SP500_SYMBOLS


def generate_stock_universe(
    limit: int = 50, output_path: str = "data/top_50_stocks_volume.json", window: str = "30d"
):
    """Fetch historical volume for S&P 500 constituents and select top assets."""
    print(f"Fetching volume data for S&P 500 constituents (window: {window})...")

    # Determine window length
    days = 30
    if window == "7d":
        days = 7
    elif window == "24h":
        days = 1

    start_date = (datetime.now() - timedelta(days=days + 5)).strftime("%Y-%m-%d")
    
    # Fetch data in batches of 100 to avoid yfinance rate limits/errors
    batch_size = 100
    all_vols = {}

    for i in range(0, len(SP500_SYMBOLS), batch_size):
        batch = SP500_SYMBOLS[i : i + batch_size]
        print(f"  Processing batch {i // batch_size + 1}/{(len(SP500_SYMBOLS) + batch_size - 1) // batch_size}...")
        
        try:
            # group_by="ticker" ensures MultiIndex structure
            df = yf.download(
                tickers=batch,
                start=start_date,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                progress=False,
            )
            
            if df.empty:
                continue

            for symbol in batch:
                if symbol not in df.columns.levels[0]:
                    continue
                
                sym_df = df[symbol].tail(days)
                if sym_df.empty:
                    continue
                
                # Dollar volume = Close * Volume
                # Compute average daily dollar volume
                dollar_vol = (sym_df["Close"] * sym_df["Volume"]).mean()
                if not pd.isna(dollar_vol):
                    all_vols[symbol] = dollar_vol

        except Exception as e:
            print(f"  Error processing batch: {e!r}")

    # Rank and select
    sorted_stocks = sorted(all_vols.items(), key=lambda x: x[1], reverse=True)
    top_stocks = sorted_stocks[:limit]

    results = []
    print(f"\nTop {len(top_stocks)} S&P 500 stocks by {window} avg daily dollar volume:")
    for i, (symbol, avg_vol) in enumerate(top_stocks, 1):
        print(f"  {i:02d}. {symbol:10s} (${avg_vol:,.0f})")
        results.append(
            {
                "rank": i,
                "symbol": symbol,
                "frequency": 1,
                "average_notional_volume": float(avg_vol),
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSaved {len(results)} symbols to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate live Top N stock universe via yfinance.")
    parser.add_argument("--limit", type=int, default=50, help="Number of assets to select.")
    parser.add_argument(
        "--out", type=str, default="data/top_50_stocks_volume.json", help="Output JSON path."
    )
    parser.add_argument(
        "--window", type=str, default="30d", choices=["24h", "7d", "30d"], help="Volume window."
    )

    args = parser.parse_args()
    generate_stock_universe(limit=args.limit, output_path=args.out, window=args.window)


if __name__ == "__main__":
    main()
