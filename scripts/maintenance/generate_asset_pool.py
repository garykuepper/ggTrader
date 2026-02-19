"""Generates a JSON file containing consistent movers based on historical volume frequency."""

import argparse
import json
import os
import sys

import pandas as pd

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from ggTrader.data.kraken.constants import SYMBOL_MAPPING
from ggTrader.data.kraken.historical_data import KrakenHistoricalData


def main():
    parser = argparse.ArgumentParser(description="Generate Consistent Movers Asset Pool")
    parser.add_argument("--n", type=int, default=25, help="Number of assets in the final pool")
    parser.add_argument(
        "--daily-n", type=int, default=200, help="Number of daily top movers to consider"
    )
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date")
    parser.add_argument("--end-date", type=str, default="2025-12-31", help="End date")
    parser.add_argument("--threshold", type=int, default=500, help="Minimum daily trades")
    parser.add_argument("--quote", type=str, default="USD", help="Quote currency (USD, EUR, etc.)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: data/top_N_USD_2023-01-01_2025-12-31.json)",
    )
    parser.add_argument("--stables", action="store_true", help="Include stablecoins and fiats")

    args = parser.parse_args()

    # Default output path if not provided
    if args.output is None:
        args.output = f"data/top_{args.n}_{args.quote}_{args.start_date}_{args.end_date}.json"

    print(f"--- Generating Consistent Movers Pool ---")
    print(f"Start Date: {args.start_date}")
    print(f"End Date: {args.end_date}")
    print(f"Quote: {args.quote}")
    print(f"Daily Top Filter: {args.daily_n}")
    print(f"Target Pool Size: {args.n}")
    print(f"Trades Threshold: {args.threshold}")

    kh = KrakenHistoricalData()
    df = kh.reader.get_consistent_movers(
        start_date=args.start_date,
        end_date=args.end_date,
        daily_top_n=args.daily_n,
        output_n=args.n,
        trades_threshold=args.threshold,
        stables=args.stables,
        quote=args.quote,
    )

    if df.empty:
        print(
            "Error: No consistent movers found. Check your database connection and trades threshold."
        )
        sys.exit(1)

    # Sort by volume descending so the top volumes get rank 1, 2, 3...
    df = df.sort_values(by=["average_notional_volume", "frequency"], ascending=[False, False])
    # Apply mapping and add rank
    results = []
    for i, row in enumerate(df.to_dict(orient="records")):
        symbol = row["symbol"]
        # Standardize symbol name
        mapped_symbol = SYMBOL_MAPPING.get(symbol, symbol)

        results.append(
            {
                "rank": i + 1,
                "symbol": mapped_symbol,
                "kraken_symbol": symbol,
                "frequency": row["frequency"],
                "average_notional_volume": row["average_notional_volume"],
            }
        )

    # Ensure directory exists
    output_path = os.path.abspath(os.path.join(kh.root_dir, args.output))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Successfully saved {len(results)} movers to {output_path}")
    print(pd.DataFrame(results).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
