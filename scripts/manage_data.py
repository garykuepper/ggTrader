"""CLI for OHLCV maintenance: ingest, aggregate, and related data tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

from ggTrader.data.core.constants import SYMBOL_MAPPING
from ggTrader.data.historical.postgres_ingestor import PostgresIngestor
from ggTrader.data.historical.timescaledb_loader import TimescaleDBLoader
from ggTrader.utils.config import get_db_connection_string
from ggTrader.utils.db_engine import create_db_engine


def cmd_backfill(args):
    """Build 4h and 30m candles from 1h and 15m data for missing date ranges."""
    AGGREGATE_4H_SQL = """
    INSERT INTO ohlcv (timestamp, symbol, interval, open, high, low, close, volume, trades)
    SELECT time_bucket('4 hours', timestamp) AS ts, symbol, '4h' AS interval,
        (ARRAY_AGG(open ORDER BY timestamp ASC))[1] AS open, MAX(high) AS high, MIN(low) AS low,
        (ARRAY_AGG(close ORDER BY timestamp DESC))[1] AS close, SUM(volume) AS volume, SUM(trades) AS trades
    FROM ohlcv WHERE interval = '1h' AND timestamp < :cutoff AND symbol = :symbol
    GROUP BY ts, symbol HAVING COUNT(*) = 4
    ON CONFLICT (timestamp, symbol, interval) DO UPDATE SET
        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
        close = EXCLUDED.close, volume = EXCLUDED.volume, trades = EXCLUDED.trades;
    """

    AGGREGATE_30M_SQL = """
    INSERT INTO ohlcv (timestamp, symbol, interval, open, high, low, close, volume, trades)
    SELECT time_bucket('30 minutes', timestamp) AS ts, symbol, '30m' AS interval,
        (ARRAY_AGG(open ORDER BY timestamp ASC))[1] AS open, MAX(high) AS high, MIN(low) AS low,
        (ARRAY_AGG(close ORDER BY timestamp DESC))[1] AS close, SUM(volume) AS volume, SUM(trades) AS trades
    FROM ohlcv WHERE interval = '15m' AND timestamp < :cutoff AND symbol = :symbol
    GROUP BY ts, symbol HAVING COUNT(*) = 2
    ON CONFLICT (timestamp, symbol, interval) DO UPDATE SET
        open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
        close = EXCLUDED.close, volume = EXCLUDED.volume, trades = EXCLUDED.trades;
    """

    CUTOFF = "2024-04-01"
    engine = create_db_engine()
    print("Fetching unique symbols...")
    with engine.connect() as conn:
        symbols = [r[0] for r in conn.execute(text("SELECT DISTINCT symbol FROM ohlcv"))]

    print(f"Found {len(symbols)} symbols to process.")

    with engine.begin() as conn:
        conn.execute(text("SET timescaledb.max_tuples_decompressed_per_dml_transaction = 0"))

        print(f"Building 4h candles from 1h data (before {CUTOFF})...")
        total_4h = sum(
            conn.execute(text(AGGREGATE_4H_SQL), {"cutoff": CUTOFF, "symbol": s}).rowcount
            for s in tqdm(symbols, desc="4h Aggregation")
        )
        print(f"  Inserted/updated {total_4h:,} 4h rows.")

        print(f"Building 30m candles from 15m data (before {CUTOFF})...")
        total_30m = sum(
            conn.execute(text(AGGREGATE_30M_SQL), {"cutoff": CUTOFF, "symbol": s}).rowcount
            for s in tqdm(symbols, desc="30m Aggregation")
        )
        print(f"  Inserted/updated {total_30m:,} 30m rows.")

    engine.dispose()

    engine = create_db_engine()
    with engine.connect() as conn:
        for iv in ["4h", "30m"]:
            row = conn.execute(
                text(
                    "SELECT MIN(timestamp), MAX(timestamp), COUNT(*) FROM ohlcv WHERE interval = :iv AND symbol = 'BTC-USD'"
                ),
                {"iv": iv},
            ).fetchone()
            if row and row[0]:
                print(f"  BTC {iv}: {str(row[0])[:10]} to {str(row[1])[:10]} ({row[2]:,} rows)")
            else:
                print(f"  BTC {iv}: No data found.")
    engine.dispose()
    print("Done!")


def cmd_ingest_kraken(args):
    """Ingest Kraken Data into PostgreSQL."""
    print("Connecting to database...")
    try:
        connection_string = get_db_connection_string()
        ingestor = PostgresIngestor(connection_string)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    except Exception as e:
        print(f"Failed to initialize ingestor: {e}")
        return

    if args.dir:
        print(f"Ingesting specific directory: {args.dir}")
        ingestor.ingest_dir(args.dir)
    else:
        print("Syncing new data directories...")
        ingestor.sync_local_data(root_dir=project_root, force=args.force)


def cmd_generate_pool(args):
    """Generate Consistent Movers Asset Pool."""
    output = args.output or f"data/top_{args.n}_{args.quote}_{args.start_date}_{args.end_date}.json"

    print(f"--- Generating Consistent Movers Pool ---")
    print(f"Start Date: {args.start_date} | End Date: {args.end_date}")
    print(f"Quote: {args.quote} | Pool Size: {args.n} | Threshold: {args.threshold}")

    loader = TimescaleDBLoader()
    df = loader.get_consistent_movers(
        start_date=args.start_date,
        end_date=args.end_date,
        daily_top_n=args.daily_n,
        output_n=args.n,
        trades_threshold=args.threshold,
        stables=args.stables,
        quote=args.quote,
    )

    if df.empty:
        print("Error: No consistent movers found.")
        sys.exit(1)

    df = df.sort_values(by=["average_notional_volume", "frequency"], ascending=[False, False])
    results = [
        {
            "rank": i + 1,
            "symbol": SYMBOL_MAPPING.get(row["symbol"], row["symbol"]),
            "kraken_symbol": row["symbol"],
            "frequency": row["frequency"],
            "average_notional_volume": row["average_notional_volume"],
        }
        for i, row in enumerate(df.to_dict(orient="records"))
    ]

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_path = os.path.abspath(os.path.join(project_root, output))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Successfully saved {len(results)} movers to {output_path}")
    print(pd.DataFrame(results).head(20).to_string(index=False))


def cmd_patch_notebook(args):
    """Patches notebook files for vectorbt exporting."""
    file_path = args.file
    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    if nb["cells"]:
        nb["cells"][0]["source"] = [
            "# Parameter Sensitivity Explorer\n",
            "\n",
            "This notebook performs a **vectorized grid search** using the `ggTrader` orchestrator api. It visualizes the profitability landscape to find robust parameter regions.\n",
            "\n",
            "### Exporting with Plots\n",
            "To ensure plots are included in your export:\n",
            "1. **HTML**: Use `File > Save and Export Notebook As... > HTML`.\n",
            "2. **PDF/WebPDF**: Use `File > Save and Export Notebook As... > WebPDF`.",
        ]

        for cell in nb["cells"]:
            if cell["cell_type"] == "code" and any(
                "import vectorbt as vbt" in line for line in cell["source"]
            ):
                new_source = []
                for line in cell["source"]:
                    if "import plotly.graph_objects as go" in line:
                        new_source.extend([line, "import plotly.io as pio\n"])
                    elif "from tabulate import tabulate" in line:
                        new_source.extend(
                            [
                                line,
                                "\n",
                                "# Configure Plotly renderer for export compatibility\n",
                                'pio.renderers.default = "notebook"\n',
                                "\n",
                                "# Ensure VectorBT uses the default plotly renderer\n",
                                "vbt.settings.plotting['layout']['template'] = 'vbt'\n",
                            ]
                        )
                    else:
                        new_source.append(line)
                cell["source"] = new_source
                break

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook patched successfully.")


def main():
    parser = argparse.ArgumentParser(description="Manage data and maintenance scripts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: backfill-4h
    subparsers.add_parser(
        "backfill-4h", help="One-time migration: build 4h and 30m candles from 1h and 15m data."
    )

    # Subcommand: ingest-kraken
    parser_ingest = subparsers.add_parser(
        "ingest-kraken", help="Ingest Kraken Data into PostgreSQL."
    )
    parser_ingest.add_argument(
        "--sync", action="store_true", help="Sync new directories from data/raw"
    )
    parser_ingest.add_argument(
        "--force", action="store_true", help="Force sync all directories (ignore manifest)"
    )
    parser_ingest.add_argument("--dir", type=str, help="Specific directory to ingest")

    # Subcommand: generate-pool
    parser_pool = subparsers.add_parser(
        "generate-pool", help="Generate Consistent Movers Asset Pool."
    )
    parser_pool.add_argument("--n", type=int, default=25, help="Number of assets in the final pool")
    parser_pool.add_argument("--daily-n", type=int, default=200, help="Number of daily top movers")
    parser_pool.add_argument("--start-date", type=str, default="2023-01-01", help="Start date")
    parser_pool.add_argument("--end-date", type=str, default="2025-12-31", help="End date")
    parser_pool.add_argument("--threshold", type=int, default=500, help="Minimum daily trades")
    parser_pool.add_argument("--quote", type=str, default="USD", help="Quote currency")
    parser_pool.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser_pool.add_argument("--stables", action="store_true", help="Include stablecoins and fiats")

    # Subcommand: patch-notebook
    parser_patch = subparsers.add_parser(
        "patch-notebook", help="Patches notebook files for vectorbt exporting."
    )
    default_nb_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "notebooks", "parameter_sensitivity_explorer.ipynb"
        )
    )
    parser_patch.add_argument("--file", type=str, default=default_nb_path, help="Path to notebook")

    args = parser.parse_args()

    try:
        if args.command == "backfill-4h":
            cmd_backfill(args)
        elif args.command == "ingest-kraken":
            cmd_ingest_kraken(args)
        elif args.command == "generate-pool":
            cmd_generate_pool(args)
        elif args.command == "patch-notebook":
            cmd_patch_notebook(args)
    except Exception as e:
        print(f"Error handling command '{args.command}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
