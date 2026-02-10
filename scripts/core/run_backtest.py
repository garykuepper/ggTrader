import os
import sys
import argparse
import json
from datetime import datetime
import pandas as pd
import vectorbt as vbt
from tabulate import tabulate

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

from ggTrader.core.trading import Trading
from ggTrader.data.kraken.historical_data import KrakenHistoricalData
from ggTrader.utils.results_manager import ResultsManager
from ggTrader.utils.config import load_symbols_from_json

DEFAULT_SYMBOLS = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SYMBOLS_FILE_DEFAULT = os.path.abspath(
    os.path.join(SCRIPT_DIR, "../../data/top_50_consistent_movers.json")
)


def run_backtest(
    params_path=None,
    symbols_file=None,
    start_date="2025-01-01",
    end_date="2025-12-31",
    interval="4h",
    start_cash=10000,
    top_n_movers=25,
    max_position=0.1,
    run_name="run_backtest",
):
    # --- 1. Configuration ---
    # Load Symbols
    symbols = None
    if symbols_file:
        symbols = load_symbols_from_json(symbols_file)
        if symbols:
            print(f"Loaded {len(symbols)} symbols from {symbols_file}")

    if not symbols:
        symbols = DEFAULT_SYMBOLS
        print(f"Using default symbols: {symbols}")

    # Initialize ResultsManager
    rm = ResultsManager(run_name)

    # Strategy Parameters
    strategy_params = {
        "adx_threshold": 25,
        "adx_length": 14,
        "sar_acceleration": 0.02,
        "sar_maximum": 0.2,
        "atr_multiplier": 3.0,
        "atr_length": 14,
        "use_dmp_cross": False,
    }

    # Load parameters if provided
    if params_path:
        print(f"Loading parameters from {params_path}...")
        strategy_params.update(rm.load_params(params_path))
    elif os.path.exists("params.json"):
        print("Loading parameters from local params.json...")
        strategy_params.update(rm.load_params("params.json"))

    print(f"--- Backtest Configuration ({run_name}) ---")
    print(f"Symbols: {len(symbols)} symbols")
    print(f"Range: {start_date} to {end_date}")
    print(f"Interval: {interval}")
    print(f"Start Cash: {start_cash}")
    print(f"Parameters: {strategy_params}")
    print(f"------------------------------\n")

    # --- 2. Data Loading ---
    k_h = KrakenHistoricalData()
    start_dt = pd.to_datetime(start_date).tz_localize("UTC")
    end_dt = pd.to_datetime(end_date).tz_localize("UTC")

    print("Loading data...")
    ohlcv_df = k_h.get_ohlcv_df(symbols, interval=interval, start=start_dt, end=end_dt)

    if isinstance(ohlcv_df.columns, pd.MultiIndex):
        ohlcv_df.columns.names = [None] * ohlcv_df.columns.nlevels
    else:
        ohlcv_df.columns.name = None
    ohlcv_df.index.name = None

    if ohlcv_df.empty:
        print("No data found for the specified range and symbols.")
        return

    date_range = ohlcv_df.index

    # --- 3. Run Simulation ---
    engine = Trading(
        ohlcv_df=ohlcv_df,
        date_range=date_range,
        start_cash=start_cash,
        top_n_movers=top_n_movers,
        max_position=max_position,
        strategy_params=strategy_params,
    )

    print("Running backtest engine...")
    engine.run()

    # --- 4. Results ---
    print("\n--- Backtest Results ---")
    portfolio_stats = engine.portfolio.stats_dict()

    print(f"Final Portfolio Value: {portfolio_stats['total_value']:.2f}")
    print(
        f"Profit/Loss: {portfolio_stats['total_profit']:.2f} ({portfolio_stats['profit_pct']:.2f}%)"
    )
    print(f"Total Transactions: {portfolio_stats['total_trades']}")

    # Standardized Metadata
    metadata = {
        "script": "run_backtest",
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "symbols": symbols if len(symbols) < 20 else f"{len(symbols)} symbols",
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date,
        "strategy_params": strategy_params,
    }
    # Update and SAVE Metadata FIRST (satisfies foreign key for trades and metrics)
    metadata.update(portfolio_stats)
    rm.save_metadata(metadata)

    # Save to metrics table
    rm.db_manager.add_metrics(rm.run_id, portfolio_stats)

    if engine.portfolio.trades:
        trades_dict = [t.as_dict() for t in engine.portfolio.trades]
        # Explicit standardization for DuckDB schema
        history_df = pd.DataFrame(trades_dict).rename(
            columns={"entry_date": "entry_time", "exit_date": "exit_time"}
        )

        # Save detailed trades to database
        rm.save_trades(history_df)

        print("\nLast 10 trades:")
        print(tabulate(history_df.tail(10), headers="keys", tablefmt="github"))

        # --- 5. Advanced Visualization (VectorBT) ---
        print("\nGenerating VectorBT plots...")
        try:
            pf = vbt.Portfolio.from_signals(
                close=engine.bulk_prices,
                entries=engine.bulk_entries,
                exits=engine.bulk_exits,
                fees=engine.portfolio.transaction_fee,
                init_cash=start_cash / len(symbols),
                freq=interval,
                size=max_position,
                size_type="percent",
                direction="longonly",
            )

            # Ensure we have a single Series for the total portfolio value
            val = pf.value()
            if isinstance(val, pd.DataFrame):
                val = val.sum(axis=1)

            # Plot Total Equity
            fig = val.vbt.plot()
            fig.update_layout(title=f"VectorBT Total Portfolio Value - {interval}")
            rm.save_plotly_figure(fig, "portfolio_value")

            # Save equity series to database
            rm.save_equity_curve(val, "portfolio_value")

            # Plot aggregate drawdowns
            fig_dd = val.vbt.drawdowns.plot()
            fig_dd.update_layout(title=f"VectorBT Portfolio Drawdowns - {interval}")
            rm.save_plotly_figure(fig_dd, "portfolio_drawdowns")

            print("VectorBT plots saved.")
        except Exception as e:
            print(f"VectorBT visualization failed: {e}")

    print(f"\nResults saved to: {rm.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a flexible strategy backtest.")
    parser.add_argument("--params", type=str, help="Path to a params.json file.")
    parser.add_argument(
        "--symbols-file",
        type=str,
        default=SYMBOLS_FILE_DEFAULT,
        help="Path to JSON file with symbols",
    )
    parser.add_argument(
        "--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--interval", type=str, default="4h", help="Data interval (e.g., 1h, 4h, 1d)"
    )
    parser.add_argument("--cash", type=float, default=10000, help="Starting cash")
    parser.add_argument(
        "--top-n", type=int, default=25, help="Number of top movers to watch"
    )
    parser.add_argument(
        "--max-pos",
        type=float,
        default=0.1,
        help="Max position size as decimal (0.1 = 10%)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="run_backtest",
        help="Name for the results directory",
    )

    args = parser.parse_args()

    run_backtest(
        params_path=args.params,
        symbols_file=args.symbols_file,
        start_date=args.start,
        end_date=args.end,
        interval=args.interval,
        start_cash=args.cash,
        top_n_movers=args.top_n,
        max_position=args.max_pos,
        run_name=args.name,
    )
