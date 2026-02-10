import os
import sys
import argparse
import json

# Ensure project root is in path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
)

import pandas as pd
import vectorbt as vbt
from tabulate import tabulate
from ggTrader.core.trading import Trading
from ggTrader.data.kraken.historical_data import KrakenHistoricalData
from ggTrader.utils.results_manager import ResultsManager, get_latest_params
from ggTrader.utils.config import load_symbols_from_json
from ggTrader.core.backtest import Backtest
from datetime import datetime

DEFAULT_SYMBOLS = ["BTC", "ETH", "XRP", "SOL", "DOGE"]
SYMBOLS_FILE_DEFAULT = "data/top_50_consistent_movers.json"


def run_backtest(params_path=None, symbols_file=None):
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
    interval = "4h"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    start_cash = 10000
    top_n_movers = 5
    max_position = 0.2

    # Initialize ResultsManager
    rm = ResultsManager("run_backtest")

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

    print(f"--- Backtest Configuration ---")
    print(f"Symbols: {symbols}")
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
    final_value = engine.portfolio.total_value
    profit = engine.portfolio.profit
    profit_pct = engine.portfolio.profit_pct * 100

    print(f"Final Portfolio Value: {final_value:.2f}")
    print(f"Profit/Loss: {profit:.2f} ({profit_pct:.2f}%)")
    print(f"Total Transactions: {len(engine.portfolio.trades)}")

    # Save Metadata and Metrics
    metadata = {
        "script": "run_backtest",
        "timestamp": datetime.now().isoformat(),
        "symbols": symbols,
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date,
        "strategy_params": strategy_params,
        "final_value": final_value,
        "profit": profit,
        "profit_pct": profit_pct,
    }
    rm.save_metadata(metadata)

    if engine.portfolio.trades:
        trades_dict = [t.as_dict() for t in engine.portfolio.trades]
        history_df = pd.DataFrame(trades_dict)
        rm.save_metrics(history_df, "trade_history.csv")
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
                init_cash=start_cash,
                freq=interval,
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

            # Save core metrics to performance_metrics table
            rm.db_manager.add_metrics(
                rm.run_id,
                {
                    "final_value": float(final_value),
                    "profit": float(profit),
                    "profit_pct": float(profit_pct),
                    "total_trades": float(len(engine.portfolio.trades)),
                },
            )

            # Plot aggregate drawdowns

            # Plot aggregate drawdowns
            fig_dd = val.vbt.drawdowns.plot()
            fig_dd.update_layout(title=f"VectorBT Portfolio Drawdowns - {interval}")
            rm.save_plotly_figure(fig_dd, "portfolio_drawdowns")

            print("VectorBT plots saved.")
        except Exception as e:
            print(f"VectorBT plotting failed: {e}")
            import traceback

            traceback.print_exc()

    # --- 6. Export to Excel ---
    print("\nExporting to Excel...")
    excel_data = {
        "Metadata": pd.DataFrame.from_dict(metadata, orient="index", columns=["Value"]),
        "Strategy Params": pd.DataFrame.from_dict(
            strategy_params, orient="index", columns=["Value"]
        ),
    }
    if engine.portfolio.trades:
        excel_data["Trade History"] = pd.DataFrame(
            [t.as_dict() for t in engine.portfolio.trades]
        )

    # Portfolio stats
    portfolio_stats = engine.portfolio.stats_dict()
    excel_data["Portfolio Stats"] = pd.DataFrame.from_dict(
        portfolio_stats, orient="index", columns=["Value"]
    )

    excel_path = rm.save_excel(excel_data, "backtest_results.xlsx")
    print(f"Excel report saved to: {excel_path}")

    print(f"\nResults saved to: {rm.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a strategy backtest.")
    parser.add_argument("--params", type=str, help="Path to a params.json file.")
    parser.add_argument(
        "--symbols-file",
        type=str,
        default=SYMBOLS_FILE_DEFAULT,
        help="Path to JSON file with symbols",
    )
    args = parser.parse_args()

    run_backtest(params_path=args.params, symbols_file=args.symbols_file)
