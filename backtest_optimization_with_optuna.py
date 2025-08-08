# run_backtest_optuna.py
import io
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from tabulate import tabulate
import optuna

from data_manager import CryptoDataManager
from backtest_simulator import BacktestSimulator
from trading_strategy import EMAStrategy


def build_price_data(symbols, interval, start_date, end_date):
    cm = CryptoDataManager()
    price_data_map = {}
    for sym in symbols:
        price_data_map[sym] = cm.get_crypto_data(sym, interval, start_date, end_date)
    return price_data_map


def run_backtest_with_params(symbols, price_data_map, ema_fast, ema_slow, trailing_pct, initial_cash=1000.0):
    # Equal position sizing across all symbols
    if not symbols:
        raise ValueError("symbols list is empty; cannot compute equal position size.")
    pos_pct = 1.0 / len(symbols)
    position_pct_map = {s: pos_pct for s in symbols}

    # One EMAStrategy per symbol, same params across symbols for simplicity
    symbol_strategy_map = {}
    for sym in symbols:
        params = {'ema_fast': ema_fast, 'ema_slow': ema_slow}
        strategy_name = f"{sym} EMA({ema_fast},{ema_slow})"
        symbol_strategy_map[sym] = EMAStrategy(strategy_name, params, trailing_pct=trailing_pct)

    sim = BacktestSimulator(
        symbol_strategy_map=symbol_strategy_map,
        price_data_map=price_data_map,
        position_pct_map=position_pct_map,
        initial_cash=initial_cash
    )
    sim.run()
    return sim


def make_objective(symbols, price_data_map, initial_cash=1000.0, metric='final_equity', quiet=True):
    """
    metric options:
      - 'final_equity' (maximize)
      - 'total_return_pct' (maximize)
      - 'sharpe' (maximize)
    """
    def objective(trial: optuna.trial.Trial):
        # Sample parameters
        ema_fast = trial.suggest_int('ema_fast', 3, 50)
        slow_min = ema_fast + 5
        slow_high = max(slow_min + 5, 100)
        ema_slow = trial.suggest_int('ema_slow', slow_min, slow_high)
        trailing_pct = trial.suggest_float('trailing_pct', 0.0, 0.1)

        # Run a trial backtest
        buf = io.StringIO()
        # Suppress verbose printing during trials for speed/noise reduction
        ctx = redirect_stdout(buf) if quiet else nullcontext()
        with ctx:
            sim = run_backtest_with_params(
                symbols, price_data_map, ema_fast, ema_slow, trailing_pct, initial_cash=initial_cash
            )
            # Compute metric
            per_sym_df, portfolio_df = sim.performance_summary()

        # Extract desired metric
        try:
            if metric == 'final_equity':
                value = float(portfolio_df['Final Equity'].iloc[0])
            elif metric == 'total_return_pct':
                value = float(portfolio_df['Total Return %'].iloc[0])
            elif metric == 'sharpe':
                value = float(portfolio_df['Sharpe'].iloc[0])
            else:
                value = float(portfolio_df['Final Equity'].iloc[0])
        except Exception:
            # Fallback if anything goes wrong computing the metric
            value = 0.0

        return value

    # Helper if quiet=False (avoid NameError)
    class nullcontext:
        def __enter__(self): return None
        def __exit__(self, exc_type, exc, tb): return False

    return objective


def main():
    # Config
    symbols = ['BTCUSDT',
               'ETHUSDT',
               'LTCUSDT',
               'ADAUSDT',
               'SOLUSDT',
               'DOGEUSDT',
               'XRPUSDT',
               'BNBUSDT',]
    interval = '4h'
    end_date = datetime(2025, 8, 1, tzinfo=timezone.utc)
    start_date = end_date - timedelta(days=30*3)
    initial_cash = 1000.0
    metric = 'final_equity'  # or: 'total_return_pct', 'sharpe'
    n_trials = 100

    # Load price data once
    price_data_map = build_price_data(symbols, interval, start_date, end_date)

    # Create study and optimize
    study = optuna.create_study(direction='maximize', study_name='ema_trailing_optimization')
    objective = make_objective(symbols, price_data_map, initial_cash=initial_cash, metric=metric, quiet=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    # Print results safely with tabulate (use list of dict rows)
    print("\nBest parameters:")
    param_rows = [{'param': k, 'value': v} for k, v in sorted(best_params.items())]
    print(tabulate(param_rows, headers='keys', tablefmt='github'))

    print("\nBest metric:")
    metric_rows = [{'metric': metric, 'value': best_value}]
    print(tabulate(metric_rows, headers='keys', tablefmt='github'))


    # Re-run backtest with best params and print full summaries
    best = study.best_params
    sim_best = run_backtest_with_params(
        symbols,
        price_data_map,
        ema_fast=int(best['ema_fast']),
        ema_slow=int(best['ema_slow']),
        trailing_pct=float(best['trailing_pct']),
        initial_cash=initial_cash
    )
    # sim_best.print_trade_history()
    sim_best.print_performance_summary()

    # Example: plot a symbol window if you want (optional)
    # sim_best.plot_symbol('LTCUSDT', start=start_date, end=end_date)


if __name__ == '__main__':
    main()
