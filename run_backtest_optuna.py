# run_backtest_optuna.py
import io
from contextlib import redirect_stdout, nullcontext
from datetime import datetime, timedelta, timezone

from optuna.samplers import RandomSampler
from tabulate import tabulate
import optuna

from data_manager import CryptoDataManager, DataManager
from backtest_simulator import BacktestSimulator
from utils import align_end_to_interval  # NEW


def make_objective(symbols, price_data_map, initial_cash=1000.0, metric='final_equity', quiet=True, interval='4h', fee_pct=0.0):
    """
    metric options:
      - 'final_equity' (maximize)
      - 'total_return_pct' (maximize)
      - 'sharpe' (maximize)
    """
    def objective(trial: optuna.trial.Trial):
        # Sample parameters
        ema_fast = trial.suggest_int('ema_fast', 10, 40)
        ema_slow = trial.suggest_int('ema_slow', ema_fast + 10, ema_fast + 40)
        trailing_pct = round(trial.suggest_float('trailing_pct', 0.01, 0.08, step=0.002), 3)


        # Run a trial backtest (suppress prints for parallel trials)
        with nullcontext():
            sim = BacktestSimulator.run_with_uniform_params(
                symbols=symbols,
                price_data_map=price_data_map,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                trailing_pct=trailing_pct,
                initial_cash=initial_cash,
                interval=interval,
                fee_pct=fee_pct,
                verbose=False
            )
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
            value = 0.0

        return value

    return objective


def save_best_to_mongodb(strategy: str, interval: str, params: dict, metric_name: str, metric_value: float, symbols: list):
    dm = DataManager()
    db = dm.db
    coll = db['strategy_optimizations']

    try:
        coll.create_index(
            [('date', 1), ('strategy', 1), ('interval', 1)],
            unique=True,
            name='uniq_date_strategy_interval'
        )
    except Exception as e:
        print(f"Index creation error (strategy_optimizations): {e}")

    utc_today_midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    doc = {
        'date': utc_today_midnight,
        'strategy': strategy,
        'interval': interval,
        'params': params,
        'metric_name': metric_name,
        'metric_value': float(metric_value),
        'symbols': list(dict.fromkeys(symbols)),
        'saved_at': datetime.now(timezone.utc),
    }

    filt = {'date': utc_today_midnight, 'strategy': strategy, 'interval': interval}
    res = coll.update_one(filt, {'$set': doc}, upsert=True)
    if res.upserted_id or res.modified_count > 0:
        print("✅ Best optimization result saved to MongoDB (strategy_optimizations).")
    else:
        print("ℹ️ No changes made (document already up to date).")


def main():
    # Config
    INTERVAL = '4h'
    FEE_PCT = 0.001  # e.g., 0.1% per trade

    # Dynamic symbols via CryptoDataManager (centralized)
    symbols = CryptoDataManager.get_top_binance_usdt_symbols(top_n=10)
    if not symbols:
        symbols = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'BNBUSDT', 'LINKUSDT', 'TRXUSDT']
    symbols = list(dict.fromkeys(symbols))

    # Align end date to last fully closed bar and set start date
    now_utc = datetime.now(timezone.utc)
    end_date = align_end_to_interval(now_utc, INTERVAL)  # aligned to closed bar
    start_date = end_date - timedelta(days=30 * 6)

    initial_cash = 1000.0
    metric = 'final_equity'  # or: 'total_return_pct', 'sharpe'
    n_trials = 50
    strategy_name = 'EMA_trailing'

    # Load price data once from CryptoDataManager (centralized)
    cm = CryptoDataManager()
    price_data_map = cm.build_price_data(symbols, INTERVAL, start_date, end_date)

    # Create study and optimize
    study = optuna.create_study(
        direction='maximize',
        study_name='ema_trailing_optimization'

    )
    objective = make_objective(
        symbols,
        price_data_map,
        initial_cash=initial_cash,
        metric=metric,
        quiet=True,
        interval=INTERVAL,
        fee_pct=FEE_PCT
    )
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    # Save best result to MongoDB
    save_best_to_mongodb(
        strategy=strategy_name,
        interval=INTERVAL,
        params=best_params,
        metric_name=metric,
        metric_value=best_value,
        symbols=symbols
    )

    # Re-run backtest with best params and print full summaries (verbose)
    sim_best = BacktestSimulator.run_with_uniform_params(
        symbols,
        price_data_map,
        ema_fast=int(best_params['ema_fast']),
        ema_slow=int(best_params['ema_slow']),
        trailing_pct=float(best_params['trailing_pct']),
        initial_cash=initial_cash,
        interval=INTERVAL,
        fee_pct=FEE_PCT,
        verbose=True
    )
    sim_best.print_performance_summary()

    # Print results
    print("\nSymbols used:")
    print(tabulate([[i + 1, s] for i, s in enumerate(symbols)], headers=['#', 'Symbol'], tablefmt='github'))

    print("\nBest parameters:")
    param_rows = [{'param': k, 'value': v} for k, v in sorted(best_params.items())]
    print(tabulate(param_rows, headers='keys', tablefmt='github'))

    print("\nBest metric:")
    metric_rows = [{'metric': metric, 'value': best_value}]
    print(tabulate(metric_rows, headers='keys', tablefmt='github'))


if __name__ == '__main__':
    main()
