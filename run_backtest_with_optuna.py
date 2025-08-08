# run_backtest_optuna.py
import io
from contextlib import redirect_stdout, nullcontext
from datetime import datetime, timedelta, timezone

from optuna.samplers import RandomSampler
from tabulate import tabulate
import optuna

from data_manager import CryptoDataManager, DataManager
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
        slow_min = ema_fast + 10
        slow_high = max(slow_min + 10, 80)
        ema_slow = trial.suggest_int('ema_slow', slow_min, slow_high)
        trailing_pct = trial.suggest_float('trailing_pct', 0, .1)

        # Run a trial backtest
        buf = io.StringIO()
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

    return objective


def save_best_to_mongodb(strategy: str, interval: str, params: dict, metric_name: str, metric_value: float, symbols: list):
    """
    Upsert best optimization result into MongoDB with unique key (date, strategy, interval).
    - date is set to UTC midnight for the current day for idempotence.
    - params contains the optimized hyperparameters.
    """
    dm = DataManager()
    db = dm.db
    coll = db['strategy_optimizations']

    # Ensure unique index on (date, strategy, interval)
    try:
        coll.create_index(
            [('date', 1), ('strategy', 1), ('interval', 1)],
            unique=True,
            name='uniq_date_strategy_interval'
        )
    except Exception as e:
        print(f"Index creation error (strategy_optimizations): {e}")

    # True UTC midnight
    utc_today_min = datetime.now(timezone.utc).replace(second=0, microsecond=0)

    doc = {
        'date': utc_today_min,
        'strategy': strategy,
        'interval': interval,
        'params': params,
        'metric_name': metric_name,
        'metric_value': float(metric_value),
        'symbols': list(dict.fromkeys(symbols)),  # store deduped list
        'saved_at': datetime.now(timezone.utc),
    }

    filt = {'date': utc_today_min, 'strategy': strategy, 'interval': interval}
    res = coll.update_one(filt, {'$set': doc}, upsert=True)
    if res.upserted_id or res.modified_count > 0:
        print("✅ Best optimization result saved to MongoDB (strategy_optimizations).")
    else:
        print("ℹ️ No changes made (document already up to date).")


def get_top_binance_usdt_symbols(top_n=10, min_change=0.0, min_trades=0, min_volume=0):
    """
    Return a deduped list of the top USDT pairs on Binance.US by 24h quote volume.
    Filters out leveraged/ETF-like products.
    """
    try:
        top_pairs = CryptoDataManager.get_24hr_top_binance(
            top_n=top_n * 2,  # overfetch a bit to allow filtering
            quote='USDT',
            min_change=min_change,
            min_trades=min_trades,
            min_volume=min_volume
        )
        raw_symbols = [p.get('symbol', '') for p in top_pairs if isinstance(p, dict)]
        # Filter out leveraged/ETF-like tickers
        banned_fragments = ('UPUSDT', 'DOWNUSDT', 'BULLUSDT', 'BEARUSDT','USDCUSDT')
        filtered = [s for s in raw_symbols if s.endswith('USDT') and not any(b in s for b in banned_fragments)]
        # Deduplicate while preserving order
        deduped = list(dict.fromkeys(filtered))
        return deduped[:top_n]
    except Exception as e:
        print(f"⚠️ Failed to fetch top Binance.US pairs: {e}")
        return []


def main():
    # Attempt to fetch top USDT symbols dynamically
    symbols = get_top_binance_usdt_symbols(top_n=10)
    if not symbols:
        # Safe fallback to a small default set if API is down or filtered out everything
        symbols = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'BNBUSDT', 'LINKUSDT', 'TRXUSDT']

    # Deduplicate while preserving order
    symbols = list(dict.fromkeys(symbols))

    interval = '4h'
    end_date = datetime(2025, 8, 1, tzinfo=timezone.utc)
    start_date = end_date - timedelta(days=30 * 6)
    initial_cash = 1000.0
    metric = 'final_equity'  # or: 'total_return_pct', 'sharpe'
    n_trials = 50
    strategy_name = 'EMA_trailing'

    # Load price data once
    price_data_map = build_price_data(symbols, interval, start_date, end_date)

    # Create study and optimize
    study = optuna.create_study(
        direction='maximize',
        study_name='ema_trailing_optimization'

    )
    objective = make_objective(symbols, price_data_map, initial_cash=initial_cash, metric=metric, quiet=True)
    study.optimize(objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    # Print results safely with tabulate (use list of dict rows)
    print("\nSymbols used:")
    print(tabulate([[i + 1, s] for i, s in enumerate(symbols)], headers=['#', 'Symbol'], tablefmt='github'))

    print("\nBest parameters:")
    param_rows = [{'param': k, 'value': v} for k, v in sorted(best_params.items())]
    print(tabulate(param_rows, headers='keys', tablefmt='github'))

    print("\nBest metric:")
    metric_rows = [{'metric': metric, 'value': best_value}]
    print(tabulate(metric_rows, headers='keys', tablefmt='github'))

    # Save best result to MongoDB (unique by date, strategy, interval)
    save_best_to_mongodb(
        strategy=strategy_name,
        interval=interval,
        params=best_params,
        metric_name=metric,
        metric_value=best_value,
        symbols=symbols
    )

    # Re-run backtest with best params and print full summaries
    sim_best = run_backtest_with_params(
        symbols,
        price_data_map,
        ema_fast=int(best_params['ema_fast']),
        ema_slow=int(best_params['ema_slow']),
        trailing_pct=float(best_params['trailing_pct']),
        initial_cash=initial_cash
    )
    sim_best.print_performance_summary()


if __name__ == '__main__':
    main()
