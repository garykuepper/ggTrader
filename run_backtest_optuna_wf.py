
# run_backtest_wf.py
from datetime import datetime, timedelta, timezone
import statistics
import optuna
import pandas as pd
from tabulate import tabulate

from data_manager import CryptoDataManager, DataManager
from backtest_simulator import BacktestSimulator
from utils import align_end_to_interval


class WalkForwardObjective:
    def __init__(
        self,
        symbols: list,
        full_price_data_map: dict,
        interval: str,
        initial_cash: float,
        fee_pct: float,
        metric: str,
        train_len_days: int = 60,
        test_len_days: int = 14,
        step_days: int = 14,
        min_trades_per_window: int = 3,
    ):
        self.symbols = symbols
        self.full_price_data_map = full_price_data_map
        self.interval = interval
        self.initial_cash = float(initial_cash)
        self.fee_pct = float(fee_pct)
        self.metric = metric
        self.train_len_days = int(train_len_days)
        self.test_len_days = int(test_len_days)
        self.step_days = int(step_days)
        self.min_trades_per_window = int(min_trades_per_window)

        # Precompute available data range
        self.full_start = min(
            (df.index.min() for df in self.full_price_data_map.values() if df is not None and not df.empty),
            default=None,
        )
        self.full_end = max(
            (df.index.max() for df in self.full_price_data_map.values() if df is not None and not df.empty),
            default=None,
        )
        self.windows = self._build_windows()

    def _build_windows(self):
        if self.full_start is None or self.full_end is None:
            return []
        cur_start = self.full_start
        wins = []
        while True:
            train_start = cur_start
            train_end = train_start + timedelta(days=self.train_len_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_len_days)
            if test_end > self.full_end:
                break
            wins.append((train_start, train_end, test_start, test_end))
            cur_start = cur_start + timedelta(days=self.step_days)
        return wins

    @staticmethod
    def _slice_price_data(price_data_map, start, end):
        return {s: df[(df.index >= start) & (df.index <= end)] for s, df in price_data_map.items()}

    @staticmethod
    def _sum_trades(df: pd.DataFrame) -> int:
        return int(df['Trades'].sum()) if df is not None and not df.empty and 'Trades' in df.columns else 0

    @staticmethod
    def _metric_value(portfolio_df: pd.DataFrame, metric: str) -> float:
        try:
            if metric == 'final_equity':
                return float(portfolio_df['Final Equity'].iloc[0])
            if metric == 'total_return_pct':
                return float(portfolio_df['Total Return %'].iloc[0])
            if metric == 'sharpe':
                return float(portfolio_df['Sharpe'].iloc[0])
            return float(portfolio_df['Final Equity'].iloc[0])
        except Exception:
            return 0.0

    def __call__(self, trial: optuna.trial.Trial) -> float:
        # Sample parameters
        ema_fast = trial.suggest_int('ema_fast', 10, 40)
        ema_slow = trial.suggest_int('ema_slow', ema_fast + 10, ema_fast + 40)
        trailing_pct = round(trial.suggest_float('trailing_pct', 0.01, 0.08, step=0.002), 3)

        if not self.windows:
            return 0.0

        oos_scores = []

        for train_start, train_end, test_start, test_end in self.windows:
            # Warmup run: from train_start up to test_start
            warm_price_map = self._slice_price_data(self.full_price_data_map, train_start, test_start)
            sim_warm = BacktestSimulator.run_with_uniform_params(
                symbols=self.symbols,
                price_data_map=warm_price_map,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                trailing_pct=trailing_pct,
                initial_cash=self.initial_cash,
                interval=self.interval,
                fee_pct=self.fee_pct,
                verbose=False,
            )
            per_sym_warm, pf_warm = sim_warm.performance_summary()
            start_equity = float(pf_warm['Final Equity'].iloc[0]) if not pf_warm.empty else self.initial_cash
            trades_warm = self._sum_trades(per_sym_warm)

            # Full run: from train_start through test_end (test uses warmed indicators)
            full_price_map = self._slice_price_data(self.full_price_data_map, train_start, test_end)
            sim_full = BacktestSimulator.run_with_uniform_params(
                symbols=self.symbols,
                price_data_map=full_price_map,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                trailing_pct=trailing_pct,
                initial_cash=self.initial_cash,
                interval=self.interval,
                fee_pct=self.fee_pct,
                verbose=False,
            )
            per_sym_full, pf_full = sim_full.performance_summary()
            end_equity = float(pf_full['Final Equity'].iloc[0]) if not pf_full.empty else self.initial_cash
            trades_full = self._sum_trades(per_sym_full)

            # Test-only deltas
            test_trades = max(0, trades_full - trades_warm)
            test_pnl = end_equity - start_equity
            test_return_pct = ((end_equity / start_equity) - 1.0) * 100.0 if start_equity > 0 else 0.0

            # Guardrail on test-only trades
            if test_trades < self.min_trades_per_window:
                oos_scores.append(0.0)
                continue

            # Score test-only performance
            if self.metric == 'final_equity':
                # Keep scale near initial_cash so scores are comparable
                oos_scores.append(self.initial_cash + test_pnl)
            elif self.metric == 'total_return_pct':
                oos_scores.append(test_return_pct)
            else:
                # Fallback to full-period metric for unsupported ones
                oos_scores.append(self._metric_value(pf_full, self.metric))

        return float(statistics.median(oos_scores)) if oos_scores else 0.0


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
    params_clean = dict(params)
    if 'trailing_pct' in params_clean:
        params_clean['trailing_pct'] = round(float(params_clean['trailing_pct']), 3)

    doc = {
        'date': utc_today_midnight,
        'strategy': strategy,
        'interval': interval,
        'params': params_clean,
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
    FEE_PCT = 0.001
    INITIAL_CASH = 1000.0
    METRIC = 'final_equity'  # or 'total_return_pct'
    N_TRIALS = 50
    STRATEGY_NAME = 'EMA_trailing'

    # Symbols (dynamic with fallback)
    symbols = CryptoDataManager.get_top_binance_usdt_symbols(top_n=10)
    if not symbols:
        symbols = ['BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'ADAUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT', 'BNBUSDT', 'LINKUSDT', 'TRXUSDT']
    symbols = list(dict.fromkeys(symbols))

    # Period to load
    now_utc = datetime.now(timezone.utc)
    end_date = align_end_to_interval(now_utc, INTERVAL)
    start_date = end_date - timedelta(days=30 * 6)

    # Load data once
    cm = CryptoDataManager()
    price_data_map = cm.build_price_data(symbols, INTERVAL, start_date, end_date)

    # Objective
    wf_objective = WalkForwardObjective(
        symbols=symbols,
        full_price_data_map=price_data_map,
        interval=INTERVAL,
        initial_cash=INITIAL_CASH,
        fee_pct=FEE_PCT,
        metric=METRIC,
        train_len_days=60,
        test_len_days=14,
        step_days=14,
        min_trades_per_window=3,
    )

    study = optuna.create_study(direction='maximize', study_name='ema_trailing_optimization_wf')
    study.optimize(wf_objective, n_trials=N_TRIALS, n_jobs=1)

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params['trailing_pct'] = round(float(best_params.get('trailing_pct', 0.0)), 3)
    best_value = float(best_trial.value)

    # Save best
    save_best_to_mongodb(
        strategy=STRATEGY_NAME,
        interval=INTERVAL,
        params=best_params,
        metric_name=METRIC,
        metric_value=best_value,
        symbols=symbols
    )

    # Sanity check: run the best params on the last test window length
    recent_start = end_date - timedelta(days=14)
    recent_price_map = {s: df[(df.index >= recent_start) & (df.index <= end_date)] for s, df in price_data_map.items()}
    sim_best = BacktestSimulator.run_with_uniform_params(
        symbols,
        recent_price_map,
        ema_fast=int(best_params['ema_fast']),
        ema_slow=int(best_params['ema_slow']),
        trailing_pct=float(best_params['trailing_pct']),
        initial_cash=INITIAL_CASH,
        interval=INTERVAL,
        fee_pct=FEE_PCT,
        verbose=True
    )
    sim_best.print_performance_summary()

    print("\nSymbols used:")
    print(tabulate([[i + 1, s] for i, s in enumerate(symbols)], headers=['#', 'Symbol'], tablefmt='github'))

    print("\nBest parameters:")
    param_rows = [{'param': k, 'value': v} for k, v in sorted(best_params.items())]
    print(tabulate(param_rows, headers='keys', tablefmt='github'))

    print("\nBest metric (median OOS across windows):")
    metric_rows = [{'metric': METRIC, 'value': best_value}]
    print(tabulate(metric_rows, headers='keys', tablefmt='github'))


if __name__ == '__main__':
    main()
