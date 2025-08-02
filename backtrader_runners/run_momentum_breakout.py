import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tabulate import tabulate

from old.ggTrader_old.data_manager import UniversalDataManager
from old.ggTrader_old.utils.optimization_report import OptimizationReport
from old.ggTrader_old.utils.backtrader_utils import BacktraderUtils
from old.ggTrader_old.strats.momentum_breakout import MomentumBreakoutStrategy

symbol = "UPRO"
interval = "1d"
day_range = 365
marketType = "stock"

# Configure optimization parameters for momentum strategy
atr_periods = range(10, 22, 4)         # 3 values: 10, 14, 18
atr_multipliers = [1.5, 2.0, 2.5]     # 3 values
ema_fast_range = range(8, 17, 3)      # 3 values: 8, 11, 14
ema_slow_range = range(21, 42, 7)     # 3 values: 21, 28, 35
breakout_lookbacks = range(10, 26, 5) # 4 values: 10, 15, 20, 25
position_sizes = [0.75, 0.85, 0.95]  # 3 values

def run_optimization():
    """Run parameter optimization for momentum breakout strategy"""
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

    report = OptimizationReport()
    bt_utils = BacktraderUtils()

    print("# 🚀 MOMENTUM BREAKOUT OPTIMIZATION")
    print("*Advanced Momentum Breakout Strategy Parameter Optimization*")

    # Initialize data manager
    dm = UniversalDataManager(mongo_uri=mongo_uri)

    # Fetch data using global variables
    print("\n## 📈 Fetching Historical Data...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=day_range)).strftime('%Y-%m-%d')

    df = dm.load_or_fetch(symbol, interval, start_date, end_date, market=marketType)

    # Validate data
    is_valid, message = bt_utils.validate_data_for_optimization(df, min_length=200)
    if not is_valid:
        print(f"❌ **Error:** {message}")
        return

    report.print_data_summary(df)

    # Display configuration
    print(f"\n## ⚙️ OPTIMIZATION CONFIGURATION")
    total_combinations = bt_utils.calculate_parameter_combinations(
        atr_periods, atr_multipliers, ema_fast_range, ema_slow_range, breakout_lookbacks, position_sizes
    )

    config_data = [
        ["Strategy", "Momentum Breakout"],
        ["Symbol", symbol],
        ["Interval", interval],
        ["Day Range", f"{day_range} days"],
        ["ATR Periods", f"{min(atr_periods)} - {max(atr_periods)}"],
        ["ATR Multipliers", f"{min(atr_multipliers)} - {max(atr_multipliers)}"],
        ["Fast EMA Range", f"{min(ema_fast_range)} - {max(ema_fast_range)}"],
        ["Slow EMA Range", f"{min(ema_slow_range)} - {max(ema_slow_range)}"],
        ["Breakout Lookback", f"{min(breakout_lookbacks)} - {max(breakout_lookbacks)}"],
        ["Position Sizes", ", ".join([f"{p:.0%}" for p in position_sizes])],
        ["Total Combinations", f"{total_combinations:,}"]
    ]

    print(tabulate(config_data, headers=["Parameter", "Value"], tablefmt="github"))

    # Setup cerebro
    cerebro = bt_utils.setup_cerebro(initial_cash=10000.0, commission=0.001)
    cerebro = bt_utils.add_standard_analyzers(cerebro, risk_free_rate=0.02)

    cerebro.optstrategy(
        MomentumBreakoutStrategy,
        atr_period=atr_periods,
        atr_multiplier=atr_multipliers,
        ema_fast=ema_fast_range,
        ema_slow=ema_slow_range,
        breakout_lookback=breakout_lookbacks,
        position_pct=position_sizes
    )

    # Add data
    data = bt_utils.create_backtrader_data(df)
    cerebro.adddata(data)

    # Run optimization
    print(f"\n🔄 **Running optimization with {total_combinations:,} parameter combinations...**")
    results = cerebro.run()
    print(f"✅ **Optimization completed!** Analyzing {len(results):,} results...")

    # Process results
    optimization_results = []
    initial_value = 10000.0

    for result in results:
        strategy = result[0]

        # Extract parameters
        params = {
            'atr_period': strategy.params.atr_period,
            'atr_multiplier': strategy.params.atr_multiplier,
            'ema_fast': strategy.params.ema_fast,
            'ema_slow': strategy.params.ema_slow,
            'breakout_lookback': strategy.params.breakout_lookback,
            'position_pct': strategy.params.position_pct
        }

        # Extract standard results using utility
        strategy_results = bt_utils.extract_strategy_results(strategy, initial_value)

        # Combine parameters and results
        result_entry = {**params, **strategy_results}
        optimization_results.append(result_entry)

    # Sort by Sharpe ratio (good for momentum strategies)
    optimization_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

    # Display results
    report.format_results_table(optimization_results, "TOP 20 OPTIMIZATION RESULTS (by Sharpe Ratio)", 20)
    report.format_best_performers_table(optimization_results)
    report.format_summary_stats_table(optimization_results)
    report.print_performance_distribution(optimization_results)
    report.print_optimization_analysis(optimization_results)
    report.print_completion_summary()

    return optimization_results

if __name__ == "__main__":
    results = run_optimization()