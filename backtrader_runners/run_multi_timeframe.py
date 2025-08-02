import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tabulate import tabulate

from old.ggTrader_old.data_manager import UniversalDataManager
from old.ggTrader_old.utils.optimization_report import OptimizationReport
from old.ggTrader_old.utils.backtrader_utils import BacktraderUtils
from old.ggTrader_old.strats.multi_timeframe import MultiTimeFrameStrategy

symbol = "LTCUSDT"
interval = "5m"
day_range = 14

def run_optimization():
    """Run parameter optimization for multi-timeframe strategy"""
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

    report = OptimizationReport()
    bt_utils = BacktraderUtils()

    print("# 🚀 MULTI-TIMEFRAME OPTIMIZATION")
    print("*Multi-Timeframe Strategy Parameter Optimization*")

    # Initialize data manager
    dm = UniversalDataManager(mongo_uri=mongo_uri)

    # Fetch data for multiple timeframes using global variables
    print("\n## 📈 Fetching Historical Data...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=day_range)).strftime('%Y-%m-%d')

    # Primary timeframe (5m for trading)
    df_primary = dm.load_or_fetch(symbol, interval, start_date, end_date, market="crypto")

    # Higher timeframe (15m for trend) - multiply interval by 3 for higher timeframe
    higher_interval = "15m" if interval == "5m" else "1h"
    df_higher = dm.load_or_fetch(symbol, higher_interval, start_date, end_date, market="crypto")

    # Validate data
    is_valid_primary, message_primary = bt_utils.validate_data_for_optimization(df_primary, min_length=200)
    is_valid_higher, message_higher = bt_utils.validate_data_for_optimization(df_higher, min_length=50)

    if not is_valid_primary:
        print(f"❌ **Error (Primary {interval} data):** {message_primary}")
        return
    if not is_valid_higher:
        print(f"❌ **Error (Higher {higher_interval} data):** {message_higher}")
        return

    print(f"\n### {interval.upper()} Data Summary (Primary)")
    report.print_data_summary(df_primary)

    print(f"\n### {higher_interval.upper()} Data Summary (Higher Timeframe)")
    report.print_data_summary(df_higher)

    # Configure optimization parameters
    fast_ma_range = range(5, 20, 3)        # Fast MA: 5, 8, 11, 14, 17
    slow_ma_range = range(20, 45, 5)       # Slow MA: 20, 25, 30, 35, 40
    trend_ma_range = range(40, 70, 5)      # Trend MA: 40, 45, 50, 55, 60, 65
    rsi_periods = range(10, 20, 2)         # RSI periods: 10, 12, 14, 16, 18
    rsi_oversold_levels = range(25, 36, 2) # Oversold: 25, 27, 29, 31, 33, 35
    position_sizes = [0.8, 0.9, 0.95]     # Position sizing

    # Display configuration
    print(f"\n## ⚙️ OPTIMIZATION CONFIGURATION")
    total_combinations = bt_utils.calculate_parameter_combinations(
        fast_ma_range, slow_ma_range, trend_ma_range, rsi_periods, rsi_oversold_levels, position_sizes
    )

    config_data = [
        ["Strategy", "Multi-Timeframe"],
        ["Symbol", symbol],
        ["Trading Timeframe", interval],
        ["Trend Timeframe", higher_interval],
        ["Day Range", f"{day_range} days"],
        ["Fast MA Range", f"{min(fast_ma_range)} - {max(fast_ma_range)}"],
        ["Slow MA Range", f"{min(slow_ma_range)} - {max(slow_ma_range)}"],
        ["Trend MA Range", f"{min(trend_ma_range)} - {max(trend_ma_range)}"],
        ["RSI Periods", f"{min(rsi_periods)} - {max(rsi_periods)}"],
        ["RSI Oversold", f"{min(rsi_oversold_levels)} - {max(rsi_oversold_levels)}"],
        ["Position Sizes", ", ".join([f"{p:.0%}" for p in position_sizes])],
        ["Total Combinations", f"{total_combinations:,}"],
    ]

    print(tabulate(config_data, headers=["Parameter", "Value"], tablefmt="github"))

    # Setup cerebro
    cerebro = bt_utils.setup_cerebro(initial_cash=10000.0, commission=0.001)
    cerebro = bt_utils.add_standard_analyzers(cerebro, risk_free_rate=0.02)

    cerebro.optstrategy(
        MultiTimeFrameStrategy,
        fast_ma=fast_ma_range,
        slow_ma=slow_ma_range,
        trend_ma=trend_ma_range,
        rsi_period=rsi_periods,
        rsi_oversold=rsi_oversold_levels,
        position_pct=position_sizes
    )

    # Add both timeframes
    data_primary = bt_utils.create_backtrader_data(df_primary)
    data_higher = bt_utils.create_backtrader_data(df_higher)

    cerebro.adddata(data_primary, name=f'primary_{interval}')  # Primary timeframe
    cerebro.adddata(data_higher, name=f'higher_{higher_interval}')  # Higher timeframe

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
            'fast_ma': strategy.params.fast_ma,
            'slow_ma': strategy.params.slow_ma,
            'trend_ma': strategy.params.trend_ma,
            'rsi_period': strategy.params.rsi_period,
            'rsi_oversold': strategy.params.rsi_oversold,
            'position_pct': strategy.params.position_pct
        }

        # Extract standard results using utility
        strategy_results = bt_utils.extract_strategy_results(strategy, initial_value)

        # Combine parameters and results
        result_entry = {**params, **strategy_results}
        optimization_results.append(result_entry)

    # Sort by Sharpe ratio
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