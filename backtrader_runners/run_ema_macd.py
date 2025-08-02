import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from old.ggTrader_old.data_manager import UniversalDataManager
from old.ggTrader_old.utils.optimization_report import OptimizationReport
from old.ggTrader_old.utils.backtrader_utils import BacktraderUtils
from old.ggTrader_old.strats.ema_macd import EMAMACDStrategy

# Configuration parameters
symbol = "SPY"
interval = "1d"
day_range = 365
marketType = "stock"

# Updated optimization parameters for dual EMA + MACD strategy
ema_fast_periods = range(8, 16, 2)    # 4 values: 8, 10, 12, 14
ema_slow_periods = range(20, 32, 4)   # 3 values: 20, 24, 28
macd_fast_periods = range(10, 16, 2)  # 3 values: 10, 12, 14
macd_slow_periods = range(24, 31, 3)  # 3 values: 24, 27, 30
macd_signal_periods = range(7, 13, 2) # 3 values: 7, 9, 11
position_sizes = [0.85, 0.95]        # 2 values
stop_losses = [0.03, 0.05, 0.07]     # 3 values



load_dotenv()
mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")


def run_optimization():
    """Run EMA MACD strategy optimization"""
    report = OptimizationReport()
    bt_utils = BacktraderUtils()

    print("# 🚀 EMA MACD STRATEGY OPTIMIZATION")
    print("*EMA Trend Filter + MACD Signal Strategy*")

    # Initialize data manager
    dm = UniversalDataManager(mongo_uri=mongo_uri)

    # Fetch data
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

    total_combinations = (len(ema_fast_periods) * len(ema_slow_periods) *
                          len(macd_fast_periods) * len(macd_slow_periods) *
                          len(macd_signal_periods) * len(position_sizes) * len(stop_losses))

    print(f"\n## 🔧 Optimization Configuration")
    print(f"| Parameter | Range | Values |")
    print(f"|-----------|-------|--------|")
    print(f"| Symbol | - | {symbol} |")
    print(f"| Market | - | {marketType} |")
    print(f"| Interval | - | {interval} |")
    print(f"| Day Range | - | {day_range} |")
    print(f"| EMA Fast | {min(ema_fast_periods)}-{max(ema_fast_periods)} | {list(ema_fast_periods)} |")
    print(f"| EMA Slow | {min(ema_slow_periods)}-{max(ema_slow_periods)} | {list(ema_slow_periods)} |")
    print(f"| MACD Fast | {min(macd_fast_periods)}-{max(macd_fast_periods)} | {list(macd_fast_periods)} |")
    print(f"| MACD Slow | {min(macd_slow_periods)}-{max(macd_slow_periods)} | {list(macd_slow_periods)} |")
    print(f"| MACD Signal | {min(macd_signal_periods)}-{max(macd_signal_periods)} | {list(macd_signal_periods)} |")
    print(f"| Position Size | - | {position_sizes} |")
    print(f"| Stop Loss | - | {stop_losses} |")
    print(f"| **Total Combinations** | **{total_combinations:,}** | |")

    # Setup cerebro
    cerebro = bt_utils.setup_cerebro(initial_cash=10000.0, commission=0.001)
    cerebro = bt_utils.add_standard_analyzers(cerebro, risk_free_rate=0.02)

    # Update the optstrategy call:
    cerebro.optstrategy(
        EMAMACDStrategy,
        ema_fast=ema_fast_periods,
        ema_slow=ema_slow_periods,
        macd_fast=macd_fast_periods,
        macd_slow=macd_slow_periods,
        macd_signal=macd_signal_periods,
        position_pct=position_sizes,
        stop_loss_pct=stop_losses
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

        params = {
            'ema_fast': strategy.params.ema_fast,
            'ema_slow': strategy.params.ema_slow,
            'macd_fast': strategy.params.macd_fast,
            'macd_slow': strategy.params.macd_slow,
            'macd_signal': strategy.params.macd_signal,
            'position_pct': strategy.params.position_pct,
            'stop_loss_pct': strategy.params.stop_loss_pct
        }

        strategy_results = bt_utils.extract_strategy_results(strategy, initial_value)
        result_entry = {**params, **strategy_results}
        optimization_results.append(result_entry)

    # Sort by Sharpe ratio for better risk-adjusted performance
    optimization_results.sort(key=lambda x: x.get('sharpe_ratio', -999), reverse=True)

    # Display results
    report.format_results_table(optimization_results, "TOP 20 EMA MACD RESULTS (by Sharpe Ratio)", 20)
    report.format_best_performers_table(optimization_results)
    report.format_summary_stats_table(optimization_results)
    report.print_performance_distribution(optimization_results)
    report.print_optimization_analysis(optimization_results)
    report.print_completion_summary()

    return optimization_results

if __name__ == "__main__":
    results = run_optimization()