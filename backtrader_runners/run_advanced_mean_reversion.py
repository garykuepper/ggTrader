import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tabulate import tabulate

from old.ggTrader_old.data_manager import UniversalDataManager
from old.ggTrader_old.utils.optimization_report import OptimizationReport
from old.ggTrader_old.utils.backtrader_utils import BacktraderUtils
from old.ggTrader_old.strats.advanced_mean_reversion import AdvancedMeanReversionStrategy

symbol= "SPY"
interval = "1d"
day_range = 365*3
marketType = "stock"
# Configure optimization parameters for advanced mean reversion
rsi_periods = range(10, 26, 2)         # 8 values: 10, 12, 14, 16, 18, 20, 22, 24
rsi_oversold_levels = range(15, 36, 5) # 5 values: 15, 20, 25, 30, 35
bb_periods = range(12, 26, 3)          # 5 values: 12, 15, 18, 21, 24
position_sizes = [0.75, 0.85, 0.95]   # 3 values

def run_optimization():
    """Run parameter optimization with sophisticated strategy"""
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

    report = OptimizationReport()
    bt_utils = BacktraderUtils()

    print("# 🚀 ADVANCED STRATEGY OPTIMIZATION")
    print("*Multi-Indicator Mean Reversion Strategy*")

    # Initialize data manager
    dm = UniversalDataManager(mongo_uri=mongo_uri)

    # Fetch longer period for advanced strategies
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
        rsi_periods, rsi_oversold_levels, bb_periods, position_sizes
    )

    config_data = [
        ["Strategy", "Advanced Mean Reversion"],
        ["RSI Periods", f"{min(rsi_periods)} - {max(rsi_periods)}"],
        ["RSI Oversold Levels", f"{min(rsi_oversold_levels)} - {max(rsi_oversold_levels)}"],
        ["Bollinger Band Periods", f"{min(bb_periods)} - {max(bb_periods)}"],
        ["Position Sizes", ", ".join([f"{p:.0%}" for p in position_sizes])],
        ["Total Combinations", f"{total_combinations:,}"],
    ]

    print(tabulate(config_data, headers=["Parameter", "Value"], tablefmt="github"))

    # Setup cerebro
    cerebro = bt_utils.setup_cerebro(initial_cash=10000.0, commission=0.001)
    cerebro = bt_utils.add_standard_analyzers(cerebro, risk_free_rate=0.02)

    cerebro.optstrategy(
        AdvancedMeanReversionStrategy,
        rsi_period=rsi_periods,
        rsi_oversold=rsi_oversold_levels,
        bb_period=bb_periods,
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
            'rsi_period': strategy.params.rsi_period,
            'rsi_oversold': strategy.params.rsi_oversold,
            'bb_period': strategy.params.bb_period,
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