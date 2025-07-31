import os
import backtrader as bt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from ggTrader.data_manager.universal_data_manager import UniversalDataManager
from ggTrader.utils.optimization_report import OptimizationReport
from ggTrader.utils.backtrader_utils import BacktraderUtils

load_dotenv()
mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")

class OptimizedSMAStrategy(bt.Strategy):
    """SMA crossover strategy with optimizable parameters"""
    params = (
        ('sma_fast', 10),
        ('sma_slow', 30),
        ('position_pct', 0.95),
    )

    def __init__(self):
        self.sma_fast = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_fast)
        self.sma_slow = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_slow)
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = (cash * self.params.position_pct) / price
                if size > 0:
                    self.buy(size=size)
        else:
            if self.crossover < 0:
                self.close()

def run_optimization():
    """Run parameter optimization with comprehensive analysis"""
    report = OptimizationReport()
    bt_utils = BacktraderUtils()

    print("# 🚀 ENHANCED BACKTRADER OPTIMIZATION")
    print("*SMA Crossover Strategy Parameter Optimization*")

    # Initialize data manager
    dm = UniversalDataManager(mongo_uri=mongo_uri)

    # Fetch data
    print("\n## 📈 Fetching Historical Data...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    df = dm.load_or_fetch("SPY", "1d", start_date, end_date, market="stock")

    # Validate data using utility
    is_valid, message = bt_utils.validate_data_for_optimization(df, min_length=100)
    if not is_valid:
        print(f"❌ **Error:** {message}")
        return

    report.print_data_summary(df)

    # Configure optimization parameters
    fast_range = range(5, 25, 3)
    slow_range = range(25, 60, 5)
    position_range = [0.8, 0.9, 0.95]

    report.print_optimization_config(fast_range, slow_range, position_range)

    # Setup cerebro using utility
    cerebro = bt_utils.setup_cerebro(initial_cash=10000.0, commission=0.001)
    cerebro = bt_utils.add_standard_analyzers(cerebro, risk_free_rate=0.02)

    cerebro.optstrategy(
        OptimizedSMAStrategy,
        sma_fast=fast_range,
        sma_slow=slow_range,
        position_pct=position_range
    )

    # Add data using utility
    data = bt_utils.create_backtrader_data(df)
    cerebro.adddata(data)

    # Run optimization
    total_combinations = bt_utils.calculate_parameter_combinations(fast_range, slow_range, position_range)
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
            'fast_sma': strategy.params.sma_fast,
            'slow_sma': strategy.params.sma_slow,
            'position_pct': strategy.params.position_pct
        }

        # Extract standard results using utility
        strategy_results = bt_utils.extract_strategy_results(strategy, initial_value)

        # Combine parameters and results
        result_entry = {**params, **strategy_results}
        optimization_results.append(result_entry)

    # Sort by return
    optimization_results.sort(key=lambda x: x['total_return'], reverse=True)

    # Display results using the report class
    report.format_results_table(optimization_results, "TOP 20 OPTIMIZATION RESULTS (by Total Return)", 20)
    report.format_best_performers_table(optimization_results)
    report.format_summary_stats_table(optimization_results)
    report.print_performance_distribution(optimization_results)
    report.print_optimization_analysis(optimization_results)
    report.print_completion_summary()

    return optimization_results

if __name__ == "__main__":
    results = run_optimization()