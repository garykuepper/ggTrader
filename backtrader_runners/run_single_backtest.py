import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from old.ggTrader_old.data_manager import UniversalDataManager
from old.ggTrader_old.utils.backtrader_utils import BacktraderUtils
from old.ggTrader_old.strats.ema_macd import EMAMACDStrategy
from old.ggTrader_old.strats.advanced_mean_reversion import AdvancedMeanReversionStrategy
from old.ggTrader_old.strats.momentum_breakout import MomentumBreakoutStrategy

# Configuration parameters
symbol = "SPY"
interval = "1d"
day_range = 365
marketType = "stock"

# Strategy selection and parameters
strategy_name = "ema_macd"  # Options: "ema_macd", "mean_reversion", "momentum_breakout"

# Strategy-specific parameters
strategy_params = {
    "ema_macd": {
        'ema_fast': 8,
        'ema_slow': 20,
        'macd_fast': 10,
        'macd_slow': 27,
        'macd_signal': 7,
        'position_pct': 0.95,
        'stop_loss_pct': 0.05
    },
    "mean_reversion": {
        'rsi_period': 14,
        'rsi_oversold': 30,
        'bb_period': 20,
        'position_pct': 0.95
    },
    "momentum_breakout": {
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'ema_fast': 12,
        'ema_slow': 26,
        'breakout_lookback': 20,
        'position_pct': 0.95
    }
}

# Backtest settings
initial_cash = 10000.0
commission = 0.001
risk_free_rate = 0.02

load_dotenv()
mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")


def get_strategy_class(strategy_name):
    """Return the strategy class based on name"""
    strategies = {
        "ema_macd": EMAMACDStrategy,
        "mean_reversion": AdvancedMeanReversionStrategy,
        "momentum_breakout": MomentumBreakoutStrategy
    }
    return strategies.get(strategy_name)


def run_single_backtest():
    """Run a single backtest with specified parameters"""
    bt_utils = BacktraderUtils()

    print(f"# 🚀 SINGLE BACKTEST - {strategy_name.upper().replace('_', ' ')} STRATEGY")
    print(f"*Testing {symbol} with specific parameters*")

    # Initialize data manager
    dm = UniversalDataManager(mongo_uri=mongo_uri)

    # Fetch data
    print("\n## 📈 Fetching Historical Data...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=day_range)).strftime('%Y-%m-%d')

    df = dm.load_or_fetch(symbol, interval, start_date, end_date, market=marketType)

    # Validate data
    is_valid, message = bt_utils.validate_data_for_optimization(df, min_length=50)
    if not is_valid:
        print(f"❌ **Error:** {message}")
        return

    print(f"✅ **Data loaded:** {len(df)} bars from {start_date} to {end_date}")

    # Get strategy class and parameters
    strategy_class = get_strategy_class(strategy_name)
    if not strategy_class:
        print(f"❌ **Error:** Unknown strategy '{strategy_name}'")
        return

    params = strategy_params[strategy_name]

    print(f"\n## 🔧 Backtest Configuration")
    print(f"| Parameter | Value |")
    print(f"|-----------|-------|")
    print(f"| Symbol | {symbol} |")
    print(f"| Market | {marketType} |")
    print(f"| Interval | {interval} |")
    print(f"| Day Range | {day_range} |")
    print(f"| Strategy | {strategy_name.replace('_', ' ').title()} |")
    print(f"| Initial Cash | ${initial_cash:,.0f} |")
    print(f"| Commission | {commission*100:.1f}% |")

    print(f"\n**Strategy Parameters:**")
    for param, value in params.items():
        if isinstance(value, float) and 0 < value < 1:
            print(f"| {param.replace('_', ' ').title()} | {value*100:.1f}% |")
        else:
            print(f"| {param.replace('_', ' ').title()} | {value} |")

    # Setup cerebro
    cerebro = bt_utils.setup_cerebro(initial_cash=initial_cash, commission=commission)
    cerebro = bt_utils.add_standard_analyzers(cerebro, risk_free_rate=risk_free_rate)

    # Add strategy with parameters
    cerebro.addstrategy(strategy_class, **params)

    # Add data
    data = bt_utils.create_backtrader_data(df)
    cerebro.adddata(data)

    # Run backtest
    print(f"\n🔄 **Running backtest...**")
    results = cerebro.run()
    print(f"✅ **Backtest completed!**")

    # Extract results
    strategy = results[0]
    strategy_results = bt_utils.extract_strategy_results(strategy, initial_cash)

    # Display results
    print(f"\n## 📊 BACKTEST RESULTS")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| **Final Value** | **${strategy_results['final_value']:,.0f}** |")
    print(f"| **Total Return** | **{strategy_results['total_return']:.2f}%** |")
    print(f"| **Sharpe Ratio** | **{strategy_results['sharpe_ratio']:.3f}** |")
    print(f"| **Max Drawdown** | **{strategy_results['max_drawdown']:.2f}%** |")
    print(f"| **Total Trades** | **{strategy_results['total_trades']}** |")
    print(f"| **Win Rate** | **{strategy_results['win_rate']:.1f}%** |")
    print(f"| **Profit Factor** | **{strategy_results.get('profit_factor', 'N/A')}** |")
    print(f"| **Avg Trade** | **{strategy_results.get('avg_trade_pct', 0):.2f}%** |")

    # Performance assessment
    print(f"\n## 🎯 PERFORMANCE ASSESSMENT")
    if strategy_results['total_return'] > 0:
        print(f"✅ **Profitable Strategy** - Generated {strategy_results['total_return']:.2f}% return")
    else:
        print(f"❌ **Unprofitable Strategy** - Lost {abs(strategy_results['total_return']):.2f}%")

    if strategy_results['sharpe_ratio'] > 1.0:
        print(f"🏆 **Excellent Risk-Adjusted Performance** - Sharpe ratio of {strategy_results['sharpe_ratio']:.3f}")
    elif strategy_results['sharpe_ratio'] > 0.5:
        print(f"👍 **Good Risk-Adjusted Performance** - Sharpe ratio of {strategy_results['sharpe_ratio']:.3f}")
    else:
        print(f"⚠️ **Poor Risk-Adjusted Performance** - Sharpe ratio of {strategy_results['sharpe_ratio']:.3f}")

    if strategy_results['max_drawdown'] < 10:
        print(f"💪 **Low Risk** - Maximum drawdown of {strategy_results['max_drawdown']:.2f}%")
    elif strategy_results['max_drawdown'] < 20:
        print(f"⚖️ **Moderate Risk** - Maximum drawdown of {strategy_results['max_drawdown']:.2f}%")
    else:
        print(f"⚠️ **High Risk** - Maximum drawdown of {strategy_results['max_drawdown']:.2f}%")

    print(f"\n✅ **BACKTEST COMPLETE**")
    print(f"*Strategy ready for analysis and potential deployment*")
    cerebro.plot()
    return strategy_results



if __name__ == "__main__":
    results = run_single_backtest()