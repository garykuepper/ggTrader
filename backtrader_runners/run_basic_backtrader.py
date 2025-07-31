import os
import backtrader as bt
import pandas as pd
from dotenv import load_dotenv
from ggTrader.data_manager.universal_data_manager import UniversalDataManager
from ggTrader.strats.simple_sma import SimpleSMAStrategy
from ggTrader.utils.backtrader_utils import BacktraderUtils
load_dotenv()
mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")

symbol= "SPY"
market= "stock"
interval = "1d"
date_range = 365*5
# Configure optimization parameters
fast_range = 5
slow_range = 25
position_range = .95


def run_backtest():
    """Run the backtest"""
    print("Starting Backtest with Universal Data Manager...")

    # Initialize data manager
    dm = UniversalDataManager(mongo_uri=mongo_uri)
    bt_utils = BacktraderUtils()
    # Fetch data
    print("Fetching data...")
    df = dm.load_or_fetch("BTCUSDT", "1h", "2025-07-20", "2025-07-29", market="crypto")


    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    # Create backtrader cerebro engine
    cerebro = bt.Cerebro()

    # Add strategy
    cerebro.addstrategy(SimpleSMAStrategy,
                        sma_fast=fast_range,
                        sma_slow=slow_range,
                        position_pct=position_range)

    # Convert and add data
    data = bt_utils.create_backtrader_data(df)
    cerebro.adddata(data)


    # Set initial cash
    cerebro.broker.setcash(10000.0)

    # Set commission (0.1%)
    cerebro.broker.setcommission(commission=0.001)
    # Allow fractional shares
    cerebro.broker.set_filler(bt.broker.fillers.FixedBarPerc(perc=100.0))
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Run backtest
    results = cerebro.run()

    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")

    # Print results
    strat = results[0]

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)

    # Sharpe Ratio
    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f"Sharpe Ratio: {sharpe.get('sharperatio', 'N/A')}")

    # Drawdown
    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f"Max Drawdown: {drawdown.get('max', {}).get('drawdown', 'N/A'):.2f}%")

    # Returns
    returns = strat.analyzers.returns.get_analysis()
    print(f"Total Return: {returns.get('rtot', 'N/A'):.4f}")

    # Trade Analysis
    trades = strat.analyzers.trades.get_analysis()
    total_trades = trades.get('total', {}).get('total', 0)
    won_trades = trades.get('won', {}).get('total', 0)
    lost_trades = trades.get('lost', {}).get('total', 0)

    print(f"Total Trades: {total_trades}")
    print(f"Won Trades: {won_trades}")
    print(f"Lost Trades: {lost_trades}")
    if total_trades > 0:
        win_rate = (won_trades / total_trades) * 100
        print(f"Win Rate: {win_rate:.1f}%")

    # Plot results (optional)
    try:
        print("\nGenerating plot...")
        cerebro.plot(style='candlestick', volume=False, figsize=(15, 8))
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    run_backtest()
