import os
import backtrader as bt
from dotenv import load_dotenv
from old.ggTrader_old.data_manager import UniversalDataManager
from old.ggTrader_old.utils.backtrader_utils import BacktraderUtils
from old.ggTrader_old.strats.ema_macd import EMAMACDStrategy

load_dotenv()
mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")

# =============================================================================
# CONFIGURATION - Change strategy and parameters here
# =============================================================================
# STRATEGY = SimpleSMAStrategy  # Change this to your desired strategy class
#
# STRATEGY_PARAMS = {
#     'sma_fast': 5,
#     'sma_slow': 25,
#     'position_pct': 0.95
# }
STRATEGY = EMAMACDStrategy  # Change this to your desired strategy class

STRATEGY_PARAMS = {
    'ema_fast': 12,         # Fast EMA period
    'ema_slow': 26,         # Slow EMA period
    'macd_fast': 12,       # MACD fast EMA period
    'macd_slow': 26,       # MACD slow EMA period
    'macd_signal': 9,      # MACD signal line EMA period
    'position_pct': 0.95   # Position sizing percentage
}
# Data configuration
SYMBOL = "BTCUSDT"
MARKET = "crypto"
INTERVAL = "1h"
START_DATE = "2025-07-20"
END_DATE = "2025-07-29"

# Broker configuration
INITIAL_CASH = 10000.0
COMMISSION = 0.001
# =============================================================================


def run_backtest():
    """Run the backtest with configurable strategy"""
    print(f"Starting Backtest with {STRATEGY.__name__}...")

    # Initialize data manager
    dm = UniversalDataManager(mongo_uri=mongo_uri)
    bt_utils = BacktraderUtils()

    # Fetch data
    print("Fetching data...")
    df = dm.load_or_fetch(SYMBOL, INTERVAL, START_DATE, END_DATE, market=MARKET)

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Create backtrader cerebro engine
    cerebro = bt.Cerebro()

    # Add strategy with parameters
    cerebro.addstrategy(STRATEGY, **STRATEGY_PARAMS)

    # Convert and add data
    data = bt_utils.create_backtrader_data(df)
    cerebro.adddata(data)

    # Set broker configuration
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
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

    # Plot results
    try:
        print("\nGenerating plot...")
        cerebro.plot(style='candlestick', volume=False, figsize=(15, 8))
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    run_backtest()