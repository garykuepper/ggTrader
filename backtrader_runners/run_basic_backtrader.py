import os
import backtrader as bt
import pandas as pd
from dotenv import load_dotenv
from ggTrader.data_manager.universal_data_manager import UniversalDataManager

load_dotenv()
mongo_uri = os.getenv('MONGO_URI', "mongodb://localhost:27017/")

class SimpleSMAStrategy(bt.Strategy):
    """Simple SMA crossover strategy with position sizing"""
    params = (
        ('sma_fast', 10),
        ('sma_slow', 30),
        ('position_pct', 0.95),  # Use 95% of available cash
    )

    def __init__(self):
        # Create SMA indicators
        self.sma_fast = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_fast)
        self.sma_slow = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_slow)

        # Create crossover signal
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)

    def next(self):
        if not self.position:  # Not in market
            if self.crossover > 0:  # Fast SMA crosses above slow SMA
                # Calculate how much we can afford
                cash = self.broker.getcash()
                price = self.data.close[0]
                size = (cash * self.params.position_pct) / price

                if size > 0:
                    self.buy(size=size)
                    print(f"BUY order: {size:.6f} BTC at {self.data.datetime.date(0)}, Price: ${price:.2f}")
        else:  # In market
            if self.crossover < 0:  # Fast SMA crosses below slow SMA
                self.close()
                print(f"SELL order at {self.data.datetime.date(0)}, Price: ${self.data.close[0]:.2f}")

    def notify_order(self, order):
        """Track order execution"""
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"BUY EXECUTED - Size: {order.executed.size:.6f}, Price: ${order.executed.price:.2f}")
            else:
                print(f"SELL EXECUTED - Size: {order.executed.size:.6f}, Price: ${order.executed.price:.2f}")

    def notify_trade(self, trade):
        """Track completed trades"""
        if trade.isclosed:
            print(f"TRADE CLOSED - P&L: ${trade.pnl:.2f}, Commission: ${trade.commission:.2f}")
def create_backtrader_data(df):
    """Convert DataFrame to backtrader data format"""
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        # Assuming index is already datetime, but make sure
        df.index = pd.to_datetime(df.index)

    # Rename columns to match backtrader expectations
    bt_df = df.copy()
    bt_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Create backtrader data feed
    data = bt.feeds.PandasData(dataname=bt_df)
    return data

def run_backtest():
    """Run the backtest"""
    print("Starting Backtest with Universal Data Manager...")

    # Initialize data manager
    dm = UniversalDataManager(mongo_uri=mongo_uri)

    # Fetch data
    print("Fetching data...")
    df = dm.load_or_fetch("BTCUSDT", "1h", "2025-07-20", "2025-07-29", market="crypto")

    if df.empty:
        print("❌ No data available for backtest")
        return

    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    # Create backtrader cerebro engine
    cerebro = bt.Cerebro()

    # Add strategy
    cerebro.addstrategy(SimpleSMAStrategy)

    # Convert and add data
    data = create_backtrader_data(df)
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

    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)

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