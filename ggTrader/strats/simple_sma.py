import backtrader as bt

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