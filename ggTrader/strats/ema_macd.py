import backtrader as bt

class EMAMACDStrategy(bt.Strategy):
    """Dual EMA + MACD strategy with optimizable parameters and optional logging"""

    params = (
        ('ema_fast', 12),             # Fast EMA period
        ('ema_slow', 26),             # Slow EMA period
        ('macd_fast', 12),            # MACD fast period
        ('macd_slow', 26),            # MACD slow period
        ('macd_signal', 9),           # MACD signal period
        ('position_pct', 0.95),       # Position size percentage
        ('stop_loss_pct', 0.05),      # Stop loss percentage
        ('log_enabled', False),       # Enable logging of trades
    )

    def __init__(self):
        self.log_enabled = self.p.log_enabled

        # Indicators
        self.ema_fast = bt.ind.EMA(self.data.close, period=self.p.ema_fast)
        self.ema_slow = bt.ind.EMA(self.data.close, period=self.p.ema_slow)
        self.ema_crossover = bt.ind.CrossOver(self.ema_fast, self.ema_slow)

        self.macd = bt.ind.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
        )
        self.macd_crossover = bt.ind.CrossOver(self.macd.macd, self.macd.signal)

        self.entry_price = None

    def notify_order(self, order):
        if order.status in [order.Completed] and self.log_enabled:
            action = "BUY" if order.isbuy() else "SELL"
            print(f"[{self.data.datetime.date(0)}] {action} @ {order.executed.price:.2f}")

    def notify_trade(self, trade):
        if trade.isclosed and self.log_enabled:
            print(f"[{self.data.datetime.date(0)}] Trade PnL: {trade.pnl:.2f}")

    def next(self):
        current_price = self.data.close[0]

        # Entry logic
        if not self.position:
            if (
                    self.ema_crossover > 0 and
                    self.macd_crossover > 0 and
                    self.ema_fast[0] > self.ema_slow[0] and
                    self.macd.macd[0] > self.macd.signal[0]
            ):
                cash = self.broker.getcash()
                size = (cash * self.p.position_pct) / current_price
                if size > 0:
                    self.buy(size=size)
                    self.entry_price = current_price

        # Exit logic
        else:
            stop_price = self.entry_price * (1 - self.p.stop_loss_pct)
            if self.entry_price and current_price <= stop_price:
                self.close()
                self.entry_price = None
            elif self.ema_crossover < 0 or self.macd_crossover < 0:
                self.close()
                self.entry_price = None
