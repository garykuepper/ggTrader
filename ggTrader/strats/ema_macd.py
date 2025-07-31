import backtrader as bt

class EMAMACDStrategy(bt.Strategy):
    """Dual EMA + MACD strategy with optimizable parameters"""
    params = (
        ('ema_fast', 12),             # Fast EMA period
        ('ema_slow', 26),             # Slow EMA period
        ('macd_fast', 12),            # MACD fast period
        ('macd_slow', 26),            # MACD slow period
        ('macd_signal', 9),           # MACD signal period
        ('position_pct', 0.95),       # Position size percentage
        ('stop_loss_pct', 0.05),      # Stop loss percentage
    )

    def __init__(self):
        # Dual EMA system
        self.ema_fast = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.params.ema_fast
        )
        self.ema_slow = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.params.ema_slow
        )

        # EMA crossover signal
        self.ema_crossover = bt.indicators.CrossOver(self.ema_fast, self.ema_slow)

        # MACD indicator
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.params.macd_fast,
            period_me2=self.params.macd_slow,
            period_signal=self.params.macd_signal
        )

        # MACD crossover signal
        self.macd_crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

        # Track entry price for stop loss
        self.entry_price = None

    def next(self):
        current_price = self.data.close[0]

        # Entry logic
        if not self.position:
            # Long entry: Both EMA bullish crossover AND MACD bullish crossover
            if (self.ema_crossover > 0 and          # Fast EMA crosses above slow EMA
                    self.macd_crossover > 0 and         # MACD crosses above signal line
                    self.ema_fast[0] > self.ema_slow[0] and  # Confirm EMA trend
                    self.macd.macd[0] > self.macd.signal[0]):  # Confirm MACD signal

                cash = self.broker.getcash()
                size = (cash * self.params.position_pct) / current_price
                if size > 0:
                    self.buy(size=size)
                    self.entry_price = current_price

        # Exit logic
        else:
            # Stop loss
            if (self.entry_price and
                    current_price <= self.entry_price * (1 - self.params.stop_loss_pct)):
                self.close()
                self.entry_price = None

            # Exit on either EMA bearish crossover OR MACD bearish crossover
            elif (self.ema_crossover < 0 or self.macd_crossover < 0):
                self.close()
                self.entry_price = None