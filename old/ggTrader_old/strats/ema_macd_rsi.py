import backtrader as bt

class EmaMacdRsiStrategy(bt.Strategy):
    params = (
        ("ema_fast", 12),
        ("ema_slow", 26),
        ("macd_fast", 12),
        ("macd_slow", 26),
        ("macd_signal", 9),
        ("rsi_period", 14),
        ("rsi_overbought", 70),
        ("rsi_oversold", 30),
    )

    def __init__(self):
        # EMA crossover
        self.ema_fast = bt.ind.EMA(period=self.p.ema_fast)
        self.ema_slow = bt.ind.EMA(period=self.p.ema_slow)
        self.crossover = bt.ind.CrossOver(self.ema_fast, self.ema_slow)

        # MACD
        self.macd = bt.ind.MACD(
            self.data,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
        )

        # RSI
        self.rsi = bt.ind.RSI(period=self.p.rsi_period)

    def next(self):
        if not self.position:
            if (
                    self.crossover > 0
                    and self.macd.macd > self.macd.signal
                    and self.rsi < self.p.rsi_overbought
            ):
                self.buy()
        else:
            if (
                    self.crossover < 0
                    or self.macd.macd < self.macd.signal
                    or self.rsi > self.p.rsi_oversold
            ):
                self.close()
