import backtrader as bt

class MomentumBreakoutStrategy(bt.Strategy):
    """Advanced momentum breakout strategy"""
    params = (
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
        ('ema_fast', 12),
        ('ema_slow', 26),
        ('volume_threshold', 1.2),
        ('breakout_lookback', 20),
        ('position_pct', 0.95),
        ('risk_per_trade', 0.02),  # 2% risk per trade
    )

    def __init__(self):
        # EMAs for trend direction
        self.ema_fast = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.params.ema_fast
        )
        self.ema_slow = bt.indicators.ExponentialMovingAverage(
            self.data.close, period=self.params.ema_slow
        )

        # ATR for volatility-based stops
        self.atr = bt.indicators.AverageTrueRange(period=self.params.atr_period)

        # Highest/Lowest for breakout detection
        self.highest = bt.indicators.Highest(
            self.data.high, period=self.params.breakout_lookback
        )
        self.lowest = bt.indicators.Lowest(
            self.data.low, period=self.params.breakout_lookback
        )

        # Volume
        self.volume_ma = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=20
        )

        self.entry_price = None
        self.stop_loss_price = None

    def next(self):
        current_price = self.data.close[0]
        current_volume = self.data.volume[0]

        if not self.position:
            # Trend condition (EMAs aligned)
            bullish_trend = self.ema_fast[0] > self.ema_slow[0]

            # Breakout condition
            breakout_high = current_price > self.highest[-1]  # Yesterday's highest

            # Volume confirmation
            volume_spike = current_volume > (self.volume_ma[0] * self.params.volume_threshold)

            if bullish_trend and breakout_high and volume_spike:
                # Position sizing based on ATR
                atr_stop_distance = self.atr[0] * self.params.atr_multiplier
                account_value = self.broker.getvalue()
                risk_amount = account_value * self.params.risk_per_trade

                if atr_stop_distance > 0:
                    position_size = risk_amount / atr_stop_distance
                    max_position = (account_value * self.params.position_pct) / current_price

                    size = min(position_size, max_position)

                    if size > 0:
                        self.buy(size=size)
                        self.entry_price = current_price
                        self.stop_loss_price = current_price - atr_stop_distance

        else:
            # ATR-based trailing stop
            trailing_stop = current_price - (self.atr[0] * self.params.atr_multiplier)

            if self.stop_loss_price is None or trailing_stop > self.stop_loss_price:
                self.stop_loss_price = trailing_stop

            # Exit on stop loss
            if current_price <= self.stop_loss_price:
                self.close()
                self.entry_price = None
                self.stop_loss_price = None