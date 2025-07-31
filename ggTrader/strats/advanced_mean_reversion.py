import backtrader as bt
import numpy as np

class AdvancedMeanReversionStrategy(bt.Strategy):
    """Advanced mean reversion strategy with multiple indicators"""
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('bb_period', 20),
        ('bb_std', 2.0),
        ('volume_ma_period', 20),
        ('volume_threshold', 1.5),
        ('position_pct', 0.95),
        ('stop_loss_pct', 0.05),
        ('take_profit_pct', 0.10),
    )

    def __init__(self):
        # RSI indicator
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.data.close,
            period=self.params.rsi_period
        )

        # Bollinger Bands
        self.bb = bt.indicators.BollingerBands(
            self.data.close,
            period=self.params.bb_period,
            devfactor=self.params.bb_std
        )

        # Volume indicators
        self.volume_ma = bt.indicators.SimpleMovingAverage(
            self.data.volume,
            period=self.params.volume_ma_period
        )

        # Price tracking for stop loss/take profit
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None

    def next(self):
        current_price = self.data.close[0]
        current_volume = self.data.volume[0]

        # Volume filter - check if volume_ma has enough data
        if len(self.volume_ma) == 0:
            return

        volume_condition = current_volume > (self.volume_ma[0] * self.params.volume_threshold)

        if not self.position:
            # Long entry conditions (oversold + below lower BB + high volume)
            oversold_condition = self.rsi[0] < self.params.rsi_oversold
            bb_lower_condition = current_price < self.bb.lines.bot[0]

            if oversold_condition and bb_lower_condition and volume_condition:
                cash = self.broker.getcash()
                size = (cash * self.params.position_pct) / current_price
                if size > 0:
                    self.buy(size=size)
                    self.entry_price = current_price
                    self.stop_loss_price = current_price * (1 - self.params.stop_loss_pct)
                    self.take_profit_price = current_price * (1 + self.params.take_profit_pct)

        else:
            # Exit conditions - only check if we have valid stop/profit prices
            overbought_condition = self.rsi[0] > self.params.rsi_overbought
            bb_upper_condition = current_price > self.bb.lines.top[0]

            # Safe comparisons with None checks
            stop_loss_hit = (self.stop_loss_price is not None and
                             current_price <= self.stop_loss_price)
            take_profit_hit = (self.take_profit_price is not None and
                               current_price >= self.take_profit_price)

            if (overbought_condition or bb_upper_condition or
                    stop_loss_hit or take_profit_hit):
                self.close()
                self.entry_price = None
                self.stop_loss_price = None
                self.take_profit_price = None