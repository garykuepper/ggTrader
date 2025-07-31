import backtrader as bt

class MultiTimeFrameStrategy(bt.Strategy):
    """Multi-timeframe strategy using higher timeframe for trend"""
    params = (
        ('fast_ma', 10),
        ('slow_ma', 30),
        ('trend_ma', 50),      # Higher timeframe trend MA
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('rsi_overbought', 70),
        ('position_pct', 0.95),
        ('volume_threshold', 1.2),
    )

    def __init__(self):
        # Primary timeframe indicators (trading timeframe)
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.fast_ma
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0].close, period=self.params.slow_ma
        )
        self.rsi = bt.indicators.RelativeStrengthIndex(
            self.datas[0].close, period=self.params.rsi_period
        )

        # Volume analysis
        self.volume_ma = bt.indicators.SimpleMovingAverage(
            self.datas[0].volume, period=20
        )

        # Higher timeframe trend (if available)
        if len(self.datas) > 1:
            self.trend_ma = bt.indicators.SimpleMovingAverage(
                self.datas[1].close, period=self.params.trend_ma
            )
        else:
            # Fallback to longer MA on same timeframe
            self.trend_ma = bt.indicators.SimpleMovingAverage(
                self.datas[0].close, period=self.params.trend_ma * 2
            )

        # Crossover signal
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        current_price = self.datas[0].close[0]
        current_volume = self.datas[0].volume[0]

        # Higher timeframe trend direction
        if len(self.datas) > 1:
            htf_price = self.datas[1].close[0]
            bullish_trend = htf_price > self.trend_ma[0]
        else:
            bullish_trend = current_price > self.trend_ma[0]

        # Volume condition
        volume_condition = current_volume > (self.volume_ma[0] * self.params.volume_threshold)

        if not self.position:
            # Long entry conditions
            ma_crossover = self.crossover > 0
            rsi_oversold = self.rsi[0] < self.params.rsi_oversold

            if bullish_trend and ma_crossover and rsi_oversold and volume_condition:
                cash = self.broker.getcash()
                size = (cash * self.params.position_pct) / current_price
                if size > 0:
                    self.buy(size=size)

        else:
            # Exit conditions
            ma_crossover_down = self.crossover < 0
            rsi_overbought = self.rsi[0] > self.params.rsi_overbought
            trend_reversal = not bullish_trend

            if ma_crossover_down or rsi_overbought or trend_reversal:
                self.close()