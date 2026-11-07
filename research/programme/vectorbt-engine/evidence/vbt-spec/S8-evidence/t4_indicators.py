import numpy as np, pandas as pd, vectorbt as vbt
from t_common import *
o,h,l,c,v = make_data(n=800, k=6)
import talib; print("talib", talib.__version__, "funcs", len(talib.get_functions()))
run("talib.RSI numpy", lambda: talib.RSI(c.iloc[:,0].values, 14)[-1])
run("vbt.IndicatorFactory.from_talib('RSI')", lambda: vbt.IndicatorFactory.from_talib('RSI').run(c, timeperiod=[10,14,20]).real.shape)
run("vbt.IndicatorFactory.from_talib('BBANDS')", lambda: vbt.IndicatorFactory.from_talib('BBANDS').run(c, timeperiod=[20], nbdevup=[2.0], nbdevdn=[2.0]).lowerband.shape)
run("vbt.IndicatorFactory.from_talib('ATR') OHLC", lambda: vbt.IndicatorFactory.from_talib('ATR').run(h,l,c, timeperiod=[14]).real.shape)
run("vbt.talib('MACD')", lambda: vbt.talib('MACD').run(c, fastperiod=12, slowperiod=26, signalperiod=9).macd.shape)
import ta
run("ta lib RSIIndicator", lambda: ta.momentum.RSIIndicator(c.iloc[:,0], 14).rsi().iloc[-1])
run("vbt.IndicatorFactory.from_ta('RSIIndicator')", lambda: vbt.IndicatorFactory.from_ta('RSIIndicator').run(c, window=[10,14]).rsi.shape)
import pandas_ta_classic as pta
run("pandas_ta_classic rsi", lambda: pta.rsi(c.iloc[:,0], 14).iloc[-1])
run("vbt.IndicatorFactory.from_pandas_ta('RSI') (needs pandas_ta)", lambda: vbt.IndicatorFactory.from_pandas_ta('RSI').run(c, length=[14]).shape if hasattr(vbt.IndicatorFactory,'from_pandas_ta') else 'no from_pandas_ta')
print("IndicatorFactory attrs:", [a for a in dir(vbt.IndicatorFactory) if a.startswith('from_')])
run("vbt.RSI.run default", lambda: vbt.RSI.run(c, window=14).rsi.shape)
# parity talib vs vbt RSI (wilder)
tal = pd.DataFrame({k: talib.RSI(c[k].values,14) for k in c.columns}, index=c.index)
ve = vbt.RSI.run(c, window=27, ewm=True).rsi  # D-report: window=2n-1 equals Wilder
d = (tal - ve.droplevel(0, axis=1) if isinstance(ve.columns, pd.MultiIndex) else tal-ve).abs().iloc[200:]
print("max |talib RSI14 - vbt RSI(27,ewm)| after bar200:", float(np.nanmax(d.values)))
tav = ta.momentum.RSIIndicator(c.iloc[:,0],14).rsi(); print("max |talib - ta lib| after 200:", float((tav-tal.iloc[:,0]).abs().iloc[200:].max()))
pv = pta.rsi(c.iloc[:,0],14); print("max |talib - pandas_ta_classic| after 200:", float((pv-tal.iloc[:,0]).abs().iloc[200:].max()))
