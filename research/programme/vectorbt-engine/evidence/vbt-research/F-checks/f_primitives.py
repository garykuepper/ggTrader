"""F: own primitive-equality checks (independent of D's script), ggTrader functions vs vbt 0.28.5."""
import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.simulate import compute_atr_stop
from ggTrader.lab.strategies import indicators as ind

rng = np.random.default_rng(11)
T, N = 1500, 8
idx = pd.bdate_range("2019-01-01", periods=T, tz="UTC")
r = rng.normal(0.0003, 0.015, (T, N))
close = pd.DataFrame(100 * np.exp(np.cumsum(r, 0)), index=idx, columns=[f"S{i}" for i in range(N)])
close.iloc[300:305, 2] = close.iloc[299, 2]  # flat run (zero-loss edge inside RSI)
high = close * (1 + np.abs(rng.normal(0, 0.006, (T, N))))
low = close * (1 - np.abs(rng.normal(0, 0.006, (T, N))))


def cmp(name, a, b, after=0):
    a, b = a.iloc[after:], b.iloc[after:]
    both = a.notna() & b.notna()
    d = float(np.nanmax(np.abs((a - b).where(both)).values))
    wa = int(a.isna().all(axis=1).values.argmin()); wb = int(b.isna().all(axis=1).values.argmin())
    print(f"{name:42s} max|diff|={d:.3e}  first-valid bar g/v={wa}/{wb}  only-one-NaN cells={int((a.isna() ^ b.isna()).values.sum())}")


# 1. SMA
cmp("SMA20 pandas vs vbt.MA", close.rolling(20).mean(), vbt.MA.run(close, 20).ma.droplevel(0, axis=1))
# 2. Bollinger lower: ggTrader bb uses rolling std ddof=1
sma = close.rolling(20).mean(); sd = close.rolling(20).std()
lower_g = sma - 2.0 * sd
bb0 = vbt.BBANDS.run(close, 20, alpha=2.0).lower.droplevel([0, 1], axis=1)
bb1 = vbt.BBANDS.run(close, 20, alpha=2.0, ddof=1).lower.droplevel([0, 1], axis=1)
cmp("BB lower ddof=1 vs vbt default (ddof=0)", lower_g, bb0)
cmp("BB lower ddof=1 vs vbt ddof=1", lower_g, bb1)
# 3. RSI Wilder (ggTrader formula copied from indicators.rsi_signals internals)
delta = close.diff()
ag = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
al = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
rsi_g = 100 - 100 / (1 + ag / al.replace(0, np.nan))
cmp("RSI14 Wilder vs vbt.RSI(14) default", rsi_g, vbt.RSI.run(close, 14).rsi.droplevel(0, axis=1))
cmp("RSI14 Wilder vs vbt.RSI(27, ewm=True)", rsi_g, vbt.RSI.run(close, 27, ewm=True).rsi.droplevel([0, 1], axis=1))
cmp("  same, after bar 200", rsi_g, vbt.RSI.run(close, 27, ewm=True).rsi.droplevel([0, 1], axis=1), after=200)
# 4. MACD hist (pandas ewm adjust=False from bar 0)
m = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
h_g = m - m.ewm(span=9, adjust=False).mean()
mv = vbt.MACD.run(close, 12, 26, 9, macd_ewm=True, signal_ewm=True)
cmp("MACD hist vs vbt (both ewm)", h_g, mv.hist.droplevel([0, 1, 2, 3, 4], axis=1))
cmp("  same, after bar 200", h_g, mv.hist.droplevel([0, 1, 2, 3, 4], axis=1), after=200)
# 5. ATR: ggTrader stop fraction = 2*ATR/close (simple mean TR)
atr_g = compute_atr_stop(high, low, close, 14, 1.0) * close
atr_v = vbt.ATR.run(high, low, close, 14, ewm=False).atr.droplevel([0, 1], axis=1)
cmp("ATR14 SMA-of-TR vs vbt.ATR(ewm=False)", atr_g, atr_v)
atr_vd = vbt.ATR.run(high, low, close, 14).atr.droplevel(0, axis=1)
cmp("ATR14 SMA-of-TR vs vbt.ATR default (ewm)", atr_g, atr_vd)
# check ggTrader floor clip
# 6. metrics on one curve
eq = close["S0"] * 100
cs = curve_stats(eq)
ret = eq.pct_change().dropna()
ra = ret.vbt.returns(freq="D", year_freq="252 days")
ra365 = ret.vbt.returns(freq="D")
print(f"Sharpe   ggT {cs['sharpe']:.10f}  vbt252 {ra.sharpe_ratio():.10f}  vbt default {ra365.sharpe_ratio():.10f}")
print(f"Sortino  ggT {cs['sortino']:.10f}  vbt252 {ra.sortino_ratio():.10f}")
print(f"MaxDD    ggT {cs['max_drawdown_pct']:.10f}  vbt {100*ra.max_drawdown():.10f}")
print(f"CAGR     ggT {cs['cagr_pct']:.6f}  vbt252 annualized {100*ra.annualized():.6f}")
# zero-variance Sharpe
flat = pd.Series(0.0, index=idx[:50])
print("flat Sharpe ggT", curve_stats(pd.Series(100.0, index=idx[:50]))["sharpe"], " vbt", flat.vbt.returns(freq="D").sharpe_ratio())
# 7. the actual ggTrader rsi_signals entry count vs vbt-based reconstruction
e_g, _ = ind.rsi_signals(close, 14, 30, 50)
rv = vbt.RSI.run(close, 27, ewm=True).rsi.droplevel([0, 1], axis=1)
e_v = (rv < 30) & (rv.shift(1) >= 30)
print("RSI entry cells ggT", int(e_g.values.sum()), " vbt-recon", int(e_v.values.sum()), " differing cells", int((e_g ^ e_v).values.sum()),
      " differing after bar 100", int((e_g ^ e_v).iloc[100:].values.sum()))
