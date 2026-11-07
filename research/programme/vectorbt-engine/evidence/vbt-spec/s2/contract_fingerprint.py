"""Prototype: emit numeric fingerprints of vbt definitions that ggTrader would rely on."""
import json, sys, warnings
import numpy as np, pandas as pd
import vectorbt as vbt
warnings.simplefilter("ignore")
rng = np.random.default_rng(7)
idx = pd.bdate_range("2020-01-01", periods=400)
close = pd.Series(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.01, 400))), idx, name="x")
high, low = close * 1.004, close * 0.996
ret = close.pct_change(fill_method=None).dropna()
out = {}
def fp(name, s):
    s = pd.Series(s).dropna() if not np.isscalar(s) else pd.Series([s])
    out[name] = dict(n=int(s.shape[0]), sum=round(float(s.sum()), 9), last=round(float(s.iloc[-1]), 9))
def first_valid(s): return int(s.reset_index(drop=True).first_valid_index())
fp("rsi14_default", vbt.RSI.run(close, 14).rsi); out["rsi14_default"]["first"] = first_valid(vbt.RSI.run(close, 14).rsi)
r = vbt.RSI.run(close, 27, ewm=True).rsi; fp("rsi_wilder27", r); out["rsi_wilder27"]["first"] = first_valid(r)
b = vbt.BBANDS.run(close, 20, alpha=2); fp("bb_default_lower", b.lower)
try:
    b1 = vbt.BBANDS.run(close, 20, alpha=2, ddof=1); fp("bb_ddof1_lower", b1.lower)
except TypeError as e: out["bb_ddof1_lower"] = f"TypeError {e}"
m = vbt.MACD.run(close, 12, 26, 9, macd_ewm=True, signal_ewm=True); fp("macd_ewm_hist", m.hist)
a = vbt.ATR.run(high, low, close, 14, ewm=False).atr; fp("atr14_sma", a); out["atr14_sma"]["first"] = first_valid(a)
fp("ma20", vbt.MA.run(close, 20).ma)
rr = ret.vbt.returns(freq="D")
for k, f in dict(sharpe_default=lambda: rr.sharpe_ratio(), sortino=lambda: rr.sortino_ratio(),
                 ann_ret=lambda: rr.annualized(), maxdd=lambda: rr.max_drawdown(),
                 ann_vol=lambda: rr.annualized_volatility(), calmar=lambda: rr.calmar_ratio(),
                 dsr=lambda: rr.deflated_sharpe_ratio(nb_trials=10, var_sharpe=0.1)).items():
    try: out[k] = round(float(f()), 9)
    except Exception as e: out[k] = f"ERR {type(e).__name__}"
rr2 = ret.vbt.returns(freq="D", year_freq="252 days"); out["sharpe_252"] = round(float(rr2.sharpe_ratio()), 9)
out["_versions"] = dict(vbt=vbt.__version__, pandas=pd.__version__, numpy=np.__version__)
print(json.dumps(out, indent=1, sort_keys=True))
