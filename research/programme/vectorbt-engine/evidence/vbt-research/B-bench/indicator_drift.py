"""Do indicator/metric outputs change between vectorbt versions? Saves arrays for cross-venv compare."""
import sys, numpy as np, pandas as pd, vectorbt as vbt, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import make_prices
px = make_prices(1900, 200)
o = {}
a = px.vbt
kw = {} if len(sys.argv) < 3 else {"engine": sys.argv[2]}
o["rolling_mean"] = a.rolling_mean(20, **kw).values
o["rolling_std1"] = a.rolling_std(20, **kw).values
o["ewm_std_adj"] = a.ewm_std(20, adjust=True, **kw).values
o["ewm_std_noadj"] = a.ewm_std(20, adjust=False, **kw).values
o["ewm_std_ddof0"] = a.ewm_std(20, adjust=True, ddof=0, **kw).values
for n, f in [("MA", lambda: vbt.MA.run(px, window=20, **kw).ma), ("MSTD", lambda: vbt.MSTD.run(px, window=20, **kw).mstd),
             ("RSI", lambda: vbt.RSI.run(px, window=14, **kw).rsi), ("BBU", lambda: vbt.BBANDS.run(px, **kw).upper),
             ("MACD", lambda: vbt.MACD.run(px, **kw).macd), ("MACDewm", lambda: vbt.MACD.run(px, macd_ewm=True, signal_ewm=True, **kw).hist)]:
    o[n] = f().values
rets = px.pct_change(fill_method=None).iloc[1:]
for m in ["sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown", "annualized_volatility", "annualized"]:
    o["ret_" + m] = np.asarray(getattr(rets.vbt.returns(freq="1D"), m)())
dsr = getattr(rets.vbt.returns(freq="1D"), "deflated_sharpe_ratio", None)
if dsr is not None:
    try: o["ret_deflated_sharpe"] = np.asarray(dsr(var_sharpe=0.1, nb_trials=10))
    except Exception as e: print("dsr err", e)
np.savez(sys.argv[1], **o)
print(vbt.__version__, "saved", len(o))
