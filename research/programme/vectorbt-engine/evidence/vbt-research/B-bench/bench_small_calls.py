"""WFO-fold-sized from_orders / from_signals calls (many small calls): per-call wall time, incl. pf.value() + trades.count()."""
import sys, time, json, numpy as np, pandas as pd, vectorbt as vbt
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import make_prices
eng = sys.argv[1]; kw = {} if eng == "none" else {"engine": eng}
N = 150
px1 = make_prices(280, 150, seed=3); rng = np.random.default_rng(5)
cols = pd.MultiIndex.from_product([["a", "b", "c"], range(50)], names=["strategy", "symbol"])
close = px1.set_axis(cols, axis=1)
grp = pd.Index(cols.get_level_values(0), name="strategy")
w = pd.DataFrame(np.nan, index=px1.index, columns=cols)
for r in range(0, 280, 21): w.iloc[r] = rng.dirichlet(np.ones(150)) * 3 * 0.3
ent = pd.DataFrame(rng.random(close.shape) < .02, index=px1.index, columns=cols); ex = pd.DataFrame(rng.random(close.shape) < .03, index=px1.index, columns=cols)
def orders(): 
    pf = vbt.Portfolio.from_orders(close, w, size_type="targetpercent", init_cash=1e5, fees=5e-4, slippage=5e-4, freq="1D", cash_sharing=True, group_by=grp, call_seq="auto", **kw).copy()
    return pf.value(), pf.trades.count()
def sigs():
    pf = vbt.Portfolio.from_signals(close, ent, ex, size=.03, size_type="percent", init_cash=1e5, fees=5e-4, slippage=5e-4, freq="1D", cash_sharing=True, group_by=grp, sl_stop=.08, sl_trail=True, **kw).copy()
    return pf.value(), pf.trades.count()
res = {"vbt": vbt.__version__, "engine": eng}
for name, fn in [("from_orders targetpercent auto, 280d x 150cols, 3 groups", orders), ("from_signals trailing SL, 280d x 150cols, 3 groups", sigs)]:
    t0 = time.perf_counter(); v1 = fn(); first = time.perf_counter() - t0
    ts = []
    for _ in range(N):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    res[name] = {"first_call_s": first, "median_ms": 1e3 * float(np.median(ts)), "min_ms": 1e3 * min(ts), "p90_ms": 1e3 * float(np.percentile(ts, 90)), "n": N}
    res[name]["checksum"] = float(np.nansum(v1[0].values[-1]))
print(json.dumps(res))
