import sys, cProfile, pstats, io, numpy as np, pandas as pd, vectorbt as vbt
sys.path.insert(0, __import__("os").path.dirname(__file__))
from common import make_prices
kw = {} if sys.argv[1] == "none" else {"engine": sys.argv[1]}
px1 = make_prices(280, 150, seed=3); rng = np.random.default_rng(5)
cols = pd.MultiIndex.from_product([["a", "b", "c"], range(50)], names=["strategy", "symbol"])
close = px1.set_axis(cols, axis=1); grp = pd.Index(cols.get_level_values(0), name="strategy")
w = pd.DataFrame(np.nan, index=px1.index, columns=cols)
for r in range(0, 280, 21): w.iloc[r] = rng.dirichlet(np.ones(150)) * 0.9
def f():
    pf = vbt.Portfolio.from_orders(close, w, size_type="targetpercent", init_cash=1e5, fees=5e-4, slippage=5e-4, freq="1D", cash_sharing=True, group_by=grp, call_seq="auto", **kw).copy()
    return pf.value(), pf.trades.count()
for _ in range(5): f()
pr = cProfile.Profile(); pr.enable()
for _ in range(60): f()
pr.disable(); s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(4); 
print(vbt.__version__, sys.argv[1]); print("\n".join(l for l in s.getvalue().splitlines() if l.strip()))
