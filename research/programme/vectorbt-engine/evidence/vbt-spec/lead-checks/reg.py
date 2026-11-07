import collections, inspect, numpy as np, pandas as pd
from ggTrader.lab.strategies import STRATEGY_REGISTRY as R
print("n", len(R))
kinds = collections.Counter(getattr(c, "target_kind", "?") for c in R.values())
print(kinds)
for k, c in R.items():
    f = inspect.getsourcefile(c).split("strategies/")[-1]
    ln = inspect.getsourcelines(c)[1]
    print(k, c.__name__, f, ln, c.target_kind)
from ggTrader.lab.strategies.indicators import bb_signals, volume_bb_signals
rng = np.random.default_rng(1)
close = pd.DataFrame(100*np.exp(np.cumsum(rng.normal(0,0.02,(3000,40)),0)))
vol = pd.DataFrame(rng.lognormal(10,0.6,(3000,40)))
be, bx = bb_signals(close, 20, 2.0)
ve, vx = volume_bb_signals(close, vol, 20, 2.0, 20, 2.0)
print("vbb entries not in bb:", int((ve & ~be).values.sum()), "vbb entries", int(ve.values.sum()), "bb", int(be.values.sum()))
print("exit frames identical:", bool((vx == bx).all().all()))
