import sys, time, numpy as np, pandas as pd
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategies import indicators as ind
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.data import STOCK_BASE_CONFIG
import vectorbt as vbt
out = sys.argv[1]
c = pd.read_parquet("/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S5-evidence/tape_close.parquet"); v = pd.read_parquet("/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/vbt-spec/S5-evidence/tape_volume.parquet")
rng = np.random.default_rng(5); cols = sorted(rng.choice(c.columns, 60, replace=False))
c = c[cols].loc["2021-01-01":"2023-06-30"]; v = v[cols].loc[c.index]
data = pd.concat({s: pd.concat({"close": c[s], "volume": v[s]}, axis=1) for s in cols}, axis=1)
strat = EnsembleSignal(LabConfig(min_history_bars=60))
grid = build_grid(EnsembleSignal)
t = time.perf_counter(); sig = strat.sweep_signals(grid, cols, data); t1 = time.perf_counter() - t
close = ind.extract_close(data, cols)
cfg = dict(STOCK_BASE_CONFIG); cfg["FREQ"] = "1D"
r1, val1, d1 = simulate_signals(sig, close, cfg)
cfg2 = dict(cfg); cfg2["ts_stop"] = 0.10
r2, val2, d2 = simulate_signals(sig, close, cfg2)
pd.concat({"plain": val1, "trail": val2}, axis=1).to_parquet(out)
# fragmentation micro: rsi chain on extract_close output vs consolidated copy
frag = ind.extract_close(data, cols); cons = frag.copy()
def tm(f, n=15):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts) * 1e3
a = tm(lambda: ind.rsi_signals(frag, 14, 30, 50)); b = tm(lambda: ind.rsi_signals(cons, 14, 30, 50))
print(pd.__version__, vbt.__version__, vbt.settings.get("engine") if hasattr(vbt.settings, "get") else "-", "combos", len(sig), "sweep_signals s", round(t1, 2), "| rsi chain ms frag/consolidated", round(a, 1), round(b, 1), "| nblocks", frag._mgr.nblocks, cons._mgr.nblocks, "| trades", sum(x.get("n_trades", 0) for x in d1.values()))
