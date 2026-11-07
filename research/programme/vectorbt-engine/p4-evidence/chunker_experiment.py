import gc, resource, sys, time, os, inspect, textwrap, ctypes
import numpy as np, pandas as pd
from ggTrader.data.bars import load_bars
from ggTrader.engine import portfolio as P
from ggTrader.lab.data import STOCK_BASE_CONFIG
from ggTrader.lab.strategies.v2.contract import tile_columns
from ggTrader.lab.strategies.v2.ensemble import Ensemble
from ggTrader.lab.universe import pit_mask
def cur(): return int(open('/proc/self/statm').read().split()[1])*4096/1e6
def peak(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
bars = load_bars(universe="sp500", universe_span=("2021-01-31","2026-04-30"), extra_symbols=["SPY"], start="2019-03-01", end="2026-10-30")
out = Ensemble().build(bars); mask = pit_mask(bars.close, min_history_bars=400, on_stale="ignore")
cols = out.entries.columns
close_t = tile_columns(bars.close, cols); mask_t = tile_columns(mask, cols)
if sys.argv[1] == "fix":
    src = inspect.getsource(P._run_chunks)
    assert "            trade_frames.append(records)\n" in src
    src = src.replace("            trade_frames.append(records)\n", "            trade_frames.append(records)\n        del pf, value\n        gc.collect()\n        ctypes.CDLL('libc.so.6').malloc_trim(0)\n")
    ns = dict(P.__dict__); ns["gc"]=gc; ns["ctypes"]=ctypes
    exec(src, ns); P._run_chunks = ns["_run_chunks"]
ref = None
res = P.run_signals(out.entries, out.exits, close_t, dict(STOCK_BASE_CONFIG), groups=out.group_labels(), mask=mask_t, max_bytes=int(float(sys.argv[2])*1024**3))
print(sys.argv[1], "chunks", res.n_chunks, "peak", peak())
np.save(f"/tmp/claude-1000/-home-flynn-ggTrader/6c9c1710-29af-4adf-b77b-ad700b6fcc23/scratchpad/eq_{sys.argv[1]}_{sys.argv[2]}.npy", res.value.to_numpy())
