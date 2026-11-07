"""Controlled timing of the 48-combo signal sweep (reads a saved PIT mask; READ-ONLY DB).
Usage: sweep_timing.py <mask.parquet> <engine: auto|numba|na> <reps>   (engine 'na' for vectorbt 0.28.5)"""
import json
import sys
import time
import warnings

import pandas as pd

warnings.simplefilter("ignore")
mask_path, engine, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
import vectorbt as vbt  # noqa: E402

if engine != "na":
    vbt.settings["engine"] = engine

from ggTrader.data.live import cached_yfinance_loader as cyl  # noqa: E402
from ggTrader.lab.data import STOCK_BASE_CONFIG  # noqa: E402
from ggTrader.lab.strategies import STRATEGY_REGISTRY  # noqa: E402
from ggTrader.lab.strategy import LabConfig  # noqa: E402
from ggTrader.lab.sweep import build_grid  # noqa: E402
from ggTrader.lab.wfo import _sweep_fold  # noqa: E402

_oc = cyl.CachedYFinanceLoader._connect
cyl.CachedYFinanceLoader._connect = lambda self: (lambda c: (c.set_session(readonly=True), c)[1])(_oc(self))
mask = pd.read_parquet(mask_path)
syms = list(mask.columns)
cfg = LabConfig()
start = mask.index[0] - pd.Timedelta(days=5)
ohlcv = cyl.CachedYFinanceLoader()._db_fetch(syms, "1d", pd.Timestamp(mask.index[0]).tz_convert("UTC") - pd.Timedelta(days=1), pd.Timestamp(mask.index[-1]).tz_convert("UTC"))
ohlcv = ohlcv.reindex(mask.index)
mask = mask.reindex(index=ohlcv.index)
cls = STRATEGY_REGISTRY["ensemble"]
grid = build_grid(cls)
inst = cls(cfg)
es, ee = ohlcv.index[ohlcv.index.searchsorted(pd.Timestamp("2021-01-31", tz="UTC"))], ohlcv.index[-1]
import cProfile
import os
import pstats

ts = []
for i in range(reps):
    prof = cProfile.Profile() if (os.environ.get("PROFILE") and i == reps - 1) else None
    if prof:
        prof.enable()
    t = time.perf_counter()
    c = time.process_time()
    _sweep_fold("ensemble", inst, ohlcv, es, ee, dict(STOCK_BASE_CONFIG), grid, entry_mask=mask)
    if prof:
        prof.disable()
        prof.dump_stats(os.environ["PROFILE"])
    ts.append((round(time.perf_counter() - t, 2), round(time.process_time() - c, 2)))
print(json.dumps({"vectorbt": vbt.__version__, "pandas": pd.__version__, "engine": engine, "cols": len(syms), "rows": len(ohlcv), "wall_cpu_per_rep": ts}))
