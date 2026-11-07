"""How many cells would a pandas-3 bare pct_change() change on REAL cached SP500 closes? READ-ONLY.
old behaviour == ffill-then-pct (pad, limit None); new bare behaviour == no padding."""
import sys
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
import psycopg2  # noqa: E402

from ggTrader.data.live import cached_yfinance_loader as cyl  # noqa: E402
from ggTrader.lab.data import equity_universe_between  # noqa: E402
from ggTrader.utils.config import get_db_connection_string  # noqa: E402

_oc = cyl.CachedYFinanceLoader._connect
cyl.CachedYFinanceLoader._connect = lambda self: (lambda c: (c.set_session(readonly=True), c)[1])(_oc(self))
out = {}
for label, (s, e) in {"2021-2022": ("2021-01-01", "2022-07-31"), "2025-2026": ("2025-01-02", "2026-10-30")}.items():
    start, end = pd.Timestamp(s, tz="UTC"), pd.Timestamp(e, tz="UTC")
    uni = equity_universe_between(start, end, universe="sp500")
    wide = cyl.CachedYFinanceLoader()._db_fetch(uni, "1d", start, end)
    close = pd.concat({sym: wide[sym]["close"] for sym in wide.columns.get_level_values(0).unique()}, axis=1)
    padded = close.ffill().pct_change(fill_method=None)  # == pandas<3 bare pct_change()
    bare = close.pct_change(fill_method=None)  # == pandas>=3 bare pct_change()
    both_nan = padded.isna() & bare.isna()
    differ = ~((padded == bare) | both_nan)
    interior = close.notna() & close.shift(1).isna() & close.notna().cumsum().gt(1)  # re-appearing after a gap
    out[label] = {
        "pandas": pd.__version__,
        "symbols": close.shape[1],
        "rows": close.shape[0],
        "nan_cells": int(close.isna().sum().sum()),
        "cells_where_pad_vs_none_differ": int(differ.sum().sum()),
        "symbols_affected": int((differ.sum() > 0).sum()),
        "max_abs_diff_ignoring_nan": float(np.nanmax(np.abs((padded - bare).to_numpy()))) if differ.any().any() else 0.0,
    }
    print(label, out[label])
import json  # noqa: E402

json.dump(out, open(sys.argv[1], "w"), indent=1)
