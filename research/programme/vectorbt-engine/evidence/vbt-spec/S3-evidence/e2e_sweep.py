"""Per-combo equity comparison on real cached data (READ-ONLY): signal sweep (48 ensemble
combos) + weight path (xs_momentum grid) through lab.wfo._sweep_fold / _sweep_fold_weights.
Dumps every combo's metrics and full equity curve so two stacks can be diffed exactly.

Usage: e2e_sweep.py <outdir> [--n-symbols 100]
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n-symbols", type=int, default=100)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
warnings.simplefilter("ignore")

import psycopg2  # noqa: E402

from ggTrader.data.live import cached_yfinance_loader as cyl  # noqa: E402
from ggTrader.data.live import yfinance_loader as yfl  # noqa: E402

_orig_connect = cyl.CachedYFinanceLoader._connect


def _ro_connect(self):
    c = _orig_connect(self)
    c.set_session(readonly=True)
    return c


cyl.CachedYFinanceLoader._connect = _ro_connect
cyl.CachedYFinanceLoader.fetch_ohlcv = lambda self, symbols, interval, start_date=None, end_date=None, limit=None: (
    self._db_fetch(symbols, interval, start_date, end_date)
)
cyl.CachedYFinanceLoader._cache_to_db = lambda self, df, interval: None
cyl.CachedYFinanceLoader._record_inceptions = lambda self, *x, **k: None
yfl.YFinanceDataLoader.fetch_ohlcv = lambda self, *x, **k: pd.DataFrame()

from ggTrader.lab.data import (  # noqa: E402
    STOCK_BASE_CONFIG,
    eligible_at,
    equity_universe_between,
    load_ohlcv,
)
from ggTrader.lab.strategies import STRATEGY_REGISTRY  # noqa: E402
from ggTrader.lab.strategy import LabConfig  # noqa: E402
from ggTrader.lab.sweep import build_grid, pit_entry_mask  # noqa: E402
from ggTrader.lab.wfo import _sweep_fold, _sweep_fold_weights  # noqa: E402
from ggTrader.utils.config import get_db_connection_string  # noqa: E402

es_s, ee_s = "2021-01-31", "2022-07-31"
cfg = LabConfig()
es, ee = pd.Timestamp(es_s, tz="UTC"), pd.Timestamp(ee_s, tz="UTC")
warm = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
universe = equity_universe_between(es, ee, universe="sp500")
with (
    psycopg2.connect(get_db_connection_string().replace("postgresql+psycopg2://", "postgresql://")) as c,
    c.cursor() as cur,
):
    c.set_session(readonly=True)
    cur.execute(
        "select symbol from ohlcv where venue='yfinance' and interval='1d' and timestamp >= %s"
        " and timestamp < %s group by symbol having count(*) > 700",
        (str((es - pd.Timedelta(days=warm)).date()), ee_s),
    )
    have = {r[0] for r in cur.fetchall()}
universe = sorted(u for u in universe if u in have and u != "SPY")
# deterministic spread rather than the first N alphabetically
step = max(1, len(universe) // a.n_symbols)
universe = universe[::step][: a.n_symbols]
ohlcv = load_ohlcv(universe + ["SPY"], str((es - pd.Timedelta(days=warm)).date()), ee_s)
spy = ohlcv["SPY"]["close"].dropna()
ohlcv = ohlcv[[s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]]
uni_fn = lambda asof, past: eligible_at(asof, past, cfg, universe="sp500")[0]  # noqa: E731
base = dict(STOCK_BASE_CONFIG)
summary = {"pandas": pd.__version__, "numpy": np.__version__, "n_symbols": len(universe), "rows": len(ohlcv)}

# ---- signal path -----------------------------------------------------------
t = time.time()
mask = pit_entry_mask(ohlcv, uni_fn)
mask.to_parquet(out / "pit_mask.parquet")
summary["pit_mask_s"] = time.time() - t
cls = STRATEGY_REGISTRY["ensemble"]
grid = build_grid(cls)
inst = cls(cfg)
t = time.time()
res, eq = _sweep_fold("ensemble", inst, ohlcv, es, ee, base, grid, entry_mask=mask)
summary["sweep_signal_s"] = time.time() - t
summary["signal_combos"] = len(res)
pd.DataFrame(
    [{"combo": r["combo"], **{k: v for k, v in r.items() if k not in ("combo", "params")}} for r in res]
).to_csv(out / "signal_metrics.csv", index=False)
pd.DataFrame({k: v for k, v in eq.items()}).to_parquet(out / "signal_equity.parquet")

# ---- weight path -----------------------------------------------------------
wcls = STRATEGY_REGISTRY["xs_momentum"]
wgrid = build_grid(wcls)[:6]
t = time.time()
wres, weq = _sweep_fold_weights("xs_momentum", wcls, cfg, ohlcv, es, ee, base, wgrid, uni_fn)
summary["sweep_weight_s"] = time.time() - t
summary["weight_combos"] = len(wres)
pd.DataFrame(
    [{"combo": r["combo"], **{k: v for k, v in r.items() if k not in ("combo", "params")}} for r in wres]
).to_csv(out / "weight_metrics.csv", index=False)
pd.DataFrame({k: v for k, v in weq.items()}).to_parquet(out / "weight_equity.parquet")
(out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps(summary, indent=1, default=str))
