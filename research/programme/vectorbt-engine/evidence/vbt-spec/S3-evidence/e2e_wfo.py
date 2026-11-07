"""Small real-data end-to-end WFO (ensemble, 2 folds) on cached symbols only. READ-ONLY.

Adapted from scratchpad/prof_wfo.py. Differences: limits to N cached symbols, forbids all
network/DB writes (the yfinance loader and cache writers are replaced), records every
warning, writes a JSON artefact + OOS equity CSV so two stacks can be diffed.

Usage: e2e_wfo.py <outdir> [--n-symbols 60] [--eval-end 2022-07-31]
"""
import argparse
import collections
import json
import resource
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n-symbols", type=int, default=60)
ap.add_argument("--eval-end", default="2022-07-31")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

warn_log: list[warnings.WarningMessage] = []
_t_import = time.time()
with warnings.catch_warnings(record=True) as warn_log:
    warnings.simplefilter("always")

    import psycopg2

    from ggTrader.data.live import cached_yfinance_loader as cyl
    from ggTrader.data.live import yfinance_loader as yfl

    # ---- read-only guards -------------------------------------------------
    _orig_connect = cyl.CachedYFinanceLoader._connect

    def _ro_connect(self):
        c = _orig_connect(self)
        c.set_session(readonly=True)
        return c

    cyl.CachedYFinanceLoader._connect = _ro_connect

    def _db_only_fetch(self, symbols, interval, start_date=None, end_date=None, limit=None):
        df = self._db_fetch(symbols, interval, start_date, end_date)
        return df.tail(limit) if limit else df

    cyl.CachedYFinanceLoader.fetch_ohlcv = _db_only_fetch
    cyl.CachedYFinanceLoader._cache_to_db = lambda self, df, interval: None
    cyl.CachedYFinanceLoader._record_inceptions = lambda self, *x, **k: None
    yfl.YFinanceDataLoader.fetch_ohlcv = lambda self, *x, **k: pd.DataFrame()

    from ggTrader.lab.data import (
        STOCK_BASE_CONFIG,
        eligible_at,
        equity_universe_between,
        load_ohlcv,
    )
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.sweep import build_grid
    from ggTrader.lab.wfo import generate_folds, run_wfo
    from ggTrader.utils.config import get_db_connection_string

    import_s = time.time() - _t_import
    es_s, ee_s = "2021-01-31", a.eval_end
    cfg = LabConfig()
    es, ee = pd.Timestamp(es_s, tz="UTC"), pd.Timestamp(ee_s, tz="UTC")
    folds = generate_folds(es, ee)
    warm = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    universe = equity_universe_between(es, ee, universe="sp500")
    with (
        psycopg2.connect(
            get_db_connection_string().replace("postgresql+psycopg2://", "postgresql://")
        ) as c,
        c.cursor() as cur,
    ):
        c.set_session(readonly=True)
        cur.execute(
            "select symbol from ohlcv where venue='yfinance' and interval='1d' and timestamp >= %s"
            " and timestamp < %s group by symbol having count(*) > 700",
            (str((es - pd.Timedelta(days=warm)).date()), ee_s),
        )
        have = {r[0] for r in cur.fetchall()}
    universe = sorted(u for u in universe if u in have and u != "SPY")[: a.n_symbols]
    t0 = time.time()
    ohlcv = load_ohlcv(universe + ["SPY"], str((es - pd.Timedelta(days=warm)).date()), ee_s)
    load_s = time.time() - t0
    spy = ohlcv["SPY"]["close"].dropna()
    ohlcv = ohlcv[[s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]]
    cls = STRATEGY_REGISTRY["ensemble"]
    grid = build_grid(cls)

    t1 = time.time()
    res = run_wfo(
        "ensemble",
        cls,
        cfg,
        ohlcv,
        spy,
        eval_start=es_s,
        eval_end=ee_s,
        market="stock",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=grid,
        universe_fn=lambda asof, past: eligible_at(asof, past, cfg, universe="sp500")[0],
    )
    wfo_s = time.time() - t1

rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (o != o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return o if isinstance(o, (int, str, bool, type(None))) else str(o)


wc = collections.Counter(
    f"{w.category.__name__}: {str(w.message)[:110]} @ {Path(w.filename).name}:{w.lineno}"
    for w in warn_log
)
summary = {
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "n_symbols": len(universe),
    "symbols": universe,
    "rows": len(ohlcv),
    "folds": len(folds),
    "grid": len(grid),
    "load_s": load_s,
    "wfo_s": wfo_s,
    "import_s": import_s,
    "peak_rss_mb": rss_mb,
    "result_is_str": isinstance(res, str),
    "ohlcv_index_dtype": str(ohlcv.index.dtype),
    "ohlcv_col_level0_dtype": str(ohlcv.columns.levels[0].dtype),
}
if not isinstance(res, str):
    summary["fold_results"] = _clean(res.fold_results)
    summary["live_params"] = _clean(res.live_params)
    summary["skipped_folds"] = _clean(list(res.skipped_folds))
    res.oos_equity.to_csv(out / "oos_equity.csv")
    summary["oos_equity_len"] = int(len(res.oos_equity))
    summary["oos_equity_last"] = float(res.oos_equity.iloc[-1])
    (out / "table.txt").write_text(res.table)
else:
    summary["result_str"] = res
summary["warnings_top"] = wc.most_common(40)
(out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps({k: v for k, v in summary.items() if k not in ("symbols", "fold_results", "warnings_top")}, indent=1))
print("WARNINGS:", len(warn_log))
for k, v in wc.most_common(25):
    print(v, k)
sys.stdout.flush()
