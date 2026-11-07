"""F: independent checks on real cached data (read-only use of ggTrader).

1. cProfile overhead per component (eligible_at, _generate_signals, simulate_signals).
2. Vectorised PIT mask vs eligible_at at sampled dates (exact equality?).
3. Prefix-vs-full-history causality through ggTrader's real _sweep_fold, with PIT mask,
   td_stop, ATR stops and vol targeting in the grid.
"""
import cProfile
import json
import sys
import time

import numpy as np
import pandas as pd

OUT = sys.argv[1]
NSUB = int(sys.argv[2]) if len(sys.argv) > 2 else 120


def main() -> None:
    from ggTrader.data.core.index_constituents import universe_members_asof
    from ggTrader.data.core.index_constituents import normalize_yf_ticker  # noqa: F401
    from ggTrader.lab.data import STOCK_BASE_CONFIG, eligible_at, equity_universe_between, load_ohlcv
    from ggTrader.lab.metrics import curve_stats
    from ggTrader.lab.simulate import simulate_signals
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategies.indicators import extract_close, extract_volume
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.sweep import build_grid
    from ggTrader.lab.wfo import _sweep_fold, generate_folds

    res: dict = {}
    es_s, ee_s = "2021-01-31", "2022-07-31"
    cfg = LabConfig()
    es, ee = pd.Timestamp(es_s, tz="UTC"), pd.Timestamp(ee_s, tz="UTC")
    warm = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    universe = equity_universe_between(es, ee, universe="sp500")
    import psycopg2
    from ggTrader.utils.config import get_db_connection_string
    with psycopg2.connect(get_db_connection_string().replace("postgresql+psycopg2://", "postgresql://")) as c, c.cursor() as cur:
        cur.execute("select symbol from ohlcv where timestamp >= %s and timestamp < %s group by symbol having count(*) > 700",
                    (str((es - pd.Timedelta(days=warm)).date()), ee_s))
        have = {r[0] for r in cur.fetchall()}
    universe = [u for u in universe if u in have and u != "SPY"]
    ohlcv = load_ohlcv(universe, str((es - pd.Timedelta(days=warm)).date()), ee_s, use_negative_cache=True)
    syms = sorted(ohlcv.columns.get_level_values(0).unique())
    print("loaded", ohlcv.shape, len(syms), flush=True)
    ufn = lambda asof, past: eligible_at(asof, past, cfg, universe="sp500")[0]  # noqa: E731

    # ---- 1. profiler overhead ----
    dates = list(ohlcv.index[::22])[:40]
    def run_elig():
        for d in dates:
            ufn(d, ohlcv.loc[:d])
    t = time.perf_counter(); run_elig(); t_plain = time.perf_counter() - t
    pr = cProfile.Profile(); t = time.perf_counter(); pr.enable(); run_elig(); pr.disable(); t_prof = time.perf_counter() - t
    res["eligible_at_40dates"] = {"plain_s": t_plain, "profiled_s": t_prof, "ratio": t_prof / t_plain}
    print(res["eligible_at_40dates"], flush=True)

    cls = STRATEGY_REGISTRY["ensemble"]
    strat = cls(cfg)
    close = extract_close(ohlcv, syms); vol = extract_volume(ohlcv, syms)
    strat._generate_signals(close, vol)  # warm
    t = time.perf_counter(); tg = strat._generate_signals(close, vol); g_plain = time.perf_counter() - t
    pr = cProfile.Profile(); t = time.perf_counter(); pr.enable(); strat._generate_signals(close, vol); pr.disable(); g_prof = time.perf_counter() - t
    res["generate_signals_1combo"] = {"plain_s": g_plain, "profiled_s": g_prof, "ratio": g_prof / g_plain}
    print(res["generate_signals_1combo"], flush=True)

    base = dict(STOCK_BASE_CONFIG)
    tgts = {f"c{i}": tg for i in range(12)}
    simulate_signals(tgts, close, base)  # warm (JIT)
    t = time.perf_counter(); simulate_signals(tgts, close, base); s_plain = time.perf_counter() - t
    pr = cProfile.Profile(); t = time.perf_counter(); pr.enable(); simulate_signals(tgts, close, base); pr.disable(); s_prof = time.perf_counter() - t
    res["simulate_signals_12groups"] = {"plain_s": s_plain, "profiled_s": s_prof, "ratio": s_prof / s_plain}
    print(res["simulate_signals_12groups"], flush=True)

    # ---- 2. vectorised PIT mask vs eligible_at ----
    t = time.perf_counter()
    nn = close.notna().cumsum() >= cfg.min_history_bars  # count of non-NaN closes in data<=d
    from ggTrader.data.core.index_constituents import normalize_yf_ticker as nyt
    memb = pd.DataFrame(False, index=close.index, columns=close.columns)
    colset = set(close.columns)
    for d in close.index:
        m = [s for s in (nyt(x) for x in universe_members_asof("sp500", d)) if s in colset]
        memb.loc[d, m] = True
    vmask = memb & nn
    t_vec = time.perf_counter() - t
    sample = list(close.index[::7])
    mism = 0
    for d in sample:
        e = set(ufn(d, ohlcv.loc[:d]))
        v = set(vmask.columns[vmask.loc[d].values])
        mism += len(e ^ v)
    res["pit_vectorised"] = {"vec_build_s": t_vec, "sample_dates": len(sample), "symbol_mismatches": mism}
    print(res["pit_vectorised"], flush=True)

    # ---- 3. causality through real _sweep_fold ----
    sub = syms[:NSUB]
    o2 = ohlcv[sub]
    mask = vmask[sub]
    g48 = build_grid(cls)
    grid = list(g48)
    grid += [{**c, "td_stop": 10} for c in g48[:8]]
    grid += [{**c, "atr_mult": 2.0, "atr_period": 14} for c in g48[:8]]
    grid += [{**c, "ts_stop": 0.05, "tp_stop": 0.10} for c in g48[:8]]
    grid += [{**c, "vol_target": 0.15, "vol_lookback": 20} for c in g48[:8]]
    folds = generate_folds(es, ee)
    full_res, full_eq = _sweep_fold("ensemble", strat, o2, folds[0].train_start, o2.index[-1], base, grid, mask)
    out = []
    for f in folds:
        pre_res, pre_eq = _sweep_fold("ensemble", strat, o2, f.train_start, f.train_end, base, grid, mask)
        eqd = 0.0; shd = 0.0; ntd = 0; nt_rel = []
        pre_by = {r["combo"]: r for r in pre_res}
        for k, s in pre_eq.items():
            fs = full_eq[k].loc[s.index]
            eqd = max(eqd, float(np.nanmax(np.abs(s.values - fs.values))))
            w = full_eq[k].loc[f.train_start:f.train_end].dropna()
            m = curve_stats(float(base["START_CASH"]) * (w / w.iloc[0]))
            if k in pre_by:
                shd = max(shd, abs(m["sharpe"] - pre_by[k]["sharpe"]) if np.isfinite(m["sharpe"]) else 0.0)
        full_by = {r["combo"]: r for r in full_res}
        for k, r in pre_by.items():
            if r["n_trades"] != full_by[k]["n_trades"]:
                ntd += 1
            nt_rel.append((r["n_trades"], full_by[k]["n_trades"]))
        out.append({"train_end": str(f.train_end.date()), "n_combos": len(pre_eq), "max_abs_equity_diff": eqd,
                    "max_abs_sharpe_diff": shd, "combos_with_n_trades_diff": ntd, "n_trades_example": nt_rel[:3]})
        print(out[-1], flush=True)
    res["causality_real_sweep"] = out
    json.dump(res, open(OUT, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
