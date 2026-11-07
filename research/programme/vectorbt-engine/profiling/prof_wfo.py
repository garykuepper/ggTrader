"""Throwaway profile of the current equity WFO path (SP500 ensemble, full grid, N folds)."""
import argparse, cProfile, pstats, time, json
from pathlib import Path
import pandas as pd

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-end", default="2022-07-31")  # 2 folds
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from ggTrader.lab.data import STOCK_BASE_CONFIG, eligible_at, equity_universe_between, load_ohlcv
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.sweep import build_grid
    from ggTrader.lab.wfo import run_wfo, generate_folds
    es_s, ee_s = "2021-01-31", a.eval_end
    cfg = LabConfig()
    es, ee = pd.Timestamp(es_s, tz="UTC"), pd.Timestamp(ee_s, tz="UTC")
    print("folds:", len(generate_folds(es, ee)), flush=True)
    warm = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    t0 = time.time()
    universe = equity_universe_between(es, ee, universe="sp500")
    import psycopg2
    from ggTrader.utils.config import get_db_connection_string
    with psycopg2.connect(get_db_connection_string().replace("postgresql+psycopg2://", "postgresql://")) as c, c.cursor() as cur:
        cur.execute("select symbol from ohlcv where timestamp >= %s and timestamp < %s group by symbol having count(*) > 700",
                    (str((es - pd.Timedelta(days=warm)).date()), ee_s))
        have = {r[0] for r in cur.fetchall()}
    print(f"universe {len(universe)} -> {len([u for u in universe if u in have])} cached", flush=True)
    universe = [u for u in universe if u in have and u != "SPY"]
    ohlcv = load_ohlcv(universe + ["SPY"], str((es - pd.Timedelta(days=warm)).date()), ee_s, use_negative_cache=True)
    load_s = time.time() - t0
    spy = ohlcv["SPY"]["close"].dropna()
    ohlcv = ohlcv[[s for s in ohlcv.columns.get_level_values(0).unique() if s != "SPY"]]
    cls = STRATEGY_REGISTRY["ensemble"]; grid = build_grid(cls)
    print(f"load {load_s:.1f}s symbols={ohlcv.columns.get_level_values(0).nunique()} rows={len(ohlcv)} grid={len(grid)}", flush=True)
    pr = cProfile.Profile(); t1 = time.time(); pr.enable()
    run_wfo("ensemble", cls, cfg, ohlcv, spy, eval_start=es_s, eval_end=ee_s, market="stock",
            base_config=dict(STOCK_BASE_CONFIG), grid=grid,
            universe_fn=lambda asof, past: eligible_at(asof, past, cfg, universe="sp500")[0])
    pr.disable(); wfo_s = time.time() - t1
    out = Path(a.out); pr.dump_stats(str(out / "wfo.prof"))
    json.dump({"load_s": load_s, "wfo_s": wfo_s, "grid": len(grid)}, open(out / "timing.json", "w"))
    print(f"run_wfo {wfo_s:.1f}s", flush=True)

if __name__ == "__main__":
    main()
