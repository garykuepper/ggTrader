"""Ad hoc diversification check: does insider_cluster_buy's OOS return
stream correlate with the deployed 5-voter ensemble core's OOS stream?
Same question idio_vol (2026-07-07), max_effect (2026-07-17), and pead
(2026-07-17) were put through -- insider_cluster_buy's standalone WFO
tied SPY on Sharpe (0.76 vs 0.77) but with a notably shallower drawdown
(-21.0% vs -33.7%), the same near-tie-with-better-risk-profile shape as
idio_vol's case, which is the actual reason to check correlation here
rather than close it as a clean standalone rejection. Uses run_wfo (not
harness.walkforward) for both strategies since it doesn't touch the
research DB -- this is a throwaway correlation measurement, not a run
worth persisting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ggTrader.lab.data import STOCK_BASE_CONFIG, eligible_at, equity_universe_between, load_ohlcv
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategies.insider_cluster import InsiderClusterBuyStrategy
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import WfoResult, run_wfo


def main() -> None:
    cfg = LabConfig(min_history_bars=400)
    eval_start, eval_end = "2015-06-01", str(pd.Timestamp.now().date())
    es = pd.Timestamp(eval_start, tz="UTC")
    ee = pd.Timestamp(eval_end, tz="UTC")

    warmup_days = int(cfg.min_history_bars * 1.6) + 60
    data_start = str((es - pd.Timedelta(days=warmup_days)).date())

    symbols = equity_universe_between(es, ee, universe="sp500")
    ohlcv = load_ohlcv(symbols, data_start, eval_end, use_negative_cache=True)
    spy_close = load_ohlcv(["SPY"], data_start, eval_end, use_negative_cache=True)["SPY"]["close"]
    spy_close = spy_close.dropna()

    base_config = dict(STOCK_BASE_CONFIG)

    def universe_fn(asof: pd.Timestamp, past: pd.DataFrame) -> list[str]:
        eligible, _coverage = eligible_at(asof, past, cfg, universe="sp500")
        return eligible

    print("Running insider_cluster_buy WFO...", flush=True)
    ic_result = run_wfo(
        InsiderClusterBuyStrategy.name,
        InsiderClusterBuyStrategy,
        cfg,
        ohlcv,
        spy_close,
        eval_start=eval_start,
        eval_end=eval_end,
        market="equity",
        base_config=base_config,
        grid=build_grid(InsiderClusterBuyStrategy),
        universe_fn=universe_fn,
    )
    print("Running ensemble (deployed core) WFO...", flush=True)
    ensemble_result = run_wfo(
        EnsembleSignal.name,
        EnsembleSignal,
        cfg,
        ohlcv,
        spy_close,
        eval_start=eval_start,
        eval_end=eval_end,
        market="equity",
        base_config=base_config,
        grid=build_grid(EnsembleSignal),
        universe_fn=universe_fn,
    )

    if not isinstance(ic_result, WfoResult) or not isinstance(ensemble_result, WfoResult):
        print(f"insider_cluster_buy: {ic_result}\nensemble: {ensemble_result}")
        return

    ic_ret = ic_result.oos_equity.pct_change().dropna()
    ens_ret = ensemble_result.oos_equity.pct_change().dropna()
    aligned = pd.concat([ic_ret, ens_ret], axis=1).dropna()
    aligned.columns = ["insider_cluster_buy", "ensemble"]
    if len(aligned) >= 2 and aligned["ensemble"].std() > 0:
        corr = aligned["insider_cluster_buy"].corr(aligned["ensemble"])
        beta = float(np.polyfit(aligned["ensemble"], aligned["insider_cluster_buy"], 1)[0])
        print(
            f"Diversification check: OOS return correlation to deployed core = "
            f"{corr:.3f}, beta = {beta:.3f}, overlapping days = {len(aligned)}"
        )
    else:
        print("Diversification check: insufficient overlapping OOS data to compute.")


if __name__ == "__main__":
    main()
