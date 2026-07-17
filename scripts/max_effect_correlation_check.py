"""Ad hoc diversification check: does max_effect's OOS return stream
correlate with the deployed 5-voter ensemble core's OOS stream? Same
question idio_vol was put through on 2026-07-07 -- a standalone-NO-GO
weight sleeve can still be worth keeping as a diversifier if its returns
are meaningfully uncorrelated with the deployed core. Uses run_wfo (not
harness.walkforward) for both strategies since it doesn't touch the
research DB -- this is a throwaway correlation measurement, not a run
worth persisting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ggTrader.lab.data import STOCK_BASE_CONFIG, eligible_at, equity_universe_between, load_ohlcv
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategies.max_effect import MaxEffectStrategy
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import WfoResult, run_wfo


def main() -> None:
    cfg = LabConfig(min_history_bars=400)
    eval_start, eval_end = "2015-01-01", str(pd.Timestamp.now().date())
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

    print("Running max_effect WFO...", flush=True)
    max_result = run_wfo(
        MaxEffectStrategy.name,
        MaxEffectStrategy,
        cfg,
        ohlcv,
        spy_close,
        eval_start=eval_start,
        eval_end=eval_end,
        market="equity",
        base_config=base_config,
        grid=build_grid(MaxEffectStrategy),
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

    if not isinstance(max_result, WfoResult) or not isinstance(ensemble_result, WfoResult):
        print(f"max_effect: {max_result}\nensemble: {ensemble_result}")
        return

    max_ret = max_result.oos_equity.pct_change().dropna()
    ens_ret = ensemble_result.oos_equity.pct_change().dropna()
    aligned = pd.concat([max_ret, ens_ret], axis=1).dropna()
    aligned.columns = ["max_effect", "ensemble"]
    if len(aligned) >= 2 and aligned["ensemble"].std() > 0:
        corr = aligned["max_effect"].corr(aligned["ensemble"])
        beta = float(np.polyfit(aligned["ensemble"], aligned["max_effect"], 1)[0])
        print(
            f"Diversification check: OOS return correlation to deployed core = "
            f"{corr:.3f}, beta = {beta:.3f}, overlapping days = {len(aligned)}"
        )
    else:
        print("Diversification check: insufficient overlapping OOS data to compute.")


if __name__ == "__main__":
    main()
