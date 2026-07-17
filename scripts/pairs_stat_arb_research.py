"""Market-neutral pairs/stat-arb research: WFO the same-sector spread-
mean-reversion strategy against the full SP500 universe, plus a
market-neutrality check (OOS return correlation/beta to SPY) since that is
the strategy's actual design goal, not just Sharpe. See
src/ggTrader/lab/strategies/pairs_stat_arb.py and
docs/research/RESEARCH_SNAPSHOT.md §6 Rank 1.

Known, deliberate v1 simplifications (see the eventual research report):
no short-borrow-fee/margin-interest cost modeling, same-sector-only pair
universe, naive 1:1 (unhedged) log-price spread, monthly-only rebalance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ggTrader.lab.data import STOCK_BASE_CONFIG, eligible_at, equity_universe_between, load_ohlcv
from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb
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

    # No borrow-fee/margin-interest modeling in v1 -- FEES/SLIPPAGE only.
    base_config = dict(STOCK_BASE_CONFIG)

    def universe_fn(asof: pd.Timestamp, past: pd.DataFrame) -> list[str]:
        eligible, _coverage = eligible_at(asof, past, cfg, universe="sp500")
        return eligible

    result = run_wfo(
        PairsStatArb.name,
        PairsStatArb,
        cfg,
        ohlcv,
        spy_close,
        eval_start=eval_start,
        eval_end=eval_end,
        market="equity",
        base_config=base_config,
        grid=build_grid(PairsStatArb),
        universe_fn=universe_fn,
    )
    if not isinstance(result, WfoResult):
        print(f"pairs_stat_arb: {result}")
        return

    print(f"pairs_stat_arb: WFO complete\n{result.table}")

    # Market-neutrality is itself a result to report, not just Sharpe: a
    # strategy showing a good Sharpe but material correlation/beta to SPY
    # has NOT achieved its design goal even if profitable.
    strat_ret = result.oos_equity.pct_change().dropna()
    spy_ret = spy_close.pct_change().reindex(strat_ret.index)
    aligned = pd.concat([strat_ret, spy_ret], axis=1).dropna()
    aligned.columns = ["strategy", "spy"]
    if len(aligned) >= 2 and aligned["spy"].std() > 0:
        realized_corr = aligned["strategy"].corr(aligned["spy"])
        realized_beta = float(np.polyfit(aligned["spy"], aligned["strategy"], 1)[0])
        print(
            f"Market-neutrality check: OOS return correlation to SPY = "
            f"{realized_corr:.3f}, beta = {realized_beta:.3f}"
        )
    else:
        print("Market-neutrality check: insufficient overlapping OOS data to compute.")


if __name__ == "__main__":
    main()
