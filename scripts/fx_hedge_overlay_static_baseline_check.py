"""Ad hoc decisive check for fx_hedge_overlay (candidate A1): does the
dynamic carry+value+trend hedge ratio actually beat a STATIC hedge policy
on the same instruments? SPY is the wrong benchmark for this candidate --
the source paper's actual claim (Castro/Hamill/Harber/Harvey/Van Hemert,
"The Best Strategies for FX Hedging," JPM 2025) is that dynamic hedging
beats static hedged/unhedged/50-50 policies, not that FX hedging beats the
US equity market outright. This is the decisive test before calling a
verdict, mirroring this project's established pattern of not trusting a
standalone number against the wrong benchmark (see the eval-window-drift
lesson from pead/insider_cluster/congress_trades).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ggTrader.lab.data import STOCK_BASE_CONFIG, load_ohlcv
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.simulate import simulate_weights
from ggTrader.lab.strategies.fx_hedge_overlay import FX_HEDGE_PAIRS, FxHedgeOverlayStrategy
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import WfoResult, run_wfo


def _static_targets(prices: pd.DataFrame, unhedged_weight: float) -> pd.DataFrame:
    """Constant-weight targets: unhedged_weight to every unhedged ticker's
    per-pair share, the rest to hedged -- held from the first bar onward
    (no rebalancing needed, weights never change)."""
    per_pair_share = 1.0 / len(FX_HEDGE_PAIRS)
    cols = [t for pair in FX_HEDGE_PAIRS for t in (pair.unhedged, pair.hedged)]
    targets = pd.DataFrame(np.nan, index=prices.index, columns=cols)
    for pair in FX_HEDGE_PAIRS:
        targets.loc[targets.index[0], pair.unhedged] = per_pair_share * unhedged_weight
        targets.loc[targets.index[0], pair.hedged] = per_pair_share * (1.0 - unhedged_weight)
    return targets


def main() -> None:
    cfg = LabConfig(min_history_bars=400)
    eval_start, eval_end = "2016-01-01", str(pd.Timestamp.now().date())
    es = pd.Timestamp(eval_start, tz="UTC")
    ee = pd.Timestamp(eval_end, tz="UTC")
    warmup_days = int(cfg.min_history_bars * 1.6) + 60
    data_start = str((es - pd.Timedelta(days=warmup_days)).date())

    tickers = [t for pair in FX_HEDGE_PAIRS for t in (pair.unhedged, pair.hedged)]
    ohlcv = load_ohlcv(tickers + ["SPY"], data_start, eval_end, use_negative_cache=True)
    spy_close = ohlcv["SPY"]["close"].dropna()
    ohlcv = ohlcv[tickers]

    def universe_fn(asof: pd.Timestamp, past: pd.DataFrame) -> list[str]:
        return tickers

    print("Running fx_hedge_overlay WFO (dynamic)...", flush=True)
    dynamic_result = run_wfo(
        FxHedgeOverlayStrategy.name,
        FxHedgeOverlayStrategy,
        cfg,
        ohlcv,
        spy_close,
        eval_start=eval_start,
        eval_end=eval_end,
        market="equity",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=build_grid(FxHedgeOverlayStrategy),
        universe_fn=universe_fn,
    )
    if not isinstance(dynamic_result, WfoResult):
        print(f"dynamic run did not produce a WfoResult: {dynamic_result}")
        return
    dynamic_equity = dynamic_result.oos_equity.dropna()
    oos_start, oos_end = dynamic_equity.index[0], dynamic_equity.index[-1]
    print(f"OOS span: {oos_start.date()} .. {oos_end.date()}", flush=True)

    prices = pd.concat({s: ohlcv[s]["close"] for s in tickers}, axis=1)
    prices = prices.loc[oos_start:oos_end]

    static_configs = {
        "static_100pct_unhedged": 1.0,
        "static_100pct_hedged": 0.0,
        "static_50_50": 0.5,
    }
    targets_by_strategy = {name: _static_targets(prices, w) for name, w in static_configs.items()}
    returns_df, equity_df, _diags = simulate_weights(
        targets_by_strategy, prices, dict(STOCK_BASE_CONFIG)
    )

    spy_window = spy_close.loc[oos_start:oos_end]
    spy_norm = spy_window / spy_window.iloc[0] * float(STOCK_BASE_CONFIG["START_CASH"])

    print("\n=== fx_hedge_overlay: dynamic vs static-hedge baselines ===")
    print(f"{'Config':<28} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8}")
    dyn_stats = curve_stats(dynamic_equity)
    print(
        f"{'dynamic (carry+value+trend)':<28} {dyn_stats['sharpe']:>8.2f} "
        f"{dyn_stats['cagr_pct']:>8.1f} {dyn_stats['max_drawdown_pct']:>8.1f}"
    )
    for name in static_configs:
        s = curve_stats(equity_df[name].dropna())
        print(f"{name:<28} {s['sharpe']:>8.2f} {s['cagr_pct']:>8.1f} {s['max_drawdown_pct']:>8.1f}")
    spy_stats = curve_stats(spy_norm)
    print(
        f"{'SPY (reference only)':<28} {spy_stats['sharpe']:>8.2f} "
        f"{spy_stats['cagr_pct']:>8.1f} {spy_stats['max_drawdown_pct']:>8.1f}"
    )

    dyn_ret = dynamic_equity.pct_change().dropna()
    static5050_ret = equity_df["static_50_50"].dropna().pct_change().dropna()
    common = dyn_ret.index.intersection(static5050_ret.index)
    corr = dyn_ret.loc[common].corr(static5050_ret.loc[common])
    print(f"\nCorrelation, dynamic vs static 50/50: {corr:.3f}")


if __name__ == "__main__":
    main()
