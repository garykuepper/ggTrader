"""Long-only leveraged-ETF trend research: WFO each universe's trend-filtered
leveraged-ETF strategy against real ETF price history, plus a naive
buy-and-hold-the-leveraged-ETF baseline for comparison. See
src/ggTrader/lab/strategies/leveraged_trend.py and the closed
breadth-driven rotation arc at
docs/research/2026-07-16-leveraged-index-rotation-nogo.md.
"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt

from ggTrader.lab.data import STOCK_BASE_CONFIG, load_ohlcv
from ggTrader.lab.metrics import curve_stats
from ggTrader.lab.strategies.leveraged_trend import (
    LeveragedTrendNasdaq100,
    LeveragedTrendRussell2000,
    LeveragedTrendSp500,
)
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import WfoResult, run_wfo

UNIVERSES: dict[str, type] = {
    "sp500": LeveragedTrendSp500,
    "nasdaq100": LeveragedTrendNasdaq100,
    "russell2000": LeveragedTrendRussell2000,
}


def _fixed_universe_fn(tickers: tuple[str, ...]):
    def _fn(asof: pd.Timestamp, past: pd.DataFrame | None) -> list[str]:
        return list(tickers)

    return _fn


def _buy_and_hold(close: pd.Series, eval_start: str, eval_end: str, base_config: dict) -> str:
    """Naive buy-day-1-hold-to-end baseline, same fee/slippage model as the
    WFO run but with no signal/overlay -- what the strategy has to beat."""
    window = close.loc[eval_start:eval_end].dropna()
    pf = vbt.Portfolio.from_holding(
        close=window,
        init_cash=float(base_config["START_CASH"]),
        fees=float(base_config["FEES"]),
        slippage=float(base_config["SLIPPAGE"]),
        freq=base_config["FREQ"],
    )
    stats = curve_stats(pf.value())
    return (
        f"Sharpe {stats['sharpe']:.2f} | CAGR {stats['cagr_pct']:.1f}% "
        f"| MaxDD {stats['max_drawdown_pct']:.1f}%"
    )


def run_universe(universe: str, eval_start: str, eval_end: str, cfg: LabConfig) -> str:
    cls = UNIVERSES[universe]
    es = pd.Timestamp(eval_start, tz="UTC")
    max_trend_window = max(cls.sweep_params()["trend_window"])
    warmup_days = int(max(cfg.lookback, max_trend_window) * 1.6) + 60
    data_start = str((es - pd.Timedelta(days=warmup_days)).date())

    # Scoped tightly to the traded ETFs only -- compute_vol_scalar averages
    # realized vol across every column in this panel, so the underlying
    # index (loaded separately, out-of-band, by the strategy's own cached
    # loader) must NOT be included here or it would dilute the vol estimate
    # used to size the leveraged ETF.
    all_symbols = [cls.ETF_3X, cls.ETF_2X]
    ohlcv = load_ohlcv(all_symbols, data_start, eval_end, use_negative_cache=True)
    spy_close = load_ohlcv(["SPY"], data_start, eval_end, use_negative_cache=True)["SPY"]["close"]
    spy_close = spy_close.dropna()

    base_config = dict(STOCK_BASE_CONFIG)
    base_config.update(
        {
            "SIGNAL_POSITION_SIZE": 1.0,  # full notional in the ETF when the entry signal is on
            "vol_lookback": 20,
            "vol_cap": 1.0,  # overlay can only de-lever, never stack more on top of 2x/3x
        }
    )

    result = run_wfo(
        cls.name,
        cls,
        cfg,
        ohlcv,
        spy_close,
        eval_start=eval_start,
        eval_end=eval_end,
        market="equity",
        base_config=base_config,
        grid=build_grid(cls),
        universe_fn=_fixed_universe_fn((cls.ETF_3X, cls.ETF_2X)),
    )
    if not isinstance(result, WfoResult):
        return f"{universe}: {result}"

    bh_config = dict(STOCK_BASE_CONFIG)
    bh_3x = _buy_and_hold(ohlcv[cls.ETF_3X]["close"], eval_start, eval_end, bh_config)
    bh_2x = _buy_and_hold(ohlcv[cls.ETF_2X]["close"], eval_start, eval_end, bh_config)
    return (
        f"{universe}: WFO complete\n{result.table}\n"
        f"Buy&Hold {cls.ETF_3X}: {bh_3x}\n"
        f"Buy&Hold {cls.ETF_2X}: {bh_2x}"
    )


def main() -> None:
    cfg = LabConfig(min_history_bars=400)
    for universe in UNIVERSES:
        print(run_universe(universe, "2019-01-01", str(pd.Timestamp.now().date()), cfg))
        print()


if __name__ == "__main__":
    main()
