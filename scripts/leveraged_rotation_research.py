"""Leveraged/inverse index rotation research: WFO each universe's
breadth-driven rotation strategy against real leveraged/inverse ETF price
history. See docs/superpowers/specs/2026-07-14-leveraged-index-rotation-design.md.
"""

from __future__ import annotations

import pandas as pd

from ggTrader.lab.data import STOCK_BASE_CONFIG, load_ohlcv, normalize_yf_ticker
from ggTrader.data.core.index_constituents import universe_members_asof
from ggTrader.lab.strategies.leveraged_rotation import (
    LeveragedRotationNasdaq100,
    LeveragedRotationRussell2000,
    LeveragedRotationSp500,
)
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.sweep import build_grid
from ggTrader.lab.wfo import WfoResult, run_wfo

UNIVERSES: dict[str, type] = {
    "sp500": LeveragedRotationSp500,
    "nasdaq100": LeveragedRotationNasdaq100,
    "russell2000": LeveragedRotationRussell2000,
}


def _fixed_universe_fn(tickers: tuple[str, ...]):
    """universe_fn is called once per (asof, combo) without seeing which
    combo is active, so it can't be leverage-tier-aware -- it always
    returns the full 4-ticker union; the strategy picks its own 2 relevant
    tickers internally via self.leverage_tier."""

    def _fn(asof: pd.Timestamp, past: pd.DataFrame | None) -> list[str]:
        return list(tickers)

    return _fn


def run_universe(universe: str, eval_start: str, eval_end: str, cfg: LabConfig) -> str:
    cls = UNIVERSES[universe]
    es = pd.Timestamp(eval_start, tz="UTC")
    ee = pd.Timestamp(eval_end, tz="UTC")
    warmup_days = int(max(cfg.lookback, cfg.min_history_bars) * 1.6) + 60
    data_start = str((es - pd.Timedelta(days=warmup_days)).date())

    now = pd.Timestamp.now(tz="UTC")
    members = [normalize_yf_ticker(m) for m in universe_members_asof(universe, now)]
    etf_tickers = sorted(set(cls.PAIR_3X) | set(cls.PAIR_2X))
    all_symbols = sorted(set(members) | set(etf_tickers) | {"SPY"})
    ohlcv = load_ohlcv(all_symbols, data_start, eval_end, use_negative_cache=True)
    spy_close = ohlcv["SPY"]["close"].dropna()

    result = run_wfo(
        cls.name,
        cls,
        cfg,
        ohlcv,
        spy_close,
        eval_start=eval_start,
        eval_end=eval_end,
        market="equity",
        base_config=dict(STOCK_BASE_CONFIG),
        grid=build_grid(cls),
        universe_fn=_fixed_universe_fn(tuple(etf_tickers)),
    )
    if not isinstance(result, WfoResult):
        return f"{universe}: {result}"
    return f"{universe}: WFO complete\n{result.table}"


def main() -> None:
    cfg = LabConfig(min_history_bars=400)
    for universe in UNIVERSES:
        print(run_universe(universe, "2019-01-01", str(pd.Timestamp.now().date()), cfg))
        print()


if __name__ == "__main__":
    main()
