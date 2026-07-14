"""Live overlay: recompute each sleeve's trailing equity curve and the
resulting inverse-vol / target-vol weights, reusing the exact research
mechanism from lab/allocation.py and lab/simulate.py.
"""

from __future__ import annotations

import pandas as pd

from ggTrader.data.core.index_constituents import normalize_yf_ticker, universe_members_asof
from ggTrader.lab.data import STOCK_BASE_CONFIG, fetch_stock_ohlcv
from ggTrader.lab.simulate import simulate_signals
from ggTrader.lab.strategies.ensemble import EnsembleSignal
from ggTrader.lab.strategy import LabConfig

SLEEVE_UNIVERSES: tuple[str, ...] = ("sp500", "midcap400", "nasdaq100")


def compute_sleeve_curve(universe: str, asof: pd.Timestamp, window_days: int = 90) -> pd.Series:
    """Trailing equity curve for one sleeve, purely from OHLCV + signals.

    Uses the identical EnsembleSignal(LabConfig(min_history_bars=60))
    construction signal_runner.generate_signals uses live, so the vol
    estimate describes the strategy actually trading (Invariant 1).
    """
    start = asof - pd.Timedelta(days=window_days + 60)  # extra warmup for indicators

    members = universe_members_asof(universe, asof)
    symbols = sorted({normalize_yf_ticker(t) for t in members})

    ohlcv = fetch_stock_ohlcv(symbols, start=str(start.date()), end=str(asof.date()))
    sym_cols = list(ohlcv.columns.get_level_values(0).unique())
    if not sym_cols:
        return pd.Series(dtype=float)

    close = pd.concat({s: ohlcv[s]["close"] for s in sym_cols}, axis=1)
    if close.empty:
        return pd.Series(dtype=float)

    cfg = LabConfig(min_history_bars=60)
    ensemble = EnsembleSignal(cfg)

    plans = {}
    for bar in close.loc[str((asof - pd.Timedelta(days=window_days)).date()) :].index:
        plan = ensemble.select(bar, ohlcv, sym_cols)
        if plan:
            plans[bar] = plan
    if not plans:
        return pd.Series(dtype=float)

    targets = ensemble.to_targets(plans, ohlcv)
    returns_df, equity_df, _diags = simulate_signals(
        {"ensemble": targets}, close, dict(STOCK_BASE_CONFIG), ohlcv=ohlcv
    )
    curve = equity_df["ensemble"].dropna()
    return curve.loc[curve.index >= (asof - pd.Timedelta(days=window_days))]


def compute_weights_and_scale(
    curves: dict[str, pd.Series],
    target_vol: float = 0.068,
    window: int = 60,
    max_leverage: float = 1.0,
) -> tuple[dict[str, float], float]:
    """Inverse-vol weights + target-vol leverage scale for the given sleeve
    curves, reusing the unmodified lab/allocation.py functions."""
    from ggTrader.lab.allocation import (
        inverse_vol_weights,
        target_vol_scale,
        trailing_realized_vol,
    )

    returns = {label: curve.pct_change().dropna() for label, curve in curves.items()}
    vols = {
        label: float(trailing_realized_vol(r, window=window).iloc[-1]) if len(r) >= window else None
        for label, r in returns.items()
    }
    weights = inverse_vol_weights(vols)

    common = None
    for label, r in returns.items():
        common = r.index if common is None else common.intersection(r.index)
    blended = sum(returns[label].reindex(common) * w for label, w in weights.items())
    blend_vol = (
        float(trailing_realized_vol(blended, window=window).iloc[-1])
        if common is not None and len(common) >= window
        else float("nan")
    )
    scale = target_vol_scale(blend_vol, target_vol, max_leverage=max_leverage)
    return weights, scale


def should_rebalance(last_rebalance_date: str | None, today: pd.Timestamp) -> bool:
    """True on the first run after entering a new calendar month, or if
    there is no prior rebalance recorded."""
    if last_rebalance_date is None:
        return True
    last = pd.Timestamp(last_rebalance_date, tz="UTC")
    return (today.year, today.month) != (last.year, last.month)
