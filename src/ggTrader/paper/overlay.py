"""Live overlay: recompute each sleeve's trailing equity curve and the
resulting inverse-vol / target-vol weights, reusing the exact research
mechanism from lab/allocation.py and the vectorbt engine (``engine.run_signals``).

The curve no longer goes through ``lab.simulate`` (legacy, deleted after the engine
rebuild): the signals are the shared live ensemble frames and the simulation is the
engine's grouped ``from_signals``, bit-identical to the legacy function on the same
signals (``tests/paper/test_overlay.py``).
"""

from __future__ import annotations

import pandas as pd

from ggTrader.data.core.index_constituents import normalize_yf_ticker, universe_members_asof
from ggTrader.data.membership import MembershipSource
from ggTrader.engine import portfolio as engine_portfolio
from ggTrader.lab.data import STOCK_BASE_CONFIG, fetch_stock_ohlcv
from ggTrader.paper.live_bars import bars_from_legacy_frame
from ggTrader.paper.live_signals import (
    history_eligibility,
    live_ensemble_frames,
    subset_bars,
)

SLEEVE_UNIVERSES: tuple[str, ...] = ("sp500", "midcap400", "nasdaq100")


def compute_sleeve_curve(
    universe: str,
    asof: pd.Timestamp,
    window_days: int = 90,
    membership: MembershipSource | None = None,
) -> pd.Series:
    """Trailing equity curve for one sleeve, purely from OHLCV + signals.

    Uses the identical live ensemble frames (`live_signals.live_ensemble_frames`,
    `min_history_bars=60`) signal_runner.generate_signals uses live, so the vol
    estimate describes the strategy actually trading (Invariant 1).
    """
    # window_days is a calendar-day span; the fetch buffer and the returned
    # curve's trim margin are both padded well past that, because:
    #  (a) min_history_bars=60 is TRADING days, so the fetch needs generous
    #      calendar-day slack beyond window_days to give the earliest kept
    #      bar a full 60-trading-day lookback, and
    #  (b) compute_weights_and_scale computes returns via curve.pct_change(),
    #      which drops one row -- a curve trimmed to exactly `window` trading
    #      days (the vol window, default 60) yields only `window - 1`
    #      returns, permanently one short of that function's `len(r) >=
    #      window` guard, silently collapsing to the equal-weight/scale=0.0
    #      fallback. The +20-calendar-day trim margin below (~14 extra
    #      trading days) keeps this comfortably clear of that boundary.
    trim_margin_days = 20
    warmup_days = 90
    start = asof - pd.Timedelta(days=window_days + trim_margin_days + warmup_days)

    members = universe_members_asof(universe, asof, membership=membership)
    symbols = sorted({normalize_yf_ticker(t) for t in members})

    ohlcv = fetch_stock_ohlcv(symbols, start=str(start.date()), end=str(asof.date()))
    bars = bars_from_legacy_frame(ohlcv, symbols)
    if bars.close.empty or not bars.symbols:
        return pd.Series(dtype=float)

    trim_start = asof - pd.Timedelta(days=window_days + trim_margin_days)
    # A symbol may trade on a bar inside the window once it has 60 valid closes (the live
    # `min_history_bars`); it is in the simulation if that holds on any window bar.
    in_window = pd.Series(bars.close.index >= trim_start.normalize(), index=bars.close.index)
    plan = history_eligibility(bars.close).mul(in_window, axis=0).astype(bool)
    sim_symbols = [s for s in bars.symbols if plan[s].any()]
    if not sim_symbols:
        return pd.Series(dtype=float)

    sub = subset_bars(bars, sim_symbols)
    entries, exits = live_ensemble_frames(sub)
    # Entries only where the PIT/history plan allows; exits stay unmasked so an open
    # position can always close (the legacy `eligibility_mask` convention).
    entries = entries & plan[sim_symbols]
    result = engine_portfolio.run_signals(
        entries,
        exits,
        sub.close,
        dict(STOCK_BASE_CONFIG),
        high=sub.high,
        low=sub.low,
    )
    curve = result.value["all"].dropna()
    return curve.loc[curve.index >= trim_start]


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

    # Curves are dropna()'d upstream, so there is no gap for pandas 2's old default padding to fill.
    returns = {
        label: curve.pct_change(fill_method=None).dropna() for label, curve in curves.items()
    }
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
