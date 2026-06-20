"""Vectorized portfolio simulation: one grouped vbt call across all strategies."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt


def simulate_weights(
    targets_by_strategy: Dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    base_config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Simulate every weight-based strategy in ONE from_orders call.

    Args:
        targets_by_strategy: name -> (time x symbol) targetpercent matrix
            (NaN = no order that bar; 0.0 = exit; w = target weight).
        prices: (time x symbol) close prices covering every target column.
        base_config: START_CASH, FEES, SLIPPAGE, FREQ.

    Returns:
        (returns_df, equity_df, diags) each keyed by strategy name (columns).
    """
    names = list(targets_by_strategy)
    size_blocks, close_blocks, groups = [], [], []
    for name in names:
        tgt = targets_by_strategy[name]
        cols = pd.MultiIndex.from_product([[name], tgt.columns], names=["strategy", "symbol"])
        size_blocks.append(tgt.set_axis(cols, axis=1))
        px = prices[tgt.columns].reindex(tgt.index).ffill()
        close_blocks.append(px.set_axis(cols, axis=1))
        groups.extend([name] * tgt.shape[1])

    size = pd.concat(size_blocks, axis=1)
    close = pd.concat(close_blocks, axis=1)

    pf = vbt.Portfolio.from_orders(
        close=close,
        size=size,
        size_type="targetpercent",
        init_cash=float(base_config["START_CASH"]),
        fees=float(base_config["FEES"]),
        slippage=float(base_config["SLIPPAGE"]),
        freq=base_config["FREQ"],
        cash_sharing=True,
        group_by=pd.Index(groups, name="strategy"),
        call_seq="auto",
    ).copy()

    value = pf.value()  # (time x strategy) once grouped
    if isinstance(value, pd.Series):
        value = value.to_frame(names[0])
    value = value[names]
    returns = value.pct_change(fill_method=None).fillna(0.0)
    diags = {
        name: {"n_strategies": 1, "n_symbols": int(targets_by_strategy[name].shape[1])}
        for name in names
    }
    return returns, value, diags


def compute_vol_scalar(
    prices: pd.DataFrame,
    vol_target: float,
    vol_lookback: int,
    vol_cap: float = 2.0,
) -> pd.Series:
    """Daily vol-targeting scalar: target_vol / realized_vol, lagged by 1 bar.

    Uses the equal-weight universe's trailing annualized vol. The scalar is
    shifted by 1 day to prevent lookahead.  Capped at ``vol_cap`` (no
    leverage when cap=1.0) and floored at 0.1.
    """
    rets = prices.pct_change(fill_method=None)
    avg_ret = rets.mean(axis=1)
    realized = avg_ret.rolling(window=vol_lookback, min_periods=vol_lookback).std() * np.sqrt(252)
    scalar = (vol_target / realized).shift(1).clip(lower=0.1, upper=vol_cap).fillna(1.0)
    return scalar


def compute_atr_stop(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    atr_period: int,
    atr_mult: float,
) -> pd.DataFrame:
    """Vectorized ATR trailing stop as fractional distance from close.

    Returns (time x symbol) DataFrame of stop fractions for vbt's sl_stop.
    """

    prev_close = close.shift(1)
    tr = pd.DataFrame(
        {
            col: np.maximum(
                np.maximum(
                    high[col] - low[col],
                    (high[col] - prev_close[col]).abs(),
                ),
                (low[col] - prev_close[col]).abs(),
            )
            for col in close.columns
        },
        index=close.index,
    )
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    return (atr_mult * atr / close).clip(lower=0.001)


def simulate_signals(
    targets_by_strategy: Dict[str, Any],  # values are SignalTargets instances
    prices: pd.DataFrame,
    base_config: Dict[str, Any],
    ohlcv: pd.DataFrame | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Simulate every signal-based strategy in ONE from_signals call.

    Args:
        targets_by_strategy: name -> SignalTargets(entries, exits) where each
            frame is (time x symbol) boolean — True = entry/exit on that bar.
        prices: (time x symbol) close prices covering every target column.
        base_config: START_CASH, FEES, SLIPPAGE, FREQ.
            Optional SIGNAL_POSITION_SIZE (fraction of portfolio per entry, default 0.02).
            Optional ts_stop (float): fixed trailing stop fraction (e.g. 0.05 = 5%).
            Optional atr_mult (float) + atr_period (int, default 14): ATR-adaptive
            trailing stop. Requires ohlcv kwarg.
        ohlcv: MultiIndex (symbol, field) OHLCV DataFrame. Required when atr_mult is set.

    Returns:
        (returns_df, equity_df, diags) each keyed by strategy name (columns).
    """
    names = list(targets_by_strategy)
    entry_blocks, exit_blocks, close_blocks, groups = [], [], [], []

    for name in names:
        st = targets_by_strategy[name]
        cols = pd.MultiIndex.from_product(
            [[name], st.entries.columns], names=["strategy", "symbol"]
        )
        entry_blocks.append(st.entries.set_axis(cols, axis=1))
        exit_blocks.append(st.exits.set_axis(cols, axis=1))
        px = prices[st.entries.columns].reindex(st.entries.index).ffill()
        close_blocks.append(px.set_axis(cols, axis=1))
        groups.extend([name] * st.entries.shape[1])

    entries = pd.concat(entry_blocks, axis=1).fillna(False)
    exits = pd.concat(exit_blocks, axis=1).fillna(False)
    close = pd.concat(close_blocks, axis=1)

    # --- Stop params ---
    stop_kwargs: Dict[str, Any] = {}
    ts_stop = base_config.get("ts_stop")
    atr_mult = base_config.get("atr_mult")
    if ts_stop is not None:
        stop_kwargs["sl_stop"] = float(ts_stop)
        stop_kwargs["sl_trail"] = True
    elif atr_mult is not None:
        atr_period = int(base_config.get("atr_period", 14))
        if atr_period < 1:
            msg = f"atr_period must be >= 1, got {atr_period}"
            raise ValueError(msg)
        if ohlcv is None:
            msg = "ohlcv required for ATR trailing stop (atr_mult set)"
            raise ValueError(msg)
        syms = list(prices.columns)
        high_df = pd.concat(
            {s: ohlcv[s]["high"] for s in syms if s in ohlcv.columns.get_level_values(0)},
            axis=1,
        )
        low_df = pd.concat(
            {s: ohlcv[s]["low"] for s in syms if s in ohlcv.columns.get_level_values(0)},
            axis=1,
        )
        close_df = pd.concat(
            {s: ohlcv[s]["close"] for s in syms if s in ohlcv.columns.get_level_values(0)},
            axis=1,
        )
        atr_stops = compute_atr_stop(high_df, low_df, close_df, atr_period, float(atr_mult))
        # Broadcast ATR stops to match the stacked multi-strategy column layout
        sl_blocks = []
        for name in names:
            st = targets_by_strategy[name]
            cols = pd.MultiIndex.from_product(
                [[name], st.entries.columns], names=["strategy", "symbol"]
            )
            sl_blocks.append(
                atr_stops[st.entries.columns].reindex(st.entries.index).set_axis(cols, axis=1)
            )
        stop_kwargs["sl_stop"] = pd.concat(sl_blocks, axis=1).ffill().fillna(np.inf)
        stop_kwargs["sl_trail"] = True

    # --- Vol targeting ---
    base_size = float(base_config.get("SIGNAL_POSITION_SIZE", 0.02))
    vol_target = base_config.get("vol_target")
    if vol_target is not None:
        vol_lookback = int(base_config.get("vol_lookback", 20))
        vol_cap = float(base_config.get("vol_cap", 2.0))
        scalar = compute_vol_scalar(prices, float(vol_target), vol_lookback, vol_cap)
        size_param: float | pd.DataFrame = pd.DataFrame(
            {
                col: base_size * scalar.reindex(close.index).ffill().fillna(1.0)
                for col in close.columns
            },
            index=close.index,
        )
    else:
        size_param = base_size

    # --- Per-bar conviction sizing (overrides vol targeting where present) ---
    for name in names:
        st = targets_by_strategy[name]
        if hasattr(st, "sizes") and st.sizes is not None:
            cols = pd.MultiIndex.from_product(
                [[name], st.sizes.columns], names=["strategy", "symbol"]
            )
            conviction_sizes = st.sizes.set_axis(cols, axis=1).reindex(index=close.index)
            if not isinstance(size_param, pd.DataFrame):
                size_param = pd.DataFrame(base_size, index=close.index, columns=close.columns)
            size_param.update(conviction_sizes)

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        size=size_param,
        size_type="percent",
        init_cash=float(base_config["START_CASH"]),
        fees=float(base_config["FEES"]),
        slippage=float(base_config["SLIPPAGE"]),
        freq=base_config["FREQ"],
        cash_sharing=True,
        group_by=pd.Index(groups, name="strategy"),
        **stop_kwargs,
    ).copy()

    value = pf.value()
    if isinstance(value, pd.Series):
        value = value.to_frame(names[0])
    value = value[names]
    returns = value.pct_change(fill_method=None).fillna(0.0)
    diags = {
        name: {"n_strategies": 1, "n_symbols": int(targets_by_strategy[name].entries.shape[1])}
        for name in names
    }
    return returns, value, diags
