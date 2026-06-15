"""Vectorized portfolio simulation: one grouped vbt call across all strategies."""

from __future__ import annotations

from typing import Any, Dict, Tuple

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
    returns = value.pct_change().fillna(0.0)
    diags = {
        name: {"n_strategies": 1, "n_symbols": int(targets_by_strategy[name].shape[1])}
        for name in names
    }
    return returns, value, diags
