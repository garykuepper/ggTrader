"""Closed-trade and open-position gate helpers align with Sharpe index (grouped portfolios).

Root cause addressed: when VectorBT returns ``trades.count()`` per underlying column
(MultiIndex) while ``sharpe_ratio()`` is per ``group_by`` combo, naive boolean masking
misaligns. Helpers aggregate by param-combo level to match ``sharpe_series.index``.
"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt

from ggTrader.core.orchestrator import (
    _open_position_count_end_for_gate,
    _trade_counts_for_train_gate,
)


def test_trade_counts_align_with_sharpe_under_cash_sharing_group_by() -> None:
    """Synthetic 2x2 column MultiIndex: gate Series matches grouped Sharpe index."""
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    cols = pd.MultiIndex.from_product([[0, 1], ["A", "B"]], names=["combo", "sym"])
    entries = pd.DataFrame(False, index=idx, columns=cols)
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    entries.iloc[5, 0] = True
    entries.iloc[20, 0] = True
    entries.iloc[8, 2] = True
    entries.iloc[22, 2] = True
    exits = entries.shift(1).fillna(False).infer_objects(copy=False).astype(bool)
    group_by = cols.droplevel(-1)
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=10000.0,
        fees=0.001,
        freq="1d",
        size=0.5,
        size_type="percent",
        cash_sharing=True,
        group_by=group_by,
    )
    sharpe = pf.sharpe_ratio()
    aligned = _trade_counts_for_train_gate(pf, sharpe)
    assert isinstance(aligned, pd.Series)
    assert aligned.index.equals(sharpe.index)
    assert len(aligned) == 2
    assert (aligned >= 0).all()
    open_end = _open_position_count_end_for_gate(pf, sharpe)
    assert open_end.index.equals(sharpe.index)


def test_trade_counts_passthrough_when_index_already_matches() -> None:
    """When VBT already aligns counts to groups, helper returns without reshaping."""
    idx = pd.date_range("2020-01-01", periods=30, freq="D")
    cols = pd.MultiIndex.from_product([[0], ["A"]], names=["combo", "sym"])
    entries = pd.DataFrame(False, index=idx, columns=cols)
    close = pd.DataFrame(100.0, index=idx, columns=cols)
    entries.iloc[3, 0] = True
    entries.iloc[15, 0] = True
    exits = entries.shift(1).fillna(False).infer_objects(copy=False).astype(bool)
    group_by = cols.droplevel(-1)
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=10000.0,
        fees=0.001,
        freq="1d",
        size=1.0,
        size_type="percent",
        cash_sharing=True,
        group_by=group_by,
    )
    sh = pf.sharpe_ratio()
    if not isinstance(sh, pd.Series):
        sh = pd.Series([float(sh)])
    raw = pf.trades.count()
    aligned = _trade_counts_for_train_gate(pf, sh)
    assert aligned.index.equals(sh.index)
    if isinstance(raw, pd.Series) and raw.index.equals(sh.index):
        pd.testing.assert_series_equal(aligned.astype(float), raw.astype(float))
    else:
        assert float(aligned.iloc[0]) == float(raw)
