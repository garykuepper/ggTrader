"""Tests for WFO train-gate trade count alignment (MultiIndex columns vs metric index)."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.orchestrator import _align_grouped_combo_series, _trade_counts_for_train_gate


class _MockTrades:
    def __init__(self, count_series: pd.Series) -> None:
        self._count_series = count_series

    def count(self) -> pd.Series:
        return self._count_series


class _MockWrapper:
    def __init__(self, columns: pd.Index) -> None:
        self.columns = columns


class _MockPF:
    """Minimal portfolio stand-in for _trade_counts_for_train_gate."""

    def __init__(self, count_series: pd.Series, columns: pd.Index) -> None:
        self.trades = _MockTrades(count_series)
        self.wrapper = _MockWrapper(columns)


def test_trade_counts_exact_index_match() -> None:
    idx = pd.MultiIndex.from_tuples([(0, "x"), (0, "y")], names=["c", "sym"])
    sharpe = pd.Series([0.1, -0.2], index=idx)
    raw = pd.Series([3.0, 7.0], index=idx)
    pf = _MockPF(raw, idx)
    out = _trade_counts_for_train_gate(pf, sharpe)
    assert len(out) == 2
    np.testing.assert_array_almost_equal(out.values, [3.0, 7.0])


def test_trade_counts_groupby_multiindex_columns() -> None:
    """Two param combos × one symbol column each: counts sum per combo level."""
    cols = pd.MultiIndex.from_tuples([(0, "S"), (1, "S")], names=["combo", "symbol"])
    sharpe = pd.MultiIndex.from_tuples([(0,), (1,)], names=["combo"])
    sh = pd.Series([0.5, -0.1], index=sharpe)
    raw = pd.Series([4.0, 2.0], index=cols)
    pf = _MockPF(raw, cols)
    out = _trade_counts_for_train_gate(pf, sh)
    assert list(out.index) == [(0,), (1,)]
    np.testing.assert_array_almost_equal(out.values, [4.0, 2.0])


def test_align_grouped_combo_scalar_index_to_metric_multiindex() -> None:
    agg = pd.Series([9.0, 1.0], index=pd.Index([0, 1], dtype=np.int64))
    sh_index = pd.MultiIndex.from_tuples([(0,), (1,)], names=["combo"])
    out = _align_grouped_combo_series(agg, sh_index)
    np.testing.assert_array_almost_equal(out.values, [9.0, 1.0])


def test_trade_counts_groupby_two_symbols_per_combo() -> None:
    cols = pd.MultiIndex.from_tuples(
        [(0, "A"), (0, "B"), (1, "A"), (1, "B")], names=["combo", "symbol"]
    )
    sh_idx = pd.MultiIndex.from_tuples([(0,), (1,)], names=["combo"])
    sh = pd.Series([1.0, 2.0], index=sh_idx)
    raw = pd.Series([2.0, 1.0, 0.0, 4.0], index=cols)
    pf = _MockPF(raw, cols)
    out = _trade_counts_for_train_gate(pf, sh)
    np.testing.assert_array_almost_equal(out.values, [3.0, 4.0])
