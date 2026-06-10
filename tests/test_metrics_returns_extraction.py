"""Regression guard for the single-extraction train-metric path.

`_returns_based_metrics` replaced three independent vbt accessor calls
(`pf.sortino_ratio()` / `pf.total_return()` / `pf.max_drawdown()`) with one `pf.returns()`
extraction fed to vbt's own numba kernels (see docs/profiling_report_2026-06-05.md). These
tests assert the optimization is **bit-identical** to the accessors it replaced, including
the inf/NaN edge cases that drive WFO param selection.
"""

import os
import sys

import numpy as np
import pandas as pd
import vectorbt as vbt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.metrics import (  # noqa: E402
    _calmar_ratio_series,
    _fold_stats_metrics,
    _returns_based_metrics,
    _train_metric_series,
)


def _make_pf():
    """Multi-column portfolio with deliberate edge cases: a monotonic winner (mdd=0,
    sortino=+inf), a no-trade column (sortino=+inf, pf=NaN), a noisy mix, a steady loser."""
    idx = pd.date_range("2024-01-01", periods=400, freq="4h", tz="UTC")
    rng = np.random.default_rng(3)
    prices = pd.DataFrame(
        {
            "win": 100 * np.cumprod(1 + np.full(400, 0.001)),
            "none": 100 + np.zeros(400),
            "mix": 100 * np.cumprod(1 + rng.normal(0, 0.012, 400)),
            "lose": 100 * np.cumprod(1 + np.full(400, -0.0006)),
        },
        index=idx,
    )
    entries = pd.DataFrame(False, index=idx, columns=prices.columns)
    exits = entries.copy()
    entries.iloc[10] = True
    exits.iloc[380] = True
    entries.loc[:, "none"] = False  # no-trade column
    return vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=1000.0,
        fees=2e-4,
        slippage=3e-3,
        freq="4h",
        size=0.3,
        size_type="percent",
    ).copy()


def _eq(a, b):
    return np.allclose(
        np.asarray(a, float), np.asarray(b, float), rtol=1e-9, atol=1e-12, equal_nan=True
    )


def test_returns_based_metrics_match_vbt_accessors():
    pf = _make_pf()
    sortino, total_ret, max_dd = _returns_based_metrics(pf)
    ref_so = pf.sortino_ratio().values
    assert _eq(sortino.values, ref_so)
    assert _eq(total_ret.values, pf.total_return().values)
    assert _eq(max_dd.values, pf.max_drawdown().values)
    # Non-finite structure must be preserved exactly (inf stays inf, NaN stays NaN —
    # never silently coerced, since the composite's .notna() gate depends on it).
    assert np.array_equal(np.isinf(sortino.values), np.isinf(ref_so))
    assert np.array_equal(np.isnan(sortino.values), np.isnan(ref_so))


def test_fold_stats_metrics_match_vbt_accessors():
    """The per-fold OOS/train diagnostic helper (wfo._process_wfo_fold) must reproduce
    pf.sharpe_ratio()/sortino_ratio()/total_return()/max_drawdown() and the reductions
    used downstream (.mean()/.min()/.max()), which feed the aggregate gates."""
    pf = _make_pf()
    m = _fold_stats_metrics(pf)
    assert _eq(m["sharpe"].values, pf.sharpe_ratio().values)
    assert _eq(m["sortino"].values, pf.sortino_ratio().values)
    assert _eq(m["total_return"].values, pf.total_return().values)
    assert _eq(m["max_drawdown"].values, pf.max_drawdown().values)
    # the reductions wfo.py applies must match too
    assert _eq([m["sharpe"].mean()], [pf.sharpe_ratio().mean()])
    assert _eq([m["sortino"].mean()], [pf.sortino_ratio().mean()])
    assert _eq([m["total_return"].mean()], [pf.total_return().mean()])
    assert _eq([m["max_drawdown"].min()], [pf.max_drawdown().min()])
    assert _eq([m["total_return"].max()], [pf.total_return().max()])
    # returned frame is exactly pf.returns() (stored downstream as oos_returns)
    assert _eq(m["returns"].values, pf.returns().values)


def test_calmar_matches_accessor_derivation():
    pf = _make_pf()
    _, total_ret, max_dd = _returns_based_metrics(pf)
    new = _calmar_ratio_series(pf, tr=total_ret, mdd=max_dd)
    ref = pf.total_return() / pf.max_drawdown().abs().replace(0, np.nan)
    assert _eq(new.values, ref.values)
    # uncomputed path (no tr/mdd supplied) must agree too
    assert _eq(_calmar_ratio_series(pf).values, ref.values)


def _native_profit_factor_reference(trades) -> pd.Series:
    """vbt's Trades.profit_factor() formula on writable copies — the real accessor
    mutates in place and crashes on the read-only views vbt can return (the bug
    metrics._profit_factor_raw exists to avoid)."""
    total_win = np.array(np.atleast_1d(np.asarray(trades.winning.pnl.sum())), dtype=float)
    total_loss = np.array(np.atleast_1d(np.asarray(trades.losing.pnl.sum())), dtype=float)
    has_values = np.atleast_1d(np.asarray(trades.count())) > 0
    total_win[np.isnan(total_win) & has_values] = 0.0
    total_loss[np.isnan(total_loss) & has_values] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        result = total_win / np.abs(total_loss)
    return pd.Series(result, index=trades.wrapper.grouper.get_columns())


def test_composite_train_metric_identical_to_raw_accessor_reference():
    pf = _make_pf()
    # Reference composite built from RAW vbt accessors + the documented rank logic.
    so = pf.sortino_ratio()
    ca = (pf.total_return() / pf.max_drawdown().abs().replace(0, np.nan)).reindex(so.index)
    pf_s = ((_native_profit_factor_reference(pf.trades) - 1.0).clip(-3, 3)).reindex(so.index)
    mean_rank = pd.concat(
        [
            so.rank(ascending=False, method="average"),
            ca.clip(-5, 5).rank(ascending=False, method="average"),
            pf_s.clip(-3, 3).rank(ascending=False, method="average"),
        ],
        axis=1,
    ).mean(axis=1, skipna=False)
    ref = (-mean_rank).where(so.notna(), other=float("nan"))

    got = _train_metric_series(pf, {"TRAIN_METRIC": "composite"})
    assert _eq(got.values, ref.values)
