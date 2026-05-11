"""Tests for the final holdout reservation and warning logic.

Step 0 of the WFO textbook reset. The most recent HOLDOUT_FRACTION of bars is
locked away before WFO. After WFO + gates pass, median params are evaluated
on the holdout exactly once. A warning is raised if:
- holdout annualized return < 0, OR
- holdout max_dd > 1.5 * worst test-fold max_dd from WFO.
The holdout is NOT a gate. Numbers are always reported.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ggTrader.core.holdout import (
    split_train_holdout,
    holdout_warning_flags,
)


def _make_ohlcv(n_bars: int) -> pd.DataFrame:
    """Multi-symbol OHLCV with (symbol, field) columns matching project convention."""
    idx = pd.date_range("2023-01-01", periods=n_bars, freq="4h")
    cols = pd.MultiIndex.from_product(
        [["BTC-USD"], ["open", "high", "low", "close", "volume"]],
        names=["symbol", "field"],
    )
    data = np.random.randn(n_bars, len(cols)).cumsum(axis=0) + 100
    data = np.abs(data)  # keep positive
    return pd.DataFrame(data, index=idx, columns=cols)


def test_split_train_holdout_fraction_20_percent():
    """80% goes to WFO train, 20% goes to holdout."""
    ohlcv = _make_ohlcv(1000)
    train, holdout = split_train_holdout(ohlcv, holdout_fraction=0.20)
    assert len(train) == 800
    assert len(holdout) == 200
    # Train precedes holdout chronologically (no overlap, no gap).
    assert train.index[-1] < holdout.index[0]
    # Concatenation must reconstruct the original.
    reassembled = pd.concat([train, holdout])
    pd.testing.assert_frame_equal(reassembled, ohlcv)


def test_split_train_holdout_disabled_at_fraction_zero():
    """HOLDOUT_FRACTION=0 returns the original OHLCV and an empty holdout."""
    ohlcv = _make_ohlcv(500)
    train, holdout = split_train_holdout(ohlcv, holdout_fraction=0.0)
    assert len(train) == 500
    assert len(holdout) == 0


def test_warning_flag_negative_return():
    """Annualized return < 0 triggers the negative-return warning."""
    flags = holdout_warning_flags(
        holdout_ann_return=-0.05,
        holdout_max_dd=-0.10,
        worst_wfo_test_dd=-0.20,
    )
    assert "negative_return" in flags


def test_warning_flag_max_dd_exceeds_threshold():
    """Holdout DD worse than 1.5x WFO worst-test-DD triggers the DD warning."""
    # WFO worst test DD: -20%. Threshold = 1.5 * 20% = 30%. Holdout DD of -35% triggers.
    flags = holdout_warning_flags(
        holdout_ann_return=0.10,  # positive, so no return-flag
        holdout_max_dd=-0.35,
        worst_wfo_test_dd=-0.20,
    )
    assert "max_dd_exceeds_threshold" in flags
    assert "negative_return" not in flags


def test_no_warning_when_all_good():
    """Positive return AND DD within 1.5x → no warnings."""
    flags = holdout_warning_flags(
        holdout_ann_return=0.20,
        holdout_max_dd=-0.15,
        worst_wfo_test_dd=-0.20,
    )
    assert flags == []


def test_no_warning_when_worst_wfo_test_dd_is_zero():
    """A WFO baseline of 0 DD should not trigger a false-positive max_dd flag."""
    # If WFO had perfect no-drawdown folds (worst_wfo_test_dd = 0), the 1.5x
    # threshold becomes 0 and would fire on any non-zero holdout DD. The
    # warning function must not fire in this degenerate case.
    flags = holdout_warning_flags(
        holdout_ann_return=0.10,  # positive, no return flag
        holdout_max_dd=-0.05,     # small holdout DD
        worst_wfo_test_dd=0.0,    # degenerate WFO baseline
    )
    assert "max_dd_exceeds_threshold" not in flags


def test_no_warning_when_worst_wfo_test_dd_is_nan():
    """A NaN WFO baseline should skip the DD check rather than masking it."""
    import math
    flags = holdout_warning_flags(
        holdout_ann_return=0.10,
        holdout_max_dd=-0.30,
        worst_wfo_test_dd=math.nan,
    )
    assert "max_dd_exceeds_threshold" not in flags
