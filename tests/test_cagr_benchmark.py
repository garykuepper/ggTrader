"""Tests for CAGR and equal-weight benchmark helpers in orchestrator."""

import math

import numpy as np
import pandas as pd
import pytest

from ggTrader.core.orchestrator import (
    _cagr_percent,
    _equal_weight_buy_hold_portfolio_stats,
    _years_from_price_index,
)


def test_cagr_percent_matches_geometric_mean() -> None:
    """13.72% over 3 calendar years -> ~4.4% CAGR."""
    cagr = _cagr_percent(13.72, 3.0)
    expected = (1.1372 ** (1.0 / 3.0) - 1.0) * 100.0
    assert cagr == pytest.approx(expected, rel=1e-6)


def test_cagr_percent_invalid_years() -> None:
    assert math.isnan(_cagr_percent(10.0, 0.0))
    assert math.isnan(_cagr_percent(10.0, float("nan")))


def test_years_from_price_index() -> None:
    idx = pd.date_range("2023-01-01", periods=10, freq="4h")
    y = _years_from_price_index(idx)
    assert y > 0
    assert math.isfinite(y)


def test_equal_weight_buy_hold_produces_trades() -> None:
    """Smoke test: vectorbt portfolio builds and returns finite metrics."""
    idx = pd.date_range("2023-01-01", periods=80, freq="D")
    close = pd.DataFrame(
        {
            "AAA": np.linspace(100.0, 108.0, 80),
            "BBB": np.linspace(50.0, 52.0, 80),
        },
        index=idx,
    )
    config = {
        "START_CASH": 1000.0,
        "FEES": 0.001,
        "SLIPPAGE": 0.0,
        "FREQ": "1D",
    }
    out = _equal_weight_buy_hold_portfolio_stats(close, config)
    assert out["total_trades"] >= 2
    assert out["profit_pct"] is not None
    assert out["cagr_pct"] is not None
