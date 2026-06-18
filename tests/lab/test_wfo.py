import numpy as np
import pandas as pd

from ggTrader.lab.strategy import LabConfig, SignalTargets
from ggTrader.lab.wfo import composite_score, generate_folds, run_wfo, select_live_params


def _ohlcv(symbols, n):
    """Build a (symbol, field) MultiIndex OHLCV DataFrame with trending prices."""
    idx = pd.date_range("2014-01-01", periods=n, freq="B", tz="UTC")
    frames = {}
    for i, s in enumerate(symbols):
        base = 100.0 + i * 10
        close = base * (1 + 0.0003 * np.arange(n))
        frames[s] = pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.005,
                "low": close * 0.995,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
    return pd.concat(frames, axis=1)


class _TinySignal:
    """Minimal signal strategy for testing: buy on bar 2, never exit."""

    name = "tiny"
    target_kind = "signals"

    def __init__(self, cfg):
        self.cfg = cfg

    @classmethod
    def sweep_params(cls):
        return {"param_a": [1, 2]}

    def sweep_signals(self, combos, symbols, data):
        from ggTrader.lab.sweep import combo_name

        close = pd.concat(
            {s: data[s]["close"] for s in symbols if s in data.columns.get_level_values(0)},
            axis=1,
        )
        result = {}
        for combo in combos:
            entries = pd.DataFrame(False, index=close.index, columns=close.columns)
            exits = pd.DataFrame(False, index=close.index, columns=close.columns)
            # Buy on bar 2 — after warmup
            entries.iloc[2] = True
            key = combo_name(self.name, combo)
            result[key] = SignalTargets(entries=entries.astype(bool), exits=exits.astype(bool))
        return result


def test_generate_folds_count_and_boundaries():
    """11-year span with 3yr train / 1yr test -> 8 folds, no overlap."""
    start = pd.Timestamp("2015-01-01", tz="UTC")
    end = pd.Timestamp("2026-01-01", tz="UTC")
    folds = generate_folds(start, end)
    assert len(folds) == 8
    for f in folds:
        assert f.test_start == f.train_end  # no gap
        train_years = (f.train_end - f.train_start).days / 365.25
        test_years = (f.test_end - f.test_start).days / 365.25
        assert abs(train_years - 3.0) < 0.05
        assert abs(test_years - 1.0) < 0.05
    # No fold's test_end exceeds data_end
    assert all(f.test_end <= end for f in folds)
    # Folds slide by 1 year
    for i in range(1, len(folds)):
        delta_years = (folds[i].train_start - folds[i - 1].train_start).days / 365.25
        assert abs(delta_years - 1.0) < 0.05


def test_generate_folds_short_data_returns_fewer():
    """Only 3.5 years of data -> not enough for train+test, returns 0 folds."""
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2023-06-01", tz="UTC")
    folds = generate_folds(start, end)
    assert len(folds) == 0


def test_generate_folds_exact_4_years():
    """Exactly 4 years -> 1 fold."""
    start = pd.Timestamp("2020-01-01", tz="UTC")
    end = pd.Timestamp("2024-01-01", tz="UTC")
    folds = generate_folds(start, end)
    assert len(folds) == 1
    assert folds[0].train_start == start
    assert folds[0].test_end == end


def test_composite_score_ranking():
    """Combo with best sharpe+sortino and least drawdown scores highest."""
    metrics = [
        {"sharpe": 1.0, "sortino": 1.2, "max_drawdown_pct": -10.0},
        {"sharpe": 0.5, "sortino": 0.6, "max_drawdown_pct": -20.0},
        {"sharpe": 0.2, "sortino": 0.3, "max_drawdown_pct": -30.0},
    ]
    scores = composite_score(metrics)
    assert len(scores) == 3
    assert scores[0] > scores[1] > scores[2]
    # Best combo should get close to 1.0 (all normalized to 1.0)
    # 0.5 * 1.0 + 0.3 * 1.0 - 0.2 * 0.0 = 0.8
    assert abs(scores[0] - 0.8) < 1e-9


def test_composite_score_single_combo():
    """Single combo: min==max for all metrics, all normalized to 0.0, score=0.0."""
    metrics = [{"sharpe": 0.5, "sortino": 0.6, "max_drawdown_pct": -15.0}]
    scores = composite_score(metrics)
    assert len(scores) == 1
    assert scores[0] == 0.0


def test_composite_score_nan_handling():
    """NaN sharpe/sortino treated as worst (floor of range)."""
    metrics = [
        {"sharpe": 1.0, "sortino": 1.0, "max_drawdown_pct": -10.0},
        {"sharpe": float("nan"), "sortino": float("nan"), "max_drawdown_pct": -20.0},
    ]
    scores = composite_score(metrics)
    assert scores[0] > scores[1]


def test_run_wfo_integration():
    """Full WFO with tiny strategy: 2014-2020 data, 3yr/1yr -> 3 folds."""
    symbols = ["X", "Y"]
    n = 252 * 7  # ~7 years of daily bars
    ohlcv = _ohlcv(symbols, n)
    spy_close = ohlcv["X"]["close"].copy()  # use X as SPY proxy
    cfg = LabConfig(top_n=10, lookback=20, skip=5, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.5,
    }
    eval_start = ohlcv.index[0]
    eval_end = ohlcv.index[-1]
    grid = [{"param_a": 1}, {"param_a": 2}]

    output = run_wfo(
        "tiny",
        _TinySignal,
        cfg,
        ohlcv,
        spy_close,
        str(eval_start.date()),
        str(eval_end.date()),
        "test",
        base_config,
        grid,
    )
    assert "WFO:" in output
    assert "OOS Aggregate:" in output
    assert "Recommended Live Params" in output
    # Should have at least 1 fold
    assert "Fold" in output


def test_select_live_params_uses_recent_window():
    """Live params trained on most recent 3yr; stability counts matching fold winners."""
    symbols = ["X", "Y"]
    n = 252 * 7
    ohlcv = _ohlcv(symbols, n)
    cfg = LabConfig(top_n=10, lookback=20, skip=5, min_history_bars=10)
    base_config = {
        "START_CASH": 10000.0,
        "FEES": 0.0,
        "SLIPPAGE": 0.0,
        "FREQ": "1d",
        "SIGNAL_POSITION_SIZE": 0.5,
    }
    grid = [{"param_a": 1}, {"param_a": 2}]
    # Simulate fold_winners that all picked param_a=1
    fold_winners = [
        {"combo": "tiny__param_a1", "params": {"param_a": 1}},
        {"combo": "tiny__param_a1", "params": {"param_a": 1}},
    ]
    eval_end = str(ohlcv.index[-1].date())
    result = select_live_params(
        "tiny",
        _TinySignal,
        cfg,
        ohlcv,
        eval_end,
        base_config,
        grid,
        fold_winners,
    )
    assert "combo" in result
    assert "stability" in result
    assert isinstance(result["stability"], int)
    assert "train_metrics" in result
    assert "sharpe" in result["train_metrics"]
