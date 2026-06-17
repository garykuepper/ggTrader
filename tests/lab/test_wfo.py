import pandas as pd

from ggTrader.lab.wfo import composite_score, generate_folds


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
