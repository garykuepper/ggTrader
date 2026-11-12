"""Check the descriptive portfolio arithmetic on invented prices."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "descriptive_protection.py"  # root-ok: sibling code
spec = importlib.util.spec_from_file_location("descriptive_protection", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pytestmark = pytest.mark.study


def test_buy_and_hold_and_unrecovered_drawdown():
    prices = pd.DataFrame(
        {"SPY": [100, 80, 90]}, index=pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"])
    )
    curve = module.portfolio(prices, {"SPY": 1.0}, 0.0)
    assert curve.tolist() == pytest.approx([1.0, 0.8, 0.9])
    result = module.summarize(curve)
    assert result["max_drawdown"] == pytest.approx(-0.2)
    assert result["recovered"] is False
    assert result["underwater_sessions"] == 2


def test_month_end_rebalance_charges_turnover():
    prices = pd.DataFrame(
        {"A": [100, 200, 200], "B": [100, 100, 100]},
        index=pd.to_datetime(["2021-01-04", "2021-01-29", "2021-02-01"]),
    )
    curve = module.portfolio(prices, {"A": 0.5, "B": 0.5}, 0.001)
    assert curve.iloc[-1] == pytest.approx(1.4980005)


def test_matched_episode_recovers_and_censors():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "matched_protection.py"  # root-ok: sibling code
    spec = importlib.util.spec_from_file_location("matched_protection", path)
    matched = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matched)
    dates = pd.date_range("2022-01-01", periods=5)
    curves = {
        "spy": pd.Series([1.0, 0.8, 0.6, 0.9, 1.0], index=dates),
        "candidate": pd.Series([1.0, 0.9, 0.8, 1.0, 1.1], index=dates),
        "unfinished": pd.Series([1.0, 0.9, 0.8, 0.85, 0.9], index=dates),
    }
    result = matched.matched_episode(curves)
    assert result["candidate"]["return_at_spy_trough"] == pytest.approx(-0.2)
    assert result["candidate"]["recovered"] is True
    assert result["candidate"]["sessions"] == 3
    assert result["unfinished"]["recovered"] is False
    assert result["unfinished"]["sessions"] == 4
