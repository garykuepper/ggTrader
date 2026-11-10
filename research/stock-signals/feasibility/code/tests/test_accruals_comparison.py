"""Check comparison arithmetic and reject incomplete synthetic data."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "accruals_comparison.py"
spec = importlib.util.spec_from_file_location("accruals_comparison", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pytestmark = pytest.mark.study


def test_compounds_and_includes_initial_wealth_in_drawdown():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    result = module.compare(
        pd.Series([-0.2, 0.25, 0.0], index=dates), pd.Series([0.0] * 3, index=dates)
    )
    assert result["candidate"]["terminal_wealth_per_1"] == pytest.approx(1.0)
    assert result["candidate"]["max_month_end_drawdown"] == pytest.approx(-0.2)


def test_rejects_missing_month_and_mismatched_period():
    dates = pd.to_datetime(["2020-01-31", "2020-03-31"])
    s = pd.Series([0.01, 0.02], index=dates)
    with pytest.raises(ValueError, match="continuous"):
        module.compare(s, s)
    with pytest.raises(ValueError, match="share"):
        module.compare(s, s.iloc[:1])


def test_cost_adjustment_compounds_and_reports_undefined_statistic():
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    candidate = pd.Series([0.01] * 24, index=dates)
    benchmark = pd.Series([0.0] * 24, index=dates)
    result = module.screen(candidate, benchmark, annual_cost=0.012)
    assert result["comparison"]["candidate"]["terminal_wealth_per_1"] == pytest.approx(1.009**24)
    assert result["annual_cost_assumption"] == 0.012
    assert result["mean_excess_t"] is None


def test_rejects_costs_that_produce_invalid_wealth():
    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    candidate = pd.Series([-0.9999] * 24, index=dates)
    with pytest.raises(ValueError, match="positive wealth"):
        module.screen(candidate, pd.Series([0.0] * 24, index=dates), annual_cost=0.012)


def test_fixed_windows_fail_closed_if_a_month_is_missing():
    dates = pd.date_range("1963-07-31", "2026-04-30", freq="ME")
    candidate = pd.Series([0.01] * len(dates), index=dates)
    market = pd.Series([0.005] * len(dates), index=dates)
    result = module.evaluate(candidate, market)
    assert result["full"]["comparison"]["months"] == 754
    assert result["since_1994"]["comparison"]["months"] == 388
    assert result["full_stress"]["annual_cost_assumption"] == 0.02
    candidate = candidate.drop(dates[30])
    with pytest.raises(ValueError):
        module.evaluate(candidate, market)


def test_spy_monthly_requires_a_starting_month_and_complete_prices():
    dates = pd.to_datetime(["2021-04-30", "2021-05-28", "2021-06-30"])
    monthly = module.monthly_returns(pd.Series([100.0, 110.0, 99.0], index=dates))
    assert list(monthly) == pytest.approx([0.1, -0.1])
    with pytest.raises(ValueError, match="Missing"):
        module.monthly_returns(pd.Series([100.0, 99.0], index=dates[[0, 2]]))


def test_driver_writes_synthetic_results_once_and_checks_spy_hash(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys

    code = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(code))
    spec = importlib.util.spec_from_file_location("screen_driver", code / "run_accruals_screen.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    dates = pd.date_range("1963-07-31", "2026-04-30", freq="ME")
    fake_table = pd.DataFrame({"BIG LoAC": [0.01] * len(dates)}, index=dates)
    monkeypatch.setattr(
        driver, "verify_manifest", lambda _: {k: {"sha256": v} for k, v in driver.EXPECTED.items()}
    )
    monkeypatch.setattr(driver, "_read_frozen_csv", lambda *_: "synthetic")
    monkeypatch.setattr(driver, "parse_french_monthly", lambda *_: fake_table)
    monkeypatch.setattr(
        driver, "load_market_total_return", lambda _: pd.DataFrame({"mkt": 0.005}, index=dates)
    )
    spy = tmp_path / "SPY.csv"
    spy_dates = pd.date_range("2021-04-30", "2026-04-30", freq="ME")
    pd.DataFrame(
        {
            "date": spy_dates.tz_localize("America/New_York"),
            "Adj Close": range(100, 100 + len(spy_dates)),
        }
    ).to_csv(spy, index=False)
    monkeypatch.setattr(driver, "SPY_HASH", hashlib.sha256(spy.read_bytes()).hexdigest())
    out = tmp_path / "result.json"
    monkeypatch.setattr(
        sys, "argv", ["screen", "--data", str(tmp_path), "--spy", str(spy), "--out", str(out)]
    )
    driver.main()
    result = json.loads(out.read_text())
    assert result["results"]["full"]["comparison"]["months"] == 754
    assert result["spy_overlap"]["comparison"]["comparison"]["months"] == 60
    with pytest.raises(FileExistsError):
        driver.main()
    out.unlink()
    spy.write_text("changed")
    with pytest.raises(ValueError, match="SPY input hash"):
        driver.main()
