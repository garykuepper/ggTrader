"""Pure checks for the registered daily-bar audit driver."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.significance import ExcessStats
from scripts import production_replica_audit as audit


def _results(primary_pass: bool = True, holdout_pass: bool = True) -> dict:
    def window(passed: bool) -> dict:
        stats = ExcessStats(1000, 3.1 if passed else 0.5, 2.0, 1.0, 2.0, 1.1 if passed else -1.0)
        return {
            "arms": {"primary": {"excess_vs_spy": asdict(stats)}},
            "stop_flags": [],
            "mean_coverage": 0.95,
        }

    return {"windows": {"primary": window(primary_pass), "holdout": window(holdout_pass)}}


def test_verdict_requires_both_windows():
    assert audit.verdict(_results(holdout_pass=False))["verdict"] == "NO-GO"
    assert audit.verdict(_results())["stage"] == "pretax_pass"
    results = _results()
    results["windows"]["holdout"]["mean_coverage"] = 0.65
    assert audit.verdict(results)["verdict"] == "INCONCLUSIVE"
    results = _results(primary_pass=False)
    results["windows"]["holdout"]["mean_coverage"] = 0.65
    assert audit.verdict(results)["verdict"] == "NO-GO"
    results["windows"]["primary"]["stop_flags"] = ["tainted_trade"]
    assert audit.verdict(results)["stage"] == "stopped"
    assert audit.verdict(results)["verdict"] == "INCONCLUSIVE"


@pytest.mark.parametrize("mean_coverage", [None, "MISSING"])
def test_verdict_holdout_incomplete_when_coverage_missing_or_none(mean_coverage):
    results = _results(primary_pass=True)
    holdout: dict = {"arms": {}, "stop_flags": []}
    if mean_coverage != "MISSING":
        holdout["mean_coverage"] = mean_coverage
    results["windows"]["holdout"] = holdout
    assert audit.verdict(results)["reasons"] == ["holdout run incomplete"]


def test_tbill_daily_converts_annual_percent(monkeypatch):
    index = pd.bdate_range("2024-01-01", periods=10, tz="UTC")
    monkeypatch.setattr(
        audit,
        "load_fred_series",
        lambda *args: pd.DataFrame({"date": [pd.Timestamp("2023-12-01")], "value": [4.0]}),
    )
    rates = audit.tbill_daily(index)
    np.testing.assert_allclose(rates, 1.04 ** (1 / 252) - 1, atol=1e-12)


def test_exposure_matched_all_cash_earns_cash_rate():
    index = pd.bdate_range("2024-01-01", periods=5, tz="UTC")
    exposure = pd.DataFrame({"stocks": 0.0, "spy": 0.0, "cash": 1.0}, index=index)
    rate = pd.Series(0.001, index=index)
    spy = pd.Series([100, 110, 90, 120, 80], index=index)
    curve = audit.exposure_matched(spy, exposure, rate, 1000.0)
    np.testing.assert_allclose(curve, 1000.0 * 1.001 ** np.arange(5))


def test_exposure_matched_uses_prior_weights():
    index = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    exposure = pd.DataFrame({"stocks": [0, 1, 0], "spy": 0.0, "cash": [1, 0, 1]}, index=index)
    curve = audit.exposure_matched(
        pd.Series([100, 110, 121], index=index), exposure, pd.Series(0.0, index=index), 1000
    )
    np.testing.assert_allclose(curve, [1000, 1000, 1100])


def test_seam_scan_does_not_fill_missing_prices():
    index = pd.bdate_range("2024-01-01", periods=5, tz="UTC")
    close = pd.DataFrame({"AAA": [100, 50, np.nan, 10, 11]}, index=index)
    flags = audit.seam_scan(close, 0.40)
    assert len(flags) == 1
    assert flags[0]["symbol"] == "AAA"
    assert flags[0]["return"] == pytest.approx(-0.5)


def test_coverage_counts_actual_point_in_time_members(monkeypatch):
    index = pd.bdate_range("2024-01-01", periods=2, tz="UTC")
    close = pd.DataFrame({"AAA": [10.0, 10.0], "BBB": [np.nan, 20.0]}, index=index)
    monkeypatch.setattr(audit, "universe_members_asof", lambda universe, date: ["AAA", "BBB"])
    assert audit.coverage(close, list(index)) == {2024: 0.75}


def test_taint_check_uses_five_bars_of_same_symbol():
    index = pd.bdate_range("2024-01-01", periods=15, tz="UTC")
    seams = [{"symbol": "AAA", "date": str(index[5]), "artifact": True}]
    trades = pd.DataFrame(
        {"date": [index[10], index[11], index[5]], "symbol": ["AAA", "AAA", "BBB"]}
    )
    matches = audit.tainted_trades(trades, seams, index)
    assert len(matches) == 1
    assert matches[0]["date"] == str(index[10])


def test_fresh_seam_comparison_checkpoints_plain_json(monkeypatch, tmp_path):
    index = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    close = pd.DataFrame({"AAA": [100.0, 50.0, 51.0]}, index=index)
    monkeypatch.setattr(
        audit.yf,
        "download",
        lambda *args, **kwargs: pd.DataFrame({"Close": close.AAA.to_numpy()}, index=index),
    )
    results = {"windows": {"primary": {"arms": {}, "stop_flags": []}}}
    path = tmp_path / "audit.json"
    audit._verify_seams(close, results["windows"]["primary"], results, path)
    saved = json.loads(path.read_text())
    assert saved["windows"]["primary"]["seam_scan"][0]["artifact"] is False


def test_checkpoint_honors_global_data_stop_over_primary_failure(tmp_path):
    results = _results(primary_pass=False)
    results["windows"]["holdout"]["stop_flags"] = ["holdout_coverage_below_70pct"]
    path = tmp_path / "audit.json"
    audit._checkpoint(path, results)
    assert json.loads(path.read_text())["verdict"]["stage"] == "stopped"
