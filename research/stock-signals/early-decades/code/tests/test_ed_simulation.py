"""Exercise the early-window ledger and output with synthetic prices only."""

import argparse
import json

import ed_run
import pandas as pd
import pytest
from ed_simulation import simulate_arm, validate_calendar

pytestmark = pytest.mark.study


@pytest.fixture
def fake_inputs() -> dict:
    """A small universe with one prior jump and a later rising stock."""
    dates = pd.bdate_range("2019-01-01", "2021-03-31")
    close = pd.DataFrame({"A": 10.0, "B": 10.0}, index=dates)
    close.loc["2019-12-02":, "A"] = 20.0
    return {
        "close": close,
        "eligible": pd.DataFrame(True, index=dates, columns=close.columns),
        "span_rows": pd.DataFrame(columns=["symbol", "last_date", "has_successor", "total_loss"]),
        "provenance": {"test": True},
    }


@pytest.mark.parametrize("arm", ["B", "C"])
def test_arm_runs_six_ledger_cells(arm: str, fake_inputs: dict) -> None:
    cells = simulate_arm(arm, fake_inputs, "2020-01-01", "2021-03-31", top_n=1, lookback=60, skip=5)
    assert {(cell["cost"], cell["dead_stock_loss"]) for cell in cells} == {
        (cost, loss) for cost in (0.00183, 0.005) for loss in (0.0, 0.5, 1.0)
    }
    assert all(cell["arm"] == arm and cell["chain_verified"] for cell in cells)
    assert all(cell["n_trades"] > 0 for cell in cells)
    assert all(
        "coverage" in cell
        and "holdings_audit" in cell
        and "trades" in cell
        and "one_permno_one_column" in cell
        for cell in cells
    )


def test_comparison_calendar_detects_missing_day() -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    with pytest.raises(ValueError, match="1 comparison dates missing"):
        validate_calendar(
            dates, pd.Series([0.0, 0.0], index=dates[[0, 2]]), "2020-01-02", "2020-01-06"
        )


def test_early_driver_writes_metadata_and_cells(monkeypatch, tmp_path, fake_inputs) -> None:
    dates = pd.to_datetime(["1969-02-03", "1993-01-29", "1993-02-01", "1998-07-31"])
    fake_inputs["close"] = pd.DataFrame({"A": [1, 1, 1, 1]}, index=dates)
    returns = pd.Series([0.0, 0.01, 0.02, 0.03], index=dates)
    monkeypatch.setattr(ed_run.ed_inputs, "read_early_inputs", lambda *args: fake_inputs)
    monkeypatch.setattr(ed_run, "comparison_returns", lambda *args: returns)
    monkeypatch.setattr(ed_run, "validate_calendar", lambda *args: None)
    monkeypatch.setattr(ed_run, "preflight", lambda: None)
    monkeypatch.setattr(
        ed_run,
        "simulate_arm",
        lambda *args: [
            {
                "arm": args[0],
                "cost": 0.00183,
                "dead_stock_loss": 0.0,
                "tax": "low",
                "after_tax_final": 100.0,
                "account_value_daily": [[str(day.date()), 100.0] for day in dates],
            }
        ],
    )
    monkeypatch.setattr(ed_run, "git_sha", lambda root: "test-commit")
    keys = [
        ed_run.study.cell_key(loss, cost, tax)
        for loss in ed_run.study.LOSSES
        for cost in ed_run.study.COSTS
        for tax in ed_run.study.TAXES
    ]
    cells = dict.fromkeys(keys, {"value": 1})
    first = tmp_path / "dead-stock-first-test"
    first.mkdir()
    (first / "_holdout-A.json").write_text(json.dumps({"result": {"strategy": cells}}))
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            {
                "start": ed_run.START,
                "end": ed_run.END,
                "matches": True,
                "cell": cells,
                "code_commit": "test-commit",
                "rule_hash": "a" * 64,
                "clean": True,
            }
        )
    )
    (tmp_path / "early-decades").mkdir()
    monkeypatch.setattr(ed_run, "study_dir", lambda _: tmp_path / "early-decades")
    monkeypatch.setattr(ed_run, "rule_hash", lambda: "a" * 64)
    out = tmp_path / "result.json"
    ed_run.run_early(
        argparse.Namespace(
            arm="C",
            eval_start=ed_run.EARLY_START,
            eval_end=ed_run.EARLY_END,
            out=out,
            regression_proof=proof,
        )
    )
    data = json.loads(out.read_text())
    assert data["code_commit"] == "test-commit" and data["release_id"] == 1
    assert data["arm"] == "C" and len(data["rule_hash"]) == 64
    assert set(data["cells"][0]["periods"]) == {"early", "later", "combined"}
    assert data["comparison"]["early"]["label"] == "CRSP market return (stand-in)"
    assert data["comparison"]["later"]["label"] == "SPY"
    assert "max_drawdown" in data["comparison"]["later"]["metrics"]
    with pytest.raises(SystemExit, match="overwrite"):
        ed_run.run_early(
            argparse.Namespace(
                arm="C",
                eval_start=ed_run.EARLY_START,
                eval_end=ed_run.EARLY_END,
                out=out,
                regression_proof=proof,
            )
        )


def test_early_requires_matching_regression_before_input(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(ed_run.ed_inputs, "read_early_inputs", lambda *args: pytest.fail("read"))
    args = argparse.Namespace(
        arm="B",
        eval_start=ed_run.EARLY_START,
        eval_end=ed_run.EARLY_END,
        out=tmp_path / "out.json",
        regression_proof=None,
    )
    with pytest.raises(SystemExit, match="regression output is required"):
        ed_run.run_early(args)


def test_cell_exposes_realized_gains_and_subperiod_curves(fake_inputs: dict) -> None:
    cells = simulate_arm("B", fake_inputs, "2020-01-01", "2021-03-31", top_n=1, lookback=60, skip=5)
    cell = cells[0]
    assert set(cell["realized_gains"]) == {"short_term", "long_term"}
    assert "account_value_daily" in cell
    assert cell["account_value_daily"][0][0] == "2020-01-01"


def test_preflight_refuses_dirty_tree_before_drift(monkeypatch) -> None:
    monkeypatch.setattr(ed_run, "git_dirty", lambda root: True, raising=False)
    monkeypatch.setattr(
        ed_run, "read_linked_drift", lambda: pytest.fail("drift checked"), raising=False
    )
    with pytest.raises(SystemExit, match="dirty"):
        ed_run.preflight()


def test_early_rejects_proof_from_different_commit_before_claim(monkeypatch, tmp_path) -> None:
    study = tmp_path / "early-decades"
    first = tmp_path / "dead-stock-first-test"
    study.mkdir()
    first.mkdir()
    keys = [
        ed_run.study.cell_key(loss, cost, tax)
        for loss in ed_run.study.LOSSES
        for cost in ed_run.study.COSTS
        for tax in ed_run.study.TAXES
    ]
    cells = dict.fromkeys(keys, {"x": 1})
    (first / "_holdout-A.json").write_text(json.dumps({"result": {"strategy": cells}}))
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            {
                "start": ed_run.START,
                "end": ed_run.END,
                "matches": True,
                "cell": cells,
                "code_commit": "old",
                "rule_hash": "old",
                "clean": True,
            }
        )
    )
    monkeypatch.setattr(ed_run, "study_dir", lambda _: study)
    monkeypatch.setattr(ed_run, "git_sha", lambda _: "new")
    monkeypatch.setattr(ed_run, "rule_hash", lambda: "new")
    monkeypatch.setattr(ed_run, "preflight", lambda: None)
    monkeypatch.setattr(ed_run.ed_inputs, "read_early_inputs", lambda *args: pytest.fail("read"))
    args = argparse.Namespace(
        arm="A",
        eval_start=ed_run.EARLY_START,
        eval_end=ed_run.EARLY_END,
        out=tmp_path / "out.json",
        regression_proof=proof,
    )
    with pytest.raises(SystemExit, match="proof"):
        ed_run.run_early(args)
    assert not (study / "_early-A.claim").exists()


def test_period_metrics_measure_growth_drawdown_and_sharpe() -> None:
    from ed_simulation import period_metrics

    dates = pd.bdate_range("2020-01-02", periods=4)
    curve = pd.Series([100.0, 110.0, 99.0, 121.0], index=dates)
    result = period_metrics(curve, str(dates[1].date()), str(dates[-1].date()))
    assert result["growth"] == pytest.approx(1.21)
    assert result["max_drawdown"] == pytest.approx(-0.1)
    assert result["sharpe"] is not None
    assert result["cagr"] is not None


def test_loss_monotone_finds_increasing_value() -> None:
    from ed_simulation import loss_monotone

    cells = [
        {"cost": 0.00183, "tax": "low", "dead_stock_loss": loss, "after_tax_final": value}
        for loss, value in [(0, 100), (0.5, 80), (1, 90)]
    ]
    assert len(loss_monotone(cells)) == 1


def test_period_metrics_preserves_total_loss() -> None:
    from ed_simulation import period_metrics

    curve = pd.Series([100.0, 0.0], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    result = period_metrics(curve, "2020-01-02", "2020-01-03")
    assert result["growth"] == 0.0
    assert result["max_drawdown"] == -1.0


def test_preflight_refuses_linked_yahoo_drift(monkeypatch) -> None:
    monkeypatch.setattr(ed_run, "git_dirty", lambda _: False)
    monkeypatch.setattr(ed_run, "read_linked_drift", lambda: ["DRIFTED"])
    with pytest.raises(SystemExit, match="drift"):
        ed_run.preflight()
