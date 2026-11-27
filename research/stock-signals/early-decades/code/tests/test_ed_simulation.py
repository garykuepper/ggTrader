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
    monkeypatch.setattr(
        ed_run,
        "simulate_arm",
        lambda *args: [{"arm": args[0], "cost": 0.00183, "dead_stock_loss": 0.0}],
    )
    monkeypatch.setattr(ed_run, "git_sha", lambda root: "test-commit")
    keys = [ed_run.study.cell_key(0.0, 0.00183, tax) for tax in ("low", "high")]
    cells = dict.fromkeys(keys, {"value": 1})
    first = tmp_path / "dead-stock-first-test"
    first.mkdir()
    (first / "_holdout-A.json").write_text(json.dumps({"result": {"strategy": cells}}))
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps({"start": ed_run.START, "end": ed_run.END, "matches": True, "cell": cells})
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
    assert data["comparison"]["early"]["label"] == "CRSP market return (stand-in)"
    assert data["comparison"]["later"]["label"] == "SPY"
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
