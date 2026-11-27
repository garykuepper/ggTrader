"""Check the regression path and proof using synthetic data only."""

import json
import sys
from types import SimpleNamespace

import ed_run
import pandas as pd
import pytest

pytestmark = pytest.mark.study


def test_regression_compares_all_cells_through_early_path(monkeypatch, tmp_path) -> None:
    cells = []
    saved = {}
    for loss in ed_run.study.LOSSES:
        for cost in ed_run.study.COSTS:
            for tax in ed_run.study.TAXES:
                key = ed_run.study.cell_key(loss, cost, tax)
                cells.append(
                    {"dead_stock_loss": loss, "cost": cost, "tax": tax, "cagr_after_tax": 0.1}
                )
                saved[key] = {
                    "loss": loss,
                    "one_way_cost": cost,
                    "tax": tax,
                    "cagr_after_tax": 0.1,
                    "gap_cagr_after_tax_pp": 10.0,
                }
    first = tmp_path / "dead-stock-first-test"
    first.mkdir()
    (first / "_holdout-A.json").write_text(json.dumps({"result": {"strategy": saved}}))
    monkeypatch.setattr(ed_run, "study_dir", lambda _: tmp_path / "early-decades")
    monkeypatch.setattr(ed_run, "git_dirty", lambda _: False)
    monkeypatch.setattr(ed_run, "git_sha", lambda _: "head")
    monkeypatch.setattr(ed_run, "rule_hash", lambda: "hash")
    monkeypatch.setattr(ed_run.ed_inputs, "read_early_inputs", lambda *args: {"fake": True})
    monkeypatch.setattr(
        ed_run.ed_inputs.bars_reader,
        "read_close_batched",
        lambda *args, **kwargs: (pd.DataFrame({"SPY": [1.0]}), None),
    )
    monkeypatch.setattr(ed_run, "simulate_arm", lambda *args, **kwargs: cells)
    monkeypatch.setattr(
        ed_run, "run_spy", lambda *args, **kwargs: SimpleNamespace(cagr_after_tax=0)
    )
    out = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ed_run",
            "--window",
            "holdout",
            "--arm",
            "A",
            "--eval-start",
            ed_run.START,
            "--eval-end",
            ed_run.END,
            "--out",
            str(out),
        ],
    )
    ed_run.main()
    proof = json.loads(out.read_text())
    assert proof["matches"] and proof["cell"] == saved and proof["clean"]
    cells[0]["cagr_after_tax"] = 0.2
    with pytest.raises(SystemExit, match="did not match"):
        ed_run.main()
