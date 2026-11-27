"""Check regression control flow without a database."""

import json
import sys

import ed_run
import pytest

pytestmark = pytest.mark.study


def test_regression_compares_saved_cell_exactly(monkeypatch, tmp_path) -> None:
    """A changed cell must fail, and the requested dates reach the engine."""
    first = tmp_path / "dead-stock-first-test"
    first.mkdir()
    keys = [ed_run.study.cell_key(0.0, 0.00183, tax) for tax in ("low", "high")]
    cells = dict.fromkeys(keys, {"x": 1})
    (first / "_holdout-A.json").write_text(json.dumps({"result": {"strategy": cells}}))
    monkeypatch.setattr(ed_run, "study_dir", lambda _: tmp_path / "early-decades")
    monkeypatch.setattr(
        ed_run.ed_inputs,
        "read_inputs",
        lambda args: dict.fromkeys(
            ["close", "spy_close", "eligible", "span_rows", "release_spans"]
        ),
    )
    seen = {}

    def run(*args, **kwargs):
        seen.update(kwargs)
        return {"strategy": cells}

    monkeypatch.setattr(ed_run.study, "run_study", run)
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
    assert json.loads(out.read_text())["matches"] is True
    assert (seen["start"], seen["end"]) == (ed_run.START, ed_run.END)
    monkeypatch.setattr(
        ed_run.study, "run_study", lambda *a, **kw: {"strategy": dict.fromkeys(keys, {"x": 2})}
    )
    with pytest.raises(SystemExit, match="did not match"):
        ed_run.main()
    assert json.loads(out.read_text())["matches"] is False
