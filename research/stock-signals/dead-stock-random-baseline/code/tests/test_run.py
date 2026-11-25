"""End-to-end test of the driver on synthetic inputs: check first, resume, refusals."""

from __future__ import annotations

import json
from pathlib import Path

import dead_stock_random_baseline as logic
import numpy as np
import pandas as pd
import pytest
import rb_dead_stock_first_test as fp
from rb_ledger_run import run_spy

from ggTrader.utils.study import load_driver, study_dir

pytestmark = pytest.mark.study

# Loaded by path: another study's driver is also called run.py.
run = load_driver(study_dir(__file__) / "code" / "run.py")

WINDOW = "holdout"
DAYS = pd.bdate_range("2018-06-01", "2024-12-31", tz="UTC")
SYMBOLS = [f"S{i:02d}" for i in range(16)]


def inputs_for(_args) -> dict:
    rng = np.random.default_rng(3)
    prices = 40.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, (len(DAYS), len(SYMBOLS))), axis=0))
    close = pd.DataFrame(prices, index=DAYS, columns=SYMBOLS)
    spy = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(DAYS)))), index=DAYS)
    # S03 dies mid-window.
    close.loc["2022-03-01":, "S03"] = np.nan
    spans = pd.DataFrame(
        {
            "symbol": ["S03"],
            "last_date": pd.to_datetime(["2022-02-28"]),
            "has_successor": [False],
            "total_loss": [False],
        }
    )
    eligible = pd.DataFrame(True, index=DAYS, columns=SYMBOLS)
    return {
        "close": close,
        "spy_close": spy,
        "eligible": eligible,
        "span_rows": spans,
        "release_spans": spans,
        "data_provenance": {},
    }


def saved_file(tmp_path: Path, *, break_it: bool = False) -> Path:
    """Write the rule's own cells for the synthetic data as the 'saved' file."""
    start, end = run.baseline_inputs.WINDOWS[WINDOW]
    prep = run.Prepared(inputs_for(None), start, end)
    cells = prep.draw_cells(prep.rule_steps)
    strategy = {fp.cell_key(loss, logic.COST, logic.TAX): cell for loss, cell in cells.items()}
    spy = run_spy(
        prep.spy_close,
        params=run._tax_params(),
        one_way_cost=logic.COST,
        start=start,
        end=end,
        start_cash=fp.START_CASH,
    )
    spy_cell = fp._cell(spy)
    for cell in strategy.values():
        cell["gap_cagr_after_tax_pp"] = (cell["cagr_after_tax"] - spy_cell["cagr_after_tax"]) * 100
    if break_it:
        next(iter(strategy.values()))["after_tax_final"] += 1.0
    path = tmp_path / "saved.json"
    path.write_text(
        json.dumps(
            {
                "result": {
                    "strategy": strategy,
                    "spy": {fp.spy_key(logic.COST, logic.TAX): spy_cell},
                }
            }
        )
    )
    return path


def args_for(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--window",
        WINDOW,
        "--release-id",
        "1",
        "--crsp-load",
        "1",
        "--draws",
        "2",
        "--sanity-draws",
        "2",
        "--out",
        str(tmp_path / "out.json"),
        "--results-dir",
        str(tmp_path / "partial"),
        *extra,
    ]


def call(tmp_path: Path, saved: Path, *extra: str) -> int:
    return run.main(
        args_for(tmp_path, *extra),
        loader=inputs_for,
        saved_paths={WINDOW: saved},
        git_dirty=lambda: False,
    )


def test_a_full_run_writes_both_arms_the_check_and_the_sanity_block(tmp_path: Path) -> None:
    assert call(tmp_path, saved_file(tmp_path)) == 0
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["check_first"]["ok"] and out["check_first"]["n_cells"] == 4
    assert [len(out["draws"][arm]) for arm in run.ARMS] == [2, 2]
    assert out["summary"]["R"]["loss1"]["position"]["n"] == 2
    assert "all_eligible_cagr_pct" in out["sanity"]
    assert len(out["sanity_dice_cagr_by_seed"]) == 2
    assert out["completed"] == {"R": 2, "M": 2, "sanity": 2}
    assert len(out["check_first"]["rule_new_names_by_step"]) == out["n_rebalance_steps"]
    assert out["check_first"]["rule_new_names_by_step"][0] > 0
    assert sum(out["check_first"]["rule_new_names_by_step"]) == len(SYMBOLS)  # all bought once
    seeds = [rec["seed"] for rec in out["draws"]["R"]]
    assert seeds == [20261125 + 2_000_000, 20261125 + 2_000_001]


def test_a_failed_check_runs_no_draw_and_writes_nothing(tmp_path: Path) -> None:
    code = call(tmp_path, saved_file(tmp_path, break_it=True))
    assert code == run.EXIT_CHECK_FAILED
    assert not (tmp_path / "out.json").exists()
    assert not (tmp_path / "partial").exists()


def test_the_run_resumes_from_its_partial_file_without_redoing_draws(tmp_path: Path) -> None:
    saved = saved_file(tmp_path)
    assert call(tmp_path, saved) == 0
    first = json.loads((tmp_path / "out.json").read_text())
    (tmp_path / "out.json").unlink()
    assert call(tmp_path, saved) == 0
    second = json.loads((tmp_path / "out.json").read_text())
    assert first["draws"] == second["draws"]
    assert first["sanity_dice_cagr_by_seed"] == second["sanity_dice_cagr_by_seed"]


def test_refusals(tmp_path: Path) -> None:
    saved = saved_file(tmp_path)
    assert (
        run.main(
            ["--window", WINDOW, "--release-id", "2", "--crsp-load", "1"],
            loader=inputs_for,
            saved_paths={WINDOW: saved},
            git_dirty=lambda: False,
        )
        == run.EXIT_REFUSED
    )
    assert (
        run.main(
            args_for(tmp_path),
            loader=inputs_for,
            saved_paths={WINDOW: saved},
            git_dirty=lambda: True,
        )
        == run.EXIT_REFUSED
    )
    (tmp_path / "out.json").write_text("{}")
    assert call(tmp_path, saved) == run.EXIT_REFUSED


def test_drifted_inputs_are_refused(tmp_path: Path) -> None:
    def drifted(args):
        data = inputs_for(args)
        data["data_provenance"] = {"candidates": {"drift": True, "drift_symbols": ["S01"]}}
        return data

    code = run.main(
        args_for(tmp_path),
        loader=drifted,
        saved_paths={WINDOW: saved_file(tmp_path)},
        git_dirty=lambda: False,
    )
    assert code == run.EXIT_REFUSED


def _only_partial(tmp_path: Path) -> Path:
    (path,) = list((tmp_path / "partial").glob("*.partial.json"))
    return path


def test_a_partial_from_other_data_or_code_is_refused(tmp_path: Path) -> None:
    saved = saved_file(tmp_path)
    assert call(tmp_path, saved) == 0
    partial = _only_partial(tmp_path)
    (tmp_path / "out.json").unlink()
    original = partial.read_text()
    for key in (
        "window",
        "saved_file_sha256",
        "data_provenance_sha256",
        "code_sha256",
        "git_head",
        "lock_sha256",
        "draws",
    ):
        data = json.loads(original)
        data["identity"][key] = "tampered" if data["identity"][key] != 999 else 998
        partial.write_text(json.dumps(data))
        assert call(tmp_path, saved) == run.EXIT_REFUSED, key
        assert not (tmp_path / "out.json").exists()
    partial.write_text(original)
    assert call(tmp_path, saved) == 0


def test_a_partial_with_a_stray_or_duplicate_draw_is_refused(tmp_path: Path) -> None:
    saved = saved_file(tmp_path)
    assert call(tmp_path, saved) == 0
    partial = _only_partial(tmp_path)
    (tmp_path / "out.json").unlink()
    data = json.loads(partial.read_text())
    data["R"].append({**data["R"][0], "seed": 123})  # a draw nobody asked for
    partial.write_text(json.dumps(data))
    assert call(tmp_path, saved) == run.EXIT_REFUSED
    data["R"][-1] = dict(data["R"][0])  # a duplicate seed
    partial.write_text(json.dumps(data))
    assert call(tmp_path, saved) == run.EXIT_REFUSED


def test_a_timing_probe_does_not_leak_into_the_real_run(tmp_path: Path) -> None:
    saved = saved_file(tmp_path)
    probe = tmp_path / "probe.json"
    code = run.main(
        [
            *args_for(tmp_path)[:-4],
            "--first-draw",
            "9999",
            "--draws",
            "1",
            "--sanity-draws",
            "1",
            "--out",
            str(probe),
            "--results-dir",
            str(tmp_path / "partial"),
        ],
        loader=inputs_for,
        saved_paths={WINDOW: saved},
        git_dirty=lambda: False,
    )
    assert code == 0
    assert call(tmp_path, saved) == 0
    out = json.loads((tmp_path / "out.json").read_text())
    seeds = {rec["seed"] for rec in out["draws"]["R"]}
    assert seeds == {20261125 + 2_000_000, 20261125 + 2_000_001}
    assert len(list((tmp_path / "partial").glob("*.partial.json"))) == 2


def test_a_sanity_key_spelled_two_ways_is_refused(tmp_path: Path) -> None:
    saved = saved_file(tmp_path)
    assert call(tmp_path, saved) == 0
    partial = _only_partial(tmp_path)
    (tmp_path / "out.json").unlink()
    data = json.loads(partial.read_text())
    key = next(iter(data["sanity"]))
    data["sanity"]["0" + key] = data["sanity"][key]
    partial.write_text(json.dumps(data))
    assert call(tmp_path, saved) == run.EXIT_REFUSED
