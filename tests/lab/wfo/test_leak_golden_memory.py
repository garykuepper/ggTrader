"""Whole-WFO leak test, a regression golden for a 3-fold mini WFO and the memory-budget test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.engine import portfolio as engine_portfolio
from ggTrader.lab.strategies.ensemble import Ensemble
from ggTrader.lab.strategies.leak import perturb_after
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.wfo import run_wfo
from tests.lab.wfo import helpers as h

pytestmark = pytest.mark.filterwarnings("ignore")
GOLDEN = Path(__file__).parent / "golden" / "mini_wfo.json"


def _run(bars, eval_end: str = h.MINI["eval_end"], **kwargs):
    mask = h.open_mask(bars)
    return run_wfo(
        "ensemble",
        Ensemble,
        LabConfig(),
        bars,
        bars.close.iloc[:, 0],
        h.MINI["eval_start"],
        eval_end,
        "stock",
        dict(h.BASE),
        Ensemble.grid,
        mask=mask,
        verbose=False,
        **kwargs,
    )


@pytest.fixture(scope="module")
def mini():
    return _run(h.small_wfo_bars())


def _same_folds(a, b) -> None:
    assert len(a.fold_results) == len(b.fold_results)
    for x, y in zip(a.fold_results, b.fold_results, strict=True):
        for key, value in x.items():
            other = y[key]
            if isinstance(value, float) and np.isnan(value):
                assert np.isnan(other), key
            else:
                assert value == other, key
    pd.testing.assert_series_equal(a.oos_equity, b.oos_equity)


# --- leak test through the whole WFO ---------------------------------------------------------


def test_bars_after_eval_end_cannot_change_any_result(mini) -> None:
    bars = h.small_wfo_bars()
    last = pd.Timestamp(h.MINI["eval_end"], tz="UTC")
    assert (bars.close.index > last).sum() > 100
    shaken = _run(perturb_after(bars, last, seed=4))
    _same_folds(mini, shaken)
    assert shaken.live_params["combo"] == mini.live_params["combo"]
    assert shaken.table == mini.table


def test_bars_after_the_last_fold_cannot_change_a_fold_result() -> None:
    bars = h.small_wfo_bars()
    late_end = "2022-05-15"  # live window reaches past the last fold's test_end (2022-03-01)
    base = _run(bars, eval_end=late_end)
    last_fold_end = base.fold_results[-1]["test_end"]
    assert last_fold_end < pd.Timestamp(late_end, tz="UTC")
    shaken = _run(perturb_after(bars, last_fold_end, seed=9), eval_end=late_end)
    _same_folds(base, shaken)  # only the live-parameter pick may use the later bars


def test_the_leak_test_has_teeth_changing_a_bar_inside_the_folds_changes_the_result(mini) -> None:
    bars = h.small_wfo_bars()
    inside = pd.Timestamp("2021-06-01", tz="UTC")
    changed = _run(perturb_after(bars, inside, seed=2))
    assert changed.table != mini.table or not changed.oos_equity.equals(mini.oos_equity)


# --- regression golden -----------------------------------------------------------------------


def _summary(result) -> dict:
    return {
        "winners": [r["winner_combo"] for r in result.fold_results],
        "gates": [bool(r["gates_passed"]) for r in result.fold_results],
        "used_anchor": [bool(r["used_anchor"]) for r in result.fold_results],
        "halted": [bool(r["halted"]) for r in result.fold_results],
        "dsr_value": [r["dsr_value"] for r in result.fold_results],
        "is_sharpe": [r["is_sharpe"] for r in result.fold_results],
        "oos_sharpe": [
            None if np.isnan(r["oos_sharpe"]) else r["oos_sharpe"] for r in result.fold_results
        ],
        "live_combo": result.live_params["combo"],
        "oos_equity": [float(v) for v in result.oos_equity.iloc[::10]],
        "oos_bars": int(len(result.oos_equity)),
    }


SCENARIOS = {
    # the full 48-combination grid: the anchor is deployed (every fold fails a gate)
    "full_grid": {"grid": None, "kwargs": {}},
    # 16 active combinations (min_agree 2): trades in the test windows, anchor deployed
    "active_16": {"grid": {"min_agree": [2]}, "kwargs": {}},
    # same grid with the gate thresholds opened, so a winner is deployed in fold 1
    "active_16_open_gates": {
        "grid": {"min_agree": [2]},
        "kwargs": {"ndh_threshold": 0.0, "dsr_threshold": 0.0},
    },
}


def _scenario(name: str):
    spec = SCENARIOS[name]
    grid = Ensemble.grid if spec["grid"] is None else Ensemble.grid.override(spec["grid"])
    bars = h.small_wfo_bars()
    return run_wfo(
        "ensemble",
        Ensemble,
        LabConfig(),
        bars,
        bars.close.iloc[:, 0],
        h.MINI["eval_start"],
        h.MINI["eval_end"],
        "stock",
        dict(h.BASE),
        grid,
        mask=h.open_mask(bars),
        verbose=False,
        **spec["kwargs"],
    )


def test_three_fold_mini_wfo_matches_its_golden(request: pytest.FixtureRequest) -> None:
    """A change detector, not evidence: the golden was written by the code under test.

    It catches any unintended edit to a result. What makes a result right is checked by the
    independent tests (the closed-form toy oracle, legacy equivalence, the causality and
    perturbation tests); with this test deselected they still kill every mutant it caught
    (P5 review MINOR-5). Regenerate with ``--update-golden --golden-reason`` and say why in
    ``docs/changelog.md``.
    """
    data = {name: _summary(_scenario(name)) for name in SCENARIOS}
    assert all(len(v["winners"]) == 3 for v in data.values())
    if request.config.getoption("--update-golden"):
        reason = request.config.getoption("--golden-reason")
        assert reason, "--update-golden needs --golden-reason (also put it in docs/changelog.md)"
        GOLDEN.write_text(json.dumps({"reason": reason, "data": data}, indent=1) + "\n")
    stored = json.loads(GOLDEN.read_text())["data"]
    assert stored.keys() == data.keys()
    for name, got in data.items():
        for key in ("winners", "gates", "used_anchor", "halted", "live_combo", "oos_bars"):
            assert got[key] == stored[name][key], (name, key)
        for key in ("dsr_value", "is_sharpe", "oos_sharpe", "oos_equity"):
            np.testing.assert_allclose(
                np.array(got[key], dtype=float),
                np.array(stored[name][key], dtype=float),
                rtol=1e-9,
                err_msg=f"{name}.{key}",
            )
    # the golden must exercise a winner deployment and a trading out-of-sample window
    assert data["active_16_open_gates"]["used_anchor"][0] is False
    assert any(x is not None for x in data["active_16"]["oos_sharpe"])


# --- memory budget ---------------------------------------------------------------------------


def test_a_small_budget_splits_the_simulation_without_changing_a_number(mini) -> None:
    budget = 600_000
    small = _run(h.small_wfo_bars(), max_bytes=budget)
    assert small.diagnostics["n_chunks"] > 1
    _same_folds(mini, small)
    assert small.table == mini.table


def test_every_planned_chunk_of_the_full_17_fold_run_fits_its_budget() -> None:
    """Pure arithmetic on the real run's dimensions: 48 groups x 605 symbols x 1,929 bars."""
    n_bars, n_groups, n_syms, budget = 1929, 48, 605, 3 * 1024**3
    codes = np.repeat(np.arange(n_groups), n_syms)
    chunks = engine_portfolio._plan_chunks(codes, n_bars, 1, budget)
    assert len(chunks) >= 2
    for chunk in chunks:
        cols = len(chunk) * n_syms
        assert engine_portfolio.estimate_peak_bytes(n_bars, cols, 1) <= budget
    assert sorted(c for chunk in chunks for c in chunk) == list(range(n_groups))


def test_the_peak_estimate_for_the_single_pass_is_bounded_by_the_budget_per_chunk() -> None:
    result = _run(h.small_wfo_bars(), max_bytes=2_000_000)
    d = result.diagnostics
    n_cols = d["n_symbols"] * d["n_combos"]
    per_chunk = engine_portfolio.estimate_peak_bytes(d["n_bars"], n_cols // d["n_chunks"] + 1, 1)
    assert d["n_chunks"] >= 2 and per_chunk < 4 * 2_000_000
