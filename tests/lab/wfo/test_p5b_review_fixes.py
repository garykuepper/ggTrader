"""P5b: fixes for the independent review of the walk-forward rewrite.

* MAJOR-1: selection windows blank Sharpe/Sortino with no entry and Sortino with fewer than
  ``MIN_LOSING_DAYS`` losing days, so an almost flat combination cannot win a composite.
* MAJOR-2: the anchor of fold ``k`` is a pure function of data up to ``train_end_k`` (a
  full-sample anchor is a lookahead that moved the 17-fold Sharpe from 0.39 to 0.66), and the
  default DSR gate threshold is 0.80.
* MAJOR-3: every fold records the anchor parameter set and why it was (or was not) deployed.
* MINOR-1/2/3: tie-safe winner pick, combination-order check, dead combinations are not anchors.
"""

from __future__ import annotations

import inspect
import json
import re

import numpy as np
import pandas as pd
import pytest

from ggTrader.engine import config as engine_config
from ggTrader.engine import portfolio as engine_portfolio
from ggTrader.engine import stats
from ggTrader.lab.data import STOCK_BASE_CONFIG
from ggTrader.lab.strategies.contract import ParamGrid, SignalOut
from ggTrader.lab.strategies.ensemble import Ensemble
from ggTrader.lab.strategies.leak import perturb_after, truncate_bars
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.wfo import folds as folds_mod
from ggTrader.lab.wfo import inputs, report, run, run_wfo, select
from ggTrader.lab.wfo.anchor import NO_ANCHOR, pick_anchor
from ggTrader.lab.wfo.scorers import SignalScorer, WindowReq
from tests.lab.wfo import helpers as h

pytestmark = pytest.mark.filterwarnings("ignore")
ES, EE = "2020-06-01", "2022-03-01"


# --- MAJOR-1: the guards reach the selection windows ------------------------------------------


class _LateToy(h.ToyHold):
    """Same closed-form strategy; the grid is chosen per test."""

    def __init__(self, delays) -> None:
        self.grid = ParamGrid.from_sweep({"delay": list(delays)})


def _scorer(delays, bars=None, **kwargs) -> SignalScorer:
    bars = bars if bars is not None else h.toy_bars()
    strat = _LateToy(delays)
    combos = strat.grid.combos()
    names = [inputs.combo_name("toy", c) for c in combos]
    scorer = SignalScorer(
        strat, bars, strat.grid, names, combos, None, dict(h.TOY_CONFIG), 1 << 30, **kwargs
    )
    scorer.prepare([])
    return scorer


def _fold1(bars):
    return folds_mod.generate_folds(pd.Timestamp(ES, tz="UTC"), pd.Timestamp(EE, tz="UTC"))[0]


def _delays_for_fold1(bars):
    """(carried in before the window, entering early inside it, entering 5 bars before its end)."""
    fold = _fold1(bars)
    a, b = np.searchsorted(bars.close.index, [fold.train_start, fold.train_end])
    return 3, int(a) + 10, int(b) - 5


def test_a_selection_window_blanks_a_combination_with_no_entry_and_few_losing_days() -> None:
    bars = h.toy_bars()
    carried, early, late = _delays_for_fold1(bars)
    scorer = _scorer([carried, early, late], bars)
    fold = _fold1(bars)
    rows = {
        r["params"]["delay"]: r
        for r in scorer.metrics(WindowReq("train", 1, fold.train_start, fold.train_end))
    }
    assert np.isnan(rows[carried]["sharpe"]) and np.isnan(rows[carried]["sortino"])  # no entry
    assert np.isfinite(rows[early]["sharpe"]) and np.isfinite(rows[early]["sortino"])
    assert np.isfinite(rows[late]["sharpe"]) and np.isnan(rows[late]["sortino"])  # 5 exposed days
    assert np.isfinite(rows[carried]["max_drawdown_pct"]) and np.isfinite(rows[carried]["cagr_pct"])


def test_the_anchor_and_live_windows_are_selection_windows_too() -> None:
    bars = h.toy_bars()
    carried, early, late = _delays_for_fold1(bars)
    scorer = _scorer([carried, early, late], bars)
    live = WindowReq("live", 0, bars.close.index[400], bars.close.index[600])
    rows = {r["params"]["delay"]: r for r in scorer.metrics(live)}
    assert np.isnan(rows[carried]["sharpe"])  # entered at bar 3, none inside the window
    anchor = WindowReq("anchor", 0, None, bars.close.index[600])
    assert all(np.isfinite(r["sharpe"]) for r in scorer.metrics(anchor))  # every entry is inside


def test_the_test_window_is_a_measurement_so_only_the_flat_curve_guard_applies() -> None:
    bars = h.toy_bars()
    carried, early, late = _delays_for_fold1(bars)
    scorer = _scorer([carried, early, late], bars)
    fold = _fold1(bars)
    rows = scorer.metrics(WindowReq("test", 1, fold.test_start, fold.test_end), [0])
    assert np.isfinite(rows[0]["sharpe"])  # the deployed book held a carried-in position


def test_the_guard_is_overridable_and_off_at_zero() -> None:
    bars = h.toy_bars()
    carried, early, late = _delays_for_fold1(bars)
    fold = _fold1(bars)
    req = WindowReq("train", 1, fold.train_start, fold.train_end)
    loose = _scorer([carried, early, late], bars, min_losing_days=0)
    # the zero-entry rule stays on (spec 5.6); only the losing-day threshold is off
    rows = {r["params"]["delay"]: r for r in loose.metrics(req)}
    assert np.isfinite(rows[late]["sortino"]) and np.isnan(rows[carried]["sharpe"])
    strict = _scorer([carried, early, late], bars, min_losing_days=10_000)
    assert all(np.isnan(r["sortino"]) for r in strict.metrics(req))


def test_a_near_flat_combination_does_not_win_the_composite() -> None:
    """The review's fold 17: one outlier Sortino used to take the full 0.3 weight."""
    bars = h.toy_bars()
    carried, early, late = _delays_for_fold1(bars)
    fold = _fold1(bars)
    req = WindowReq("train", 1, fold.train_start, fold.train_end)
    rows = _scorer([carried, early, late], bars, min_losing_days=0).metrics(req)
    unguarded_pick = rows[select.best_index(select.composite_score(rows))]["params"]["delay"]
    rows_g = _scorer([carried, early, late], bars).metrics(req)
    guarded_pick = rows_g[select.best_index(select.composite_score(rows_g))]["params"]["delay"]
    assert guarded_pick == early
    assert unguarded_pick in (early, late)  # documents what the guard is compared against


def test_the_run_stamps_the_guards_it_used() -> None:
    bars = h.toy_bars()
    res = run_wfo(
        "toy", h.ToyHold, LabConfig(), bars, bars.close["X"], ES, EE, "t", dict(h.TOY_CONFIG),
        h.ToyHold.grid, verbose=False, min_losing_days=7, allow_unmasked=True
    )  # fmt: skip
    assert res.diagnostics["guards"] == {
        "min_losing_days": 7,
        "zero_trade_rule": "selection windows (anchor, train, live)",
    }
    default = run_wfo(
        "toy", h.ToyHold, LabConfig(), bars, bars.close["X"], ES, EE, "t", dict(h.TOY_CONFIG),
        h.ToyHold.grid, verbose=False, allow_unmasked=True
    )  # fmt: skip
    assert default.diagnostics["guards"]["min_losing_days"] == engine_config.MIN_LOSING_DAYS == 20


# --- MAJOR-3: the anchor of every fold is recorded ---------------------------------------------


@pytest.fixture(scope="module")
def toy():
    bars = h.toy_bars()
    result = run_wfo(
        "toy", h.ToyHold, LabConfig(), bars, bars.close["X"], ES, EE, "t", dict(h.TOY_CONFIG),
        h.ToyHold.grid, verbose=False, allow_unmasked=True
    )  # fmt: skip
    return bars, result


def _oracle_anchor(close: pd.Series, end) -> object:
    """Independent anchor: closed-form equity of every delay, sliced by label up to ``end``."""
    from tests.lab.wfo.test_toy_known_answers import _rows

    return pick_anchor(_rows(close, close.index[0], end))


def test_every_fold_records_its_anchor_and_why_it_was_used(toy) -> None:
    bars, result = toy
    close = bars.close["X"]
    for row in result.fold_results:
        expected = _oracle_anchor(close, row["train_end"])
        assert row["anchor_combo"] == expected.combo
        assert row["anchor_params"] == expected.params
        assert row["deployed_combo"] == (
            expected.combo if row["used_anchor"] else row["winner_combo"]
        )
        if not row["used_anchor"]:
            assert row["anchor_reason"] is None
        elif row["halted_entering"]:
            assert row["anchor_reason"] == "halted"
        else:
            assert row["anchor_reason"] == "gates_failed"
        assert isinstance(row["halt_cleared"], bool)


def test_the_anchor_history_is_in_the_json_payload_and_the_report(toy) -> None:
    _, result = toy
    payload = json.loads(json.dumps(report.result_payload(result), default=str))
    assert [r["anchor_combo"] for r in payload["fold_table"]] == [
        r["anchor_combo"] for r in result.fold_results
    ]
    assert payload["anchor_history"][0]["fold"] == 1
    assert set(payload["anchor_history"][0]) >= {"anchor_combo", "reason", "deployed_combo"}
    assert "── Anchor Per Fold" in result.table
    # callers that parse the legacy table (`^\d+\s` fold rows) must not see new rows
    fold_rows = [ln for ln in result.table.splitlines() if re.match(r"^\d+\s", ln)]
    assert len(fold_rows) == len(result.fold_results)


def test_the_default_table_text_is_still_the_legacy_text() -> None:
    """Without the anchor history the formatter is byte-for-byte the legacy format."""
    assert "anchor_history" in inspect.signature(report.format_wfo_table).parameters
    assert inspect.signature(report.format_wfo_table).parameters["anchor_history"].default is None


# --- MAJOR-2: the anchor is causal --------------------------------------------------------------


def test_the_anchor_of_each_fold_differs_from_the_full_sample_anchor_on_the_toy(toy) -> None:
    """Teeth: on this series a full-sample anchor would pick another combination."""
    bars, result = toy
    close = bars.close["X"]
    full = _oracle_anchor(close, close.index[-1]).combo
    assert all(r["anchor_combo"] != full for r in result.fold_results)


def _flipping_perturbation(bars, fold_row):
    """Bars shaken after ``fold_row["train_end"]`` so that an anchor fitted up to the last
    evaluation day (a lookahead) would be another combination than the causal one.

    Searches the first seed that does it, by the closed-form oracle; the search is deterministic
    and the assertion that one exists keeps the test honest if the toy changes.
    """
    ee = pd.Timestamp(EE, tz="UTC")
    for seed in range(200):
        shaken = perturb_after(bars, fold_row["train_end"], seed=seed)
        close = shaken.close["X"]
        causal = _oracle_anchor(close, fold_row["train_end"]).combo
        lookahead = _oracle_anchor(close, ee).combo
        if causal != lookahead:
            return shaken, causal, lookahead
    raise AssertionError("no perturbation separates the causal anchor from a full-sample one")


@pytest.mark.parametrize("k", [0, 1])
def test_perturbing_bars_after_an_interior_train_end_leaves_that_anchor_unchanged(toy, k) -> None:
    """Kills the full-sample anchor mutant: it would follow the perturbed later bars."""
    bars, result = toy
    row = result.fold_results[k]
    shaken, causal, lookahead = _flipping_perturbation(bars, row)
    again = run_wfo(
        "toy", h.ToyHold, LabConfig(), shaken, shaken.close["X"], ES, EE, "t", dict(h.TOY_CONFIG),
        h.ToyHold.grid, verbose=False, allow_unmasked=True
    )  # fmt: skip
    assert again.fold_results[k]["anchor_combo"] == causal != lookahead
    for old, new in list(zip(result.fold_results, again.fold_results, strict=True))[: k + 1]:
        assert (new["anchor_combo"], new["anchor_params"]) == (
            old["anchor_combo"],
            old["anchor_params"],
        )
        assert new["winner_combo"] == old["winner_combo"]
        assert new["train_score"] == old["train_score"]
    assert not shaken.close["X"].equals(bars.close["X"])  # the perturbation really happened


def test_the_anchor_equals_a_fresh_simulation_on_data_truncated_at_train_end() -> None:
    """Ensemble grid: the recorded anchor is what a run that never saw later bars would pick."""
    bars = h.small_wfo_bars()
    mask = h.open_mask(bars)
    result = _ensemble_run(bars, mask)
    grid = Ensemble.grid
    combos = inputs.resolve_grid(Ensemble, grid)[1]
    names = [inputs.combo_name("ensemble", c) for c in combos]
    for row in result.fold_results:
        cut = truncate_bars(bars, row["train_end"])
        out = Ensemble().build(cut, grid)
        sim = engine_portfolio.run_signals(
            out.entries,
            out.exits,
            _tile(cut.close, out.entries.columns),
            dict(h.BASE),
            groups=out.group_labels(),
            mask=_tile(mask.loc[: row["train_end"]], out.entries.columns),
        )
        table = stats.window_stats(sim.returns, 0, len(cut.close))
        rows = [
            {
                "combo": names[g],
                "params": combos[g],
                "cagr_pct": float(table.loc["cagr"].iloc[g]) * 100,
                "max_drawdown_pct": float(table.loc["max_drawdown"].iloc[g]) * 100,
                "sharpe": float(table.loc["sharpe"].iloc[g]),
            }
            for g in range(len(names))
        ]
        assert row["anchor_combo"] == pick_anchor(rows).combo


def test_bars_after_an_interior_train_end_cannot_change_that_folds_selection_or_anchor() -> None:
    bars = h.small_wfo_bars()
    mask = h.open_mask(bars)
    base = _ensemble_run(bars, mask)
    k = 1  # perturb after fold 2's train_end: folds 1 and 2 selection must not move
    cut = base.fold_results[k]["train_end"]
    shaken = _ensemble_run(perturb_after(bars, cut, seed=3), mask)
    keys = ("winner_combo", "train_score", "ndh_passed", "dsr_passed", "gates_passed",
            "dsr_value", "is_sharpe", "anchor_combo", "anchor_params")  # fmt: skip
    for old, new in list(zip(base.fold_results, shaken.fold_results, strict=True))[: k + 1]:
        for key in keys:
            a, b = old[key], new[key]
            same = (a == b) or (isinstance(a, float) and np.isnan(a) and np.isnan(b))
            assert same, (old["fold_num"], key, a, b)


def _tile(frame: pd.DataFrame, columns) -> pd.DataFrame:
    from ggTrader.lab.strategies.contract import tile_columns

    return tile_columns(frame, columns)


def _ensemble_run(bars, mask, **kwargs):
    return run_wfo(
        "ensemble", Ensemble, LabConfig(), bars, bars.close.iloc[:, 0], h.MINI["eval_start"],
        h.MINI["eval_end"], "stock", dict(STOCK_BASE_CONFIG), Ensemble.grid, mask=mask,
        verbose=False, **kwargs,
    )  # fmt: skip


# --- MAJOR-2 (b) / MINOR-4: the default DSR threshold -------------------------------------------


def test_the_default_dsr_threshold_is_point_eight() -> None:
    assert inspect.signature(run_wfo).parameters["dsr_threshold"].default == 0.80
    assert inspect.signature(run_wfo).parameters["ndh_threshold"].default == 0.85


def test_a_default_run_hands_the_default_threshold_to_the_gate(monkeypatch) -> None:
    seen: list[float] = []
    real = run.run_dsr

    def spy(returns, n_trials, threshold):
        seen.append(threshold)
        return real(returns, n_trials, threshold)

    monkeypatch.setattr(run, "run_dsr", spy)
    bars = h.toy_bars()
    run_wfo(
        "toy", h.ToyHold, LabConfig(), bars, bars.close["X"], ES, EE, "t", dict(h.TOY_CONFIG),
        h.ToyHold.grid, verbose=False, allow_unmasked=True
    )  # fmt: skip
    assert seen and set(seen) == {0.80}


# --- MINOR-1: ties --------------------------------------------------------------------------------


def test_a_tie_within_float_noise_goes_to_the_first_combination() -> None:
    assert select.best_index([0.5, 0.5 + 3e-13, 0.2]) == 0
    assert select.best_index([0.5, 0.5 + 1e-6, 0.2]) == 1  # a real difference still wins
    metrics = [{"combo": "a"}, {"combo": "b"}, {"combo": "c"}]
    assert select.pick_live_winner(metrics, [0.5, 0.5 + 3e-13, 0.1], {})[0] == 0


# --- MINOR-2: combination order ------------------------------------------------------------------


class _Reordered(h.ToyHold):
    """Builds its combinations in the opposite order to ``ParamGrid.combos()``."""

    def build(self, bars, grid=None) -> SignalOut:
        out = super().build(bars, grid)
        n = len(bars.symbols)
        cols = out.entries.columns
        order = [cols[i * n : (i + 1) * n] for i in range(len(out.combos))][::-1]
        flat = [c for block in order for c in block]
        return SignalOut(
            entries=out.entries[flat], exits=out.exits[flat], combos=tuple(reversed(out.combos))
        )


def test_a_strategy_that_orders_combinations_differently_is_refused() -> None:
    bars = h.toy_bars()
    combos = h.ToyHold.grid.combos()
    names = [inputs.combo_name("toy", c) for c in combos]
    scorer = SignalScorer(
        _Reordered(), bars, h.ToyHold.grid, names, combos, None, dict(h.TOY_CONFIG), 1 << 30
    )
    with pytest.raises(ValueError, match="order"):
        scorer.prepare([])


def test_a_strategy_in_the_grid_order_is_accepted() -> None:
    _scorer([3, 50, 150])  # does not raise


# --- MINOR-3: a dead combination is not an anchor -------------------------------------------------


def _row(name, dd, cagr, sharpe):
    return {"combo": name, "params": {"n": name}, "max_drawdown_pct": dd, "cagr_pct": cagr,
            "sharpe": sharpe}  # fmt: skip


def test_a_combination_that_never_traded_cannot_be_the_anchor() -> None:
    dead = _row("dead", 0.0, 0.0, float("nan"))
    live = _row("live", -9.0, 1.0, 0.2)
    assert pick_anchor([dead, live]).combo == "live"  # least drawdown, but only among live ones
    assert pick_anchor([live, dead]).combo == "live"


def test_cagr_filter_still_applies_among_live_combinations() -> None:
    assert pick_anchor([_row("a", -12.0, 6.0, 0.5), _row("b", -5.0, 2.0, 0.1)]).combo == "a"


def test_all_dead_means_no_anchor() -> None:
    assert pick_anchor([_row("d1", 0.0, 0.0, float("nan"))]) == NO_ANCHOR
    assert pick_anchor([]) == NO_ANCHOR


def test_a_run_with_no_anchor_sits_in_cash_over_the_test_windows() -> None:
    """Every combination enters after the last fold: no anchor exists and the book stays in cash
    (a flat equity at the starting cash, as the legacy dead anchor gave), not an empty curve."""

    class NeverTrades(h.ToyHold):
        grid = ParamGrid.from_sweep({"delay": [1200, 1250, 1290]})

    bars = h.toy_bars()
    res = run_wfo(
        "toy", NeverTrades, LabConfig(), bars, bars.close["X"], ES, EE, "t", dict(h.TOY_CONFIG),
        NeverTrades.grid, verbose=False, allow_unmasked=True
    )  # fmt: skip
    assert all(r["anchor_combo"] == "none" and r["used_anchor"] for r in res.fold_results)
    assert all(r["deployed_combo"] == "none" for r in res.fold_results)
    assert len(res.oos_equity) > 100
    assert (res.oos_equity == h.START_CASH).all()
    assert np.isfinite(res.diagnostics["spy_metrics"]["cagr_pct"])  # the SPY baseline survives


# --- P10: a signal run without a point-in-time mask is refused unless asked for -----------------


def test_a_signal_run_without_a_mask_raises_unless_allowed() -> None:
    bars = h.toy_bars()
    args = ("toy", h.ToyHold, LabConfig(), bars, bars.close["X"], ES, EE, "t")
    with pytest.raises(ValueError, match=r"allow_unmasked=True.*|survivorship"):
        run_wfo(*args, dict(h.TOY_CONFIG), h.ToyHold.grid, verbose=False)
    res = run_wfo(*args, dict(h.TOY_CONFIG), h.ToyHold.grid, verbose=False, allow_unmasked=True)
    assert res.fold_results


def test_signal_scorer_rejects_columns_that_disagree_with_reported_combinations() -> None:
    class ReorderedColumns(h.ToyHold):
        def build(self, bars, grid=None) -> SignalOut:
            out = super().build(bars, grid)
            return SignalOut(
                entries=out.entries.iloc[:, ::-1],
                exits=out.exits.iloc[:, ::-1],
                combos=out.combos,
            )

    bars = h.toy_bars()
    combos = h.ToyHold.grid.combos()
    names = [inputs.combo_name("toy", c) for c in combos]
    scorer = SignalScorer(
        ReorderedColumns(), bars, h.ToyHold.grid, names, combos, None, dict(h.TOY_CONFIG), 1 << 30
    )
    with pytest.raises(ValueError, match="signal.*columns"):
        scorer.prepare([])
