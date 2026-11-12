"""Report format, input normalisation and the fresh-start weight path (synthetic)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.xs_momentum import XsMomentum
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.wfo import inputs, report, run_wfo
from ggTrader.lab.wfo.anchor import AnchorSet
from ggTrader.lab.wfo.scorers import WeightScorer, WindowReq
from tests.lab.wfo import helpers as h
from tests.oracles.curve_stats import curve_stats

pytestmark = pytest.mark.filterwarnings("ignore")


# --- report ----------------------------------------------------------------------------------


def _fold(n: int, **over):
    base = {
        "fold_num": n,
        "train_start": pd.Timestamp("2021-01-31", tz="UTC"),
        "train_end": pd.Timestamp("2022-01-31", tz="UTC"),
        "test_start": pd.Timestamp("2022-01-31", tz="UTC"),
        "test_end": pd.Timestamp("2022-04-30", tz="UTC"),
        "winner_combo": "ensemble__" + "x" * 40,
        "gates_passed": n % 2 == 0,
        "wfe": None if n == 1 else 0.5,
        "halted": n == 3,
        "used_anchor": n != 2,
        "is_sharpe": 1.234,
        "oos_sharpe": float("nan") if n == 3 else 0.456,
    }
    return {**base, **over}


def test_the_table_text_equals_the_frozen_legacy_table_for_the_same_inputs() -> None:
    folds_ = [_fold(1), _fold(2), _fold(3)]
    oos = {"sharpe": 0.62, "cagr_pct": 3.3, "max_drawdown_pct": -5.3}
    spy = {"sharpe": 0.78, "cagr_pct": 13.0, "max_drawdown_pct": -22.1}
    live = {
        "combo": "ensemble__a",
        "stability": 2,
        "train_start": folds_[0]["train_start"],
        "train_end": folds_[0]["train_end"],
        "train_metrics": {"sharpe": 1.3, "cagr_pct": 20.2, "max_drawdown_pct": -3.0},
    }
    anchor = AnchorSet("ensemble__b", {}, -7.7, 4.2, 0.69)
    args = (folds_, oos, spy, live, "ensemble", 48, 3)
    new = report.format_wfo_table(*args, halted=True, anchor=anchor)
    golden = json.loads((Path(__file__).parent / "golden" / "legacy_logic_units.json").read_text())
    assert new == golden["table_text"]


def test_curve_summary_agrees_with_curve_stats_on_the_shared_conventions() -> None:
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2021-01-01", periods=300, tz="UTC")
    curve = pd.Series(10000 * np.cumprod(1 + rng.normal(0.0005, 0.01, 300)), index=idx)
    mine, ref = report.curve_summary(curve), curve_stats(curve)
    for key in ("sharpe", "max_drawdown_pct", "total_return_pct", "ann_vol_pct"):
        assert mine[key] == pytest.approx(ref[key], rel=1e-10)
    flat = report.curve_summary(pd.Series(1.0, index=idx))
    assert math.isnan(flat["sharpe"])  # the flat-curve guard
    assert all(math.isnan(v) for v in report.curve_summary(curve.iloc[:1]).values())


def test_result_payload_is_json_able_and_complete() -> None:
    bars = h.toy_bars()
    result = run_wfo(
        "toy",
        h.ToyHold,
        LabConfig(),
        bars,
        bars.close["X"],
        "2020-06-01",
        "2022-03-01",
        "t",
        dict(h.TOY_CONFIG),
        h.ToyHold.grid,
        verbose=False,
        allow_unmasked=True,
    )
    from ggTrader.utils.results import write_result_json

    payload = report.result_payload(result, run="toy")
    assert payload["n_folds"] == 3 and payload["gates"].endswith("/3")
    path = write_result_json("/tmp/_wfo_payload_test.json", payload)
    data = json.loads(path.read_text())
    assert data["provenance"]["engine_effective"] == "rust" and len(data["winners"]) == 3
    path.unlink()


# --- inputs ----------------------------------------------------------------------------------


def test_as_bars_from_a_legacy_frame_keeps_fields_and_sorts_symbols() -> None:
    bars = h.small_wfo_bars(n_syms=4)
    frame = bars.legacy_frame()[["S02", "S00", "S03", "S01"]]
    got = inputs.as_bars(frame)
    assert got.symbols == ["S00", "S01", "S02", "S03"]
    pd.testing.assert_frame_equal(got.close, bars.close, check_freq=False, check_names=False)
    with pytest.raises(ValueError, match="tz-aware"):
        inputs.as_bars(frame.tz_localize(None))
    with pytest.raises(ValueError, match="close"):
        inputs.as_bars(frame.drop(columns="close", level=1))


def test_resolve_grid_accepts_the_legacy_list_and_rejects_a_ragged_one() -> None:
    from ggTrader.lab.strategies.ensemble import Ensemble

    combos = Ensemble.grid.combos()
    grid, got = inputs.resolve_grid(Ensemble, combos)
    assert got == combos and len(grid) == 48
    with pytest.raises(ValueError, match="Cartesian"):
        inputs.resolve_grid(Ensemble, combos[:-1])


def test_universe_fn_and_mask_paths_agree() -> None:
    bars = h.small_wfo_bars(n_syms=4)
    mask = h.open_mask(bars)
    frame = bars.legacy_frame()

    def fn(asof, past):
        return [s for s in bars.symbols if mask.loc[asof, s]]

    pd.testing.assert_frame_equal(
        inputs.mask_from_universe_fn(frame, bars.symbols, fn),
        mask,
        check_freq=False,
        check_names=False,
    )


# --- weight path -----------------------------------------------------------------------------


def _weight_scorer(bars, mask) -> WeightScorer:
    cls = XsMomentum
    combos = cls.grid.combos()
    names = [inputs.combo_name("xs_momentum", c) for c in combos]
    return WeightScorer(cls(), bars, cls.grid, names, combos, mask, dict(h.BASE), 1 << 30)


@pytest.fixture(scope="module")
def weight_setup():
    bars = h.make_bars(seed=8, n_bars=900, n_syms=14)
    mask = bars.close.notna().cumsum() >= 60
    reqs = [
        WindowReq(
            "train", 0, pd.Timestamp("2020-02-03", tz="UTC"), pd.Timestamp("2021-02-02", tz="UTC")
        ),
        WindowReq(
            "train", 1, pd.Timestamp("2020-05-04", tz="UTC"), pd.Timestamp("2021-05-03", tz="UTC")
        ),
        WindowReq(
            "test", 0, pd.Timestamp("2021-02-02", tz="UTC"), pd.Timestamp("2021-05-03", tz="UTC")
        ),
    ]
    return bars, mask, reqs


def test_every_weight_window_starts_flat_with_fresh_cash(weight_setup) -> None:
    bars, mask, reqs = weight_setup
    scorer = _weight_scorer(bars, mask)
    scorer.prepare(reqs)
    for req in reqs:
        for g in range(len(scorer.names)):
            eq = scorer.equity(req, g)
            assert eq.iloc[0] == h.START_CASH and eq.index[0] >= req.start
            # cash until the first decision takes effect on the second bar
            assert abs(eq.iloc[1] / h.START_CASH - 1) < 1e-3  # only slippage on the first buys


def test_stacking_windows_changes_no_number(weight_setup) -> None:
    bars, mask, reqs = weight_setup
    together = _weight_scorer(bars, mask)
    together.prepare(reqs)
    for req in reqs:
        alone = _weight_scorer(bars, mask)
        alone.prepare([req])
        for g in range(len(alone.names)):
            assert np.array_equal(
                alone.equity(req, g).to_numpy(), together.equity(req, g).to_numpy()
            )
        assert alone.metrics(req) == together.metrics(req) or all(
            _nan_equal(a, b) for a, b in zip(alone.metrics(req), together.metrics(req), strict=True)
        )
    assert together.n_simulations == 2  # one call for the train windows, one for the test window


def _nan_equal(a: dict, b: dict) -> bool:
    return all(
        (x == y) or (isinstance(x, float) and math.isnan(x) and math.isnan(y))
        for x, y in ((a[k], b[k]) for k in a if k != "params")
    )


def test_weight_anchor_is_a_prefix_of_one_simulation_from_the_first_bar(weight_setup) -> None:
    bars, mask, _ = weight_setup
    scorer = _weight_scorer(bars, mask)
    early = WindowReq("anchor", 0, None, pd.Timestamp("2020-12-01", tz="UTC"))
    late = WindowReq("anchor", 1, None, pd.Timestamp("2021-06-01", tz="UTC"))
    scorer.prepare([early, late])
    assert scorer.n_simulations == 1
    for g in (0, 5):
        a, b = scorer.equity(early, g), scorer.equity(late, g)
        assert np.array_equal(a.to_numpy(), b.to_numpy()[: len(a)])
        assert a.index[0] == bars.close.index[0]


def test_a_weight_wfo_runs_end_to_end_and_needs_an_eligibility_mask() -> None:
    bars = h.make_bars(seed=8, n_bars=1100, n_syms=14)
    mask = bars.close.notna().cumsum() >= 60
    args = (
        "xs_momentum",
        XsMomentum,
        LabConfig(),
        bars,
        bars.close.iloc[:, 0],
        "2020-06-01",
        "2021-12-01",
        "t",
        dict(h.BASE),
        XsMomentum.grid,
    )
    result = run_wfo(*args, mask=mask, verbose=False)
    assert len(result.fold_results) >= 1 and result.diagnostics["strategy_kind"] == "weights"
    assert result.diagnostics["order_sequence"] == "auto"
    assert result.diagnostics["stop_exit_price"] is None
    assert result.diagnostics["n_simulations"] <= 4  # anchor, train, test, live
    with pytest.raises(ValueError, match="weight strategies require"):
        run_wfo(*args, verbose=False)


def test_weight_scorer_refuses_reordered_symbols_within_one_group() -> None:
    class ReorderedSymbols(XsMomentum):
        def build(self, bars, grid=None, rebal=None, eligible=None):
            result = super().build(bars, grid, rebal, eligible)
            n = len(bars.symbols)
            order = list(range(n)) + list(reversed(range(n, 2 * n)))
            order += list(range(2 * n, len(result.columns)))
            return result.iloc[:, order]

    bars = h.make_bars(seed=8, n_bars=400, n_syms=4)
    scorer = _weight_scorer(bars, bars.close.notna())
    scorer.strategy = ReorderedSymbols()
    req = WindowReq("train", 0, bars.close.index[20], bars.close.index[300])
    with pytest.raises(ValueError, match="symbol.*order"):
        scorer.prepare([req])


def test_weight_scorer_refuses_unrecognized_combination_level() -> None:
    class WrongLevel(XsMomentum):
        def build(self, bars, grid=None, rebal=None, eligible=None):
            result = super().build(bars, grid, rebal, eligible)
            names = list(result.columns.names)
            names[0] = "unrecognized"
            return result.rename_axis(columns=names)

    bars = h.make_bars(seed=8, n_bars=400, n_syms=4)
    scorer = _weight_scorer(bars, bars.close.notna())
    scorer.strategy = WrongLevel()
    req = WindowReq("train", 0, bars.close.index[20], bars.close.index[300])
    with pytest.raises(ValueError, match="combination.*levels"):
        scorer.prepare([req])


@pytest.mark.parametrize("damage", ["incomplete", "mixed_labels", "duplicate_symbol"])
def test_weight_scorer_rejects_ambiguous_group_blocks(damage: str) -> None:
    from ggTrader.lab.strategies.contract import group_columns

    bars = h.make_bars(seed=8, n_bars=20, n_syms=4)
    scorer = _weight_scorer(bars, bars.close.notna())
    columns = group_columns(scorer.grid.combos(), scorer.grid.swept(), bars.symbols)
    labels = columns.to_list()
    n = len(bars.symbols)
    if damage == "incomplete":
        labels.pop()
    elif damage == "mixed_labels":
        labels[1] = (*labels[n][:-1], labels[1][-1])
    else:
        labels[1] = labels[0]
    frame = pd.DataFrame(columns=pd.MultiIndex.from_tuples(labels, names=columns.names))
    with pytest.raises(ValueError, match="weight"):
        scorer._check_weight_order(frame, len(labels) // len(scorer.names), len(scorer.names))
