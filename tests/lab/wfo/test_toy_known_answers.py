"""Integration of the engine walk-forward on a closed-form toy, against an independent oracle.

``ToyHold`` has an exact equity curve (see helpers), so the simulation, the window mapping, the
scoring, the selection, the gates, the anchor, the breaker and the OOS stitching can each be
checked against numbers that no vectorbt call produced. The oracle below re-derives every fold
from label slicing of the closed-form equity and the pure policy functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.engine import stats
from ggTrader.lab.strategies.contract import ParamGrid
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.wfo import breaker, folds, gates, inputs, run_wfo, select
from ggTrader.lab.wfo.anchor import pick_anchor
from tests.lab.wfo import helpers as h

ES, EE = "2020-06-01", "2022-03-01"  # 3 folds on 1300 bars from 2019-03-01


def _rows(
    close: pd.Series, start, end, delays=h.DELAYS, name="toy", guarded: bool = True
) -> list[dict]:
    """Metric rows of the oracle equity over [start, end] by label slicing (independent path).

    ``guarded`` rows (train and anchor windows, which feed a selection) apply the P5b degeneracy
    guards from the oracle's own entry count: no entry in the window blanks Sharpe and Sortino.
    """
    rows = []
    for delay in delays:
        eq = h.oracle_equity(close, delay).loc[:end]
        window = eq.loc[start:end]
        rets = window.pct_change(fill_method=None)
        entries = int(delay >= 0 and window.index[0] <= close.index[delay] <= end)
        table = (
            stats.window_stats(rets, 0, len(window), n_trades=pd.Series([entries]))
            if guarded
            else stats.window_stats(rets, 0, len(window), min_losing_days=0)
        )
        rows.append(
            {
                "combo": inputs.combo_name(name, {"delay": delay}),
                "params": {"delay": delay},
                "sharpe": float(table.loc["sharpe"].iloc[0]),
                "sortino": float(table.loc["sortino"].iloc[0]),
                "cagr_pct": float(table.loc["cagr"].iloc[0]) * 100,
                "max_drawdown_pct": float(table.loc["max_drawdown"].iloc[0]) * 100,
                "total_return_pct": (window.iloc[-1] / window.iloc[0] - 1) * 100,
                "n_trades": entries,
            }
        )
    return rows


@pytest.fixture(scope="module")
def toy():
    bars = h.toy_bars()
    result = run_wfo(
        "toy",
        h.ToyHold,
        LabConfig(),
        bars,
        bars.close["X"],
        ES,
        EE,
        "test",
        dict(h.TOY_CONFIG),
        h.ToyHold.grid,
        verbose=False,
        allow_unmasked=True,
    )
    return bars, result


def test_engine_equity_equals_the_closed_form(toy) -> None:
    from ggTrader.lab.wfo.scorers import SignalScorer

    bars, _ = toy
    combos = h.ToyHold.grid.combos()
    names = [inputs.combo_name("toy", c) for c in combos]
    scorer = SignalScorer(
        h.ToyHold(), bars, h.ToyHold.grid, names, combos, None, dict(h.TOY_CONFIG), 1 << 30
    )
    scorer.prepare([])
    for g, delay in enumerate(h.DELAYS):
        np.testing.assert_allclose(
            scorer._val[:, g], h.oracle_equity(bars.close["X"], delay).to_numpy(), rtol=1e-12
        )


def test_folds_match_the_legacy_recurrence(toy) -> None:
    _, result = toy
    expected = folds.generate_folds(pd.Timestamp(ES, tz="UTC"), pd.Timestamp(EE, tz="UTC"))
    assert len(result.fold_results) == len(expected) == 3
    for row, fold in zip(result.fold_results, expected, strict=True):
        assert (row["train_start"], row["train_end"]) == (fold.train_start, fold.train_end)
        assert (row["test_start"], row["test_end"]) == (fold.test_start, fold.test_end)


class ToyInsideWindows(h.ToyHold):
    """Entries fall inside the first folds' train windows, so the selection is not degenerate."""

    grid = ParamGrid.from_sweep({"delay": [330, 400, 480]})


def test_every_fold_matches_an_independent_oracle(toy) -> None:
    _check_against_oracle(*toy, h.DELAYS, h.ToyHold.grid)


def test_the_oracle_also_holds_when_entries_fall_inside_the_train_windows() -> None:
    """With the zero-entry guard the carried-in toy above ranks every combination NaN (P5b); this
    grid has entries inside folds 1 and 2, so winners, gates and the anchor really differ."""
    bars = h.toy_bars()
    result = run_wfo(
        "toy", ToyInsideWindows, LabConfig(), bars, bars.close["X"], ES, EE, "test",
        dict(h.TOY_CONFIG), ToyInsideWindows.grid, verbose=False, allow_unmasked=True
    )  # fmt: skip
    _check_against_oracle(bars, result, (330, 400, 480), ToyInsideWindows.grid)
    finite = [r for r in result.fold_results if np.isfinite(r["is_sharpe"])]
    assert len(finite) >= 2  # the guard did not blank every fold


def _check_against_oracle(bars, result, delays, grid) -> None:
    close = bars.close["X"]
    oracle_state = breaker.WfoState()
    running = h.START_CASH
    curves = []
    for fold, row in zip(
        folds.generate_folds(pd.Timestamp(ES, tz="UTC"), pd.Timestamp(EE, tz="UTC")),
        result.fold_results,
        strict=True,
    ):
        train = _rows(close, fold.train_start, fold.train_end, delays=delays)
        anchor = pick_anchor(_rows(close, close.index[0], fold.train_end, delays=delays))
        best = select.best_index(select.composite_score(train))
        winner = train[best]
        ndh = gates.run_ndh(train, grid.combos(), best, 0.85)
        window = h.oracle_equity(close, winner["params"]["delay"]).loc[
            fold.train_start : fold.train_end
        ]
        dsr_ok, dsr_value = gates.run_dsr(
            window.pct_change(fill_method=None).iloc[1:], len(delays), 0.80
        )
        gates_ok = ndh.passed and dsr_ok
        used_anchor = (not gates_ok) or oracle_state.halted
        deploy = anchor.params if used_anchor else winner["params"]
        test_eq = h.oracle_equity(close, deploy["delay"]).loc[fold.test_start : fold.test_end]
        curves.append(running * (test_eq / test_eq.iloc[0]))
        running = float(curves[-1].iloc[-1])
        test_row = _rows(
            close, fold.test_start, fold.test_end, delays=[deploy["delay"]], guarded=False
        )[0]
        wfe = breaker.compute_wfe(winner["sharpe"], test_row["sharpe"])
        oracle_state.wfe_history.append(wfe)
        if np.isfinite(test_row["sharpe"]):
            oracle_state.oos_sharpes.append(test_row["sharpe"])
        entering = oracle_state.halted
        if not oracle_state.halted:
            oracle_state = breaker.check_circuit_breaker(oracle_state)
        else:
            oracle_state = breaker.check_shadow_reentry(oracle_state, ndh.passed, dsr_ok, wfe)
        assert row["winner_combo"] == winner["combo"]
        assert (row["ndh_passed"], row["dsr_passed"], row["gates_passed"]) == (
            ndh.passed,
            dsr_ok,
            gates_ok,
        )
        assert row["dsr_value"] == pytest.approx(dsr_value, abs=1e-12)
        assert row["used_anchor"] == used_anchor and row["halted_entering"] == entering
        assert row["halted"] == oracle_state.halted
        assert row["is_sharpe"] == pytest.approx(winner["sharpe"], rel=1e-12, nan_ok=True)
        assert row["oos_sharpe"] == pytest.approx(test_row["sharpe"], rel=1e-12, nan_ok=True)
    expected = pd.concat(curves)
    expected = expected[~expected.index.duplicated(keep="last")]
    pd.testing.assert_series_equal(
        result.oos_equity, expected, check_names=False, check_freq=False, rtol=1e-12
    )


def test_the_oos_curve_is_continuous_across_folds(toy) -> None:
    _, result = toy
    curve = result.oos_equity
    assert curve.index.is_unique and curve.index.is_monotonic_increasing
    assert curve.iloc[0] == pytest.approx(h.START_CASH)
    for row in result.fold_results[1:]:
        before = curve.loc[: row["test_start"]]
        step = curve.loc[row["test_start"]] / before.iloc[-2] - 1 if len(before) > 1 else 0.0
        assert abs(step) < 0.2  # one ordinary day, not a jump from re-basing


def test_live_params_come_from_the_last_twelve_months(toy) -> None:
    _, result = toy
    live = result.live_params
    assert live["train_end"] == pd.Timestamp(EE, tz="UTC")
    assert live["train_start"] == pd.Timestamp(EE, tz="UTC") - pd.DateOffset(months=12)
    assert live["combo"].startswith("toy__delay")


def test_table_and_diagnostics(toy) -> None:
    _, result = toy
    assert "3 combos x 3 folds" in result.table and "Anchor Set" in result.table
    d = result.diagnostics
    assert d["n_simulations"] == 1 and d["legacy_equivalent_simulations"] >= 7
    assert d["od12_ndh_flips_folds"] == []
    assert d["definitions"] == "vbt" and d["engine"] == "rust"


def test_no_valid_folds_returns_the_legacy_string() -> None:
    bars = h.toy_bars()
    out = run_wfo(
        "toy",
        h.ToyHold,
        LabConfig(),
        bars,
        bars.close["X"],
        "2021-01-01",
        "2021-06-01",
        "t",
        dict(h.TOY_CONFIG),
        h.ToyHold.grid,
        verbose=False,
        allow_unmasked=True,
    )
    assert isinstance(out, str) and "no valid folds" in out


def test_list_grids_and_legacy_frames_are_accepted() -> None:
    bars = h.toy_bars()
    out = run_wfo(
        "toy",
        h.ToyHold,
        LabConfig(),
        bars.legacy_frame(),
        bars.close["X"],
        ES,
        EE,
        "t",
        dict(h.TOY_CONFIG, FREQ="1d"),
        [{"delay": d} for d in h.DELAYS],
        verbose=False,
        allow_unmasked=True,
    )
    ref = run_wfo(
        "toy",
        h.ToyHold,
        LabConfig(),
        bars,
        bars.close["X"],
        ES,
        EE,
        "t",
        dict(h.TOY_CONFIG),
        h.ToyHold.grid,
        verbose=False,
        allow_unmasked=True,
    )
    assert out.table == ref.table
    pd.testing.assert_series_equal(out.oos_equity, ref.oos_equity)


def test_legacy_strategy_classes_are_refused_with_a_pointer() -> None:
    bars = h.toy_bars()
    with pytest.raises(ValueError, match="legacy-engine-vbt0285"):
        run_wfo(
            "toy",
            h.LegacyToy,
            LabConfig(),
            bars,
            bars.close["X"],
            ES,
            EE,
            "t",
            dict(h.TOY_CONFIG),
            [{"delay": 3}],
            verbose=False,
            allow_unmasked=True,
        )
