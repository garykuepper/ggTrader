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
    assert toy[1].diagnostics["intraday_stops"] is False
    assert toy[1].diagnostics["halt_mask_sha256"] is None
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


def test_signal_scorer_can_opt_into_intraday_stops() -> None:
    from dataclasses import replace

    from ggTrader.lab.wfo.scorers import SignalScorer

    original = h.toy_bars(n_bars=20)
    close = original.close * 0 + 100.0
    low = close.copy()
    low.iloc[4, 0] = 80.0
    bars = replace(original, close=close, open=close.copy(), high=close.copy(), low=low)
    grid = ParamGrid.single({"delay": 3})
    args = (
        h.ToyHold(),
        bars,
        grid,
        ["toy"],
        grid.combos(),
        None,
        {**h.TOY_CONFIG, "ts_stop": 0.1},
        1 << 30,
    )
    baseline = SignalScorer(*args)
    baseline.prepare([])
    intraday = SignalScorer(*args, intraday_stops=True)
    intraday.prepare([])
    assert baseline._val[4, 0] == 10000.0
    assert intraday._val[4, 0] == 9500.0


def test_signal_scorer_applies_explicit_close_stop_fill() -> None:
    from dataclasses import replace

    from ggTrader.lab.wfo.scorers import SignalScorer

    original = h.toy_bars(n_bars=20)
    close = original.close * 0 + 100.0
    close.iloc[4:, :] = 85.0
    bars = replace(original, close=close, open=close, high=close, low=close)
    grid = ParamGrid.single({"delay": 3})
    args = (
        h.ToyHold(),
        bars,
        grid,
        ["toy"],
        grid.combos(),
        None,
        {**h.TOY_CONFIG, "ts_stop": 0.1, "SLIPPAGE": 0.01},
        1 << 30,
    )
    default = SignalScorer(*args, order_sequence="auto")
    market = SignalScorer(*args, order_sequence="auto", stop_exit_price="stopmarket")
    default.prepare([])
    market.prepare([])
    assert market._val[4, 0] < default._val[4, 0]


def test_signal_scorer_applies_explicit_order_sequence() -> None:
    from dataclasses import replace

    from ggTrader.lab.strategies.contract import SignalOut, group_columns
    from ggTrader.lab.wfo.scorers import SignalScorer

    original = h.toy_bars(n_bars=3)
    close = pd.DataFrame({"A": [100.0, 100.0, 110.0], "B": [100.0] * 3}, index=original.close.index)
    bars = replace(original, close=close, open=close, high=close, low=close)
    grid = ParamGrid.single({"delay": 1})

    class Swap:
        def build(self, bars, grid):
            columns = group_columns(grid.combos(), [], ["A", "B"])
            entries = pd.DataFrame(
                [[False, True], [True, False], [False, False]],
                index=bars.close.index,
                columns=columns,
            )
            exits = pd.DataFrame(
                [[False, False], [False, True], [False, False]],
                index=bars.close.index,
                columns=columns,
            )
            return SignalOut(entries=entries, exits=exits, combos=tuple(grid.combos()))

    args = (
        Swap(),
        bars,
        grid,
        ["toy"],
        grid.combos(),
        None,
        {**h.TOY_CONFIG, "SIGNAL_POSITION_SIZE": 1.0},
        1 << 30,
    )
    default = SignalScorer(*args, order_sequence="default")
    automatic = SignalScorer(*args, order_sequence="auto")
    default.prepare([])
    automatic.prepare([])
    assert default._val[-1, 0] == pytest.approx(10000.0)
    assert automatic._val[-1, 0] == pytest.approx(11000.0)


@pytest.mark.parametrize("setting", [{}, {"stop_entry_price": None}])
def test_wfo_records_and_uses_intraday_stop_mode(monkeypatch, tmp_path, setting) -> None:
    import json
    from dataclasses import replace

    from ggTrader.lab.wfo.cli_support import write_wfo_json
    from ggTrader.lab.wfo.report import result_payload
    from ggTrader.lab.wfo.scorers import SignalScorer

    original = h.toy_bars()
    low = original.low.copy()
    low.iloc[4, 0] = original.close.iloc[3, 0] * 0.5
    bars = replace(original, low=low)
    observed = []
    prepare = SignalScorer.prepare

    def capture(scorer, requests):
        prepare(scorer, requests)
        observed.append(scorer._val[4, 0])

    monkeypatch.setattr(SignalScorer, "prepare", capture)
    result = run_wfo(
        "toy",
        h.ToyHold,
        LabConfig(),
        bars,
        bars.close["X"],
        ES,
        EE,
        "test",
        {**h.TOY_CONFIG, "ts_stop": 0.1, **setting},
        h.ToyHold.grid,
        verbose=False,
        allow_unmasked=True,
        intraday_stops=True,
    )
    assert result.diagnostics["intraday_stops"] is True
    assert result.diagnostics["order_sequence"] == "auto"
    assert result.diagnostics["stop_exit_price"] == "stopmarket"
    assert result.diagnostics["stop_entry_price"] == "close"
    assert result.diagnostics["stop_settings"] == {
        "trailing_fraction": 0.1,
        "take_profit_fraction": None,
        "atr_multiplier": None,
        "atr_period": None,
        "time_stop_bars": None,
    }
    assert result_payload(result)["diagnostics"]["intraday_stops"] is True
    assert observed == pytest.approx([9500.0])
    result_file = tmp_path / "synthetic_wfo.json"
    write_wfo_json(result, bars, str(result_file), strategy="toy")
    assert json.loads(result_file.read_text())["diagnostics"]["intraday_stops"] is True
    assert (
        json.loads(result_file.read_text())["diagnostics"]["stop_settings"]["trailing_fraction"]
        == 0.1
    )


def test_wfo_reports_no_stop_fill_when_no_price_stop_is_used() -> None:
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
        stop_exit_price="stopmarket",
    )
    assert result.diagnostics["stop_exit_price"] is None
    assert result.diagnostics["stop_entry_price"] is None
    assert result.diagnostics["time_stop_warning"] is None


@pytest.mark.parametrize("intraday", [False, True])
def test_wfo_halt_mask_reaches_simulation_and_is_recorded(intraday, tmp_path) -> None:
    import json

    from ggTrader.lab.wfo.cli_support import write_wfo_json

    bars = h.toy_bars()
    halt_mask = pd.DataFrame(False, index=bars.close.index, columns=bars.close.columns)
    args = (
        "toy",
        h.ToyHold,
        LabConfig(),
        bars,
        bars.close["X"],
        ES,
        EE,
        "test",
        {**h.TOY_CONFIG, **({"ts_stop": 0.9} if intraday else {})},
        h.ToyHold.grid,
    )
    result = run_wfo(
        *args, verbose=False, allow_unmasked=True, halt_mask=halt_mask, intraday_stops=intraday
    )
    assert result.diagnostics["halt_mask_supplied"] is True
    original_hash = result.diagnostics["halt_mask_sha256"]
    assert isinstance(original_hash, str) and len(original_hash) == 64
    output = tmp_path / "result.json"
    write_wfo_json(result, bars, str(output), strategy="toy")
    assert json.loads(output.read_text())["diagnostics"]["halt_mask_sha256"] == original_hash
    halt_mask.iloc[0, 0] = True  # Earlier than every synthetic entry.
    changed = run_wfo(
        *args, verbose=False, allow_unmasked=True, halt_mask=halt_mask, intraday_stops=intraday
    )
    assert changed.diagnostics["halt_mask_sha256"] != original_hash
    halt_mask.iloc[0, 0] = False
    halt_mask.iloc[5, 0] = True
    with pytest.raises(ValueError, match="halt during an open position"):
        run_wfo(
            *args,
            verbose=False,
            allow_unmasked=True,
            halt_mask=halt_mask,
            intraday_stops=intraday,
            max_bytes=1,
        )


def test_halt_mask_hash_includes_dates_and_symbol_labels() -> None:
    from ggTrader.lab.wfo.run import _halt_mask_hash

    bars = h.toy_bars(n_bars=4)
    mask = pd.DataFrame(False, index=bars.close.index, columns=bars.close.columns)
    assert _halt_mask_hash(mask) == _halt_mask_hash(mask.copy())
    assert _halt_mask_hash(mask) != _halt_mask_hash(mask.rename(columns={"X": "Y"}))
    changed_dates = mask.copy()
    changed_dates.index = changed_dates.index + pd.Timedelta(days=1)
    assert _halt_mask_hash(mask) != _halt_mask_hash(changed_dates)


def test_halt_mask_hash_preserves_recorded_format() -> None:
    from ggTrader.lab.wfo.run import _halt_mask_hash

    mask = pd.DataFrame({"X": [True, False]}, index=pd.date_range("2024-01-01", periods=2))
    assert _halt_mask_hash(mask) == (
        "05338c5ca3cd00131015282333301f5f742bfaee86cfc23c965a083ea9b18d77"
    )


def test_wfo_persists_time_stop_warning(tmp_path) -> None:
    import json
    from dataclasses import replace

    from ggTrader.lab.wfo.cli_support import write_wfo_json

    class TimeStop(h.ToyHold):
        def build(self, bars, grid=None):
            return replace(super().build(bars, grid), td_stop=5)

    bars = h.toy_bars()
    with pytest.warns(UserWarning, match="td_stop"):
        result = run_wfo(
            "toy",
            TimeStop,
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
    path = tmp_path / "time_stop.json"
    write_wfo_json(result, bars, str(path), strategy="toy")
    warning = json.loads(path.read_text())["diagnostics"]["time_stop_warning"]
    assert "entry signals" in warning
    assert "filled positions" in warning


def test_wfo_records_explicit_close_only_sequence(monkeypatch) -> None:
    from ggTrader.engine import portfolio

    captured = []
    original = portfolio.run_signals

    def capture(*args, **kwargs):
        captured.append(args[3].get("stop_entry_price"))
        return original(*args, **kwargs)

    monkeypatch.setattr(portfolio, "run_signals", capture)
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
        {**h.TOY_CONFIG, "ts_stop": 0.1, "stop_entry_price": "fillprice"},
        h.ToyHold.grid,
        verbose=False,
        allow_unmasked=True,
        order_sequence="auto",
        stop_exit_price="stopmarket",
    )
    assert result.diagnostics["intraday_stops"] is False
    assert result.diagnostics["order_sequence"] == "auto"
    assert result.diagnostics["stop_entry_price"] == "fillprice"
    assert result.diagnostics["stop_exit_price"] == "stopmarket"
    assert captured == ["fillprice"]


def test_wfo_rejects_invalid_stop_fill_rule() -> None:
    bars = h.toy_bars()
    with pytest.raises(ValueError, match="stop_exit_price"):
        run_wfo(
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
            stop_exit_price="invalid",
        )


def test_intraday_stop_mode_rejects_weight_strategies() -> None:
    from types import SimpleNamespace

    from ggTrader.lab.wfo.run import _make_scorer

    with pytest.raises(ValueError, match="signal strategy"):
        _make_scorer(
            SimpleNamespace(kind="weights"),
            h.toy_bars(n_bars=20),
            h.ToyHold.grid,
            [],
            [],
            None,
            h.TOY_CONFIG,
            1 << 30,
            0,
            True,
        )


@pytest.mark.parametrize(
    "option",
    [{"order_sequence": "auto"}, {"stop_exit_price": "stopmarket"}, {"halt_mask": pd.DataFrame()}],
)
def test_signal_execution_options_reject_weight_strategies(option) -> None:
    from types import SimpleNamespace

    from ggTrader.lab.wfo.run import _make_scorer

    with pytest.raises(ValueError, match="signal strategy"):
        _make_scorer(
            SimpleNamespace(kind="weights"),
            h.toy_bars(n_bars=20),
            h.ToyHold.grid,
            [],
            [],
            None,
            h.TOY_CONFIG,
            1 << 30,
            0,
            False,
            **option,
        )


def test_intraday_scorer_allows_late_listing_before_first_entry() -> None:
    from dataclasses import replace

    from ggTrader.lab.wfo.scorers import SignalScorer

    original = h.toy_bars(n_bars=20)
    close = original.close * 0 + 100.0
    close.iloc[:2, 0] = float("nan")
    low = close.copy()
    low.iloc[4, 0] = 80.0
    bars = replace(original, close=close, open=close.copy(), high=close.copy(), low=low)
    grid = ParamGrid.single({"delay": 3})
    scorer = SignalScorer(
        h.ToyHold(),
        bars,
        grid,
        ["toy"],
        grid.combos(),
        None,
        {**h.TOY_CONFIG, "ts_stop": 0.1},
        1 << 30,
        intraday_stops=True,
    )
    scorer.prepare([])
    assert scorer._val[4, 0] == 9500.0


def test_scorer_records_actual_strategy_stop_overrides() -> None:
    from dataclasses import replace

    from ggTrader.lab.wfo.scorers import SignalScorer

    class WithStops(h.ToyHold):
        def build(self, bars, grid=None):
            return replace(super().build(bars, grid), sl_stop=0.2, tp_stop=0.3, td_stop=5)

    bars = h.toy_bars(n_bars=20)
    grid = ParamGrid.single({"delay": 3})
    scorer = SignalScorer(
        WithStops(),
        bars,
        grid,
        ["toy"],
        grid.combos(),
        None,
        {**h.TOY_CONFIG, "ts_stop": 0.1, "tp_stop": 0.05, "atr_mult": 2.0},
        1 << 30,
    )
    with pytest.warns(UserWarning, match="td_stop"):
        scorer.prepare([])
    assert scorer.stop_settings == {
        "trailing_fraction": 0.2,
        "take_profit_fraction": 0.3,
        "atr_multiplier": None,
        "atr_period": None,
        "time_stop_bars": 5,
    }


def test_scorer_records_atr_stop_period() -> None:
    from ggTrader.lab.wfo.scorers import SignalScorer

    bars = h.toy_bars(n_bars=20)
    grid = ParamGrid.single({"delay": 3})
    scorer = SignalScorer(
        h.ToyHold(),
        bars,
        grid,
        ["toy"],
        grid.combos(),
        None,
        {**h.TOY_CONFIG, "atr_mult": 2.0, "atr_period": 5},
        1 << 30,
    )
    scorer.prepare([])
    assert scorer.stop_settings["atr_multiplier"] == 2.0
    assert scorer.stop_settings["atr_period"] == 5
