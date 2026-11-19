"""The private helpers of ``run_wfo`` (``_score_fold`` and the ones added after it) on their own.

The scripted scorer of the characterization tests answers every window request from a script, so
a single fold can be scored without a simulation and every call it makes can be read back.
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from ggTrader.lab.wfo import inputs, run
from ggTrader.lab.wfo.anchor import NO_ANCHOR
from ggTrader.lab.wfo.breaker import WfoPolicy, WfoState
from ggTrader.lab.wfo.fold_record import FoldRecord
from ggTrader.lab.wfo.folds import Fold
from tests.lab.wfo import helpers as h
from tests.lab.wfo.test_run_wfo_characterization import (
    BAD_TRAIN,
    _fold_script,
    _folds,
    _row,
    _rows,
    scripted_scorer_class,
)


def _context(
    script: dict[str, Any], calls: list, *, verbose: bool = False, n_folds: int | None = None
) -> run._FoldContext:
    """A fold context over the toy lattice with the scripted scorer."""
    _, combos = inputs.resolve_grid(h.ToyHold, h.ToyHold.grid)
    names = [inputs.combo_name("toy", c) for c in combos]
    return run._FoldContext(
        scorer=scripted_scorer_class(script, calls)(),
        strategy_name="toy",
        names=names,
        name_to_idx={n: i for i, n in enumerate(names)},
        combos=combos,
        ndh_threshold=0.85,
        dsr_threshold=0.80,
        policy=WfoPolicy(),
        n_folds=len(script["folds"]) if n_folds is None else n_folds,
        verbose=verbose,
    )


def _score(
    script: dict[str, Any],
    i: int = 0,
    *,
    state: WfoState | None = None,
    running_value: float = 100_000.0,
    verbose: bool = False,
) -> tuple[run._FoldOutcome, list, str, Fold, WfoState]:
    calls: list = []
    ctx = _context(script, calls, verbose=verbose)
    folds = _folds()
    state = WfoState() if state is None else state
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        outcome = run._score_fold(ctx, i, folds[i], state, running_value)
    return outcome, calls, out.getvalue(), folds[i], state


def _script(**over: Any) -> dict[str, Any]:
    folds = _folds()
    return {"live": [], "folds": [_fold_script(f, **over) for f in folds]}


def test_score_fold_skipped_returns_state_and_running_value_unchanged() -> None:
    state = WfoState(wfe_history=[0.5], oos_sharpes=[0.3])
    outcome, _, _, fold, _ = _score(_script(train=[]), state=state, running_value=123.5)
    assert outcome.record is None
    assert outcome.skipped == {"fold_num": 1, **fold._asdict(), "reason": "no_train_metrics"}
    assert list(outcome.skipped) == ["fold_num", *fold._fields, "reason"]
    assert outcome.state is state
    assert outcome.running_value == 123.5
    assert (outcome.winner, outcome.oos_curve) == (None, None)
    assert (outcome.ran_test, outcome.od12_flip) == (False, False)
    assert state.wfe_history == [0.5] and state.oos_sharpes == [0.3]
    # a skipped fold still carries the anchor picked from the anchor window
    assert outcome.anchor.combo != NO_ANCHOR.combo


def test_score_fold_appends_to_the_state_it_was_given() -> None:
    state = WfoState(wfe_history=[0.9], oos_sharpes=[0.7])
    history, sharpes = state.wfe_history, state.oos_sharpes
    outcome, *_ = _score(_script(), state=state)
    assert state.wfe_history is history and len(history) == 2
    assert state.oos_sharpes is sharpes and len(sharpes) == 2
    # the breaker hands back a copy, so the returned state is a different object
    assert outcome.state is not state
    assert outcome.state.wfe_history == history


def test_score_fold_calls_the_scorer_in_the_pinned_order() -> None:
    outcome, calls, _, fold, _ = _score(_script())
    assert [(c[0], c[1]) for c in calls] == [
        ("metrics", "anchor"),
        ("metrics", "train"),
        ("returns", "train"),
        ("trade_counts", "train"),
        ("metrics", "test"),
        ("equity", "test"),
    ]
    by_method = {(c[0], c[1]): c for c in calls}
    assert by_method[("trade_counts", "train")][5] is True
    assert by_method[("metrics", "anchor")][3:5] == (None, fold.train_end)
    assert by_method[("metrics", "train")][3:5] == (fold.train_start, fold.train_end)
    assert by_method[("metrics", "test")][3:5] == (fold.test_start, fold.test_end)
    assert by_method[("metrics", "test")][5] == [1]  # delay 50, the deployed winner
    assert outcome.ran_test is True and isinstance(outcome.record, FoldRecord)


def test_score_fold_verbose_off_prints_nothing() -> None:
    _, _, text, *_ = _score(_script(), verbose=False)
    assert text == ""


def test_score_fold_verbose_prints_the_two_fragments() -> None:
    outcome, _, text, fold, _ = _score(_script(), verbose=True)
    header, _, result = text.partition(" → done")
    assert header.startswith(f"  Fold 1/3: train {fold.train_start.date()}→")
    assert "\n" not in header  # the header is an unterminated line
    assert result.endswith("\n") and result.count("\n") == 1
    assert outcome.record is not None


def test_score_fold_verbose_skipped_fold_leaves_the_header_unterminated() -> None:
    outcome, _, text, *_ = _score(_script(train=[]), verbose=True)
    assert outcome.record is None
    assert text.startswith("  Fold 1/3:") and not text.endswith("\n")


def test_score_fold_flat_curve_when_no_anchor() -> None:
    no_anchor = [_row(d, float("nan")) for d in h.DELAYS]  # never traded: no candidate
    script = _script(anchor=no_anchor, train=_rows(BAD_TRAIN))
    outcome, calls, _, fold, _ = _score(script, running_value=777.0)
    assert outcome.record is not None and outcome.record.deployed_combo == "none"
    assert outcome.ran_test is False
    assert [c[0] for c in calls][-1] == "window_dates"
    curve = outcome.oos_curve
    assert curve is not None
    assert (curve == 777.0).all()
    assert curve.index.equals(pd.bdate_range(fold.test_start, fold.test_end, tz="UTC"))
    assert outcome.running_value == 777.0
    assert np.isnan(outcome.record.oos_sharpe)


# --- _stitch_oos ---------------------------------------------------------------------------


def _curve(values: list[float], start: str) -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=index, dtype=float)


def test_stitch_oos_no_curves_gives_empty_float_series_and_nan_dicts() -> None:
    spy = _curve([100.0, 101.0, 102.0], "2024-01-01")
    equity, oos_metrics, spy_metrics = run._stitch_oos([], spy, 1000.0)
    assert equity.empty
    assert equity.dtype == float
    assert all(np.isnan(v) for v in oos_metrics.values())
    assert all(np.isnan(v) for v in spy_metrics.values())


def test_stitch_oos_keeps_last_of_duplicated_dates() -> None:
    first = _curve([100.0, 101.0, 102.0], "2024-01-01")
    second = _curve([500.0, 501.0], "2024-01-03")
    spy = _curve([1.0] * 6, "2024-01-01")
    equity, _, _ = run._stitch_oos([first, second], spy, 100.0)
    assert list(equity.index) == list(pd.date_range("2024-01-01", periods=4, freq="D", tz="UTC"))
    assert list(equity.values) == [100.0, 101.0, 500.0, 501.0]


def test_stitch_oos_nan_dicts_have_three_keys() -> None:
    spy = _curve([100.0, 101.0], "2024-01-01")
    _, oos_metrics, spy_metrics = run._stitch_oos([], spy, 1000.0)
    keys = ["sharpe", "cagr_pct", "max_drawdown_pct"]
    assert list(oos_metrics) == keys
    assert list(spy_metrics) == keys


def test_stitch_oos_nan_dicts_are_not_shared() -> None:
    spy = _curve([100.0, 101.0], "2024-01-01")
    _, oos_metrics, spy_metrics = run._stitch_oos([], spy, 1000.0)
    assert oos_metrics is not spy_metrics
    oos_metrics["sharpe"] = 1.0
    assert np.isnan(spy_metrics["sharpe"])
    _, again, _ = run._stitch_oos([], spy, 1000.0)
    assert np.isnan(again["sharpe"])


def test_stitch_oos_spy_with_one_date_gives_nan_spy() -> None:
    curve = _curve([100.0, 101.0, 103.0], "2024-01-01")
    # Only the last curve date has a SPY close: the leading NaNs are dropped, one date is left.
    spy = _curve([400.0], "2024-01-03")
    equity, oos_metrics, spy_metrics = run._stitch_oos([curve], spy, 100.0)
    assert list(equity.values) == [100.0, 101.0, 103.0]
    assert len(oos_metrics) == 6
    assert list(spy_metrics) == ["sharpe", "cagr_pct", "max_drawdown_pct"]
    assert all(np.isnan(v) for v in spy_metrics.values())


def test_stitch_oos_scales_spy_to_start_cash(monkeypatch: Any) -> None:
    from ggTrader.lab.wfo.report import curve_summary

    curve = _curve([100.0, 102.0, 101.0, 105.0], "2024-01-01")
    spy = _curve([400.0, 404.0, 396.0, 408.0], "2024-01-01")
    seen = []

    def record_summary(values: pd.Series) -> dict:
        seen.append(values)
        return curve_summary(values)

    monkeypatch.setattr(run, "curve_summary", record_summary)
    _, _, spy_metrics = run._stitch_oos([curve], spy, 100.0)
    expected = curve_summary(100.0 * (spy / spy.iloc[0]))
    assert spy_metrics == expected
    pd.testing.assert_series_equal(seen[1], _curve([100.0, 101.0, 99.0, 102.0], "2024-01-01"))


# --- _build_diagnostics --------------------------------------------------------------------

STOP_SETTINGS = {
    "trailing_fraction": 0.1,
    "take_profit_fraction": None,
    "atr_multiplier": None,
    "atr_period": None,
    "time_stop_bars": 5,
}


class _StubScorer:
    """Only what ``_build_diagnostics`` reads from a scorer."""

    def __init__(self, *, stop_settings: dict | None = None, with_stops: bool = False) -> None:
        self.n_simulations = 7
        self.n_chunks = 3
        self.timings = {"build_s": 1.23456, "simulate_s": 2.0}
        if with_stops:
            self.stop_settings = stop_settings


def _diag_inputs(kind: str = "signals", **over: Any) -> run.ScoringInputs:
    bars = h.toy_bars(n_bars=40)
    strategy = type("Strategy", (), {"kind": kind})()
    params = [{"delay": 3}, {"delay": 50}]
    fields: dict[str, Any] = {
        "strategy": strategy,
        "bars": bars,
        "grid": None,
        "names": ["a", "b"],
        "params": params,
        "mask": None,
        "base_config": {},
        "max_bytes": 12345,
        "min_losing_days": 4,
    }
    fields.update(over)
    return run.ScoringInputs(**fields)


def _tally(**over: Any) -> run._Tally:
    fields: dict[str, Any] = {
        "n_folds": 4,
        "n_test_runs": 2,
        "od12_flips": [1, 3],
        "oos_metrics": {"sharpe": 1.0},
        "spy_metrics": {"sharpe": 0.5},
    }
    fields.update(over)
    return run._Tally(**fields)


def _diagnostics(
    scorer: Any = None,
    inputs_: run.ScoringInputs | None = None,
    options: run.SignalSimOptions | None = None,
    tally: run._Tally | None = None,
    *,
    started: float = 0.0,
    fold_started: float = 0.0,
) -> dict[str, Any]:
    return run._build_diagnostics(
        _StubScorer() if scorer is None else scorer,
        _diag_inputs() if inputs_ is None else inputs_,
        run.SignalSimOptions() if options is None else options,
        _tally() if tally is None else tally,
        started=started,
        fold_started=fold_started,
    )


def test_build_diagnostics_has_the_26_keys_in_order() -> None:
    from tests.lab.wfo.test_run_wfo_characterization import DIAGNOSTICS_KEYS_26

    diagnostics = _diagnostics()
    assert tuple(diagnostics) == DIAGNOSTICS_KEYS_26
    assert diagnostics["n_combos"] == 2 and diagnostics["n_symbols"] == 1
    assert diagnostics["n_bars"] == 40 and diagnostics["max_bytes"] == 12345
    assert (diagnostics["n_simulations"], diagnostics["n_chunks"]) == (7, 3)
    assert diagnostics["od12_ndh_flips_folds"] == [1, 3]
    assert diagnostics["oos_metrics"] == {"sharpe": 1.0}
    assert diagnostics["spy_metrics"] == {"sharpe": 0.5}
    assert diagnostics["pit_masked"] is False
    assert diagnostics["guards"] == {
        "min_losing_days": 4,
        "zero_trade_rule": "selection windows (anchor, train, live)",
    }
    assert isinstance(diagnostics["peak_rss_mb"], int)
    masked = _diagnostics(inputs_=_diag_inputs(mask=pd.DataFrame()))
    assert masked["pit_masked"] is True


def test_build_diagnostics_legacy_equivalent_simulations_counts_skipped_folds() -> None:
    # n_folds is every generated fold: a skipped one still adds its two legacy simulations
    assert _diagnostics(tally=_tally(n_folds=4, n_test_runs=2))[
        "legacy_equivalent_simulations"
    ] == (4 * 2 + 2 + 1)
    diagnostics = _diagnostics(tally=_tally(n_folds=5, n_test_runs=0))
    assert diagnostics["n_folds"] == 5
    assert diagnostics["legacy_equivalent_simulations"] == 11


def test_build_diagnostics_weights_run_has_no_stop_settings() -> None:
    diagnostics = _diagnostics(inputs_=_diag_inputs("weights"))
    assert diagnostics["strategy_kind"] == "weights"
    assert diagnostics["stop_settings"] is None
    for key in ("stop_exit_price", "stop_entry_price", "time_stop_warning"):
        assert diagnostics[key] is None
    assert diagnostics["order_sequence"] == "auto"


def test_build_diagnostics_stop_settings_drive_the_stop_keys() -> None:
    scorer = _StubScorer(stop_settings=dict(STOP_SETTINGS), with_stops=True)
    base_config = {"stop_entry_price": "fillprice"}
    diagnostics = _diagnostics(scorer, _diag_inputs(base_config=base_config))
    assert diagnostics["stop_settings"] == STOP_SETTINGS
    assert diagnostics["stop_exit_price"] == "stoplimit"
    assert diagnostics["stop_entry_price"] == "fillprice"
    assert diagnostics["time_stop_warning"] == run.engine_portfolio.TIME_STOP_WARNING
    intraday = _diagnostics(scorer, options=run.SignalSimOptions(intraday_stops=True))
    assert intraday["stop_exit_price"] == "stopmarket"
    assert intraday["stop_entry_price"] == "close"
    explicit = _diagnostics(scorer, options=run.SignalSimOptions(stop_exit_price="stopmarket"))
    assert explicit["stop_exit_price"] == "stopmarket"
    # a settings dict with no price stop leaves the price keys None, the time stop still shows
    time_only = dict(STOP_SETTINGS, trailing_fraction=None)
    quiet = _diagnostics(_StubScorer(stop_settings=time_only, with_stops=True))
    assert quiet["stop_exit_price"] is None and quiet["stop_entry_price"] is None
    assert quiet["time_stop_warning"] == run.engine_portfolio.TIME_STOP_WARNING
    no_time = dict(STOP_SETTINGS, time_stop_bars=None)
    assert (
        _diagnostics(_StubScorer(stop_settings=no_time, with_stops=True))["time_stop_warning"]
        is None
    )


def test_build_diagnostics_order_sequence_precedence() -> None:
    # ``a or b if kind == "signals" else "auto"`` parses as ``(a or b) if ... else "auto"``
    def order(kind: str, **options: Any) -> str:
        return _diagnostics(inputs_=_diag_inputs(kind), options=run.SignalSimOptions(**options))[
            "order_sequence"
        ]

    assert order("signals") == "default"
    assert order("signals", intraday_stops=True) == "auto"
    assert order("weights") == "auto"
    assert order("weights", intraday_stops=True) == "auto"
    assert order("signals", order_sequence="default", intraday_stops=True) == "default"
    assert order("signals", order_sequence="auto") == "auto"
    # an explicit value does not reach a weight run
    assert order("weights", order_sequence="default") == "auto"


def test_build_diagnostics_halt_mask_keys() -> None:
    plain = _diagnostics()
    assert plain["halt_mask_supplied"] is False and plain["halt_mask_sha256"] is None
    idx = pd.bdate_range("2024-01-01", periods=3, tz="UTC")
    mask = pd.DataFrame({"X": [False, True, False]}, index=idx)
    flagged = _diagnostics(options=run.SignalSimOptions(halt_mask=mask))
    assert flagged["halt_mask_supplied"] is True
    assert flagged["halt_mask_sha256"] == run._halt_mask_hash(mask)


def test_build_diagnostics_timings_order(monkeypatch: Any) -> None:
    clock = iter([110.4567, 130.1234])
    monkeypatch.setattr(run, "time", SimpleNamespace(time=lambda: next(clock)))
    diagnostics = _diagnostics(started=100.0, fold_started=105.0)
    timings = diagnostics["timings_s"]
    assert list(timings) == ["build_s", "simulate_s", "folds_s", "total_s"]
    # folds_s reads the clock first, total_s second
    assert timings == {
        "build_s": 1.235,
        "simulate_s": 2.0,
        "folds_s": round(110.4567 - 105.0, 3),
        "total_s": round(130.1234 - 100.0, 3),
    }
