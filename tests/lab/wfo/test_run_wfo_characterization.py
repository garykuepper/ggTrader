"""Pin run_wfo: fold records, scorer call order, verbose output and branch outcomes.

Two kinds of scenario, both recorded in ``tests/lab/wfo/characterization/run_wfo.json``:

* **Real engine** on the closed-form toy of ``helpers.py`` (``ToyHold``): the numbers come from
  the Rust simulator, so the golden also pins how the engine, the gates and the stitching work
  together.
* **Scripted scorer**: ``ScriptedScorer`` answers every window request from a per-scenario
  script, so each branch of ``run_wfo`` (skipped folds, halt and shadow re-entry, NaN and None
  values, no anchor, empty curves, the OD12 flip) can be reached with chosen inputs. Every call
  the run makes is appended to ``calls`` and goes into the golden, which pins the order and the
  arguments of the scorer calls.

The tests that read the golden (``test_run_wfo_scenario_matches_golden``) are a record of what
the code does today. The tests below them (keys, signature, seams, identity, claims) do not
trust that record: the key lists and the signature are written out here, so regenerating the
golden cannot weaken them.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import io
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from ggTrader.engine import config as engine_config
from ggTrader.engine import portfolio as engine_portfolio
from ggTrader.lab.strategies.contract import ParamGrid
from ggTrader.lab.strategies.xs_momentum import XsMomentum
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.wfo import inputs, run
from ggTrader.lab.wfo.breaker import WfoPolicy
from ggTrader.lab.wfo.folds import Fold, generate_folds
from ggTrader.lab.wfo.gates import NdhOutcome
from ggTrader.lab.wfo.scorers import SignalScorer, WindowReq
from tests.characterization import RTOL, check_golden, compare, dumps, encode
from tests.lab.strategies import helpers as strategy_helpers
from tests.lab.wfo import helpers as h

GOLDEN = Path(__file__).parent / "characterization" / "run_wfo.json"
ES, EE = "2020-06-01", "2022-03-01"  # 3 folds on the 1300 toy bars from 2019-03-01
UTC = "UTC"

SCENARIOS: dict[str, Callable[[], dict]] = {}
#: Scenarios whose own test is marked slow (a single test of 3 s or more leaves the fast core).
SLOW_SCENARIOS: set[str] = set()

FOLD_KEYS_28 = (
    "fold_num",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "winner_combo",
    "winner_params",
    "train_score",
    "oos_score",
    "ndh_passed",
    "dsr_passed",
    "gates_passed",
    "ndh_density",
    "dsr_value",
    "wfe",
    "is_sharpe",
    "oos_sharpe",
    "halted",
    "halted_entering",
    "used_anchor",
    "anchor_combo",
    "anchor_params",
    "anchor_reason",
    "deployed_combo",
    "halt_cleared",
    "winner_trades_in_window",
    "winner_trades_whole_simulation",
    "od12_ndh_flip",
)
DIAGNOSTICS_KEYS_26 = (
    "engine",
    "definitions",
    "strategy_kind",
    "intraday_stops",
    "halt_mask_supplied",
    "halt_mask_sha256",
    "order_sequence",
    "stop_exit_price",
    "stop_entry_price",
    "stop_settings",
    "time_stop_warning",
    "n_combos",
    "n_symbols",
    "n_bars",
    "n_folds",
    "n_simulations",
    "n_chunks",
    "legacy_equivalent_simulations",
    "od12_ndh_flips_folds",
    "timings_s",
    "peak_rss_mb",
    "max_bytes",
    "oos_metrics",
    "spy_metrics",
    "pit_masked",
    "guards",
)


@pytest.fixture(autouse=True)
def _rust_engine() -> Iterator[None]:
    """Run every scenario on the Rust engine and put the process-wide setting back."""
    before = engine_config.active_engine()
    engine_config.set_engine("rust")
    yield
    engine_config.set_engine(before)


# --- recording -----------------------------------------------------------------------------


def _stable_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """The diagnostics with the volatile values replaced by their type names.

    ``peak_rss_mb`` and every value of ``timings_s`` change from run to run; the keys, their
    order and the types are what is pinned.
    """
    stable = dict(diagnostics)
    stable["peak_rss_mb"] = type(diagnostics["peak_rss_mb"]).__name__
    stable["timings_s"] = {k: type(v).__name__ for k, v in diagnostics["timings_s"].items()}
    return stable


def _record(result: Any, stdout: str, calls: list | None = None) -> dict[str, Any]:
    """One scenario's record: the result, what it printed and (scripted only) the calls."""
    if isinstance(result, str):
        record: dict[str, Any] = {"returned": "str", "value": result, "stdout": stdout}
    else:
        record = {
            "fold_results": result.fold_results,
            "skipped_folds": result.skipped_folds,
            "live_params": result.live_params,
            "table": result.table,
            "oos_equity": result.oos_equity,
            "diagnostics": _stable_diagnostics(result.diagnostics),
            "stdout": stdout,
        }
    if calls is not None:
        record["calls"] = calls
    return record


def scenario(func: Callable[[], dict]) -> Callable[[], dict]:
    """Register a scenario under its function name, in definition order."""
    SCENARIOS[func.__name__] = func
    return func


# --- real engine on the closed-form toy ----------------------------------------------------


def _toy_run(
    delays: tuple[int, ...] = h.DELAYS,
    *,
    eval_start: str = ES,
    eval_end: str = EE,
    base: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run ``run_wfo`` on ``ToyHold`` over the toy bars with the printed output captured."""
    bars = h.toy_bars()
    kwargs.setdefault("verbose", False)
    kwargs.setdefault("allow_unmasked", True)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            result: Any = run.run_wfo(
                "toy",
                h.ToyHold,
                LabConfig(),
                bars,
                bars.close["X"],
                eval_start,
                eval_end,
                "test",
                dict(h.TOY_CONFIG) if base is None else base,
                ParamGrid.from_sweep({"delay": list(delays)}),
                **kwargs,
            )
        except run.WfoNotRun as exc:
            # run_wfo used to return this text; the golden keeps the record of its message
            # (the type and message are pinned by test_toy_known_answers.py)
            result = str(exc)
    return _record(result, out.getvalue())


INSIDE_TRAIN = (330, 400, 480)
#: Two delays on which zero thresholds change the deployed combination in fold 2: the winner
#: (delay 400) passes both gates only when they are open, and the default run falls back to the
#: anchor (delay 330). Three delays never show this (the neighborhood variance cap or a
#: negative peak Sharpe still blocks the winner).
OPEN_GATES_DELAYS = (330, 400)


@scenario
def toy_default() -> dict:
    return _toy_run()


@scenario
def toy_entries_inside_train() -> dict:
    return _toy_run(INSIDE_TRAIN)


@scenario
def toy_open_gates() -> dict:
    return _toy_run(OPEN_GATES_DELAYS, ndh_threshold=0.0, dsr_threshold=0.0)


@scenario
def toy_never_trades() -> dict:
    return _toy_run((1200, 1250, 1290))


@scenario
def toy_policy_exclude() -> dict:
    """The run-level policy on a run it does not change.

    Exclusion changes the breaker's history, but this three-fold run cannot trigger its
    four-observation chronic check; acute failure halts both policies. The record equals
    ``toy_entries_inside_train`` (``_claim_policy_exclude`` asserts the equality). A changed
    verdict is pinned
    by the two ``scripted_chronic_decay_*`` scenarios, and that it reaches the breaker by
    ``test_policy_forwards_nan_exclusion_to_breaker``.
    """
    policy = WfoPolicy(nan_wfe="exclude", anchor_fold_wfe="exclude_from_breaker")
    return _toy_run(INSIDE_TRAIN, policy=policy)


INTRADAY_BASE = {**h.TOY_CONFIG, "ts_stop": 0.1, "stop_entry_price": "fillprice"}
#: A halt on a bar before the earliest entry (delay 330): no position is open, so nothing refuses.
FLAT_HALT_BAR = 1


def _toy_halt_mask(*bars_halted: int) -> pd.DataFrame:
    """The toy grid's halt mask, true on the given bar positions only."""
    close = h.toy_bars().close
    mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    for bar in bars_halted:
        mask.iloc[bar] = True
    return mask


@scenario
def toy_intraday_halt_mask() -> dict:
    return _toy_run(
        INSIDE_TRAIN,
        base=INTRADAY_BASE,
        intraday_stops=True,
        halt_mask=_toy_halt_mask(FLAT_HALT_BAR),
    )


@scenario
def toy_verbose() -> dict:
    return _toy_run(INSIDE_TRAIN, verbose=True)


@scenario
def toy_no_valid_folds() -> dict:
    return _toy_run(eval_start="2021-01-01", eval_end="2021-06-01")


@scenario
def weights_xs_momentum() -> dict:
    bars = strategy_helpers.make_bars(seed=8, n_bars=1100, n_syms=14)
    mask = bars.close.notna().cumsum() >= 60
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = run.run_wfo(
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
            mask=mask,
            verbose=True,
        )
    return _record(result, out.getvalue())


# --- scripted scorer -----------------------------------------------------------------------

#: The kinds ``run_wfo`` asks about per fold, and the one global window.
FOLD_KINDS = ("anchor", "train", "test")


def _call_record(
    method: str, req: WindowReq | None, extra: Any = None
) -> tuple[str, Any, Any, Any, Any, Any]:
    """``(method, kind, fold, start, end, extra)`` of one scorer call."""
    if req is None:
        return (method, None, None, None, None, extra)
    return (method, req.kind, req.fold, req.start, req.end, extra)


def scripted_scorer_class(script: dict[str, Any], calls: list) -> type[SignalScorer]:
    """A ``SignalScorer`` subclass that answers from ``script`` and logs every call.

    It ignores the constructor arguments, so it works whether ``run_wfo`` builds a scorer by
    ``SignalScorer(...)`` or by ``SignalScorer.from_inputs(...)``. ``stop_settings`` is
    deliberately absent (``getattr(scorer, "stop_settings", None)`` is a branch of ``run_wfo``).
    """

    class ScriptedScorer(SignalScorer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.n_simulations = script.get("n_simulations", 1)
            self.n_chunks = script.get("n_chunks", 1)
            self.timings: dict[str, float] = {"build_s": 0.0, "simulate_s": 0.0}

        @classmethod
        def from_inputs(cls, *args: Any, **kwargs: Any) -> ScriptedScorer:
            return cls()

        def _fold(self, req: WindowReq) -> dict[str, Any]:
            return script["folds"][req.fold]

        def prepare(self, requests: Any) -> None:
            calls.append(
                _call_record("prepare", None, [(r.kind, r.fold, r.start, r.end) for r in requests])
            )

        def metrics(self, req: WindowReq, combos: Any = None) -> list[dict[str, Any]]:
            calls.append(_call_record("metrics", req, None if combos is None else list(combos)))
            if req.kind == "live":
                return script["live"]
            return self._fold(req)[req.kind]

        def returns(self, req: WindowReq, combo: int) -> pd.Series:
            calls.append(_call_record("returns", req, combo))
            return self._fold(req)["returns"]

        def equity(self, req: WindowReq, combo: int) -> pd.Series:
            calls.append(_call_record("equity", req, combo))
            return self._fold(req)["equity"]

        def window_dates(self, req: WindowReq) -> pd.DatetimeIndex:
            calls.append(_call_record("window_dates", req))
            return self._fold(req)["dates"]

        def trade_counts(self, req: WindowReq, *, cumulative: bool) -> np.ndarray:
            calls.append(_call_record("trade_counts", req, cumulative))
            return self._fold(req)["cumulative"]

    return ScriptedScorer


def _row(
    delay: int,
    sharpe: float,
    *,
    sortino: float | None = None,
    cagr: float = 12.0,
    drawdown: float = -8.0,
    total: float = 9.0,
    trades: int = 3,
) -> dict[str, Any]:
    """A metric row on the ``ToyHold`` lattice, as ``SignalScorer.metrics`` returns it."""
    return {
        "combo": inputs.combo_name("toy", {"delay": delay}),
        "params": {"delay": delay},
        "total_return_pct": total,
        "cagr_pct": cagr,
        "sharpe": sharpe,
        "sortino": sharpe * 1.2 if sortino is None else sortino,
        "ann_vol_pct": 10.0,
        "max_drawdown_pct": drawdown,
        "n_trades": trades,
    }


def _rows(
    sharpes: tuple[float, float, float], *, drawdowns: tuple[float, float, float] = (-6.0,) * 3
) -> list[dict[str, Any]]:
    """One row per toy combination (delays 3, 50, 150), with total return 9 % and 3 trades."""
    return [_row(d, s, drawdown=dd) for d, s, dd in zip(h.DELAYS, sharpes, drawdowns)]


#: Train rows whose winner (delay 50) has two profitable neighbours: the NDH gate passes.
GOOD_TRAIN = (1.1, 1.6, 1.2)
#: Anchor rows: delay 3 has the smallest drawdown among the rows that clear the risk-free rate.
ANCHOR_ROWS = [
    _row(3, 0.9, cagr=9.0, drawdown=-3.0),
    _row(50, 1.4, cagr=11.0, drawdown=-7.0),
    _row(150, 1.0, cagr=10.0, drawdown=-9.0),
]
#: Train rows whose winner has loss-making neighbours: the NDH gate fails.
BAD_TRAIN = (-0.5, 1.5, -0.4)


def _returns(n: int = 80) -> pd.Series:
    """Deterministic train returns whose DSR passes the 0.80 gate."""
    k = np.arange(n)
    return pd.Series(0.004 + 0.004 * np.sin(k * 1.3), index=pd.bdate_range("2021-01-04", periods=n))


def _equity(fold: Fold, slope: float = 0.0006) -> pd.Series:
    """Deterministic test-window equity, positive and moving."""
    index = pd.bdate_range(fold.test_start, fold.test_end, tz=UTC)
    k = np.arange(len(index))
    return pd.Series(100.0 * (1 + slope * k) + np.sin(k / 3.0), index=index)


def _flat_dates(fold: Fold) -> pd.DatetimeIndex:
    return pd.bdate_range(fold.test_start, fold.test_end, tz=UTC)


def _fold_script(fold: Fold, **over: Any) -> dict[str, Any]:
    """One fold's script: anchor and train rows, the deployed test row, curves and counts."""
    script: dict[str, Any] = {
        "anchor": ANCHOR_ROWS,
        "train": _rows(GOOD_TRAIN),
        "test": [_row(50, 1.0)],
        "returns": _returns(),
        "equity": _equity(fold),
        "dates": _flat_dates(fold),
        "cumulative": np.array([3, 4, 5]),
    }
    script.update(over)
    return script


def _scripted_run(
    script: dict[str, Any],
    *,
    eval_start: str = ES,
    eval_end: str = EE,
    spy_close: pd.Series | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run ``run_wfo`` with the scripted scorer patched in; record the result and the calls."""
    bars = h.toy_bars()
    calls: list = []
    kwargs.setdefault("verbose", True)
    kwargs.setdefault("allow_unmasked", True)
    out = io.StringIO()
    with (
        mock.patch.object(run, "SignalScorer", scripted_scorer_class(script, calls)),
        contextlib.redirect_stdout(out),
    ):
        result = run.run_wfo(
            "toy",
            h.ToyHold,
            LabConfig(),
            bars,
            bars.close["X"] if spy_close is None else spy_close,
            eval_start,
            eval_end,
            "test",
            dict(h.TOY_CONFIG),
            h.ToyHold.grid,
            **kwargs,
        )
    return _record(result, out.getvalue(), calls)


def _folds(eval_start: str = ES, eval_end: str = EE) -> list[Fold]:
    return generate_folds(pd.Timestamp(eval_start, tz=UTC), pd.Timestamp(eval_end, tz=UTC))


def _live_rows() -> list[dict[str, Any]]:
    return _rows((1.0, 1.5, 1.1))


@scenario
def scripted_skipped_fold() -> dict:
    # four folds: the second and the last have no train metrics, so they are skipped
    ee = "2022-06-01"
    folds = _folds(eval_end=ee)
    script = {
        "live": _live_rows(),
        "folds": [
            _fold_script(folds[0]),
            _fold_script(folds[1], train=[]),
            _fold_script(folds[2]),
            _fold_script(
                folds[3], train=[], anchor=_rows((0.9, 1.4, 1.0), drawdowns=(-9.0, -7.0, -3.0))
            ),
        ],
    }
    return _scripted_run(script, eval_end=ee)


@scenario
def scripted_halt_and_reentry() -> dict:
    # five folds. Folds 1 and 2 lose out of sample (acute failure halts after fold 2); folds 3
    # and 4 run shadow re-entry as clean windows and fold 4 clears the halt; fold 5 is free.
    ee = "2022-09-01"
    folds = _folds(eval_end=ee)
    anchor_test = [_row(3, 0.9)]
    script = {
        "live": _live_rows(),
        "folds": [
            _fold_script(folds[0], test=[_row(50, -0.4)]),
            _fold_script(folds[1], test=[_row(50, -0.3)]),
            _fold_script(folds[2], test=anchor_test),
            _fold_script(folds[3], test=anchor_test),
            _fold_script(folds[4], test=[_row(50, 0.5)]),
        ],
    }
    return _scripted_run(script, eval_end=ee)


@scenario
def scripted_nan_and_none_values() -> dict:
    folds = _folds()
    below_floor = _rows((0.2, 0.3, 0.25))  # winner Sharpe 0.3 is under the 0.4 floor: wfe None
    nan_is_sharpe = [
        _row(3, 0.5, sortino=0.6, drawdown=-6.0),
        _row(50, float("nan"), sortino=3.0, drawdown=-2.0),  # NaN in-sample Sharpe: wfe NaN
        _row(150, 0.5, sortino=0.7, drawdown=-9.0),
    ]
    no_sharpe_test = [{k: v for k, v in _row(50, 1.0).items() if k != "sharpe"}]
    script = {
        "live": _live_rows(),
        "folds": [
            _fold_script(folds[0], train=below_floor, test=[_row(50, 0.5)]),
            _fold_script(folds[1], train=nan_is_sharpe, test=[_row(50, 0.5)]),
            _fold_script(folds[2], test=no_sharpe_test),  # no 'sharpe': oos_sharpe is NaN
        ],
    }
    return _scripted_run(script)


@scenario
def scripted_no_anchor_and_empty_curves() -> dict:
    folds = _folds()
    no_anchor = [_row(d, float("nan")) for d in h.DELAYS]  # never traded: no candidate
    script = {
        "live": _live_rows(),
        "folds": [
            # no anchor and gates fail: no deployed combination, a flat curve from the dates
            _fold_script(folds[0], anchor=no_anchor, train=_rows(BAD_TRAIN)),
            # the same, but the window has no dates: no curve at all
            _fold_script(
                folds[1], anchor=no_anchor, train=_rows(BAD_TRAIN), dates=pd.DatetimeIndex([])
            ),
            # an anchor exists and is deployed, but its test equity is empty
            _fold_script(
                folds[2],
                train=_rows(BAD_TRAIN),
                test=[_row(3, 0.9)],
                equity=pd.Series(dtype=float),
            ),
        ],
    }
    return _scripted_run(script)


def _ndh_double(
    train_metrics: list[dict[str, Any]], grid: Any, best_idx: int, threshold: float
) -> NdhOutcome:
    """NOT the real gate: passes when every row has fewer than 100 trades, else fails.

    The real gate cannot flip between a per-window and a cumulative trade count (see
    ``scripted_od12_no_flip_real_gate``), so this double is the only way to reach the branch
    that records an OD12 flip.
    """
    passed = all(row["n_trades"] < 100 for row in train_metrics)
    return NdhOutcome(passed, 1.0 if passed else 0.0, 0.1)


@scenario
def scripted_od12_flip_gate_double() -> dict:
    folds = _folds()
    window_rows = _rows(GOOD_TRAIN)  # three trades each, below 100
    flip = np.array([120, 130, 140])  # cumulative counts of 100 or more: the double flips
    steady = np.array([5, 6, 7])
    script = {
        "live": _live_rows(),
        "folds": [
            _fold_script(folds[0], train=window_rows, cumulative=flip),
            _fold_script(folds[1], train=window_rows, cumulative=steady),
            _fold_script(folds[2], train=window_rows, cumulative=flip),
        ],
    }
    with mock.patch.object(run, "run_ndh", _ndh_double):
        record = _scripted_run(script)
    return {"gate": "double", **record}


@scenario
def scripted_od12_no_flip_real_gate() -> dict:
    folds = _folds()
    rows = [dict(row, n_trades=0) for row in _rows(GOOD_TRAIN)]
    cumulative = np.array([7, 7, 7])
    script = {
        "live": _live_rows(),
        "folds": [_fold_script(f, train=rows, cumulative=cumulative) for f in folds],
    }
    return {"gate": "real", **_scripted_run(script)}


@scenario
def scripted_empty_test_metrics() -> dict:
    folds = _folds()
    script = {
        "live": _live_rows(),
        "folds": [_fold_script(f, test=[]) for f in folds],
    }
    return _scripted_run(script)


@scenario
def scripted_spy_missing_over_oos() -> dict:
    folds = _folds()
    script = {"live": _live_rows(), "folds": [_fold_script(f) for f in folds]}
    elsewhere = pd.Series([100.0], index=pd.DatetimeIndex(["2030-01-02"], tz=UTC))
    return _scripted_run(script, spy_close=elsewhere)


def _chronic_decay_script() -> tuple[dict[str, Any], str]:
    # five folds, every winner fails its gate (anchor deployed) and the OOS Sharpe is positive
    # but small: WFE 0.2 / 2.0 = 0.1 in each, so only chronic decay can halt the run
    ee = "2022-09-01"
    folds = _folds(eval_end=ee)
    weak = _rows((-0.5, 2.0, -0.4))
    anchor_test = [_row(3, 0.2)]
    script = {
        "live": _live_rows(),
        "folds": [_fold_script(f, train=weak, test=anchor_test) for f in folds],
    }
    return script, ee


@scenario
def scripted_chronic_decay_legacy() -> dict:
    script, ee = _chronic_decay_script()
    return _scripted_run(script, eval_end=ee)


@scenario
def scripted_chronic_decay_policy_exclude() -> dict:
    script, ee = _chronic_decay_script()
    policy = WfoPolicy(nan_wfe="exclude", anchor_fold_wfe="exclude_from_breaker")
    return _scripted_run(script, eval_end=ee, policy=policy)


# --- reading the golden --------------------------------------------------------------------


def _entries(tree: Any) -> dict[Any, Any]:
    """The key -> encoded value mapping of an encoded dict."""
    return {key: value for key, value in tree["$dict"]}


@functools.cache
def _golden_text() -> str:
    return GOLDEN.read_text(encoding="utf-8")


def _golden_scenarios() -> dict[str, Any]:
    """Each scenario's encoded record, read from the golden file."""
    return _entries(_entries(json.loads(_golden_text()))["scenarios"])


def _plain(tree: Any) -> Any:
    """An encoded tree as plain Python (dicts, tuples, floats, timestamps); curves stay tagged."""
    if isinstance(tree, list):
        return [_plain(item) for item in tree]
    if not isinstance(tree, dict):
        return tree
    if "$dict" in tree:
        return {_plain(k): _plain(v) for k, v in tree["$dict"]}
    if "$tuple" in tree:
        return tuple(_plain(item) for item in tree["$tuple"])
    if "$float" in tree:
        return float(tree["$float"].lower().replace("infinity", "inf"))
    if tree.get("$type") == "pandas.Timestamp":
        return pd.Timestamp(tree["v"])
    if str(tree.get("$type", "")).startswith("numpy."):
        return _plain(tree["v"])
    return tree


def _record_of(name: str) -> dict[str, Any]:
    return _plain(_golden_scenarios()[name])


# --- golden -------------------------------------------------------------------------------

_REGENERATED = False


def _regenerate_everything() -> None:
    """Rebuild the whole golden once per run, then fail so the diff is reviewed."""
    global _REGENERATED
    if _REGENERATED:
        pytest.skip("golden already regenerated by an earlier test of this run")
    _REGENERATED = True
    check_golden(GOLDEN, {"scenarios": {key: build() for key, build in SCENARIOS.items()}})


def _scenario_params() -> list[Any]:
    return [
        pytest.param(name, marks=pytest.mark.slow) if name in SLOW_SCENARIOS else name
        for name in SCENARIOS
    ]


@pytest.mark.parametrize("name", _scenario_params())
def test_run_wfo_scenario_matches_golden(name: str) -> None:
    if os.environ.get("GGT_REGEN_CHARACTERIZATION") == "1":
        _regenerate_everything()
    actual = encode(SCENARIOS[name]())
    golden = _golden_scenarios()[name]
    exact = os.environ.get("GGT_CHARACTERIZATION_EXACT") == "1"
    paths = compare(actual, golden, rtol=RTOL, exact=exact)
    assert not paths, f"{name} differs at {paths}\n{dumps(actual)[:3000]}"


def test_every_scenario_has_a_golden_entry() -> None:
    assert not [name for name in SCENARIOS if name not in _golden_scenarios()]


def test_no_orphan_golden_entries() -> None:
    assert not [name for name in _golden_scenarios() if name not in SCENARIOS]


# --- what each scenario's name claims ------------------------------------------------------


def _rows_of(record: dict[str, Any]) -> list[dict[str, Any]]:
    return record["fold_results"]


def _claim_default(record: dict[str, Any]) -> None:
    assert len(_rows_of(record)) == 3
    assert record["skipped_folds"] == ()


def _claim_entries_inside_train(record: dict[str, Any]) -> None:
    assert any(row["winner_trades_in_window"] >= 1 for row in _rows_of(record))


def _claim_open_gates(record: dict[str, Any]) -> None:
    # fold 2's winner (delay 400) is deployed because the gates are open; the same grid under the
    # default thresholds refuses it (test_open_gates_change_the_deployed_combination)
    first, second = _rows_of(record)[:2]
    assert first["gates_passed"] is True and first["used_anchor"] is False
    assert second["gates_passed"] is True and second["used_anchor"] is False
    assert second["deployed_combo"] == second["winner_combo"] == "toy__delay400"
    assert second["ndh_density"] == 0.0 and second["dsr_value"] < 0.8


def _claim_never_trades(record: dict[str, Any]) -> None:
    rows = _rows_of(record)
    assert all(row["anchor_combo"] == "none" and row["deployed_combo"] == "none" for row in rows)


def _claim_policy_exclude(record: dict[str, Any]) -> None:
    assert any(row["used_anchor"] for row in _rows_of(record))
    # the policy is a no-op on this input: the whole record equals the default-policy run's
    golden = _golden_scenarios()
    this = _entries(golden["toy_policy_exclude"])
    default = _entries(golden["toy_entries_inside_train"])
    for key in ("fold_results", "skipped_folds", "live_params", "table", "oos_equity", "stdout"):
        assert this[key] == default[key], key


def _claim_intraday(record: dict[str, Any]) -> None:
    # The halt mask here flags a flat bar, so it refuses nothing; that a halt on an open position
    # refuses is test_halt_on_an_open_position_refuses_through_run_wfo. What this record shows
    # is the intraday trailing stop acting: the stopped positions leave the out-of-sample curve
    # flat, where the stop-less run on the same grid loses 28.7%.
    diagnostics = record["diagnostics"]
    assert diagnostics["intraday_stops"] is True
    assert diagnostics["halt_mask_supplied"] is True
    assert diagnostics["stop_entry_price"] == "fillprice"
    assert diagnostics["stop_settings"]["trailing_fraction"] == 0.1
    assert diagnostics["oos_metrics"]["total_return_pct"] == 0.0
    stopless = _record_of("toy_entries_inside_train")["diagnostics"]["oos_metrics"]
    assert stopless["total_return_pct"] < -25.0
    assert any(row["winner_trades_whole_simulation"] >= 1 for row in _rows_of(record))


def _claim_verbose(record: dict[str, Any]) -> None:
    assert "  Fold 1/3:" in record["stdout"] and record["table"] in record["stdout"]


def _claim_no_valid_folds(record: dict[str, Any]) -> None:
    assert record["returned"] == "str" and "no valid folds" in record["value"]


def _claim_weights(record: dict[str, Any]) -> None:
    assert record["diagnostics"]["strategy_kind"] == "weights"
    assert len(_rows_of(record)) >= 1


def _claim_skipped(record: dict[str, Any]) -> None:
    skipped = record["skipped_folds"]
    assert [item["fold_num"] for item in skipped] == [2, 4]
    assert all(item["reason"] == "no_train_metrics" for item in skipped)
    assert len(_rows_of(record)) == 2
    assert record["diagnostics"]["n_folds"] == 4
    fallback = record["table"].split("Anchor Set (Defensive Fallback)", 1)[1]
    assert inputs.combo_name("toy", {"delay": 150}) in fallback


def test_open_gates_change_the_deployed_combination() -> None:
    """Zero thresholds deploy fold 2's winner; the default thresholds fall back to the anchor."""
    default = _toy_run(OPEN_GATES_DELAYS)["fold_results"][1]
    open_ = _toy_run(OPEN_GATES_DELAYS, ndh_threshold=0.0, dsr_threshold=0.0)["fold_results"][1]
    assert default["winner_combo"] == open_["winner_combo"] == "toy__delay400"
    assert default["gates_passed"] is False and open_["gates_passed"] is True
    assert default["used_anchor"] is True and open_["used_anchor"] is False
    assert default["deployed_combo"] == "toy__delay330"
    assert open_["deployed_combo"] == "toy__delay400"


def test_a_halt_flag_changes_the_mask_hash_but_not_a_flat_run() -> None:
    """A flagged flat bar is a different mask from an all-false one but refuses nothing."""
    flagged = SCENARIOS["toy_intraday_halt_mask"]()
    all_false = _toy_run(
        INSIDE_TRAIN, base=INTRADAY_BASE, intraday_stops=True, halt_mask=_toy_halt_mask()
    )
    assert (
        flagged["diagnostics"]["halt_mask_sha256"] != all_false["diagnostics"]["halt_mask_sha256"]
    )
    assert dumps(encode(flagged["fold_results"])) == dumps(encode(all_false["fold_results"]))


def test_halt_on_an_open_position_refuses_through_run_wfo() -> None:
    """The engine's halt refusal is not swallowed: a halt inside the delay-330 trade raises."""
    with pytest.raises(ValueError, match="halt during an open position at 2020-09-11"):
        _toy_run(
            INSIDE_TRAIN,
            base=INTRADAY_BASE,
            intraday_stops=True,
            halt_mask=_toy_halt_mask(400),
        )


def test_policy_forwards_nan_exclusion_to_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """The run-level policy reaches the circuit breaker on each fold."""
    original = run.check_circuit_breaker
    calls = []

    def recording_breaker(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("nan_policy"))
        return original(*args, **kwargs)

    monkeypatch.setattr(run, "check_circuit_breaker", recording_breaker)
    toy_policy_exclude()
    assert calls and all(policy == "exclude" for policy in calls)


def _claim_halt(record: dict[str, Any]) -> None:
    rows = _rows_of(record)
    assert len(rows) >= 4
    assert [row["halted"] for row in rows] == [False, True, True, False, False]
    assert [row["halt_cleared"] for row in rows] == [False, False, False, True, False]
    assert [row["anchor_reason"] for row in rows] == [None, None, "halted", "halted", None]


def _claim_nan_and_none(record: dict[str, Any]) -> None:
    rows = _rows_of(record)
    assert rows[0]["wfe"] is None
    assert rows[1]["wfe"] != rows[1]["wfe"]  # NaN
    assert rows[2]["oos_sharpe"] != rows[2]["oos_sharpe"]


def _claim_no_anchor(record: dict[str, Any]) -> None:
    rows = _rows_of(record)
    assert [row["anchor_combo"] for row in rows][:2] == ["none", "none"]
    assert [row["deployed_combo"] for row in rows][:2] == ["none", "none"]
    assert rows[2]["deployed_combo"] != "none"


def _claim_flip_double(record: dict[str, Any]) -> None:
    assert record["gate"] == "double"
    assert [row["od12_ndh_flip"] for row in _rows_of(record)] == [True, False, True]
    assert record["diagnostics"]["od12_ndh_flips_folds"] == [1, 3]


def _claim_no_flip_real(record: dict[str, Any]) -> None:
    assert record["gate"] == "real"
    assert not any(row["od12_ndh_flip"] for row in _rows_of(record))
    assert record["diagnostics"]["od12_ndh_flips_folds"] == []


def _claim_empty_test(record: dict[str, Any]) -> None:
    rows = _rows_of(record)
    assert rows and all(row["oos_score"] == 0.0 for row in rows)
    assert all(row["oos_sharpe"] != row["oos_sharpe"] for row in rows)


def _claim_spy_missing(record: dict[str, Any]) -> None:
    spy = record["diagnostics"]["spy_metrics"]
    assert set(spy) == {"sharpe", "cagr_pct", "max_drawdown_pct"}
    assert all(value != value for value in spy.values())


def _claim_chronic_legacy(record: dict[str, Any]) -> None:
    rows = _rows_of(record)
    assert all(row["used_anchor"] for row in rows)
    assert any(row["halted"] for row in rows)


def _claim_chronic_exclude(record: dict[str, Any]) -> None:
    rows = _rows_of(record)
    assert all(row["used_anchor"] for row in rows)
    assert not any(row["halted"] for row in rows)


WHAT_THE_NAME_CLAIMS: dict[str, Callable[[dict[str, Any]], None]] = {
    "toy_default": _claim_default,
    "toy_entries_inside_train": _claim_entries_inside_train,
    "toy_open_gates": _claim_open_gates,
    "toy_never_trades": _claim_never_trades,
    "toy_policy_exclude": _claim_policy_exclude,
    "toy_intraday_halt_mask": _claim_intraday,
    "toy_verbose": _claim_verbose,
    "toy_no_valid_folds": _claim_no_valid_folds,
    "weights_xs_momentum": _claim_weights,
    "scripted_skipped_fold": _claim_skipped,
    "scripted_halt_and_reentry": _claim_halt,
    "scripted_nan_and_none_values": _claim_nan_and_none,
    "scripted_no_anchor_and_empty_curves": _claim_no_anchor,
    "scripted_od12_flip_gate_double": _claim_flip_double,
    "scripted_od12_no_flip_real_gate": _claim_no_flip_real,
    "scripted_empty_test_metrics": _claim_empty_test,
    "scripted_spy_missing_over_oos": _claim_spy_missing,
    "scripted_chronic_decay_legacy": _claim_chronic_legacy,
    "scripted_chronic_decay_policy_exclude": _claim_chronic_exclude,
}


def test_what_the_name_claims_covers_every_scenario() -> None:
    assert list(WHAT_THE_NAME_CLAIMS) == list(SCENARIOS)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_shows_what_its_name_claims(name: str) -> None:
    WHAT_THE_NAME_CLAIMS[name](_record_of(name))


# --- frozen shape: keys, order, signature ---------------------------------------------------


def _fold_rows_of(name: str) -> list[dict[str, Any]]:
    return _rows_of(_record_of(name))


@pytest.mark.parametrize("name", ["toy_default", "toy_open_gates", "scripted_halt_and_reentry"])
def test_fold_keys_are_the_28_in_order(name: str) -> None:
    assert len(FOLD_KEYS_28) == 28
    assert _fold_rows_of(name), name
    for row in _fold_rows_of(name):
        assert tuple(row) == FOLD_KEYS_28


@pytest.mark.parametrize(
    "name", ["toy_default", "scripted_halt_and_reentry", "weights_xs_momentum"]
)
def test_diagnostics_keys_are_the_26_in_order(name: str) -> None:
    assert len(DIAGNOSTICS_KEYS_26) == 26
    assert tuple(_record_of(name)["diagnostics"]) == DIAGNOSTICS_KEYS_26


def test_the_live_run_has_the_same_fold_and_diagnostics_keys() -> None:
    """The golden is not the only witness: a fresh scripted run has the same key lists."""
    record = SCENARIOS["scripted_halt_and_reentry"]()
    assert all(tuple(row) == FOLD_KEYS_28 for row in record["fold_results"])
    assert tuple(record["diagnostics"]) == DIAGNOSTICS_KEYS_26


#: name, kind, default for the 23 parameters of ``run_wfo`` (``policy`` is checked separately).
SIGNATURE: list[tuple[str, str, Any]] = [
    ("strategy_name", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("strategy_cls", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("cfg", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("ohlcv", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("spy_close", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("eval_start", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("eval_end", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("market", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("base_config", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("grid", "POSITIONAL_OR_KEYWORD", inspect.Parameter.empty),
    ("ndh_threshold", "POSITIONAL_OR_KEYWORD", 0.85),
    ("dsr_threshold", "POSITIONAL_OR_KEYWORD", 0.80),
    ("universe_fn", "POSITIONAL_OR_KEYWORD", None),
    ("policy", "POSITIONAL_OR_KEYWORD", "WfoPolicy()"),
    ("mask", "KEYWORD_ONLY", None),
    ("allow_unmasked", "KEYWORD_ONLY", False),
    ("max_bytes", "KEYWORD_ONLY", "DEFAULT_MAX_BYTES"),
    ("min_losing_days", "KEYWORD_ONLY", "MIN_LOSING_DAYS"),
    ("intraday_stops", "KEYWORD_ONLY", False),
    ("halt_mask", "KEYWORD_ONLY", None),
    ("order_sequence", "KEYWORD_ONLY", None),
    ("stop_exit_price", "KEYWORD_ONLY", None),
    ("verbose", "KEYWORD_ONLY", True),
    ("dead_stock_loss", "KEYWORD_ONLY", 0.0),
]


def test_run_wfo_signature_is_frozen() -> None:
    parameters = list(inspect.signature(run.run_wfo).parameters.values())
    assert len(parameters) == 24
    resolved = {
        "WfoPolicy()": WfoPolicy(),
        "DEFAULT_MAX_BYTES": engine_portfolio.DEFAULT_MAX_BYTES,
        "MIN_LOSING_DAYS": engine_config.MIN_LOSING_DAYS,
    }
    actual = [(p.name, p.kind.name, p.default) for p in parameters]
    expected = [
        (name, kind, resolved[default] if isinstance(default, str) else default)
        for name, kind, default in SIGNATURE
    ]
    assert actual == expected


# --- seams ----------------------------------------------------------------------------------


def _scripted_default() -> tuple[dict[str, Any], list[Fold]]:
    """A script of three default folds, and those folds."""
    folds = _folds()
    script = {"live": _live_rows(), "folds": [_fold_script(f) for f in folds]}
    return script, folds


def test_scorer_seam_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing ``run.SignalScorer`` is honored: the run reads the script, not the engine."""
    script, folds = _scripted_default()
    consulted: list[str] = []
    calls: list = []
    scripted = scripted_scorer_class(script, calls)

    class Spy(scripted):  # type: ignore[valid-type, misc]
        def metrics(self, req: WindowReq, combos: Any = None) -> list[dict[str, Any]]:
            consulted.append(req.kind)
            return super().metrics(req, combos)

    monkeypatch.setattr(run, "SignalScorer", Spy)
    bars = h.toy_bars()
    with contextlib.redirect_stdout(io.StringIO()):
        result = run.run_wfo(
            "toy", h.ToyHold, LabConfig(), bars, bars.close["X"], ES, EE, "test",
            dict(h.TOY_CONFIG), h.ToyHold.grid, verbose=False, allow_unmasked=True,
        )  # fmt: skip
    assert len(result.fold_results) == len(folds)
    assert "train" in consulted and calls[0][0] == "prepare"
    assert result.diagnostics["n_simulations"] == 1  # the script's, not the engine's


def test_gate_seam_is_live(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run.run_dsr`` and ``run.run_ndh`` are looked up on the module at call time."""
    dsr_calls: list[tuple[tuple, dict]] = []
    ndh_calls: list[tuple[tuple, dict]] = []
    real_dsr, real_ndh = run.run_dsr, run.run_ndh

    def record_dsr(*args: Any, **kwargs: Any) -> Any:
        dsr_calls.append((args, kwargs))
        return real_dsr(*args, **kwargs)

    def record_ndh(*args: Any, **kwargs: Any) -> Any:
        ndh_calls.append((args, kwargs))
        return real_ndh(*args, **kwargs)

    monkeypatch.setattr(run, "run_dsr", record_dsr)
    monkeypatch.setattr(run, "run_ndh", record_ndh)
    record = _toy_run(INSIDE_TRAIN)
    n_folds = len(record["fold_results"])
    assert n_folds == 3
    assert len(dsr_calls) == n_folds and len(ndh_calls) == 2 * n_folds
    for args, kwargs in dsr_calls:  # (returns, n_trials, threshold), positionally
        assert kwargs == {} and len(args) == 3
        assert isinstance(args[0], pd.Series) and args[1] == 3 and args[2] == 0.80
    for args, kwargs in ndh_calls:  # (train_metrics, combos, best_idx, threshold), positionally
        assert kwargs == {} and len(args) == 4
        assert isinstance(args[0], list) and len(args[1]) == 3 and args[3] == 0.85


def test_winner_and_anchor_params_are_the_scorers_own_dicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fold row holds the very dict objects the scorer supplied, not copies."""
    folds = _folds()
    train = _rows(GOOD_TRAIN)
    anchor = [dict(row) for row in ANCHOR_ROWS]
    script = {
        "live": _live_rows(),
        "folds": [_fold_script(f, train=train, anchor=anchor) for f in folds],
    }
    calls: list = []
    monkeypatch.setattr(run, "SignalScorer", scripted_scorer_class(script, calls))
    bars = h.toy_bars()
    with contextlib.redirect_stdout(io.StringIO()):
        result = run.run_wfo(
            "toy", h.ToyHold, LabConfig(), bars, bars.close["X"], ES, EE, "test",
            dict(h.TOY_CONFIG), h.ToyHold.grid, verbose=False, allow_unmasked=True,
        )  # fmt: skip
    winner_row = train[1]  # delay 50 wins the scripted train rows
    for fold_row in result.fold_results:
        assert fold_row["winner_params"] is winner_row["params"]
        assert fold_row["anchor_params"] is anchor[0]["params"]  # delay 3: least drawdown
