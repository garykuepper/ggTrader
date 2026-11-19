"""The private helpers of ``run_wfo`` (``_score_fold`` and the ones added after it) on their own.

The scripted scorer of the characterization tests answers every window request from a script, so
a single fold can be scored without a simulation and every call it makes can be read back.
"""

from __future__ import annotations

import contextlib
import io
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
