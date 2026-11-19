"""``ScoringInputs`` and ``SignalSimOptions`` carry the scorer arguments without changing them."""

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from ggTrader.engine import config as engine_config
from ggTrader.lab.wfo import run
from ggTrader.lab.wfo.scorers import ScoringInputs, SignalScorer, SignalSimOptions, WeightScorer
from tests.lab.wfo import helpers as h


def _inputs(strategy: Any = None, **over: Any) -> ScoringInputs:
    bars = h.toy_bars(n_bars=20)
    fields: dict[str, Any] = {
        "strategy": strategy if strategy is not None else SimpleNamespace(kind="signals"),
        "bars": bars,
        "grid": h.ToyHold.grid,
        "names": ("a", "b"),
        "params": ({"delay": 3}, {"delay": 50}),
        "mask": None,
        "base_config": dict(h.TOY_CONFIG),
        "max_bytes": 1 << 30,
    }
    fields.update(over)
    return ScoringInputs(**fields)


def test_scoring_inputs_as_args_matches_the_constructor_order() -> None:
    inp = _inputs(min_losing_days=7)
    assert inp.as_args() == (
        inp.strategy,
        inp.bars,
        inp.grid,
        inp.names,
        inp.params,
        inp.mask,
        inp.base_config,
        inp.max_bytes,
        7,
    )
    assert [f.name for f in dataclasses.fields(ScoringInputs)] == [
        *list(inspect.signature(SignalScorer.__init__).parameters)[1:10]
    ]
    assert inspect.signature(WeightScorer.__init__).parameters.keys() == (
        inspect.signature(SignalScorer.__init__).parameters.keys()
        - {
            "intraday_stops",
            "halt_mask",
            "order_sequence",
            "stop_exit_price",
        }
    )


def test_min_losing_days_defaults_to_the_engine_constant() -> None:
    assert _inputs().min_losing_days == engine_config.MIN_LOSING_DAYS


def test_from_inputs_builds_the_same_scorer_as_the_constructor() -> None:
    mask = pd.DataFrame({"X": [True]})
    halt = pd.DataFrame({"X": [False]})
    inp = _inputs(mask=mask, min_losing_days=3)
    opts = SignalSimOptions(
        intraday_stops=True, halt_mask=halt, order_sequence="auto", stop_exit_price="stopmarket"
    )
    direct = SignalScorer(
        *inp.as_args(),
        intraday_stops=True,
        halt_mask=halt,
        order_sequence="auto",
        stop_exit_price="stopmarket",
    )
    built = SignalScorer.from_inputs(inp, opts)
    assert type(built) is SignalScorer
    assert vars(built).keys() == vars(direct).keys()
    for key, value in vars(direct).items():
        if isinstance(value, pd.DataFrame):
            assert getattr(built, key) is value or getattr(built, key).equals(value)
        else:
            assert getattr(built, key) == value, key

    wdirect = WeightScorer(*inp.as_args())
    wbuilt = WeightScorer.from_inputs(inp)
    assert type(wbuilt) is WeightScorer
    assert vars(wbuilt).keys() == vars(wdirect).keys()
    for key, value in vars(wdirect).items():
        if isinstance(value, pd.Index):
            assert wbuilt.index.equals(value)
        elif not isinstance(value, pd.DataFrame):
            assert getattr(wbuilt, key) == value, key


def test_from_inputs_copies_like_the_constructor() -> None:
    inp = _inputs()
    scorer = SignalScorer.from_inputs(inp)
    assert scorer.names == ["a", "b"] and scorer.names is not inp.names
    assert scorer.base_config == inp.base_config and scorer.base_config is not inp.base_config


def test_from_inputs_does_not_prepare() -> None:
    class Boom:
        kind = "signals"

        def build(self, *_a: Any, **_k: Any) -> None:
            raise AssertionError("build must not run")

    scorer = SignalScorer.from_inputs(_inputs(Boom()))
    assert scorer._prepared is False and scorer.n_simulations == 0
    weights = WeightScorer.from_inputs(_inputs(SimpleNamespace(kind="weights")))
    assert weights.n_simulations == 0 and weights._sims == {}


def test_signal_options_default_as_kwargs() -> None:
    opts = SignalSimOptions()
    assert opts.as_kwargs() == {
        "intraday_stops": False,
        "halt_mask": None,
        "order_sequence": None,
        "stop_exit_price": None,
    }
    assert list(opts.as_kwargs()) == [
        "intraday_stops",
        "halt_mask",
        "order_sequence",
        "stop_exit_price",
    ]
    opts.require_signal_strategy()  # defaults ask for nothing


def test_require_signal_strategy_messages_and_order() -> None:
    frame = pd.DataFrame({"X": [True]})
    everything = SignalSimOptions(
        intraday_stops=True, halt_mask=frame, order_sequence="auto", stop_exit_price="stopmarket"
    )
    with pytest.raises(ValueError) as caught:
        everything.require_signal_strategy()
    assert str(caught.value) == "intraday stops require a signal strategy"
    alone = [
        (SignalSimOptions(halt_mask=frame), "halt_mask requires a signal strategy"),
        (SignalSimOptions(order_sequence="auto"), "order_sequence requires a signal strategy"),
        (
            SignalSimOptions(stop_exit_price="stopmarket"),
            "stop_exit_price requires a signal strategy",
        ),
        (SignalSimOptions(intraday_stops=1), "intraday stops require a signal strategy"),
    ]
    for options, message in alone:
        with pytest.raises(ValueError) as caught:
            options.require_signal_strategy()
        assert str(caught.value) == message
    # the order below the first one: halt_mask before order_sequence before stop_exit_price
    rest = SignalSimOptions(halt_mask=frame, order_sequence="auto", stop_exit_price="x")
    with pytest.raises(ValueError, match="halt_mask"):
        rest.require_signal_strategy()
    rest = SignalSimOptions(order_sequence="auto", stop_exit_price="x")
    with pytest.raises(ValueError, match="order_sequence"):
        rest.require_signal_strategy()


def test_options_and_inputs_are_frozen_and_hashable_by_identity() -> None:
    inp, opts = _inputs(), SignalSimOptions(halt_mask=pd.DataFrame({"X": [True]}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        opts.intraday_stops = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        inp.max_bytes = 1  # type: ignore[misc]
    assert inp == inp and inp != _inputs()  # eq=False: identity, so a frame never reaches __eq__
    assert opts == opts


def test_scorer_for_weights_rejects_signal_options() -> None:
    inp = _inputs(SimpleNamespace(kind="weights"))
    with pytest.raises(ValueError, match="halt_mask requires a signal strategy"):
        run._scorer_for(inp.strategy, inp, SignalSimOptions(halt_mask=pd.DataFrame()))
    with pytest.raises(ValueError, match="intraday stops require a signal strategy"):
        run._scorer_for(inp.strategy, inp, SignalSimOptions(intraday_stops=True))
    assert type(run._scorer_for(inp.strategy, inp, SignalSimOptions())) is WeightScorer


def test_scorer_for_signals_accepts_them() -> None:
    inp = _inputs()
    opts = SignalSimOptions(intraday_stops=True, order_sequence="auto", stop_exit_price="stoplimit")
    scorer = run._scorer_for(inp.strategy, inp, opts)
    assert type(scorer) is SignalScorer
    assert (scorer.intraday_stops, scorer.order_sequence, scorer.stop_exit_price) == (
        True,
        "auto",
        "stoplimit",
    )


def test_scorer_for_looks_the_classes_up_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    class SpySignal(SignalScorer):
        pass

    class SpyWeight(WeightScorer):
        pass

    monkeypatch.setattr(run, "SignalScorer", SpySignal)
    monkeypatch.setattr(run, "WeightScorer", SpyWeight)
    sig, wgt = _inputs(), _inputs(SimpleNamespace(kind="weights"))
    seen += [type(run._scorer_for(sig.strategy, sig, SignalSimOptions())).__name__]
    seen += [type(run._scorer_for(wgt.strategy, wgt, SignalSimOptions())).__name__]
    assert seen == ["SpySignal", "SpyWeight"]


def test_make_scorer_wrapper_keeps_its_signature() -> None:
    params = inspect.signature(run._make_scorer).parameters
    assert [(n, p.default) for n, p in params.items()] == [
        ("strategy", inspect.Parameter.empty),
        ("bars", inspect.Parameter.empty),
        ("grid", inspect.Parameter.empty),
        ("names", inspect.Parameter.empty),
        ("combos", inspect.Parameter.empty),
        ("mask", inspect.Parameter.empty),
        ("base_config", inspect.Parameter.empty),
        ("max_bytes", inspect.Parameter.empty),
        ("min_losing_days", inspect.Parameter.empty),
        ("intraday_stops", inspect.Parameter.empty),
        ("order_sequence", None),
        ("stop_exit_price", None),
        ("halt_mask", None),
    ]
    assert all(p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for p in params.values())


def test_make_scorer_wrapper_passes_everything_through() -> None:
    frame = pd.DataFrame({"X": [True]})
    scorer = run._make_scorer(
        SimpleNamespace(kind="signals"),
        h.toy_bars(n_bars=20),
        h.ToyHold.grid,
        ["a"],
        [{"delay": 3}],
        None,
        h.TOY_CONFIG,
        123,
        4,
        True,
        "auto",
        "stopmarket",
        frame,
    )
    assert isinstance(scorer, SignalScorer)
    assert scorer.min_losing_days == 4 and scorer.max_bytes == 123
    assert scorer.intraday_stops is True and scorer.halt_mask is frame
    assert scorer.order_sequence == "auto" and scorer.stop_exit_price == "stopmarket"
