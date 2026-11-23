"""Dead-series execution on signal simulations."""

from dataclasses import replace

import pandas as pd
import pytest

from ggTrader.lab.dead_stocks import forced_exit_frames, haircut_close
from ggTrader.lab.strategies.contract import (
    ParamGrid,
    SignalOut,
    group_columns,
    simulate_signal_out,
)
from ggTrader.lab.strategy import LabConfig
from ggTrader.lab.wfo.run import run_wfo
from tests.lab.wfo.helpers import TOY_CONFIG, ToyHold, toy_bars


def data(n=100):
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    close = pd.DataFrame({"DEAD": [10.0] * 5 + [float("nan")] * (n - 5), "LIVE": 20.0}, index=idx)
    spans = pd.DataFrame(
        {
            "symbol": ["DEAD"],
            "last_date": [idx[4].date()],
            "has_successor": [False],
            "total_loss": [False],
        }
    )
    cols = group_columns([{}], [], close.columns)
    entries = pd.DataFrame(False, index=idx, columns=cols)
    entries.iloc[0, :] = True
    entries.iloc[8, 0] = True
    return close, spans, SignalOut(entries, entries & False, ({},))


def test_combined_run_applies_forced_exit():
    close, spans, out = data()
    frames = forced_exit_frames(close, spans)
    for loss, expected in [(0.0, 10000.0), (0.5, 7500.0)]:
        sim = simulate_signal_out(out, haircut_close(close, spans, loss), TOY_CONFIG, dead=frames)
        assert sim.value.iloc[-1, 0] == pytest.approx(expected, abs=1e-9)
        assert sim.trades.loc[sim.trades.col == 0, "exit_idx"].tolist() == [4]
        assert sim.entries_suppressed == 1
        assert sim.returns.iloc[4, 0] == pytest.approx(-loss * 0.5, abs=1e-9)


def test_two_end_dates_agree_through_common_history():
    sims = []
    for n in (35, 95):
        close, spans, out = data(n)
        sims.append(
            simulate_signal_out(out, close, TOY_CONFIG, dead=forced_exit_frames(close, spans))
        )
    pd.testing.assert_frame_equal(sims[0].value, sims[1].value.iloc[:35])


def test_non_combined_run_unchanged():
    close, _, out = data()
    a = simulate_signal_out(out, close, TOY_CONFIG)
    b = simulate_signal_out(out, close.copy(), TOY_CONFIG, dead=None)
    pd.testing.assert_frame_equal(a.value, b.value)


def test_combined_without_spans_raises():
    bars = replace(toy_bars(), provenance={"source": "combined"}, spans=None)
    with pytest.raises(ValueError, match="combined bars carry no release spans"):
        run_wfo(
            "toy",
            ToyHold,
            LabConfig(),
            bars,
            bars.close.iloc[:, 0],
            "2020-06-01",
            "2022-03-01",
            "equity",
            TOY_CONFIG,
            ParamGrid.single({"delay": 3}),
            allow_unmasked=True,
            verbose=False,
        )


def test_combined_run_records_diagnostics_and_keeps_spans_in_subset():
    from ggTrader.lab.wfo.inputs import subset_bars

    original = toy_bars()
    spans = pd.DataFrame({"symbol": ["X"], "last_date": [original.close.index[200].date()]})
    bars = replace(original, provenance={"source": "combined"}, spans=spans)
    bars = subset_bars(bars, ["X"])
    result = run_wfo(
        "toy",
        ToyHold,
        LabConfig(),
        bars,
        bars.close.iloc[:, 0],
        "2020-06-01",
        "2022-03-01",
        "equity",
        TOY_CONFIG,
        ParamGrid.single({"delay": 3}),
        allow_unmasked=True,
        verbose=False,
    )
    assert result.diagnostics["dead_stock_rule"]["n_ended_in_window"] == 1
    assert result.diagnostics["dead_stock_rule"]["entries_suppressed"] == 0
    assert result.diagnostics["dead_stock_rule"]["loss"] == 0
