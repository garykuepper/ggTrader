"""Dead-series execution on signal simulations."""

from dataclasses import replace

import pandas as pd
import pytest

from ggTrader.lab.dead_stocks import forced_exit_frames, haircut_close, simulation_close
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
        sim = simulate_signal_out(out, close, TOY_CONFIG, dead=forced_exit_frames(close, spans))
        assert sim.trades.loc[sim.trades.col == 0, "exit_idx"].tolist() == [4]
        assert sim.trades.loc[sim.trades.col == 0, "status"].tolist() == [1]
        sims.append(sim)
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


def test_total_loss_signal_recovers_only_floor():
    close, spans, out = data()
    sim = simulate_signal_out(
        out, simulation_close(close, spans, 1.0), TOY_CONFIG, dead=forced_exit_frames(close, spans)
    )
    recovery = sim.value.iloc[-1, 0] - 5000.0
    assert 0 < recovery < 1e-6 * 5000.0
    assert sim.trades.loc[sim.trades.col == 0, "exit_idx"].tolist() == [4]


def test_crsp_end_guard_refuses_only_columns_that_stop_at_file_end():
    from ggTrader.lab.wfo.inputs import subset_bars

    original = toy_bars()
    cliff = original.close.index[800]
    frame = original.close.copy()
    frame.loc[frame.index > cliff, "X"] = float("nan")
    bars = replace(
        original,
        close=frame,
        last_valid_bar=pd.Series({"X": cliff}),
        provenance={"source": "combined", "crsp_end": str(cliff.date())},
        spans=pd.DataFrame(columns=["symbol", "last_date"]),
    )

    def run(b):
        return run_wfo(
            "toy",
            ToyHold,
            LabConfig(),
            b,
            original.close.X,
            "2020-06-01",
            "2022-03-01",
            "equity",
            TOY_CONFIG,
            ParamGrid.single({"delay": 3}),
            allow_unmasked=True,
            verbose=False,
        )

    with pytest.raises(ValueError, match="X.*end the window on or before.*CRSP end"):
        run(bars)
    trimmed = replace(
        bars,
        **{
            name: bars.frame(name).loc[:cliff]
            for name in ("close", "open", "high", "low", "volume")
        },
    )
    assert run(trimmed).diagnostics["n_symbols"] == 1
    continued = replace(bars, close=original.close, last_valid_bar=original.last_valid_bar)
    assert run(continued).diagnostics["n_symbols"] == 1
    empty = subset_bars(bars, [])
    assert empty.close.shape[1] == 0


def test_dead_stock_diagnostics_distinguish_warmup_and_total_loss():
    bars = toy_bars()
    early, later = bars.close.index[100], bars.close.index[400]
    extra = {
        name: bars.frame(name).rename(columns={"X": "Y"})
        for name in ("close", "open", "high", "low", "volume")
    }
    frames = {name: pd.concat([bars.frame(name), extra[name]], axis=1) for name in extra}
    spans = pd.DataFrame({"symbol": ["X", "Y"], "last_date": [early.date(), later.date()]})
    both = replace(
        bars,
        **frames,
        requested=("X", "Y"),
        last_valid_bar=pd.Series({"X": bars.close.index[-1], "Y": bars.close.index[-1]}),
        provenance={"source": "combined"},
        spans=spans,
    )
    result = run_wfo(
        "toy",
        ToyHold,
        LabConfig(),
        both,
        bars.close.X,
        "2020-06-01",
        "2022-03-01",
        "equity",
        TOY_CONFIG,
        ParamGrid.single({"delay": 3}),
        allow_unmasked=True,
        verbose=False,
        dead_stock_loss=1.0,
    )
    rule = result.diagnostics["dead_stock_rule"]
    assert rule["n_ended_in_window"] == 2 and rule["n_ended_in_evaluation"] == 1
    assert "written off completely" in rule["plain_words"]
    assert "optimistic" not in rule["plain_words"]


def test_negative_control_turning_off_rule_changes_equity(monkeypatch):
    import ggTrader.lab.dead_stocks as rules

    close, spans, out = data(35)
    close.loc[close.index[9] :, "LIVE"] = 40.0
    entries = out.entries.copy()
    entries.iloc[0, 1] = False
    entries.iloc[8, 1] = True
    out = replace(out, entries=entries)
    original = rules.forced_exit_frames
    active = simulate_signal_out(out, close, TOY_CONFIG, dead=original(close, spans))

    def disabled(prices, series):
        frames = original(prices, series)
        return replace(frames, exits=frames.exits & False, entry_ok=frames.entry_ok | True)

    monkeypatch.setattr(rules, "forced_exit_frames", disabled)
    control = simulate_signal_out(
        out, close, TOY_CONFIG, dead=rules.forced_exit_frames(close, spans)
    )
    assert active.trades.loc[active.trades.col == 0, "exit_idx"].tolist() == [4]
    assert active.value.iloc[-1, 0] != control.value.iloc[-1, 0]


def test_combined_intraday_stops_are_rejected():
    bars = replace(
        toy_bars(),
        provenance={"source": "combined"},
        spans=pd.DataFrame(columns=["symbol", "last_date"]),
    )
    with pytest.raises(ValueError, match="intraday.*combined"):
        run_wfo(
            "toy",
            ToyHold,
            LabConfig(),
            bars,
            bars.close.X,
            "2020-06-01",
            "2022-03-01",
            "equity",
            TOY_CONFIG,
            ParamGrid.single({"delay": 3}),
            allow_unmasked=True,
            intraday_stops=True,
            verbose=False,
        )
