"""Dead-series execution on target-weight simulations."""

from dataclasses import replace

import pandas as pd
import pytest

from ggTrader.lab.dead_stocks import forced_exit_frames, simulation_close
from ggTrader.lab.strategies.contract import ParamGrid, group_columns
from ggTrader.lab.wfo import scorers
from tests.lab.wfo.helpers import TOY_CONFIG, toy_bars


class HoldWeight:
    def build(self, bars, grid, rebal, eligible=None):
        cols = group_columns([{}], [], bars.symbols)
        weights = pd.DataFrame(float("nan"), index=bars.close.index, columns=cols)
        weights.iloc[0, 0] = 0.5
        weights.iloc[8, 0] = 0.5
        return weights


def simulate(monkeypatch, loss, combined=True):
    bars = toy_bars(n_bars=40)
    close = bars.close.copy()
    close.iloc[:, 0] = 10.0
    close.iloc[5:, 0] = float("nan")
    spans = pd.DataFrame({"symbol": ["X"], "last_date": [close.index[4].date()]})
    bars = replace(
        bars,
        close=close,
        provenance={"source": "combined" if combined else "db"},
        spans=spans if combined else None,
    )
    frames = forced_exit_frames(close, spans) if combined else None
    sim_close = simulation_close(close, spans, loss) if combined else None
    inputs = scorers.ScoringInputs(
        HoldWeight(),
        bars,
        ParamGrid.single({}),
        ["test"],
        [{}],
        None,
        TOY_CONFIG,
        1_000_000_000,
        dead_frames=frames,
        sim_close=sim_close,
    )
    monkeypatch.setattr(scorers, "rebalance_dates", lambda *args: [close.index[0]])
    scorer = scorers.WeightScorer.from_inputs(inputs)
    req = scorers.WindowReq("test", 0, close.index[0], close.index[-1])
    scorer.prepare([req])
    return scorer, req


def test_combined_run_weight_strategy_exits_dead_stock(monkeypatch):
    scorer, req = simulate(monkeypatch, 0.5)
    equity = scorer.equity(req, 0)
    assert equity.iloc[-1] == pytest.approx(7500.0, abs=1e-9)
    assert equity.pct_change().iloc[4] == pytest.approx(-0.25, abs=1e-9)
    assert scorer._sims[scorer._key(req)].trades.exit_idx.tolist() == [4]


def test_total_loss_is_zero_recovery(monkeypatch):
    scorer, req = simulate(monkeypatch, 1.0)
    recovery = scorer.equity(req, 0).iloc[-1] - 5000.0
    assert 0 < recovery < 1e-6 * 5000.0
    assert scorer._sims[scorer._key(req)].trades.exit_idx.tolist() == [4]


def test_weight_rule_not_applied_for_db_source(monkeypatch):
    scorer, req = simulate(monkeypatch, 0.0, combined=False)
    assert scorer.equity(req, 0).iloc[-1] == pytest.approx(10000.0, abs=1e-9)


def test_weight_diagnostics_suppressed_entries_are_not_applicable():
    from ggTrader.lab.strategies.xs_momentum import XsMomentum
    from ggTrader.lab.strategy import LabConfig
    from ggTrader.lab.wfo.run import run_wfo

    bars = toy_bars()
    spans = pd.DataFrame({"symbol": ["X"], "last_date": [bars.close.index[400].date()]})
    bars = replace(bars, provenance={"source": "combined"}, spans=spans)
    result = run_wfo(
        "xs_momentum",
        XsMomentum,
        LabConfig(),
        bars,
        bars.close.X,
        "2020-06-01",
        "2022-03-01",
        "equity",
        TOY_CONFIG,
        ParamGrid.single({"lookback": 30, "skip": 1, "top_n": 1}),
        mask=bars.close.notna(),
        verbose=False,
    )
    assert result.diagnostics["dead_stock_rule"]["entries_suppressed"] is None
