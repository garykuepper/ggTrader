"""The crux of the single-pass design: slicing one full-history simulation equals simulating
each window on data truncated at its end (what the legacy walk-forward did).

If this fails the single-pass WFO is invalid; do not adapt the WFO to make it pass.
Variants cover the exit machinery the ported strategies and the lab config can use: the time
stop (``td_stop``), a trailing stop, an ATR trailing stop and a take-profit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.ensemble import Ensemble
from ggTrader.lab.strategies.leak import truncate_bars
from ggTrader.lab.wfo import inputs
from ggTrader.lab.wfo.scorers import SignalScorer, WindowReq
from tests.lab.wfo import helpers as h

pytestmark = pytest.mark.filterwarnings("ignore")

VARIANTS = {
    "plain": ({}, None),
    "time_stop_10": ({}, 10),
    "trailing_stop_5pct": ({"ts_stop": 0.05}, None),
    "atr_stop_2x": ({"atr_mult": 2.0, "atr_period": 14}, None),
    "take_profit_8pct": ({"tp_stop": 0.08}, None),
}


def _scorer(bars, mask, extra, td_stop) -> SignalScorer:
    cls = Ensemble
    combos = cls.grid.combos()
    names = [inputs.combo_name("ensemble", c) for c in combos]
    scorer = SignalScorer(
        cls(td_stop=td_stop),
        bars,
        cls.grid,
        names,
        combos,
        mask,
        {**h.BASE, **extra},
        1 << 30,
    )
    scorer.prepare([])
    return scorer


def _check(bars, make_mask, ends, extra, td_stop) -> None:
    full = _scorer(bars, make_mask(bars), extra, td_stop)
    assert int(full._trades.shape[0]) > 0, "the scenario must trade or the test proves nothing"
    for end in ends:
        cut = truncate_bars(bars, end)
        part = _scorer(cut, make_mask(cut), extra, td_stop)
        n = len(cut.close)
        # equity: bit for bit, every combination, from the first loaded bar
        assert np.array_equal(full._val[:n], part._val), f"equity differs at {end.date()}"
        # entries counted up to the end agree (the legacy whole-simulation trade count)
        req = WindowReq("train", 0, end - pd.DateOffset(months=12), end)
        assert np.array_equal(
            full.trade_counts(req, cumulative=True), part.trade_counts(req, cumulative=True)
        )
        # and so do the window statistics that selection reads
        rows_full, rows_part = full.metrics(req), part.metrics(req)
        for a, b in zip(rows_full, rows_part, strict=True):
            for key in ("sharpe", "sortino", "cagr_pct", "max_drawdown_pct", "total_return_pct"):
                assert a[key] == b[key] or (np.isnan(a[key]) and np.isnan(b[key])), (key, end)
        anchor = WindowReq("anchor", 0, None, end)
        for a, b in zip(full.metrics(anchor), part.metrics(anchor), strict=True):
            assert a["max_drawdown_pct"] == b["max_drawdown_pct"]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.slow
def test_single_pass_equals_truncated_simulations_on_a_synthetic_panel(variant: str) -> None:
    bars = h.small_wfo_bars(seed=17, n_syms=6)
    extra, td_stop = VARIANTS[variant]
    index = bars.close.index
    ends = [index[600], index[800], index[1100]]
    _check(bars, h.open_mask, ends, extra, td_stop)


@h.needs_real
@pytest.mark.slow
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_single_pass_equals_truncated_simulations_on_the_real_tape(variant: str) -> None:
    from ggTrader.lab.strategy import LabConfig

    bars, _ = h.real_bars(n_symbols=40, seed=3)
    extra, td_stop = VARIANTS[variant]
    cfg = LabConfig()

    def make_mask(b):
        return inputs.universe_mask(b.close, "sp500", min_history_bars=cfg.min_history_bars)

    index = bars.close.index
    ends = [pd.Timestamp("2021-07-30", tz="UTC"), pd.Timestamp("2022-01-31", tz="UTC"), index[-1]]
    _check(bars, make_mask, ends, extra, td_stop)


def test_the_check_has_teeth_a_forward_looking_signal_is_caught() -> None:
    """A strategy whose signals peek at the next bar must fail the same comparison."""
    from ggTrader.lab.strategies.contract import SignalOut

    class Peeker(Ensemble):
        def build(self, bars, grid=None):  # type: ignore[override]
            out = super().build(bars, grid)
            ahead = (bars.close.shift(-1) > bars.close).reindex(
                columns=out.entries.columns.get_level_values(-1)
            )
            ahead.columns = out.entries.columns
            return SignalOut(
                entries=out.entries | ahead.fillna(False), exits=out.exits, combos=out.combos
            )

    bars = h.small_wfo_bars(seed=17, n_syms=6)
    combos = Peeker.grid.combos()
    names = [inputs.combo_name("ensemble", c) for c in combos]

    def run(b):
        s = SignalScorer(
            Peeker(), b, Peeker.grid, names, combos, h.open_mask(b), dict(h.BASE), 1 << 30
        )
        s.prepare([])
        return s._val

    end = bars.close.index[800]
    full, part = run(bars), run(truncate_bars(bars, end))
    assert not np.array_equal(full[: len(part)], part)
