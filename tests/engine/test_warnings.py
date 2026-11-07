"""L3: the whole engine surface runs clean under -W error for pandas/vectorbt deprecations.

A pandas or vectorbt deprecation that the adapter triggers is an upgrade hazard (the next
release removes the behaviour). The adapter must call each API in a way that does not warn.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from ggTrader.engine import indicators as ind
from ggTrader.engine import portfolio as pfm
from ggTrader.engine import signals as sig
from ggTrader.engine import stats, windows

from .panels import make_panel

pytestmark = pytest.mark.tripwire


def _exercise(engine_name: str) -> None:
    panel = make_panel(seed=21, n_bars=320, n_syms=6, holes=True)
    close, high, low = panel["close"], panel["high"], panel["low"]
    rsi = ind.wilder_rsi(close, 14).rsi
    bb = ind.bbands(close, 20, 2.0)
    lower, middle = bb.lower, bb.middle  # each access builds a new frame
    ind.ema(close, [10, 20])
    ind.sma(close, 20)
    ind.macd(close)
    ind.atr(high, low, close)
    rsi.columns = lower.columns = middle.columns = close.columns
    entries = sig.cross_below(rsi, 30) | sig.cross_below(close, lower)
    exits = sig.cross_above(close, middle)
    cfg = {"START_CASH": 10000.0, "FEES": 0.0005, "SLIPPAGE": 0.0005, "FREQ": "1D"}
    groups = ["a", "a", "a", "b", "b", "b"]
    result = pfm.run_signals(
        entries,
        exits,
        close,
        {**cfg, "atr_mult": 2.0},
        groups=groups,
        td_stop=5,
        high=high,
        low=low,
        mask=close.notna(),
    )
    weights = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    weights.iloc[40] = 0.1
    pfm.run_weights(weights, close, cfg, groups=groups)
    stats.window_stats(result.returns, 50, 250)
    stats.dsr(result.returns["a"], 48)
    stats.trade_counts(result.trades, 0, 200)
    windows.stack_windows(close, [(0, 100), (50, 200)])
    windows.slice_positions(close.index, close.index[3], close.index[40])


@pytest.mark.parametrize("name", ["rust", "numba"])
def test_engine_surface_is_free_of_pandas_and_vectorbt_deprecations(name: str) -> None:
    from ggTrader.engine import config

    config.set_engine(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _exercise(name)  # warm any one-off compilation outside the error filter
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        warnings.simplefilter("error", FutureWarning)
        _exercise(name)
