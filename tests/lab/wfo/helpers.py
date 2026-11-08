"""Shared fixtures for the vectorbt-engine walk-forward tests: a closed-form toy strategy.

``ToyHold`` buys one symbol on bar ``delay`` and holds forever, so a combination's equity has a
closed form that does not depend on any simulator: ``C`` before the entry and
``C * (1 - s + s * P_t / P_entry)`` after it (``s`` is the position size, no fees or slippage).
``oracle_equity`` is that closed form; the walk-forward tests compare the engine walk-forward
with it fold by fold.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from ggTrader.data.bars import Bars
from ggTrader.lab.strategies.contract import ParamGrid, SignalOut, group_columns, resolve_combos

START_CASH = 10000.0
SIZE = 0.5
TOY_CONFIG: Dict[str, Any] = {
    "START_CASH": START_CASH,
    "FEES": 0.0,
    "SLIPPAGE": 0.0,
    "FREQ": "1D",
    "SIGNAL_POSITION_SIZE": SIZE,
}
DELAYS = (3, 50, 150)


def toy_bars(n_bars: int = 1300, seed: int = 5, symbol: str = "X") -> Bars:
    """One symbol whose log price swings up and down so some folds lose money."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-03-01", periods=n_bars, tz="UTC")
    t = np.arange(n_bars)
    log_price = 0.45 * np.sin(2 * np.pi * t / 420.0) + np.cumsum(rng.normal(0, 0.008, n_bars))
    close = pd.DataFrame({symbol: 100.0 * np.exp(log_price)}, index=index)
    frames = {
        "close": close,
        "open": close * 0.999,
        "high": close * 1.004,
        "low": close * 0.996,
        "volume": pd.DataFrame({symbol: np.full(n_bars, 1e6)}, index=index),
    }
    return Bars(
        requested=(symbol,),
        missing_symbols=(),
        last_valid_bar=pd.Series({symbol: index[-1]}),
        provenance={"source": "toy"},
        **frames,
    )


class ToyHold:
    """Buy on bar ``delay`` and hold; one entry per combination, no exits."""

    name = "toy"
    kind = "signals"
    grid = ParamGrid.from_sweep({"delay": list(DELAYS)})
    live = None
    defaults = {"delay": DELAYS[0]}

    def build(self, bars: Bars, grid: ParamGrid | None = None) -> SignalOut:
        grid = grid if grid is not None else self.grid
        combos = resolve_combos(grid, self.defaults)
        columns = group_columns(combos, grid.swept(), bars.symbols)
        entries = pd.DataFrame(False, index=bars.close.index, columns=columns)
        n_sym = len(bars.symbols)
        for g, combo in enumerate(combos):
            entries.iloc[int(combo["delay"]), g * n_sym : (g + 1) * n_sym] = True
        exits = pd.DataFrame(False, index=bars.close.index, columns=columns)
        return SignalOut(entries=entries, exits=exits, combos=tuple(combos))


class LegacyToy:
    """A pre-engine strategy class (``target_kind``, no ``build``): ``run_wfo`` must refuse it."""

    name = "toy"
    target_kind = "signals"

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg


def oracle_equity(close: pd.Series, delay: int) -> pd.Series:
    """Closed-form equity of buying on bar ``delay`` and holding (no costs)."""
    equity = pd.Series(START_CASH, index=close.index)
    entry_price = close.iloc[delay]
    equity.iloc[delay:] = START_CASH * ((1 - SIZE) + SIZE * close.iloc[delay:] / entry_price)
    return equity


def legacy_frame(bars: Bars) -> pd.DataFrame:
    """The ``(symbol, field)`` frame the legacy walk-forward consumes."""
    return bars.legacy_frame()


# --------------------------------------------------------------------------- real tape

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

from ggTrader.data import snapshot  # noqa: E402
from ggTrader.lab.data import STOCK_BASE_CONFIG  # noqa: E402
from ggTrader.lab.strategy import LabConfig  # noqa: E402
from tests.lab.strategies.helpers import make_bars  # noqa: E402

REAL_PARQUET = Path(snapshot.DEFAULT_SNAPSHOT_DIR) / "ohlcv_20261106.parquet"
needs_real = pytest.mark.skipif(
    not REAL_PARQUET.exists(),
    reason="frozen tape data/snapshots/ohlcv_20261106.parquet is absent (git-ignored; rebuild "
    "with scripts/make_tape_snapshot.py)",
)
SMOKE_START, SMOKE_END = "2021-01-31", "2022-07-31"


def real_bars(n_symbols: int = 40, seed: int = 7, end: str = SMOKE_END):
    """A seeded sample of long-history SP500 symbols plus SPY from the frozen tape.

    Returns ``(bars_without_spy, spy_close)``; the window is the 2-fold smoke window.
    """
    from ggTrader.data.bars import legacy_exclusive_end, load_bars
    from ggTrader.data.membership import FixtureMembershipSource
    from ggTrader.lab.wfo.inputs import subset_bars

    cfg = LabConfig()
    start = pd.Timestamp(SMOKE_START) - pd.Timedelta(days=int(cfg.min_history_bars * 1.6) + 60)
    bars = load_bars(
        universe="sp500",
        universe_span=(SMOKE_START, end),
        extra_symbols=["SPY"],
        start=str(start.date()),
        end=legacy_exclusive_end(end),
        membership=FixtureMembershipSource("sp500_v1_2026-06-02"),
    )
    pool = [s for s in bars.symbols if s != "SPY" and bars.close[s].notna().sum() > 700]
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(pool, n_symbols, replace=False))
    return subset_bars(bars, chosen), bars.close["SPY"].dropna()


def small_wfo_bars(seed: int = 41, n_syms: int = 6):
    """Synthetic bars (with holes) long enough for a 3-fold WFO from 2020-06-01."""
    return make_bars(seed=seed, n_bars=1300, n_syms=n_syms, holes=True)


def open_mask(bars) -> pd.DataFrame:
    """Everything eligible once a symbol has 60 closes (a synthetic PIT mask)."""
    return bars.close.notna().cumsum() >= 60


MINI = {"eval_start": "2020-06-01", "eval_end": "2022-03-01"}
BASE = dict(STOCK_BASE_CONFIG)
