"""Scoring sources of the walk-forward: one full-history simulation sliced per window (signal
strategies) and fresh-start stacked windows (weight strategies).

Both answer the same questions for a window (a :class:`WindowReq`): the per-combination
metrics the legacy ``curve_stats`` produced, the window returns and equity of one combination,
and trade counts. The statistics come from ``engine.stats.window_stats``.

**Signal path** (spec 5.4 items 2, 4, 7): the legacy sweep simulated ``ohlcv.loc[:window_end]``
from the first loaded bar, so positions and cash carry into every window and the equity at bar
``t`` depends only on bars ``<= t``. One simulation over the whole history therefore gives
every window's equity by slicing, including the anchor (first bar to ``train_end``), the train
sweep, the deployed combination's test window and the live sweep. This is proved against
per-window simulations on truncated data in ``tests/lab/wfo/test_causality.py``.

**Weight path** (spec 5.4 item 3): the legacy weight sweep plans only from ``window_start``, so
every (combination, window) starts flat with fresh cash. Slicing a full-history simulation
would carry positions in, so each window is simulated on its own, stacked side by side into one
``run_weights`` call per window kind. The one exception is the anchor: every anchor window
starts at the first loaded bar and ends at that fold's ``train_end``, so all anchors are
prefixes of one simulation from the first bar (decisions after ``train_end`` cannot change the
equity before it) and are sliced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from ggTrader.data.bars import Bars
from ggTrader.engine import config as engine_config
from ggTrader.engine import portfolio as engine_portfolio
from ggTrader.engine import stats, windows
from ggTrader.lab.data import rebalance_dates
from ggTrader.lab.strategies.contract import (
    COMBO_LEVEL,
    SYMBOL_LEVEL,
    ParamGrid,
    group_columns,
    simulate_signal_out,
)

KINDS = ("anchor", "train", "test", "live")
#: Windows whose statistics feed a selection (winner, anchor, live pick) carry the degeneracy
#: guards. The test window measures the book that was deployed, so only the flat-curve guard
#: applies to it (a carried-in position earns real returns without an entry).
GUARDED_KINDS = frozenset({"anchor", "train", "live"})
PSEUDO_START = "2000-01-03"


@dataclass(frozen=True)
class WindowReq:
    """A scoring window: ``start=None`` means the first loaded bar (anchor windows)."""

    kind: str
    fold: int
    start: pd.Timestamp | None
    end: pd.Timestamp


class Scorer(Protocol):
    """What the walk-forward asks of a scoring source."""

    kind: str
    n_simulations: int
    n_chunks: int
    timings: Dict[str, float]

    def prepare(self, requests: Sequence[WindowReq]) -> None:
        """Run the simulations the requests need (once)."""
        ...

    def metrics(self, req: WindowReq, combos: Sequence[int] | None = None) -> List[Dict[str, Any]]:
        """Legacy-style metric rows (empty when the window has fewer than two bars)."""
        ...

    def returns(self, req: WindowReq, combo: int) -> pd.Series:
        """The window's returns of one combination (from the second bar on)."""
        ...

    def equity(self, req: WindowReq, combo: int) -> pd.Series:
        """The window's equity bars of one combination."""
        ...

    def window_dates(self, req: WindowReq) -> pd.DatetimeIndex:
        """The window's bar dates (empty when the window could not be simulated)."""
        ...

    def trade_counts(self, req: WindowReq, *, cumulative: bool) -> np.ndarray:
        """Trades per combination: entries inside the window, or all entries up to its end."""
        ...


def metric_rows(
    table: pd.DataFrame,
    total: pd.Series,
    vol: pd.Series,
    columns: Sequence[int],
    n_trades: np.ndarray,
    names: Sequence[str],
    params: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Legacy ``curve_stats`` rows (percent units) from an engine statistics table."""
    rows: List[Dict[str, Any]] = []
    for pos, col in enumerate(columns):
        rows.append(
            {
                "combo": names[col],
                "params": params[col],
                "total_return_pct": float(total.iloc[pos]) * 100,
                "cagr_pct": float(table.loc["cagr"].iloc[pos]) * 100,
                "sharpe": float(table.loc["sharpe"].iloc[pos]),
                "sortino": float(table.loc["sortino"].iloc[pos]),
                "ann_vol_pct": float(vol.iloc[pos]) * 100,
                "max_drawdown_pct": float(table.loc["max_drawdown"].iloc[pos]) * 100,
                "n_trades": int(n_trades[col]),
            }
        )
    return rows


def _window_metrics(
    ret: pd.DataFrame,
    start: int,
    end: int,
    columns: Sequence[int],
    n_trades: np.ndarray,
    names: Sequence[str],
    params: Sequence[Mapping[str, Any]],
    *,
    guarded: bool = False,
    min_losing_days: int = engine_config.MIN_LOSING_DAYS,
) -> List[Dict[str, Any]]:
    """Metric rows of bars ``start..end-1`` for the given columns of ``ret``.

    ``guarded`` windows (those that feed a selection) blank Sharpe and Sortino of a combination
    with no entry inside the window (spec 5.6) and Sortino below ``min_losing_days`` losing days.
    """
    if end - start < 2:
        return []
    if guarded:
        entries = pd.Series(np.asarray(n_trades), index=ret.columns)
        table = stats.window_stats(
            ret, start, end, n_trades=entries, min_losing_days=min_losing_days
        )
    else:
        table = stats.window_stats(ret, start, end, min_losing_days=0)
    window = ret.iloc[start + 1 : end]
    total = stats.total_return(window)
    vol = pd.Series(stats.volatility(window))
    return metric_rows(table, total, vol, columns, n_trades, names, params)


def check_combination_order(
    labels: Sequence[tuple], params: Sequence[Mapping[str, Any]], swept: Sequence[str]
) -> None:
    """Raise unless the strategy built its combinations in the order of ``params``.

    ``labels`` is the grid-level tuple of each combination in the order the strategy emitted it
    (the swept parameter values, outermost level first). Column ``i`` is labelled ``names[i]``
    by position, so a different order would silently mislabel every winner (review MINOR-2).
    """
    if not swept:
        return
    expected = [tuple(p[k] for k in swept) for p in params]
    if list(labels) != expected:
        raise ValueError(
            "strategy built its combinations in a different order from ParamGrid.combos(); "
            f"first built {list(labels)[:2]}, expected {expected[:2]}"
        )


class SignalScorer:
    """One grouped full-history simulation per strategy configuration, sliced per window."""

    kind = "signals"

    def __init__(
        self,
        strategy: Any,
        bars: Bars,
        grid: ParamGrid,
        names: Sequence[str],
        params: Sequence[Mapping[str, Any]],
        mask: pd.DataFrame | None,
        base_config: Mapping[str, Any],
        max_bytes: int,
        min_losing_days: int = engine_config.MIN_LOSING_DAYS,
        *,
        intraday_stops: bool = False,
        halt_mask: pd.DataFrame | None = None,
        order_sequence: str | None = None,
        stop_exit_price: str | None = None,
    ) -> None:
        self.strategy, self.bars, self.grid = strategy, bars, grid
        self.names, self.params, self.mask = list(names), list(params), mask
        self.base_config, self.max_bytes = dict(base_config), max_bytes
        self.min_losing_days = min_losing_days
        self.intraday_stops = intraday_stops
        self.halt_mask = halt_mask
        self.order_sequence = order_sequence
        self.stop_exit_price = stop_exit_price
        self.n_simulations = 0
        self.n_chunks = 0
        self.timings: Dict[str, float] = {}
        self._prepared = False

    def prepare(self, requests: Sequence[WindowReq]) -> None:  # noqa: ARG002
        """Build every combination's signals and simulate the whole history once."""
        if self._prepared:
            return
        t0 = time.time()
        out = self.strategy.build(self.bars, self.grid)
        n_groups = len(self.names)
        if len(out.combos) != n_groups:
            raise ValueError(f"strategy built {len(out.combos)} combinations, expected {n_groups}")
        swept = self.grid.swept()
        built = [tuple(c[k] for k in swept) for c in out.combos]
        check_combination_order(built, self.params, swept)
        if not isinstance(out.entries.columns, pd.MultiIndex):
            raise ValueError("signal columns must identify combinations and symbols")
        expected = group_columns(out.combos, swept, out.symbols)
        if not out.entries.columns.equals(expected) or out.entries.columns.names != expected.names:
            raise ValueError("signal columns do not match combination and symbol order")
        self.timings["build_s"] = time.time() - t0
        t1 = time.time()
        use_atr = self.base_config.get("atr_mult") is not None
        sim = simulate_signal_out(
            out,
            self.bars.close,
            self.base_config,
            mask=self.mask,
            high=self.bars.high if use_atr or self.intraday_stops else None,
            low=self.bars.low if use_atr or self.intraday_stops else None,
            stop_open=self.bars.open if self.intraday_stops else None,
            halt_mask=self.halt_mask,
            order_sequence=self.order_sequence,
            stop_exit_price=self.stop_exit_price,
            max_bytes=self.max_bytes,
        )
        trailing = out.sl_stop if out.sl_stop is not None else self.base_config.get("ts_stop")
        profit = out.tp_stop if out.tp_stop is not None else self.base_config.get("tp_stop")
        atr = self.base_config.get("atr_mult") if trailing is None else None
        self.stop_settings = {
            "trailing_fraction": None if trailing is None else float(trailing),
            "take_profit_fraction": None if profit is None else float(profit),
            "atr_multiplier": None if atr is None else float(atr),
            "atr_period": (
                None
                if atr is None
                else int(self.base_config.get("atr_period", engine_portfolio.DEFAULT_ATR_PERIOD))
            ),
            "time_stop_bars": None if out.td_stop is None else int(out.td_stop),
        }
        self.timings["simulate_s"] = time.time() - t1
        self.n_simulations, self.n_chunks = 1, sim.n_chunks
        if sim.value.shape[1] != n_groups:
            raise ValueError("simulation returned an unexpected number of groups")
        self.index = self.bars.close.index
        self._ret = sim.returns.set_axis(range(n_groups), axis=1)
        self._val = sim.value.to_numpy()
        n_sym = len(out.symbols)
        trades = sim.trades[["col", "entry_idx"]]
        self._trades = pd.DataFrame(
            {
                "group": (trades["col"].to_numpy() // n_sym).astype("int64"),
                "entry_idx": trades["entry_idx"].to_numpy(),
            }
        )
        self._prepared = True

    def _span(self, req: WindowReq) -> tuple[int, int]:
        start = self.index[0] if req.start is None else req.start
        return windows.slice_positions(self.index, start, req.end)

    def metrics(self, req: WindowReq, combos: Sequence[int] | None = None) -> List[Dict[str, Any]]:
        """Metric rows of the window for ``combos`` (default all), sliced from the simulation."""
        a, b = self._span(req)
        columns = list(range(len(self.names))) if combos is None else list(combos)
        counts = self._counts(a, b)
        return _window_metrics(
            self._ret.iloc[:, columns],
            a,
            b,
            range(len(columns)),
            counts[columns],
            [self.names[c] for c in columns],
            [self.params[c] for c in columns],
            guarded=req.kind in GUARDED_KINDS,
            min_losing_days=self.min_losing_days,
        )

    def returns(self, req: WindowReq, combo: int) -> pd.Series:
        """Returns ``a+1..b-1`` of one combination."""
        a, b = self._span(req)
        return self._ret.iloc[a + 1 : b, combo]

    def equity(self, req: WindowReq, combo: int) -> pd.Series:
        """Equity bars ``a..b-1`` of one combination, indexed by date."""
        a, b = self._span(req)
        return pd.Series(self._val[a:b, combo], index=self.index[a:b], name=self.names[combo])

    def window_dates(self, req: WindowReq) -> pd.DatetimeIndex:
        """Bar dates of the window."""
        a, b = self._span(req)
        return self.index[a:b]

    def _counts(self, start: int, end: int) -> np.ndarray:
        inside = self._trades[
            (self._trades["entry_idx"] >= start) & (self._trades["entry_idx"] < end)
        ]
        return np.bincount(inside["group"].to_numpy(), minlength=len(self.names))

    def trade_counts(self, req: WindowReq, *, cumulative: bool) -> np.ndarray:
        """Entries inside the window (OD12) or every entry up to its end (the legacy count)."""
        a, b = self._span(req)
        return self._counts(0 if cumulative else a, b)


@dataclass(frozen=True, eq=False)
class _WindowSim:
    """One simulated weight window: dates, per-combination equity/returns and its trades.

    ``trades`` holds one row per vectorbt trade record (a partial reduction is its own
    record): ``group`` is the combination, ``col`` the column, ``entry_idx``/``exit_idx`` bar
    positions inside the window and ``open`` whether the position was still open.
    """

    index: pd.DatetimeIndex
    value: np.ndarray
    returns: np.ndarray
    trades: pd.DataFrame


class WeightScorer:
    """Fresh-start windows stacked into one ``run_weights`` call per window kind."""

    kind = "weights"

    def __init__(
        self,
        strategy: Any,
        bars: Bars,
        grid: ParamGrid,
        names: Sequence[str],
        params: Sequence[Mapping[str, Any]],
        mask: pd.DataFrame | None,
        base_config: Mapping[str, Any],
        max_bytes: int,
        min_losing_days: int = engine_config.MIN_LOSING_DAYS,
    ) -> None:
        self.strategy, self.bars, self.grid = strategy, bars, grid
        self.names, self.params, self.mask = list(names), list(params), mask
        self.base_config, self.max_bytes = dict(base_config), max_bytes
        self.min_losing_days = min_losing_days
        self.index = bars.close.index
        self.n_simulations = 0
        self.n_chunks = 0
        self.timings: Dict[str, float] = {}
        self._sims: Dict[tuple, _WindowSim | None] = {}

    def _key(self, req: WindowReq) -> tuple:
        if req.kind == "anchor":  # every anchor is a prefix of one window from the first bar
            return ("anchor", self.index[0], self.index[-1])
        return (req.kind, req.start, req.end)

    def prepare(self, requests: Sequence[WindowReq]) -> None:
        """Simulate every requested window, one stacked ``run_weights`` call per kind."""
        wanted: Dict[str, List[tuple]] = {}
        for req in requests:
            key = self._key(req)
            if key not in self._sims and key not in wanted.setdefault(key[0], []):
                wanted[key[0]].append(key)
        for kind in KINDS:
            if wanted.get(kind):
                self._simulate(wanted[kind])

    def _check_weight_order(self, w: pd.DataFrame, n_sym: int, n_combos: int) -> None:
        """Refuse ambiguous combination blocks or differing symbol orders."""
        swept = self.grid.swept()
        cols = w.columns
        if not len(cols):
            return
        if not n_sym or len(cols) != n_sym * n_combos or not isinstance(cols, pd.MultiIndex):
            raise ValueError("weight columns must contain complete combination blocks")
        symbols = cols.get_level_values(-1)[:n_sym]
        if not symbols.is_unique:
            raise ValueError("weight symbol order contains duplicates")
        for group in range(n_combos):
            block = cols[group * n_sym : (group + 1) * n_sym]
            if not block.get_level_values(-1).equals(symbols):
                raise ValueError("weight symbol order differs between combination blocks")
            if len({tuple(col[:-1]) for col in block}) != 1:
                raise ValueError("weight combination block contains mixed labels")
        if list(cols.names) != [*(swept or [COMBO_LEVEL]), SYMBOL_LEVEL]:
            raise ValueError("weight combination column levels do not match the grid")
        labels = [tuple(cols[g * n_sym][:-1]) for g in range(n_combos)]
        if swept:
            check_combination_order(labels, self.params, swept)
        elif labels != [(g,) for g in range(n_combos)]:
            raise ValueError("weight combination order does not match the grid")

    def _simulate(self, keys: List[tuple]) -> None:
        t0 = time.time()
        n_combos = len(self.names)
        close = self.bars.close.ffill()
        blocks: List[tuple | None] = []
        for key in keys:
            _, start, end = key
            i0, i1 = windows.slice_positions(self.index, start, end)
            rebal = rebalance_dates(self.index, start, end)
            if not rebal:
                blocks.append(None)
                continue
            weights = self.strategy.build(
                self.bars, self.grid, pd.DatetimeIndex(rebal), eligible=self.mask
            )
            w = weights.iloc[i0:i1]
            n_sym = w.shape[1] // n_combos
            self._check_weight_order(w, n_sym, n_combos)
            symbols = list(w.columns.get_level_values(-1)[:n_sym])
            c = close[symbols].iloc[i0:i1].to_numpy() if n_sym else np.empty((i1 - i0, 0))
            blocks.append((i0, i1, w.to_numpy(), np.tile(c, (1, n_combos)), n_sym))
        self.timings["build_s"] = self.timings.get("build_s", 0.0) + time.time() - t0

        t1 = time.time()
        length = max((b[1] - b[0] for b in blocks if b is not None), default=0)
        w_parts, c_parts, labels = [], [], []
        for k, block in enumerate(blocks):
            if block is None:
                continue
            i0, i1, w, c, n_sym = block
            pad = length - (i1 - i0)
            w_parts.append(np.vstack([w, np.full((pad, w.shape[1]), np.nan)]))
            c_parts.append(np.vstack([c, np.full((pad, c.shape[1]), np.nan)]))
            labels.extend(np.repeat([k * n_combos + g for g in range(n_combos)], n_sym))
        all_groups = [
            k * n_combos + g
            for k, block in enumerate(blocks)
            if block is not None
            for g in range(n_combos)
        ]
        start_cash = float(self.base_config["START_CASH"])
        value = returns = None
        trades = pd.DataFrame({k: [] for k in ("group", "col", "entry_idx", "exit_idx", "status")})
        if labels:
            pseudo = pd.bdate_range(PSEUDO_START, periods=length, tz="UTC")
            frame_w = pd.DataFrame(np.hstack(w_parts), index=pseudo)
            frame_c = pd.DataFrame(np.hstack(c_parts), index=pseudo)
            sim = engine_portfolio.run_weights(
                frame_w,
                frame_c,
                self.base_config,
                groups=labels,
                all_groups=all_groups,
                max_bytes=self.max_bytes,
            )
            self.n_simulations += 1
            self.n_chunks += sim.n_chunks
            value, returns = sim.value, sim.returns
            trades = sim.trades[["group", "col", "entry_idx", "exit_idx", "status"]]
        for k, (key, block) in enumerate(zip(keys, blocks, strict=True)):
            if block is None:
                self._sims[key] = None
                continue
            i0, i1 = block[0], block[1]
            cols = [k * n_combos + g for g in range(n_combos)]
            if value is None:
                val = np.full((i1 - i0, n_combos), start_cash)
                ret = np.zeros((i1 - i0, n_combos))
            else:
                val = value[cols].to_numpy()[: i1 - i0]
                ret = returns[cols].to_numpy()[: i1 - i0]
            mine = trades[trades["group"].isin(cols)].copy()
            mine["group"] = (mine["group"] - k * n_combos).astype("int64")
            self._sims[key] = _WindowSim(
                index=self.index[i0:i1], value=val, returns=ret, trades=mine.reset_index(drop=True)
            )
        self.timings["simulate_s"] = self.timings.get("simulate_s", 0.0) + time.time() - t1

    def _window(self, req: WindowReq) -> tuple[_WindowSim | None, int]:
        sim = self._sims[self._key(req)]
        if sim is None:
            return None, 0
        end = len(sim.index)
        if req.kind == "anchor":
            end = windows.slice_positions(sim.index, sim.index[0], req.end)[1]
        return sim, end

    def metrics(self, req: WindowReq, combos: Sequence[int] | None = None) -> List[Dict[str, Any]]:
        """Metric rows of the window for ``combos`` (default all)."""
        sim, end = self._window(req)
        if sim is None:
            return []
        columns = list(range(len(self.names))) if combos is None else list(combos)
        ret = pd.DataFrame(sim.returns[:, columns], index=sim.index, columns=range(len(columns)))
        counts = self.trade_counts(req, cumulative=False)
        return _window_metrics(
            ret,
            0,
            end,
            range(len(columns)),
            counts[columns],
            [self.names[c] for c in columns],
            [self.params[c] for c in columns],
            guarded=req.kind in GUARDED_KINDS,
            min_losing_days=self.min_losing_days,
        )

    def returns(self, req: WindowReq, combo: int) -> pd.Series:
        """Returns from the window's second bar on, for one combination."""
        sim, end = self._window(req)
        assert sim is not None
        return pd.Series(sim.returns[1:end, combo], index=sim.index[1:end])

    def equity(self, req: WindowReq, combo: int) -> pd.Series:
        """Equity bars of the window, for one combination."""
        sim, end = self._window(req)
        assert sim is not None
        return pd.Series(sim.value[:end, combo], index=sim.index[:end], name=self.names[combo])

    def window_dates(self, req: WindowReq) -> pd.DatetimeIndex:
        """Bar dates of the window (empty when it has no rebalance date)."""
        sim, end = self._window(req)
        return pd.DatetimeIndex([]) if sim is None else sim.index[:end]

    def trade_counts(self, req: WindowReq, *, cumulative: bool) -> np.ndarray:  # noqa: ARG002
        """Trade records of a simulation truncated at the window's end, per combination.

        vectorbt writes one record per exit order, so a position that is reduced several times
        after ``end`` is several records in the full window but one open record in a simulation
        that stops at ``end`` (the legacy count). Records closed before ``end`` count as they
        are; every position still open at ``end`` counts once. For the stacked train/test/live
        windows ``end`` is the window's last bar, so this is the plain record count.
        """
        sim, end = self._window(req)
        if sim is None:
            return np.zeros(len(self.names), dtype="int64")
        t = sim.trades
        t = t[t["entry_idx"] < end]
        closed_before = (t["status"] != 0) & (t["exit_idx"] < end)
        counts = np.bincount(t.loc[closed_before, "group"], minlength=len(self.names))
        still = t.loc[~closed_before, ["group", "col", "entry_idx"]].drop_duplicates()
        return counts + np.bincount(still["group"], minlength=len(self.names))
