"""Grouped portfolio simulation: ``run_signals`` and ``run_weights`` around vectorbt.

Both functions make one ``Portfolio.from_signals`` / ``from_orders`` call per memory chunk
with ``cash_sharing=True`` and one group per strategy/combination, the engine passed
explicitly. They take the same ``base_config`` keys as the legacy ``lab/simulate.py``
(START_CASH, FEES, SLIPPAGE, FREQ, SIGNAL_POSITION_SIZE, ts_stop, tp_stop, atr_mult,
atr_period) and return a frozen :class:`SimResult`. Equity curves are bit-identical to the
legacy functions on the same inputs (tests/engine/test_portfolio.py).

Memory (spec 5.5): vectorbt OSS has no chunking, so the grid columns are split into chunks
whose estimated peak stays under ``max_bytes`` (default 5 GiB). A group is never split
because its columns share one cash pool. Each chunk is reduced to the arrays ``SimResult``
needs (its equity columns and trade records, copied) and then released, with the allocator
asked to hand the freed pages back. Input grids allocated before chunking are outside the
estimate, so ``max_bytes`` is not a process-wide memory cap.
"""

from __future__ import annotations

import ctypes
import gc
import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from ggTrader.engine import config as engine_config
from ggTrader.engine import indicators as ind
from ggTrader.engine import signals as sig

#: Base-config keys the legacy ``lab/simulate.py`` applied and this module does not: passing
#: one raises instead of silently simulating un-scaled sizes (review MINOR-2).
UNSUPPORTED_LEGACY_KEYS = ("vol_target", "vol_lookback", "vol_cap")
#: Stop keys that only ``from_signals`` can apply; ``run_weights`` would drop them silently.
SIGNAL_ONLY_KEYS = ("ts_stop", "atr_mult", "tp_stop", "stop_entry_price")
DEFAULT_MAX_BYTES = 5 * 1024**3
#: Working arrays vectorbt allocates per column beyond the dense inputs. Measured on the full
#: ensemble grid (48 x 605 x 1,929 bars, research/programme/vectorbt-engine/p4-evidence/
#: p2b_memory_summary.json): a chunk peaks at about 10.7 bar-column cells of 8 bytes above the
#: process baseline; 12 keeps the estimate ~12% above that (the spec's first guess, 8, left the
#: measured single-call peak 18% above its estimate).
WORKING_ARRAYS = 12
BYTES_PER_CELL = 8
DEFAULT_SIGNAL_POSITION_SIZE = 0.02
DEFAULT_ATR_PERIOD = 14
MIN_ATR_STOP = 0.001
TIME_STOP_WARNING = (
    "td_stop is based on entry signals, not filled positions; an earlier entry signal "
    "can close a later trade early, prevent an entry, or cause a hold beyond the limit."
)
TRADE_COLUMNS = (
    "id",
    "col",
    "group",
    "symbol",
    "size",
    "entry_idx",
    "entry_price",
    "entry_fees",
    "exit_idx",
    "exit_price",
    "exit_fees",
    "pnl",
    "return",
    "direction",
    "status",
    "parent_id",
)


@dataclass(frozen=True, eq=False)
class SimResult:
    """Result of one grouped simulation.

    Attributes:
        value: equity curve, one column per group, in first-appearance order.
        returns: per-bar returns of ``value`` (first bar 0.0), the legacy definition.
        trades: one row per trade; ``col`` is the global column position, ``entry_idx`` and
            ``exit_idx`` are bar positions in ``value.index``.
        n_trades: trade count per group over the whole simulation.
        engine: the vectorbt engine the simulation was asked to use.
        n_chunks: how many memory chunks the columns were split into.
    """

    value: pd.DataFrame
    returns: pd.DataFrame
    trades: pd.DataFrame
    n_trades: pd.Series
    engine: str
    n_chunks: int


def estimate_peak_bytes(n_bars: int, n_cols: int, n_dense: int = 1) -> int:
    """Estimated peak bytes: bars x columns x 8 x (dense inputs + working arrays)."""
    return int(n_bars) * int(n_cols) * BYTES_PER_CELL * (int(n_dense) + WORKING_ARRAYS)


def atr_stop_fraction(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int, mult: float
) -> pd.DataFrame:
    """ATR trailing-stop distance as a fraction of close, for ``sl_stop``.

    ``mult * ATR / close`` floored at 0.1%, forward-filled, ``inf`` (no stop) before the
    first valid ATR, as the legacy ``compute_atr_stop`` plus the ``simulate_signals`` fill.

    Known differences from the legacy stop (review MINOR-3, class (ii), pinned in
    tests/engine/test_portfolio.py): the ATR is taken on the forward-filled close, and
    vectorbt's true range skips a NaN previous close where the legacy ``np.maximum`` propagates
    it. Series with interior gaps or late listings therefore differ in single cells (the first
    valid ATR, and the bar where the 14-bar window leaves a gap); gap-free series agree to 1e-9.
    """
    atr = ind.atr(high, low, close, window).atr
    atr.columns = close.columns
    stop = (mult * atr / close).clip(lower=MIN_ATR_STOP)
    return stop.ffill().fillna(np.inf)


def run_signals(
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    close: pd.DataFrame,
    base_config: Mapping[str, Any],
    *,
    groups: Sequence[Hashable] | None = None,
    all_groups: Sequence[Hashable] | None = None,
    mask: pd.DataFrame | None = None,
    td_stop: int | None = None,
    size: float | pd.DataFrame | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    stop_open: pd.DataFrame | None = None,
    halt_mask: pd.DataFrame | None = None,
    order_sequence: str | None = None,
    stop_exit_price: str | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> SimResult:
    """Simulate boolean entry/exit signals, one cash pool per group, in one grouped call.

    Args:
        entries: bool (time x column) entry signals.
        exits: bool frame, same shape as ``entries``.
        close: float (time x column) closes, same shape; forward-filled here.
        base_config: START_CASH, FEES, SLIPPAGE, FREQ; optional SIGNAL_POSITION_SIZE (default
            0.02), ts_stop (fixed trailing stop fraction), atr_mult (+ atr_period, needs
            ``high``/``low``), tp_stop (take-profit fraction), stop_entry_price (``close``
            or ``fillprice``; defaults to ``close``).
        groups: group label per column (default one group ``"all"``).
        all_groups: labels that must appear in the output; those with no columns stay flat.
        mask: bool frame ANDed into ``entries`` before the time stop (point-in-time universe).
        td_stop: schedule an exit signal this many bars after each masked entry signal.
            This is not a maximum holding-period guarantee: fills and later positions
            can differ from entry signals.
        size: per-entry size as a fraction of the group's equity, a float or a per-bar frame
            (conviction or volatility-target sizing); overrides SIGNAL_POSITION_SIZE.
        high: highs for the ATR stop; ``low`` likewise.
        stop_open: opt into intraday stops; missing bars are forbidden during positions.
            With grid columns, open/high/low may instead have one column per symbol.
            Cannot be combined with the unreliable ``td_stop``.
        halt_mask: verified boolean halt bars, either grid-aligned or one column per symbol.
            Any simulated position overlapping one is rejected; this is not a halt-fill model.
            The caller must establish coverage, timing and identity; false is not proof of trading.
        order_sequence: ``default`` or ``auto``; omitted uses auto only for intraday stops.
        stop_exit_price: ``stoplimit`` skips slippage on stop exits; ``stopmarket`` applies it.
            Neither models an unfilled limit order. Omitted preserves mode defaults.
        max_bytes: chunking budget for the estimated peak memory.
    """
    _reject_unsupported(base_config)
    if order_sequence not in (None, "default", "auto"):
        raise ValueError("order_sequence must be 'default' or 'auto'")
    if stop_exit_price not in (None, "stoplimit", "stopmarket"):
        raise ValueError("stop_exit_price must be 'stoplimit' or 'stopmarket'")
    stop_entry_price = base_config.get("stop_entry_price", "close")
    if stop_entry_price is None:
        stop_entry_price = "close"
    if stop_entry_price not in ("close", "fillprice"):
        raise ValueError("stop_entry_price must be 'close' or 'fillprice'")
    _check_shapes(close, entries=entries, exits=exits)
    halt_positions = {}
    if halt_mask is not None:
        if (
            not isinstance(halt_mask, pd.DataFrame)
            or not halt_mask.index.equals(close.index)
            or not halt_mask.columns.is_unique
            or halt_mask.isna().to_numpy().any()
            or not all(pd.api.types.is_bool_dtype(dtype) for dtype in halt_mask.dtypes)
        ):
            raise ValueError("halt_mask must have matching dates and known boolean states")
        if halt_mask.columns.equals(close.columns):
            halt_columns = np.arange(len(close.columns))
        elif isinstance(close.columns, pd.MultiIndex):
            halt_columns = halt_mask.columns.get_indexer(close.columns.get_level_values(-1))
            if (halt_columns < 0).any():
                raise ValueError("halt_mask must cover every symbol")
        else:
            raise ValueError("halt_mask must match close columns")
        for column in np.unique(halt_columns):
            positions = np.flatnonzero(halt_mask.iloc[:, column].to_numpy(dtype=bool))
            if len(positions):
                halt_positions[int(column)] = positions
    entries = _as_bool(entries)
    exits = _as_bool(exits)
    if mask is not None:
        entries = entries & _as_bool(
            mask.reindex(index=entries.index, columns=entries.columns, fill_value=False)
        )
    gaps: dict[int, np.ndarray] = {}
    stop_columns = None
    if stop_open is not None:
        if td_stop is not None:
            raise ValueError("td_stop is unreliable and cannot be combined with intraday stops")
        if base_config.get("atr_mult") is not None and base_config.get("ts_stop") is None:
            raise ValueError("ATR stops are not supported with intraday stop prices")
        if base_config.get("ts_stop") is None and base_config.get("tp_stop") is None:
            raise ValueError("intraday stop prices require a configured price stop")
        for key in ("ts_stop", "tp_stop"):
            value = base_config.get(key)
            if value is None:
                continue
            try:
                fraction = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid {key}: expected a finite positive fraction") from exc
            if not np.isfinite(fraction) or fraction <= 0 or (key == "ts_stop" and fraction >= 1):
                raise ValueError(f"invalid {key}: expected a finite positive fraction")
        if high is None or low is None:
            raise ValueError("stop_open requires high and low")
        if not stop_open.columns.equals(close.columns):
            if not isinstance(close.columns, pd.MultiIndex) or not stop_open.columns.is_unique:
                raise ValueError("stop prices must have the same index and columns as close")
            stop_columns = stop_open.columns.get_indexer(close.columns.get_level_values(-1))
            if (stop_columns < 0).any():
                raise ValueError("stop prices must have the same index and columns as close")
        for frame in (stop_open, high, low):
            if not frame.index.equals(close.index) or not frame.columns.equals(stop_open.columns):
                raise ValueError("stop prices must have the same index and columns as close")
    if td_stop is not None:
        warnings.warn(
            TIME_STOP_WARNING,
            UserWarning,
            stacklevel=2,
        )
        exits = exits | sig.time_stop(entries, td_stop)
    raw_close = close if stop_open is not None else None
    close = close.ffill()

    stops, n_dense = _stop_arguments(base_config, close, high, low)
    stops["stop_entry_price"] = stop_entry_price
    if stop_exit_price is not None:
        stops["stop_exit_price"] = stop_exit_price
    if stop_open is not None:
        stops.update(open=stop_open, high=high, low=low)
        stops.setdefault("stop_exit_price", "stopmarket")
        n_dense += 3
    if size is None:
        size = float(base_config.get("SIGNAL_POSITION_SIZE", DEFAULT_SIGNAL_POSITION_SIZE))
    if isinstance(size, pd.DataFrame):
        _check_shapes(close, size=size)
        n_dense += 1

    def simulate(sel: np.ndarray, group_codes: np.ndarray) -> Any:
        def pick(obj: Any) -> Any:
            return obj.iloc[:, sel] if isinstance(obj, pd.DataFrame) else obj

        stop_args = {k: pick(v) for k, v in stops.items() if k not in {"open", "high", "low"}}
        if stop_open is not None:
            selected = sel if stop_columns is None else stop_columns[sel]
            bars = [
                frame.iloc[:, selected].set_axis(close.columns[sel], axis=1)
                for frame in (stop_open, high, low)
            ]
            missing = _validate_stop_bars(raw_close.iloc[:, sel], entries.iloc[:, sel], *bars)
            for local in np.flatnonzero(missing.any(axis=0)):
                gaps[int(sel[local])] = np.flatnonzero(missing[:, local])
            stop_args.update(open=bars[0], high=bars[1], low=bars[2])

        return vbt.Portfolio.from_signals(
            close=close.iloc[:, sel],
            entries=entries.iloc[:, sel],
            exits=exits.iloc[:, sel],
            size=pick(size),
            size_type="percent",
            init_cash=_cfg_float(base_config, "START_CASH"),
            fees=_cfg_float(base_config, "FEES"),
            slippage=_cfg_float(base_config, "SLIPPAGE"),
            freq=base_config["FREQ"],
            cash_sharing=True,
            group_by=group_codes,
            call_seq=order_sequence or ("auto" if stop_open is not None else "default"),
            engine=engine_config.active_engine(),
            **stop_args,
        )

    result = _run_chunks(simulate, close, groups, all_groups, base_config, n_dense, max_bytes)
    if halt_positions:
        for col, entry, exit_ in result.trades[["col", "entry_idx", "exit_idx"]].itertuples(
            index=False, name=None
        ):
            dates = halt_positions.get(int(halt_columns[int(col)]))
            if dates is not None:
                at = int(np.searchsorted(dates, int(entry)))
                if at < len(dates) and dates[at] <= int(exit_):
                    raise ValueError(
                        f"halt during an open position at "
                        f"{close.index[dates[at]]}, {close.columns[int(col)]}"
                    )
    if gaps:
        for col, entry, exit_ in result.trades[["col", "entry_idx", "exit_idx"]].itertuples(
            index=False, name=None
        ):
            dates = gaps.get(int(col))
            if dates is not None:
                at = int(np.searchsorted(dates, int(entry)))
                if at < len(dates) and dates[at] <= int(exit_):
                    raise ValueError(
                        f"missing price bar during an open position at "
                        f"{close.index[dates[at]]}, {close.columns[int(col)]}"
                    )
    return result


def run_weights(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    base_config: Mapping[str, Any],
    *,
    groups: Sequence[Hashable] | None = None,
    all_groups: Sequence[Hashable] | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> SimResult:
    """Simulate target-percent weights, one cash pool per group, in one grouped call.

    Args:
        weights: (time x column) target weights; NaN means no order that bar, 0.0 exits.
        close: float (time x column) closes, same shape; forward-filled here.
        base_config: START_CASH, FEES, SLIPPAGE, FREQ.
        groups: group label per column (default one group ``"all"``).
        all_groups: labels that must appear in the output; those with no columns stay flat
            (an event strategy can legitimately hold nothing for a whole fold).
        max_bytes: chunking budget for the estimated peak memory.
    """
    _reject_unsupported(base_config, weights=True)
    _check_shapes(close, weights=weights)
    close = close.ffill()

    def simulate(sel: np.ndarray, group_codes: np.ndarray) -> Any:
        return vbt.Portfolio.from_orders(
            close=close.iloc[:, sel],
            size=weights.iloc[:, sel],
            size_type="targetpercent",
            init_cash=_cfg_float(base_config, "START_CASH"),
            fees=_cfg_float(base_config, "FEES"),
            slippage=_cfg_float(base_config, "SLIPPAGE"),
            freq=base_config["FREQ"],
            cash_sharing=True,
            group_by=group_codes,
            call_seq="auto",
            engine=engine_config.active_engine(),
        )

    return _run_chunks(simulate, close, groups, all_groups, base_config, 2, max_bytes)


def _reject_unsupported(base_config: Mapping[str, Any], *, weights: bool = False) -> None:
    """Raise on a config key the legacy simulator honoured and this one cannot apply."""
    keys = UNSUPPORTED_LEGACY_KEYS + (SIGNAL_ONLY_KEYS if weights else ())
    bad = [k for k in keys if base_config.get(k) is not None]
    if bad:
        where = "run_weights has no stops" if weights else "vol-target sizing is not ported"
        raise ValueError(
            f"base_config keys {bad} are not supported by the engine ({where}); "
            "legacy lab/simulate.py applied them, so ignoring them would change the result. "
            "Pass per-bar sizes through run_signals(size=...) instead."
        )


def _release_memory() -> None:
    """Collect cycles and ask glibc to return freed pages to the OS (no-op elsewhere)."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _as_bool(frame: pd.DataFrame) -> pd.DataFrame:
    """Bool frame with NaN meaning False (a plain fillna(False) warns on float dtypes)."""
    return frame.where(frame.notna(), False).astype(bool)


def _cfg_float(base_config: Mapping[str, Any], key: str) -> float:
    """A required numeric config key, with a clear error instead of a bare KeyError."""
    if key not in base_config:
        raise ValueError(f"base_config is missing {key!r}")
    return float(base_config[key])


def _check_shapes(close: pd.DataFrame, **frames: pd.DataFrame) -> None:
    """All frames must share close's index and columns (vectorbt uses positions)."""
    for name, frame in frames.items():
        if (
            frame.shape != close.shape
            or not frame.index.equals(close.index)
            or not frame.columns.equals(close.columns)
        ):
            raise ValueError(f"{name} must have the same index, columns and shape as close")


def _validate_stop_bars(
    close: pd.DataFrame,
    entries: pd.DataFrame,
    opening: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
) -> np.ndarray:
    """Validate one chunk and return its entirely missing price bars."""
    prices = [frame.to_numpy(dtype=float) for frame in (opening, high, low, close)]
    missing = np.logical_and.reduce([np.isnan(p) for p in prices])
    invalid = np.logical_or.reduce([(~np.isfinite(p) | (p <= 0)) & ~missing for p in prices])
    if invalid.any():
        row, col = np.argwhere(invalid)[0]
        raise ValueError(
            f"stop prices must be finite and positive at {close.index[row]}, {close.columns[col]}"
        )
    if (missing & entries.to_numpy()).any():
        raise ValueError("entry requested on a missing price bar")
    opening_values, highs, lows, closing = prices
    min_price, max_price = np.minimum(opening_values, closing), np.maximum(opening_values, closing)
    invalid = (lows > min_price * (1 + 1e-10)) | (highs < max_price * (1 - 1e-10))
    if invalid.any():
        row, col = np.argwhere(invalid)[0]
        raise ValueError(f"invalid OHLC prices at {close.index[row]}, {close.columns[col]}")
    return missing


def _stop_arguments(
    base_config: Mapping[str, Any],
    close: pd.DataFrame,
    high: pd.DataFrame | None,
    low: pd.DataFrame | None,
) -> tuple[dict[str, Any], int]:
    """Stop keyword arguments for ``from_signals`` and the number of dense per-column arrays."""
    stops: dict[str, Any] = {}
    n_dense = 1
    ts_stop = base_config.get("ts_stop")
    atr_mult = base_config.get("atr_mult")
    if ts_stop is not None:
        stops["sl_stop"] = float(ts_stop)
        stops["sl_trail"] = True
    elif atr_mult is not None:
        window = int(base_config.get("atr_period", DEFAULT_ATR_PERIOD))
        if window < 1:
            raise ValueError(f"atr_period must be >= 1, got {window}")
        if high is None or low is None:
            raise ValueError("high and low are required for the ATR trailing stop (atr_mult set)")
        _check_shapes(close, high=high, low=low)
        stops["sl_stop"] = atr_stop_fraction(high, low, close, window, float(atr_mult))
        stops["sl_trail"] = True
        n_dense += 1
    tp_stop = base_config.get("tp_stop")
    if tp_stop is not None:
        stops["tp_stop"] = float(tp_stop)
    return stops, n_dense


def _group_codes(
    n_cols: int, groups: Sequence[Hashable] | None
) -> tuple[np.ndarray, list[Hashable]]:
    """Integer code per column and the labels in first-appearance order."""
    labels = ["all"] * n_cols if groups is None else list(groups)
    if len(labels) != n_cols:
        raise ValueError(f"groups has {len(labels)} labels for {n_cols} columns")
    order: dict[Hashable, int] = {}
    for label in labels:
        order.setdefault(label, len(order))
    return np.array([order[label] for label in labels], dtype=np.int64), list(order)


def _plan_chunks(codes: np.ndarray, n_bars: int, n_dense: int, max_bytes: int) -> list[list[int]]:
    """Greedy chunks of whole groups (by code) whose estimated peak stays under the budget."""
    sizes = np.bincount(codes)
    chunks: list[list[int]] = []
    current: list[int] = []
    current_cols = 0
    for code, n_cols in enumerate(sizes):
        over = estimate_peak_bytes(n_bars, current_cols + int(n_cols), n_dense) > max_bytes
        if current and over:
            chunks.append(current)
            current, current_cols = [], 0
        current.append(code)
        current_cols += int(n_cols)
    if current:
        chunks.append(current)
    for chunk in chunks:
        cols = int(sum(sizes[c] for c in chunk))
        if len(chunk) == 1 and estimate_peak_bytes(n_bars, cols, n_dense) > max_bytes:
            warnings.warn(
                f"one group of {cols} columns is estimated above max_bytes={max_bytes}; "
                "it cannot be split because its columns share one cash pool",
                ResourceWarning,
                stacklevel=3,
            )
    return chunks


def _run_chunks(
    simulate: Any,
    close: pd.DataFrame,
    groups: Sequence[Hashable] | None,
    all_groups: Sequence[Hashable] | None,
    base_config: Mapping[str, Any],
    n_dense: int,
    max_bytes: int,
) -> SimResult:
    """Run ``simulate`` per chunk of whole groups and assemble one SimResult."""
    codes, labels = _group_codes(close.shape[1], groups)
    out_labels = list(labels)
    for label in all_groups or []:
        if label not in out_labels:
            out_labels.append(label)
    start_cash = _cfg_float(base_config, "START_CASH")
    symbols = _symbol_labels(close.columns)

    values: dict[int, np.ndarray] = {}
    trade_frames: list[pd.DataFrame] = []
    id_offset = 0
    chunks = _plan_chunks(codes, len(close), n_dense, max_bytes) if close.shape[1] else []
    for chunk in chunks:
        sel = np.flatnonzero(np.isin(codes, chunk))
        local = {code: i for i, code in enumerate(chunk)}
        local_codes = np.array([local[c] for c in codes[sel]], dtype=np.int64)
        pf = simulate(sel, local_codes)
        value = pf.value()
        if isinstance(value, pd.Series):
            value = value.to_frame(0)
        for code in chunk:
            # copy: a view would keep the whole chunk frame alive until the result is built
            values[code] = value.iloc[:, local[code]].to_numpy().copy()
        records = pf.trades.records.copy()
        if len(records):
            records["id"] += id_offset
            records["parent_id"] += id_offset
            id_offset = int(records["id"].max()) + 1
            records["col"] = sel[records["col"].to_numpy()]
            trade_frames.append(records)
        del pf, value, records
        _release_memory()

    flat = pd.Series(start_cash, index=close.index)
    equity = {}
    for label in out_labels:
        code = labels.index(label) if label in labels else None
        equity[label] = values[code] if code is not None else flat.to_numpy()
    value_frame = pd.DataFrame(equity, index=close.index)
    value_frame.columns = pd.Index(out_labels)
    returns = value_frame.pct_change(fill_method=None).fillna(0.0)

    trades = _assemble_trades(trade_frames, codes, labels, symbols)
    counts = trades.groupby("group", sort=False).size() if len(trades) else pd.Series(dtype="int64")
    n_trades = pd.Series({label: int(counts.get(label, 0)) for label in out_labels}, dtype="int64")
    return SimResult(
        value=value_frame,
        returns=returns,
        trades=trades,
        n_trades=n_trades,
        engine=engine_config.active_engine(),
        n_chunks=len(chunks),
    )


def _symbol_labels(columns: pd.Index) -> list[Hashable]:
    """Symbol label per column: the last level of a MultiIndex, else the label itself."""
    if isinstance(columns, pd.MultiIndex):
        return list(columns.get_level_values(-1))
    return list(columns)


def _assemble_trades(
    frames: list[pd.DataFrame],
    codes: np.ndarray,
    labels: list[Hashable],
    symbols: list[Hashable],
) -> pd.DataFrame:
    """Concatenate chunk trade records and attach group and symbol labels."""
    if not frames:
        return pd.DataFrame({name: pd.Series(dtype="float64") for name in TRADE_COLUMNS})
    trades = pd.concat(frames, ignore_index=True)
    trades["group"] = pd.Series([labels[codes[c]] for c in trades["col"]], dtype=object)
    trades["symbol"] = pd.Series([symbols[c] for c in trades["col"]], dtype=object)
    trades = trades.sort_values(["col", "id"], kind="stable").reset_index(drop=True)
    return trades[list(TRADE_COLUMNS)]
