"""Inputs of the vectorbt-engine walk-forward: bars, point-in-time masks and grids.

Legacy callers hand ``run_wfo`` a ``(symbol, field)`` OHLCV frame and a ``universe_fn``; new
callers hand it :class:`~ggTrader.data.bars.Bars` and a precomputed boolean mask. Both are
accepted and normalised here.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import pandas as pd

from ggTrader.data.bars import BAR_FIELDS, Bars
from ggTrader.data.core.index_constituents import snapshot_members
from ggTrader.data.membership import MembershipSource
from ggTrader.lab import universe as universe_mod
from ggTrader.lab.strategies.contract import ParamGrid

UniverseFn = Callable[[pd.Timestamp, pd.DataFrame], List[str]]


def as_bars(data: Bars | pd.DataFrame) -> Bars:
    """A :class:`Bars` from a ``Bars`` or a legacy ``(symbol, field)`` MultiIndex frame.

    Fields missing from a legacy frame become all-NaN frames; ``close`` is required.

    Raises:
        ValueError: no ``close`` field, or a tz-naive index (folds are tz-aware UTC).
    """
    if isinstance(data, Bars):
        return data
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("ohlcv must be a Bars or a (symbol, field) MultiIndex frame")
    index = pd.DatetimeIndex(data.index)
    if index.tz is None:
        raise ValueError("ohlcv index must be tz-aware (UTC): the folds are tz-aware dates")
    fields = set(data.columns.get_level_values(-1))
    if "close" not in fields:
        raise ValueError("ohlcv has no 'close' field")
    symbols = sorted(data.columns.get_level_values(0).unique())
    frames: Dict[str, pd.DataFrame] = {}
    for name in BAR_FIELDS:
        if name in fields:
            frame = data.xs(name, axis=1, level=-1)[symbols].astype("float64")
        else:
            frame = pd.DataFrame(float("nan"), index=data.index, columns=symbols)
        frame.columns.name = "symbol"
        frames[name] = frame.copy()
    close = frames["close"]
    last_valid = close.apply(lambda s: s.last_valid_index())
    return Bars(
        requested=tuple(symbols),
        missing_symbols=(),
        last_valid_bar=last_valid,
        provenance={"source": "legacy_frame"},
        **frames,
    )


def subset_bars(bars: Bars, symbols: Sequence[str]) -> Bars:
    """Bars restricted to ``symbols`` (kept in the original sorted column order)."""
    keep = [s for s in bars.symbols if s in set(symbols)]
    frames = {name: bars.frame(name)[keep] for name in BAR_FIELDS}
    return Bars(
        requested=tuple(keep),
        missing_symbols=(),
        last_valid_bar=bars.last_valid_bar.reindex(keep),
        provenance=dict(bars.provenance),
        **frames,
    )


def universe_mask(
    close: pd.DataFrame,
    universe: str,
    *,
    min_history_bars: int,
    max_stocks: int | None = None,
    membership: MembershipSource | None = None,
) -> pd.DataFrame:
    """Daily eligibility mask equal to ``eligible_at(asof, past, cfg, universe)`` on every bar.

    ``sp500`` is point-in-time and reads ``membership`` (``None``: the source selected by
    ``GGT_MEMBERSHIP``, else an error); every other universe is a static snapshot list (so a bar
    earlier than the snapshot still counts the symbol, as the legacy call did).
    """
    if universe == "sp500":
        return universe_mod.pit_mask(
            close,
            membership=membership,
            min_history_bars=min_history_bars,
            max_stocks=max_stocks,
            on_stale="ignore",
        )
    return universe_mod.pit_mask(
        close,
        min_history_bars=min_history_bars,
        max_stocks=max_stocks,
        static_members=snapshot_members(universe),
    )


def mask_from_universe_fn(
    legacy_frame: pd.DataFrame, symbols: Sequence[str], universe_fn: UniverseFn
) -> pd.DataFrame:
    """Compatibility path: a mask from a legacy ``universe_fn`` (one call per bar, O(T^2)).

    Only for callers that cannot pass a vectorised mask; the member sets are exactly what the
    legacy ``pit_entry_mask`` built. Slow on a long, wide frame.
    """
    rows = {d: set(universe_fn(d, legacy_frame.loc[:d])) for d in legacy_frame.index}
    return pd.DataFrame(
        [[s in rows[d] for s in symbols] for d in legacy_frame.index],
        index=legacy_frame.index,
        columns=list(symbols),
        dtype=bool,
    )


def resolve_grid(
    strategy_cls: Any, grid: ParamGrid | Sequence[Dict[str, Any]]
) -> tuple[ParamGrid, List[Dict[str, Any]]]:
    """The strategy's :class:`ParamGrid` and its combinations as plain dicts (legacy order).

    A list of combination dicts (the legacy ``build_grid`` output) is accepted when it equals
    the combinations of a grid over the same values with the strategy's own validity rules.
    Combinations are evaluated, and ties broken, in the legacy order (the product over sorted
    parameter names), whatever order the list had.

    Raises:
        ValueError: the list is not a Cartesian product the strategy's rules reproduce.
    """
    if isinstance(grid, ParamGrid):
        return grid, grid.combos()
    combos = [dict(c) for c in grid]
    if not combos:
        raise ValueError("grid is empty")
    keys = sorted(combos[0])
    values = {k: tuple(dict.fromkeys(c[k] for c in combos)) for k in keys}
    rules = tuple(getattr(getattr(strategy_cls, "grid", None), "rules", ()))
    candidate = ParamGrid(values, rules)
    rebuilt = candidate.combos()
    if sorted(map(_key, rebuilt)) != sorted(map(_key, combos)):
        raise ValueError(
            "grid is not the Cartesian product of its values under the strategy's validity "
            "rules; pass a ParamGrid instead"
        )
    return candidate, rebuilt


def _key(combo: Dict[str, Any]) -> tuple:
    return tuple(sorted((k, repr(v)) for k, v in combo.items()))


def combo_name(strategy_name: str, params: Dict[str, Any]) -> str:
    """Deterministic label from strategy name + sorted param key-value pairs (legacy format)."""
    parts = [f"{k}{v}" for k, v in sorted(params.items())]
    return strategy_name + "__" + "_".join(parts)
