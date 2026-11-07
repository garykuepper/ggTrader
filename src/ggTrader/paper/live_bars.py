"""Turn the trader's live OHLCV frame into the engine's ``Bars`` container.

The research reader :mod:`ggTrader.data.bars` serves a frozen tape or a read-only database
session. It cannot serve the trader's case: at 12:45 PT the newest bar is a *partial* intraday
bar that only the live loader chain (``lab.data.fetch_stock_ohlcv`` -> TimescaleDB cache ->
yfinance gap fetch) can supply. That one step therefore stays on the live loader (spec 8.2,
5.7); this module only reshapes its ``(symbol, field)`` MultiIndex frame into ``Bars`` so the
validated ``Ensemble.build()`` runs on exactly the data the loader returned.

Nothing here fetches or writes. Validation is left to :mod:`ggTrader.paper.gates`, so a bad tape
is reported by the trader's sanity gate instead of being repaired silently.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ggTrader.data.bars import BAR_FIELDS, Bars

SYMBOL_LEVEL = "symbol"


def bars_from_legacy_frame(ohlcv: pd.DataFrame, requested: Sequence[str] = ()) -> Bars:
    """Build ``Bars`` from a ``(symbol, field)`` MultiIndex OHLCV frame.

    The index is normalised to tz-aware UTC midnight (the engine contract). Duplicate days are
    **kept**, not collapsed, so the sanity gate can see and refuse them.

    Args:
        ohlcv: frame as returned by ``lab.data.fetch_stock_ohlcv``.
        requested: symbols that were asked for (for ``missing_symbols``); defaults to the
            symbols present.

    Returns:
        Bars with float64 frames and sorted symbol columns.

    Raises:
        ValueError: the frame is not a two-level ``(symbol, field)`` frame or lacks a field.
    """
    asked = tuple(sorted(set(requested)))
    if ohlcv.empty:
        empty = pd.DataFrame(
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"), dtype="float64"
        )
        return Bars(
            close=empty.copy(),
            open=empty.copy(),
            high=empty.copy(),
            low=empty.copy(),
            volume=empty.copy(),
            requested=asked,
            missing_symbols=asked,
            last_valid_bar=pd.Series([], dtype="datetime64[ns, UTC]", name="last_valid_bar"),
            provenance={"source": "live_loader"},
        )
    if not isinstance(ohlcv.columns, pd.MultiIndex) or ohlcv.columns.nlevels != 2:
        raise ValueError("expected a (symbol, field) MultiIndex frame")
    fields = set(ohlcv.columns.get_level_values(1))
    missing_fields = [f for f in BAR_FIELDS if f not in fields]
    if missing_fields:
        raise ValueError(f"live frame lacks fields {missing_fields}")

    index = pd.DatetimeIndex(ohlcv.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    index = index.normalize()
    index.name = "timestamp"

    symbols = sorted(ohlcv.columns.get_level_values(0).unique())
    frames: dict[str, pd.DataFrame] = {}
    for name in BAR_FIELDS:
        frame = ohlcv.xs(name, axis=1, level=1).reindex(columns=symbols).astype("float64")
        frame.index = index
        frame.columns.name = SYMBOL_LEVEL
        frames[name] = frame

    asked = asked or tuple(symbols)
    close = frames["close"]
    last_valid = close.apply(lambda col: col.last_valid_index())
    last_valid.name = "last_valid_bar"
    return Bars(
        requested=asked,
        missing_symbols=tuple(s for s in asked if s not in set(symbols)),
        last_valid_bar=last_valid,
        provenance={"source": "live_loader"},
        **frames,
    )
