"""Build conservative daily research masks from supplied halt intervals."""

import pandas as pd


def _utc(value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid halt or session timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError("Halt and session timestamps need explicit timezones")
    return timestamp.tz_convert("UTC")


def daily_halt_mask(sessions: pd.DataFrame, symbols: list[str], halts: list[dict]) -> pd.DataFrame:
    """Flag sessions touched by halt intervals, conservatively including the close.

    US cash sessions need explicit timezone-aware opens and closes, and a UTC
    midnight index labelled with the New York session date. The index is retained
    for alignment with daily bars. Halts are dictionaries with symbol, start and
    end; end=None means no resumption has been established and stays flagged to
    the end of the supplied calendar. A resumption at the open clears that session;
    a halt at the close flags it. This is a whole-session diagnostic, not a fill
    model. The caller must establish source coverage, identity, availability and
    the market hours relevant to its strategy; false does not prove tradability.
    A retrospective mask may audit a run but must not select securities using
    events that were not known at the decision time.
    """
    if (
        not isinstance(sessions, pd.DataFrame)
        or sessions.empty
        or sessions.columns.has_duplicates
        or not {"open", "close"}.issubset(sessions.columns)
        or not isinstance(sessions.index, pd.DatetimeIndex)
        or str(sessions.index.tz) != "UTC"
        or sessions.index.has_duplicates
        or not sessions.index.is_monotonic_increasing
        or not sessions.index.equals(sessions.index.normalize())
    ):
        raise ValueError("Invalid sessions")
    if (
        not isinstance(symbols, list)
        or not symbols
        or any(not isinstance(s, str) or not s or s != s.strip() for s in symbols)
        or len(set(symbols)) != len(symbols)
        or not isinstance(halts, list)
    ):
        raise ValueError("Invalid symbols or halts")
    opening = pd.DatetimeIndex([_utc(value) for value in sessions["open"]])
    closing = pd.DatetimeIndex([_utc(value) for value in sessions["close"]])
    if (
        (opening >= closing).any()
        or (closing[:-1] > opening[1:]).any()
        or list(opening.tz_convert("America/New_York").date) != list(sessions.index.date)
        or list(closing.tz_convert("America/New_York").date) != list(sessions.index.date)
    ):
        raise ValueError("Invalid or overlapping sessions")
    mask = pd.DataFrame(False, index=sessions.index, columns=symbols)
    for halt in halts:
        if not isinstance(halt, dict) or not {"symbol", "start", "end"}.issubset(halt):
            raise ValueError("Invalid halt interval")
        symbol = halt["symbol"]
        if not isinstance(symbol, str) or symbol not in symbols:
            raise ValueError("Unknown halt symbol")
        start = _utc(halt["start"])
        end = None if halt["end"] is None else _utc(halt["end"])
        if end is not None and end <= start:
            raise ValueError("Invalid halt interval")
        affected = closing >= start
        if end is not None:
            affected &= opening < end
        mask.loc[affected, symbol] = True
    return mask
