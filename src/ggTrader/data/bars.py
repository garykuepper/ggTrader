"""Read-only daily bars for research: per-field frames from the frozen tape or a read-only DB.

``load_bars`` is the single research entry point to daily OHLCV (spec
docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md, sections 5.1 and 5.7). It returns a
:class:`Bars` container: one ``dates x symbols`` frame per field plus ``missing_symbols`` and the
last valid bar of every symbol.

Guarantees
----------
* **Read-only, no network.** The default source is the hashed Parquet snapshot
  (:mod:`ggTrader.data.snapshot`); the ``db`` source opens a server-enforced read-only session
  and runs one SELECT. Nothing here fetches from a provider and nothing writes a row or runs DDL;
  a test scans this file for write verbs.
* **Hard assertions instead of repair.** Stored stamps must be unique per ``(timestamp, symbol)``,
  naive-UTC midnight and weekdays, otherwise :class:`BarsValidationError` is raised. This
  replaces the silent heuristic ``ggTrader.lab.data.collapse_daily_duplicates`` (which stays
  untouched for legacy callers) now that the tape has no such rows.
* **Date bounds are inclusive on both ends** (naive UTC dates). The legacy loader compared naive
  stamps with tz-aware bounds in the database session zone, which silently dropped the bar
  stamped on ``end``; that quirk is not reproduced. To match a legacy window ending on day D,
  pass :func:`legacy_exclusive_end` ``(D)`` (the previous weekday) as ``end``. Every
  legacy-parity run must do so, or its last fold gains one out-of-sample bar.

Output index is tz-aware UTC midnight (what the lab already uses), columns are sorted symbols.
:func:`to_legacy_frame` gives the ``(symbol, field)`` MultiIndex frame the not-yet-ported
strategies still consume.

Why not ``vbt.Data`` (spec 5.3, S6 section 2)
---------------------------------------------
Open-source ``vbt.Data`` is a dict-of-DataFrames holder: it has no ``.run``/``.close``/features
(those are PRO), its ``download`` loop has no skip policy for delisted symbols, and the vectorbt
import is confined to ``ggTrader.engine``. A 69-line subclass worked but only reproduced what a
frozen dataclass of frames gives, so it is not adopted. ``YFData``/``AlpacaData`` are rejected
for the reasons in S6 section 3 (tz-ambiguous stamps, serial fetch, crash on delisted symbols).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ggTrader.data import snapshot
from ggTrader.data.core.index_constituents import normalize_yf_ticker, universe_all_between

BAR_FIELDS = ("close", "open", "high", "low", "volume")
#: Field order of the legacy ``(symbol, field)`` frame (alphabetical, as ``sort_index`` gave).
LEGACY_FIELD_ORDER = ("close", "high", "low", "open", "volume")
SOURCES = ("snapshot", "db")
VENUE = "yfinance"
INTERVAL = "1d"
STATEMENT_TIMEOUT_MS = 900_000
#: Server-enforced: any write in this session is rejected by PostgreSQL itself.
READ_ONLY_OPTIONS = (
    f"-c default_transaction_read_only=on -c statement_timeout={STATEMENT_TIMEOUT_MS}"
)
SELECT_SQL = """
    SELECT timestamp, symbol, open, high, low, close, volume
    FROM ohlcv
    WHERE symbol = ANY(%s) AND interval = %s AND venue = %s
      AND timestamp >= %s AND timestamp <= %s
    ORDER BY symbol ASC, timestamp ASC
"""
_FAR_PAST = pd.Timestamp("1900-01-01")
_FAR_FUTURE = pd.Timestamp("2200-01-01")


class BarsValidationError(ValueError):
    """Stored bars violate the research-tape contract (duplicates, non-midnight, weekend)."""


@dataclass(frozen=True, eq=False)
class Bars:
    """Per-field ``dates x symbols`` frames plus what was asked for and what was found.

    Attributes:
        close, open, high, low, volume: float64 frames, tz-aware UTC midnight index, sorted
            symbol columns (only symbols with at least one bar in the window).
        requested: normalised, de-duplicated, sorted symbols that were asked for.
        missing_symbols: requested symbols with no bar in the window (sorted tuple).
        last_valid_bar: per loaded symbol, the date of its last non-NaN close.
        provenance: source description for run JSON (snapshot hash or ``db`` read-only flag).
    """

    close: pd.DataFrame
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    requested: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    last_valid_bar: pd.Series
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def symbols(self) -> list[str]:
        """Loaded symbols (columns of every field frame)."""
        return list(self.close.columns)

    def frame(self, name: str) -> pd.DataFrame:
        """The frame for a field name (``close``, ``open``, ``high``, ``low``, ``volume``)."""
        if name not in BAR_FIELDS:
            raise KeyError(f"unknown bar field {name!r}; expected one of {BAR_FIELDS}")
        return getattr(self, name)

    def legacy_frame(self) -> pd.DataFrame:
        """The ``(symbol, field)`` MultiIndex frame for strategies not yet ported."""
        return to_legacy_frame(self)

    def last_bar_coverage(self) -> float:
        """Share of loaded symbols with a valid close on the final index bar."""
        if self.close.empty:
            return 0.0
        return float(self.close.iloc[-1].notna().mean())

    def early_ending(self, min_bars: int = 5) -> list[str]:
        """Symbols whose last valid close is at least ``min_bars`` bars before the final bar."""
        if self.close.empty:
            return []
        last_pos = self.close.index.get_indexer(self.last_valid_bar.dropna())
        gap = (len(self.close) - 1) - last_pos
        names = self.last_valid_bar.dropna().index
        return sorted(str(s) for s, g in zip(names, gap) if g >= min_bars)

    def summary(self, early_min_bars: int = 5) -> dict[str, Any]:
        """JSON-able coverage facts that every run records (spec 5.7)."""
        early = self.early_ending(early_min_bars)
        return {
            "source": self.provenance.get("source"),
            "n_symbols": len(self.symbols),
            "n_bars": len(self.close),
            "first_bar": str(self.close.index[0].date()) if len(self.close) else None,
            "last_bar": str(self.close.index[-1].date()) if len(self.close) else None,
            "requested": len(self.requested),
            "missing_symbols": list(self.missing_symbols),
            "early_ending_min_bars": early_min_bars,
            "early_ending": {s: str(self.last_valid_bar[s].date()) for s in early},
            "last_bar_coverage": self.last_bar_coverage(),
        }


# --------------------------------------------------------------------------- validation


def validate_long(flat: pd.DataFrame) -> None:
    """Assert the stored-row contract; raises :class:`BarsValidationError` with examples."""
    if flat.empty:
        return
    stamps = pd.DatetimeIndex(flat["timestamp"])
    if stamps.tz is not None:
        raise BarsValidationError("stored timestamps must be naive UTC, found a tz-aware column")
    if stamps.isna().any():
        raise BarsValidationError("null timestamps in bars")
    problems: list[str] = []
    dup = flat.duplicated(["timestamp", "symbol"], keep=False)
    if dup.any():
        sample = flat.loc[dup, ["symbol", "timestamp"]].head(3).to_dict("records")
        problems.append(f"{int(dup.sum())} duplicate (timestamp, symbol) rows, e.g. {sample}")
    off = np.asarray(stamps != stamps.normalize())
    if off.any():
        sample = [str(t) for t in stamps[off][:3]]
        problems.append(f"{int(off.sum())} non-midnight stamps, e.g. {sample}")
    weekend = np.asarray(stamps.day_of_week >= 5)
    if weekend.any():
        sample = [str(t.date()) for t in stamps[weekend][:3]]
        problems.append(f"{int(weekend.sum())} weekend stamps, e.g. {sample}")
    if problems:
        raise BarsValidationError("bars violate the research-tape contract: " + "; ".join(problems))


# --------------------------------------------------------------------------- read paths


def connect_readonly(conn_str: Optional[str] = None) -> Any:
    """Open a server-enforced read-only psycopg2 session; refuses if the server disagrees."""
    import psycopg2

    if conn_str is None:
        from ggTrader.utils.config import get_db_connection_string

        conn_str = get_db_connection_string()
    conn_str = conn_str.replace("postgresql+psycopg2://", "postgresql://")
    conn = psycopg2.connect(conn_str, options=READ_ONLY_OPTIONS)
    conn.set_session(readonly=True, autocommit=True)
    with conn.cursor() as cur:
        cur.execute("show default_transaction_read_only")
        state = cur.fetchone()[0]
    if state != "on":
        conn.close()
        raise RuntimeError("database session is not read-only; refusing to continue")
    return conn


def _read_db_long(symbols: Sequence[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """One SELECT for ``symbols`` in [start, end] (naive-UTC bounds) on a read-only session."""
    conn = connect_readonly()
    try:
        with conn.cursor() as cur:
            cur.execute(SELECT_SQL, (list(symbols), INTERVAL, VENUE, start, end))
            rows = cur.fetchall()
    finally:
        conn.close()
    flat = pd.DataFrame(rows, columns=["timestamp", "symbol", *snapshot.FIELDS])
    flat["timestamp"] = pd.to_datetime(flat["timestamp"]).astype("datetime64[ns]")
    for col in snapshot.FIELDS:
        flat[col] = pd.to_numeric(flat[col]).astype("float64")
    return flat


def _naive_utc(value: Any, default: pd.Timestamp) -> pd.Timestamp:
    """A bound as a naive UTC timestamp (tz-aware inputs are converted, not shifted)."""
    if value is None:
        return default
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def legacy_exclusive_end(end: Any) -> pd.Timestamp:
    """The inclusive ``end`` that reproduces a legacy window ending (exclusively) on ``end``.

    The legacy loader dropped the bar stamped on ``--eval-end``; ``load_bars`` keeps it. This
    returns the previous weekday as a naive UTC date, so ``load_bars(end=legacy_exclusive_end(D))``
    has the last bar the legacy loader had (P0: ``--eval-end 2026-04-30`` ended OOS on 04-29).
    A holiday just means that day has no bar, which is the same under both conventions.

    Args:
        end: the legacy exclusive end (date string, naive or tz-aware timestamp; the time of
            day is dropped after conversion to UTC).
    """
    day = _naive_utc(end, _FAR_FUTURE).normalize() - pd.Timedelta(days=1)
    while day.weekday() >= 5:  # Saturday and Sunday carry no stamps
        day -= pd.Timedelta(days=1)
    return day


def _utc(value: Any) -> pd.Timestamp:
    """A bound as a tz-aware UTC timestamp (naive is read as UTC, aware is converted)."""
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _normalise_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({normalize_yf_ticker(s) for s in symbols}))


def load_bars(
    symbols: Optional[Sequence[str]] = None,
    start: Any = None,
    end: Any = None,
    *,
    universe: Optional[str] = None,
    universe_span: Optional[tuple[Any, Any]] = None,
    extra_symbols: Sequence[str] = (),
    source: str = "snapshot",
    snapshot_path: Optional[Any] = None,
    deep_verify: bool = False,
) -> Bars:
    """Load daily bars for research, read-only.

    Args:
        symbols: tickers to load (normalised to yfinance form, e.g. ``BRK.B`` to ``BRK-B``).
        start, end: inclusive date bounds; ``None`` leaves that side open.
        universe: instead of ``symbols``, a universe name (``sp500`` = point-in-time union over
            ``universe_span``; static universes use their snapshot list).
        universe_span: ``(start, end)`` over which the membership union is taken; required with
            ``universe`` (the bar window usually starts earlier, for indicator warm-up).
        extra_symbols: added to a ``universe`` request (benchmark and sleeve ETFs).
        source: ``snapshot`` (default, hash-verified Parquet) or ``db`` (read-only session).
        snapshot_path: snapshot file; default is :func:`snapshot.default_snapshot_path`.
        deep_verify: also recompute the snapshot content hash.

    Raises:
        BarsValidationError: duplicate, non-midnight or weekend stamps.
        snapshot.SnapshotIntegrityError: snapshot does not match its manifest.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    if universe is not None and symbols is not None:
        raise ValueError("pass symbols or universe, not both")
    if universe is None and symbols is None:
        raise ValueError("pass symbols or universe")

    if universe is not None:
        if universe_span is None:
            raise ValueError("universe_span=(start, end) is required with universe")
        span_start, span_end = _utc(universe_span[0]), _utc(universe_span[1])
        members = universe_all_between(universe, span_start, span_end)
        requested = _normalise_symbols([*members, *extra_symbols])
    else:
        requested = _normalise_symbols([*(symbols or ()), *extra_symbols])

    lo = _naive_utc(start, _FAR_PAST)
    hi = _naive_utc(end, _FAR_FUTURE)

    if source == "snapshot":
        flat, info = snapshot.load_snapshot_long(snapshot_path, requested, deep=deep_verify)
        provenance: dict[str, Any] = {"source": "snapshot", "snapshot": info.as_dict()}
    else:
        flat = _read_db_long(requested, lo, hi)
        provenance = {"source": "db", "read_only": True}

    validate_long(flat)
    stamps = flat["timestamp"]
    flat = flat[(stamps >= lo) & (stamps <= hi)]
    provenance = {
        **provenance,
        "start": None if start is None else str(lo),
        "end": None if end is None else str(hi),
    }

    if flat.empty:
        empty = pd.DataFrame(
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"), dtype="float64"
        )
        frames = {name: empty.copy() for name in BAR_FIELDS}
    else:
        wide = flat.set_index(["timestamp", "symbol"])[list(snapshot.FIELDS)].unstack("symbol")
        wide.index = pd.DatetimeIndex(wide.index).tz_localize("UTC")
        wide.index.name = "timestamp"
        frames = {}
        for name in BAR_FIELDS:
            frame = wide[name].astype("float64")
            frame.columns.name = "symbol"
            frames[name] = frame

    loaded = set(frames["close"].columns)
    missing = tuple(s for s in requested if s not in loaded)
    last_valid = _last_valid_bar(frames["close"])
    return Bars(
        requested=requested,
        missing_symbols=missing,
        last_valid_bar=last_valid,
        provenance=provenance,
        **frames,
    )


def _last_valid_bar(close: pd.DataFrame) -> pd.Series:
    """Date of each column's last non-NaN close (vectorised; NaT for an all-NaN column)."""
    if close.empty:
        return pd.Series([], dtype="datetime64[ns, UTC]", name="last_valid_bar")
    valid = close.notna().to_numpy()
    n = valid.shape[0]
    last_pos = n - 1 - valid[::-1].argmax(axis=0)
    has_any = valid.any(axis=0)
    stamps = close.index[last_pos].where(has_any, pd.NaT)
    return pd.Series(stamps, index=close.columns, name="last_valid_bar")


def to_legacy_frame(bars: Bars) -> pd.DataFrame:
    """``(symbol, field)`` MultiIndex frame, exactly the layout ``lab.data.load_ohlcv`` returns.

    Columns are sorted by symbol then field (close, high, low, open, volume); names are
    ``["symbol", "field"]``. A shim for strategies that have not been ported to per-field frames.
    """
    if bars.close.empty:
        return pd.DataFrame()
    parts = {name: bars.frame(name) for name in LEGACY_FIELD_ORDER}
    wide = pd.concat(parts, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)
    wide.columns.names = ["symbol", "field"]
    return wide.copy()
