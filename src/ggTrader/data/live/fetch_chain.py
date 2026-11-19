"""The equity price-loader fallback chain: DB cache, then yfinance, then Tiingo for the gaps."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from ggTrader.data.core.index_constituents import normalize_yf_ticker


class OhlcvProvider(Protocol):
    """Anything that fetches a (symbol, field) OHLCV frame for a window."""

    def fetch_ohlcv(
        self,
        symbols: list[str],
        interval: str,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return the bars for ``symbols`` (empty when the provider has none)."""
        ...


class CacheWriter(Protocol):
    """Anything that can save a fetched frame so the next run finds it."""

    def persist(self, df: pd.DataFrame, interval: str) -> None:
        """Write ``df`` to the cache."""
        ...


class CacheProvider(CacheWriter, Protocol):
    """The DB cache: a writer whose fetch also takes a ``limit``."""

    def fetch_ohlcv(
        self,
        symbols: list[str],
        interval: str,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Return the cached bars, refreshing stale or absent symbols itself."""
        ...


class NegativeCache(Protocol):
    """Symbols known to return no data from any provider, kept for a time window."""

    def load_skip(self, interval: str) -> set[str]:
        """The symbols to skip for ``interval``."""
        ...

    def record(self, symbols: list[str], interval: str) -> None:
        """Remember ``symbols`` as no-data for ``interval``."""
        ...


@dataclass(frozen=True)
class GapFiller:
    """One provider that fills the symbols the earlier steps left out.

    Attributes:
        make: Builds the provider. Called inside the guard when ``swallow_errors`` is set.
        intro: Progress line with an ``{n}`` placeholder for the number of symbols asked for.
        announce_before_make: True prints ``intro`` before ``make`` runs (yfinance), False
            prints it after (Tiingo).
        persist_error: Label after "failed to persist " when saving the fetched frame fails.
        swallow_errors: None lets every error propagate; otherwise the label printed after
            "[data] " when the whole block (construction, fetch, persist, merge) raises.
    """

    make: Callable[[], OhlcvProvider]
    intro: str
    announce_before_make: bool
    persist_error: str
    swallow_errors: str | None


def _symbols(df: pd.DataFrame) -> set[str]:
    """The distinct symbols of a (symbol, field) frame."""
    return set(df.columns.get_level_values(0).unique())


class OhlcvFetchChain:
    """Fetch daily OHLCV through a cache, a primary provider and ordered gap fillers."""

    def __init__(
        self,
        *,
        primary: Callable[[], OhlcvProvider],
        fillers: Sequence[GapFiller],
        cache: Callable[[], CacheProvider] | None = None,
        negative_cache: NegativeCache | None = None,
    ) -> None:
        """Wire the chain.

        Args:
            primary: Builds the provider asked for every ticker when the cache left nothing.
                Its instance is also the first filler's provider.
            fillers: Gap fillers tried in order, each for the symbols still missing.
            cache: Builds the DB cache, tried first; None skips it and nothing is persisted.
            negative_cache: Optional no-data list; never touched when None.
        """
        self._primary = primary
        self._fillers = tuple(fillers)
        self._cache = cache
        self._negative_cache = negative_cache

    def fetch(
        self,
        symbols: list[str],
        start: str,
        end: str | None = None,
        interval: str = "1d",
        min_coverage: float = 0.0,
    ) -> pd.DataFrame:
        """Return OHLCV for ``symbols`` as a (symbol, field) frame trimmed to the window.

        Raises:
            ValueError: When the cache has nothing and the primary provider returns nothing.
        """
        tickers = sorted({normalize_yf_ticker(s) for s in symbols})
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") if end else None

        skip = self._load_skip(interval)

        writer: CacheWriter | None = None
        df = pd.DataFrame()
        if self._cache is not None:
            try:
                cache = self._cache()
                writer = cache
                df = cache.fetch_ohlcv(
                    tickers, interval, start_date=start_ts, end_date=end_ts, limit=None
                )
            except Exception as exc:
                print(f"  [data] DB cache unavailable ({exc!r}); falling back to yfinance only")
                writer = None

        primary: OhlcvProvider | None = None
        if df.empty:
            primary = self._primary()
            df = primary.fetch_ohlcv(tickers, interval, start_date=start_ts, end_date=end_ts)
            if df.empty:
                raise ValueError("yfinance returned no data for the requested universe")

        for index, filler in enumerate(self._fillers):
            missing = [t for t in tickers if t not in _symbols(df) and t not in skip]
            if not missing:
                continue
            reuse = primary if index == 0 else None
            if filler.swallow_errors is None:
                df = self._fill(df, filler, missing, reuse, writer, interval, start_ts, end_ts)
                continue
            try:
                df = self._fill(df, filler, missing, reuse, writer, interval, start_ts, end_ts)
            except Exception as exc:
                print(f"  [data] {filler.swallow_errors}: {exc!r}")

        self._record_no_data(df, tickers, skip, interval)

        df = df[df.index >= start_ts]
        if end_ts is not None:
            df = df[df.index <= end_ts]

        if min_coverage > 0.0:
            keep = [
                sym
                for sym in df.columns.get_level_values(0).unique()
                if df[sym]["close"].notna().mean() >= min_coverage
            ]
            df = df[keep]

        n_syms = len(df.columns.get_level_values(0).unique())
        print(f"  [data] {len(df)} rows x {n_syms} symbols ({interval})")
        df.columns.names = ["symbol", "field"]
        return df

    def _load_skip(self, interval: str) -> set[str]:
        """The no-data symbols to leave out, or an empty set when unavailable."""
        if self._negative_cache is None:
            return set()
        try:
            skip = self._negative_cache.load_skip(interval)
            if skip:
                print(f"  [data] negative cache: skipping {len(skip)} known no-data symbols")
            return skip
        except Exception as exc:
            print(f"  [data] negative cache unavailable ({exc!r}); fetching all")
            return set()

    def _fill(
        self,
        df: pd.DataFrame,
        filler: GapFiller,
        missing: list[str],
        reuse: OhlcvProvider | None,
        writer: CacheWriter | None,
        interval: str,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp | None,
    ) -> pd.DataFrame:
        """Run one gap filler for ``missing`` and return ``df`` with what it supplied."""
        intro = filler.intro.format(n=len(missing))
        if filler.announce_before_make:
            print(intro)
        provider = reuse if reuse is not None else filler.make()
        if not filler.announce_before_make:
            print(intro)
        extra = provider.fetch_ohlcv(missing, interval, start_date=start_ts, end_date=end_ts)
        if not extra.empty:
            if writer is not None:
                try:
                    writer.persist(extra, interval)
                except Exception as exc:
                    print(f"  [data] failed to persist {filler.persist_error}: {exc!r}")
            df = pd.concat([df, extra], axis=1)
            df.sort_index(axis=1, inplace=True)
        return df

    def _record_no_data(
        self, df: pd.DataFrame, tickers: list[str], skip: set[str], interval: str
    ) -> None:
        """Remember symbols no provider supplied so later opt-in runs skip them."""
        if self._negative_cache is None:
            return
        have = _symbols(df)
        newly_dead = [t for t in tickers if t not in have and t not in skip]
        if newly_dead:
            try:
                self._negative_cache.record(newly_dead, interval)
                print(f"  [data] negative cache: recorded {len(newly_dead)} no-data symbols")
            except Exception as exc:
                print(f"  [data] negative cache write failed ({exc!r})")


def _make_cache() -> CacheProvider:
    """Build the DB-first loader, resolving the class when called so patches on the module hold."""
    from ggTrader.data.live import cached_yfinance_loader as module

    return module.CachedYFinanceLoader()


def _make_yfinance() -> OhlcvProvider:
    """Build the plain yfinance loader, resolving the class when called."""
    from ggTrader.data.live import yfinance_loader as module

    return module.YFinanceDataLoader()


def _make_tiingo() -> OhlcvProvider:
    """Build the Tiingo loader, resolving the class when called."""
    from ggTrader.data.live import tiingo_loader as module

    return module.TiingoDataLoader()


def default_chain(
    use_db_cache: bool = True, negative_cache: NegativeCache | None = None
) -> OhlcvFetchChain:
    """The production chain: DB cache, yfinance, then yfinance gap and Tiingo gap fillers."""
    return OhlcvFetchChain(
        primary=_make_yfinance,
        fillers=[
            GapFiller(
                make=_make_yfinance,
                intro="  [data] fetching {n} symbols missing from cache...",
                announce_before_make=True,
                persist_error="gap fetch",
                swallow_errors=None,
            ),
            GapFiller(
                make=_make_tiingo,
                intro="  [data] trying Tiingo for {n} symbols yfinance missed...",
                announce_before_make=False,
                persist_error="Tiingo data",
                swallow_errors="Tiingo fallback unavailable",
            ),
        ],
        cache=_make_cache if use_db_cache else None,
        negative_cache=negative_cache,
    )


def fetch_stock_ohlcv(
    symbols: list[str],
    start: str,
    end: str | None = None,
    interval: str = "1d",
    use_db_cache: bool = True,
    min_coverage: float = 0.0,
    negative_cache: NegativeCache | None = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV for ``symbols`` as a (symbol, field) MultiIndex frame.

    DB-first via the cached loader when reachable, else plain yfinance; symbols absent from the
    result are fetched for the full range (yfinance, then Tiingo) and persisted. Symbols the
    optional ``negative_cache`` lists are skipped, and ones no provider supplies are recorded.
    """
    return default_chain(use_db_cache, negative_cache).fetch(
        symbols, start, end, interval=interval, min_coverage=min_coverage
    )
