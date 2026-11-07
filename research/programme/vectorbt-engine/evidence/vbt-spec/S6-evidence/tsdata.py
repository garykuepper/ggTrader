"""S6 prototype: vbt.Data subclass reading the existing TimescaleDB cache READ-ONLY.

No network, no writes: bypasses CachedYFinanceLoader.fetch_ohlcv (which fetches Yahoo
and upserts) and uses only its SELECT path (_db_fetch) over a read-only session.
"""
from __future__ import annotations

import time
import warnings

import pandas as pd
import psycopg2
import vectorbt as vbt

from ggTrader.data.live.cached_yfinance_loader import CachedYFinanceLoader

FIELDS = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}


class ReadOnlyLoader(CachedYFinanceLoader):
    """Same SELECT path as production, but the session is read-only (writes raise)."""

    def _connect(self):
        c = psycopg2.connect(self.connection_string)
        c.set_session(readonly=True)
        return c


class TimescaleData(vbt.Data):
    """vbt.Data whose symbols come from the ggTrader ohlcv cache (venue='yfinance').

    Index: tz-aware UTC midnight (matches lab convention). Columns: Open High Low Close
    Volume (vbt convention, so data.get('Close') works). Symbols with no rows are dropped
    and recorded in ``missing_symbols`` (not raised) -- the delisted-ticker policy.
    """

    @classmethod
    def download(cls, symbols, start=None, end=None, interval="1d", loader=None,
                 min_bars=1, **kwargs):
        loader = loader or ReadOnlyLoader()
        syms = sorted(set(symbols))
        s = None if start is None else pd.Timestamp(start, tz="UTC")
        e = None if end is None else pd.Timestamp(end, tz="UTC")
        wide = loader._db_fetch(syms, interval, s, e)  # (symbol, field) MultiIndex, tz-aware UTC
        if wide.empty:
            raise ValueError("no rows for requested symbols")
        have = list(wide.columns.get_level_values(0).unique())
        data = {}
        for sym in have:
            df = wide[sym].rename(columns=FIELDS)[list(FIELDS.values())]
            if df["Close"].notna().sum() >= min_bars:
                data[sym] = df
        missing = [x for x in syms if x not in data]
        # one shared index up-front so Data.align_index does not warn/union per symbol
        idx = wide.index
        data = {k: v.reindex(idx) for k, v in data.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = cls.from_data(data, missing_index="nan", missing_columns="nan",
                                 download_kwargs=dict(start=start, end=end, interval=interval, min_bars=min_bars), **kwargs)
        obj.missing_symbols = missing  # plain attribute: informational only
        return obj

    def concat(self, level_name="symbol"):
        """Vectorised override of Data.concat (stock one loops symbol x column with .loc)."""
        out = {}
        for c in self.data[self.symbols[0]].columns:
            out[c] = pd.concat({s: self.data[s][c] for s in self.symbols}, axis=1, names=[level_name])
        return out
