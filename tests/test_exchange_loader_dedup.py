"""Regression test: live exchange loader dedupes per-symbol Kraken duplicates.

Kraken's OHLC endpoint occasionally returns the same timestamp twice (typically
the in-progress partial bar repeated). Without per-symbol deduping the
horizontal `pd.concat(axis=1)` raises `InvalidIndexError`.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from ggTrader.data.live.exchange_loader import LiveExchangeLoader


def test_concat_survives_duplicate_kraken_bar():
    """Two symbols, only-overlapping indexes, second has a duplicate trailing bar.

    Without the per-symbol dedupe, `pd.concat(axis=1)` raises
    InvalidIndexError because pandas needs to reindex the duplicate-containing
    frame to the union of timestamps.
    """
    loader = LiveExchangeLoader.__new__(LiveExchangeLoader)
    loader.exchange = MagicMock()
    loader._markets_cached = True
    now_ms = int(time.time() * 1000)
    h4 = 4 * 60 * 60 * 1000

    # BTC: bars at t-8h, t-4h, t  (clean)
    btc = [
        [now_ms - 2 * h4, 1.0, 1.1, 0.9, 1.05, 100.0],
        [now_ms - h4, 1.05, 1.2, 1.0, 1.15, 120.0],
        [now_ms, 1.15, 1.25, 1.1, 1.20, 80.0],
    ]
    # ETH: bars at t-12h, t-4h, t, t (last duplicated — Kraken partial-bar quirk)
    eth = [
        [now_ms - 3 * h4, 2.0, 2.1, 1.9, 2.05, 50.0],
        [now_ms - h4, 2.05, 2.2, 2.0, 2.15, 60.0],
        [now_ms, 2.15, 2.25, 2.1, 2.20, 40.0],
        [now_ms, 2.15, 2.25, 2.1, 2.21, 41.0],  # <- dup ts, "keep last" wins
    ]
    loader.exchange.fetch_ohlcv.side_effect = [btc, eth]

    df = loader.fetch_ohlcv(symbols=["BTC-USD", "ETH-USD"], interval="4h", limit=200)

    assert not df.empty
    assert df.index.is_unique, "concat output must have a unique row index"
    assert ("ETH-USD", "close") in df.columns
    # "keep last" → 2.21 wins on the duplicated bar.
    eth_last = df[("ETH-USD", "close")].dropna().iloc[-1]
    assert eth_last == 2.21
