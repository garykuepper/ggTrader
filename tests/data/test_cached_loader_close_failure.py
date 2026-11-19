"""Check that cache connection close failures are logged."""

import logging
from unittest.mock import MagicMock

import pandas as pd

from ggTrader.data.live.cached_yfinance_loader import CachedYFinanceLoader


def test_connection_close_failure_is_logged_at_debug(monkeypatch, caplog) -> None:
    loader = CachedYFinanceLoader(connection_string="postgresql://test")
    conn = MagicMock()
    conn.close.side_effect = RuntimeError("close failed")
    monkeypatch.setattr(loader, "_connect", lambda: conn)
    writes = []
    monkeypatch.setattr(
        "ggTrader.data.live.cached_yfinance_loader.execute_values",
        lambda *args, **kwargs: writes.append(args),
    )
    df = pd.DataFrame(
        [[1, 2, 1, 2, 100]],
        index=pd.to_datetime(["2024-01-02"]),
        columns=pd.MultiIndex.from_product([["SPY"], ["open", "high", "low", "close", "volume"]]),
    )
    with caplog.at_level(logging.DEBUG, logger=loader.logger.name):
        loader._cache_to_db(df, "1d")
    assert writes and conn.close.called
    assert any("RuntimeError" in record.getMessage() for record in caplog.records)
