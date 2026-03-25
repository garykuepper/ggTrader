"""Unit tests for TimescaleDB Loader."""

from unittest.mock import MagicMock, patch

import pandas as pd

from ggTrader.data.historical.timescaledb_loader import TimescaleDBLoader


def test_timescaledb_loader_fetch_ohlcv():
    with patch("ggTrader.data.historical.timescaledb_loader.create_engine") as mock_create_engine:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Mock pd.read_sql
        with patch("ggTrader.data.historical.timescaledb_loader.pd.read_sql") as mock_read_sql:
            mock_read_sql.return_value = pd.DataFrame(
                {
                    "datetime": [pd.Timestamp("2023-01-01", tz="UTC")],
                    "symbol": ["BTC-USD"],
                    "open": [100.0],
                    "high": [110.0],
                    "low": [90.0],
                    "close": [105.0],
                    "volume": [1.0],
                    "trades": [10],
                }
            )

            loader = TimescaleDBLoader("postgresql://mock")
            df = loader.fetch_ohlcv(symbols=["BTC-USD"], interval="4h")

            assert not df.empty
            # Output should be pivoted to a MultiIndex: symbol -> metric
            assert isinstance(df.columns, pd.MultiIndex)
            assert "BTC-USD" in df.columns.get_level_values(0)
            assert "close" in df.columns.get_level_values(1)


def test_timescaledb_loader_list_symbols():
    with patch("ggTrader.data.historical.timescaledb_loader.create_engine") as mock_create_engine:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        mock_conn = MagicMock()
        # Mock what engine.connect() context manager returns
        mock_engine.connect.return_value.__enter__.return_value = mock_conn

        # SQLAlchemy execute returns rows
        mock_conn.execute.return_value = [("BTC-USD",), ("ETH-EUR",), ("SOL-USD",)]

        loader = TimescaleDBLoader("mock")
        symbols = loader.list_symbols(quote="USD")

        assert "BTC-USD" in symbols
        assert "SOL-USD" in symbols
        assert "ETH-EUR" not in symbols
