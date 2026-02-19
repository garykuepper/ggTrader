"""Unit tests for Kraken data adapters."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from ggTrader.data.kraken.historical_data import KrakenHistoricalData
from ggTrader.data.kraken.postgres_reader import KrakenPostgresReader


@pytest.fixture
def mock_reader():
    # Patch create_engine inside the module
    with patch("ggTrader.data.kraken.postgres_reader.create_engine"):
        reader = KrakenPostgresReader("postgresql://mock")
        yield reader


def test_kraken_postgres_reader_read_ohlcv(mock_reader):
    # Mock pd.read_sql inside the module
    with patch("ggTrader.data.kraken.postgres_reader.pd.read_sql") as mock_read_sql:
        mock_read_sql.return_value = pd.DataFrame(
            {
                "timestamp": ["2023-01-01"],
                "symbol": ["BTC-USD"],
                "interval": ["4h"],
                "open": [100.0],
                "high": [110.0],
                "low": [90.0],
                "close": [105.0],
                "volume": [1.0],
                "trades": [10],
            }
        )

        df = mock_reader.read_ohlcv(symbol="BTC", interval="4h")

        assert not df.empty
        # Column names should be exactly as expected by the code
        assert "symbol" in df.columns
        assert df["symbol"].iloc[0] == "BTC"  # Quote stripped by reader logic


def test_kraken_historical_data_get_ohlcv_df():
    # Patch all heavy dependencies in historical_data.py
    with patch(
        "ggTrader.data.kraken.historical_data.get_db_connection_string",
        return_value="postgresql://mock",
    ), patch("ggTrader.data.kraken.historical_data.KrakenPostgresIngestor"), patch(
        "ggTrader.data.kraken.historical_data.KrakenPostgresReader"
    ) as mock_reader_cls:

        instance = mock_reader_cls.return_value
        idx = pd.date_range("2023-01-01", periods=1, freq="1d").tz_localize("UTC")
        mock_df = pd.DataFrame({("BTC", "close"): [105.0]}, index=idx)
        mock_df.columns = pd.MultiIndex.from_tuples(mock_df.columns, names=["symbol", "field"])
        instance.get_ohlcv_df.return_value = mock_df

        hdata = KrakenHistoricalData()
        df = hdata.get_ohlcv_df(["BTC"], interval="1d")

        assert isinstance(df.columns, pd.MultiIndex)
        assert "BTC" in df.columns.get_level_values("symbol")
        instance.get_ohlcv_df.assert_called_once()
