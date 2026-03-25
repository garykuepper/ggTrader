"""Unit tests for the Screener class."""

from unittest.mock import patch

import pandas as pd
import pytest

from ggTrader.core.screener import Screener


@pytest.fixture
def mock_screener():
    with patch("ggTrader.core.screener.LiveExchangeLoader") as mock_data, patch(
        "ggTrader.core.screener.TimescaleDBLoader"
    ) as mock_hdata:
        screener = Screener()
        yield screener, mock_data.return_value, mock_hdata.return_value


def test_get_daily_top_kraken_by_volume(mock_screener):
    screener, m_data, _ = mock_screener
    m_data.get_top_by_volume.return_value = pd.DataFrame({"symbol": ["BTC/USD"], "volume": [1000]})

    res = screener.get_daily_top_kraken_by_volume(top_n=5)

    assert len(res) == 1
    assert "BTC/USD" in res["symbol"].values
    m_data.get_top_by_volume.assert_called_with(
        limit=5, quote="USD", exclude_stables=True, verbose=False
    )


def test_get_historical_daily_kraken_by_volume(mock_screener):
    screener, _, m_hdata = mock_screener

    date = pd.Timestamp("2024-01-01")
    # Screener extracts the symbols from mask.columns[mask.loc[date]]
    mock_mask = pd.DataFrame({"ETH/USD": [True]}, index=[date])
    m_hdata.get_daily_mover_mask.return_value = mock_mask

    res = screener.get_historical_daily_kraken_by_volume(date, top_n=10)

    assert len(res) == 1
    assert res["symbol"].iloc[0] == "ETH/USD"
    m_hdata.get_daily_mover_mask.assert_called_with(start=date, end=date, top_n=10)
