"""Unit tests for the Screener class."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from ggTrader.core.screener import Screener


@pytest.fixture
def mock_screener():
    with patch("ggTrader.core.screener.KrakenData") as mock_data, patch(
        "ggTrader.core.screener.KrakenHistoricalData"
    ) as mock_hdata:
        screener = Screener()
        yield screener, mock_data.return_value, mock_hdata.return_value


def test_get_daily_top_kraken_by_volume(mock_screener):
    screener, m_data, _ = mock_screener
    m_data.top_kraken_by_volume.return_value = pd.DataFrame(
        {"symbol": ["BTC/USD"], "volume": [1000]}
    )

    res = screener.get_daily_top_kraken_by_volume(top_n=5)

    assert len(res) == 1
    assert "BTC/USD" in res["symbol"].values
    m_data.top_kraken_by_volume.assert_called_with(
        limit=5, only_usd=True, exclude_stables=True, verbose=False
    )


def test_get_historical_daily_kraken_by_volume(mock_screener):
    screener, _, m_hdata = mock_screener
    m_hdata.get_historical_movers_by_day.return_value = pd.DataFrame({"symbol": ["ETH/USD"]})

    date = pd.Timestamp("2024-01-01")
    res = screener.get_historical_daily_kraken_by_volume(date, top_n=10)

    assert len(res) == 1
    m_hdata.get_historical_movers_by_day.assert_called_with(date, top_n=10)
