"""Unit tests for the Trading class."""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from ggTrader.core.trading import Trading
from ggTrader.core.portfolio import Portfolio
from ggTrader.core.position import Position


@pytest.fixture
def dummy_ohlcv():
    idx = pd.date_range("2023-01-01", periods=10, freq="1d").tz_localize("UTC")
    data = {
        ("BTC", "open"): [100.0] * 10,
        ("BTC", "high"): [110.0] * 10,
        ("BTC", "low"): [90.0] * 10,
        ("BTC", "close"): [105.0] * 10,
    }
    df = pd.DataFrame(data, index=idx)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    return df


@pytest.fixture
def mock_trading(dummy_ohlcv):
    with patch("ggTrader.core.trading.Screener") as mock_screener:
        t = Trading(
            ohlcv_df=dummy_ohlcv, date_range=dummy_ohlcv.index, start_cash=10000, use_movers=False
        )
        yield t, mock_screener.return_value


def test_trading_init(mock_trading, dummy_ohlcv):
    t, _ = mock_trading
    assert t.portfolio.cash == 10000
    assert t.current_date == dummy_ohlcv.index[0]


def test_prepare_simulation_data_no_movers(mock_trading, dummy_ohlcv):
    t, _ = mock_trading
    with patch("ggTrader.indicators.signals.Signals.calc_signals") as mock_signals:
        mock_signals.return_value = (
            pd.DataFrame({"BTC": [False] * 10}, index=dummy_ohlcv.index),
            pd.DataFrame({"BTC": [False] * 10}, index=dummy_ohlcv.index),
            pd.DataFrame({"BTC": [90.0] * 10}, index=dummy_ohlcv.index),
            pd.DataFrame({"BTC": [105.0] * 10}, index=dummy_ohlcv.index),
        )
        t.prepare_simulation_data()

    assert "BTC" in t.all_movers_per_day[dummy_ohlcv.index[0]]
    assert "BTC" in t.precalculated_signals


def test_check_buy_signal(mock_trading, dummy_ohlcv):
    t, _ = mock_trading
    sig_df = pd.DataFrame({"signal": [1] * 10, "close": [105.0] * 10}, index=dummy_ohlcv.index)
    t.signals_dict = {"BTC": sig_df}
    t.daily_movers = pd.DataFrame({"symbol": ["BTC"]})
    t.current_date = dummy_ohlcv.index[0]

    t.check_buy()

    assert len(t.portfolio.positions) == 1
    assert t.portfolio.positions[0].symbol == "BTC"


def test_check_sell_stop_loss(mock_trading, dummy_ohlcv):
    t, _ = mock_trading
    # Initialize Portfolio and Position correctly
    t.portfolio = Portfolio(cash=10000)
    pos = Position("BTC", 1.0, 105.0, t.current_date)
    pos.stop_loss = 110.0  # Force SL condition: current_price(105) <= stop_loss(110)
    t.portfolio.add_position(pos)

    # Ensure current_date is first index
    t.current_date = dummy_ohlcv.index[0]

    # Mock signals dict with expected structure
    sig_df = pd.DataFrame(
        {"signal": [0] * 10, "stop_loss": [110.0] * 10, "close": [105.0] * 10},
        index=dummy_ohlcv.index,
    )
    t.signals_dict = {"BTC": sig_df}

    t.check_sell()

    # Position should be closed
    assert len(t.portfolio.positions) == 0
    assert len(t.portfolio.trades) == 1
    assert t.portfolio.trades[0].status == "closed"
