"""Tests for the Alpaca paper trading broker adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_broker():
    """Create an AlpacaBroker with mocked credentials."""
    with patch.dict(
        "os.environ",
        {
            "APCA_API_KEY_ID": "test-key",
            "APCA_API_SECRET_KEY": "test-secret",
        },
    ):
        from ggTrader.paper.alpaca_broker import AlpacaBroker

        with patch("ggTrader.paper.alpaca_broker.TradingClient"):
            return AlpacaBroker()


class TestAlpacaBrokerInit:
    @patch("ggTrader.paper.alpaca_broker._load_env")
    def test_requires_api_keys(self, _mock_load_env):
        # Patch _load_env so it can't repopulate keys from the real .env file
        # on disk, which would defeat the cleared environment below.
        from ggTrader.paper.alpaca_broker import AlpacaBroker

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="APCA_API_KEY_ID"):
                AlpacaBroker()

    def test_creates_paper_client(self):
        with patch.dict(
            "os.environ",
            {
                "APCA_API_KEY_ID": "test-key",
                "APCA_API_SECRET_KEY": "test-secret",
            },
        ):
            with patch("ggTrader.paper.alpaca_broker.TradingClient") as mock_tc:
                from ggTrader.paper.alpaca_broker import AlpacaBroker

                AlpacaBroker()
                mock_tc.assert_called_once_with("test-key", "test-secret", paper=True)


class TestGetAccount:
    def test_returns_account_dict(self):
        broker = _make_broker()
        mock_acct = MagicMock()
        mock_acct.cash = "50000.00"
        mock_acct.portfolio_value = "52000.00"
        mock_acct.buying_power = "100000.00"
        broker._client.get_account.return_value = mock_acct
        result = broker.get_account()
        assert result == {"cash": 50000.0, "portfolio_value": 52000.0, "buying_power": 100000.0}


class TestGetPositions:
    def test_returns_position_dict(self):
        broker = _make_broker()
        mock_pos = MagicMock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = "10"
        mock_pos.market_value = "1500.00"
        mock_pos.avg_entry_price = "145.00"
        mock_pos.unrealized_pl = "50.00"
        broker._client.get_all_positions.return_value = [mock_pos]
        result = broker.get_positions()
        assert result == {
            "AAPL": {"qty": 10.0, "market_value": 1500.0, "avg_entry": 145.0, "unrealized_pl": 50.0}
        }

    def test_empty_positions(self):
        broker = _make_broker()
        broker._client.get_all_positions.return_value = []
        assert broker.get_positions() == {}


class TestSubmitOrders:
    def test_submit_buy_notional(self):
        broker = _make_broker()
        mock_order = MagicMock()
        mock_order.id = "order-123"
        broker._client.submit_order.return_value = mock_order
        oid = broker.submit_buy("AAPL", notional=1000.0)
        assert oid == "order-123"
        call_kwargs = broker._client.submit_order.call_args
        req = call_kwargs[0][0] if call_kwargs[0] else call_kwargs[1].get("order_data")
        assert req.symbol == "AAPL"

    def test_submit_sell_quantity(self):
        broker = _make_broker()
        mock_order = MagicMock()
        mock_order.id = "order-456"
        broker._client.submit_order.return_value = mock_order
        oid = broker.submit_sell("AAPL", qty=10.0)
        assert oid == "order-456"


class TestGetClock:
    def test_returns_clock_dict(self):
        broker = _make_broker()
        mock_clock = MagicMock()
        mock_clock.is_open = False
        mock_clock.next_open = "2026-06-22T09:30:00-04:00"
        mock_clock.next_close = "2026-06-22T16:00:00-04:00"
        broker._client.get_clock.return_value = mock_clock
        result = broker.get_clock()
        assert result == {
            "is_open": False,
            "next_open": "2026-06-22T09:30:00-04:00",
            "next_close": "2026-06-22T16:00:00-04:00",
        }
