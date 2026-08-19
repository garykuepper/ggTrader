"""Tests for the Alpaca paper trading broker adapter."""

from __future__ import annotations

from datetime import date
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
        mock_acct.multiplier = "1"
        broker._client.get_account.return_value = mock_acct
        result = broker.get_account()
        assert result == {
            "cash": 50000.0,
            "portfolio_value": 52000.0,
            "buying_power": 100000.0,
            "multiplier": 1.0,
        }

    def test_includes_multiplier(self):
        broker = _make_broker()
        mock_acct = MagicMock()
        mock_acct.cash = "50000.00"
        mock_acct.portfolio_value = "52000.00"
        mock_acct.buying_power = "100000.00"
        mock_acct.multiplier = "1"
        broker._client.get_account.return_value = mock_acct
        result = broker.get_account()
        assert result["multiplier"] == 1.0


class TestGetPositions:
    def test_returns_position_dict(self):
        broker = _make_broker()
        mock_pos = MagicMock()
        mock_pos.symbol = "AAPL"
        mock_pos.qty = "10"
        mock_pos.market_value = "1500.00"
        mock_pos.current_price = "150.00"
        mock_pos.avg_entry_price = "145.00"
        mock_pos.unrealized_pl = "50.00"
        mock_pos.unrealized_plpc = "0.0345"
        mock_pos.change_today = "0.012"
        mock_pos.cost_basis = "1450.00"
        broker._client.get_all_positions.return_value = [mock_pos]
        result = broker.get_positions()
        assert result == {
            "AAPL": {
                "qty": 10.0,
                "market_value": 1500.0,
                "current_price": 150.0,
                "avg_entry": 145.0,
                "unrealized_pl": 50.0,
                "unrealized_plpc": 0.0345,
                "change_today": 0.012,
                "cost_basis": 1450.0,
            }
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


def _forward_split_action(symbol="MNST", old_rate=1.0, new_rate=2.0, ex_date=date(2026, 8, 11)):
    action = MagicMock()
    action.symbol = symbol
    action.old_rate = old_rate
    action.new_rate = new_rate
    action.ex_date = ex_date
    return action


class TestGetSplitCorrections:
    @patch("ggTrader.paper.alpaca_broker.CorporateActionsClient")
    def test_unapplied_forward_split_is_flagged(self, mock_ca_cls):
        broker = _make_broker()
        mock_ca_cls.return_value.get_corporate_actions.return_value.data = {
            "forward_splits": [_forward_split_action()],
        }
        broker._client.get.return_value = []  # no matching SPLIT activity

        result = broker.get_split_corrections(["MNST"], since=date(2026, 8, 1))

        assert result == {"MNST": 2.0}

    @patch("ggTrader.paper.alpaca_broker.CorporateActionsClient")
    def test_split_with_matching_activity_not_flagged(self, mock_ca_cls):
        broker = _make_broker()
        mock_ca_cls.return_value.get_corporate_actions.return_value.data = {
            "forward_splits": [_forward_split_action()],
        }
        broker._client.get.return_value = [{"symbol": "MNST", "date": "2026-08-11"}]

        result = broker.get_split_corrections(["MNST"], since=date(2026, 8, 1))

        assert result == {}

    @patch("ggTrader.paper.alpaca_broker.CorporateActionsClient")
    def test_reverse_split_factor_below_one(self, mock_ca_cls):
        broker = _make_broker()
        action = _forward_split_action(symbol="XYZ", old_rate=10.0, new_rate=1.0)
        mock_ca_cls.return_value.get_corporate_actions.return_value.data = {
            "reverse_splits": [action],
        }
        broker._client.get.return_value = []

        result = broker.get_split_corrections(["XYZ"], since=date(2026, 8, 1))

        assert result == {"XYZ": 0.1}

    @patch("ggTrader.paper.alpaca_broker.CorporateActionsClient")
    def test_no_known_splits_returns_empty(self, mock_ca_cls):
        broker = _make_broker()
        mock_ca_cls.return_value.get_corporate_actions.return_value.data = {}

        result = broker.get_split_corrections(["VZ"], since=date(2026, 8, 1))

        assert result == {}
        # No account SPLIT activity is unnecessary when nothing is corp-flagged.
        broker._client.get.assert_not_called()

    def test_empty_symbol_list_returns_empty_without_api_call(self):
        broker = _make_broker()

        result = broker.get_split_corrections([], since=date(2026, 8, 1))

        assert result == {}
        broker._client.get.assert_not_called()

    @patch("ggTrader.paper.alpaca_broker.CorporateActionsClient")
    def test_corporate_actions_api_failure_fails_soft(self, mock_ca_cls):
        broker = _make_broker()
        mock_ca_cls.return_value.get_corporate_actions.side_effect = Exception("API down")

        result = broker.get_split_corrections(["MNST"], since=date(2026, 8, 1))

        assert result == {}

    @patch("ggTrader.paper.alpaca_broker.CorporateActionsClient")
    def test_activities_api_failure_fails_soft(self, mock_ca_cls):
        broker = _make_broker()
        mock_ca_cls.return_value.get_corporate_actions.return_value.data = {
            "forward_splits": [_forward_split_action()],
        }
        broker._client.get.side_effect = Exception("API down")

        result = broker.get_split_corrections(["MNST"], since=date(2026, 8, 1))

        assert result == {}
