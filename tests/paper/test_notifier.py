"""Tests for the Telegram notifier."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def _make_notifier(**env_overrides):
    env = {
        "TELEGRAM_BOT_TOKEN": "fake-token",
        "TELEGRAM_CHAT_ID": "12345",
        **env_overrides,
    }
    with patch.dict("os.environ", env):
        from ggTrader.paper.notifier import TelegramNotifier

        return TelegramNotifier()


class TestTelegramNotifierInit:
    def test_disabled_when_no_token(self):
        with patch.dict("os.environ", {}, clear=True):
            from ggTrader.paper.notifier import TelegramNotifier

            n = TelegramNotifier()
            assert n._enabled is False

    def test_enabled_when_configured(self):
        n = _make_notifier()
        assert n._enabled is True


class TestSend:
    def test_send_disabled_returns_false(self):
        with patch.dict("os.environ", {}, clear=True):
            from ggTrader.paper.notifier import TelegramNotifier

            n = TelegramNotifier()
            assert n.send("hello") is False

    @patch("ggTrader.paper.notifier.requests.post")
    def test_send_posts_to_telegram(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        n = _make_notifier()
        assert n.send("test message") is True
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "test message" in str(call_kwargs)

    @patch("ggTrader.paper.notifier.requests.post")
    def test_send_returns_false_on_http_error(self, mock_post):
        mock_post.return_value = MagicMock(status_code=500)
        n = _make_notifier()
        assert n.send("test") is False

    @patch("ggTrader.paper.notifier.requests.post")
    def test_send_returns_false_on_exception(self, mock_post):
        mock_post.side_effect = ConnectionError("network down")
        n = _make_notifier()
        assert n.send("test") is False


class TestTradeAlert:
    @patch("ggTrader.paper.notifier.requests.post")
    def test_trade_alert_formats_message(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        n = _make_notifier()
        result = n.trade_alert("BUY", "AAPL", 1000.0, "order-123")
        assert result is True
        body = mock_post.call_args[1]["json"]["text"]
        assert "BUY" in body
        assert "AAPL" in body
        assert "1000" in body


class TestDailySummary:
    @patch("ggTrader.paper.notifier.requests.post")
    def test_daily_summary_formats_message(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        n = _make_notifier()
        positions = {"AAPL": {"qty": 10.0, "unrealized_pl": 50.0}}
        result = n.daily_summary(102000.0, 500.0, positions)
        assert result is True
        body = mock_post.call_args[1]["json"]["text"]
        assert "102,000" in body or "102000" in body
        assert "AAPL" in body
