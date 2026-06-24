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
    @patch("ggTrader.paper.notifier._load_env")
    def test_disabled_when_no_token(self, _mock_load_env):
        # Patch _load_env so it can't repopulate the token from the real .env
        # file on disk, which would defeat the cleared environment below.
        with patch.dict("os.environ", {}, clear=True):
            from ggTrader.paper.notifier import TelegramNotifier

            n = TelegramNotifier()
            assert n._enabled is False

    def test_enabled_when_configured(self):
        n = _make_notifier()
        assert n._enabled is True


class TestSend:
    @patch("ggTrader.paper.notifier.requests.post")
    @patch("ggTrader.paper.notifier._load_env")
    def test_send_disabled_returns_false(self, _mock_load_env, mock_post):
        # _load_env patched so the cleared env actually disables the notifier;
        # requests.post patched as a hard safety net so this can never make a
        # real Telegram API call even if the disabled-guard regresses.
        with patch.dict("os.environ", {}, clear=True):
            from ggTrader.paper.notifier import TelegramNotifier

            n = TelegramNotifier()
            assert n.send("hello") is False
            mock_post.assert_not_called()

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
