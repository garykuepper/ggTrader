"""Tests for paper trading persistence layer."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


@patch("ggTrader.paper.persist._get_engine")
class TestInitSchema:
    def test_creates_tables(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        from ggTrader.paper.persist import init_paper_schema

        init_paper_schema()

        executed_sql = " ".join(
            str(call[0][0]) for call in mock_conn.execute.call_args_list
        )
        assert "paper_trades" in executed_sql
        assert "paper_snapshots" in executed_sql


@patch("ggTrader.paper.persist._get_engine")
class TestLogTrade:
    def test_inserts_trade_row(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        from ggTrader.paper.persist import log_trade

        log_trade("2026-06-20", "BUY", "AAPL", 2000.0, "order-123")

        mock_conn.execute.assert_called_once()
        sql_str = str(mock_conn.execute.call_args[0][0])
        assert "paper_trades" in sql_str


@patch("ggTrader.paper.persist._get_engine")
class TestLogSnapshot:
    def test_inserts_snapshot_row(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        from ggTrader.paper.persist import log_snapshot

        log_snapshot("2026-06-20", 102000.0, 50000.0, {"AAPL": {"qty": 10}})

        mock_conn.execute.assert_called_once()
        sql_str = str(mock_conn.execute.call_args[0][0])
        assert "paper_snapshots" in sql_str
