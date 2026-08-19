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

        executed_sql = " ".join(str(call[0][0]) for call in mock_conn.execute.call_args_list)
        assert "paper_trades" in executed_sql
        assert "paper_snapshots" in executed_sql
        assert "paper_rebalance_state" in executed_sql


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


@patch("ggTrader.paper.persist._get_engine")
class TestGetLatestSnapshot:
    def test_returns_value_when_snapshot_exists(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = (102000.0,)

        from ggTrader.paper.persist import get_latest_snapshot

        assert get_latest_snapshot() == 102000.0

    def test_returns_none_when_no_snapshots(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = None

        from ggTrader.paper.persist import get_latest_snapshot

        assert get_latest_snapshot() is None


@patch("ggTrader.paper.persist._get_engine")
class TestGetLatestSnapshotPositions:
    def test_returns_positions_when_snapshot_exists(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = ({"AAPL": {"qty": 10.0}},)

        from ggTrader.paper.persist import get_latest_snapshot_positions

        assert get_latest_snapshot_positions() == {"AAPL": {"qty": 10.0}}

    def test_returns_none_when_no_snapshots(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = None

        from ggTrader.paper.persist import get_latest_snapshot_positions

        assert get_latest_snapshot_positions() is None


@patch("ggTrader.paper.persist._get_engine")
class TestRebalanceState:
    def test_save_rebalance_state_inserts_row(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        from ggTrader.paper.persist import save_rebalance_state

        save_rebalance_state("2026-07-01", {"sp500": 0.5, "midcap400": 0.3, "nasdaq100": 0.2}, 0.87)

        mock_conn.execute.assert_called_once()
        sql_str = str(mock_conn.execute.call_args[0][0])
        assert "paper_rebalance_state" in sql_str
        params = mock_conn.execute.call_args[0][1]
        assert params["rd"] == "2026-07-01"
        assert params["s"] == 0.87

    def test_get_rebalance_state_returns_parsed_row(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = (
            "2026-07-01",
            {"sp500": 0.5, "midcap400": 0.3, "nasdaq100": 0.2},
            0.87,
        )

        from ggTrader.paper.persist import get_rebalance_state

        state = get_rebalance_state()

        assert state["rebalance_date"] == "2026-07-01"
        assert state["weights"] == {"sp500": 0.5, "midcap400": 0.3, "nasdaq100": 0.2}
        assert state["scale"] == 0.87

    def test_get_rebalance_state_returns_none_when_empty(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = None

        from ggTrader.paper.persist import get_rebalance_state

        assert get_rebalance_state() is None


@patch("ggTrader.paper.persist._get_engine")
class TestPeakValue:
    def test_save_peak_value_upserts_row(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        from ggTrader.paper.persist import save_peak_value

        save_peak_value(120000.0)

        mock_conn.execute.assert_called_once()
        sql_str = str(mock_conn.execute.call_args[0][0])
        assert "paper_risk_state" in sql_str
        params = mock_conn.execute.call_args[0][1]
        assert params["peak"] == 120000.0

    def test_get_peak_value_returns_value(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = (120000.0,)

        from ggTrader.paper.persist import get_peak_value

        assert get_peak_value() == 120000.0

    def test_get_peak_value_returns_none_when_unset(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = None

        from ggTrader.paper.persist import get_peak_value

        assert get_peak_value() is None


@patch("ggTrader.paper.persist._get_engine")
class TestPendingOrderStaleness:
    def test_get_pending_orders_selects_created_at_and_flagged_stale(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.mappings.return_value = []

        from ggTrader.paper.persist import get_pending_orders

        get_pending_orders()

        sql_str = str(mock_conn.execute.call_args[0][0])
        assert "created_at" in sql_str
        assert "flagged_stale" in sql_str

    def test_mark_pending_order_stale_updates_row(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        from ggTrader.paper.persist import mark_pending_order_stale

        mark_pending_order_stale("order-123")

        mock_conn.execute.assert_called_once()
        sql_str = str(mock_conn.execute.call_args[0][0])
        assert "flagged_stale = TRUE" in sql_str
        params = mock_conn.execute.call_args[0][1]
        assert params["order_id"] == "order-123"


@patch("ggTrader.paper.persist._get_engine")
class TestGetLatestSnapshotRunDate:
    def test_returns_run_date_when_snapshot_exists(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = ("2026-06-18",)

        from ggTrader.paper.persist import get_latest_snapshot_run_date

        assert get_latest_snapshot_run_date() == "2026-06-18"

    def test_returns_none_when_no_snapshots(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = None

        from ggTrader.paper.persist import get_latest_snapshot_run_date

        assert get_latest_snapshot_run_date() is None
