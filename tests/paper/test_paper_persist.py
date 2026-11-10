"""Tests for paper trading persistence layer."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest


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

        mock_conn.execute.return_value.all.return_value = []
        from ggTrader.paper.persist import log_trade

        log_trade("2026-06-20", "BUY", "AAPL", 2000.0, "order-123")

        assert "INSERT INTO paper_trades" in str(mock_conn.execute.call_args[0][0])


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
class TestInitSchemaIncludesDividendAccruals:
    def test_creates_dividend_accrual_table(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        from ggTrader.paper.persist import init_paper_schema

        init_paper_schema()

        executed_sql = " ".join(str(call[0][0]) for call in mock_conn.execute.call_args_list)
        assert "paper_dividend_accruals" in executed_sql


@patch("ggTrader.paper.persist._get_engine")
class TestGetSnapshotHistory:
    def test_returns_ascending_run_date_positions_pairs(self, mock_engine):
        from datetime import date

        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = [
            (date(2026, 5, 20), {"VZ": {"qty": 50.0}}),
            (date(2026, 6, 1), {"VZ": {"qty": 77.05}}),
        ]

        from ggTrader.paper.persist import get_snapshot_history

        result = get_snapshot_history()

        assert result == [
            (date(2026, 5, 20), {"VZ": {"qty": 50.0}}),
            (date(2026, 6, 1), {"VZ": {"qty": 77.05}}),
        ]

    def test_parses_json_string_positions(self, mock_engine):
        from datetime import date

        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = [
            (date(2026, 5, 20), '{"VZ": {"qty": 50.0}}'),
        ]

        from ggTrader.paper.persist import get_snapshot_history

        result = get_snapshot_history()

        assert result == [(date(2026, 5, 20), {"VZ": {"qty": 50.0}})]

    def test_empty_history(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = []

        from ggTrader.paper.persist import get_snapshot_history

        assert get_snapshot_history() == []


@patch("ggTrader.paper.persist._get_engine")
class TestDividendAccruals:
    def test_log_dividend_accrual_inserts_row(self, mock_engine):
        from datetime import date

        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        from ggTrader.paper.persist import log_dividend_accrual

        log_dividend_accrual("VZ", date(2026, 8, 1), 0.6725, 77.05, 51.82)

        mock_conn.execute.assert_called_once()
        sql_str = str(mock_conn.execute.call_args[0][0])
        assert "paper_dividend_accruals" in sql_str
        assert "DO NOTHING" in sql_str
        params = mock_conn.execute.call_args[0][1]
        assert params["symbol"] == "VZ"
        assert params["ex_date"] == date(2026, 8, 1)
        assert params["amount"] == 51.82

    def test_get_accrued_dividend_keys_returns_set_of_pairs(self, mock_engine):
        from datetime import date

        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = [
            ("VZ", date(2026, 8, 1)),
            ("AEE", date(2026, 7, 1)),
        ]

        from ggTrader.paper.persist import get_accrued_dividend_keys

        result = get_accrued_dividend_keys()

        assert result == {("VZ", date(2026, 8, 1)), ("AEE", date(2026, 7, 1))}

    def test_get_accrued_dividend_keys_empty(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = []

        from ggTrader.paper.persist import get_accrued_dividend_keys

        assert get_accrued_dividend_keys() == set()

    def test_get_total_dividend_accrual_returns_sum(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = (151.37,)

        from ggTrader.paper.persist import get_total_dividend_accrual

        assert get_total_dividend_accrual() == 151.37

    def test_get_total_dividend_accrual_zero_when_no_rows(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.first.return_value = (0,)

        from ggTrader.paper.persist import get_total_dividend_accrual

        assert get_total_dividend_accrual() == 0.0


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


@patch("ggTrader.paper.persist._get_engine")
class TestGetTradeHistoryDates:
    """Feeds `split_check.find_split_applied_symbols` its evidence for
    ruling out a plain trade explaining a qty change across a split's
    ex-date."""

    def test_returns_ascending_dates_for_symbol(self, mock_engine):
        from datetime import date

        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = [
            (date(2026, 8, 5),),
            (date(2026, 8, 20),),
        ]

        from ggTrader.paper.persist import get_trade_history_dates

        result = get_trade_history_dates("MNST")

        assert result == [date(2026, 8, 5), date(2026, 8, 20)]

    def test_empty_history(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = []

        from ggTrader.paper.persist import get_trade_history_dates

        assert get_trade_history_dates("MNST") == []

    def test_filters_by_symbol(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = []

        from ggTrader.paper.persist import get_trade_history_dates

        get_trade_history_dates("MNST")

        params = mock_conn.execute.call_args[0][1]
        assert params == {"symbol": "MNST"}


@patch("ggTrader.paper.persist._get_engine")
class TestSplitState:
    def test_creates_split_state_table(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        from ggTrader.paper.persist import init_paper_schema

        init_paper_schema()

        executed_sql = " ".join(str(call[0][0]) for call in mock_conn.execute.call_args_list)
        assert "paper_split_state" in executed_sql

    def test_get_open_split_corrections_returns_persisted_rows(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = [("MNST", 2.0), ("XYZ", 0.5)]

        from ggTrader.paper.persist import get_open_split_corrections

        result = get_open_split_corrections()

        assert result == {"MNST": 2.0, "XYZ": 0.5}

    def test_get_open_split_corrections_empty_when_no_rows(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = []

        from ggTrader.paper.persist import get_open_split_corrections

        assert get_open_split_corrections() == {}

    def test_get_open_split_states_returns_factor_and_ex_date(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = [
            ("MNST", 2.0, date(2026, 8, 11)),
            ("XYZ", 0.5, date(2026, 7, 1)),
        ]

        from ggTrader.paper.persist import get_open_split_states

        result = get_open_split_states()

        assert result == {
            "MNST": {"factor": 2.0, "ex_date": date(2026, 8, 11)},
            "XYZ": {"factor": 0.5, "ex_date": date(2026, 7, 1)},
        }

    def test_get_open_split_states_empty_when_no_rows(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.all.return_value = []

        from ggTrader.paper.persist import get_open_split_states

        assert get_open_split_states() == {}

    def test_save_split_correction_upserts(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        from ggTrader.paper.persist import save_split_correction

        save_split_correction("MNST", "2026-08-11", 2.0)

        mock_conn.execute.assert_called_once()
        sql_str = str(mock_conn.execute.call_args[0][0])
        assert "INSERT INTO paper_split_state" in sql_str
        assert "ON CONFLICT (symbol)" in sql_str
        params = mock_conn.execute.call_args[0][1]
        assert params == {"symbol": "MNST", "ex_date": "2026-08-11", "factor": 2.0}
        mock_conn.commit.assert_called_once()

    def test_clear_split_correction_deletes_row(self, mock_engine):
        mock_conn = MagicMock()
        mock_engine.return_value.connect.return_value.__enter__ = lambda s: mock_conn
        mock_engine.return_value.connect.return_value.__exit__ = MagicMock(return_value=False)

        from ggTrader.paper.persist import clear_split_correction

        clear_split_correction("MNST")

        mock_conn.execute.assert_called_once()
        sql_str = str(mock_conn.execute.call_args[0][0])
        assert "DELETE FROM paper_split_state" in sql_str
        assert mock_conn.execute.call_args[0][1] == {"symbol": "MNST"}
        mock_conn.commit.assert_called_once()


def test_settlement_rejects_invalid_fill_before_database_access():
    import pytest

    from ggTrader.paper.persist import settle_order_intent

    for qty, price, status in [
        (0, None, "filled"),
        (-1, 2, "canceled"),
        (float("nan"), 2, "canceled"),
        (1, None, "filled"),
        (1, float("inf"), "filled"),
        (1, 0, "filled"),
    ]:
        with patch("ggTrader.paper.persist._get_engine") as engine:
            with pytest.raises(ValueError):
                settle_order_intent("client", "broker", qty, price, status)
            engine.assert_not_called()


def test_order_intents_with_disposable_postgres(monkeypatch, request):
    """Exercise commits, rollback, replay, and lock ownership on an isolated cluster."""
    import os
    import uuid

    import pytest
    from sqlalchemy import create_engine, event, text
    from sqlalchemy.engine import make_url

    from ggTrader.paper import persist

    dsn = os.environ.get("GGT_TASK2_TEST_DSN")
    if not dsn:
        pytest.skip("GGT_TASK2_TEST_DSN not set for dedicated disposable database")
    url = make_url(dsn)
    if (
        url.get_backend_name() != "postgresql"
        or url.database != "ggtrader_task2_test"
        or url.username != "ggtrader_task2_test"
        or url.host not in ("localhost", "127.0.0.1")
        or url.query
    ):
        raise ValueError("Refusing unexpected Task 2 test database identity")
    admin = create_engine(url)
    with admin.connect() as conn:
        identity = conn.execute(text("SELECT current_database(), current_user")).one()
        if tuple(identity) != ("ggtrader_task2_test", "ggtrader_task2_test"):
            raise ValueError("Refusing unexpected Task 2 server identity")
    schema = "ggt_task2_" + uuid.uuid4().hex
    with admin.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))

    def cleanup():
        engine.dispose()
        with admin.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()

    request.addfinalizer(cleanup)
    engine = create_engine(
        url,
        connect_args={
            "options": f"-csearch_path={schema} -clock_timeout=3000 -cstatement_timeout=5000"
        },
    )
    monkeypatch.setattr(persist, "_get_engine", lambda: engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE paper_trades (id SERIAL PRIMARY KEY, run_date DATE NOT NULL, "
                "side TEXT NOT NULL, symbol TEXT NOT NULL, amount DOUBLE PRECISION NOT NULL, "
                "order_id TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
        )
        conn.execute(
            text(
                "INSERT INTO paper_trades (run_date, side, symbol, amount, order_id) "
                "VALUES ('2026-01-01', 'BUY', 'OLD', 1, 'old')"
            )
        )
    persist.init_paper_schema()
    persist.log_pending_order("2026-01-02", "BUY", "ABC", 24, "legacy", "test")

    def fail_delete(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("DELETE FROM paper_pending_orders"):
            raise RuntimeError("injected delete failure")

    event.listen(engine, "before_cursor_execute", fail_delete)
    with pytest.raises(RuntimeError, match="injected"):
        persist.settle_legacy_pending_order("legacy", 2, 12, "filled")
    event.remove(engine, "before_cursor_execute", fail_delete)
    with engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT count(*) FROM paper_trades WHERE order_id = 'legacy'")
            ).scalar_one()
            == 0
        )
        assert (
            conn.execute(
                text("SELECT count(*) FROM paper_pending_orders WHERE order_id = 'legacy'")
            ).scalar_one()
            == 1
        )
    persist.settle_legacy_pending_order("legacy", 2, 12, "filled")
    with engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT count(*) FROM paper_trades WHERE order_id = 'legacy'")
            ).scalar_one()
            == 1
        )
        assert (
            conn.execute(
                text("SELECT count(*) FROM paper_pending_orders WHERE order_id = 'legacy'")
            ).scalar_one()
            == 0
        )
    persist.log_trade("2026-01-01", "BUY", "OLD", 1, "old")
    persist.init_paper_schema()
    persist.create_order_intent("c1", "2026-01-02", "BUY", "ABC", 100, "notional", "test")
    with pytest.raises(Exception):
        persist.create_order_intent("c1", "2026-01-02", "BUY", "ABC", 100, "notional", None)
    assert len(persist.get_unresolved_order_intents()) == 1
    persist.attach_broker_order("c1", "b1")
    persist.attach_broker_order("c1", "b1")
    with pytest.raises(ValueError):
        persist.attach_broker_order("c1", "other")
    assert persist.get_unresolved_order_intents()[0]["state"] == "submitted"

    def fail_update(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE paper_order_intents SET broker_order_id"):
            raise RuntimeError("injected failure")

    event.listen(engine, "before_cursor_execute", fail_update)
    with pytest.raises(RuntimeError, match="injected"):
        persist.settle_order_intent("c1", "b1", 2, 12, "canceled")
    event.remove(engine, "before_cursor_execute", fail_update)
    with engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT count(*) FROM paper_trades WHERE order_id = 'b1'")
            ).scalar_one()
            == 0
        )
    assert len(persist.get_unresolved_order_intents()) == 1
    persist.settle_order_intent("c1", "b1", 2, 12, "canceled")
    persist.settle_order_intent("c1", "b1", 2, 12, "canceled")
    with pytest.raises(ValueError):
        persist.settle_order_intent("c1", "b1", 3, 8, "canceled")
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT run_date, amount, reason FROM paper_trades WHERE order_id = 'b1'")
        ).all()
        assert rows == [(date(2026, 1, 2), 24, "test")]
        assert (
            conn.execute(
                text("SELECT count(*) FROM paper_trades WHERE order_id = 'old'")
            ).scalar_one()
            == 1
        )
    persist.create_order_intent("c2", "2026-01-02", "SELL", "ABC", 2, "qty", None)
    persist.settle_order_intent("c2", "b2", 0, None, "canceled")
    assert persist.get_unresolved_order_intents() == []
    with persist.paper_run_lock() as lock:
        lock.verify()
        with pytest.raises(RuntimeError, match="already locked"):
            with persist.paper_run_lock():
                pass
    with persist.paper_run_lock() as lock:
        lock.verify()

    # Existing matching rows are reused; conflicts and historical duplicates fail closed.
    for suffix, amounts, succeeds in [
        ("match", [24], True),
        ("conflict", [25], False),
        ("duplicate", [24, 24], False),
    ]:
        persist.create_order_intent(suffix, "2026-01-02", "BUY", "ABC", 24, "notional", None)
        with engine.begin() as conn:
            for amount in amounts:
                conn.execute(
                    text(
                        "INSERT INTO paper_trades (run_date, side, symbol, amount, order_id) "
                        "VALUES ('2026-01-02', 'BUY', 'ABC', :amount, :id)"
                    ),
                    {"amount": amount, "id": suffix},
                )
        if succeeds:
            persist.settle_order_intent(suffix, suffix, 2, 12, "filled")
        else:
            with pytest.raises(ValueError, match="Conflicting historical trade"):
                persist.settle_order_intent(suffix, suffix, 2, 12, "filled")
    for side, amount, kind in [("BAD", 1, "qty"), ("BUY", 0, "notional"), ("BUY", 1, "qty")]:
        with pytest.raises(Exception):
            persist.create_order_intent("invalid", "2026-01-02", side, "ABC", amount, kind, None)

    # Existing legacy representations block both attach and settlement.
    for action in ("attach", "settle"):
        persist.create_order_intent(action, "2026-01-02", "BUY", "ABC", 24, "notional", None)
        persist.log_pending_order("2026-01-02", "BUY", "ABC", 24, action, None)
        with pytest.raises(ValueError, match="both"):
            if action == "attach":
                persist.attach_broker_order(action, action)
            else:
                persist.settle_order_intent(action, action, 2, 12, "filled")
        with engine.connect() as conn:
            assert (
                conn.execute(
                    text("SELECT count(*) FROM paper_trades WHERE order_id = :id"), {"id": action}
                ).scalar_one()
                == 0
            )

    # Once attached, the legacy settlement must also refuse the duplicate.
    persist.create_order_intent("attached", "2026-01-02", "BUY", "ABC", 24, "notional", None)
    persist.attach_broker_order("attached", "attached-b")
    persist.log_pending_order("2026-01-02", "BUY", "ABC", 24, "attached-b", None)
    with pytest.raises(ValueError, match="both"):
        persist.settle_legacy_pending_order("attached-b", 2, 12, "filled")

    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    persist.create_order_intent("cross-race", "2026-01-02", "BUY", "ABC", 24, "notional", None)
    persist.log_pending_order("2026-01-02", "BUY", "ABC", 24, "cross-b", None)
    cross_barrier = Barrier(2)

    def cross_attach():
        cross_barrier.wait(timeout=5)
        try:
            persist.attach_broker_order("cross-race", "cross-b")
            return "attached"
        except ValueError:
            return "rejected"

    def cross_legacy():
        cross_barrier.wait(timeout=5)
        persist.settle_legacy_pending_order("cross-b", 2, 12, "filled")

    with ThreadPoolExecutor(max_workers=2) as pool:
        attach_future = pool.submit(cross_attach)
        legacy_future = pool.submit(cross_legacy)
        legacy_future.result(timeout=10)
        assert attach_future.result(timeout=10) in ("attached", "rejected")
    with engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT count(*) FROM paper_trades WHERE order_id = 'cross-b'")
            ).scalar_one()
            == 1
        )

    # Two simultaneous writers serialize on the broker order ID.
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    persist.create_order_intent("race", "2026-01-02", "BUY", "ABC", 24, "notional", None)
    barrier = Barrier(2)

    def settle():
        barrier.wait(timeout=5)
        persist.settle_order_intent("race", "race-b", 2, 12, "filled")

    def legacy():
        barrier.wait(timeout=5)
        persist.log_trade("2026-01-02", "BUY", "ABC", 24, "race-b")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(settle), pool.submit(legacy)]
        for future in futures:
            future.result(timeout=10)
    persist.create_order_intent("race2", "2026-01-02", "BUY", "ABC", 24, "notional", None)
    barrier2 = Barrier(2)

    def settle_twice():
        barrier2.wait(timeout=5)
        persist.settle_order_intent("race2", "race2-b", 2, 12, "filled")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(settle_twice) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)
    with engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT count(*) FROM paper_trades WHERE order_id = 'race2-b'")
            ).scalar_one()
            == 1
        )
    with engine.connect() as conn:
        assert (
            conn.execute(
                text("SELECT count(*) FROM paper_trades WHERE order_id = 'race-b'")
            ).scalar_one()
            == 1
        )

    with pytest.raises(Exception):
        with persist.paper_run_lock() as lock:
            lock.verify()
            lock._conn.invalidate()
            lock.verify()


def test_legacy_trade_checks_existing_rows_under_lock():
    import pytest

    from ggTrader.paper.persist import log_trade

    matching = (date(2026, 1, 2), "BUY", "ABC", 24.0, None)
    for rows, error, inserts in [
        ([], False, 1),
        ([matching], False, 0),
        ([matching, matching], True, 0),
        ([(date(2026, 1, 2), "BUY", "ABC", 25.0, None)], True, 0),
    ]:
        with patch("ggTrader.paper.persist._get_engine") as engine:
            conn = engine.return_value.connect.return_value.__enter__.return_value
            conn.execute.return_value.all.return_value = rows
            if error:
                with pytest.raises(ValueError, match="Conflicting"):
                    log_trade("2026-01-02", "BUY", "ABC", 24, "b")
            else:
                log_trade("2026-01-02", "BUY", "ABC", 24, "b")
            sql = [str(call.args[0]) for call in conn.execute.call_args_list]
            assert "pg_advisory_xact_lock" in sql[0]
            assert sum("INSERT INTO paper_trades" in statement for statement in sql) == inserts


@pytest.mark.parametrize(
    ("qty", "price", "status"),
    [
        (None, 1, "filled"),
        (float("nan"), 1, "filled"),
        (-1, 1, "canceled"),
        (1, None, "filled"),
        (1, 0, "filled"),
        (1, float("inf"), "filled"),
        (0, None, "filled"),
        (1, 1, "unknown"),
    ],
)
def test_legacy_settlement_rejects_invalid_fill_before_database(qty, price, status):
    from ggTrader.paper.persist import settle_legacy_pending_order

    with patch("ggTrader.paper.persist._get_engine") as engine:
        with pytest.raises(ValueError, match="Invalid"):
            settle_legacy_pending_order("legacy", qty, price, status)
        engine.assert_not_called()
