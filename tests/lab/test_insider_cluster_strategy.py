"""Tests for the insider cluster-buying (SEC Form 4) strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.insider_cluster import (
    InsiderClusterBuyStrategy,
    cluster_events,
)
from ggTrader.lab.strategy import LabConfig


def _tx(symbol, insider_cik, date, code="P", is_10b5_1=False, filing_date=None):
    """A Form 4 row. ``filing_date`` defaults to the SEC's 2-business-day
    deadline after the trade, which is also the measured median lag across
    the 783,982 stored rows (p95 = 5 days)."""
    tx_date = pd.Timestamp(date)
    return {
        "symbol": symbol,
        "insider_cik": insider_cik,
        "transaction_date": tx_date,
        "transaction_code": code,
        "is_10b5_1_plan": is_10b5_1,
        "filing_date": pd.Timestamp(filing_date) if filing_date else tx_date + pd.Timedelta(days=2),
    }


class TestPointInTimeAvailability:
    """A Form 4 cluster is tradeable when the filing becomes public, not
    when the insider traded. Using ``transaction_date`` as the event date
    gave a median 2-day lookahead (audit 2026-07-25 §2.1B); the sibling
    ``congress_trades`` strategy already keys off ``filing_date``."""

    def test_event_date_is_the_filing_date_not_the_transaction_date(self):
        txs = pd.DataFrame(
            [
                _tx("AAPL", 1, "2020-01-01", filing_date="2020-01-08"),
                _tx("AAPL", 2, "2020-01-02", filing_date="2020-01-08"),
                _tx("AAPL", 3, "2020-01-03", filing_date="2020-01-08"),
            ]
        )
        events = cluster_events(txs, cluster_window_days=30, min_insiders=3)
        assert len(events) == 1
        assert events.iloc[0]["event_date"] == pd.Timestamp("2020-01-08"), (
            "event must fire when the last filing became public, not on the insider's trade date"
        )

    def test_cluster_completes_on_the_latest_filing_of_its_members(self):
        """Three insiders traded within days of each other but the third
        filed late -- the cluster is not knowable until that filing."""
        txs = pd.DataFrame(
            [
                _tx("AAPL", 1, "2020-01-01", filing_date="2020-01-03"),
                _tx("AAPL", 2, "2020-01-02", filing_date="2020-01-04"),
                _tx("AAPL", 3, "2020-01-03", filing_date="2020-02-20"),
            ]
        )
        events = cluster_events(txs, cluster_window_days=90, min_insiders=3)
        assert len(events) == 1
        assert events.iloc[0]["event_date"] == pd.Timestamp("2020-02-20")


class TestClusterEvents:
    def test_detects_a_cluster_of_three_distinct_insiders(self):
        txs = pd.DataFrame(
            [
                _tx("AAPL", 1, "2020-01-01"),
                _tx("AAPL", 2, "2020-01-05"),
                _tx("AAPL", 3, "2020-01-10"),
            ]
        )
        events = cluster_events(txs, cluster_window_days=14, min_insiders=3)
        assert len(events) == 1
        assert events.iloc[0]["symbol"] == "AAPL"
        # The cluster completes on the last insider's trade (2020-01-10) but
        # only becomes public when that Form 4 is filed -- here the default
        # two-business-day deadline. See TestPointInTimeAvailability.
        assert events.iloc[0]["event_date"] == pd.Timestamp("2020-01-12")

    def test_two_distinct_insiders_is_not_a_cluster(self):
        txs = pd.DataFrame([_tx("AAPL", 1, "2020-01-01"), _tx("AAPL", 2, "2020-01-05")])
        events = cluster_events(txs, cluster_window_days=14, min_insiders=3)
        assert events.empty

    def test_same_insider_buying_twice_does_not_count_as_two_distinct_insiders(self):
        txs = pd.DataFrame(
            [
                _tx("AAPL", 1, "2020-01-01"),
                _tx("AAPL", 1, "2020-01-03"),  # same insider again
                _tx("AAPL", 2, "2020-01-05"),
            ]
        )
        events = cluster_events(txs, cluster_window_days=14, min_insiders=3)
        assert events.empty

    def test_purchases_outside_the_window_do_not_cluster(self):
        txs = pd.DataFrame(
            [
                _tx("AAPL", 1, "2020-01-01"),
                _tx("AAPL", 2, "2020-01-05"),
                _tx("AAPL", 3, "2020-02-01"),  # 27 days after the first -- outside a 14-day window
            ]
        )
        events = cluster_events(txs, cluster_window_days=14, min_insiders=3)
        assert events.empty

    def test_excludes_10b5_1_plan_trades(self):
        txs = pd.DataFrame(
            [
                _tx("AAPL", 1, "2020-01-01"),
                _tx("AAPL", 2, "2020-01-05"),
                _tx("AAPL", 3, "2020-01-10", is_10b5_1=True),  # scheduled plan, must not count
            ]
        )
        events = cluster_events(txs, cluster_window_days=14, min_insiders=3)
        assert events.empty

    def test_excludes_non_purchase_transaction_codes(self):
        txs = pd.DataFrame(
            [
                _tx("AAPL", 1, "2020-01-01"),
                _tx("AAPL", 2, "2020-01-05"),
                _tx("AAPL", 3, "2020-01-10", code="M"),  # option exercise, not an open-market buy
            ]
        )
        events = cluster_events(txs, cluster_window_days=14, min_insiders=3)
        assert events.empty

    def test_events_are_scoped_per_symbol(self):
        txs = pd.DataFrame(
            [
                _tx("AAPL", 1, "2020-01-01"),
                _tx("AAPL", 2, "2020-01-05"),
                _tx("MSFT", 3, "2020-01-01"),
            ]
        )
        events = cluster_events(txs, cluster_window_days=14, min_insiders=3)
        assert events.empty  # neither symbol alone has 3 distinct insiders

    def test_only_one_event_per_cluster_not_one_per_subsequent_purchase(self):
        """A 4th purchase inside an already-triggered cluster's window must
        not create a second event -- otherwise the same cluster would
        re-enter the position repeatedly."""
        txs = pd.DataFrame(
            [
                _tx("AAPL", 1, "2020-01-01"),
                _tx("AAPL", 2, "2020-01-05"),
                _tx("AAPL", 3, "2020-01-10"),  # cluster completes here
                _tx("AAPL", 4, "2020-01-12"),  # still within window, must not double-count
            ]
        )
        events = cluster_events(txs, cluster_window_days=14, min_insiders=3)
        assert len(events) == 1


def _idx(n, start="2019-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv_from_returns(returns: pd.DataFrame) -> pd.DataFrame:
    frames = {}
    for col in returns.columns:
        close = 100.0 * (1.0 + returns[col]).cumprod()
        frames[col] = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.001,
                "low": close * 0.999,
                "close": close,
                "volume": np.full(len(close), 1e6),
            },
            index=returns.index,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


def _flat_ohlcv(symbols, n=400, start="2019-01-01"):
    idx = _idx(n, start)
    returns = pd.DataFrame({s: np.zeros(n) for s in symbols}, index=idx)
    return _ohlcv_from_returns(returns)


class TestSelect:
    def test_holds_a_symbol_with_an_active_cluster_event(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        txs = pd.DataFrame(
            [
                _tx("B", 1, asof - pd.Timedelta(days=20)),
                _tx("B", 2, asof - pd.Timedelta(days=16)),
                _tx("B", 3, asof - pd.Timedelta(days=10)),
            ]
        )
        strat = InsiderClusterBuyStrategy(
            LabConfig(min_history_bars=30),
            hold_days=252,
            _tx_loader=lambda symbols, start, end: txs,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert len(sels) == 1
        assert sels[0]["symbol"] == "B"

    def test_no_transactions_returns_empty_plan(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        strat = InsiderClusterBuyStrategy(
            LabConfig(min_history_bars=30),
            _tx_loader=lambda symbols, start, end: pd.DataFrame(
                columns=[
                    "symbol",
                    "insider_cik",
                    "transaction_date",
                    "transaction_code",
                    "is_10b5_1_plan",
                ]
            ),
        )
        sels = strat.select(ohlcv.index[-1], ohlcv, symbols)
        assert sels == []

    def test_event_older_than_hold_days_is_excluded(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        txs = pd.DataFrame(
            [
                _tx("B", 1, asof - pd.Timedelta(days=300)),
                _tx("B", 2, asof - pd.Timedelta(days=296)),
                _tx("B", 3, asof - pd.Timedelta(days=290)),
            ]
        )
        strat = InsiderClusterBuyStrategy(
            LabConfig(min_history_bars=30),
            hold_days=126,
            _tx_loader=lambda symbols, start, end: txs,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert sels == []

    def test_respects_min_history(self):
        symbols = ["A", "B"]
        ohlcv = _flat_ohlcv(symbols, n=25)
        strat = InsiderClusterBuyStrategy(
            LabConfig(min_history_bars=400),
            _tx_loader=lambda symbols, start, end: pd.DataFrame(
                columns=[
                    "symbol",
                    "insider_cik",
                    "transaction_date",
                    "transaction_code",
                    "is_10b5_1_plan",
                ]
            ),
        )
        sels = strat.select(ohlcv.index[-1], ohlcv, symbols)
        assert sels == []

    def test_multiple_active_clusters_are_equal_weighted(self):
        symbols = ["A", "B", "C", "D"]
        ohlcv = _flat_ohlcv(symbols, n=400)
        asof = ohlcv.index[-1]
        txs = pd.DataFrame(
            [
                _tx("B", 1, asof - pd.Timedelta(days=20)),
                _tx("B", 2, asof - pd.Timedelta(days=16)),
                _tx("B", 3, asof - pd.Timedelta(days=10)),
                _tx("D", 4, asof - pd.Timedelta(days=25)),
                _tx("D", 5, asof - pd.Timedelta(days=20)),
                _tx("D", 6, asof - pd.Timedelta(days=15)),
            ]
        )
        strat = InsiderClusterBuyStrategy(
            LabConfig(min_history_bars=30),
            hold_days=252,
            _tx_loader=lambda symbols, start, end: txs,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert {s["symbol"] for s in sels} == {"B", "D"}
        for s in sels:
            assert s["weight"] == pytest.approx(0.5)


class TestToTargets:
    def test_returns_weight_dataframe(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=60)
        strat = InsiderClusterBuyStrategy(LabConfig(min_history_bars=30))
        plans = {ohlcv.index[50]: [{"symbol": "B", "weight": 1.0}]}
        targets = strat.to_targets(plans, ohlcv)
        assert isinstance(targets, pd.DataFrame)
        assert set(targets.columns) == {"B"}


def test_sweep_params_has_hold_days():
    params = InsiderClusterBuyStrategy.sweep_params()
    assert "hold_days" in params


def test_insider_cluster_buy_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    assert "insider_cluster_buy" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["insider_cluster_buy"] is InsiderClusterBuyStrategy
