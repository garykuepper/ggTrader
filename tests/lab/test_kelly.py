"""Tests for ggTrader.lab.kelly — pooled expanding Kelly-fraction sizing."""

import pandas as pd
import pytest
from ggTrader.lab.kelly import extract_trades


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


class TestExtractTrades:
    def test_single_symbol_single_trade(self):
        idx = _idx(5)
        entries = pd.DataFrame({"A": [True, False, False, False, False]}, index=idx)
        exits = pd.DataFrame({"A": [False, False, True, False, False]}, index=idx)
        close = pd.DataFrame({"A": [100.0, 101, 110, 111, 112]}, index=idx)
        trades = extract_trades(entries, exits, close)
        assert len(trades) == 1
        row = trades.iloc[0]
        assert row["symbol"] == "A"
        assert row["entry_time"] == idx[0]
        assert row["exit_time"] == idx[2]
        assert row["ret"] == pytest.approx(0.10)

    def test_redundant_entries_while_in_position_are_ignored(self):
        idx = _idx(6)
        entries = pd.DataFrame({"A": [True, True, False, False, False, False]}, index=idx)
        exits = pd.DataFrame({"A": [False, False, False, True, False, False]}, index=idx)
        close = pd.DataFrame({"A": [100.0, 100, 100, 105, 105, 105]}, index=idx)
        trades = extract_trades(entries, exits, close)
        assert len(trades) == 1
        assert trades.iloc[0]["entry_time"] == idx[0]

    def test_unrealized_trade_at_end_is_dropped(self):
        idx = _idx(4)
        entries = pd.DataFrame({"A": [False, True, False, False]}, index=idx)
        exits = pd.DataFrame({"A": [False, False, False, False]}, index=idx)
        close = pd.DataFrame({"A": [100.0, 100, 100, 100]}, index=idx)
        trades = extract_trades(entries, exits, close)
        assert trades.empty

    def test_multiple_symbols_pooled_and_sorted_by_exit_time(self):
        idx = _idx(6)
        entries = pd.DataFrame(
            {
                "A": [True, False, False, False, False, False],
                "B": [False, True, False, False, False, False],
            },
            index=idx,
        )
        exits = pd.DataFrame(
            {
                "A": [False, False, False, True, False, False],
                "B": [False, False, True, False, False, False],
            },
            index=idx,
        )
        close = pd.DataFrame(
            {
                "A": [100.0, 100, 100, 105, 105, 105],
                "B": [50.0, 50, 55, 55, 55, 55],
            },
            index=idx,
        )
        trades = extract_trades(entries, exits, close)
        assert list(trades["symbol"]) == ["B", "A"]
        assert trades["exit_time"].is_monotonic_increasing

    def test_new_entry_after_close_opens_a_new_trade(self):
        idx = _idx(6)
        entries = pd.DataFrame({"A": [True, False, False, True, False, False]}, index=idx)
        exits = pd.DataFrame({"A": [False, True, False, False, True, False]}, index=idx)
        close = pd.DataFrame({"A": [100.0, 110, 110, 110, 121, 121]}, index=idx)
        trades = extract_trades(entries, exits, close)
        assert len(trades) == 2
        assert trades.iloc[0]["entry_time"] == idx[0]
        assert trades.iloc[1]["entry_time"] == idx[3]
