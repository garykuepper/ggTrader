"""Tests for ggTrader.lab.kelly — pooled expanding Kelly-fraction sizing."""

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.kelly import expanding_kelly_fraction, extract_trades, kelly_fraction_asof


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


class TestExpandingKellyFraction:
    def test_nan_before_min_trades(self):
        trades = pd.DataFrame({"ret": [0.05, -0.02, 0.03], "exit_time": _idx(3)})
        f_star = expanding_kelly_fraction(trades, min_trades=5)
        assert f_star.isna().all()

    def test_nan_when_only_wins_or_only_losses(self):
        trades = pd.DataFrame({"ret": [0.05] * 10, "exit_time": _idx(10)})
        f_star = expanding_kelly_fraction(trades, min_trades=3)
        assert f_star.isna().all()

    def test_matches_hand_computed_value(self):
        # 6 wins of +0.10, 4 losses of -0.05 -> W=0.6, avg_win=0.10, avg_loss=0.05, R=2
        # f* = W - (1-W)/R = 0.6 - 0.4/2 = 0.4
        rets = [0.10] * 6 + [-0.05] * 4
        trades = pd.DataFrame({"ret": rets, "exit_time": _idx(10)})
        f_star = expanding_kelly_fraction(trades, min_trades=3)
        assert f_star.iloc[-1] == pytest.approx(0.4)

    def test_is_causal_expanding(self):
        """f*.iloc[i] must be unaffected by trades after position i."""
        rets = [0.10, -0.05, 0.10, -0.05, 0.10, -0.05]
        trades = pd.DataFrame({"ret": rets, "exit_time": _idx(6)})
        full = expanding_kelly_fraction(trades, min_trades=2)
        prefix = expanding_kelly_fraction(trades.iloc[:4], min_trades=2)
        pd.testing.assert_series_equal(full.iloc[:4], prefix, check_names=False)

    def test_empty_trades_returns_empty_series(self):
        trades = pd.DataFrame(columns=["symbol", "entry_time", "exit_time", "ret"])
        f_star = expanding_kelly_fraction(trades)
        assert f_star.empty


class TestKellyFractionAsof:
    def test_uses_only_trades_strictly_before_asof(self):
        idx = _idx(5)
        f_star = pd.Series([0.1, 0.2, 0.3], index=idx[[0, 2, 4]])
        assert np.isnan(kelly_fraction_asof(f_star, idx[0]))
        assert kelly_fraction_asof(f_star, idx[1]) == pytest.approx(0.1)
        assert kelly_fraction_asof(f_star, idx[3]) == pytest.approx(0.2)

    def test_empty_series_returns_nan(self):
        assert np.isnan(kelly_fraction_asof(pd.Series(dtype=float), pd.Timestamp("2020-01-01")))
