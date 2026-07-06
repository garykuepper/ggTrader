"""Tests for ggTrader.lab.kelly — pooled expanding Kelly-fraction sizing."""

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.kelly import (
    expanding_kelly_fraction,
    extract_trades,
    kelly_fraction_asof,
    kelly_sizes,
)


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


class TestKellySizes:
    def test_nan_where_no_entry(self):
        idx = _idx(5)
        entries = pd.DataFrame({"A": [True, False, False, False, False]}, index=idx)
        exits = pd.DataFrame({"A": [False, True, False, False, False]}, index=idx)
        close = pd.DataFrame({"A": [100.0, 101, 102, 103, 104]}, index=idx)
        sizes = kelly_sizes(
            entries, exits, close, kelly_multiplier=0.5, base_size=0.03, max_size=0.05
        )
        no_entry = ~entries
        assert sizes[no_entry].isna().all().all()

    def test_falls_back_to_base_size_without_measurable_edge(self):
        idx = _idx(5)
        entries = pd.DataFrame({"A": [True, False, False, False, False]}, index=idx)
        exits = pd.DataFrame({"A": [False, True, False, False, False]}, index=idx)
        close = pd.DataFrame({"A": [100.0, 101, 102, 103, 104]}, index=idx)
        sizes = kelly_sizes(
            entries, exits, close, kelly_multiplier=0.5, base_size=0.03, max_size=0.05
        )
        assert sizes.at[idx[0], "A"] == pytest.approx(0.03)

    def test_capped_at_max_size(self):
        idx = _idx(40)
        entries = pd.DataFrame({"A": [False] * 40}, index=idx)
        exits = pd.DataFrame({"A": [False] * 40}, index=idx)
        prices = [100.0] * 40
        # 6 winning round-trips (+10%) then 4 losing round-trips (-5%).
        for i in range(10):
            entry_bar, exit_bar = 2 * i, 2 * i + 1
            entries.iloc[entry_bar, 0] = True
            exits.iloc[exit_bar, 0] = True
            prices[exit_bar] = 110.0 if i < 6 else 95.0
        entries.iloc[35, 0] = True
        exits.iloc[36, 0] = True
        close = pd.DataFrame({"A": prices}, index=idx)

        sizes = kelly_sizes(
            entries,
            exits,
            close,
            kelly_multiplier=5.0,
            base_size=0.03,
            max_size=0.05,
            min_trades=3,
        )
        # W=0.6, avg_win=0.10, avg_loss=0.05, R=2 -> f*=0.4; k*f*=2.0, must cap.
        assert sizes.at[idx[35], "A"] == pytest.approx(0.05)


class TestKellySizesCausality:
    def test_future_trades_do_not_affect_earlier_sizes(self):
        """Appending more trades/bars to the end of the data must not change
        the Kelly size computed for an earlier entry — the no-look-ahead
        property this sizing mechanism relies on for honest walk-forward."""
        idx = _idx(30)
        entries = pd.DataFrame({"A": [False] * 30}, index=idx)
        exits = pd.DataFrame({"A": [False] * 30}, index=idx)
        prices = [100.0] * 30
        for i in range(8):
            entry_bar, exit_bar = 2 * i, 2 * i + 1
            entries.iloc[entry_bar, 0] = True
            exits.iloc[exit_bar, 0] = True
            prices[exit_bar] = 110.0 if i % 2 == 0 else 95.0
        entries.iloc[20, 0] = True
        exits.iloc[21, 0] = True
        close_short = pd.DataFrame({"A": prices}, index=idx)
        sizes_short = kelly_sizes(
            entries,
            exits,
            close_short,
            kelly_multiplier=0.5,
            base_size=0.03,
            max_size=0.05,
            min_trades=3,
        )

        idx_long = _idx(38)
        entries_long = entries.reindex(idx_long, fill_value=False)
        exits_long = exits.reindex(idx_long, fill_value=False)
        prices_long = prices + [100.0] * 8
        entries_long.iloc[36, 0] = True
        exits_long.iloc[37, 0] = True
        prices_long[37] = 50.0  # a huge future loss
        close_long = pd.DataFrame({"A": prices_long}, index=idx_long)
        sizes_long = kelly_sizes(
            entries_long,
            exits_long,
            close_long,
            kelly_multiplier=0.5,
            base_size=0.03,
            max_size=0.05,
            min_trades=3,
        )
        assert sizes_long.at[idx[20], "A"] == pytest.approx(sizes_short.at[idx[20], "A"])
