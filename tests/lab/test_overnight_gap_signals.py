"""Tests for overnight-gap reversion indicator functions and signal class."""

import numpy as np
import pandas as pd

from ggTrader.lab.strategies.indicators import extract_open, overnight_gap_signals


def _idx(n, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="B", tz="UTC")


def _ohlcv(symbols, n=300, seed=42):
    rng = np.random.default_rng(seed)
    idx = _idx(n)
    frames = {}
    for i, s in enumerate(symbols):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
        open_ = close + rng.normal(0, 0.05, n)  # tiny day-to-day gaps by default
        frames[s] = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) * 1.005,
                "low": np.minimum(open_, close) * 0.995,
                "close": close,
                "volume": np.full(n, 1e6),
            },
            index=idx,
        )
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


class TestExtractOpen:
    def test_shape_matches_close(self):
        ohlcv = _ohlcv(["A", "B"], n=100)
        opens = extract_open(ohlcv, ["A", "B"])
        assert opens.shape == (100, 2)
        assert list(opens.columns) == ["A", "B"]

    def test_missing_symbol_skipped(self):
        ohlcv = _ohlcv(["A"], n=50)
        opens = extract_open(ohlcv, ["A", "NOSUCH"])
        assert "A" in opens.columns
        assert "NOSUCH" not in opens.columns


class TestOvernightGapSignals:
    def test_output_shape_and_dtype(self):
        ohlcv = _ohlcv(["A"], n=200)
        close = ohlcv["A"]["close"].to_frame("A")
        open_ = ohlcv["A"]["open"].to_frame("A")
        entries, exits = overnight_gap_signals(close, open_, 20, -1.5, -0.5)
        assert entries.shape == close.shape
        assert exits.shape == close.shape
        assert entries.dtypes.eq(bool).all()
        assert exits.dtypes.eq(bool).all()

    def test_entry_on_extreme_gap_down(self):
        """A large overnight gap-down should trigger an entry that same bar."""
        idx = _idx(100)
        rng = np.random.default_rng(1)
        close_vals = np.full(100, 100.0)
        open_vals = close_vals + rng.normal(0, 0.05, 100)  # tiny gaps most days
        open_vals[50] = close_vals[49] * 0.85  # 15% overnight gap down on day 50
        close = pd.DataFrame({"A": close_vals}, index=idx)
        open_ = pd.DataFrame({"A": open_vals}, index=idx)
        entries, _ = overnight_gap_signals(
            close, open_, gap_lookback=20, gap_z_entry=-1.5, gap_z_exit=-0.5
        )
        assert entries["A"].iloc[50], "Should enter on extreme overnight gap-down"

    def test_no_entries_in_warmup(self):
        """No signals before gap_lookback bars of history exist."""
        idx = _idx(50)
        close = pd.DataFrame({"A": np.full(50, 100.0)}, index=idx)
        open_ = pd.DataFrame({"A": np.full(50, 100.0)}, index=idx)
        entries, _ = overnight_gap_signals(
            close, open_, gap_lookback=20, gap_z_entry=-1.5, gap_z_exit=-0.5
        )
        assert not entries["A"].iloc[:20].any()

    def test_stricter_threshold_fewer_entries(self):
        """A more extreme (more negative) gap_z_entry should produce fewer entries."""
        idx = _idx(300)
        rng = np.random.default_rng(7)
        close_vals = np.full(300, 100.0)
        open_vals = close_vals + rng.normal(0, 1.0, 300)
        close = pd.DataFrame({"A": close_vals}, index=idx)
        open_ = pd.DataFrame({"A": open_vals}, index=idx)
        entries_loose, _ = overnight_gap_signals(close, open_, 20, -1.0, -0.5)
        entries_strict, _ = overnight_gap_signals(close, open_, 20, -2.5, -0.5)
        assert entries_loose["A"].sum() >= entries_strict["A"].sum()
