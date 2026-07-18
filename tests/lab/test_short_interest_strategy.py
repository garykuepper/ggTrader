"""Tests for the short-interest (free-data-only cut) strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.short_interest import ShortInterestStrategy
from ggTrader.lab.strategy import LabConfig


def _idx(n, start="2020-01-01"):
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


def _flat_ohlcv(symbols, n=300):
    idx = _idx(n)
    returns = pd.DataFrame({s: np.zeros(n) for s in symbols}, index=idx)
    return _ohlcv_from_returns(returns)


def _si_row(symbol, settlement_date, dtc):
    return {
        "symbol": symbol,
        "settlement_date": pd.Timestamp(settlement_date),
        "days_to_cover": dtc,
    }


class TestSelect:
    def test_avoids_rising_days_to_cover_symbol(self):
        """B's days-to-cover roughly doubled cycle over cycle (increasingly
        hard to borrow); A's stayed flat. The strategy must pick A, not B."""
        symbols = ["A", "B", "C", "D", "E"]
        ohlcv = _flat_ohlcv(symbols, n=300)
        asof = ohlcv.index[-1]

        si_rows = []
        for sym, dtc_series in [
            ("A", [2.0, 2.0, 2.0]),  # flat -> unambiguously the smallest change
            ("B", [1.0, 2.0, 4.0]),  # rising fast -> avoid
            ("C", [3.0, 3.0, 3.3]),
            ("D", [1.5, 1.5, 1.65]),
            ("E", [2.5, 2.5, 2.75]),
        ]:
            dates = ["2020-01-15", "2020-02-15", "2020-03-15"]
            si_rows.extend(_si_row(sym, d, v) for d, v in zip(dates, dtc_series))
        si_df = pd.DataFrame(si_rows)

        strat = ShortInterestStrategy(
            LabConfig(min_history_bars=100),
            lookback_cycles=1,
            quintile=5,
            publish_lag_days=0,
            _si_loader=lambda symbols, start, end: si_df,
        )
        # asof well after the last settlement date so the publish-lag gate doesn't exclude it.
        sels = strat.select(asof, ohlcv, symbols)
        assert len(sels) == 1
        assert sels[0]["symbol"] == "A"
        assert "B" not in [s["symbol"] for s in sels]

    def test_no_short_interest_data_returns_empty_plan(self):
        symbols = ["A", "B", "C", "D", "E"]
        ohlcv = _flat_ohlcv(symbols, n=300)
        strat = ShortInterestStrategy(
            LabConfig(min_history_bars=100),
            _si_loader=lambda symbols, start, end: pd.DataFrame(
                columns=["symbol", "settlement_date", "days_to_cover"]
            ),
        )
        sels = strat.select(ohlcv.index[-1], ohlcv, symbols)
        assert sels == []

    def test_respects_min_history(self):
        symbols = ["A", "B"]
        ohlcv = _flat_ohlcv(symbols, n=50)
        strat = ShortInterestStrategy(
            LabConfig(min_history_bars=400),
            _si_loader=lambda symbols, start, end: pd.DataFrame(
                [_si_row("A", "2020-01-15", 2.0), _si_row("B", "2020-01-15", 3.0)]
            ),
        )
        sels = strat.select(ohlcv.index[-1], ohlcv, symbols)
        assert sels == []

    def test_publish_lag_excludes_too_recent_settlement(self):
        """A settlement dated right before asof, within the publish lag, must
        not be visible yet -- without it there aren't enough cycles to rank."""
        symbols = ["A", "B", "C", "D", "E"]
        ohlcv = _flat_ohlcv(symbols, n=300)
        asof = ohlcv.index[-1]

        si_rows = []
        for sym in symbols:
            # Only ONE settlement date, dated the day before asof -- with a
            # real publish lag this can't be seen yet regardless of ranking.
            si_rows.append(_si_row(sym, asof - pd.Timedelta(days=1), 2.0))
        si_df = pd.DataFrame(si_rows)

        strat = ShortInterestStrategy(
            LabConfig(min_history_bars=100),
            publish_lag_days=15,
            _si_loader=lambda symbols, start, end: si_df,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert sels == []

    def test_weight_is_one_over_bucket_size(self):
        symbols = ["A", "B", "C", "D"]
        ohlcv = _flat_ohlcv(symbols, n=300)
        asof = ohlcv.index[-1]
        si_rows = []
        for sym, dtc_series in [
            ("A", [2.0, 2.0]),
            ("B", [1.0, 3.0]),
            ("C", [3.0, 3.0]),
            ("D", [1.5, 1.4]),
        ]:
            dates = ["2020-01-15", "2020-02-15"]
            si_rows.extend(_si_row(sym, d, v) for d, v in zip(dates, dtc_series))
        si_df = pd.DataFrame(si_rows)

        strat = ShortInterestStrategy(
            LabConfig(min_history_bars=100),
            quintile=4,
            publish_lag_days=0,
            _si_loader=lambda symbols, start, end: si_df,
        )
        sels = strat.select(asof, ohlcv, symbols)
        assert sels
        for s in sels:
            assert s["weight"] == pytest.approx(1.0 / len(sels))


class TestToTargets:
    def test_returns_weight_dataframe(self):
        symbols = ["A", "B", "C"]
        ohlcv = _flat_ohlcv(symbols, n=60)
        strat = ShortInterestStrategy(LabConfig(min_history_bars=30))
        plans = {ohlcv.index[50]: [{"symbol": "A", "weight": 1.0}]}
        targets = strat.to_targets(plans, ohlcv)
        assert isinstance(targets, pd.DataFrame)
        assert set(targets.columns) == {"A"}


def test_sweep_params_has_lookback_cycles_and_quintile():
    params = ShortInterestStrategy.sweep_params()
    assert "lookback_cycles" in params
    assert "quintile" in params


def test_short_interest_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY

    assert "short_interest" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["short_interest"] is ShortInterestStrategy
