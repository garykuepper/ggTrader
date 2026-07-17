"""Tests for the long-only leveraged-ETF trend-following strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _clear_underlying_cache():
    """The module-level underlying-close cache is keyed by (ticker, start,
    end) -- safe for real market data, but tests below reuse the same
    default synthetic date range across different fixtures, which would
    collide on that key without clearing between tests."""
    from ggTrader.lab.strategies.leveraged_trend import _underlying_cache

    _underlying_cache.clear()
    yield
    _underlying_cache.clear()


def _idx(n, start="2020-01-01", freq="B"):
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


def _concrete_cls():
    from ggTrader.lab.strategies.leveraged_trend import _LeveragedTrendBase

    class _Concrete(_LeveragedTrendBase):
        name = "leveraged_trend_test"
        ETF_3X = "LONG3X"
        ETF_2X = "LONG2X"
        UNDERLYING = "UND"

    return _Concrete


def _mock_underlying(monkeypatch, close: pd.Series):
    """Patch load_ohlcv so _cached_underlying_close returns a fixed series."""
    import ggTrader.lab.strategies.leveraged_trend as mod

    def _fake_load_ohlcv(symbols, start, end, use_negative_cache=True):
        frame = pd.DataFrame({"close": close})
        frame.columns = pd.MultiIndex.from_product([[symbols[0]], frame.columns])
        return frame

    monkeypatch.setattr(mod, "load_ohlcv", _fake_load_ohlcv, raising=False)


def _uptrend(n, start="2020-01-01"):
    idx = _idx(n, start)
    return pd.Series(100.0 * np.exp(np.linspace(0, 0.5, n)), index=idx)


def _downtrend(n, start="2020-01-01"):
    idx = _idx(n, start)
    return pd.Series(100.0 * np.exp(np.linspace(0.5, 0, n)), index=idx)


class TestTrendState:
    def test_uptrend_is_long_after_warmup(self):
        from ggTrader.lab.strategies.leveraged_trend import _trend_state

        close = _uptrend(300)
        is_long = _trend_state(close, close.index, trend_window=50, min_hold_days=1)
        assert not is_long.iloc[:49].any()
        assert is_long.iloc[100:].all()

    def test_downtrend_is_never_long(self):
        from ggTrader.lab.strategies.leveraged_trend import _trend_state

        close = _downtrend(300)
        is_long = _trend_state(close, close.index, trend_window=50, min_hold_days=1)
        assert not is_long.iloc[100:].any()

    def test_min_hold_requires_confirmation(self):
        """A single-day blip below the SMA should not flip state if
        min_hold_days requires multiple consecutive confirming readings."""
        from ggTrader.lab.strategies.leveraged_trend import _trend_state

        idx = _idx(10)
        # Flat-ish series engineered so day 5 dips just below its trailing
        # mean for one day, then resumes -- with trend_window=3 the raw
        # signal flips down for exactly one reading.
        vals = [100, 101, 102, 103, 104, 90, 106, 107, 108, 109]
        close = pd.Series(vals, index=idx, dtype=float)
        is_long = _trend_state(close, close.index, trend_window=3, min_hold_days=3)
        # Once the initial long confirmation lands (index 4, after 3
        # consecutive up readings), the single-day dip at index 5 never
        # holds 3 consecutive down readings -> state never flips back.
        assert is_long.iloc[4:].all()


class TestEntriesExits:
    def test_entries_fire_only_on_transition_up(self):
        from ggTrader.lab.strategies.leveraged_trend import _entries_exits

        is_long = pd.Series([False, False, True, True, True], index=_idx(5))
        entries, exits = _entries_exits(is_long, "ETF")
        assert list(entries["ETF"]) == [False, False, True, False, False]
        assert not exits["ETF"].any()

    def test_exits_fire_only_on_transition_down(self):
        from ggTrader.lab.strategies.leveraged_trend import _entries_exits

        is_long = pd.Series([False, True, True, False, False], index=_idx(5))
        entries, exits = _entries_exits(is_long, "ETF")
        assert list(exits["ETF"]) == [False, False, False, True, False]
        assert list(entries["ETF"]) == [False, True, False, False, False]

    def test_long_from_first_bar_enters_immediately(self):
        from ggTrader.lab.strategies.leveraged_trend import _entries_exits

        is_long = pd.Series([True, True, True], index=_idx(3))
        entries, _exits = _entries_exits(is_long, "ETF")
        assert list(entries["ETF"]) == [True, False, False]


class TestSelect:
    def test_select_returns_active_tier_only(self):
        from ggTrader.lab.strategy import LabConfig

        strat = _concrete_cls()(LabConfig(), leverage_tier="3x")
        plan = strat.select(
            pd.Timestamp("2020-06-30", tz="UTC"), pd.DataFrame(), ["LONG3X", "LONG2X"]
        )
        assert plan == [{"symbol": "LONG3X", "weight": 1.0}]

    def test_select_2x_tier(self):
        from ggTrader.lab.strategy import LabConfig

        strat = _concrete_cls()(LabConfig(), leverage_tier="2x")
        plan = strat.select(
            pd.Timestamp("2020-06-30", tz="UTC"), pd.DataFrame(), ["LONG3X", "LONG2X"]
        )
        assert plan == [{"symbol": "LONG2X", "weight": 1.0}]

    def test_select_empty_when_etf_not_eligible(self):
        from ggTrader.lab.strategy import LabConfig

        strat = _concrete_cls()(LabConfig(), leverage_tier="3x")
        plan = strat.select(pd.Timestamp("2020-06-30", tz="UTC"), pd.DataFrame(), ["LONG2X"])
        assert plan == []


class TestToTargets:
    def test_degenerate_to_buy_and_hold(self, monkeypatch):
        """A monotonic uptrend should produce exactly one entry and no
        exits -- the strategy degenerates to plain buy-and-hold."""
        from ggTrader.lab.strategy import LabConfig

        close = _uptrend(300)
        _mock_underlying(monkeypatch, close)
        strat = _concrete_cls()(LabConfig(), trend_window=50, leverage_tier="3x", min_hold_days=1)
        targets = strat.to_targets({}, pd.DataFrame(index=close.index))

        assert targets.entries["LONG3X"].sum() == 1
        assert not targets.exits["LONG3X"].any()

    def test_never_holds_inactive_tier_column(self, monkeypatch):
        from ggTrader.lab.strategy import LabConfig

        close = _uptrend(300)
        _mock_underlying(monkeypatch, close)
        strat = _concrete_cls()(LabConfig(), leverage_tier="2x")
        targets = strat.to_targets({}, pd.DataFrame(index=close.index))
        assert list(targets.entries.columns) == ["LONG2X"]
        assert list(targets.exits.columns) == ["LONG2X"]

    def test_empty_data_returns_empty_frames(self):
        from ggTrader.lab.strategy import LabConfig

        strat = _concrete_cls()(LabConfig())
        targets = strat.to_targets({}, pd.DataFrame())
        assert len(targets.entries) == 0
        assert len(targets.exits) == 0


class TestSweepSignals:
    def test_sweep_signals_returns_one_result_per_combo(self, monkeypatch):
        from ggTrader.lab.strategy import LabConfig
        from ggTrader.lab.sweep import combo_name

        close = _uptrend(300)
        _mock_underlying(monkeypatch, close)
        strat = _concrete_cls()(LabConfig())
        combos = [
            {"trend_window": 50, "leverage_tier": "3x", "min_hold_days": 1},
            {"trend_window": 100, "leverage_tier": "2x", "min_hold_days": 5},
        ]
        result = strat.sweep_signals(combos, ["LONG3X", "LONG2X"], pd.DataFrame(index=close.index))

        assert set(result.keys()) == {combo_name(strat.name, c) for c in combos}
        key0 = combo_name(strat.name, combos[0])
        key1 = combo_name(strat.name, combos[1])
        assert list(result[key0].entries.columns) == ["LONG3X"]
        assert list(result[key1].entries.columns) == ["LONG2X"]

    def test_sweep_signals_caches_underlying_load(self, monkeypatch):
        """WFO builds a fresh strategy instance per grid combo and calls
        sweep_signals once per stop-config group on the same data window --
        the underlying OHLCV load must not repeat per combo."""
        from unittest.mock import MagicMock

        import ggTrader.lab.strategies.leveraged_trend as mod
        from ggTrader.lab.strategy import LabConfig

        close = _uptrend(300)
        frame = pd.DataFrame({"close": close})
        frame.columns = pd.MultiIndex.from_product([["UND"], frame.columns])
        mock_load = MagicMock(return_value=frame)
        monkeypatch.setattr(mod, "load_ohlcv", mock_load, raising=False)

        strat = _concrete_cls()(LabConfig())
        combos = [
            {"trend_window": 50, "leverage_tier": "3x", "min_hold_days": 1},
            {"trend_window": 100, "leverage_tier": "2x", "min_hold_days": 5},
        ]
        strat.sweep_signals(combos, ["LONG3X", "LONG2X"], pd.DataFrame(index=close.index))
        strat.sweep_signals(combos, ["LONG3X", "LONG2X"], pd.DataFrame(index=close.index))

        assert mock_load.call_count == 1


class TestSweepParams:
    def test_sweep_params_grid(self):
        params = _concrete_cls().sweep_params()
        assert "trend_window" in params
        assert "leverage_tier" in params
        assert "min_hold_days" in params
        assert "vol_target" in params
        assert set(params["leverage_tier"]) == {"2x", "3x"}


class TestPerUniverseSubclasses:
    def test_sp500(self):
        from ggTrader.lab.strategies.leveraged_trend import LeveragedTrendSp500

        assert LeveragedTrendSp500.ETF_3X == "UPRO"
        assert LeveragedTrendSp500.ETF_2X == "SSO"
        assert LeveragedTrendSp500.UNDERLYING == "SPY"
        assert LeveragedTrendSp500.name == "leveraged_trend_sp500"

    def test_nasdaq100(self):
        from ggTrader.lab.strategies.leveraged_trend import LeveragedTrendNasdaq100

        assert LeveragedTrendNasdaq100.ETF_3X == "TQQQ"
        assert LeveragedTrendNasdaq100.ETF_2X == "QLD"
        assert LeveragedTrendNasdaq100.UNDERLYING == "QQQ"

    def test_russell2000(self):
        from ggTrader.lab.strategies.leveraged_trend import LeveragedTrendRussell2000

        assert LeveragedTrendRussell2000.ETF_3X == "TNA"
        assert LeveragedTrendRussell2000.ETF_2X == "UWM"
        assert LeveragedTrendRussell2000.UNDERLYING == "IWM"

    def test_bare_construction_works_without_extra_args(self):
        """wfo.py calls strategy_cls(cfg) with no extra args in some paths
        (anchor-set computation) -- every subclass must support this."""
        from ggTrader.lab.strategies.leveraged_trend import (
            LeveragedTrendNasdaq100,
            LeveragedTrendRussell2000,
            LeveragedTrendSp500,
        )
        from ggTrader.lab.strategy import LabConfig

        for cls in (LeveragedTrendSp500, LeveragedTrendNasdaq100, LeveragedTrendRussell2000):
            strat = cls(LabConfig())
            assert strat.leverage_tier == "3x"
            assert strat.target_kind == "signals"


def test_all_three_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategies.leveraged_trend import (
        LeveragedTrendNasdaq100,
        LeveragedTrendRussell2000,
        LeveragedTrendSp500,
    )

    assert STRATEGY_REGISTRY["leveraged_trend_sp500"] is LeveragedTrendSp500
    assert STRATEGY_REGISTRY["leveraged_trend_nasdaq100"] is LeveragedTrendNasdaq100
    assert STRATEGY_REGISTRY["leveraged_trend_russell2000"] is LeveragedTrendRussell2000
