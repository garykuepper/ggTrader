"""Tests for the daily signal runner."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ggTrader.paper.gates import SanityConfig, SignalGateError

#: The synthetic panels below have 1-20 symbols and 120 bars: only the structural gates apply.
PERMISSIVE = SanityConfig.permissive()


@pytest.fixture(autouse=True)
def _permissive_default_sanity():
    """Small synthetic panels: swap the production bounds for the structural-only ones."""
    with patch("ggTrader.paper.signal_runner.SANITY", PERMISSIVE):
        yield


@pytest.fixture(autouse=True)
def _stub_engine_selfcheck():
    """The real self-check runs a Rust probe (~0.5 s); `TestEngineSelfcheck` exercises it."""
    with patch(
        "ggTrader.paper.signal_runner.engine_selfcheck.selfcheck",
        return_value={"engine": "rust", "vectorbt": "1.2.0", "vectorbt_rust": "1.2.0"},
    ):
        yield


def _mock_ohlcv(symbols: list[str], n_days: int = 120, end: str = "2026-06-19") -> pd.DataFrame:
    """Create a fake MultiIndex OHLCV DataFrame."""
    dates = pd.bdate_range(end=end, periods=n_days, tz="UTC")
    np.random.seed(42)
    frames = {}
    for sym in symbols:
        price = 100.0 * np.exp(np.random.randn(n_days).cumsum() * 0.02)
        frames[sym] = pd.DataFrame(
            {
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1e6,
            },
            index=dates,
        )
    df = pd.concat(frames, axis=1)
    df.columns.names = ["symbol", "field"]
    return df


class TestGenerateSignals:
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_returns_buys_and_sells(self, mock_fetch, mock_members):
        symbols = ["AAPL", "MSFT", "GOOG"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)

        from ggTrader.paper.signal_runner import generate_signals

        result = generate_signals(lookback_days=120, sanity=PERMISSIVE)

        assert "buys" in result
        assert "sells" in result
        assert "as_of" in result
        assert "universe_size" in result
        assert isinstance(result["buys"], list)
        assert isinstance(result["sells"], list)
        assert result["universe_size"] == 3

    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_buys_and_sells_are_disjoint(self, mock_fetch, mock_members):
        symbols = [f"SYM{i}" for i in range(20)]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols, n_days=120)

        from ggTrader.paper.signal_runner import generate_signals

        result = generate_signals(lookback_days=120, sanity=PERMISSIVE)

        buys = set(result["buys"])
        sells = set(result["sells"])
        assert buys.isdisjoint(sells), "A symbol cannot be both a buy and sell"

    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_last_exit_covers_every_bar_not_just_the_last(self, mock_fetch, mock_members):
        symbols = [f"SYM{i}" for i in range(20)]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols, n_days=120)

        from ggTrader.paper.signal_runner import generate_signals

        result = generate_signals(lookback_days=120, sanity=PERMISSIVE)

        last_exit = result["last_exit"]
        # A random walk over 120 bars crosses RSI 50 for most names, almost
        # all of them before the final bar -- the ones live used to miss.
        assert len(last_exit) > len(result["sells"])
        assert all(d <= result["as_of"] for d in last_exit.values())
        assert all(last_exit[s] == result["as_of"] for s in result["sells"])

    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_empty_data_fails_closed(self, mock_fetch, mock_members):
        mock_members.return_value = ["AAPL"]
        empty = pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(
                [
                    ("AAPL", "open"),
                    ("AAPL", "high"),
                    ("AAPL", "low"),
                    ("AAPL", "close"),
                    ("AAPL", "volume"),
                ],
                names=["symbol", "field"],
            )
        )
        mock_fetch.return_value = empty

        from ggTrader.paper.signal_runner import generate_signals

        # Fail closed: an empty frame used to return "no signals" silently (and the trader's
        # freshness gate then passed); it is now a gate failure that stops the run and alerts.
        with pytest.raises(SignalGateError, match="empty_frame"):
            generate_signals(lookback_days=120, sanity=PERMISSIVE)

    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_as_of_is_last_bar_date(self, mock_fetch, mock_members):
        symbols = ["AAPL"]
        mock_members.return_value = symbols
        ohlcv = _mock_ohlcv(symbols)
        mock_fetch.return_value = ohlcv

        from ggTrader.paper.signal_runner import generate_signals

        result = generate_signals(lookback_days=120, sanity=PERMISSIVE)

        expected_date = str(ohlcv.index[-1].date())
        assert result["as_of"] == expected_date

    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_universe_param_passed_through(self, mock_fetch, mock_members):
        symbols = ["AAPL", "MSFT", "GOOG"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)

        from ggTrader.paper.signal_runner import generate_signals

        result = generate_signals(universe="midcap400", lookback_days=120, sanity=PERMISSIVE)

        mock_members.assert_called_once()
        assert mock_members.call_args[0][0] == "midcap400"
        assert result["universe_size"] == 3


class TestGenerateBlendedSignals:
    @patch("ggTrader.paper.signal_runner.save_rebalance_state")
    @patch("ggTrader.paper.signal_runner.compute_weights_and_scale")
    @patch("ggTrader.paper.signal_runner.compute_sleeve_curve")
    @patch("ggTrader.paper.signal_runner.get_rebalance_state")
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_first_run_rebalances_and_returns_all_sleeves(
        self, mock_fetch, mock_members, mock_get_state, mock_curve, mock_weights, mock_save
    ):
        symbols = ["AAPL", "MSFT"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)
        mock_get_state.return_value = None  # no prior rebalance
        mock_curve.return_value = pd.Series([1.0, 1.01, 1.02])
        mock_weights.return_value = ({"sp500": 0.4, "midcap400": 0.3, "nasdaq100": 0.3}, 0.9)

        from ggTrader.paper.signal_runner import generate_blended_signals

        result = generate_blended_signals()

        assert set(result["sleeves"]) == {"sp500", "midcap400", "nasdaq100"}
        assert result["weights"] == {"sp500": 0.4, "midcap400": 0.3, "nasdaq100": 0.3}
        assert result["scale"] == 0.9
        assert result["rebalanced_today"] is True
        assert result["fallback_used"] is False
        mock_save.assert_called_once()

    @patch("ggTrader.paper.signal_runner.save_rebalance_state")
    @patch("ggTrader.paper.signal_runner.compute_weights_and_scale")
    @patch("ggTrader.paper.signal_runner.compute_sleeve_curve")
    @patch("ggTrader.paper.signal_runner.get_rebalance_state")
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_mid_month_reuses_stored_weights(
        self, mock_fetch, mock_members, mock_get_state, mock_curve, mock_weights, mock_save
    ):
        symbols = ["AAPL", "MSFT"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)
        today_str = str(pd.Timestamp.now(tz="UTC").normalize().date())
        mock_get_state.return_value = {
            "rebalance_date": today_str,
            "weights": {"sp500": 0.5, "midcap400": 0.25, "nasdaq100": 0.25},
            "scale": 0.8,
        }

        from ggTrader.paper.signal_runner import generate_blended_signals

        result = generate_blended_signals()

        assert result["weights"] == {"sp500": 0.5, "midcap400": 0.25, "nasdaq100": 0.25}
        assert result["scale"] == 0.8
        assert result["rebalanced_today"] is False
        mock_curve.assert_not_called()
        mock_save.assert_not_called()

    @patch("ggTrader.paper.signal_runner.save_rebalance_state")
    @patch("ggTrader.paper.signal_runner.compute_weights_and_scale")
    @patch("ggTrader.paper.signal_runner.compute_sleeve_curve")
    @patch("ggTrader.paper.signal_runner.get_rebalance_state")
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_rebalance_fetch_failure_falls_back_to_stored_weights(
        self, mock_fetch, mock_members, mock_get_state, mock_curve, mock_weights, mock_save
    ):
        symbols = ["AAPL", "MSFT"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)
        mock_get_state.return_value = {
            "rebalance_date": "2026-05-01",  # stale -- would normally trigger a rebalance
            "weights": {"sp500": 0.6, "midcap400": 0.2, "nasdaq100": 0.2},
            "scale": 0.7,
        }
        mock_curve.side_effect = RuntimeError("OHLCV fetch failed")

        from ggTrader.paper.signal_runner import generate_blended_signals

        result = generate_blended_signals()

        assert result["weights"] == {"sp500": 0.6, "midcap400": 0.2, "nasdaq100": 0.2}
        assert result["scale"] == 0.7
        assert result["fallback_used"] is True
        mock_save.assert_not_called()

    @patch("ggTrader.paper.signal_runner.save_rebalance_state")
    @patch("ggTrader.paper.signal_runner.compute_weights_and_scale")
    @patch("ggTrader.paper.signal_runner.compute_sleeve_curve")
    @patch("ggTrader.paper.signal_runner.get_rebalance_state")
    @patch("ggTrader.paper.signal_runner.universe_members_asof")
    @patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv")
    def test_first_run_rebalance_failure_raises(
        self, mock_fetch, mock_members, mock_get_state, mock_curve, mock_weights, mock_save
    ):
        symbols = ["AAPL", "MSFT"]
        mock_members.return_value = symbols
        mock_fetch.return_value = _mock_ohlcv(symbols)
        mock_get_state.return_value = None  # no prior rebalance -- nothing to fall back to
        mock_curve.side_effect = RuntimeError("OHLCV fetch failed")

        from ggTrader.paper.signal_runner import generate_blended_signals

        with pytest.raises(RuntimeError, match="OHLCV fetch failed"):
            generate_blended_signals()

        mock_save.assert_not_called()


class TestGenerateCoreSignals:
    @patch("ggTrader.paper.signal_runner.generate_signals")
    def test_wraps_sp500_signals_in_blend_shape(self, mock_generate):
        mock_generate.return_value = {
            "buys": ["AAPL", "MSFT"],
            "sells": ["TSLA"],
            "as_of": "2026-09-10",
            "universe_size": 503,
        }

        from ggTrader.paper.signal_runner import generate_core_signals

        result = generate_core_signals()

        mock_generate.assert_called_once_with(universe="sp500", persist_tape=True, membership=None)
        assert result["sleeves"] == {"sp500": mock_generate.return_value}
        assert result["weights"] == {"sp500": 1.0}
        assert result["scale"] == 1.0
        assert result["rebalanced_today"] is False
        assert result["fallback_used"] is False

    @patch("ggTrader.paper.signal_runner.generate_signals")
    def test_the_chosen_membership_source_is_passed_down(self, mock_generate):
        from ggTrader.paper.signal_runner import generate_core_signals

        source = object()
        mock_generate.return_value = {"buys": [], "sells": [], "as_of": "2026-09-16"}
        generate_core_signals(persist_tape=False, membership=source)
        mock_generate.assert_called_once_with(
            universe="sp500", persist_tape=False, membership=source
        )

    @patch("ggTrader.paper.signal_runner.generate_signals")
    def test_core_weights_restore_full_slot_size(self, mock_generate):
        """weights=1.0, scale=1.0 must give sleeve_position_notional ==
        position_notional (3.3% of PV), i.e. no blend-overlay shrink."""
        from ggTrader.paper.signal_runner import generate_core_signals

        mock_generate.return_value = {
            "buys": [],
            "sells": [],
            "as_of": "2026-09-16",
            "universe_size": 503,
        }
        core = generate_core_signals()
        from ggTrader.paper.risk import RiskConfig, RiskGuard

        guard = RiskGuard(RiskConfig())
        pv = 102_000.0
        sleeve = guard.sleeve_position_notional(pv, core["weights"]["sp500"], core["scale"])
        assert sleeve == guard.position_notional(pv)
        assert sleeve == pytest.approx(pv * 0.033, abs=0.01)


class TestRefreshBenchmarkTape:
    @patch("ggTrader.data.live.cached_yfinance_loader.CachedYFinanceLoader")
    @patch("ggTrader.data.live.yfinance_loader.YFinanceDataLoader")
    def test_refetches_trailing_window_and_upserts(self, mock_plain_cls, mock_cached_cls):
        from ggTrader.paper.signal_runner import BENCHMARK_SYMBOLS, refresh_benchmark_tape

        frame = _mock_ohlcv(list(BENCHMARK_SYMBOLS), n_days=30)
        mock_plain_cls.return_value.fetch_ohlcv.return_value = frame

        got = refresh_benchmark_tape(lookback_days=30)

        args, kwargs = mock_plain_cls.return_value.fetch_ohlcv.call_args
        assert sorted(args[0]) == sorted(BENCHMARK_SYMBOLS)
        assert args[1] == "1d"
        assert (kwargs["end_date"] - kwargs["start_date"]).days == 30
        mock_cached_cls.return_value._cache_to_db.assert_called_once_with(frame, "1d")
        assert sorted(got) == sorted(BENCHMARK_SYMBOLS)

    @patch("ggTrader.data.live.cached_yfinance_loader.CachedYFinanceLoader")
    @patch("ggTrader.data.live.yfinance_loader.YFinanceDataLoader")
    def test_never_raises_on_fetch_error(self, mock_plain_cls, mock_cached_cls):
        from ggTrader.paper.signal_runner import refresh_benchmark_tape

        mock_plain_cls.return_value.fetch_ohlcv.side_effect = RuntimeError("yf down")

        assert refresh_benchmark_tape() == []
        mock_cached_cls.return_value._cache_to_db.assert_not_called()

    @patch("ggTrader.data.live.cached_yfinance_loader.CachedYFinanceLoader")
    @patch("ggTrader.data.live.yfinance_loader.YFinanceDataLoader")
    def test_never_raises_on_db_write_error(self, mock_plain_cls, mock_cached_cls):
        from ggTrader.paper.signal_runner import refresh_benchmark_tape

        mock_plain_cls.return_value.fetch_ohlcv.return_value = _mock_ohlcv(
            ["SPY", "TLT"], n_days=30
        )
        mock_cached_cls.return_value._cache_to_db.side_effect = RuntimeError("db down")

        assert refresh_benchmark_tape() == []

    @patch("ggTrader.data.live.cached_yfinance_loader.CachedYFinanceLoader")
    @patch("ggTrader.data.live.yfinance_loader.YFinanceDataLoader")
    def test_reports_only_symbols_that_returned_data(self, mock_plain_cls, mock_cached_cls):
        from ggTrader.paper.signal_runner import refresh_benchmark_tape

        mock_plain_cls.return_value.fetch_ohlcv.return_value = _mock_ohlcv(
            ["SPY", "TLT"], n_days=30
        )

        assert sorted(refresh_benchmark_tape()) == ["SPY", "TLT"]


def test_default_lookback_covers_indicator_warmup():
    """365 calendar days (~252 bars) gives 0 entry/exit mismatches vs the
    long-history backtest; 120 days mismatched 4.0% / 3.4%
    (research/paper-system/indicator-warmup/2026-09-28-live-indicator-warmup.md)."""
    import inspect

    from ggTrader.paper.signal_runner import generate_signals

    assert inspect.signature(generate_signals).parameters["lookback_days"].default >= 365


# ---------------------------------------------------------------------------------------------
# P7: the signal path on the vectorbt engine, fail closed
# ---------------------------------------------------------------------------------------------

from ggTrader.engine import selfcheck as _engine_selfcheck  # noqa: E402

_REAL_SELFCHECK = _engine_selfcheck.selfcheck  # captured before the autouse stub replaces it
_NOW = pd.Timestamp("2026-06-22 19:45", tz="UTC").to_pydatetime()  # Monday 12:45 PT


def _universe_patches(symbols, ohlcv):
    return (
        patch("ggTrader.paper.signal_runner.universe_members_asof", return_value=symbols),
        patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv", return_value=ohlcv),
    )


class TestEngineSelfcheck:
    def test_the_real_selfcheck_passes_and_reports_rust(self):
        """The trader's start-up probe on this box: Rust dispatched, versions paired."""
        from ggTrader.paper import signal_runner

        with patch.object(_engine_selfcheck, "selfcheck", _REAL_SELFCHECK):
            result = signal_runner.require_rust_engine()
        assert result["engine"] == "rust" and result["rust_calls"] >= 1
        assert result["vectorbt"].split(".")[:2] == result["vectorbt_rust"].split(".")[:2]

    def test_selfcheck_failure_stops_the_run_before_any_data_is_fetched(self):
        from ggTrader.paper.signal_runner import generate_signals

        members, fetch = _universe_patches(["AAPL"], _mock_ohlcv(["AAPL"]))
        with (
            members,
            fetch as mock_fetch,
            patch(
                "ggTrader.paper.signal_runner.engine_selfcheck.selfcheck",
                side_effect=RuntimeError("version mismatch"),
            ),
        ):
            with pytest.raises(SignalGateError, match="engine_selfcheck.*version mismatch"):
                generate_signals(lookback_days=120)
        mock_fetch.assert_not_called()

    def test_a_non_rust_engine_is_refused_even_if_the_probe_passes(self):
        from ggTrader.paper.signal_runner import generate_signals

        with patch(
            "ggTrader.paper.signal_runner.engine_selfcheck.selfcheck",
            return_value={"engine": "numba"},
        ):
            with pytest.raises(SignalGateError, match="engine_not_rust"):
                generate_signals(lookback_days=120)


class TestSignalPathOnTheEngine:
    def test_the_live_defaults_grid_is_what_runs(self):
        """OD14: one combination, divergence window 20, volume multiple 2.0, min_agree 2."""
        from ggTrader.lab.strategies.ensemble import Ensemble
        from ggTrader.paper.signal_runner import generate_signals

        symbols = ["AAPL", "MSFT"]
        members, fetch = _universe_patches(symbols, _mock_ohlcv(symbols))
        real_build, grids = Ensemble.build, []

        def spy(self, bars, grid=None):
            grids.append(grid)
            return real_build(self, bars, grid)

        with members, fetch, patch.object(Ensemble, "build", spy):
            generate_signals(lookback_days=120, sanity=PERMISSIVE)
        (grid,) = grids
        (combo,) = grid.combos()
        assert combo["divergence_window"] == 20 and combo["vol_mult"] == 2.0
        assert combo["min_agree"] == 2 and combo["min_agree_exit"] == 2

    def test_an_engine_exception_propagates_there_is_no_legacy_fallback(self):
        from ggTrader.paper import signal_runner

        assert not hasattr(signal_runner, "EnsembleSignal")
        symbols = ["AAPL", "MSFT"]
        members, fetch = _universe_patches(symbols, _mock_ohlcv(symbols))
        with (
            members,
            fetch,
            patch.object(signal_runner, "live_ensemble_frames", side_effect=ValueError("rust")),
        ):
            with pytest.raises(ValueError, match="rust"):
                signal_runner.generate_signals(lookback_days=120, sanity=PERMISSIVE)

    def test_production_bounds_reject_a_small_universe(self):
        from ggTrader.paper.signal_runner import generate_signals

        symbols = [f"S{i}" for i in range(20)]
        members, fetch = _universe_patches(symbols, _mock_ohlcv(symbols, n_days=300))
        with members, fetch:
            with pytest.raises(SignalGateError, match="small_universe"):
                generate_signals(lookback_days=365, sanity=SanityConfig())

    def test_no_symbol_with_enough_history_fails_closed(self):
        from ggTrader.paper.signal_runner import generate_signals

        symbols = ["AAPL", "MSFT"]
        members, fetch = _universe_patches(symbols, _mock_ohlcv(symbols, n_days=40))
        with members, fetch:
            with pytest.raises(SignalGateError, match="no_eligible_symbols"):
                generate_signals(lookback_days=120, sanity=PERMISSIVE)

    def test_absurd_signal_counts_fail_closed(self):
        from ggTrader.paper.signal_runner import generate_signals

        symbols = [f"S{i}" for i in range(20)]
        members, fetch = _universe_patches(symbols, _mock_ohlcv(symbols))
        tiny = SanityConfig(
            min_bars=1,
            min_universe=1,
            min_loaded_share=0.0,
            min_last_bar_coverage=0.0,
            max_buys=0,
            max_sells=0,
        )
        with members, fetch:
            with pytest.raises(SignalGateError, match="absurd_signal_count"):
                generate_signals(lookback_days=120, sanity=tiny)

    def test_a_thin_bar_dated_today_is_dropped_and_signals_use_the_previous_session(self):
        """The 1-of-646 case: today's bar is not complete yet, yesterday's is."""
        from ggTrader.paper.signal_runner import generate_signals

        symbols = [f"S{i:02d}" for i in range(40)]
        ohlcv = _mock_ohlcv(symbols, n_days=120, end="2026-06-22")  # last bar = `_NOW`'s ET date
        ohlcv.iloc[-1, :] = np.nan
        ohlcv.loc[ohlcv.index[-1], ("S00", slice(None))] = 100.0
        members, fetch = _universe_patches(symbols, ohlcv)
        strict_cov = SanityConfig(
            min_bars=50, min_universe=20, min_loaded_share=0.9, min_last_bar_coverage=0.95
        )
        with members, fetch:
            result = generate_signals(lookback_days=120, sanity=strict_cov, now=_NOW)
        assert result["bars"]["trimmed_partial_bar"] is True
        assert result["bars"]["dropped_bar"] == "2026-06-22"
        assert result["as_of"] == "2026-06-19"  # the Friday before that Monday

    def test_a_thin_bar_of_an_older_day_is_a_bad_tape(self):
        from ggTrader.paper.signal_runner import generate_signals

        symbols = [f"S{i:02d}" for i in range(40)]
        ohlcv = _mock_ohlcv(symbols, n_days=120)  # last bar 2026-06-19, before `_NOW`
        ohlcv.iloc[-1, :] = np.nan
        members, fetch = _universe_patches(symbols, ohlcv)
        strict_cov = SanityConfig(
            min_bars=50, min_universe=20, min_loaded_share=0.9, min_last_bar_coverage=0.95
        )
        with members, fetch:
            with pytest.raises(SignalGateError, match="last_bar_coverage"):
                generate_signals(lookback_days=120, sanity=strict_cov, now=_NOW)

    def test_result_carries_the_engine_and_bar_facts_for_the_run_log(self):
        from ggTrader.paper.signal_runner import generate_signals

        symbols = ["AAPL", "MSFT", "GOOG"]
        members, fetch = _universe_patches(symbols, _mock_ohlcv(symbols))
        with members, fetch:
            result = generate_signals(lookback_days=120, sanity=PERMISSIVE)
        assert result["engine"]["engine"] == "rust"
        assert result["bars"]["n_bars"] == 120 and result["bars"]["eligible"] == 3
        assert result["bars"]["trimmed_partial_bar"] is False


class TestTapeIsNotWrittenWhenAskedNot:
    def test_persist_tape_false_never_touches_the_caching_loader(self):
        """The read-only dry run: bars come straight from yfinance and stay in memory."""
        from ggTrader.paper.signal_runner import generate_signals

        symbols = ["AAPL", "MSFT"]
        ohlcv = _mock_ohlcv(symbols)
        with (
            patch("ggTrader.paper.signal_runner.universe_members_asof", return_value=symbols),
            patch("ggTrader.paper.signal_runner.fetch_stock_ohlcv") as cached_path,
            patch("ggTrader.data.live.cached_yfinance_loader.CachedYFinanceLoader") as cached_cls,
            patch("ggTrader.data.live.yfinance_loader.YFinanceDataLoader") as plain_cls,
        ):
            plain_cls.return_value.fetch_ohlcv.return_value = ohlcv
            result = generate_signals(lookback_days=120, sanity=PERMISSIVE, persist_tape=False)
        cached_path.assert_not_called()
        cached_cls.assert_not_called()
        plain_cls.return_value.fetch_ohlcv.assert_called_once()
        assert result["universe_size"] == 2

    def test_persist_tape_true_is_the_existing_loader_chain(self):
        from ggTrader.paper.signal_runner import generate_signals

        symbols = ["AAPL", "MSFT"]
        members, fetch = _universe_patches(symbols, _mock_ohlcv(symbols))
        with members, fetch as cached_path:
            generate_signals(lookback_days=120, sanity=PERMISSIVE)
        cached_path.assert_called_once()
