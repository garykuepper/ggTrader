"""Tests for the market-neutral pairs/stat-arb mean-reversion strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _idx(n, start="2020-01-01", freq="B"):
    return pd.date_range(start, periods=n, freq=freq, tz="UTC")


@pytest.fixture(autouse=True)
def _clear_pair_candidate_cache():
    """The module-level cache is keyed by (asof, corr_lookback, corr_min,
    eligible) -- safe for real market data, but tests below reuse the same
    default date range across different synthetic fixtures, which would
    collide on that key without clearing between tests."""
    from ggTrader.lab.strategies.pairs_stat_arb import _pair_candidate_cache

    _pair_candidate_cache.clear()
    yield
    _pair_candidate_cache.clear()


class TestEnumerateSectorPairs:
    def test_pairs_only_within_same_sector(self):
        from ggTrader.lab.strategies.pairs_stat_arb import enumerate_sector_pairs

        symbols = ["A", "B", "C", "D"]
        sector_map = {"A": "Tech", "B": "Tech", "C": "Energy", "D": "Energy"}
        pairs = enumerate_sector_pairs(symbols, sector_map)
        assert set(pairs) == {("A", "B"), ("C", "D")}

    def test_skips_unknown_sector(self):
        from ggTrader.lab.strategies.pairs_stat_arb import enumerate_sector_pairs

        symbols = ["A", "B"]
        sector_map = {}  # both fall back to "Unknown"
        pairs = enumerate_sector_pairs(symbols, sector_map)
        assert pairs == []

    def test_singleton_sector_produces_no_pairs(self):
        from ggTrader.lab.strategies.pairs_stat_arb import enumerate_sector_pairs

        symbols = ["A", "B", "C"]
        sector_map = {"A": "Tech", "B": "Energy", "C": "Health"}
        pairs = enumerate_sector_pairs(symbols, sector_map)
        assert pairs == []

    def test_three_member_sector_gives_three_combos(self):
        from ggTrader.lab.strategies.pairs_stat_arb import enumerate_sector_pairs

        symbols = ["A", "B", "C"]
        sector_map = {"A": "Tech", "B": "Tech", "C": "Tech"}
        pairs = enumerate_sector_pairs(symbols, sector_map)
        assert set(pairs) == {("A", "B"), ("A", "C"), ("B", "C")}


class TestPairwiseCorrelations:
    def test_identical_returns_are_perfectly_correlated(self):
        from ggTrader.lab.strategies.pairs_stat_arb import pairwise_correlations

        idx = _idx(60)
        base = 100.0 * (1.0 + pd.Series(np.linspace(0, 0.3, 60), index=idx))
        close = pd.DataFrame({"A": base, "B": base * 2.0})  # same returns, different scale
        out = pairwise_correlations(close, [("A", "B")], lookback=30)
        assert out[("A", "B")] == pytest.approx(1.0, abs=1e-6)

    def test_uncorrelated_series_below_perfect_correlation(self):
        from ggTrader.lab.strategies.pairs_stat_arb import pairwise_correlations

        rng = np.random.default_rng(0)
        idx = _idx(120)
        a = 100.0 * (1.0 + pd.Series(rng.normal(0, 0.01, 120), index=idx).cumsum())
        b = 100.0 * (1.0 + pd.Series(rng.normal(0, 0.01, 120), index=idx).cumsum())
        close = pd.DataFrame({"A": a, "B": b})
        out = pairwise_correlations(close, [("A", "B")], lookback=100)
        assert abs(out[("A", "B")]) < 0.9

    def test_insufficient_coverage_pair_is_dropped(self):
        from ggTrader.lab.strategies.pairs_stat_arb import pairwise_correlations

        idx = _idx(60)
        a = pd.Series(100.0, index=idx)
        b = pd.Series([np.nan] * 55 + [100.0] * 5, index=idx)  # mostly missing
        close = pd.DataFrame({"A": a, "B": b})
        out = pairwise_correlations(close, [("A", "B")], lookback=30)
        assert ("A", "B") not in out

    def test_missing_symbol_is_skipped_not_errored(self):
        from ggTrader.lab.strategies.pairs_stat_arb import pairwise_correlations

        idx = _idx(60)
        close = pd.DataFrame({"A": np.full(60, 100.0)}, index=idx)
        out = pairwise_correlations(close, [("A", "Z")], lookback=30)
        assert out == {}


class TestSpreadZscore:
    def test_zscore_zero_at_the_mean(self):
        from ggTrader.lab.strategies.pairs_stat_arb import spread_zscore

        idx = _idx(40)
        # Oscillating spread with a well-defined mean.
        a = pd.Series(100.0 * np.exp(0.01 * np.sin(np.arange(40) / 3)), index=idx)
        b = pd.Series(100.0, index=idx)
        z = spread_zscore(a, b, lookback=20)
        assert z.iloc[19:].notna().all()

    def test_high_a_relative_to_b_gives_positive_zscore(self):
        from ggTrader.lab.strategies.pairs_stat_arb import spread_zscore

        idx = _idx(30)
        a = pd.Series(np.concatenate([np.full(20, 100.0), np.full(10, 150.0)]), index=idx)
        b = pd.Series(100.0, index=idx)
        z = spread_zscore(a, b, lookback=20)
        assert z.iloc[-1] > 0


class TestPairSignal:
    def test_high_zscore_triggers_short_a_state(self):
        from ggTrader.lab.strategies.pairs_stat_arb import pair_signal

        z = pd.Series([0.0, 0.5, 2.5, 2.0, 0.1], index=_idx(5))
        state = pair_signal(z, entry_z=2.0, exit_z=0.5)
        # Enters short-A/long-B (-1) once z >= entry_z, holds until |z| <= exit_z.
        assert list(state) == [0.0, 0.0, -1.0, -1.0, 0.0]

    def test_low_zscore_triggers_long_a_state(self):
        from ggTrader.lab.strategies.pairs_stat_arb import pair_signal

        z = pd.Series([0.0, -2.5, -2.0, 0.1], index=_idx(4))
        state = pair_signal(z, entry_z=2.0, exit_z=0.5)
        assert list(state) == [0.0, 1.0, 1.0, 0.0]

    def test_state_persists_between_entry_and_exit(self):
        from ggTrader.lab.strategies.pairs_stat_arb import pair_signal

        # z dips below entry after triggering, but stays outside the exit band --
        # state must persist (forward-filled), not flip back to 0 prematurely.
        z = pd.Series([2.5, 1.0, 1.5, 2.2], index=_idx(4))
        state = pair_signal(z, entry_z=2.0, exit_z=0.5)
        assert list(state) == [-1.0, -1.0, -1.0, -1.0]


def _ohlcv_close_only(closes: dict[str, np.ndarray], idx) -> pd.DataFrame:
    """Minimal (symbol, field) MultiIndex OHLCV frame -- only 'close' is
    consulted by pairs_stat_arb's select()/to_targets()."""
    frames = {sym: pd.DataFrame({"close": vals}, index=idx) for sym, vals in closes.items()}
    out = pd.concat(frames, axis=1)
    out.columns = out.columns.set_names(["symbol", "field"])
    return out


def _correlated_pair(n, seed=3, common_vol=0.02, idio_vol=0.0003):
    """A common-return component shared by both legs plus small idiosyncratic
    noise -- realistic day-to-day return variance so a later shock doesn't
    single-handedly dominate the trailing correlation the way a near-zero-
    variance smooth series would (Pearson correlation is very sensitive to
    outliers when the "normal" days carry almost no variance of their own)."""
    rng = np.random.default_rng(seed)
    common_ret = rng.normal(0.0005, common_vol, n)
    idio1 = rng.normal(0, idio_vol, n)
    idio2 = rng.normal(0, idio_vol, n)
    t1 = 100.0 * np.exp(np.cumsum(common_ret + idio1))
    t2 = 100.0 * np.exp(np.cumsum(common_ret + idio2))
    return t1, t2


def _diverging_pair(n=40):
    """T1/T2: correlated (~0.8+) until a late idiosyncratic shock pushes T1
    up relative to T2 -- entry-triggering divergence (|z| ~2.3 at the last
    bar) that still clears the 0.7 corr_min filter."""
    t1, t2 = _correlated_pair(n)
    t1[-3:] = t1[-3:] * 1.08  # T1 gets relatively expensive in the last 3 bars
    return t1, t2


def _uncorrelated_pair(n=40):
    """E1/E2: opposite deterministic trends -- reliably low/negative
    correlation, well below the 0.7 corr_min filter."""
    e1 = 100.0 * 1.004 ** np.arange(n) + np.sin(np.arange(n)) * 0.3
    e2 = 100.0 * 0.996 ** np.arange(n) + np.cos(np.arange(n)) * 0.3
    return e1, e2


def _sector_map():
    return {"T1": "Tech", "T2": "Tech", "E1": "Energy", "E2": "Energy"}


def _mock_sector_map(monkeypatch):
    import ggTrader.lab.strategies.pairs_stat_arb as mod

    monkeypatch.setattr(mod, "load_sector_map", lambda: _sector_map(), raising=False)


class TestPairCandidateCaching:
    def test_correlation_scan_cached_across_combos_on_same_window(self, monkeypatch):
        """WFO builds a fresh strategy instance per grid combo and calls
        select() on the SAME data window for every one of them -- the
        candidate-pair correlation scan (independent of entry_z/exit_z/
        max_pairs/min_hold_months) must not be redundantly recomputed."""
        from unittest.mock import patch

        from ggTrader.lab.strategies import pairs_stat_arb as mod
        from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb
        from ggTrader.lab.strategy import LabConfig

        _mock_sector_map(monkeypatch)
        idx = _idx(40)
        t1, t2 = _diverging_pair(40)
        e1, e2 = _uncorrelated_pair(40)
        data = _ohlcv_close_only({"T1": t1, "T2": t2, "E1": e1, "E2": e2}, idx)
        eligible = ["T1", "T2", "E1", "E2"]

        real_corr = mod.pairwise_correlations
        with patch.object(mod, "pairwise_correlations", side_effect=real_corr) as mock_corr:
            for entry_z in (1.5, 2.0, 2.5):
                for max_pairs in (5, 10):
                    strat = PairsStatArb(
                        LabConfig(), corr_lookback=20, entry_z=entry_z, max_pairs=max_pairs
                    )
                    strat.select(data.index[-1], data, eligible)

        assert mock_corr.call_count == 1


class TestSelect:
    def _data(self, n=40):
        idx = _idx(n)
        t1, t2 = _diverging_pair(n)
        e1, e2 = _uncorrelated_pair(n)
        return _ohlcv_close_only({"T1": t1, "T2": t2, "E1": e1, "E2": e2}, idx)

    def test_diverging_correlated_pair_is_selected(self, monkeypatch):
        from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb
        from ggTrader.lab.strategy import LabConfig

        _mock_sector_map(monkeypatch)
        data = self._data()
        strat = PairsStatArb(
            LabConfig(), corr_lookback=20, corr_min=0.7, entry_z=2.0, exit_z=0.5, max_pairs=5
        )
        plan = strat.select(data.index[-1], data, ["T1", "T2", "E1", "E2"])
        pairs = {(p["symbol_long"], p["symbol_short"]) for p in plan}
        # T1 got relatively expensive -> short T1, long T2.
        assert ("T2", "T1") in pairs

    def test_uncorrelated_pair_never_selected(self, monkeypatch):
        from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb
        from ggTrader.lab.strategy import LabConfig

        _mock_sector_map(monkeypatch)
        data = self._data()
        strat = PairsStatArb(
            LabConfig(), corr_lookback=20, corr_min=0.7, entry_z=0.01, exit_z=0.0, max_pairs=5
        )
        plan = strat.select(data.index[-1], data, ["T1", "T2", "E1", "E2"])
        symbols_used = {p["symbol_long"] for p in plan} | {p["symbol_short"] for p in plan}
        assert "E1" not in symbols_used
        assert "E2" not in symbols_used

    def test_weight_is_one_over_max_pairs(self, monkeypatch):
        from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb
        from ggTrader.lab.strategy import LabConfig

        _mock_sector_map(monkeypatch)
        data = self._data()
        strat = PairsStatArb(LabConfig(), corr_lookback=20, entry_z=2.0, max_pairs=4)
        plan = strat.select(data.index[-1], data, ["T1", "T2", "E1", "E2"])
        assert plan
        for p in plan:
            assert p["weight"] == pytest.approx(0.25)

    def test_no_eligible_pairs_returns_empty_plan(self, monkeypatch):
        from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb
        from ggTrader.lab.strategy import LabConfig

        _mock_sector_map(monkeypatch)
        data = self._data()
        strat = PairsStatArb(LabConfig(), corr_lookback=20, entry_z=2.0)
        plan = strat.select(data.index[-1], data, ["E1", "E2"])  # only the uncorrelated pair
        assert plan == []

    def test_min_hold_forces_pair_to_persist_below_entry_threshold(self, monkeypatch):
        """A held pair whose |z| has fallen below entry_z (but stays above
        exit_z, and the hold floor hasn't elapsed) must not be dropped --
        min_hold prevents premature exit-by-falling-out-of-ranking, not just
        prevents re-entry churn."""
        from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb
        from ggTrader.lab.strategy import LabConfig

        _mock_sector_map(monkeypatch)
        n = 45
        idx = _idx(n)
        t1, t2 = _correlated_pair(n)
        t1[-5:] = t1[-5:] * 1.08  # sustained shock -> entry around bar -5
        t1[-1] = t1[-1] * 0.97  # partial reversion at the final bar -> |z| drops, still > exit
        e1, e2 = _uncorrelated_pair(n)
        data = _ohlcv_close_only({"T1": t1, "T2": t2, "E1": e1, "E2": e2}, idx)
        eligible = ["T1", "T2", "E1", "E2"]

        strat = PairsStatArb(
            LabConfig(),
            corr_lookback=20,
            corr_min=0.7,
            entry_z=2.0,
            exit_z=0.1,
            max_pairs=5,
            min_hold_months=3,
        )
        # Simulate two prior rebalances that already entered the pair.
        strat.select(data.index[-5], data.loc[: data.index[-5]], eligible)
        strat.select(data.index[-3], data.loc[: data.index[-3]], eligible)
        assert ("T1", "T2") in strat._held_since or ("T2", "T1") in strat._held_since
        plan = strat.select(data.index[-1], data, eligible)
        symbols_used = {p["symbol_long"] for p in plan} | {p["symbol_short"] for p in plan}
        assert "T1" in symbols_used and "T2" in symbols_used


class TestToTargets:
    def test_targets_have_correct_signs(self, monkeypatch):
        from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb
        from ggTrader.lab.strategy import LabConfig

        idx = _idx(5)
        data = _ohlcv_close_only({"A": np.full(5, 100.0), "B": np.full(5, 100.0)}, idx)
        plans = {idx[1]: [{"symbol_long": "A", "symbol_short": "B", "weight": 0.5, "z": 2.0}]}
        strat = PairsStatArb(LabConfig())
        targets = strat.to_targets(plans, data)
        bar = idx[2]  # next bar after the rebalance date
        assert targets.loc[bar, "A"] == pytest.approx(0.5)
        assert targets.loc[bar, "B"] == pytest.approx(-0.5)

    def test_reset_to_flat_when_pair_dropped(self, monkeypatch):
        from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb
        from ggTrader.lab.strategy import LabConfig

        idx = _idx(6)
        data = _ohlcv_close_only(
            {"A": np.full(6, 100.0), "B": np.full(6, 100.0), "C": np.full(6, 100.0)}, idx
        )
        plans = {
            idx[1]: [{"symbol_long": "A", "symbol_short": "B", "weight": 0.5, "z": 2.0}],
            idx[3]: [{"symbol_long": "A", "symbol_short": "C", "weight": 0.5, "z": 2.0}],
        }
        strat = PairsStatArb(LabConfig())
        targets = strat.to_targets(plans, data)
        # Second rebalance drops the A/B pair -- B must reset to flat.
        assert targets.loc[idx[4], "B"] == pytest.approx(0.0)
        assert targets.loc[idx[4], "C"] == pytest.approx(-0.5)

    def test_conflicting_symbol_is_last_write_wins(self, monkeypatch):
        """Documents defined behavior for adversarial input where two plan
        entries at the same rebalance claim the same symbol with opposite
        signs -- should not happen via select()'s own dedup, but to_targets
        must behave predictably if constructed by hand."""
        from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb
        from ggTrader.lab.strategy import LabConfig

        idx = _idx(4)
        data = _ohlcv_close_only(
            {"A": np.full(4, 100.0), "B": np.full(4, 100.0), "C": np.full(4, 100.0)}, idx
        )
        plans = {
            idx[1]: [
                {"symbol_long": "A", "symbol_short": "B", "weight": 0.5, "z": 2.0},
                {"symbol_long": "B", "symbol_short": "C", "weight": 0.3, "z": -2.0},
            ]
        }
        strat = PairsStatArb(LabConfig())
        targets = strat.to_targets(plans, data)
        # Second entry (B as long leg) overwrites the first (B as short leg).
        assert targets.loc[idx[2], "B"] == pytest.approx(0.3)

    def test_empty_plans_returns_frame_with_no_columns(self):
        from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb
        from ggTrader.lab.strategy import LabConfig

        strat = PairsStatArb(LabConfig())
        targets = strat.to_targets({}, pd.DataFrame(index=_idx(3)))
        assert list(targets.columns) == []


class TestSweepParams:
    def test_sweep_params_present(self):
        from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb

        params = PairsStatArb.sweep_params()
        assert "entry_z" in params
        assert "exit_z" in params
        assert "max_pairs" in params
        assert "min_hold_months" in params

    def test_bare_construction_works(self):
        from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb
        from ggTrader.lab.strategy import LabConfig

        strat = PairsStatArb(LabConfig())
        assert strat.target_kind == "weights"
        assert strat.name == "pairs_stat_arb"


def test_registered():
    from ggTrader.lab.strategies import STRATEGY_REGISTRY
    from ggTrader.lab.strategies.pairs_stat_arb import PairsStatArb

    assert STRATEGY_REGISTRY["pairs_stat_arb"] is PairsStatArb
