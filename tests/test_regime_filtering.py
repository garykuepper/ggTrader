"""Unit tests for regime_filtering.py — no DB required (all synthetic data)."""

import numpy as np
import pandas as pd

from ggTrader.core.orchestrator import (
    _apply_tiered_regime_mask,
    _apply_wfo_selection_gates,
    _compute_allocation_weights,
)
from ggTrader.core.regime_filtering import _compute_btc_correlations

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlcv(symbols, n_bars=300, seed=42, freq="4h"):
    """Build a synthetic MultiIndex OHLCV DataFrame for the given symbols."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_bars, freq=freq, tz="UTC")
    data = {}
    for sym in symbols:
        close = 1000.0 + np.cumsum(rng.standard_normal(n_bars) * 10)
        data[(sym, "open")] = close
        data[(sym, "high")] = close + np.abs(rng.standard_normal(n_bars) * 5)
        data[(sym, "low")] = close - np.abs(rng.standard_normal(n_bars) * 5)
        data[(sym, "close")] = close
        data[(sym, "volume")] = rng.integers(100, 1000, n_bars).astype(float)
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["symbol", "field"])
    return df


def _base_config():
    return {
        "BENCHMARK_SYMBOL": "BTC-USD",
        "EMA_WARMUP_BARS": 20,  # small for speed in tests
        "LEADER_CORR_THRESHOLD": 0.7,
    }


# ---------------------------------------------------------------------------
# _compute_btc_correlations
# ---------------------------------------------------------------------------

class TestComputeBtcCorrelations:
    def test_btc_with_itself_is_one(self):
        ohlcv = _make_ohlcv(["BTC-USD", "ETH-USD"])
        corrs = _compute_btc_correlations(ohlcv, _base_config())
        assert "BTC-USD" in corrs
        assert abs(corrs["BTC-USD"] - 1.0) < 1e-9

    def test_returns_dict_for_all_symbols(self):
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
        ohlcv = _make_ohlcv(symbols)
        corrs = _compute_btc_correlations(ohlcv, _base_config())
        for sym in symbols:
            assert sym in corrs
            assert -1.0 <= corrs[sym] <= 1.0

    def test_no_btc_returns_empty(self):
        ohlcv = _make_ohlcv(["ETH-USD", "SOL-USD"])
        corrs = _compute_btc_correlations(ohlcv, _base_config())
        assert corrs == {}

    def test_perfectly_correlated_coin(self):
        """A coin whose close is identical to BTC should have corr ≈ 1."""
        ohlcv = _make_ohlcv(["BTC-USD"])
        # Copy BTC close into a second symbol
        ohlcv[("BTC-USD", "close")]
        for field in ["open", "high", "low", "close", "volume"]:
            ohlcv[("COPY-USD", field)] = ohlcv[("BTC-USD", field)]
        ohlcv.columns = pd.MultiIndex.from_tuples(ohlcv.columns, names=["symbol", "field"])
        corrs = _compute_btc_correlations(ohlcv, _base_config())
        assert abs(corrs.get("COPY-USD", 0) - 1.0) < 1e-6



# ---------------------------------------------------------------------------
# _apply_tiered_regime_mask
# ---------------------------------------------------------------------------

class TestApplyTieredRegimeMask:
    """Tests for the BTC single-leader regime mask. Threshold = 0.7."""

    def _make_entries(self, symbols, n_bars=100):
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="4h", tz="UTC")
        data = {sym: np.ones(n_bars, dtype=bool) for sym in symbols}
        return pd.DataFrame(data, index=dates)

    def test_high_corr_coin_blocked_in_btc_bear(self):
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        btc_regime = pd.Series([True] * 60 + [False] * 40, index=dates)
        entries = self._make_entries(["XRP-USD"], n)
        # corr_btc=0.8 ≥ 0.7 → gated by BTC
        result = _apply_tiered_regime_mask(
            entries, {"XRP-USD": 0.8}, btc_regime, _base_config()
        )
        assert result.iloc[:60]["XRP-USD"].all()
        assert not result.iloc[60:]["XRP-USD"].any()

    def test_low_corr_coin_passes_through(self):
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        btc_regime = pd.Series([False] * n, index=dates)  # BTC bear throughout
        entries = self._make_entries(["XMR-USD"], n)
        # corr_btc=0.4 < 0.7 → Free tier — entries pass through
        result = _apply_tiered_regime_mask(
            entries, {"XMR-USD": 0.4}, btc_regime, _base_config()
        )
        assert result["XMR-USD"].all()

    def test_no_mask_returns_unchanged(self):
        entries = self._make_entries(["BTC-USD", "ETH-USD"])
        result = _apply_tiered_regime_mask(entries, {}, None, _base_config())
        pd.testing.assert_frame_equal(result, entries)

    def test_unknown_corr_defaults_to_filtered(self):
        """Symbols missing from corrs default to corr=1.0 (conservative — gated)."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
        btc_regime = pd.Series([False] * n, index=dates)
        entries = self._make_entries(["UNKNOWN-USD"], n)
        result = _apply_tiered_regime_mask(entries, {}, btc_regime, _base_config())
        assert not result["UNKNOWN-USD"].any()


# ---------------------------------------------------------------------------
# _compute_allocation_weights
# ---------------------------------------------------------------------------

class TestComputeAllocationWeights:
    def test_weights_sum_to_one(self):
        scores = [0.5, 0.3, 0.8, 0.2, 0.6]
        w = _compute_allocation_weights(scores, {"MAX_COIN_ALLOCATION": 0.25})
        assert abs(w.sum() - 1.0) < 1e-9

    def test_no_coin_exceeds_cap(self):
        # With 5 coins and cap=0.25, there is room to cap (5×0.25=1.25 > 1.0)
        scores = [10.0, 0.1, 0.1, 0.1, 0.1]
        w = _compute_allocation_weights(scores, {"MAX_COIN_ALLOCATION": 0.25})
        assert (w <= 0.25 + 1e-9).all()
        assert abs(w.sum() - 1.0) < 1e-9

    def test_equal_fallback_when_all_zero(self):
        scores = [0.0, 0.0, -1.0]
        w = _compute_allocation_weights(scores, {"MAX_COIN_ALLOCATION": 0.5})
        assert abs(w.sum() - 1.0) < 1e-9
        np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3], atol=1e-9)

    def test_single_coin_gets_full_weight(self):
        scores = [1.0]
        w = _compute_allocation_weights(scores, {"MAX_COIN_ALLOCATION": 0.25})
        assert abs(w[0] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# _apply_wfo_selection_gates
# ---------------------------------------------------------------------------

class TestApplyWfoSelectionGates:
    def _coin(self, rob=0.5, oos=0.4, fc=0.5, strategy="psar_adx"):
        return {
            "best_strategy": strategy,
            "best_exit": "atr_trailing",
            "best_params": {},
            "robustness_score": rob,
            "oos_robustness_score": oos,
            "fold_consistency": fc,
        }

    def test_min_robustness_drops_low_coins(self):
        results = {
            "BTC-USD": self._coin(rob=0.5),
            "ETH-USD": self._coin(rob=0.05),  # below gate
        }
        filtered = _apply_wfo_selection_gates(results, {"MIN_ROBUSTNESS_SCORE": 0.1})
        assert "BTC-USD" in filtered
        assert "ETH-USD" not in filtered

    def test_min_fold_consistency_drops_inconsistent(self):
        results = {
            "BTC-USD": self._coin(fc=0.6),
            "SOL-USD": self._coin(fc=0.1),  # below gate
        }
        filtered = _apply_wfo_selection_gates(results, {"MIN_FOLD_CONSISTENCY": 0.33})
        assert "BTC-USD" in filtered
        assert "SOL-USD" not in filtered

    def test_diversity_cap_keeps_top_oos(self):
        """With MAX_COINS_PER_STRATEGY=1, the coin with higher OOS is kept."""
        results = {
            "BTC-USD": self._coin(oos=0.9, strategy="psar_adx"),
            "ETH-USD": self._coin(oos=0.2, strategy="psar_adx"),
        }
        filtered = _apply_wfo_selection_gates(results, {"MAX_COINS_PER_STRATEGY": 1})
        assert "BTC-USD" in filtered
        assert "ETH-USD" not in filtered

    def test_all_gates_disabled_returns_unchanged(self):
        results = {
            "BTC-USD": self._coin(),
            "ETH-USD": self._coin(rob=-5.0, fc=0.0),  # would fail all gates if enabled
        }
        filtered = _apply_wfo_selection_gates(results, {})
        assert filtered == results

    def test_does_not_mutate_input(self):
        results = {"BTC-USD": self._coin(rob=0.5)}
        original = dict(results)
        _apply_wfo_selection_gates(results, {"MIN_ROBUSTNESS_SCORE": 0.9})
        assert results == original

