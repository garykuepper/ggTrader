"""L1/L3b: each indicator preset against an independent numpy reference (rtol 1e-9).

Also pins the legacy equivalence: the configured presets reproduce ggTrader's pandas
indicators (the formulas below are copied into the test, so they outlive the legacy module).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.engine import indicators as ind

from .panels import (
    make_panel,
    ref_atr_sma,
    ref_bbands,
    ref_ema,
    ref_sma,
    ref_wilder_rsi,
)

RTOL = 1e-9


@pytest.fixture(scope="module")
def panel() -> dict[str, pd.DataFrame]:
    return make_panel(seed=11, n_bars=600, n_syms=4)


def _col(frame: pd.DataFrame, j: int) -> np.ndarray:
    return frame.iloc[:, j].to_numpy(dtype=float)


def _assert_valid_close(actual: np.ndarray, expected: np.ndarray, start: int) -> None:
    """Both NaN before ``start``; equal to rtol 1e-9 from ``start`` on."""
    assert np.isnan(actual[:start]).all()
    np.testing.assert_allclose(actual[start:], expected[start:], rtol=RTOL, atol=1e-12)


# --- Wilder RSI -----------------------------------------------------------------------


@pytest.mark.parametrize("n", [5, 14])
def test_wilder_rsi_matches_reference(engine: str, panel, n: int) -> None:
    out = ind.wilder_rsi(panel["close"], n).rsi
    for j in range(panel["close"].shape[1]):
        _assert_valid_close(
            out.iloc[:, j].to_numpy(), ref_wilder_rsi(_col(panel["close"], j), n), 2 * n - 1
        )


def test_wilder_rsi_is_configured_as_window_2n_minus_1_ewm(panel) -> None:
    out = ind.wilder_rsi(panel["close"], 14)
    assert list(out.rsi.columns.get_level_values("rsi_window").unique()) == [27]
    assert out.rsi.notna().to_numpy().argmax(axis=0).tolist() == [27] * 4  # first valid bar


def test_wilder_rsi_accepts_a_grid_of_periods(panel) -> None:
    out = ind.wilder_rsi(panel["close"], [5, 14])
    assert list(out.rsi.columns.get_level_values("rsi_window").unique()) == [9, 27]


def test_wilder_rsi_zero_loss_is_100() -> None:
    rising = pd.DataFrame({"x": np.arange(1.0, 80.0)})
    tail = ind.wilder_rsi(rising, 14).rsi.iloc[30:, 0]
    assert (tail == 100.0).all()


def test_wilder_rsi_equals_legacy_pandas_form(panel) -> None:
    """Legacy: delta.ewm(alpha=1/n, min_periods=n, adjust=False); equal wherever both valid."""
    close = panel["close"]
    delta = close.diff()
    gain, loss = delta.clip(lower=0.0), -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    legacy = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss.replace(0, np.nan))
    new = ind.wilder_rsi(close, 14).rsi
    np.testing.assert_allclose(new.to_numpy()[27:], legacy.to_numpy()[27:], rtol=RTOL, atol=1e-10)


# --- Bollinger bands ------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [2.0, 2.5])
def test_bbands_use_sample_std_ddof_1(engine: str, panel, alpha: float) -> None:
    out = ind.bbands(panel["close"], window=20, alpha=alpha)
    for j in range(panel["close"].shape[1]):
        mid, up, lo = ref_bbands(_col(panel["close"], j), 20, alpha, ddof=1)
        _assert_valid_close(out.middle.iloc[:, j].to_numpy(), mid, 19)
        _assert_valid_close(out.upper.iloc[:, j].to_numpy(), up, 19)
        _assert_valid_close(out.lower.iloc[:, j].to_numpy(), lo, 19)


def test_bbands_ddof_1_differs_from_the_vbt_default_ddof_0(panel) -> None:
    """The preset is a different strategy from vectorbt's default; the gap is not noise."""
    mid, up0, lo0 = ref_bbands(_col(panel["close"], 0), 20, 2.0, ddof=0)
    mid, up1, lo1 = ref_bbands(_col(panel["close"], 0), 20, 2.0, ddof=1)
    out = ind.bbands(panel["close"], 20, 2.0).lower.iloc[:, 0].to_numpy()
    assert np.nanmax(np.abs(out - lo1)) < 1e-9
    assert np.nanmax(np.abs(out - lo0)) > 1e-2


def test_bbands_equal_legacy_pandas_rolling_std(panel) -> None:
    close = panel["close"]
    legacy_lower = (
        close.rolling(20, min_periods=20).mean() - 2.0 * close.rolling(20, min_periods=20).std()
    )
    new = ind.bbands(close, 20, 2.0).lower
    np.testing.assert_allclose(
        new.to_numpy()[19:], legacy_lower.to_numpy()[19:], rtol=RTOL, atol=1e-9
    )


# --- moving averages ------------------------------------------------------------------


def test_sma_matches_reference(engine: str, panel) -> None:
    out = ind.sma(panel["close"], 20).ma
    for j in range(panel["close"].shape[1]):
        _assert_valid_close(out.iloc[:, j].to_numpy(), ref_sma(_col(panel["close"], j), 20), 19)


@pytest.mark.parametrize("window", [10, 50])
def test_ema_matches_reference_and_pads_window_minus_1_with_nan(
    engine: str, panel, window: int
) -> None:
    out = ind.ema(panel["close"], window).ma
    for j in range(panel["close"].shape[1]):
        _assert_valid_close(
            out.iloc[:, j].to_numpy(), ref_ema(_col(panel["close"], j), window), window - 1
        )


def test_ema_equals_legacy_pandas_ewm_adjust_false(panel) -> None:
    legacy = panel["close"].ewm(span=20, adjust=False).mean()
    new = ind.ema(panel["close"], 20).ma
    np.testing.assert_allclose(new.to_numpy()[19:], legacy.to_numpy()[19:], rtol=RTOL, atol=1e-12)


def test_ema_grid_of_windows(panel) -> None:
    out = ind.ema(panel["close"], [10, 20]).ma
    assert list(out.columns.get_level_values("ma_window").unique()) == [10, 20]


# --- MACD -----------------------------------------------------------------------------


def test_macd_is_the_ema_form(engine: str, panel) -> None:
    out = ind.macd(panel["close"], 12, 26, 9)
    for j in range(panel["close"].shape[1]):
        x = _col(panel["close"], j)
        line = ref_ema(x, 12) - ref_ema(x, 26)
        _assert_valid_close(out.macd.iloc[:, j].to_numpy(), line, 25)
        # vbt seeds the signal EMA at the first valid MACD bar (index 25) and masks 8 more bars
        signal = ref_ema(line, 9, seed_at=25)
        _assert_valid_close(out.signal.iloc[:, j].to_numpy(), signal, 33)
        _assert_valid_close(out.hist.iloc[:, j].to_numpy(), line - signal, 33)


def test_macd_equals_legacy_pandas_after_200_bars(panel) -> None:
    """Legacy seeds the signal at bar 0; the seed gap decays below 1e-9 by bar ~200."""
    close = panel["close"]
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = line.ewm(span=9, adjust=False).mean()
    new = ind.macd(close, 12, 26, 9)
    np.testing.assert_allclose(
        new.macd.to_numpy()[25:], line.to_numpy()[25:], rtol=RTOL, atol=1e-12
    )
    np.testing.assert_allclose(
        new.hist.to_numpy()[200:], (line - signal).to_numpy()[200:], rtol=RTOL, atol=1e-9
    )


# --- ATR ------------------------------------------------------------------------------


@pytest.mark.parametrize("window", [14, 20])
def test_atr_is_the_simple_average_of_true_range(engine: str, panel, window: int) -> None:
    out = ind.atr(panel["high"], panel["low"], panel["close"], window)
    for j in range(panel["close"].shape[1]):
        ref = ref_atr_sma(
            _col(panel["high"], j), _col(panel["low"], j), _col(panel["close"], j), window
        )
        _assert_valid_close(out.atr.iloc[:, j].to_numpy(), ref, window - 1)


def test_atr_equals_legacy_rolling_mean_one_bar_later(panel) -> None:
    """Legacy has NaN true range on bar 0 so its first ATR is one bar after vbt's."""
    high, low, close = panel["high"], panel["low"], panel["close"]
    prev = close.shift(1)
    tr = pd.DataFrame(
        np.maximum(np.maximum(high - low, (high - prev).abs()), (low - prev).abs()),
        index=close.index,
        columns=close.columns,
    )
    legacy = tr.rolling(14, min_periods=14).mean()
    new = ind.atr(high, low, close, 14).atr
    np.testing.assert_allclose(new.to_numpy()[14:], legacy.to_numpy()[14:], rtol=RTOL, atol=1e-12)
    assert new.iloc[13].notna().all() and legacy.iloc[13].isna().all()


# --- structure ------------------------------------------------------------------------


def test_presets_return_vectorbt_indicator_objects(panel) -> None:
    for obj, outputs in (
        (ind.wilder_rsi(panel["close"], 14), ("rsi",)),
        (ind.bbands(panel["close"], 20, 2.0), ("middle", "upper", "lower")),
        (ind.ema(panel["close"], 20), ("ma",)),
        (ind.sma(panel["close"], 20), ("ma",)),
        (ind.macd(panel["close"]), ("macd", "signal", "hist")),
        (ind.atr(panel["high"], panel["low"], panel["close"]), ("tr", "atr")),
    ):
        for name in outputs:
            assert isinstance(getattr(obj, name), pd.DataFrame)


def test_presets_hide_the_constant_behaviour_flags_from_the_columns(panel) -> None:
    """Level names carry only what varies (window, alpha), not the pinned ewm flags."""
    assert "rsi_ewm" not in ind.wilder_rsi(panel["close"], 14).rsi.columns.names
    assert "bb_ewm" not in ind.bbands(panel["close"], 20, 2.0).middle.columns.names
    assert "macd_macd_ewm" not in ind.macd(panel["close"]).macd.columns.names


def test_window_must_be_positive_integers(panel) -> None:
    with pytest.raises(ValueError):
        ind.wilder_rsi(panel["close"], 0)
    with pytest.raises(ValueError):
        ind.ema(panel["close"], [10, -1])


# --- numba vs Rust (L3b) --------------------------------------------------------------


def test_numba_and_rust_agree_on_every_preset(panel) -> None:
    from ggTrader.engine import config

    results = {}
    for name in ("rust", "numba"):
        config.set_engine(name)
        results[name] = [
            ind.wilder_rsi(panel["close"], 14).rsi,
            ind.bbands(panel["close"], 20, 2.0).lower,
            ind.ema(panel["close"], 20).ma,
            ind.sma(panel["close"], 20).ma,
            ind.macd(panel["close"]).hist,
            ind.atr(panel["high"], panel["low"], panel["close"]).atr,
        ]
    for rust, numba in zip(results["rust"], results["numba"], strict=True):
        np.testing.assert_allclose(rust.to_numpy(), numba.to_numpy(), rtol=1e-10, atol=1e-10)


# ---- values(): strip the parameter level --------------------------------------------------------


def test_values_strips_the_parameter_level_for_a_single_window(panel) -> None:
    close = panel["close"]
    out = ind.sma(close, 10).ma
    assert out.columns.nlevels == 2  # (ma_window, symbol)
    flat = ind.values(out, like=close)
    assert flat.columns.equals(close.columns)
    assert flat.index.equals(close.index)
    np.testing.assert_array_equal(flat.to_numpy(), out.to_numpy())


def test_values_makes_two_indicators_directly_comparable(panel) -> None:
    close = panel["close"]
    fast, slow = (ind.values(ind.sma(close, w).ma, like=close) for w in (5, 20))
    assert (fast > slow).shape == close.shape  # no column-label mismatch


def test_values_refuses_a_grid_of_parameters(panel) -> None:
    close = panel["close"]
    with pytest.raises(ValueError, match="parameter"):
        ind.values(ind.sma(close, [5, 10]).ma, like=close)


def test_values_handles_a_series(panel) -> None:
    close = panel["close"].iloc[:, 0]
    out = ind.sma(close, 10).ma
    flat = ind.values(out, like=close)
    assert isinstance(flat, pd.Series) and flat.name == close.name
