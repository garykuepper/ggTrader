import numpy as np
import pandas as pd
import pytest


def test_trailing_realized_vol_annualizes_and_warms_up():
    from ggTrader.lab.allocation import trailing_realized_vol

    idx = pd.date_range("2021-01-01", periods=5, freq="D")
    rets = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01], index=idx)
    vol = trailing_realized_vol(rets, window=3)

    # First two are warmup (NaN), third onward defined
    assert vol.iloc[:2].isna().all()
    expected = rets.iloc[:3].std() * np.sqrt(252)
    assert vol.iloc[2] == pytest.approx(expected)


def test_inverse_vol_weights_favor_low_vol_and_sum_to_one():
    from ggTrader.lab.allocation import inverse_vol_weights

    w = inverse_vol_weights({"a": 0.04, "b": 0.08})
    assert w["a"] > w["b"]  # lower vol -> higher weight
    assert w["a"] + w["b"] == pytest.approx(1.0)
    # a has half the vol of b -> twice the weight: 2/3 vs 1/3
    assert w["a"] == pytest.approx(2 / 3)


def test_inverse_vol_weights_drop_invalid_and_fallback_equal():
    from ggTrader.lab.allocation import inverse_vol_weights

    # NaN/zero dropped
    w = inverse_vol_weights({"a": 0.05, "b": float("nan"), "c": 0.0})
    assert set(w) == {"a"}
    assert w["a"] == pytest.approx(1.0)

    # all invalid -> equal weights across original keys
    w2 = inverse_vol_weights({"a": 0.0, "b": float("nan")})
    assert w2 == pytest.approx({"a": 0.5, "b": 0.5})


def test_target_vol_scale_levers_up_and_caps():
    from ggTrader.lab.allocation import target_vol_scale

    # blend vol 0.04, target 0.068 -> scale 1.7
    assert target_vol_scale(0.04, 0.068) == pytest.approx(1.7)
    # would be 3.4x but capped at 2.0
    assert target_vol_scale(0.02, 0.068, max_leverage=2.0) == pytest.approx(2.0)
    # already above target -> scale down below 1.0
    assert target_vol_scale(0.10, 0.068) == pytest.approx(0.68)
    # degenerate blend vol -> no exposure
    assert target_vol_scale(0.0, 0.068) == 0.0
    assert target_vol_scale(float("nan"), 0.068) == 0.0
