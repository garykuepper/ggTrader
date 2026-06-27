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
