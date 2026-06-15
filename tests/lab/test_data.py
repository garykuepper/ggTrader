import pandas as pd
import pytest

from ggTrader.lab.data import rebalance_dates


def test_rebalance_dates_are_month_ends_excluding_last():
    idx = pd.date_range("2021-01-01", "2021-06-30", freq="B", tz="UTC")
    # eval_start=2021-01-31 is past Jan's last trading day (Jan 29), so January is
    # excluded — matching the production harness. June is the final span month (dropped).
    dates = rebalance_dates(
        idx, pd.Timestamp("2021-01-31", tz="UTC"), pd.Timestamp("2021-06-30", tz="UTC")
    )
    assert [d.strftime("%Y-%m") for d in dates] == ["2021-02", "2021-03", "2021-04", "2021-05"]
    assert all(d.tz is not None for d in dates)


def test_rebalance_dates_empty_when_no_overlap():
    idx = pd.date_range("2021-01-01", "2021-01-10", freq="B", tz="UTC")
    assert (
        rebalance_dates(
            idx, pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-12-31", tz="UTC")
        )
        == []
    )


@pytest.mark.integration
def test_load_ohlcv_returns_multiindex_frame():
    from ggTrader.lab.data import load_ohlcv

    df = load_ohlcv(["SPY"], "2024-01-01", "2024-03-01")
    assert df.columns.names == ["symbol", "field"]
    assert "close" in df["SPY"].columns
    assert len(df) > 20
