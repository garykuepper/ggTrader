import pandas as pd
import pytest

from ggTrader.lab.data import rebalance_dates


def test_cached_loader_interval_to_timedelta():
    from ggTrader.data.live.cached_yfinance_loader import CachedYFinanceLoader

    loader = CachedYFinanceLoader.__new__(CachedYFinanceLoader)  # skip __init__ (no DB needed)
    assert loader._interval_to_timedelta("1d") == pd.Timedelta(days=1)
    assert loader._interval_to_timedelta("1h") == pd.Timedelta(hours=1)
    assert loader._interval_to_timedelta("1wk") == pd.Timedelta(weeks=1)
    assert loader._interval_to_timedelta("1mo") == pd.Timedelta(days=30)


def test_rebalance_dates_are_month_ends_excluding_last():
    idx = pd.date_range("2021-01-01", "2021-06-30", freq="B", tz="UTC")
    # eval_start=2021-01-31 is past Jan's last trading day (Jan 29), so January is
    # excluded — matching the production harness. June is the final span month (dropped).
    dates = rebalance_dates(
        idx, pd.Timestamp("2021-01-31", tz="UTC"), pd.Timestamp("2021-06-30", tz="UTC")
    )
    # First entry is the window's first trading day itself (2021-02-01, a
    # Monday) -- item 12 fix: a decision must exist at/immediately after
    # window start, not just at the first month-end, or the fold starts in
    # pure cash for up to a month. See
    # docs/research/2026-08-18-wfo-anchor-leakage-fix.md.
    assert dates[0] == pd.Timestamp("2021-02-01", tz="UTC")
    assert [d.strftime("%Y-%m") for d in dates] == [
        "2021-02",
        "2021-02",
        "2021-03",
        "2021-04",
        "2021-05",
    ]
    assert all(d.tz is not None for d in dates)


def test_rebalance_dates_window_start_not_duplicated_when_already_month_end():
    """If the window's first trading day IS already the first month-end
    (e.g. eval_start falls on the last trading day of its month), the
    window-start decision must not be duplicated."""
    idx = pd.date_range("2021-01-01", "2021-04-30", freq="B", tz="UTC")
    jan_end = idx[idx.tz_convert(None).to_period("M") == "2021-01"][-1]
    dates = rebalance_dates(idx, jan_end, pd.Timestamp("2021-04-30", tz="UTC"))
    assert dates[0] == jan_end
    assert dates.count(jan_end) == 1


def test_rebalance_dates_single_bar_window_has_no_forward_period():
    """A window that is a single trading day has nothing to trade forward
    into, so it must still produce zero rebalance dates."""
    idx = pd.date_range("2021-01-01", "2021-01-10", freq="B", tz="UTC")
    single = idx[3]
    assert rebalance_dates(idx, single, single) == []


def test_rebalance_dates_empty_when_no_overlap():
    idx = pd.date_range("2021-01-01", "2021-01-10", freq="B", tz="UTC")
    assert (
        rebalance_dates(
            idx, pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-12-31", tz="UTC")
        )
        == []
    )


def test_universe_members_asof_sp500():
    from ggTrader.data.core.index_constituents import universe_members_asof

    members = universe_members_asof("sp500", pd.Timestamp("2025-01-15", tz="UTC"))
    assert len(members) > 400
    assert "AAPL" in members


def test_universe_members_asof_nasdaq100():
    from ggTrader.data.core.index_constituents import universe_members_asof

    members = universe_members_asof("nasdaq100", pd.Timestamp("2025-01-15", tz="UTC"))
    assert len(members) > 90
    assert "NVDA" in members


def test_universe_all_between_nasdaq100():
    from ggTrader.data.core.index_constituents import universe_all_between

    members = universe_all_between(
        "nasdaq100",
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"),
    )
    assert len(members) > 90
    assert all(isinstance(m, str) for m in members)


def test_universe_members_asof_unknown_raises():
    from ggTrader.data.core.index_constituents import universe_members_asof

    with pytest.raises(ValueError, match="Unknown snapshot universe"):
        universe_members_asof("nikkei225", pd.Timestamp("2025-01-15", tz="UTC"))


def test_snapshot_members_nasdaq100():
    from ggTrader.data.core.index_constituents import snapshot_members

    members = snapshot_members("nasdaq100")
    assert len(members) == 101
    assert "NVDA" in members


def test_equity_universe_between_with_universe_arg():
    from ggTrader.lab.data import equity_universe_between

    members = equity_universe_between(
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"),
        universe="nasdaq100",
    )
    assert len(members) > 90
    assert "NVDA" in members


@pytest.mark.integration
def test_load_ohlcv_returns_multiindex_frame():
    from ggTrader.lab.data import load_ohlcv

    df = load_ohlcv(["SPY"], "2024-01-01", "2024-03-01")
    assert df.columns.names == ["symbol", "field"]
    assert "close" in df["SPY"].columns
    assert len(df) > 20


@pytest.mark.integration
def test_fetch_stock_ohlcv_returns_multiindex_frame():
    from ggTrader.lab.data import fetch_stock_ohlcv

    df = fetch_stock_ohlcv(["SPY", "AAPL"], start="2024-01-01", end="2024-03-01")
    assert df.columns.names == ["symbol", "field"]
    assert "close" in df["SPY"].columns
    assert len(df) > 20


def test_crypto_base_config_exists_and_has_low_volume_defaults():
    from ggTrader.lab.data import CRYPTO_BASE_CONFIG

    assert CRYPTO_BASE_CONFIG["FEES"] == 0.0040
    assert CRYPTO_BASE_CONFIG["SLIPPAGE"] == 0.0015
    assert CRYPTO_BASE_CONFIG["BENCHMARK_SYMBOL"] == "BTC"
