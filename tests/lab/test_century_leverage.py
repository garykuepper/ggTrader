"""Checks for the frozen century leverage inputs and mechanics."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ggTrader.lab.century_leverage import (
    band_trigger,
    build_legs,
    load_shiller_yield,
    parse_french_daily,
    recommendation_rule,
    simulate,
)
from ggTrader.lab.fred_data import parse_fred_csv
from scripts import static_leverage_aftertax as a23


def test_a23_frozen_loader_matches_live_columns_and_values(monkeypatch) -> None:
    """Frozen A23 files must follow the same code path as an identical live fetch."""
    source = Path("data/external/a23_inputs")

    class Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **_kwargs: object) -> pd.DataFrame:
            frame = pd.read_csv(source / f"{self.symbol}.csv", index_col="date", parse_dates=True)
            frame.index = frame.index.tz_localize("America/New_York")
            return frame

    monkeypatch.setattr(a23.yf, "Ticker", Ticker)
    monkeypatch.setattr(
        a23,
        "load_fred_series",
        lambda *_args: parse_fred_csv((source / "DTB3.csv").read_text(), "DTB3"),
    )
    live = a23.load_prices(0.0075)
    frozen = a23.load_prices(0.0075, frozen_inputs=source)
    for actual, expected in zip(frozen[:2], live[:2], strict=True):
        assert_frame_equal(actual, expected)
    assert frozen[2] == live[2]


def test_daily_french_parser_checks_schema_dates_and_returns() -> None:
    """The daily parser accepts the registered section and rejects a missing session."""
    text = (
        "source\n,Mkt-RF,SMB,HML,RF\n"
        "19260701,0.09,-0.23,-0.28,0.01\n"
        "19260702,0.45,-0.34,-0.03,0.01\n\nfooter"
    )
    actual = parse_french_daily(text)
    assert actual.loc["1926-07-01", "mkt"] == pytest.approx(0.001)
    assert actual.loc["1926-07-02", "rf"] == pytest.approx(0.0001)
    with pytest.raises(ValueError, match="header"):
        parse_french_daily(text.replace("Mkt-RF", "Mkt_RF"))
    with pytest.raises(ValueError, match="date gap"):
        parse_french_daily(text.replace("19260702", "19260712"))
    with pytest.raises(ValueError, match="non-finite"):
        parse_french_daily(text.replace("0.45,-0.34", "NaN,-0.34"))


def test_synthetic_daily_reset_and_dividend_split() -> None:
    """The two legs preserve the fixed daily formula and monthly dividend treatment."""
    dates = pd.to_datetime(["2000-01-28", "2000-01-31", "2000-02-01"])
    daily = pd.DataFrame({"mkt": [0.01, 0.02, -0.01], "rf": [0.0001] * 3}, index=dates)
    yield_monthly = pd.Series([0.12, 0.12], index=pd.to_datetime(["2000-01-01", "2000-02-01"]))
    close, dist = build_legs(daily, yield_monthly, 0.0084, 0.009)
    expected_2x = 2 * daily.loc["2000-01-31", "mkt"] - 0.0001 - (0.0084 + 0.009) / 252
    assert close.loc["2000-01-31", "SSO"] == pytest.approx(1 + expected_2x)
    assert dist.loc["2000-01-31", "SPY"] == pytest.approx(0.01)
    total_1x = (close.loc["2000-01-31", "SPY"] + dist.loc["2000-01-31", "SPY"]) - 1
    assert total_1x == pytest.approx(0.02 - 0.000945 / 252)
    zero_close, zero_dist = build_legs(daily, yield_monthly, 0.0084, 0.009, True)
    assert zero_dist.to_numpy().sum() == 0
    assert zero_close.loc["2000-01-31", "SPY"] == pytest.approx(1 + 0.02 - 0.000945 / 252)


def test_band_trigger_is_strict() -> None:
    """The 30% arm trades only beyond 22-38%, never on the boundary."""
    assert not band_trigger(0.22, 0.30)
    assert not band_trigger(0.38, 0.30)
    assert band_trigger(0.219, 0.30)
    assert band_trigger(0.381, 0.30)


def test_recommendation_rule_includes_lower_weight_extra_gate() -> None:
    """An otherwise passing reduction still needs 30% ruin or underperformance."""

    def row(cagr: float, drawdown: float, ruin: bool, era: float) -> dict:
        return {
            "full": {
                "after_tax_cagr": cagr,
                "max_drawdown": drawdown,
                "ruin": ruin,
                "after_tax_terminal": 100_000 * (1 + cagr),
            },
            "1973-1981": {"after_tax_cagr": era},
        }

    rows = {
        "0": row(0.07, -0.30, False, 0.05),
        "15": row(0.09, -0.40, False, 0.05),
        "30": row(0.08, -0.45, False, 0.04),
        "45": row(0.09, -0.52, False, 0.05),
        "60": row(0.10, -0.95, True, 0.06),
    }
    result = recommendation_rule(rows)
    assert result["alternatives"]["15"]["conditions"]["below_30_extra"] is False
    assert result["alternatives"]["45"]["passes"] is True
    assert result["alternatives"]["60"]["conditions"]["no_ruin"] is False
    assert result["proposal"] == "45"
    rows["30"]["full"]["ruin"] = True
    assert recommendation_rule(rows)["alternatives"]["15"]["passes"] is True


def test_event_simulator_matches_a23_lot_engine_on_daily_path() -> None:
    """The event loop reuses A23's trade and tax behavior on a synthetic daily path."""
    dates = pd.bdate_range("2020-12-01", "2022-05-31")
    close = pd.DataFrame(
        {
            "SPY": np.linspace(100, 140, len(dates)),
            "SSO": np.linspace(100, 180, len(dates)),
            "BIL": np.ones(len(dates)),
        },
        index=dates,
    )
    dist = pd.DataFrame(0.0, index=dates, columns=["SPY", "SSO", "BIL"])
    dist.loc["2020-12-31", "SPY"] = 1.0
    dist.loc["2021-12-31", "SPY"] = 1.2
    original = a23.run(close, dist, "2020-12-01", "2022-05-31", 0.3, 0.08, 0.24, 0.15, False)
    event = simulate(close, dist, "2020-12-01", "2022-05-31", 0.3, 0.24, 0.15, False)
    assert event["after_tax_terminal"] == pytest.approx(original["after_tax_terminal"], abs=1)
    assert event["max_drawdown"] == pytest.approx(original["max_dd"], abs=0.0001)


def test_frozen_shiller_ends_with_required_dividend_month() -> None:
    """The r2 endpoint has a real dividend input and no filled later month."""
    yields = load_shiller_yield(Path("data/external/shiller"))
    assert yields.index[-1] == pd.Timestamp("2026-06-01")
    assert len(yields) == 1200
