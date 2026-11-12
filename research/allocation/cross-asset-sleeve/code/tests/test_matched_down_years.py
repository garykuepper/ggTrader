"""Exercise the proposed matched-down-year calculation with synthetic returns."""

import pandas as pd
import pytest

from ggTrader.utils.study import load_driver, study_dir

module = load_driver(study_dir(__file__) / "code" / "matched_down_years.py")
pytestmark = pytest.mark.study


def test_selects_only_same_negative_spy_years_and_uses_net_comparison():
    gross_spy = pd.Series([-0.20, 0.0, 0.15, -0.05], index=[2020, 2021, 2022, 2023])
    net = pd.DataFrame(
        {
            "candidate": [-0.10, 0.01, 0.10, -0.08],
            "fixed": [-0.15, 0.0, 0.12, -0.06],
            "spy": [-0.21, -0.01, 0.14, -0.06],
        },
        index=gross_spy.index,
    )
    result = module.matched_down_years(gross_spy, net)
    assert result.index.tolist() == [2020, 2023]
    assert result["candidate_minus_fixed"].tolist() == pytest.approx([0.05, -0.02])
    assert result["spy"].tolist() == pytest.approx([-0.21, -0.06])


def test_rejects_misalignment_and_invalid_returns():
    spy = pd.Series([-0.1], index=[2020])
    with pytest.raises(ValueError, match="year"):
        module.matched_down_years(
            spy, pd.DataFrame({"candidate": [0.0], "fixed": [0.0], "spy": [0.0]}, index=[2021])
        )
    with pytest.raises(ValueError, match="return"):
        module.matched_down_years(
            spy,
            pd.DataFrame({"candidate": [float("nan")], "fixed": [0.0], "spy": [0.0]}, index=[2020]),
        )


def test_no_down_years_returns_empty_table():
    spy = pd.Series([0.0, 0.1], index=[2020, 2021])
    net = pd.DataFrame(
        {"candidate": [0.0, 0.1], "fixed": [0.0, 0.1], "spy": [0.0, 0.1]}, index=spy.index
    )
    assert module.matched_down_years(spy, net).empty


def test_annual_returns_use_prior_year_close_and_shared_sessions():
    calendar = pd.to_datetime(["2020-12-31", "2021-01-04", "2021-12-31", "2022-01-03"])
    curves = pd.DataFrame(
        {
            "spy_gross": [100, 110, 80],
            "candidate": [100, 110, 90],
            "fixed": [100, 105, 85],
            "spy": [100, 109, 79],
        },
        index=calendar[:3],
    )
    gross, net = module.calendar_year_returns(curves, calendar, [2021])
    result = module.matched_down_years(gross, net)
    assert result.index.tolist() == [2021]
    assert result.loc[2021, "candidate_minus_fixed"] == pytest.approx(0.05)
    assert result.loc[2021, "spy"] == pytest.approx(-0.21)


def test_missing_interior_session_or_unfinished_year_is_rejected():
    calendar = pd.to_datetime(["2020-12-31", "2021-01-04", "2021-12-31", "2022-01-03"])
    curves = pd.DataFrame(
        100.0, index=calendar[[0, 2]], columns=["spy_gross", "candidate", "fixed", "spy"]
    )
    with pytest.raises(ValueError, match="session"):
        module.calendar_year_returns(curves, calendar, [2021])
    with pytest.raises(ValueError, match="calendar"):
        module.calendar_year_returns(curves, calendar[:-1], [2021])


@pytest.mark.parametrize("invalid_value", [True, "100", 100 + 1j, float("nan"), float("inf"), 0])
def test_wealth_curves_reject_nonnumeric_values(invalid_value):
    calendar = pd.to_datetime(["2020-12-31", "2021-12-31", "2022-01-03"])
    curves = pd.DataFrame(
        100.0, index=calendar[:2], columns=["spy_gross", "candidate", "fixed", "spy"]
    )
    curves["candidate"] = [invalid_value, invalid_value]
    with pytest.raises(ValueError, match="curve"):
        module.calendar_year_returns(curves, calendar, [2021])


def test_annual_returns_reject_complex_values():
    spy = pd.Series([-0.1 + 0.01j], index=[2020])
    net = pd.DataFrame({"candidate": [0.0], "fixed": [0.0], "spy": [0.0]}, index=[2020])
    with pytest.raises(ValueError, match="return"):
        module.matched_down_years(spy, net)


def test_total_loss_during_year_is_reported():
    calendar = pd.to_datetime(["2020-12-31", "2021-06-30", "2021-12-31", "2022-01-03"])
    curves = pd.DataFrame(
        {
            "spy_gross": [100, 90, 80],
            "candidate": [100, 0, 0],
            "fixed": [100, 95, 90],
            "spy": [100, 90, 80],
        },
        index=calendar[:3],
    )
    gross, net = module.calendar_year_returns(curves, calendar, [2021])
    assert module.matched_down_years(gross, net).loc[2021, "candidate"] == -1.0


def test_zero_wealth_cannot_recover_without_external_flows():
    calendar = pd.to_datetime(["2020-12-31", "2021-06-30", "2021-12-31", "2022-01-03"])
    curves = pd.DataFrame(
        100.0, index=calendar[:3], columns=["spy_gross", "candidate", "fixed", "spy"]
    )
    curves["candidate"] = [100, 0, 1]
    with pytest.raises(ValueError, match="curve"):
        module.calendar_year_returns(curves, calendar, [2021])


def test_zero_wealth_cannot_recover_between_nonconsecutive_requested_years():
    calendar = pd.to_datetime(
        ["2020-12-31", "2021-12-31", "2022-12-30", "2023-12-29", "2024-01-02"]
    )
    curves = pd.DataFrame(
        100.0, index=calendar[:-1], columns=["spy_gross", "candidate", "fixed", "spy"]
    )
    curves.loc["2021-12-31", "candidate"] = 0.0
    with pytest.raises(ValueError, match="curve"):
        module.calendar_year_returns(curves, calendar, [2021, 2023])


def test_nonconsecutive_years_remain_valid_when_wealth_is_positive():
    calendar = pd.to_datetime(
        ["2020-12-31", "2021-12-31", "2022-12-30", "2023-12-29", "2024-01-02"]
    )
    curves = pd.DataFrame(
        100.0, index=calendar[:-1], columns=["spy_gross", "candidate", "fixed", "spy"]
    )
    gross, net = module.calendar_year_returns(curves, calendar, [2021, 2023])
    assert gross.index.tolist() == [2021, 2023]
    assert net["candidate"].tolist() == [0.0, 0.0]


def test_zero_wealth_in_skipped_year_cannot_silently_recover():
    calendar = pd.to_datetime(
        ["2020-12-31", "2021-12-31", "2022-06-30", "2022-12-30", "2023-12-29", "2024-01-02"]
    )
    curves = pd.DataFrame(
        100.0, index=calendar[:-1], columns=["spy_gross", "candidate", "fixed", "spy"]
    )
    curves.loc["2022-06-30", "candidate"] = 0.0
    with pytest.raises(ValueError, match="curve"):
        module.calendar_year_returns(curves, calendar, [2021, 2023])


def test_nonconsecutive_years_require_intervening_sessions():
    calendar = pd.to_datetime(
        ["2020-12-31", "2021-12-31", "2022-06-30", "2022-12-30", "2023-12-29", "2024-01-02"]
    )
    curves = pd.DataFrame(
        100.0, index=calendar[[0, 1, 3, 4]], columns=["spy_gross", "candidate", "fixed", "spy"]
    )
    with pytest.raises(ValueError, match="session"):
        module.calendar_year_returns(curves, calendar, [2021, 2023])
