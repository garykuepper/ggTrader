"""The final link finder uses the approved settings."""

from datetime import date

from scripts.kaggle_find_links import settings_from_dict


def test_settings_defaults_and_overrides():
    settings = settings_from_dict({"window_start": "2010-01-01", "min_sessions_short": 2})
    assert settings.window_start == date(2010, 1, 1)
    assert settings.min_sessions_short == 2
    assert settings.agreement_bar == 0.999


def test_run_adds_yahoo_lifetimes(monkeypatch):
    import argparse

    import pandas as pd

    from scripts import kaggle_find_links as script

    days = pd.date_range("2020-01-01", periods=6)
    values = [0.01, -0.02, 0.03, 0.01, -0.01, 0.02]
    yahoo = pd.DataFrame({"ABC": values}, index=days)
    crsp = pd.DataFrame({1: values}, index=days)
    monkeypatch.setenv("TEST_YAHOO_URL", "unused")
    monkeypatch.setattr(
        script,
        "yahoo_data",
        lambda *args: (yahoo, pd.Series({"ABC": days[0]}), pd.Series({"ABC": days[-1]})),
    )
    monkeypatch.setattr(
        script, "crsp_data", lambda *args: (crsp, pd.Series({1: days[0]}), pd.Series({1: days[-1]}))
    )
    monkeypatch.setattr(script, "calendar_from_file", lambda *args: days)
    args = argparse.Namespace(yahoo_url_env="TEST_YAHOO_URL", crsp_file=None, sample_permnos=None)
    settings = script.settings_from_dict(
        {
            "window_start": "2020-01-01",
            "window_end": "2020-12-31",
            "min_sessions": 4,
            "min_sessions_short": 2,
        }
    )
    result = script.run(args, settings)
    assert result.loc[0, "status"] == "linked"
    assert result.loc[0, "first_yf_date"] == days[0]
    assert result.loc[0, "last_yf_date"] == days[-1]
