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


def test_error_prints_class_and_message_without_connection_secrets(monkeypatch, capsys, tmp_path):
    import psycopg2

    from scripts import kaggle_find_links as script

    def fail(*args):
        raise psycopg2.OperationalError(
            "bad database postgresql://alice:secret@db.example/ggtrader "
            "password=hidden dbname=ggtrader"
        )

    monkeypatch.setattr(script, "run", fail)
    assert script.main(["--crsp-file", "x", "--out", str(tmp_path / "links.csv")]) == 1
    error = capsys.readouterr().err
    assert "OperationalError" in error and "bad database" in error
    assert "secret" not in error and "hidden" not in error and "db.example" not in error


def test_link_finder_writes_manifest_of_actual_inputs(tmp_path, monkeypatch):
    import hashlib
    import json

    import pandas as pd

    from scripts import kaggle_find_links as script

    raw = tmp_path / "crsp.parquet"
    raw.write_bytes(b"synthetic crsp")
    out = tmp_path / "links.csv"
    frame = pd.DataFrame({"symbol": ["ABC"], "permno": [1], "status": ["linked"]})
    frame.attrs["yahoo_fingerprint"] = {
        "n_symbols": 1,
        "newest_timestamp": "2020-01-01",
        "symbols_sha256": "a" * 64,
    }
    monkeypatch.setattr(script, "run", lambda *args: frame)
    monkeypatch.setattr(script, "git_sha", lambda: "revision")
    assert (
        script.main(["--crsp-file", str(raw), "--out", str(out), "--window-start", "2010-01-01"])
        == 0
    )
    manifest = json.loads((tmp_path / "links.csv.manifest.json").read_text())
    assert manifest["csv_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest["crsp_file_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert manifest["settings"]["window_start"] == "2010-01-01"
    assert manifest["yahoo_fingerprint"] == frame.attrs["yahoo_fingerprint"]
    assert manifest["git_sha"] == "revision" and manifest["created_at"]
