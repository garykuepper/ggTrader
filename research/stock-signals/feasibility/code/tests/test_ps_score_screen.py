"""Check the fixed long-only PS extraction and window checks."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "ps_score_screen.py"  # root-ok: sibling code
spec = importlib.util.spec_from_file_location("ps_score_screen", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pytestmark = pytest.mark.study


def test_selects_only_highest_long_only_portfolio_and_converts_percent():
    frame = pd.DataFrame(
        {
            "signalname": ["PS", "PS", "Other"],
            "port": ["10", "LS", "10"],
            "date": pd.to_datetime(["2021-01-29"] * 3),
            "ret": [2.5, 900.0, 900.0],
            "Nlong": [100, 100, 100],
            "Nshort": [0, 100, 0],
        }
    )
    result = module.select_long(frame)
    assert result.index.tolist() == [pd.Timestamp("2021-01-31")]
    assert result.iloc[0] == pytest.approx(0.025)


def test_rejects_duplicate_month_or_non_long_only():
    frame = pd.DataFrame(
        {
            "signalname": ["PS", "PS"],
            "port": ["10", "10"],
            "date": pd.to_datetime(["2021-01-29", "2021-01-30"]),
            "ret": [1.0, 1.1],
            "Nlong": [10, 10],
            "Nshort": [0, 0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        module.select_long(frame)
    frame.loc[1, "date"] = pd.Timestamp("2021-02-26")
    frame.loc[1, "Nshort"] = 1
    with pytest.raises(ValueError, match="long-only"):
        module.select_long(frame)


def test_rejects_provider_missing_return_sentinel():
    frame = pd.DataFrame(
        {
            "signalname": ["PS"],
            "port": ["10"],
            "date": ["2020-01-31"],
            "ret": [-99.99],
            "Nlong": [10],
            "Nshort": [0],
        }
    )
    with pytest.raises(ValueError, match="sentinel"):
        module.select_long(frame)


def test_windows_require_every_month():
    dates = pd.date_range("1972-01-31", "2025-12-31", freq="ME")
    candidate = pd.Series([0.01] * len(dates), index=dates)
    benchmark = pd.Series([0.005] * len(dates), index=dates)
    result = module.evaluate(candidate, benchmark)
    assert result["full"]["comparison"]["months"] == 648
    assert result["since_1994"]["comparison"]["months"] == 384
    with pytest.raises(ValueError):
        module.evaluate(candidate.drop(dates[20]), benchmark)


def test_cli_checks_hash_and_refuses_to_overwrite(tmp_path, monkeypatch):
    import hashlib
    import json
    import sys

    dates = pd.date_range("1972-01-31", "2025-12-31", freq="ME")
    file = tmp_path / "portfolios.parquet"
    pd.DataFrame(
        {"signalname": "PS", "port": "10", "date": dates, "ret": 1.0, "Nlong": 100, "Nshort": 0}
    ).to_parquet(file, index=False)
    monkeypatch.setattr(module, "PORTFOLIO_SHA256", hashlib.sha256(file.read_bytes()).hexdigest())
    value_file = tmp_path / module.VALUE_FILE
    value_file.write_bytes(b"synthetic value archive")
    monkeypatch.setattr(module, "VALUE_DIR", tmp_path)
    monkeypatch.setattr(module, "VALUE_SHA256", hashlib.sha256(value_file.read_bytes()).hexdigest())
    monkeypatch.setattr(
        module, "verify_manifest", lambda _: {module.MARKET_FILE: {"sha256": module.MARKET_SHA256}}
    )
    monkeypatch.setattr(
        module, "load_market_total_return", lambda _: pd.DataFrame({"mkt": 0.005}, index=dates)
    )
    monkeypatch.setattr(
        module.french_data,
        "load_size_factor_vw",
        lambda *_: pd.DataFrame({"big": 0.004}, index=dates),
    )
    with monkeypatch.context() as check:
        check.setattr(
            sys,
            "argv",
            ["screen", "--portfolio", str(file), "--french", str(tmp_path), "--check-only"],
        )
        check.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("return data opened"))
        module.main()
    out = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["screen", "--portfolio", str(file), "--french", str(tmp_path), "--out", str(out)],
    )
    module.main()
    payload = json.loads(out.read_text())
    assert payload["results"]["full"]["comparison"]["months"] == 648
    assert payload["descriptive_candidate_vs_big_value"]["full"]["comparison"]["months"] == 648
    with pytest.raises(FileExistsError):
        module.main()
    out.unlink()
    file.write_text("changed")
    with pytest.raises(ValueError, match="portfolio hash"):
        module.main()
    assert not out.exists()
