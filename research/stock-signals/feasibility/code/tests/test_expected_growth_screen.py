"""Check the proposed expected-growth driver using invented observations."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.study


def test_driver_checks_hashes_and_runs_only_when_requested(tmp_path, monkeypatch, capsys):
    code = Path(__file__).resolve().parents[1]  # root-ok: sibling code
    monkeypatch.syspath_prepend(str(code))
    spec = importlib.util.spec_from_file_location(
        "expected_growth_screen", code / "expected_growth_screen.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    dates = pd.period_range("1967-01", "2025-12", freq="M")
    month_number = np.arange(len(dates))
    portfolios = pd.DataFrame(
        {
            "year": dates.year,
            "month": dates.month,
            "rank_ME": 2,
            "rank_EG": 3,
            "nstocks": 50,
            "ret_vw": 1.0 + 0.2 * np.sin(month_number),
            "retx_vw": 0.5,
        }
    )
    factors = pd.DataFrame(
        {
            "year": dates.year,
            "month": dates.month,
            "R_F": 0.1,
            "R_MKT": 0.5,
        }
    )
    expected = {}
    for name, frame in ((driver.PORTFOLIO_FILE, portfolios), (driver.MARKET_FILE, factors)):
        path = tmp_path / name
        frame.to_csv(path, index=False)
        expected[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(driver, "EXPECTED", expected)
    monkeypatch.setattr(
        sys, "argv", ["expected_growth_screen", "--data", str(tmp_path), "--check-only"]
    )
    with monkeypatch.context() as check:
        check.setattr(driver.pd, "read_csv", lambda *_a, **_k: pytest.fail("check read returns"))
        driver.main()
    assert "verified" in capsys.readouterr().out
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys, "argv", ["expected_growth_screen", "--data", str(tmp_path), "--out", str(output)]
    )
    driver.main()
    payload = json.loads(output.read_text())
    assert payload["results"]["full"]["comparison"]["months"] == 708
    assert payload["results"]["since_1994"]["comparison"]["months"] == 384
    assert payload["results"]["full"]["annual_cost_assumption"] == 0.01
    assert payload["results"]["full_stress"]["annual_cost_assumption"] == 0.02
    assert payload["input_sha256"] == expected
    assert isinstance(payload["proposed_hurdles"]["since_1994_pass"], bool)
    assert "full_pass" in payload["proposed_hurdles"]
    assert isinstance(payload["git"]["dirty"], bool)
    assert len(payload["git"]["commit"]) == 40
    assert len(payload["code_sha256"]["brief"]) == 64
    assert payload["proposed_hurdles"]["overall_pass"] == (
        payload["proposed_hurdles"]["full_pass"] and payload["proposed_hurdles"]["since_1994_pass"]
    )
    with pytest.raises(FileExistsError):
        driver.main()
    portfolios["ret_vw"] = 1.0
    portfolios.to_csv(tmp_path / driver.PORTFOLIO_FILE, index=False)
    expected[driver.PORTFOLIO_FILE] = hashlib.sha256(
        (tmp_path / driver.PORTFOLIO_FILE).read_bytes()
    ).hexdigest()
    undefined = tmp_path / "undefined.json"
    monkeypatch.setattr(
        sys, "argv", ["expected_growth_screen", "--data", str(tmp_path), "--out", str(undefined)]
    )
    driver.main()
    assert json.loads(undefined.read_text())["proposed_hurdles"]["overall_pass"] is None
    (tmp_path / driver.PORTFOLIO_FILE).write_text("corrupt")
    output.unlink()
    monkeypatch.setattr(
        sys, "argv", ["expected_growth_screen", "--data", str(tmp_path), "--out", str(output)]
    )
    with pytest.raises(ValueError, match="hash"):
        driver.main()
    assert not output.exists()


def test_duplicate_header_is_rejected(monkeypatch):
    code = Path(__file__).resolve().parents[1]  # root-ok: sibling code
    monkeypatch.syspath_prepend(str(code))
    spec = importlib.util.spec_from_file_location(
        "expected_growth_screen", code / "expected_growth_screen.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    with pytest.raises(ValueError, match="duplicate"):
        driver._read_csv(b"year,month,year\n2020,1,2020\n")


def test_hurdle_combination_and_brief_pins(monkeypatch):
    code = Path(__file__).resolve().parents[1]  # root-ok: sibling code
    monkeypatch.syspath_prepend(str(code))
    spec = importlib.util.spec_from_file_location(
        "expected_growth_screen", code / "expected_growth_screen.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    assert driver._hurdles(None, None, 0.02)["overall_pass"] is None
    assert driver._hurdles(3.1, 0.001, 0.009)["overall_pass"] is False
    assert driver._hurdles(3.1, 0.001, 0.011)["overall_pass"] is True
    brief = (driver.ROOT / driver.BRIEF).read_text()
    for filename, digest in driver.EXPECTED.items():
        assert filename in brief and digest in brief
