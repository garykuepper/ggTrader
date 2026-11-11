"""Exercise the proposed reversal screen using invented inputs only."""

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.study


def test_reversal_driver_uses_only_pinned_inputs_and_never_overwrites(
    tmp_path, monkeypatch, capsys
):
    code = Path(__file__).resolve().parents[1]  # root-ok: sibling code
    monkeypatch.syspath_prepend(str(code))
    spec = importlib.util.spec_from_file_location("reversal_screen", code / "reversal_screen.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    months = pd.date_range("1963-07-31", "2026-04-30", freq="ME")
    candidate = (
        "Average Value Weighted Returns -- Monthly\n,BIG LoPRIOR\n"
        + "\n".join(f"{d:%Y%m},1.0" for d in months)
        + "\n\n"
    )
    market = ",Mkt-RF,SMB,HML,RF\n" + "\n".join(f"{d:%Y%m},0.4,0,0,0.1" for d in months) + "\n\n"
    manifest = {}
    for name, contents in ((driver.PORTFOLIO_FILE, candidate), (driver.MARKET_FILE, market)):
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(name.removesuffix(".zip") + ".csv", contents)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[name] = {"sha256": digest}
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(driver, "EXPECTED", {name: row["sha256"] for name, row in manifest.items()})
    for args in ([], ["--out", str(tmp_path / "unused.json"), "--check-only"]):
        monkeypatch.setattr(sys, "argv", ["reversal_screen", "--data", str(tmp_path), *args])
        with pytest.raises(SystemExit):
            driver.main()
    monkeypatch.setattr(sys, "argv", ["reversal_screen", "--data", str(tmp_path), "--check-only"])
    with monkeypatch.context() as check:
        check.setattr(
            driver.accruals_comparison,
            "evaluate",
            lambda *_: pytest.fail("input check must not evaluate returns"),
        )
        check.setattr(
            driver.french_data,
            "_read_frozen_csv",
            lambda *_: pytest.fail("input check must not open a return table"),
        )
        driver.main()
    assert "verified" in capsys.readouterr().out
    out = tmp_path / "result.json"
    monkeypatch.setattr(
        sys, "argv", ["reversal_screen", "--data", str(tmp_path), "--out", str(out)]
    )
    driver.main()
    payload = json.loads(out.read_text())
    assert payload["candidate"] == "BIG LoPRIOR, value weighted"
    assert payload["results"]["full"]["comparison"]["months"] == 754
    assert payload["results"]["since_1994"]["comparison"]["months"] == 388
    assert payload["results"]["full_stress"]["annual_cost_assumption"] == 0.02
    with pytest.raises(FileExistsError):
        driver.main()
    (tmp_path / driver.PORTFOLIO_FILE).write_bytes(b"corrupted")
    out.unlink()
    with pytest.raises(ValueError, match="Checksum mismatch"):
        driver.main()
    assert not out.exists()
    monkeypatch.setattr(sys, "argv", ["reversal_screen", "--data", str(tmp_path), "--check-only"])
    with pytest.raises(ValueError, match="Checksum mismatch"):
        driver.main()
    portfolio = tmp_path / driver.PORTFOLIO_FILE
    with zipfile.ZipFile(portfolio, "w") as archive:
        archive.writestr("portfolio.csv", candidate.replace("BIG LoPRIOR", "BIG HiPRIOR"))
    digest = hashlib.sha256(portfolio.read_bytes()).hexdigest()
    manifest[driver.PORTFOLIO_FILE]["sha256"] = digest
    (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(driver, "EXPECTED", {name: row["sha256"] for name, row in manifest.items()})
    monkeypatch.setattr(
        sys, "argv", ["reversal_screen", "--data", str(tmp_path), "--out", str(out)]
    )
    with pytest.raises(ValueError, match="Missing required BIG LoPRIOR"):
        driver.main()
    assert not out.exists()
