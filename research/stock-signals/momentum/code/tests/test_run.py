"""Check the momentum driver: download mode only freezes, the default run writes results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.utils.study import load_driver, study_dir

pytestmark = pytest.mark.study


def _panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Provide complete monthly histories with a nonconstant market."""
    index = pd.date_range("1927-01-31", "2026-04-30", freq="ME")
    market = 0.005 + 0.002 * np.sin(np.arange(len(index)))
    sm = pd.DataFrame(
        {
            f"{size}_{rank}": market + 0.0001 * np.cos(np.arange(len(index)))
            for size in ("small", "big")
            for rank in ("lo", "mid", "hi")
        },
        index=index,
    )
    return sm, pd.DataFrame({"mkt": market, "rf": 0.001}, index=index)


def test_driver_download_only_and_offline_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Download mode freezes only; the default run writes provenance and screen results."""
    path = study_dir(__file__) / "code" / "run.py"
    driver = load_driver(path)
    sm, mk = _panels()
    manifest = {"frozen.zip": {"sha256": "test-checksum"}}
    downloads = []
    monkeypatch.setattr(driver, "download_and_freeze", lambda files: downloads.append(files))
    monkeypatch.setattr(driver, "verify_manifest", lambda: manifest)
    monkeypatch.setattr(driver, "load_size_momentum_vw", lambda: sm)
    monkeypatch.setattr(driver, "load_market_total_return", lambda: mk)
    out = tmp_path / "results" / "screen.json"
    monkeypatch.setattr("sys.argv", [str(path), "--download", "--out", str(out)])
    driver.main()
    assert downloads == [[driver.SIZE_MOM_FILE, driver.FACTORS_FILE]]
    assert not out.exists()
    monkeypatch.setattr("sys.argv", [str(path), "--out", str(out)])
    driver.main()
    result = json.loads(out.read_text())
    assert downloads == [[driver.SIZE_MOM_FILE, driver.FACTORS_FILE]]
    assert result["provenance"] == manifest
    assert result["brief"] == "research/stock-signals/momentum/2026-10-20-momentum-retest.md"
    assert result["runtime_sec"] >= 0
    assert result["pass"] == {"big": False, "small": False}
