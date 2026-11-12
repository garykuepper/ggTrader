"""Exercise the research-only ticker and date coverage audit."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).resolve().parents[1] / "stock_tape_coverage.py"  # root-ok: sibling code
spec = importlib.util.spec_from_file_location("stock_tape_coverage", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
pytestmark = pytest.mark.study


def test_membership_changes_and_ticker_normalization():
    history = pd.DataFrame(
        {"tickers": [["A", "BRK.B"], ["BRK.B", "C"]]},
        index=pd.to_datetime(["2021-01-01", "2021-02-01"], utc=True),
    )
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2021-01-29", "2021-01-29", "2021-02-26"]),
            "symbol": ["A", "BRK-B", "BRK-B"],
        }
    )
    dates = pd.to_datetime(["2021-01-29", "2021-02-26"])
    rows = module.coverage(bars, history, dates)
    assert rows[0]["missing"] == []
    assert rows[1]["missing"] == ["C"]
    assert rows[1]["membership_snapshot"] == "2021-02-01"


def test_missing_market_date_is_not_skipped():
    history = pd.DataFrame({"tickers": [["A"]]}, index=pd.to_datetime(["2021-01-01"], utc=True))
    bars = pd.DataFrame({"timestamp": pd.to_datetime(["2021-01-28"]), "symbol": ["A"]})
    rows = module.coverage(bars, history, pd.to_datetime(["2021-01-29"]))
    assert rows[0]["missing"] == ["A"]
    assert rows[0]["n_with_data"] == 0


def test_duplicate_bars_or_unknown_membership_fail_closed():
    history = pd.DataFrame({"tickers": [["A"]]}, index=pd.to_datetime(["2021-01-01"], utc=True))
    bars = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2021-01-29", "2021-01-29"]), "symbol": ["A", "A"]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        module.coverage(bars, history, pd.to_datetime(["2021-01-29"]))
    with pytest.raises(ValueError, match="predates"):
        module.coverage(bars.iloc[:1], history, pd.to_datetime(["2020-12-31"]))


def test_cli_pins_inputs_and_refuses_overwrite(tmp_path, monkeypatch):
    import json
    import sys

    tape = tmp_path / "tape.parquet"
    pd.DataFrame({"timestamp": pd.to_datetime(["2021-01-29"]), "symbol": ["BRK-B"]}).to_parquet(
        tape, index=False
    )
    members = tmp_path / "members.csv"
    members.write_text("date,tickers\n2021-01-01,BRK.B\n")
    out = tmp_path / "audit.json"
    args = [
        "coverage",
        "--tape",
        str(tape),
        "--tape-sha256",
        module.sha256_file(tape),
        "--membership",
        str(members),
        "--membership-sha256",
        module.sha256_file(members),
        "--start",
        "2021-01",
        "--end",
        "2021-02",
        "--out",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", args)
    module.main()
    result = json.loads(out.read_text())
    assert [row["date"] for row in result["month_ends"]] == ["2021-01-29", "2021-02-26"]
    assert result["month_ends"][0]["missing"] == []
    assert result["month_ends"][1]["missing"] == ["BRK-B"]
    with pytest.raises(FileExistsError):
        module.main()
    out.unlink()
    members.write_text("modified")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.main()
    assert not out.exists()
