"""Reproduce the frozen real-SSO tracking spread used by the century study."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ggTrader.lab.century_leverage import FRENCH_FILE, checked_file, load_french_daily  # noqa: E402

INPUTS = ROOT / "data" / "external" / "a23_inputs"
FRENCH = ROOT / "data" / "external" / "french"
CALIBRATION = ROOT / "docs" / "research" / "_century_leverage_calibration.json"


def main() -> None:
    """Fit each registered fee on the same aligned daily overlap and verify the artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=INPUTS, help="frozen A23 input directory")
    args = parser.parse_args()
    expected = json.loads(CALIBRATION.read_text())
    start, end = expected["registered_window"]
    _, french_record = checked_file(FRENCH, FRENCH_FILE)
    if french_record["sha256"] != expected["input_sha256"]["French_daily_zip"]:
        raise ValueError("French input differs from committed calibration")
    returns = {}
    for symbol in ("SPY", "SSO"):
        path, record = checked_file(args.inputs, f"{symbol}.csv")
        if record["sha256"] != expected["input_sha256"][f"{symbol}.csv"]:
            raise ValueError(f"{symbol} input differs from committed calibration")
        frame = pd.read_csv(path, index_col="date", parse_dates=True, float_precision="round_trip")
        returns[symbol] = (frame["Close"] + frame["Dividends"]) / frame["Close"].shift(1) - 1
    daily = load_french_daily(FRENCH)
    aligned = (
        pd.concat(
            [returns["SPY"].rename("spy"), returns["SSO"].rename("sso"), daily["rf"]],
            axis=1,
        )
        .loc[start:end]
        .dropna()
    )
    if [str(aligned.index[0].date()), str(aligned.index[-1].date())] != expected[
        "aligned_window"
    ] or len(aligned) != expected["aligned_sessions"]:
        raise ValueError("Aligned overlap differs from committed calibration")
    for name, row in expected["fees"].items():
        zero_spread = 2 * aligned["spy"] - aligned["rf"] - row["expense_ratio_annual"] / 252
        spread = float((zero_spread - aligned["sso"]).mean() * 252)
        if spread != row["fitted_spread_annual"]:
            raise ValueError(f"{name} spread differs from committed calibration")
        print(f"{name}: spread={spread!r} ({len(aligned)} sessions)")


if __name__ == "__main__":
    main()
