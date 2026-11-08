"""Run commit 53b739c's unmodified A23 logic on verified frozen inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import types
from hashlib import sha256
from pathlib import Path

import pandas as pd

from ggTrader.utils.fred_data import parse_fred_csv
from ggTrader.utils.study import bootstrap

ROOT = bootstrap(__file__)

INPUTS = ROOT / "data" / "external" / "a23_inputs"
GATE = ROOT / "research/allocation/century-leverage/_century_leverage_a23_gate.json"


def main() -> None:
    """Verify inputs, substitute only the old fetches, and check X's output hash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=INPUTS, help="frozen A23 input directory")
    args = parser.parse_args()
    gate = json.loads(GATE.read_text())
    manifest = json.loads((args.inputs / "MANIFEST.json").read_text())
    for filename, expected_hash in gate["frozen_inputs"].items():
        actual_hash = sha256((args.inputs / filename).read_bytes()).hexdigest()
        if actual_hash != expected_hash or actual_hash != manifest[filename]["sha256"]:
            raise ValueError(f"Frozen input differs from A23 gate: {filename}")

    old_script = f"{gate['baseline_commit']}:scripts/static_leverage_aftertax.py"  # history-read
    source = subprocess.run(
        ["git", "show", old_script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    baseline = types.ModuleType("a23_frozen_baseline")
    baseline.__file__ = str(ROOT / "scripts" / "static_leverage_aftertax.py")  # history-read
    sys.modules[baseline.__name__] = baseline
    # The replayed source imports the pre-move module path, so serve it the moved module.
    import ggTrader.utils.fred_data as _moved_fred  # history-read

    sys.modules.setdefault("ggTrader.lab.fred_data", _moved_fred)  # history-read: pre-move path
    exec(compile(source, baseline.__file__, "exec"), baseline.__dict__)

    class FrozenTicker:
        """Provide the old yfinance call with its frozen frame."""

        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **_kwargs: object) -> pd.DataFrame:
            return pd.read_csv(
                args.inputs / f"{self.symbol}.csv",
                index_col="date",
                parse_dates=True,
                float_precision="round_trip",
            )

    def frozen_fred(series_id: str, start: str, end: str) -> pd.DataFrame:
        """Provide the old FRED call with the same dated rows as the frozen loader."""
        frame = parse_fred_csv((args.inputs / f"{series_id}.csv").read_text(), series_id)
        return frame.loc[frame["date"].between(start, end)]

    baseline.yf = types.SimpleNamespace(Ticker=FrozenTicker)
    baseline.load_fred_series = frozen_fred
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "x.json"
        sys.argv = ["static_leverage_aftertax.py", "--out", str(out)]  # history-read
        # shortcut: baseline A23 cannot resume, add checkpointing if gate runs become routine.
        baseline.main()
        actual_hash = sha256(out.read_bytes()).hexdigest()
    expected_hash = gate["runs"]["X"]["output_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(f"X hash {actual_hash} != {expected_hash}")
    print(f"X SHA-256: {actual_hash} (matches gate)")


if __name__ == "__main__":
    main()
