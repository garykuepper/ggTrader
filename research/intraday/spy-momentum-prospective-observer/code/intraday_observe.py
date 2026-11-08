"""Manual command line for the isolated prospective SPY intraday recorder (no orders)."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # root-ok: stdlib-only family
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "_lib")]  # root-ok: venv has no repo package


def main() -> int:
    """Import after path setup; argument errors and --help never touch state or the network."""
    from intraday_observer import main as run

    return run()


if __name__ == "__main__":
    sys.exit(main())
