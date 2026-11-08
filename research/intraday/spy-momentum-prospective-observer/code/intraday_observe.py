"""Manual command line for the isolated prospective SPY intraday recorder (no orders)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    """Import after path setup; argument errors and --help never touch state or the network."""
    from ggTrader.lab.intraday_observer import main as run

    return run()


if __name__ == "__main__":
    sys.exit(main())
