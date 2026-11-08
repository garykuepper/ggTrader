"""Timer wrapper for one stage of the prospective SPY intraday recorder (no orders)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> int:
    """Import after path setup; --help and argument errors never touch state or the network."""
    from ggTrader.lab.intraday_observer import job_main

    return job_main()


if __name__ == "__main__":
    sys.exit(main())
