"""Timer wrapper for one stage of the prospective SPY intraday recorder (no orders)."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # root-ok: stdlib-only family
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "_lib")]  # root-ok: venv has no repo package


def main() -> int:
    """Import after path setup; --help and argument errors never touch state or the network."""
    from intraday_observer import job_main

    return job_main()


if __name__ == "__main__":
    sys.exit(main())
