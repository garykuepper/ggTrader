#!/usr/bin/env python3
"""
ggTrader Unified CLI (ggt)

Entry point for Research, Production, Backtesting, and Live Execution.
"""

import sys
from pathlib import Path

# Provide native path resolutions for src
sys.path.append(str(Path(__file__).parent / "src"))

from ggTrader.cli.main import main

if __name__ == "__main__":
    main()
