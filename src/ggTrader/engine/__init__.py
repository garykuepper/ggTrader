"""ggTrader.engine: the only package that imports vectorbt (spec 5.1, 5.2).

Importing it applies the engine settings (``config.configure``) so every vectorbt call in
the process uses the explicit engine, the 252-day year and the daily frequency.
"""

from __future__ import annotations

from ggTrader.engine import config

config.configure()
