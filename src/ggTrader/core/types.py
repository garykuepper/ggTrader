"""Shared type aliases used across the core domain."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

Timeframe = Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
Bps = Decimal
