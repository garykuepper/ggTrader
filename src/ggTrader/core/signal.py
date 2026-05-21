"""Signal: a strategy's directional view on an instrument at a point in time.

Pure data. Sized into Orders downstream by the Sizer; strategies never construct Orders.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ggTrader.core.instrument import Instrument


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        return value.astimezone(timezone.utc)
    return value


class Signal(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    ts: datetime
    instrument: Instrument
    direction: Direction
    confidence: Decimal = Field(default=Decimal("1.0"))
    strategy_id: str
    target_notional_usd: Optional[Decimal] = None
    """Optional fixed-notional sizing hint from the strategy.

    Strategies that already know their position size (carry, calendar spreads,
    fixed-notional momentum) set this. A Sizer downstream may override it.
    Phase 3.5 replaces the metadata-string hack used in Phase 3.
    """

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ts")
    @classmethod
    def _ts_utc(cls, v: datetime) -> datetime:
        return _require_utc(v)
