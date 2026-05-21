"""Ticker: snapshot of best bid/ask/last for an instrument at a point in time."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from ggTrader.core.instrument import Instrument
from ggTrader.core.signal import _require_utc


class Ticker(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    ts: datetime
    instrument: Instrument
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last: Decimal
    volume_24h: Optional[Decimal] = None

    @field_validator("ts")
    @classmethod
    def _ts_utc(cls, v: datetime) -> datetime:
        return _require_utc(v)
