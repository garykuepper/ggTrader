"""Order and Fill: the broker-facing pair of types.

An Order is what a Sizer hands to a Broker. A Fill is what a Broker reports back
after (partial or full) execution. Both are immutable; partial fills produce
multiple Fill records sharing the same order_id.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from ggTrader.core.instrument import Instrument
from ggTrader.core.signal import Direction, _require_utc


class Order(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    ts: datetime
    instrument: Instrument
    side: Direction
    quantity: Decimal
    order_type: str
    limit_price: Optional[Decimal] = None
    time_in_force: str = "GTC"
    client_order_id: str

    @field_validator("ts")
    @classmethod
    def _ts_utc(cls, v: datetime) -> datetime:
        return _require_utc(v)


class Fill(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    ts: datetime
    order_id: str
    instrument: Instrument
    side: Direction
    quantity: Decimal
    price: Decimal
    fee: Decimal
    fee_currency: str

    @field_validator("ts")
    @classmethod
    def _ts_utc(cls, v: datetime) -> datetime:
        return _require_utc(v)
