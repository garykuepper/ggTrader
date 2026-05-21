"""Instrument: canonical, immutable description of a tradable asset on a venue.

Strategies depend on Instrument; concrete brokers and data adapters consume it.
Differences between asset classes (crypto perp vs equity) are captured here so
strategy code stays asset-class-agnostic.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AssetClass(str, Enum):
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_PERP = "crypto_perp"
    CRYPTO_DATED_FUTURE = "crypto_dated_future"
    EQUITY = "equity"
    EQUITY_ETF = "equity_etf"


class Venue(str, Enum):
    KRAKEN_SPOT = "kraken_spot"
    KRAKEN_FUTURES = "kraken_futures"
    KRAKEN_SECURITIES = "kraken_securities"
    ALPACA = "alpaca"
    BINANCEUS_SPOT = "binanceus_spot"


class Instrument(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    symbol: str
    asset_class: AssetClass
    venue: Venue
    quote_currency: str
    base_currency: str
    tick_size: Decimal
    min_order_size: Decimal
    contract_multiplier: Decimal = Field(default=Decimal("1"))
    maker_fee_bps: Decimal
    taker_fee_bps: Decimal
    calendar_id: str
    expiry: Optional[str] = None
    venue_specific_id: Optional[str] = None
