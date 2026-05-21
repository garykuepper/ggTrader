"""Balance: per-currency account balance (free, used, total)."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, model_validator


class Balance(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    currency: str
    free: Decimal
    used: Decimal
    total: Decimal

    @model_validator(mode="after")
    def _check_total(self) -> "Balance":
        if self.free + self.used != self.total:
            raise ValueError(
                f"Balance invariant violated for {self.currency}: "
                f"free({self.free}) + used({self.used}) != total({self.total})"
            )
        return self
