"""Risk guardrails for paper/live trading."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    """Position sizing and risk parameters."""

    max_positions: int = 30
    position_pct: float = 0.033
    max_concentration_pct: float = 0.05
    daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15


class RiskGuard:
    """Enforces risk limits before order submission."""

    def __init__(self, cfg: RiskConfig | None = None) -> None:
        self.cfg = cfg or RiskConfig()
        self._peak_value: float | None = None

    def update_peak(self, portfolio_value: float) -> None:
        if self._peak_value is None or portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

    def check_drawdown_halt(self, portfolio_value: float) -> tuple[bool, str]:
        """Returns (halted, reason). Halted=True means stop all trading."""
        if self._peak_value is None:
            return False, ""
        drawdown = (self._peak_value - portfolio_value) / self._peak_value
        if drawdown >= self.cfg.max_drawdown_pct:
            return True, (
                f"Max drawdown breached: {drawdown:.1%} "
                f"(peak ${self._peak_value:,.2f} → ${portfolio_value:,.2f})"
            )
        return False, ""

    def check_daily_loss(self, portfolio_value: float, day_start_value: float) -> tuple[bool, str]:
        """Returns (halted, reason). Halted=True means stop trading for today."""
        if day_start_value <= 0:
            return False, ""
        daily_loss = (day_start_value - portfolio_value) / day_start_value
        if daily_loss >= self.cfg.daily_loss_pct:
            return True, (
                f"Daily loss limit: {daily_loss:.1%} "
                f"(${day_start_value:,.2f} → ${portfolio_value:,.2f})"
            )
        return False, ""

    def max_new_positions(self, current_count: int) -> int:
        """How many new positions can be opened."""
        return max(0, self.cfg.max_positions - current_count)

    def position_notional(self, portfolio_value: float) -> float:
        """Dollar amount for a single new position."""
        return round(portfolio_value * self.cfg.position_pct, 2)

    def check_concentration(
        self, symbol: str, positions: dict[str, dict], portfolio_value: float
    ) -> bool:
        """Returns True if adding to this symbol would exceed concentration limit."""
        if symbol not in positions:
            return False
        current_value = positions[symbol].get("market_value", 0.0)
        return (current_value / portfolio_value) >= self.cfg.max_concentration_pct
