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
        self,
        symbol: str,
        positions: dict[str, dict],
        portfolio_value: float,
        prospective_notional: float = 0.0,
    ) -> bool:
        """Returns True if adding to this symbol would exceed concentration limit."""
        current_value = positions.get(symbol, {}).get("market_value", 0.0)
        total_prospective_value = current_value + prospective_notional
        return (total_prospective_value / portfolio_value) >= self.cfg.max_concentration_pct

    def sleeve_slot_caps(self, weights: dict[str, float]) -> dict[str, int]:
        """Per-sleeve share of max_positions, proportional to weight.

        floor(weight_i * max_positions), minimum 1 slot for any sleeve with
        weight_i > 0, with leftover slots from rounding assigned to the
        highest-weight sleeve so the total never exceeds max_positions.
        """
        raw = {
            label: (max(1, int(w * self.cfg.max_positions)) if w > 0 else 0)
            for label, w in weights.items()
        }
        total = sum(raw.values())
        if total > self.cfg.max_positions and raw:
            top = max(raw, key=lambda k: weights[k])
            raw[top] -= total - self.cfg.max_positions
            raw[top] = max(raw[top], 1 if weights[top] > 0 else 0)
        return raw

    def sleeve_position_notional(
        self,
        portfolio_value: float,
        sleeve_weight: float,
        scale: float,
    ) -> float:
        """Dollar amount for a single new position within one sleeve.

        A fixed fraction (position_pct) of that sleeve's allocated capital
        (portfolio_value * sleeve_weight * scale) -- independent of how many
        signals fire that day. Matches the same fixed-fraction-per-entry
        convention simulate_signals used to generate each sleeve's own
        validated backtest curve; the weight*scale overlay caps how much
        total capital a sleeve may deploy (via sleeve_slot_caps), it does
        not resize individual positions based on signal count.
        """
        return round(portfolio_value * sleeve_weight * scale * self.cfg.position_pct, 2)
