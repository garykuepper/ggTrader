"""Risk guardrails for paper/live trading."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    """Position sizing and risk parameters.

    max_positions is a PER-SLEEVE slot budget, not a global cap. Each sleeve
    with positive weight may hold up to max_positions concurrent positions
    (see sleeve_slot_caps) -- independent of that sleeve's weight, since
    sleeve_position_notional() already scales position size by weight.
    Defaults to floor(1/position_pct) = 30, the count a sleeve needs to fully
    deploy its own allocated capital at position_pct sizing. The global cap
    across all sleeves is max_positions * n_positive_sleeves -- see
    max_new_positions.
    """

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

    @property
    def peak_value(self) -> float | None:
        return self._peak_value

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

    def max_new_positions(self, current_count: int, weights: dict[str, float]) -> int:
        """How many new positions can be opened, across all sleeves.

        The global cap is the sum of the per-sleeve slot budgets (see
        sleeve_slot_caps) -- max_positions * n_positive_sleeves -- not a flat
        max_positions. weights must be the same mapping passed to
        sleeve_slot_caps so the two stay consistent within one call.
        """
        total_cap = sum(self.sleeve_slot_caps(weights).values())
        return max(0, total_cap - current_count)

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
        """Every positive-weight sleeve gets the SAME slot budget.

        Each sleeve with weight_i > 0 gets max_positions slots -- NOT
        floor(weight_i * max_positions). The number of concurrent positions a
        sleeve needs to fully deploy its allocated capital is ~1/position_pct,
        independent of its weight: a low-weight sleeve holds proportionally
        smaller positions (sleeve_position_notional already scales dollar
        size by weight * scale), not fewer of them. Applying weight a second
        time here -- the historical bug -- squared its effect on max
        deployable exposure: cap_i * notional_i ~ weight_i**2 instead of
        weight_i, so full deployment topped out at
        scale * position_pct * sum(weight_i**2) of portfolio (~26% with the
        live sp500/midcap400/nasdaq100 weights) instead of the intended
        scale * sum(weight_i) (~75-99%). Zero-weight sleeves get 0 slots.

        Invariant: sum(caps.values()) == max_positions * n_positive_sleeves.
        That sum IS the global position cap this system uses -- see
        max_new_positions -- there is no separate, smaller global cap to also
        respect here, unlike the old proportional scheme.
        """
        return {label: (self.cfg.max_positions if w > 0 else 0) for label, w in weights.items()}

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
