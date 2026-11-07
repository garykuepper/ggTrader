"""Circuit breaker, shadow re-entry and walk-forward efficiency (WFE) for the walk-forward.

The policy half of the legacy ``lab.wfo`` moved here unchanged: it is ggTrader glue that
vectorbt has no equivalent for (spec 5.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Literal


def compute_wfe(
    is_sharpe: float,
    oos_sharpe: float,
    is_floor: float = 0.4,
) -> float | None:
    """Walk-Forward Efficiency: OOS Sharpe / IS Sharpe.

    Returns None for neutral windows (IS Sharpe < floor): they are excluded from the rolling
    WFE average, not counted as zero.
    """
    if is_sharpe < is_floor:
        return None
    return oos_sharpe / is_sharpe


@dataclass(frozen=True)
class WfoPolicy:
    """Optional changes to WFO breaker semantics."""

    nan_wfe: Literal["legacy", "exclude"] = "legacy"
    anchor_fold_wfe: Literal["legacy", "exclude_from_breaker"] = "legacy"


@dataclass
class WfoState:
    """Tracks WFE history and circuit breaker state across WFO folds."""

    wfe_history: List[float | None] = field(default_factory=list)
    oos_sharpes: List[float] = field(default_factory=list)
    halted: bool = False
    halt_reason: str | None = None
    #: Rolling clean/dirty flags for the last 3 shadow windows during a halt.
    shadow_window: List[bool] = field(default_factory=list)


def _copy_state(state: WfoState) -> WfoState:
    return WfoState(
        wfe_history=list(state.wfe_history),
        oos_sharpes=list(state.oos_sharpes),
        halted=state.halted,
        halt_reason=state.halt_reason,
        shadow_window=list(state.shadow_window),
    )


def check_circuit_breaker(
    state: WfoState,
    window_size: int = 4,
    wfe_floor: float = 0.25,
    nan_policy: Literal["legacy", "exclude"] = "legacy",
) -> WfoState:
    """Check the OR-gate circuit breaker.

    Triggers halt if EITHER:
    1. Chronic decay: trailing window_size non-None WFE average < wfe_floor
    2. Acute failure: two consecutive negative OOS Sharpe Ratios
    """
    new_state = _copy_state(state)

    valid_wfes = [
        w
        for w in state.wfe_history
        if w is not None and (nan_policy == "legacy" or not math.isnan(w))
    ]
    recent = valid_wfes[-window_size:] if len(valid_wfes) >= window_size else valid_wfes
    if len(recent) >= window_size:
        avg_wfe = sum(recent) / len(recent)
        if avg_wfe < wfe_floor:
            new_state.halted = True
            new_state.halt_reason = (
                f"Chronic decay: trailing {window_size}-window WFE avg {avg_wfe:.3f} < {wfe_floor}"
            )
            return new_state

    if len(state.oos_sharpes) >= 2:
        if state.oos_sharpes[-1] < 0 and state.oos_sharpes[-2] < 0:
            new_state.halted = True
            new_state.halt_reason = (
                f"Acute failure: 2 consecutive negative OOS Sharpe"
                f" ({state.oos_sharpes[-2]:.2f}, {state.oos_sharpes[-1]:.2f})"
            )
            return new_state

    return new_state


def check_shadow_reentry(
    state: WfoState,
    ndh_passed: bool,
    dsr_passed: bool,
    wfe: float | None,
    wfe_healthy: float = 0.5,
    window_size: int = 3,
    clean_required: int = 2,
) -> WfoState:
    """Shadow re-entry: 2 of the last 3 clean windows restore live trading.

    A clean window requires all three: NDH pass, DSR pass, WFE >= wfe_healthy. "k of the last
    n" rather than consecutive windows because on a noisy universe good strategies alternate
    clean and dirty windows, so a tripped halt otherwise never releases.
    """
    new_state = _copy_state(state)

    clean = ndh_passed and dsr_passed and wfe is not None and wfe >= wfe_healthy
    new_state.shadow_window.append(clean)
    new_state.shadow_window = new_state.shadow_window[-window_size:]

    if sum(new_state.shadow_window) >= clean_required:
        new_state.halted = False
        new_state.halt_reason = None
        new_state.shadow_window = []

    return new_state
