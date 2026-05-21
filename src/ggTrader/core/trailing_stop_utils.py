"""Post-buy stop-pct derivation for the crypto execution engine."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional


def compute_post_buy_stop_pct(
    symbol: str,
    coin_params: Dict[str, Any],
    exit_name: str,
    fill_price: float,
    atr_value: float = float("nan"),
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Final stop-pct for placing the actual exit order after a fill.

    Per exit type:
      - ``atr_trailing``: ``atr_mult * atr / fill_price * 100``; falls back to
        ``atr_multiplier`` as a percentage if ATR is unavailable.
      - ``trailing_stop``: WFO ``trailing_stop_pct``.
      - ``fixed_sl_tp``: WFO ``stop_pct`` (also used for the
        belt-and-suspenders ``fixed_sl_tp → trailing`` conversion).

    Trailing types are floor-clamped to ``MIN_ATR_TRAILING_PCT`` /
    ``MIN_TRAILING_STOP_PCT`` so an aggressive WFO param can't get decapitated
    in the first bar. The same floor is applied to converted ``fixed_sl_tp``
    placements (treated as ``trailing_stop`` post-conversion).
    """
    cfg = config or {}
    log = logger or logging.getLogger(__name__)

    if exit_name == "atr_trailing":
        atr_mult = float(coin_params.get("atr_multiplier", 3.0))
        if not math.isnan(atr_value) and fill_price and atr_value > 0:
            stop_pct = (atr_mult * atr_value / fill_price) * 100.0
            log.info(
                f"  [Stop] {symbol} atr_trailing: ATR=${atr_value:.6g} "
                f"× {atr_mult} ÷ ${fill_price:.6g} -> {stop_pct:.2f}%"
            )
        else:
            stop_pct = atr_mult
            log.warning(
                f"  [Stop] {symbol} atr_trailing: ATR unavailable "
                f"— falling back to atr_multiplier={stop_pct:.2f}%"
            )
    elif exit_name == "trailing_stop":
        stop_pct = float(coin_params.get("trailing_stop_pct", 3.0))
    else:  # fixed_sl_tp (or unknown — treat as fixed)
        stop_pct = float(coin_params.get("stop_pct", 3.0))

    floor_key: Optional[str] = None
    if exit_name == "atr_trailing":
        floor_key = "MIN_ATR_TRAILING_PCT"
    elif exit_name in ("trailing_stop", "fixed_sl_tp"):
        # fixed_sl_tp gets the trailing-stop floor because we convert it to a
        # Kraken-native trailing-stop at placement time.
        floor_key = "MIN_TRAILING_STOP_PCT"

    if floor_key is not None:
        floor = float(cfg.get(floor_key, 4.0))
        if stop_pct < floor:
            log.info(
                f"  [Stop] {symbol}: stop_pct {stop_pct:.2f}% clamped to {floor_key}={floor:.2f}%"
            )
            stop_pct = floor

    return stop_pct
