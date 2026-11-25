"""Anchor comparison, SPY gap table, headline gaps and price-path pass-through."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import fp_dead_stock_first_test as study

HALF_CENT = 0.005  # dollars: maximum allowed drift on after_tax_final for anchor reproduction
CELL_SECTIONS: tuple[str, ...] = ("strategy", "spy")
FINAL_VALUE_FIELD = "after_tax_final"
CAGR_FIELD = "cagr_after_tax"
DRAWDOWN_FIELD = "max_drawdown_pct"
SHARPE_FIELD = "sharpe"
PRICE_PATH_KEY = "death_price_paths"
HEADLINE_COST = study.DECOMPOSITION_COST  # 0.00183 (18.3 bps one way)
HEADLINE_TAX = study.DECOMPOSITION_TAX  # "low"
HEADLINE_LOSSES: tuple[tuple[str, float], ...] = (
    ("worst", 1.0),
    ("middle", 0.5),
    ("best", 0.0),
)
BPS_PER_UNIT = 1e4
PERCENT_POINTS = 100.0


def _compare_cell_final(
    section: str,
    key: str,
    result_cell: Any,
    saved_cell: Any,
) -> str | None:
    """Compare one cell's after-tax final dollar value to the saved value within half a cent."""
    if not isinstance(result_cell, Mapping):
        return f"{section} cell {key}: result cell is not a mapping"
    if not isinstance(saved_cell, Mapping):
        return f"{section} cell {key}: saved cell is not a mapping"
    if FINAL_VALUE_FIELD not in result_cell or result_cell[FINAL_VALUE_FIELD] is None:
        return f"{section} cell {key}: missing {FINAL_VALUE_FIELD} in result"
    if FINAL_VALUE_FIELD not in saved_cell or saved_cell[FINAL_VALUE_FIELD] is None:
        return f"{section} cell {key}: missing {FINAL_VALUE_FIELD} in saved"

    actual = float(result_cell[FINAL_VALUE_FIELD])
    expected = float(saved_cell[FINAL_VALUE_FIELD])
    if not (math.isfinite(actual) and math.isfinite(expected)):
        return (
            f"{section} cell {key}: non-finite {FINAL_VALUE_FIELD} "
            f"(result={actual}, saved={expected})"
        )

    diff = actual - expected
    # Compare raw unrounded dollars against half a cent rather than rounded-cent equality so a
    # tiny floating-point jitter near a .005 boundary cannot trigger a false mismatch.
    if abs(diff) > HALF_CENT:
        return (
            f"{section} cell {key}: {FINAL_VALUE_FIELD} differs by {diff:+.6f} "
            f"(result={actual:.6f}, saved={expected:.6f}, tolerance={HALF_CENT})"
        )
    return None


def compare_to_saved(result: dict[str, Any], saved: dict[str, Any]) -> list[str]:
    """Compare a freshly computed result against the saved first-test file to within half a cent.

    Checks every strategy cell (18) and SPY cell (6) under ``saved["result"]`` (or ``saved``
    directly if already unwrapped). Reports any missing cell, extra cell, or ``after_tax_final``
    difference larger than half a cent.

    Args:
        result: Result dictionary produced by ``fp_dead_stock_first_test.run_study``.
        saved: Loaded saved first-test JSON payload (with ``saved["result"]``) or an unwrapped
            result dictionary.

    Returns:
        List of plain-word difference messages; empty when every cell reproduces within half a
        cent.
    """
    saved_body = saved["result"] if "result" in saved else saved
    if not isinstance(saved_body, Mapping):
        return ["saved result block is not a mapping"]
    if not isinstance(result, Mapping):
        return ["result block is not a mapping"]

    messages: list[str] = []
    for section in CELL_SECTIONS:
        saved_cells = saved_body.get(section)
        result_cells = result.get(section)
        if not isinstance(saved_cells, Mapping):
            messages.append(f"{section}: missing or invalid in saved result")
            continue
        if not isinstance(result_cells, Mapping):
            messages.append(f"{section}: missing or invalid in result")
            continue
        if not saved_cells:
            messages.append(f"{section}: saved result has no cells")
            continue

        for key in saved_cells:
            if key not in result_cells:
                messages.append(f"{section} cell {key}: missing from result")
        for key in sorted(set(result_cells) - set(saved_cells)):
            messages.append(f"{section} cell {key}: extra cell not in saved")
        for key, saved_cell in saved_cells.items():
            if key not in result_cells:
                continue
            diff_msg = _compare_cell_final(section, key, result_cells[key], saved_cell)
            if diff_msg is not None:
                messages.append(diff_msg)

    return messages


def gap_table(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build one gap-to-SPY row per registered strategy cell in canonical grid order.

    Every gap uses the same sign convention: rule minus SPY. Growth-rate gaps are in percentage
    points a year (``cagr_after_tax`` difference times 100); drawdown gaps subtract
    ``max_drawdown_pct`` directly because it is already stored in percentage points.

    Args:
        result: Result dictionary containing the 18 ``strategy`` cells and 6 ``spy`` cells.

    Returns:
        List of 18 row dictionaries with keys ``cell``, ``loss``, ``cost_bps``, ``tax``,
        ``cagr_gap_pts``, ``drawdown_gap_pts`` and ``sharpe_gap``.
    """
    strategy = result["strategy"]
    spy = result["spy"]
    rows: list[dict[str, Any]] = []
    for loss in study.LOSSES:
        for cost in study.COSTS:
            cost_bps = round(float(cost) * BPS_PER_UNIT, 6)
            for tax in study.TAXES:
                key = study.cell_key(loss, cost, tax)
                cell = strategy[key]
                spy_cell = spy[study.spy_key(cost, tax)]
                rows.append(
                    {
                        "cell": key,
                        "loss": float(loss),
                        "cost_bps": cost_bps,
                        "tax": tax,
                        "cagr_gap_pts": (float(cell[CAGR_FIELD]) - float(spy_cell[CAGR_FIELD]))
                        * PERCENT_POINTS,
                        "drawdown_gap_pts": (
                            float(cell[DRAWDOWN_FIELD]) - float(spy_cell[DRAWDOWN_FIELD])
                        ),
                        "sharpe_gap": float(cell[SHARPE_FIELD]) - float(spy_cell[SHARPE_FIELD]),
                    }
                )
    return rows


def headline(result: dict[str, Any]) -> dict[str, float]:
    """Return the three registered headline growth-rate gaps to SPY (18.3 bps, low tax).

    Args:
        result: Result dictionary containing the registered headline strategy cells and matching
            SPY cell.

    Returns:
        Dictionary with keys ``worst`` (loss 1.0), ``middle`` (loss 0.5) and ``best`` (loss 0.0),
        each in percentage points a year (rule minus SPY).
    """
    strategy = result["strategy"]
    spy_cell = result["spy"][study.spy_key(HEADLINE_COST, HEADLINE_TAX)]
    spy_cagr = float(spy_cell[CAGR_FIELD])
    return {
        label: (
            float(strategy[study.cell_key(loss, HEADLINE_COST, HEADLINE_TAX)][CAGR_FIELD])
            - spy_cagr
        )
        * PERCENT_POINTS
        for label, loss in HEADLINE_LOSSES
    }


def price_path_block(result: dict[str, Any]) -> Any:
    """Return the result's dead-stock price-path block unchanged.

    Args:
        result: Result dictionary produced by ``fp_dead_stock_first_test.run_study``.

    Returns:
        The ``death_price_paths`` block exactly as stored in ``result``.
    """
    return result[PRICE_PATH_KEY]
