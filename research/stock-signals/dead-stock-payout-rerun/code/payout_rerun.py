"""Rerun grid, reproduction anchor, identity check and gap decomposition for dead-stock payouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import ft_dead_stock_first_test as ft
import pandas as pd
from ft_dead_stock_first_test import (
    COSTS,
    CRSP_END,
    DECOMPOSITION_COST,
    DECOMPOSITION_TAX,
    LOOKBACK,
    SKIP,
    START_CASH,
    TAXES,
    TOP_N,
    _cell,
    _naive_stamp,
    _normalise_spans,
    _window_days,
    coverage_block,
    decomposition_for_loss,
    drop_file_end_rows,
    holdings_audit,
    loss_monotone_check,
    permno_double_holds,
    spy_key,
    spy_zero_cost_check,
    validate_window_inputs,
)
from ft_ledger_run import BookResult, TaxParams, run_book, run_spy
from ft_replay import DEAD_EXIT_REASON, TargetStep, plan_trades, span_end_dates
from payout_prices import (
    ANCHOR_SET,
    EVIDENCE_SETS,
    READINGS,
    coverage_of,
    exit_prices,
    held_dead_symbols,
    pinned_losses,
)

ANCHOR_READINGS: tuple[float, ...] = (1.0, 0.0)
ANCHOR_COST = DECOMPOSITION_COST
ANCHOR_TAX = DECOMPOSITION_TAX
HEADLINE_SET = "confirmed"
IDENTITY_GAP_TOL_PP = 1e-9
RANGE_TOL_PP = 1e-9
ANCHOR_SUMMARY_FIELDS = (
    "after_tax_final",
    "cagr_after_tax",
    "gap_cagr_after_tax_pp",
    "tax_paid",
    "cost_paid",
    "n_dead_exits",
)


def cell_key(evidence_set: str, reading: float, cost: float, tax: str) -> str:
    """Return the dictionary key of one rerun strategy cell."""
    return f"{evidence_set}_{ft.cell_key(reading, cost, tax)}"


def _compare_final_values(
    rerun_cell: Mapping[str, Any], saved_cell: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Compare every final-value field present in the cell after rounding both sides to cents."""
    compared: dict[str, dict[str, Any]] = {}
    all_ok = True

    def check_scalar(name: str, rerun_val: float, saved_val: float) -> None:
        nonlocal all_ok
        r_cents = round(float(rerun_val), 2)
        s_cents = round(float(saved_val), 2)
        ok = r_cents == s_cents
        compared[name] = {
            "rerun": float(rerun_val),
            "first_test": float(saved_val),
            "rerun_cents": r_cents,
            "first_test_cents": s_cents,
            "matches": ok,
        }
        all_ok = all_ok and ok

    # after_tax_final is the sole scalar final-value field returned by _cell.
    check_scalar("after_tax_final", rerun_cell["after_tax_final"], saved_cell["after_tax_final"])

    # Also check the last month-end account value when present in both records.
    rerun_series = rerun_cell.get("account_value_month_end")
    saved_series = saved_cell.get("account_value_month_end")
    if rerun_series and saved_series:
        check_scalar("account_value_month_end_last", rerun_series[-1][1], saved_series[-1][1])

    return compared, all_ok


def _anchor_summary(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Compact scalar summary of a cell stored inside the anchor comparison block."""
    return {field: cell[field] for field in ANCHOR_SUMMARY_FIELDS if field in cell}


def _build_anchor(
    close: pd.DataFrame,
    spy_close: pd.Series,
    steps: list[TargetStep],
    ends: pd.Series,
    spans: pd.DataFrame,
    successors: frozenset[str],
    anchor_tax_params: TaxParams,
    first_test: Mapping[str, Any],
    *,
    start: str,
    end: str,
    start_cash: float,
) -> tuple[dict[str, Any], bool, BookResult, pd.DataFrame]:
    """Run the no-pin reproduction anchor before any evidence-set cell is computed.

    Returns:
        ``(anchor_block, anchor_ok, anchor_spy_book, no_pin_loss0_prices)``.
        The SPY book and reading-0.0 no-pin price frame are reused downstream so neither is
        recomputed.
    """
    anchor_spy_book = run_spy(
        spy_close,
        params=anchor_tax_params,
        one_way_cost=ANCHOR_COST,
        start=start,
        end=end,
        start_cash=start_cash,
    )
    anchor_spy_cell = {"one_way_cost": ANCHOR_COST, "tax": ANCHOR_TAX, **_cell(anchor_spy_book)}
    ft_spy_key = spy_key(ANCHOR_COST, ANCHOR_TAX)
    saved_spy = first_test["spy"][ft_spy_key]
    spy_compared, spy_ok = _compare_final_values(anchor_spy_cell, saved_spy)

    strategy_anchor: dict[str, dict[str, Any]] = {}
    all_ok = spy_ok
    no_pin_loss0_prices: pd.DataFrame | None = None

    for reading in ANCHOR_READINGS:
        px = exit_prices(close, spans, reading, {})
        if reading == 0.0:
            no_pin_loss0_prices = px
        book = run_book(
            close,
            steps,
            ends,
            px,
            params=anchor_tax_params,
            one_way_cost=ANCHOR_COST,
            start=start,
            end=end,
            successors=successors,
            start_cash=start_cash,
        )
        rerun_cell = {
            "evidence_set": ANCHOR_SET,
            "loss": reading,
            "one_way_cost": ANCHOR_COST,
            "tax": ANCHOR_TAX,
            **_cell(book),
            "gap_cagr_after_tax_pp": (book.cagr_after_tax - anchor_spy_book.cagr_after_tax) * 100.0,
        }
        ft_key = ft.cell_key(reading, ANCHOR_COST, ANCHOR_TAX)
        saved_cell = first_test["strategy"][ft_key]
        compared, ok = _compare_final_values(rerun_cell, saved_cell)
        all_ok = all_ok and ok
        strategy_anchor[ft_key] = {
            "reading": reading,
            "rerun": _anchor_summary(rerun_cell),
            "first_test": _anchor_summary(saved_cell),
            "compared_fields": compared,
            "matches": ok,
        }

    assert no_pin_loss0_prices is not None
    anchor_block = {
        "evidence_set": ANCHOR_SET,
        "one_way_cost": ANCHOR_COST,
        "tax": ANCHOR_TAX,
        "strategy": strategy_anchor,
        "spy": {
            ft_spy_key: {
                "rerun": _anchor_summary(anchor_spy_cell),
                "first_test": _anchor_summary(saved_spy),
                "compared_fields": spy_compared,
                "matches": spy_ok,
            }
        },
    }
    return anchor_block, all_ok, anchor_spy_book, no_pin_loss0_prices


def _build_identity_check(
    strategy: Mapping[str, Mapping[str, Any]], first_test: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare every reading-0.0 rerun cell against the first test's saved loss-0.0 cell."""
    checks: dict[str, dict[str, Any]] = {}
    mismatches: list[dict[str, Any]] = []
    for ev in EVIDENCE_SETS:
        for cost in COSTS:
            for tax in TAXES:
                key = cell_key(ev, 0.0, cost, tax)
                ft_key = ft.cell_key(0.0, cost, tax)
                cell = strategy[key]
                saved = first_test["strategy"][ft_key]
                r_final = float(cell["after_tax_final"])
                s_final = float(saved["after_tax_final"])
                r_gap = float(cell["gap_cagr_after_tax_pp"])
                s_gap = float(saved["gap_cagr_after_tax_pp"])
                final_ok = round(r_final, 2) == round(s_final, 2)
                gap_ok = abs(r_gap - s_gap) <= IDENTITY_GAP_TOL_PP
                ok = final_ok and gap_ok
                entry = {
                    "first_test_cell": ft_key,
                    "evidence_set": ev,
                    "one_way_cost": cost,
                    "tax": tax,
                    "rerun_after_tax_final": r_final,
                    "first_test_after_tax_final": s_final,
                    "rerun_gap_cagr_after_tax_pp": r_gap,
                    "first_test_gap_cagr_after_tax_pp": s_gap,
                    "final_matches": final_ok,
                    "gap_matches": gap_ok,
                    "matches": ok,
                }
                checks[key] = entry
                if not ok:
                    mismatches.append({"cell": key, **entry})
    return {
        "checks": checks,
        "mismatches": mismatches,
        "identity_ok": len(mismatches) == 0,
    }


def _build_lead_numbers(
    strategy: Mapping[str, Mapping[str, Any]], first_test: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, float]]:
    """Headline three gaps for confirmed @ 18.3 bps low tax, alongside the first test's range."""
    worst = float(
        strategy[cell_key(HEADLINE_SET, 1.0, DECOMPOSITION_COST, DECOMPOSITION_TAX)][
            "gap_cagr_after_tax_pp"
        ]
    )
    middle = float(
        strategy[cell_key(HEADLINE_SET, 0.5, DECOMPOSITION_COST, DECOMPOSITION_TAX)][
            "gap_cagr_after_tax_pp"
        ]
    )
    best = float(
        strategy[cell_key(HEADLINE_SET, 0.0, DECOMPOSITION_COST, DECOMPOSITION_TAX)][
            "gap_cagr_after_tax_pp"
        ]
    )
    ft_worst = float(
        first_test["strategy"][ft.cell_key(1.0, DECOMPOSITION_COST, DECOMPOSITION_TAX)][
            "gap_cagr_after_tax_pp"
        ]
    )
    ft_best = float(
        first_test["strategy"][ft.cell_key(0.0, DECOMPOSITION_COST, DECOMPOSITION_TAX)][
            "gap_cagr_after_tax_pp"
        ]
    )
    lo, hi = min(worst, best), max(worst, best)
    wider = abs(best - worst) > abs(ft_best - ft_worst) + RANGE_TOL_PP
    lead_numbers = {
        "worst": worst,
        "middle": middle,
        "best": best,
        "range_wider_than_first_test": bool(wider),
        "middle_between_worst_and_best": bool(lo - RANGE_TOL_PP <= middle <= hi + RANGE_TOL_PP),
    }
    first_test_range = {
        "worst": ft_worst,
        "best": ft_best,
        "width_pp": abs(ft_best - ft_worst),
    }
    return lead_numbers, first_test_range


def run_rerun(
    close: pd.DataFrame,
    spy_close: pd.Series,
    eligible: pd.DataFrame,
    span_rows: pd.DataFrame,
    rows: Sequence[Mapping[str, str]],
    first_test: Mapping[str, Any],
    *,
    start: str,
    end: str,
    release_spans: pd.DataFrame | None = None,
    start_cash: float = START_CASH,
    trade_sink: dict[str, list] | None = None,
) -> dict[str, Any]:
    """Run the 36-cell payout rerun grid, reproduction anchor, checks and decomposition.

    Args:
        close: Daily closes, trading days by release columns, with warm-up history.
        spy_close: SPY daily closes.
        eligible: Boolean eligibility frame.
        span_rows: Ended-series rows (``symbol``, ``last_date``, ``has_successor``,
            ``total_loss``).
        rows: Payout-table rows (text mappings with ``symbol``, ``suggested_loss``,
            ``match_confidence``).
        first_test: Saved first-test window result dict (containing ``strategy`` and ``spy``).
        start: First day of the window, ``YYYY-MM-DD``.
        end: Last day of the window, ``YYYY-MM-DD``.
        release_spans: Optional release spans with ``permno`` and ``kind`` for audit checks.
        start_cash: Opening cash.
        trade_sink: If provided, populated with every grid cell's trade list (SPY cells prefixed
            ``spy_``).

    Returns:
        Result dictionary with ``anchor`` first, followed by ``anchor_ok``, the 36 ``strategy``
        cells, the 6 ``spy`` cells, ``identity_check``, ``coverage_by_cell``, ``lead_numbers``,
        ``first_test_range``, ``pinned_share_cell``, ``gap_decomposition`` and audit blocks.
    """
    window_check = validate_window_inputs(close, spy_close, start, end)
    spans, n_censored = drop_file_end_rows(_normalise_spans(span_rows), end)
    if pd.Timestamp(end) < pd.Timestamp(CRSP_END) and n_censored:
        raise AssertionError("a window that ends before the CRSP file ends has censored rows")

    # Parse pinned maps up front so any bad payout row fails before running the ledger.
    pinned_by_set = {ev: pinned_losses(rows, ev) for ev in EVIDENCE_SETS}
    table_symbols = [
        str(row.get("symbol") or "").strip() for row in rows if str(row.get("symbol") or "").strip()
    ]

    steps: list[TargetStep] = plan_trades(
        close,
        eligible,
        spans,
        top_n=TOP_N,
        lookback=LOOKBACK,
        skip=SKIP,
        dead_loss=READINGS[0],
        start=start,
        end=end,
    )
    ends = span_end_dates(close, spans)
    flagged = spans.drop_duplicates("symbol").set_index("symbol")["has_successor"]
    successors = frozenset(s for s in ends.index if bool(flagged.get(s, False)))
    taxes = {name: TaxParams(*rates) for name, rates in TAXES.items()}
    no_tax = TaxParams(0.0, 0.0)

    # 1. Reproduction anchor BEFORE any evidence-set cell is computed.
    anchor, anchor_ok, anchor_spy_book, no_pin_loss0_prices = _build_anchor(
        close,
        spy_close,
        steps,
        ends,
        spans,
        successors,
        taxes[ANCHOR_TAX],
        first_test,
        start=start,
        end=end,
        start_cash=start_cash,
    )

    # 2. The 6 SPY cells (reusing the anchor's SPY book for the anchor cost/tax pair).
    spy: dict[str, dict[str, Any]] = {}
    all_trades: dict[str, list] = {}
    for cost in COSTS:
        for tax, params in taxes.items():
            if cost == ANCHOR_COST and tax == ANCHOR_TAX:
                book = anchor_spy_book
            else:
                book = run_spy(
                    spy_close,
                    params=params,
                    one_way_cost=cost,
                    start=start,
                    end=end,
                    start_cash=start_cash,
                )
            spy[spy_key(cost, tax)] = {"one_way_cost": cost, "tax": tax, **_cell(book)}
            all_trades[f"spy_{spy_key(cost, tax)}"] = book.trades

    # Baseline zero-cost zero-tax books for gap decomposition: SPY and unpinned reading 0.0.
    spy_zero = (
        run_spy(
            spy_close, params=no_tax, one_way_cost=0.0, start=start, end=end, start_cash=start_cash
        ).cagr_after_tax
        * 100.0
    )
    rule_loss0_zero = (
        run_book(
            close,
            steps,
            ends,
            no_pin_loss0_prices,
            params=no_tax,
            one_way_cost=0.0,
            start=start,
            end=end,
            successors=successors,
            start_cash=start_cash,
        ).cagr_after_tax
        * 100.0
    )
    del no_pin_loss0_prices

    spy_final = spy[spy_key(DECOMPOSITION_COST, DECOMPOSITION_TAX)]["cagr_after_tax"] * 100.0

    # 3. The 36 strategy cells plus per-(evidence_set, reading) gap decomposition.
    # Build one exit_prices frame per (evidence_set, reading) and drop it immediately after
    # its books finish to keep peak memory small on full-history runs.
    strategy: dict[str, dict[str, Any]] = {}
    coverage_by_cell: dict[str, dict[str, Any]] = {}
    by_evidence_set: dict[str, dict[str, dict[str, Any]]] = {ev: {} for ev in EVIDENCE_SETS}
    doubles: list[dict[str, Any]] = []
    checked = False

    for ev in EVIDENCE_SETS:
        pinned = pinned_by_set[ev]
        for reading in READINGS:
            px = exit_prices(close, spans, reading, pinned)
            for cost in COSTS:
                for tax, params in taxes.items():
                    book = run_book(
                        close,
                        steps,
                        ends,
                        px,
                        params=params,
                        one_way_cost=cost,
                        start=start,
                        end=end,
                        successors=successors,
                        start_cash=start_cash,
                    )
                    key = cell_key(ev, reading, cost, tax)
                    comparison = spy[spy_key(cost, tax)]["cagr_after_tax"]
                    strategy[key] = {
                        "evidence_set": ev,
                        "loss": reading,
                        "one_way_cost": cost,
                        "tax": tax,
                        **_cell(book),
                        "gap_cagr_after_tax_pp": (book.cagr_after_tax - comparison) * 100.0,
                    }
                    all_trades[key] = book.trades
                    held = held_dead_symbols(book.trades, successors)
                    coverage_by_cell[key] = coverage_of(held, pinned, table_symbols)
                    check = permno_double_holds(book.trades, release_spans)
                    checked = checked or check["checked"]
                    doubles += [
                        {"evidence_set": ev, "loss": reading, "cost": cost, "tax": tax, **v}
                        for v in check["violations"]
                    ]

            rule_zero = (
                run_book(
                    close,
                    steps,
                    ends,
                    px,
                    params=no_tax,
                    one_way_cost=0.0,
                    start=start,
                    end=end,
                    successors=successors,
                    start_cash=start_cash,
                ).cagr_after_tax
                * 100.0
            )
            rule_cost_no_tax = (
                run_book(
                    close,
                    steps,
                    ends,
                    px,
                    params=no_tax,
                    one_way_cost=DECOMPOSITION_COST,
                    start=start,
                    end=end,
                    successors=successors,
                    start_cash=start_cash,
                ).cagr_after_tax
                * 100.0
            )
            final_cell = strategy[cell_key(ev, reading, DECOMPOSITION_COST, DECOMPOSITION_TAX)]
            part = decomposition_for_loss(
                spy_zero=spy_zero,
                rule_loss0_zero=rule_loss0_zero,
                rule_zero=rule_zero,
                rule_cost_no_tax=rule_cost_no_tax,
                rule_final=final_cell["cagr_after_tax"] * 100.0,
                spy_final=spy_final,
            )
            part["matches_cell_gap"] = bool(
                abs(part["total_gap_pp"] - final_cell["gap_cagr_after_tax_pp"]) < 1e-9
            )
            by_evidence_set[ev][f"loss{reading:g}"] = part
            del px

    identity_check = _build_identity_check(strategy, first_test)
    lead_numbers, first_test_range = _build_lead_numbers(strategy, first_test)

    ref_key = cell_key(HEADLINE_SET, 1.0, DECOMPOSITION_COST, DECOMPOSITION_TAX)
    pinned_share_cell = dict(coverage_by_cell[ref_key])
    reference_trades = all_trades[ref_key]
    held_at_death = sum(
        1 for t in reference_trades if t.reason == DEAD_EXIT_REASON and t.symbol not in successors
    )
    monotone_violations: list[dict[str, Any]] = []
    for ev in EVIDENCE_SETS:
        sliced = {
            ft.cell_key(reading, cost, tax): strategy[cell_key(ev, reading, cost, tax)]
            for reading in READINGS
            for cost in COSTS
            for tax in TAXES
        }
        for v in loss_monotone_check(sliced)["violations"]:
            monotone_violations.append({"evidence_set": ev, **v})
    sanity = {
        "spy_zero_cost_growth": spy_zero_cost_check(spy_close, start, end),
        "loss_monotone": {"violations": monotone_violations},
        "held_at_death": held_at_death,
        "no_dead_stock_held": held_at_death == 0,
        "one_permno_one_column": {"checked": checked, "violations": doubles},
        "reference_cell": ref_key,
    }
    gap_decomposition = {
        "cost": DECOMPOSITION_COST,
        "tax": DECOMPOSITION_TAX,
        "note": (
            "Steps in yearly growth, percentage points: SPY with no cost and no tax, then the "
            "rule's stock picking (no pins, reading 0.0, no cost, no tax), then dead-stock losses "
            "(pinned figures and reading together), then trading costs, then taxes. SPY's own cost "
            "and tax drag is its own line."
        ),
        "by_evidence_set": by_evidence_set,
    }

    days = _window_days(close.index, start, end)
    month_ends = list(pd.Series(days, index=days).groupby(days.to_period("M")).last())
    kinds = None
    if release_spans is not None and "kind" in release_spans.columns:
        kinds = dict(zip(release_spans["symbol"], release_spans["kind"], strict=False))
    holdings = {
        cell_key(HEADLINE_SET, reading, DECOMPOSITION_COST, DECOMPOSITION_TAX): holdings_audit(
            all_trades[cell_key(HEADLINE_SET, reading, DECOMPOSITION_COST, DECOMPOSITION_TAX)],
            month_ends,
            ends,
            successors,
            kinds,
        )
        for reading in READINGS
    }
    if trade_sink is not None:
        trade_sink.update(all_trades)

    low, high = pd.Timestamp(start), pd.Timestamp(end)
    in_window = [t for t in ends if low <= _naive_stamp(t) <= high]
    return {
        "anchor": anchor,
        "anchor_ok": anchor_ok,
        "strategy": strategy,
        "spy": spy,
        "identity_check": identity_check,
        "coverage_by_cell": coverage_by_cell,
        "lead_numbers": lead_numbers,
        "first_test_range": first_test_range,
        "pinned_share_cell": pinned_share_cell,
        "gap_decomposition": gap_decomposition,
        "coverage": coverage_block(eligible, close, ends, successors, reference_trades, start, end),
        "sanity": sanity,
        "window_check": window_check,
        "holdings_audit": holdings,
        "payout_table": {
            "rows": len(rows),
            "pinned_confirmed": len(pinned_by_set["confirmed"]),
            "pinned_all": len(pinned_by_set["all"]),
        },
        "n_file_end_censored": n_censored,
        "crsp_end": CRSP_END,
        "n_rebalance_steps": len(steps),
        "n_columns": int(close.shape[1]),
        "n_ended_series_in_window": len(in_window),
        "n_ended_series_in_read": int(len(ends)),
        "skipped_buys_note": (
            "skipped_buys counts every target that could not be bought (no price that day, or "
            "its series had ended); the ledger does not split the reasons."
        ),
    }
