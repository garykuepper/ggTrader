"""Tests for the dead-stock payout rerun grid, anchor check, identity check and decomposition."""

from __future__ import annotations

import builtins
import copy
from pathlib import Path
from typing import Any

import ft_dead_stock_first_test as ft
import numpy as np
import pandas as pd
import payout_prices
import payout_rerun
import pytest

pytestmark = pytest.mark.study

SYNTH_START = "2019-09-02"
SYNTH_END = "2020-01-31"


def _synthetic_window() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a small 6-stock synthetic window with warmup and 4 dying stocks held by the rule.

    The warmup runs from 2018-01-02 through 2019-08-30 (>400 trading days for the 252-day
    lookback). The window runs 2019-09-02 through 2020-01-31 so month-end rebalances execute on
    the first trading days of October, November, December and January, and year-end tax runs on
    the first trading day of 2020. All 6 stocks are eligible and chosen (fewer than TOP_N=50),
    so all 4 dying stocks are held when they stop trading.
    """
    dates = pd.bdate_range("2018-01-02", SYNTH_END)
    n = len(dates)
    t = np.arange(n, dtype=float)

    # Gentle distinct trends so every stock has a valid finite momentum score.
    s_alive_1 = 50.0 * (1.0005**t)
    s_alive_2 = 40.0 * (1.0003**t)
    s_dead_1 = 30.0 * (1.0008**t)  # PERMNO-101: confirmed pinned
    s_dead_2 = 35.0 * (1.0006**t)  # PERMNO-102: name_only pinned
    s_dead_3 = 25.0 * (1.0007**t)  # PERMNO-103: blank loss in table
    s_dead_4 = 45.0 * (1.0004**t)  # PERMNO-104~2: missing from table (later spell)

    close = pd.DataFrame(
        {
            "ALIVE-1": s_alive_1,
            "ALIVE-2": s_alive_2,
            "PERMNO-101": s_dead_1,
            "PERMNO-102": s_dead_2,
            "PERMNO-103": s_dead_3,
            "PERMNO-104~2": s_dead_4,
        },
        index=dates,
    )
    death_dates = {
        "PERMNO-101": pd.Timestamp("2019-10-15"),
        "PERMNO-102": pd.Timestamp("2019-11-15"),
        "PERMNO-103": pd.Timestamp("2019-12-16"),
        "PERMNO-104~2": pd.Timestamp("2020-01-15"),
    }
    for sym, death in death_dates.items():
        close.loc[close.index > death, sym] = np.nan

    spy_close = pd.Series(100.0 * (1.0004**t), index=dates, name="SPY")
    eligible = pd.DataFrame(True, index=dates, columns=close.columns)
    for sym, death in death_dates.items():
        eligible.loc[eligible.index > death, sym] = False

    span_rows = pd.DataFrame(
        [
            {
                "symbol": sym,
                "last_date": death,
                "has_successor": False,
                "total_loss": False,
            }
            for sym, death in death_dates.items()
        ]
    )
    release_spans = pd.DataFrame(
        [
            {"symbol": "ALIVE-1", "permno": 1, "kind": "linked_yahoo"},
            {"symbol": "ALIVE-2", "permno": 2, "kind": "placeholder"},
            {"symbol": "PERMNO-101", "permno": 101, "kind": "placeholder"},
            {"symbol": "PERMNO-102", "permno": 102, "kind": "placeholder"},
            {"symbol": "PERMNO-103", "permno": 103, "kind": "placeholder"},
            {"symbol": "PERMNO-104~2", "permno": 104, "kind": "spell"},
        ]
    )
    return close, spy_close, eligible, span_rows, release_spans


def _synthetic_payout_rows(loss_101: str = "0.0") -> list[dict[str, str]]:
    """Payout rows: 101 confirmed, 102 name_only, 103 blank loss; 104 is missing from table."""
    return [
        {"symbol": "PERMNO-101", "suggested_loss": loss_101, "match_confidence": "confirmed"},
        {"symbol": "PERMNO-102", "suggested_loss": "0.0", "match_confidence": "name_only"},
        {"symbol": "PERMNO-103", "suggested_loss": "", "match_confidence": "confirmed"},
    ]


def _first_test_result(
    close: pd.DataFrame,
    spy_close: pd.Series,
    eligible: pd.DataFrame,
    span_rows: pd.DataFrame,
    release_spans: pd.DataFrame,
    *,
    start: str = SYNTH_START,
    end: str = SYNTH_END,
) -> dict[str, Any]:
    """Run the copied first-test engine on the same synthetic data to form ``first_test``."""
    return ft.run_study(
        close,
        spy_close,
        eligible,
        span_rows,
        start=start,
        end=end,
        release_spans=release_spans,
    )


def _tax_kink_window() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Synthetic fixture where reading 0.5 beats both reading 1.0 and reading 0.0 after tax.

    Mechanics: DEAD is bought on Oct 1 at 50, doubles to 100 by Nov 15, and dies unpinned.
    On Dec 2 the book puts all equity into CRASH, which drops 85 percent during December.
    On Jan 2 (first trading day of 2020) the 24 percent short-term tax on 2019 gains is paid.
    At reading 1.0 DEAD yields ~0 (no gain, no tax; only SAFE's half survives the crash).
    At reading 0.5 DEAD yields 50 (at cost basis, still 0 tax, more capital survives).
    At reading 0.0 DEAD yields 100 (+50 taxable gain: tax is 0.24 * 50 = 12, while the extra
    50 reinvested in CRASH keeps only 0.15 * 50 = 7.5, so net after tax is lower than at 0.5).
    """
    dates = pd.bdate_range("2018-01-02", SYNTH_END)
    n = len(dates)
    t = np.arange(n, dtype=float)

    safe = pd.Series(50.0, index=dates)
    dead = pd.Series(50.0, index=dates)
    # Rise DEAD after Oct 1 so purchase basis is ~50 and death close on Nov 15 is 100.
    dead.loc[dates >= pd.Timestamp("2019-10-10")] = 100.0
    dead_death = pd.Timestamp("2019-11-15")
    dead.loc[dates > dead_death] = np.nan

    crash = pd.Series(100.0, index=dates)
    # CRASH drops 85 percent between Dec 2 (rebalance buy) and Dec 31.
    crash.loc[dates >= pd.Timestamp("2019-12-10")] = 15.0

    close = pd.DataFrame({"SAFE": safe, "DEAD": dead, "CRASH": crash}, index=dates)
    spy_close = pd.Series(100.0 * (1.0001**t), index=dates, name="SPY")

    eligible = pd.DataFrame(False, index=dates, columns=close.columns)
    # September and October month-ends pick SAFE and DEAD (50/50); November picks only CRASH.
    eligible.loc[dates <= pd.Timestamp("2019-11-28"), ["SAFE", "DEAD"]] = True
    eligible.loc[dates >= pd.Timestamp("2019-11-29"), "CRASH"] = True

    span_rows = pd.DataFrame(
        [{"symbol": "DEAD", "last_date": dead_death, "has_successor": False, "total_loss": False}]
    )
    return close, spy_close, eligible, span_rows


def test_cell_key_format() -> None:
    assert (
        payout_rerun.cell_key("confirmed", 1.0, 0.00183, "low")
        == "confirmed_loss1_cost18.3bps_taxlow"
    )
    assert payout_rerun.cell_key("all", 0.5, 0.00064, "high") == "all_loss0.5_cost6.4bps_taxhigh"
    assert (
        payout_rerun.cell_key("confirmed", 0.0, 0.0040, "low") == "confirmed_loss0_cost40bps_taxlow"
    )


def test_run_rerun_returns_36_plus_6_cells_and_anchor_first() -> None:
    close, spy_close, eligible, span_rows, release_spans = _synthetic_window()
    first_test = _first_test_result(close, spy_close, eligible, span_rows, release_spans)
    rows = _synthetic_payout_rows()
    sink: dict[str, list] = {}

    result = payout_rerun.run_rerun(
        close,
        spy_close,
        eligible,
        span_rows,
        rows,
        first_test,
        start=SYNTH_START,
        end=SYNTH_END,
        release_spans=release_spans,
        trade_sink=sink,
    )

    assert list(result.keys())[0] == "anchor"
    assert len(result["strategy"]) == 36
    assert len(result["spy"]) == 6
    assert len(result["coverage_by_cell"]) == 36
    assert len(sink) == 42
    assert result["payout_table"] == {"rows": 3, "pinned_confirmed": 1, "pinned_all": 2}
    # Every strategy cell carries evidence_set, loss, cost, tax and gap to SPY.
    sample = result["strategy"]["confirmed_loss1_cost18.3bps_taxlow"]
    assert sample["evidence_set"] == "confirmed"
    assert sample["loss"] == 1.0
    assert sample["one_way_cost"] == ft.DECOMPOSITION_COST
    assert sample["tax"] == ft.DECOMPOSITION_TAX
    assert "gap_cagr_after_tax_pp" in sample


def test_anchor_with_no_pins_equals_first_test_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    close, spy_close, eligible, span_rows, release_spans = _synthetic_window()
    first_test = _first_test_result(close, spy_close, eligible, span_rows, release_spans)
    rows = _synthetic_payout_rows()

    matched = payout_rerun.run_rerun(
        close,
        spy_close,
        eligible,
        span_rows,
        rows,
        first_test,
        start=SYNTH_START,
        end=SYNTH_END,
        release_spans=release_spans,
    )
    assert matched["anchor_ok"] is True
    assert "stopped" not in matched
    assert matched["anchor"]["strategy"]["loss1_cost18.3bps_taxlow"]["matches"] is True
    assert matched["anchor"]["strategy"]["loss0_cost18.3bps_taxlow"]["matches"] is True
    assert matched["anchor"]["spy"]["cost18.3bps_taxlow"]["matches"] is True

    # Perturb one saved value by one cent: anchor_ok must turn False and run_rerun must stop
    # immediately before running any evidence-set book or touching trade_sink.
    perturbed = copy.deepcopy(first_test)
    perturbed["strategy"]["loss1_cost18.3bps_taxlow"]["after_tax_final"] += 0.01

    real_run_book = payout_rerun.run_book
    real_run_spy = payout_rerun.run_spy
    real_exit_prices = payout_rerun.exit_prices
    book_calls = 0
    spy_calls = 0
    pinned_args: list[dict[str, float]] = []

    def counting_run_book(*args: Any, **kwargs: Any) -> Any:
        nonlocal book_calls
        book_calls += 1
        return real_run_book(*args, **kwargs)

    def counting_run_spy(*args: Any, **kwargs: Any) -> Any:
        nonlocal spy_calls
        spy_calls += 1
        return real_run_spy(*args, **kwargs)

    def counting_exit_prices(
        px_close: pd.DataFrame,
        px_spans: pd.DataFrame,
        reading: float,
        pinned: Any,
    ) -> pd.DataFrame:
        pinned_args.append(dict(pinned))
        return real_exit_prices(px_close, px_spans, reading, pinned)

    monkeypatch.setattr(payout_rerun, "run_book", counting_run_book)
    monkeypatch.setattr(payout_rerun, "run_spy", counting_run_spy)
    monkeypatch.setattr(payout_rerun, "exit_prices", counting_exit_prices)

    sink: dict[str, list] = {}
    mismatched = payout_rerun.run_rerun(
        close,
        spy_close,
        eligible,
        span_rows,
        rows,
        perturbed,
        start=SYNTH_START,
        end=SYNTH_END,
        release_spans=release_spans,
        trade_sink=sink,
    )
    assert list(mismatched.keys())[0] == "anchor"
    assert mismatched["anchor_ok"] is False
    assert mismatched["stopped"] == "anchor_mismatch"
    assert mismatched["anchor"]["strategy"]["loss1_cost18.3bps_taxlow"]["matches"] is False
    assert mismatched["anchor"]["strategy"]["loss0_cost18.3bps_taxlow"]["matches"] is True
    for forbidden_key in (
        "strategy",
        "spy",
        "identity_check",
        "coverage_by_cell",
        "lead_numbers",
        "gap_decomposition",
    ):
        assert forbidden_key not in mismatched
    assert set(mismatched.keys()) == {"anchor", "anchor_ok", "stopped", "first_test_range"}
    assert sink == {}
    # Only the anchor's own books ran: two strategy books (readings 1.0 and 0.0) and one SPY book,
    # with no pinned prices ever requested.
    assert book_calls == 2
    assert spy_calls == 1
    assert pinned_args == [{}, {}]


def test_identity_check_flags_a_nonzero_pinned_loss() -> None:
    close, spy_close, eligible, span_rows, release_spans = _synthetic_window()
    first_test = _first_test_result(close, spy_close, eligible, span_rows, release_spans)

    ok_result = payout_rerun.run_rerun(
        close,
        spy_close,
        eligible,
        span_rows,
        _synthetic_payout_rows(loss_101="0.0"),
        first_test,
        start=SYNTH_START,
        end=SYNTH_END,
        release_spans=release_spans,
    )
    assert ok_result["identity_check"]["identity_ok"] is True
    assert ok_result["identity_check"]["mismatches"] == []

    bad_result = payout_rerun.run_rerun(
        close,
        spy_close,
        eligible,
        span_rows,
        _synthetic_payout_rows(loss_101="0.3"),
        first_test,
        start=SYNTH_START,
        end=SYNTH_END,
        release_spans=release_spans,
    )
    assert bad_result["identity_check"]["identity_ok"] is False
    assert len(bad_result["identity_check"]["mismatches"]) == 12


def test_coverage_by_cell_counts_pinned_and_unpinned() -> None:
    close, spy_close, eligible, span_rows, release_spans = _synthetic_window()
    first_test = _first_test_result(close, spy_close, eligible, span_rows, release_spans)
    rows = _synthetic_payout_rows()

    result = payout_rerun.run_rerun(
        close,
        spy_close,
        eligible,
        span_rows,
        rows,
        first_test,
        start=SYNTH_START,
        end=SYNTH_END,
        release_spans=release_spans,
    )
    conf_ref = payout_rerun.cell_key("confirmed", 1.0, ft.DECOMPOSITION_COST, ft.DECOMPOSITION_TAX)
    all_ref = payout_rerun.cell_key("all", 1.0, ft.DECOMPOSITION_COST, ft.DECOMPOSITION_TAX)

    assert result["coverage_by_cell"][conf_ref] == {
        "held": 4,
        "pinned": 1,
        "unpinned": 3,
        "unpinned_names": ["PERMNO-102", "PERMNO-103", "PERMNO-104"],
        "missing_from_table": ["PERMNO-104"],
        "pinned_names": ["PERMNO-101"],
    }
    assert result["coverage_by_cell"][all_ref] == {
        "held": 4,
        "pinned": 2,
        "unpinned": 2,
        "unpinned_names": ["PERMNO-103", "PERMNO-104"],
        "missing_from_table": ["PERMNO-104"],
        "pinned_names": ["PERMNO-101", "PERMNO-102"],
    }
    assert result["pinned_share_cell"] == result["coverage_by_cell"][conf_ref]


def test_lead_numbers_keys_and_flags() -> None:
    # Part 1: standard fixture where pinning one stock narrows the range and middle is between.
    close, spy_close, eligible, span_rows, release_spans = _synthetic_window()
    first_test = _first_test_result(close, spy_close, eligible, span_rows, release_spans)
    result = payout_rerun.run_rerun(
        close,
        spy_close,
        eligible,
        span_rows,
        _synthetic_payout_rows(),
        first_test,
        start=SYNTH_START,
        end=SYNTH_END,
        release_spans=release_spans,
    )
    lead = result["lead_numbers"]
    assert set(lead.keys()) == {
        "worst",
        "middle",
        "best",
        "range_wider_than_first_test",
        "middle_between_worst_and_best",
    }
    assert lead["worst"] < lead["middle"] < lead["best"]
    assert lead["middle_between_worst_and_best"] is True
    assert lead["range_wider_than_first_test"] is False
    ft_range = result["first_test_range"]
    assert ft_range["worst"] == pytest.approx(
        first_test["strategy"]["loss1_cost18.3bps_taxlow"]["gap_cagr_after_tax_pp"]
    )
    assert ft_range["best"] == pytest.approx(
        first_test["strategy"]["loss0_cost18.3bps_taxlow"]["gap_cagr_after_tax_pp"]
    )

    # Part 2: tax-kink fixture where reading 0.5 finishes strictly above both reading 1.0 and 0.0.
    k_close, k_spy, k_elig, k_spans = _tax_kink_window()
    k_first = ft.run_study(
        k_close, k_spy, k_elig, k_spans, start=SYNTH_START, end=SYNTH_END, release_spans=None
    )
    k_result = payout_rerun.run_rerun(
        k_close,
        k_spy,
        k_elig,
        k_spans,
        rows=[],
        first_test=k_first,
        start=SYNTH_START,
        end=SYNTH_END,
        release_spans=None,
    )
    k_lead = k_result["lead_numbers"]
    assert k_lead["middle"] > k_lead["best"]
    assert k_lead["middle"] > k_lead["worst"]
    assert k_lead["middle_between_worst_and_best"] is False


def test_gap_decomposition_reconciles() -> None:
    close, spy_close, eligible, span_rows, release_spans = _synthetic_window()
    first_test = _first_test_result(close, spy_close, eligible, span_rows, release_spans)
    result = payout_rerun.run_rerun(
        close,
        spy_close,
        eligible,
        span_rows,
        _synthetic_payout_rows(loss_101="0.3"),
        first_test,
        start=SYNTH_START,
        end=SYNTH_END,
        release_spans=release_spans,
    )
    decomp = result["gap_decomposition"]
    assert decomp["cost"] == ft.DECOMPOSITION_COST
    assert decomp["tax"] == ft.DECOMPOSITION_TAX

    baseline_loss0_values: set[float] = set()
    for ev in payout_prices.EVIDENCE_SETS:
        by_loss = decomp["by_evidence_set"][ev]
        for reading in payout_prices.READINGS:
            part = by_loss[f"loss{reading:g}"]
            assert part["reconciles"] is True
            assert part["matches_cell_gap"] is True
            step_sum = (
                part["selection_effect_pp"]
                + part["dead_stock_effect_pp"]
                + part["trading_cost_effect_pp"]
                + part["tax_effect_pp"]
                + part["spy_cost_and_tax_drag_pp"]
            )
            assert step_sum == pytest.approx(part["total_gap_pp"], abs=1e-9)
            baseline_loss0_values.add(part["annual_growth_pct"]["rule_loss0_zero_cost_zero_tax"])
        # Because PERMNO-101 is pinned to 0.3, even at reading 0.0 dead_stock_effect_pp is non-zero
        # (proving rule_loss0_zero uses NO pins at reading 0.0, while rule_zero applies pins).
        assert by_loss["loss0"]["dead_stock_effect_pp"] < 0.0
    # Every cell shares the exact same unpinned loss-0 baseline.
    assert len(baseline_loss0_values) == 1


def test_run_rerun_writes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    close, spy_close, eligible, span_rows, release_spans = _synthetic_window()
    first_test = _first_test_result(close, spy_close, eligible, span_rows, release_spans)
    rows = _synthetic_payout_rows()

    def forbid(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("run_rerun must not touch the filesystem")

    monkeypatch.setattr(builtins, "open", forbid)
    monkeypatch.setattr(Path, "open", forbid)
    monkeypatch.setattr(Path, "write_text", forbid)
    monkeypatch.setattr(Path, "write_bytes", forbid)

    result = payout_rerun.run_rerun(
        close,
        spy_close,
        eligible,
        span_rows,
        rows,
        first_test,
        start=SYNTH_START,
        end=SYNTH_END,
        release_spans=release_spans,
    )
    assert result["anchor_ok"] is True
    assert len(result["strategy"]) == 36
