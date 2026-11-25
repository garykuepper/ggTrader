"""Tests for fresh_compare: anchor comparison, gap table, headline and price-path pass-through."""

from __future__ import annotations

import copy
from typing import Any

import fp_dead_stock_first_test as study
import fresh_compare
import pytest

pytestmark = pytest.mark.study


def _make_cell(
    *,
    after_tax_final: float,
    cagr_after_tax: float,
    max_drawdown_pct: float,
    sharpe: float,
) -> dict[str, Any]:
    """Minimal cell dictionary using the field names fp_dead_stock_first_test._cell writes."""
    return {
        "after_tax_final": after_tax_final,
        "cagr_after_tax": cagr_after_tax,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
    }


def _make_full_result(
    *,
    strategy_overrides: dict[str, dict[str, Any]] | None = None,
    spy_overrides: dict[str, dict[str, Any]] | None = None,
    price_paths: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete 18-strategy + 6-SPY synthetic result matching run_study's layout."""
    spy: dict[str, dict[str, Any]] = {}
    for idx, cost in enumerate(study.COSTS):
        for t_idx, tax in enumerate(study.TAXES):
            key = study.spy_key(cost, tax)
            cell = {
                "one_way_cost": cost,
                "tax": tax,
                **_make_cell(
                    after_tax_final=180_000.0 - idx * 100.0 - t_idx * 500.0,
                    cagr_after_tax=0.10 - idx * 0.001 - t_idx * 0.005,
                    max_drawdown_pct=-30.0 - idx * 0.5,
                    sharpe=0.70 - idx * 0.01 - t_idx * 0.02,
                ),
            }
            if spy_overrides and key in spy_overrides:
                cell.update(spy_overrides[key])
            spy[key] = cell

    strategy: dict[str, dict[str, Any]] = {}
    for l_idx, loss in enumerate(study.LOSSES):
        for c_idx, cost in enumerate(study.COSTS):
            for t_idx, tax in enumerate(study.TAXES):
                key = study.cell_key(loss, cost, tax)
                cell = {
                    "loss": loss,
                    "one_way_cost": cost,
                    "tax": tax,
                    **_make_cell(
                        after_tax_final=150_000.0 + l_idx * 5_000.0 - c_idx * 200.0 - t_idx * 700.0,
                        cagr_after_tax=0.12 + l_idx * 0.01 - c_idx * 0.002 - t_idx * 0.006,
                        max_drawdown_pct=-40.0 + l_idx * 2.0 - c_idx * 0.5,
                        sharpe=0.80 + l_idx * 0.05 - c_idx * 0.02 - t_idx * 0.03,
                    ),
                }
                if strategy_overrides and key in strategy_overrides:
                    cell.update(strategy_overrides[key])
                strategy[key] = cell

    paths = (
        price_paths
        if price_paths is not None
        else {
            "descriptive_only": True,
            "note": "test price path block",
            "reference_cell": study.cell_key(*study.REFERENCE),
            "exits": [{"symbol": "DEAD1", "ratio": 0.95, "bucket": "0.9_to_1.1"}],
            "buckets": {"0.9_to_1.1": {"count": 1, "dollars": 2_000.0}},
        }
    )
    return {
        "strategy": strategy,
        "spy": spy,
        "death_price_paths": paths,
    }


def _wrap_saved(result: dict[str, Any]) -> dict[str, Any]:
    """Wrap a result dictionary the way _holdout-A.json stores it on disk."""
    return {
        "study": "dead-stock-first-test",
        "result": copy.deepcopy(result),
        "provenance": {"git_head": "deadbeef"},
    }


def test_identical_results_compare_clean() -> None:
    result = _make_full_result()
    saved = _wrap_saved(result)
    assert fresh_compare.compare_to_saved(result, saved) == []
    # Also clean when compared directly to an unwrapped result dict.
    assert fresh_compare.compare_to_saved(result, copy.deepcopy(result)) == []


def test_within_half_cent_compares_clean() -> None:
    saved_result = _make_full_result()
    result = copy.deepcopy(saved_result)
    target = study.cell_key(1.0, 0.00183, "low")
    result["strategy"][target]["after_tax_final"] += 0.0049
    spy_target = study.spy_key(0.00183, "low")
    result["spy"][spy_target]["after_tax_final"] -= 0.0049
    assert fresh_compare.compare_to_saved(result, _wrap_saved(saved_result)) == []


def test_one_cent_difference_is_reported() -> None:
    saved_result = _make_full_result()
    result = copy.deepcopy(saved_result)
    target = study.cell_key(0.5, 0.00183, "low")
    result["strategy"][target]["after_tax_final"] += 0.01

    diffs = fresh_compare.compare_to_saved(result, _wrap_saved(saved_result))
    assert len(diffs) == 1
    assert target in diffs[0]

    # A one-cent difference in a SPY cell is also reported.
    spy_result = copy.deepcopy(saved_result)
    spy_target = study.spy_key(0.0040, "high")
    spy_result["spy"][spy_target]["after_tax_final"] -= 0.01
    spy_diffs = fresh_compare.compare_to_saved(spy_result, _wrap_saved(saved_result))
    assert len(spy_diffs) == 1
    assert spy_target in spy_diffs[0]


def test_missing_cell_is_reported() -> None:
    saved_result = _make_full_result()
    result = copy.deepcopy(saved_result)
    missing_strat = study.cell_key(0.0, 0.00064, "high")
    del result["strategy"][missing_strat]

    diffs = fresh_compare.compare_to_saved(result, _wrap_saved(saved_result))
    assert len(diffs) == 1
    assert missing_strat in diffs[0]

    # Missing SPY cell and extra strategy cell are each reported with one message.
    result2 = copy.deepcopy(saved_result)
    missing_spy = study.spy_key(0.00064, "low")
    del result2["spy"][missing_spy]
    extra_key = "loss0.25_cost18.3bps_taxlow"
    result2["strategy"][extra_key] = _make_cell(
        after_tax_final=100_000.0,
        cagr_after_tax=0.0,
        max_drawdown_pct=0.0,
        sharpe=0.0,
    )
    diffs2 = fresh_compare.compare_to_saved(result2, _wrap_saved(saved_result))
    assert len(diffs2) == 2
    assert any(missing_spy in msg for msg in diffs2)
    assert any(extra_key in msg for msg in diffs2)


def test_headline_picks_the_registered_cells() -> None:
    spy_low_18 = study.spy_key(0.00183, "low")
    result = _make_full_result(
        spy_overrides={spy_low_18: {"cagr_after_tax": 0.10}},
        strategy_overrides={
            study.cell_key(1.0, 0.00183, "low"): {"cagr_after_tax": 0.075},
            study.cell_key(0.5, 0.00183, "low"): {"cagr_after_tax": 0.1025},
            study.cell_key(0.0, 0.00183, "low"): {"cagr_after_tax": 0.13},
        },
    )

    head = fresh_compare.headline(result)
    assert set(head) == {"worst", "middle", "best"}
    assert head["worst"] == pytest.approx(-2.5)
    assert head["middle"] == pytest.approx(0.25)
    assert head["best"] == pytest.approx(3.0)


def test_gap_table_has_18_rows_and_sign() -> None:
    spy_low_18 = study.spy_key(0.00183, "low")
    strat_above = study.cell_key(0.0, 0.00183, "low")
    strat_below = study.cell_key(1.0, 0.00183, "low")
    result = _make_full_result(
        spy_overrides={
            spy_low_18: {
                "cagr_after_tax": 0.10,
                "max_drawdown_pct": -30.0,
                "sharpe": 0.60,
            }
        },
        strategy_overrides={
            strat_above: {
                "cagr_after_tax": 0.14,
                "max_drawdown_pct": -25.0,
                "sharpe": 0.85,
            },
            strat_below: {
                "cagr_after_tax": 0.06,
                "max_drawdown_pct": -50.0,
                "sharpe": 0.35,
            },
        },
    )

    rows = fresh_compare.gap_table(result)
    assert len(rows) == 18
    expected_keys = {
        "cell",
        "loss",
        "cost_bps",
        "tax",
        "cagr_gap_pts",
        "drawdown_gap_pts",
        "sharpe_gap",
    }
    for row in rows:
        assert set(row) == expected_keys

    by_cell = {row["cell"]: row for row in rows}
    above = by_cell[strat_above]
    assert above["loss"] == 0.0
    assert above["cost_bps"] == pytest.approx(18.3)
    assert above["tax"] == "low"
    assert above["cagr_gap_pts"] == pytest.approx(4.0)
    assert above["drawdown_gap_pts"] == pytest.approx(5.0)
    assert above["sharpe_gap"] == pytest.approx(0.25)

    below = by_cell[strat_below]
    assert below["cagr_gap_pts"] == pytest.approx(-4.0)
    assert below["drawdown_gap_pts"] == pytest.approx(-20.0)
    assert below["sharpe_gap"] == pytest.approx(-0.25)


def test_price_path_block_is_passed_through_unchanged() -> None:
    sentinel_paths = {
        "descriptive_only": True,
        "note": "exact sentinel",
        "reference_cell": study.cell_key(*study.REFERENCE),
        "exits": [{"symbol": "XYZ", "ratio": 1.02, "bucket": "0.9_to_1.1"}],
        "buckets": {"0.9_to_1.1": {"count": 1, "dollars": 1234.56}},
    }
    result = _make_full_result(price_paths=sentinel_paths)

    out = fresh_compare.price_path_block(result)
    assert out is sentinel_paths
    assert out == sentinel_paths


def test_shared_omission_of_strategy_cell_is_reported() -> None:
    omitted = study.cell_key(1.0, 0.00064, "low")
    result = _make_full_result()
    del result["strategy"][omitted]

    diffs = fresh_compare.compare_to_saved(result, _wrap_saved(result))
    assert any("missing registered cell" in msg and omitted in msg for msg in diffs)
    assert any("missing from result" in msg and omitted in msg for msg in diffs)


def test_shared_omission_of_spy_cell_is_reported() -> None:
    omitted = study.spy_key(0.00064, "low")
    result = _make_full_result()
    del result["spy"][omitted]

    diffs = fresh_compare.compare_to_saved(result, _wrap_saved(result))
    assert any("missing registered cell" in msg and omitted in msg for msg in diffs)
    assert any("missing from result" in msg and omitted in msg for msg in diffs)


def test_matching_unregistered_keys_in_both_are_reported() -> None:
    extra_strat = "loss0.25_cost18.3bps_taxlow"
    extra_spy = "cost12bps_taxlow"
    result = _make_full_result()
    result["strategy"][extra_strat] = _make_cell(
        after_tax_final=100_000.0,
        cagr_after_tax=0.0,
        max_drawdown_pct=0.0,
        sharpe=0.0,
    )
    result["spy"][extra_spy] = _make_cell(
        after_tax_final=100_000.0,
        cagr_after_tax=0.0,
        max_drawdown_pct=0.0,
        sharpe=0.0,
    )

    diffs = fresh_compare.compare_to_saved(result, _wrap_saved(result))
    assert any("unregistered cell" in msg and extra_strat in msg for msg in diffs)
    assert any("unregistered cell" in msg and extra_spy in msg for msg in diffs)


def test_empty_section_nonfinite_and_missing_final_are_reported() -> None:
    # Empty section in both inputs cannot pass.
    empty = _make_full_result()
    empty["spy"] = {}
    empty_diffs = fresh_compare.compare_to_saved(empty, _wrap_saved(empty))
    assert any("has no cells" in msg for msg in empty_diffs)
    assert any("missing registered cell" in msg for msg in empty_diffs)

    # Non-finite and missing after_tax_final fields are each reported naming their cell.
    saved_result = _make_full_result()
    bad_result = copy.deepcopy(saved_result)
    nan_key = study.cell_key(1.0, 0.00183, "low")
    missing_field_key = study.spy_key(0.00183, "low")
    bad_result["strategy"][nan_key]["after_tax_final"] = float("nan")
    del bad_result["spy"][missing_field_key]["after_tax_final"]

    bad_diffs = fresh_compare.compare_to_saved(bad_result, _wrap_saved(saved_result))
    assert len(bad_diffs) == 2
    assert any(nan_key in msg and "non-finite" in msg for msg in bad_diffs)
    assert any(missing_field_key in msg and "after_tax_final" in msg for msg in bad_diffs)
