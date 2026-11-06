"""Tests for classifying saved WFO fold rows by gate/anchor outcome."""

from ggTrader.lab.wfo_attribution import (
    classify_fold,
    regime_from_ancestry,
    summarize_run,
)


def _row(num: int, gates: bool, anchor: bool, wfe=0.5) -> dict:
    return {"fold_num": num, "gates_passed": gates, "used_anchor": anchor, "wfe": wfe}


def test_classify_fold_gate_fail():
    assert classify_fold(_row(1, False, True)) == "gate_fail"


def test_classify_fold_gates_pass_anchor():
    assert classify_fold(_row(1, True, True)) == "gates_pass_anchor"


def test_classify_fold_gates_pass_winner():
    assert classify_fold(_row(1, True, False)) == "gates_pass_winner"


def test_classify_fold_flags_anomaly_when_neither_gates_nor_anchor():
    assert classify_fold(_row(1, False, False)) == "anomaly"


def test_classify_fold_missing_keys_is_anomaly_not_false():
    assert classify_fold({"fold_num": 1}) == "anomaly"


def test_summarize_run_counts_sum_to_n_folds():
    rows = [_row(1, False, True), _row(2, True, True), _row(3, True, False), _row(4, True, False)]
    out = summarize_run(rows)
    assert out["n_folds"] == 4
    assert out["gate_fail"] == 1
    assert out["gates_pass_anchor"] == 1
    assert out["gates_pass_winner"] == 2
    assert out["anomaly"] == 0
    assert (
        out["gate_fail"] + out["gates_pass_anchor"] + out["gates_pass_winner"] + out["anomaly"]
        == out["n_folds"]
    )


def test_summarize_run_counts_nan_and_none_wfe_separately():
    rows = [
        _row(1, True, False, wfe=float("nan")),
        _row(2, True, False, wfe=None),
        _row(3, True, False, wfe=0.4),
    ]
    out = summarize_run(rows)
    assert out["nan_wfe"] == 1
    assert out["none_wfe"] == 1


def test_summarize_run_reports_missing_fold_numbers():
    rows = [_row(1, True, False), _row(2, True, False), _row(4, True, False)]
    assert summarize_run(rows)["fold_numbers_missing"] == [3]


def test_classify_fold_rejects_non_boolean_flags_as_anomalies():
    assert classify_fold({"fold_num": 1, "gates_passed": "false", "used_anchor": True}) == "anomaly"
    assert classify_fold({"fold_num": 1, "gates_passed": True, "used_anchor": float("nan")}) == (
        "anomaly"
    )
    assert classify_fold({"fold_num": 1, "gates_passed": 1, "used_anchor": 0}) == "anomaly"


def test_summarize_run_counts_absent_wfe_separately_from_explicit_none():
    rows = [
        {"fold_num": 1, "gates_passed": True, "used_anchor": False},  # key absent
        {"fold_num": 2, "gates_passed": True, "used_anchor": False, "wfe": None},
    ]
    out = summarize_run(rows)
    assert out["wfe_absent"] == 1
    assert out["none_wfe"] == 1


def test_regime_from_ancestry_pre_fix_when_last_commit_predates_fix():
    assert regime_from_ancestry(first_after_fix=False, last_after_fix=False) == "pre_dsr_fix"


def test_regime_from_ancestry_after_fix_commit_is_not_proof_of_gate_version():
    assert regime_from_ancestry(first_after_fix=True, last_after_fix=True) == "committed_after_fix"


def test_regime_from_ancestry_unknown_on_straddle_or_git_error():
    assert regime_from_ancestry(first_after_fix=False, last_after_fix=True) == "unknown"
    assert regime_from_ancestry(first_after_fix=None, last_after_fix=True) == "unknown"
    assert regime_from_ancestry(first_after_fix=False, last_after_fix=None) == "unknown"
