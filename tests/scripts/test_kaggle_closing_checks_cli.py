"""CLI contract for the read-only import checks."""

from datetime import date

import pytest

from scripts.kaggle_closing_checks import build_parser, default_output, initial_checks


def test_target_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_arguments_and_default_output():
    args = build_parser().parse_args(["--db-url-env", "TEST_URL", "--release-id", "3"])
    assert args.db_url_env == "TEST_URL" and args.release_id == 3
    assert (
        str(default_output(date(2026, 11, 23), "pilot"))
        == "results/kaggle_checks/2026-11-23-pilot.json"
    )


def test_registry():
    checks = initial_checks()
    assert [c["id"] for c in checks] == [f"check_{i:02}" for i in range(1, 14)]
    assert all(c["status"] == "covered_by_tests" for c in checks[8:])


def test_baseline_records_and_detects_changed_hash(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from scripts import kaggle_closing_checks as closing

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, *args):
            pass

        def fetchall(self):
            return [("AAA",), ("BBB",)]

    class Connection:
        def cursor(self):
            return Cursor()

    state = {"hash": "first"}
    monkeypatch.setattr(
        closing,
        "_load",
        lambda *a, **kw: SimpleNamespace(
            close=SimpleNamespace(shape=(2, 2)), provenance={"bars_sha256": state["hash"]}
        ),
    )
    path = tmp_path / "baseline.json"
    closing._baseline(Connection(), "unused", save=path)
    assert closing._baseline(Connection(), "unused", compare=path)["matches"] is True
    state["hash"] = "changed"
    assert closing._baseline(Connection(), "unused", compare=path)["matches"] is False


def test_return_statistics_and_thresholds():
    from scripts.kaggle_closing_checks import return_statistics

    good = return_statistics([0.01, -0.02, 0.03], [0.01, -0.02, 0.03])
    assert good["count"] == 3 and good["correlation"] == pytest.approx(1.0)
    assert good["mean_abs_points"] == 0 and good["max_abs_points"] == 0
    bad = return_statistics([0.01, -0.02, 0.03], [0.02, -0.01, 0.04])
    assert bad["mean_abs_points"] == pytest.approx(1.0)
    assert bad["max_abs_points"] == pytest.approx(1.0)
    assert return_statistics([0.01], [0.01])["correlation"] is None


def test_timezone_sample_and_chunk_filter_detection():
    from scripts.kaggle_closing_checks import select_timezone_symbols, symbol_filter_in_plan

    old = [f"PERMNO-{i}" for i in range(10)]
    all_names = old + [f"S{i}" for i in range(20)]
    picked = select_timezone_symbols(old, all_names)
    assert len(picked) == 20 and len(set(picked) & set(old)) >= 5
    assert symbol_filter_in_plan(
        {"Plan": {"Node Type": "Index Scan", "Index Cond": "symbol = ANY (...)"}}
    )
    assert not symbol_filter_in_plan(
        {"Plan": {"Node Type": "Seq Scan", "Filter": "interval = '1d'"}}
    )


def test_duplicate_stock_days_require_both_flags():
    import pandas as pd

    from scripts.kaggle_closing_checks import duplicate_stock_days

    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2000-01-03", tz="UTC"), "AAA"),
            (pd.Timestamp("2000-01-03", tz="UTC"), "PERMNO-1"),
        ],
        names=["timestamp", "symbol"],
    )
    flags = pd.DataFrame({"possible_double": [True, True]}, index=index)
    assert duplicate_stock_days(flags, {"AAA": 1, "PERMNO-1": 1}) == []
    flags.iloc[1, 0] = False
    bad = duplicate_stock_days(flags, {"AAA": 1, "PERMNO-1": 1})
    assert len(bad) == 1 and bad[0]["permno"] == 1


def test_exit_policy_only_allows_named_skips():
    from scripts.kaggle_closing_checks import checks_succeeded

    good = [
        {"status": "pass", "reason": ""},
        {"status": "covered_by_tests", "reason": ""},
        {"status": "skipped", "reason": "no baseline given"},
        {"status": "skipped", "reason": "name not in release"},
    ]
    assert checks_succeeded(good)
    assert not checks_succeeded([*good, {"status": "skipped", "reason": "not implemented"}])
    assert not checks_succeeded([*good, {"status": "fail", "reason": ""}])


def test_ten_stock_comparison_requires_full_month_coverage():
    from scripts.kaggle_closing_checks import ten_stock_passes

    result = {"count": 238, "correlation": 1.0, "mean_abs_points": 0.0, "max_abs_points": 0.0}
    assert not ten_stock_passes(result, minimum_months=239)
    assert ten_stock_passes({**result, "count": 239}, minimum_months=239)
    assert not ten_stock_passes({**result, "count": 239, "max_abs_points": 1.7}, minimum_months=239)
