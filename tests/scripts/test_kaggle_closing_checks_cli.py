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


def test_duplicate_crsp_stock_days_fail_even_if_flagged():
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
    assert len(duplicate_stock_days(flags, {"AAA": 1, "PERMNO-1": 1})) == 1
    flags.iloc[1, 0] = False
    bad = duplicate_stock_days(flags, {"AAA": 1, "PERMNO-1": 1})
    assert len(bad) == 1 and bad[0]["permno"] == 1


def test_exit_policy_only_allows_named_skips():
    from scripts.kaggle_closing_checks import checks_succeeded

    good = [
        {"status": "pass", "reason": ""},
        {"status": "covered_by_tests", "reason": ""},
        {"status": "skipped", "reason": "name not in release"},
    ]
    assert checks_succeeded(good)
    assert not checks_succeeded([*good, {"status": "skipped", "reason": "no baseline given"}])
    assert checks_succeeded(
        [*good, {"status": "skipped", "reason": "no baseline given"}], allow_synthetic_skips=True
    )
    assert not checks_succeeded([*good, {"status": "skipped", "reason": "not implemented"}])
    assert not checks_succeeded([*good, {"status": "fail", "reason": ""}])


def test_baseline_origin_is_a_cli_option():
    args = build_parser().parse_args(["--db-url-env", "TEST_URL", "--baseline-origin", "main abc"])
    assert args.baseline_origin == "main abc"


def test_ten_stock_comparison_requires_full_month_coverage():
    from scripts.kaggle_closing_checks import ten_stock_passes

    result = {"count": 238, "correlation": 1.0, "mean_abs_points": 0.0, "max_abs_points": 0.0}
    assert not ten_stock_passes(result, minimum_months=239)
    assert ten_stock_passes({**result, "count": 239}, minimum_months=239)
    assert not ten_stock_passes({**result, "count": 239, "max_abs_points": 1.7}, minimum_months=239)


def test_named_frame_structural_validation():
    from types import SimpleNamespace

    import pandas as pd

    from scripts.kaggle_closing_checks import validate_named_frame

    days = pd.to_datetime(["1980-12-15", "1999-12-31", "2000-01-03"], utc=True)
    index = pd.MultiIndex.from_arrays([days, ["AAPL"] * 3], names=["timestamp", "symbol"])
    flags = pd.DataFrame(
        {"source": ["crsp", "crsp", "yfinance"], "possible_double": [False] * 3}, index=index
    )
    close = pd.DataFrame({"AAPL": [10.0, 10.0, 11.0]}, index=days)
    bars = SimpleNamespace(flags=flags, close=close)
    span = {
        "symbol": "AAPL",
        "kind": "linked_yahoo",
        "first_date": days[0].date(),
        "last_date": days[-1].date(),
        "first_yf_date": days[-1].date(),
        "crsp_until": days[-1].date(),
        "possible_double": False,
    }
    result = validate_named_frame(bars, span)
    assert result["status"] == "pass" and result["join_return"] == pytest.approx(0.1)
    flags.iloc[0, 0] = "yfinance"
    assert validate_named_frame(bars, span)["status"] == "fail"


def test_synthetic_skip_requires_explicit_flag():
    from scripts.kaggle_closing_checks import checks_succeeded

    checks = [{"status": "skipped", "reason": "synthetic data"}]
    assert not checks_succeeded(checks)
    assert checks_succeeded(checks, allow_synthetic_skips=True)
    args = build_parser().parse_args(["--db-url-env", "TEST_URL", "--allow-synthetic-skips"])
    assert args.allow_synthetic_skips


def test_yahoo_span_checks_its_recorded_first_yahoo_date():
    from types import SimpleNamespace

    import pandas as pd

    from scripts.kaggle_closing_checks import validate_named_frame

    day = pd.Timestamp("2020-01-03", tz="UTC")
    index = pd.MultiIndex.from_tuples([(day, "AAA")], names=["timestamp", "symbol"])
    bars = SimpleNamespace(
        flags=pd.DataFrame({"source": ["yfinance"], "possible_double": [False]}, index=index),
        close=pd.DataFrame({"AAA": [10.0]}, index=[day]),
    )
    span = {
        "symbol": "AAA",
        "kind": "yahoo",
        "first_date": day.date(),
        "last_date": day.date(),
        "first_yf_date": pd.Timestamp("2020-01-06").date(),
        "crsp_until": None,
        "possible_double": False,
    }
    assert validate_named_frame(bars, span)["status"] == "fail"


def test_nested_plan_evidence_requires_both_conditions(tmp_path):
    import json

    from scripts.kaggle_closing_checks import check8_status, checks_succeeded, nested_plan_evidence

    assert nested_plan_evidence(None)["status"] == "collected_by_orchestrator"
    assert check8_status(True, True, "collected_by_orchestrator", False) == "partial"
    assert not checks_succeeded([{"status": "partial"}])
    assert check8_status(True, True, "collected_by_orchestrator", True) == "skipped"
    path = tmp_path / "plans.json"
    path.write_text(
        json.dumps(
            {
                "yahoo": "Filter: symbol = ANY; timestamp >= 2000",
                "crsp": "Index Cond: symbol = ANY; date >= 2000",
            }
        )
    )
    assert nested_plan_evidence(path)["status"] == "pass"
    assert check8_status(True, True, "pass", False) == "pass"
    path.write_text(json.dumps({"yahoo": "Filter: symbol = ANY", "crsp": "date >= 2000"}))
    assert nested_plan_evidence(path)["status"] == "fail"
    assert check8_status(True, True, "fail", False) == "fail"
    assert (
        build_parser().parse_args(["--db-url-env", "DB", "--nested-plans", str(path)]).nested_plans
        == path
    )


def test_identity_groups_are_packed_without_splitting_or_exceeding_limit():
    from scripts.kaggle_closing_checks import pack_identity_groups

    groups = [(n, [f"{n}-{i}" for i in range(size)]) for n, size in enumerate([50, 50, 30, 120, 2])]
    batches = list(pack_identity_groups(groups, limit=120))
    assert [sum(len(names) for _, names in batch) for batch in batches] == [100, 30, 120, 2]
    assert [item for batch in batches for item in batch] == groups
    with pytest.raises(ValueError, match="120"):
        list(pack_identity_groups([(1, [str(i) for i in range(121)])], limit=120))


def test_main_redacts_connection_details_from_failure(monkeypatch, capsys):
    import sys

    from scripts import kaggle_closing_checks as closing

    monkeypatch.setenv("GGT_FAKE_URL", "postgresql://alice:secret@db.invalid/scratch")
    monkeypatch.setattr(sys, "argv", ["checks", "--db-url-env", "GGT_FAKE_URL"])

    def fail(*args, **kwargs):
        raise ValueError("bad postgresql://alice:secret@db.invalid/scratch password=hidden")

    monkeypatch.setattr(closing, "run_checks", fail)
    with pytest.raises(SystemExit) as error:
        closing.main()
    message = str(error.value)
    assert "ValueError" in message and "bad" in message
    assert all(word not in message for word in ("alice", "secret", "hidden", "db.invalid"))


def test_identity_audit_distinguishes_crsp_overlaps_and_yahoo_candidates():
    import pandas as pd

    from scripts.kaggle_closing_checks import classify_identity_records, duplicate_stock_days

    d1, d2 = date(2000, 1, 1), date(2000, 12, 31)
    records = [
        ("WRB", 1, "linked_yahoo", d1, d2, False),
        ("ACGL", 1, "yahoo", d1, d2, False),
        ("CMVT", 2, "yahoo", d1, d2, True),
        ("PERMNO-2", 2, "placeholder", d1, d2, True),
        ("BAD", 3, "yahoo", d1, d2, True),
        ("PERMNO-3", 3, "placeholder", d1, d2, False),
        ("PERMNO-4", 4, "placeholder", d1, d2, True),
        ("PERMNO-4~1", 4, "spell", d1, d2, True),
    ]
    groups, issues, different, checked = classify_identity_records(records)
    assert checked == 2
    assert different == [{"yahoo_symbol": "ACGL", "linked_symbols": ["WRB"], "permno": 1}]
    assert issues == [{"yahoo_symbol": "BAD", "crsp_symbol": "PERMNO-3", "permno": 3}]
    assert groups == [(4, ["PERMNO-4", "PERMNO-4~1"])]
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2000-01-03", tz="UTC"), name) for name in groups[0][1]],
        names=["timestamp", "symbol"],
    )
    flags = pd.DataFrame({"possible_double": [True, True]}, index=index)
    assert duplicate_stock_days(flags, dict.fromkeys(groups[0][1], 4)) == [
        {"date": "2000-01-03", "permno": 4, "symbols": ["PERMNO-4", "PERMNO-4~1"]}
    ]


def test_identity_audit_packs_440_groups_into_eight_reads(monkeypatch):
    from types import SimpleNamespace

    import pandas as pd

    from scripts import kaggle_closing_checks as closing

    calls = []
    day = pd.Timestamp("2000-01-03", tz="UTC")
    groups = [(n, [f"A{n}", f"B{n}"]) for n in range(440)]

    def fake_load(url, names, source, release_id):
        calls.append(names)
        index = pd.MultiIndex.from_tuples(
            [(day, name) for name in names], names=["timestamp", "symbol"]
        )
        return SimpleNamespace(flags=pd.DataFrame({"possible_double": False}, index=index))

    monkeypatch.setattr(closing, "_load", fake_load)
    overlaps = closing.read_identity_overlaps("scratch", 1, groups)
    assert len(calls) == 8 and max(map(len, calls)) == 120
    assert len(overlaps) == 440
    assert overlaps[0] == {"date": "2000-01-03", "permno": 0, "symbols": ["A0", "B0"]}


@pytest.mark.parametrize(
    "problem", [None, "changed", "late_crsp", "missing_yahoo", "missing_source"]
)
def test_yahoo_comparison_handles_crsp_additions_and_rejects_bad_rows(monkeypatch, problem):
    from types import SimpleNamespace

    import pandas as pd

    from scripts import kaggle_closing_checks as closing

    dates = pd.to_datetime(
        ["1999-01-04", "2005-01-03", "2009-12-31", "2010-01-04", "2010-01-05"], utc=True
    )
    close = pd.DataFrame({"SHOP": [3.0, 5.0, 9.0, 10.0, 11.0]}, index=dates)
    sources = ["crsp", "crsp", "crsp", "yfinance", "yfinance"]
    if problem == "changed":
        close.iloc[3, 0] = 20.0
    if problem == "late_crsp":
        sources[3] = "crsp"
    if problem == "missing_source":
        sources[2] = pd.NA
    if problem == "missing_yahoo":
        close = close.iloc[:-1]
        sources = sources[:-1]
    index = pd.MultiIndex.from_arrays(
        [close.index, ["SHOP"] * len(close)], names=["timestamp", "symbol"]
    )
    combined = SimpleNamespace(
        flags=pd.DataFrame({"source": sources}, index=index), frame=lambda field: close
    )
    db_close = pd.DataFrame({"SHOP": [10.0, 11.0]}, index=dates[3:])
    db_bars = SimpleNamespace(frame=lambda field: db_close)
    monkeypatch.setattr(
        closing,
        "_load",
        lambda url, names, source, *args, **kwargs: combined if source == "combined" else db_bars,
    )

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, sql, args):
            self.sql = sql

        def fetchall(self):
            if "public.ohlcv" in self.sql:
                return [(day.tz_localize(None), "SHOP") for day in dates[3:]]
            return [("SHOP", date(2010, 1, 4))]

    conn = SimpleNamespace(cursor=Cursor)
    status, measured, _ = closing._check_yahoo_keys(conn, "scratch", 1, ["SHOP"])
    assert status == ("pass" if problem is None else "fail")
    assert measured["crsp_rows_before_2000"] == 1
    assert measured["crsp_rows_from_2000"] == 2 + (problem == "late_crsp") - (
        problem == "missing_source"
    )
    if problem == "changed":
        assert measured["different_fields"]
    if problem == "missing_yahoo":
        assert measured["missing_rows"] == 1
    if problem == "late_crsp":
        assert measured["invalid_crsp_rows"] == 1
    if problem == "missing_source":
        assert measured["invalid_source_rows"] == 1


def test_nested_plans_accept_quoted_timestamps_and_compressed_time_bounds(tmp_path):
    import json

    from scripts.kaggle_closing_checks import nested_plan_evidence

    yahoo_symbol = (
        "Index Cond: ((symbol = q.symbol) AND (symbol = ANY ('{SMCI,VC,AAPL,TTWO}'::text[])) "
        "AND (\"interval\" = '1d'::text))"
    )
    quoted = (
        "Vectorized Filter: ((\"timestamp\" < '2026-11-06 00:00:00'::timestamp without time zone) "
        "AND (\"timestamp\" >= '2015-01-01 00:00:00'::timestamp without time zone))"
    )
    compressed = (
        "Index Cond: ((load_id = l.load_id) AND (permno = s.permno) "
        "AND (_ts_meta_min_1 <= '1991-12-31'::date) "
        "AND (_ts_meta_max_1 >= '1990-01-01'::date))"
    )
    crsp_symbol = "Index Cond: ((release_id = '2'::bigint) AND (symbol = q.symbol))"
    path = tmp_path / "plans.json"
    for yahoo in (quoted, compressed):
        path.write_text(
            json.dumps(
                {"yahoo": yahoo_symbol + "\n" + yahoo, "crsp": crsp_symbol + "\n" + compressed}
            )
        )
        assert nested_plan_evidence(path)["status"] == "pass"
    path.write_text(json.dumps({"yahoo": yahoo_symbol, "crsp": crsp_symbol + "\n" + compressed}))
    assert nested_plan_evidence(path)["status"] == "fail"


def test_read_timing_limits_allow_fixed_single_symbol_cost():
    from scripts.kaggle_closing_checks import read_timings_pass

    times = {
        "combined_one_seconds": 9.99,
        "db_one_seconds": 9.99,
        "combined_hundred_seconds": 59.99,
        "db_hundred_seconds": 59.99,
    }
    assert read_timings_pass(times)
    for key in ("combined_one_seconds", "db_one_seconds"):
        assert not read_timings_pass({**times, key: 10.0})
    for key in ("combined_hundred_seconds", "db_hundred_seconds"):
        assert not read_timings_pass({**times, key: 60.0})
