"""Tests for the dead-stock first test driver: the grid, the windows and the holdout guard."""

from __future__ import annotations

import gzip
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import dead_stock_first_test as study
import numpy as np
import pandas as pd
import pytest
from ledger_run import BookResult
from replay import Trade

from ggTrader.utils.study import load_driver as load_driver_by_path
from ggTrader.utils.study import repo_root, study_dir

pytestmark = pytest.mark.study

CODE = study_dir(__file__) / "code"
DAYS = pd.bdate_range("2008-01-01", "2011-12-30")
START = "2010-01-04"
END = "2011-12-30"
LOSSES = (1.0, 0.5, 0.0)
COSTS = (0.00064, 0.00183, 0.0040)
TAXES = ("low", "high")


def load_driver():
    """The driver module, loaded by path so its file name cannot clash with another study's."""
    return load_driver_by_path(CODE / "run.py")


def make_close(n: int = 12, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0004, 0.015, size=(len(DAYS), n))
    return pd.DataFrame(
        40.0 * np.exp(np.cumsum(steps, axis=0)),
        index=DAYS,
        columns=[f"S{i:02d}" for i in range(n)],
    )


def make_spy() -> pd.Series:
    rng = np.random.default_rng(99)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(DAYS)))), index=DAYS)


def no_death_inputs() -> dict:
    close = make_close()
    return {
        "close": close,
        "spy_close": make_spy(),
        "eligible": pd.DataFrame(True, index=close.index, columns=close.columns),
        "span_rows": pd.DataFrame(
            {
                "symbol": pd.Series([], dtype=object),
                "last_date": pd.to_datetime([]),
                "has_successor": pd.Series([], dtype=bool),
                "total_loss": pd.Series([], dtype=bool),
            }
        ),
    }


def death_inputs() -> dict:
    """A runaway stock D that is held for months, then stops trading on 2011-03-31."""
    data = no_death_inputs()
    close = data["close"]
    last = pd.Timestamp("2011-03-31")
    close["D"] = 10.0 * np.exp(0.004 * np.arange(len(close)))
    close.loc[close.index > last, "D"] = np.nan
    data["eligible"] = pd.DataFrame(True, index=close.index, columns=close.columns)
    data["span_rows"] = pd.DataFrame(
        {
            "symbol": ["D"],
            "last_date": [last],
            "has_successor": [False],
            "total_loss": [False],
        }
    )
    return data


def run(data: dict, **kwargs) -> dict:
    return study.run_study(
        data["close"],
        data["spy_close"],
        data["eligible"],
        data["span_rows"],
        start=START,
        end=END,
        **kwargs,
    )


@pytest.fixture(scope="module")
def clean() -> dict:
    return run(no_death_inputs())


@pytest.fixture(scope="module")
def dead() -> dict:
    return run(death_inputs())


def test_window_dates_pinned() -> None:
    assert study.WINDOWS["dev"] == ("2005-01-03", "2019-12-31")
    assert study.WINDOWS["holdout"] == ("2020-01-02", "2024-12-31")
    assert study.CRSP_END == "2024-12-31"


def test_output_has_every_grid_cell(clean: dict) -> None:
    cells = {(c["loss"], c["one_way_cost"], c["tax"]) for c in clean["strategy"].values()}
    assert cells == {(x, c, t) for x in LOSSES for c in COSTS for t in TAXES}
    assert len(clean["strategy"]) == 18
    spy = {(c["one_way_cost"], c["tax"]) for c in clean["spy"].values()}
    assert spy == {(c, t) for c in COSTS for t in TAXES}
    assert len(clean["spy"]) == 6
    needed = {
        "after_tax_final",
        "cagr_pretax",
        "cagr_after_tax",
        "tax_paid",
        "cost_paid",
        "unpaid_tax_left",
        "n_dead_exits",
        "dead_exit_writeoff",
        "skipped_buys",
        "sharpe",
        "max_drawdown_pct",
        "turnover_by_year",
    }
    for cell in [*clean["strategy"].values(), *clean["spy"].values()]:
        assert needed <= set(cell)
        assert np.isfinite(cell["after_tax_final"])


def test_loss_changes_only_dead_exit_prices(clean: dict, dead: dict) -> None:
    for cost in COSTS:
        for tax in TAXES:
            finals = {
                clean["strategy"][study.cell_key(loss, cost, tax)]["after_tax_final"]
                for loss in LOSSES
            }
            assert len(finals) == 1  # no deaths: the loss level never touches a price
    for cell in clean["strategy"].values():
        assert cell["n_dead_exits"] == 0 and cell["dead_exit_writeoff"] == 0.0
    mid = study.cell_key(1.0, 0.00183, "low")
    best = study.cell_key(0.0, 0.00183, "low")
    assert dead["strategy"][mid]["after_tax_final"] < dead["strategy"][best]["after_tax_final"]
    assert dead["strategy"][mid]["n_dead_exits"] == 1
    assert dead["strategy"][best]["n_dead_exits"] == 1


def test_sanity_block_flags_no_dead_stock_held(clean: dict, dead: dict) -> None:
    assert clean["sanity"]["no_dead_stock_held"] is True
    assert clean["sanity"]["held_at_death"] == 0
    assert dead["sanity"]["no_dead_stock_held"] is False
    assert dead["sanity"]["held_at_death"] == 1


def test_sanity_block_checks(clean: dict, dead: dict) -> None:
    for result in (clean, dead):
        sanity = result["sanity"]
        assert sanity["spy_zero_cost_growth"]["within_tolerance"] is True
        assert sanity["loss_monotone"]["violations"] == []
        assert sanity["one_permno_one_column"]["checked"] is False  # no release spans given


def test_permno_double_hold_is_reported_not_hidden() -> None:
    data = no_death_inputs()
    close = data["close"]
    grow = np.arange(len(close))
    close["S00"] = 10.0 * np.exp(0.004 * grow)  # two runaway columns of one company
    close["S01"] = 10.0 * np.exp(0.0039 * grow)
    release_spans = pd.DataFrame({"symbol": list(close.columns), "permno": list(range(12))})
    release_spans.loc[1, "permno"] = release_spans.loc[0, "permno"]
    check = run(data, release_spans=release_spans)["sanity"]["one_permno_one_column"]
    assert check["checked"] is True
    assert check["violations"]
    assert set(check["violations"][0]["symbols"]) == {"S00", "S01"}
    separate = release_spans.assign(permno=range(12))
    done = run(data, release_spans=separate)["sanity"]["one_permno_one_column"]
    assert done["checked"] is True and done["violations"] == []


def test_gap_is_strategy_minus_spy_in_points(clean: dict) -> None:
    for cell in clean["strategy"].values():
        spy = clean["spy"][study.spy_key(cell["one_way_cost"], cell["tax"])]
        expected = (cell["cagr_after_tax"] - spy["cagr_after_tax"]) * 100.0
        assert cell["gap_cagr_after_tax_pp"] == pytest.approx(expected)


def test_planner_called_once_per_run(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = study.plan_trades

    def counting(*args, **kwargs):
        calls.append(kwargs.get("dead_loss"))
        return real(*args, **kwargs)

    monkeypatch.setattr(study, "plan_trades", counting)
    run(no_death_inputs())
    assert len(calls) == 1


def test_coverage_block_counts_the_death(dead: dict) -> None:
    year = dead["coverage"]["2011"]
    assert year["ever_eligible_dead"] == 1
    assert year["held_at_death"] == 1
    assert year["dollars_written_off"] > 0.0
    assert dead["coverage"]["2010"]["held_at_death"] == 0
    assert dead["coverage"]["2010"]["eligible_per_month_mean"] == pytest.approx(13.0)


def test_turnover_excludes_the_final_liquidation(clean: dict) -> None:
    cell = clean["spy"][study.spy_key(0.00183, "low")]
    assert cell["turnover_by_year"]["2010"] == pytest.approx(1.0, rel=0.05)  # the one purchase
    assert cell["turnover_by_year"]["2011"] == pytest.approx(0.0, abs=1e-9)


def runaway_inputs(last_day: str) -> dict:
    """Like ``death_inputs`` but the runaway stock D stops on ``last_day`` (and has a span row)."""
    data = death_inputs()
    close = data["close"]
    last = pd.Timestamp(last_day)
    close["D"] = 10.0 * np.exp(0.004 * np.arange(len(close)))
    close.loc[close.index > last, "D"] = np.nan
    data["span_rows"] = pd.DataFrame(
        {"symbol": ["D"], "last_date": [last], "has_successor": [False], "total_loss": [False]}
    )
    return data


def test_stock_alive_at_the_file_end_is_not_a_dead_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(study, "CRSP_END", "2011-03-31")
    result = run(runaway_inputs("2011-03-31"))
    assert result["n_file_end_censored"] == 1
    assert result["sanity"]["held_at_death"] == 0
    assert result["sanity"]["no_dead_stock_held"] is True
    assert result["coverage"]["2011"]["ever_eligible_dead"] == 0
    assert result["coverage"]["2011"]["held_at_death"] == 0
    assert result["death_price_paths"]["exits"] == []
    for cost in COSTS:
        for tax in TAXES:
            cells = [result["strategy"][study.cell_key(x, cost, tax)] for x in LOSSES]
            assert all(c["n_dead_exits"] == 0 and c["dead_exit_writeoff"] == 0.0 for c in cells)
            # sold at the raw last close on the window's last day: the loss level never matters
            assert len({c["after_tax_final"] for c in cells}) == 1


def test_stock_that_died_a_day_before_the_file_end_is_a_dead_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(study, "CRSP_END", "2011-03-31")
    result = run(runaway_inputs("2011-03-30"))
    assert result["n_file_end_censored"] == 0
    assert result["sanity"]["held_at_death"] == 1
    assert result["coverage"]["2011"]["ever_eligible_dead"] == 1
    mid = study.cell_key(1.0, 0.00183, "low")
    best = study.cell_key(0.0, 0.00183, "low")
    assert result["strategy"][mid]["n_dead_exits"] == 1
    assert result["strategy"][mid]["after_tax_final"] < result["strategy"][best]["after_tax_final"]


def test_development_window_has_no_censored_rows(dead: dict) -> None:
    assert dead["n_file_end_censored"] == 0
    assert dead["crsp_end"] == "2024-12-31"
    assert study.WINDOWS["dev"][1] < study.CRSP_END


def test_censored_rows_are_dropped_and_counted() -> None:
    spans = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "last_date": pd.to_datetime(["2024-12-31", "2024-12-30", "2023-05-01"]),
            "has_successor": [False] * 3,
            "total_loss": [False] * 3,
        }
    )
    kept, count = study.drop_file_end_rows(spans, "2024-12-31")
    assert list(kept["symbol"]) == ["B", "C"] and count == 1
    kept, count = study.drop_file_end_rows(spans, "2019-12-31")
    assert list(kept["symbol"]) == ["B", "C"] and count == 0  # outside a window: not counted


@pytest.mark.parametrize(
    ("ratio", "bucket"),
    [
        (0.4999, "lt_0.5"),
        (0.5, "0.5_to_0.9"),
        (0.8999, "0.5_to_0.9"),
        (0.9, "0.9_to_1.1"),
        (1.0999, "0.9_to_1.1"),
        (1.1, "ge_1.1"),
        (3.0, "ge_1.1"),
        (None, "no_history"),
        (float("nan"), "no_history"),
    ],
)
def test_ratio_bucket_edges(ratio: float | None, bucket: str) -> None:
    assert study.ratio_bucket(ratio) == bucket
    assert 10.0 * 0.5 / 10.0 == 0.5 and 9.0 / 10.0 == 0.9 and 11.0 / 10.0 == 1.1


def test_death_price_paths_buckets_and_no_history() -> None:
    days = pd.bdate_range("2010-01-04", periods=40)
    exit_day = days[30]
    earlier = days[30 - 21]
    close = pd.DataFrame(100.0, index=days, columns=["LOW", "MID", "EQ", "UP", "SHORT"])
    for name, ratio in {"LOW": 0.2, "MID": 0.5, "EQ": 1.0, "UP": 1.1}.items():
        close.loc[earlier, name] = 100.0 / ratio  # last / earlier == ratio
    close.loc[days[:5], "SHORT"] = np.nan
    trades = [
        Trade(exit_day, name, -10.0, 1e-9, "dead_exit") for name in ("LOW", "MID", "EQ", "UP")
    ]
    trades.append(
        Trade(days[10], "SHORT", -10.0, 1e-9, "dead_exit")
    )  # fewer than 21 days of history
    trades.append(Trade(exit_day, "EQ", -1.0, 1.0, "rebalance"))  # not a dead exit: ignored
    block = study.death_price_paths(trades, close, frozenset())
    assert block["descriptive_only"] is True and block["note"]
    counts = {name: b["count"] for name, b in block["buckets"].items()}
    assert counts == {"lt_0.5": 1, "0.5_to_0.9": 1, "0.9_to_1.1": 1, "ge_1.1": 1, "no_history": 1}
    assert block["buckets"]["lt_0.5"]["dollars"] == pytest.approx(10.0 * 100.0)
    row = next(e for e in block["exits"] if e["symbol"] == "SHORT")
    assert row["ratio"] is None and row["bucket"] == "no_history"
    low = next(e for e in block["exits"] if e["symbol"] == "LOW")
    assert low["ratio"] == pytest.approx(0.2)
    assert low["last_close"] == 100.0 and low["exit_date"] == str(exit_day.date())


def test_death_price_paths_skips_successors() -> None:
    days = pd.bdate_range("2010-01-04", periods=40)
    close = pd.DataFrame(100.0, index=days, columns=["BUY"])
    trades = [Trade(days[30], "BUY", -10.0, 100.0, "dead_exit")]
    block = study.death_price_paths(trades, close, frozenset({"BUY"}))
    assert block["exits"] == []


def test_death_block_reports_the_held_death(dead: dict) -> None:
    block = dead["death_price_paths"]
    assert block["descriptive_only"] is True
    assert [e["symbol"] for e in block["exits"]] == ["D"]
    assert block["exits"][0]["ratio"] == pytest.approx(np.exp(0.004 * 21))  # runaway path
    assert block["buckets"]["0.9_to_1.1"]["count"] == 1  # exp(0.084) = 1.088
    assert block["buckets"]["0.9_to_1.1"]["dollars"] == block["exits"][0]["dollars_at_last_close"]


def test_death_block_never_changes_a_cell(dead: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(study, "death_price_paths", lambda *a, **k: {})
    without = run(death_inputs())
    assert without["strategy"] == dead["strategy"]
    assert without["spy"] == dead["spy"]
    assert without["coverage"] == dead["coverage"]
    assert without["sanity"] == dead["sanity"]


def fake_inputs(*_args) -> dict:
    data = no_death_inputs()
    data["release_spans"] = None
    data["data_provenance"] = {"data_release": {"release_id": 1}, "bars_sha256": ["abc"]}
    return data


def stub_study(monkeypatch: pytest.MonkeyPatch, driver) -> None:
    monkeypatch.setattr(driver.study, "run_study", lambda *a, **k: {"strategy": {}, "spy": {}})


def holdout_argv(universe: str = "A") -> list[str]:
    return ["--window", "holdout", "--universe", universe, "--release-id", "1"]


def test_holdout_runs_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    driver = load_driver()
    stub_study(monkeypatch, driver)
    assert driver.main(holdout_argv(), loader=fake_inputs, results_dir=tmp_path) == 0
    target = tmp_path / "holdout-A.json"
    payload = json.loads(target.read_text())
    assert payload["provenance"]["window"] == {"start": "2020-01-02", "end": "2024-12-31"}
    first = target.read_text()
    with pytest.raises(SystemExit) as refused:
        driver.main(holdout_argv(), loader=fake_inputs, results_dir=tmp_path)
    assert "holdout-A.json" in str(refused.value) or "holdout-A.claim" in str(refused.value)
    assert "only the owner may delete the claim" in str(refused.value)
    assert target.read_text() == first  # untouched


def test_second_holdout_refuses_even_without_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = load_driver()

    def crash(*_a, **_k):
        raise RuntimeError("database went away")

    monkeypatch.setattr(driver.study, "run_study", crash)
    with pytest.raises(RuntimeError):
        driver.main(holdout_argv(), loader=fake_inputs, results_dir=tmp_path)
    assert not (tmp_path / "holdout-A.json").exists()
    claim = tmp_path / "holdout-A.claim"
    assert claim.exists()  # a crash leaves the claim on purpose
    note = json.loads(claim.read_text())
    assert note["started_at"] and "git_head" in note and "--window" in " ".join(note["command"])
    loaded = []
    with pytest.raises(SystemExit) as refused:
        driver.main(holdout_argv(), loader=lambda a: loaded.append(a), results_dir=tmp_path)
    assert "holdout-A.claim" in str(refused.value)
    assert loaded == []  # refused before reading any data
    assert claim.exists()  # never deleted by code


def test_claim_is_taken_before_any_data_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = load_driver()
    seen = []

    def loader(args):
        seen.append((tmp_path / "holdout-A.claim").exists())
        return fake_inputs(args)

    stub_study(monkeypatch, driver)
    driver.main(holdout_argv(), loader=loader, results_dir=tmp_path)
    assert seen == [True]


def test_concurrent_holdout_claims_only_one_wins(tmp_path: Path) -> None:
    barrier = threading.Barrier(8)
    outcomes: list[str] = []

    def attempt() -> None:
        barrier.wait()
        try:
            study.claim_holdout("A", tmp_path, git_head="abc", command=["x"])
            outcomes.append("won")
        except SystemExit:
            outcomes.append("refused")

    threads = [threading.Thread(target=attempt) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["refused"] * 7 + ["won"]


def test_precreated_claim_refuses(tmp_path: Path) -> None:
    (tmp_path / "holdout-B.claim").write_text("{}")
    with pytest.raises(SystemExit) as refused:
        study.resolve_output("holdout", "B", None, tmp_path)
    assert "holdout-B.claim" in str(refused.value)
    with pytest.raises(SystemExit):
        study.claim_holdout("B", tmp_path, git_head=None, command=[])


def test_dev_run_has_no_claim_and_may_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = load_driver()
    stub_study(monkeypatch, driver)
    argv = ["--window", "dev", "--universe", "B", "--release-id", "1"]
    for _ in range(2):
        driver.main(argv, loader=fake_inputs, results_dir=tmp_path)
    assert (tmp_path / "dev-B.json").exists()
    assert list(tmp_path.glob("*.claim")) == []


def test_holdout_ignores_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    driver = load_driver()
    stub_study(monkeypatch, driver)
    elsewhere = tmp_path / "elsewhere.json"
    driver.main(
        [*holdout_argv("B"), "--out", str(elsewhere)], loader=fake_inputs, results_dir=tmp_path
    )
    assert (tmp_path / "holdout-B.json").exists() and not elsewhere.exists()


def test_result_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    driver = load_driver()
    stub_study(monkeypatch, driver)
    used = []
    real = driver.write_json
    monkeypatch.setattr(driver, "write_json", lambda *a, **k: used.append(a[0]) or real(*a, **k))
    driver.main(holdout_argv(), loader=fake_inputs, results_dir=tmp_path)
    assert used == [tmp_path / "holdout-A.json"]
    assert list(tmp_path.glob("*.tmp")) == []


def test_cli_has_no_results_dir_flag() -> None:
    parser = load_driver().build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([*holdout_argv(), "--results-dir", "/tmp/elsewhere"])
    assert "--results-dir" not in parser.format_help()


def test_default_results_folder_is_the_canonical_one(monkeypatch: pytest.MonkeyPatch) -> None:
    driver = load_driver()
    canonical = repo_root(__file__) / "results" / "dead_stock_first_test"
    assert canonical == driver.DEFAULT_RESULTS_DIR
    asked = []

    def stop(window, universe, out, folder):
        asked.append(folder)
        raise KeyboardInterrupt  # nothing is touched: stop right after the folder is chosen

    monkeypatch.setattr(driver.study, "resolve_output", stop)
    with pytest.raises(KeyboardInterrupt):
        driver.main(holdout_argv())
    assert asked == [canonical]


def spy_copy(data: dict, **changes) -> dict:
    return {**data, **changes}


def check_window(close: pd.DataFrame, spy: pd.Series) -> dict:
    return study.validate_window_inputs(close, spy, START, END)


def test_valid_window_is_accepted() -> None:
    facts = check_window(make_close(), make_spy())
    assert facts == {"first_day": START, "last_day": END, "spy_missing_closes": 0}


def test_spy_ending_a_year_early_is_refused() -> None:
    spy = make_spy()
    short = spy[spy.index <= "2010-12-31"]
    with pytest.raises(ValueError, match="SPY close ends early"):
        check_window(make_close(), short)
    with pytest.raises(ValueError, match="ends early"):
        run({**no_death_inputs(), "spy_close": short})


def test_spy_starting_late_is_refused() -> None:
    spy = make_spy()
    with pytest.raises(ValueError, match="SPY close starts late"):
        check_window(make_close(), spy[spy.index >= "2010-03-01"])


def test_strategy_close_ending_early_is_refused() -> None:
    close = make_close()
    with pytest.raises(ValueError, match="strategy close ends early"):
        check_window(close[close.index <= "2011-10-31"], make_spy())


def test_first_and_last_days_must_match_exactly() -> None:
    spy = make_spy()
    with pytest.raises(ValueError, match="first day differs"):
        check_window(make_close(), spy[spy.index != START])  # SPY's first is 2010-01-05
    with pytest.raises(ValueError, match="last day differs"):
        check_window(make_close(), spy[spy.index != END])


def test_spy_nan_run_longer_than_five_days_is_refused() -> None:
    spy = make_spy()
    spy.loc["2010-06-01":"2010-06-08"] = np.nan  # six trading days
    with pytest.raises(ValueError, match="6 closes in a row"):
        check_window(make_close(), spy)
    shorter = make_spy()
    shorter.loc["2010-06-01":"2010-06-03"] = np.nan
    assert check_window(make_close(), shorter)["spy_missing_closes"] == 3


def test_spy_nan_on_the_last_day_is_refused() -> None:
    spy = make_spy()
    spy.loc[END] = np.nan
    with pytest.raises(ValueError, match="first or last day"):
        check_window(make_close(), spy)


def test_window_with_a_utc_index_is_accepted() -> None:
    close, spy = make_close(), make_spy()
    close.index = close.index.tz_localize("UTC")
    spy.index = spy.index.tz_localize("UTC")
    assert check_window(close, spy)["first_day"] == START


def test_run_py_has_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as done:
        load_driver().main(["--help"])
    assert done.value.code == 0
    assert "--window" in capsys.readouterr().out


# ---- fix round 3: statistics, evidence, audit, drift, pins, labels -------------------------


def test_sharpe_and_drawdown_come_from_account_value_and_pretax_pair_is_kept() -> None:
    days = pd.bdate_range("2010-01-04", periods=60)
    pretax = pd.Series(np.linspace(100.0, 120.0, 60), index=days)
    account = pd.Series(
        np.linspace(100.0, 110.0, 60) * (1 + 0.01 * np.sin(np.arange(60))), index=days
    )
    book = BookResult(
        pretax_equity=pretax,
        after_tax_final=110.0,
        tax_paid=10.0,
        cost_paid=1.0,
        n_dead_exits=0,
        dead_exit_writeoff=0.0,
        cagr_pretax=0.2,
        cagr_after_tax=0.1,
        trades=[],
        account_value=account,
        unpaid_tax_flag=True,
    )
    cell = study._cell(book)
    from ggTrader.lab.wfo.report import curve_summary

    assert cell["sharpe"] == pytest.approx(curve_summary(account)["sharpe"])
    assert cell["max_drawdown_pct"] == pytest.approx(curve_summary(account)["max_drawdown_pct"])
    assert cell["sharpe_pretax_curve"] == pytest.approx(curve_summary(pretax)["sharpe"])
    assert cell["max_drawdown_pct_pretax_curve"] == pytest.approx(
        curve_summary(pretax)["max_drawdown_pct"]
    )
    assert cell["unpaid_tax_flag"] is True
    assert cell["account_value_month_end"][-1][0] == str(days[-1].date())


def test_negative_final_value_does_not_crash_the_cell_or_the_gap() -> None:
    days = pd.bdate_range("2010-01-04", periods=40)
    account = pd.Series(np.linspace(100.0, -5.0, 40), index=days)
    book = BookResult(
        pretax_equity=pd.Series(np.linspace(100.0, 50.0, 40), index=days),
        after_tax_final=-5.0,
        tax_paid=0.0,
        cost_paid=0.0,
        n_dead_exits=0,
        dead_exit_writeoff=0.0,
        cagr_pretax=-0.5,
        cagr_after_tax=-1.0,
        trades=[],
        account_value=account,
        unpaid_tax_flag=True,
    )
    cell = study._cell(book)
    assert cell["account_value_nonpositive"] is True
    assert np.isnan(cell["sharpe"]) and cell["max_drawdown_pct"] == -100.0
    parts = study.decomposition_for_loss(
        spy_zero=8.0,
        rule_loss0_zero=9.0,
        rule_zero=7.0,
        rule_cost_no_tax=6.0,
        rule_final=-100.0,
        spy_final=6.5,
    )
    assert parts["reconciles"] is True and parts["total_gap_pp"] == pytest.approx(-106.5)


def test_every_cell_has_a_month_end_account_value_series(dead: dict) -> None:
    for cell in [*dead["strategy"].values(), *dead["spy"].values()]:
        series = cell["account_value_month_end"]
        assert 20 <= len(series) <= 24 and all(len(row) == 2 for row in series)
        assert series[-1][1] == pytest.approx(cell["after_tax_final"])
        assert "unpaid_tax_flag" in cell and "sharpe_pretax_curve" in cell


def test_gap_decomposition_reconciles_exactly(dead: dict) -> None:
    block = dead["gap_decomposition"]
    assert set(block["by_loss"]) == {"loss1", "loss0.5", "loss0"}
    for name, part in block["by_loss"].items():
        loss = float(name.removeprefix("loss"))
        total = (
            part["selection_effect_pp"]
            + part["dead_stock_effect_pp"]
            + part["trading_cost_effect_pp"]
            + part["tax_effect_pp"]
        )
        assert total == pytest.approx(part["steps_sum_pp"])
        assert total + part["spy_cost_and_tax_drag_pp"] == pytest.approx(part["total_gap_pp"])
        assert part["reconciles"] is True and part["matches_cell_gap"] is True
        cell = dead["strategy"][study.cell_key(loss, 0.00183, "low")]
        assert part["total_gap_pp"] == pytest.approx(cell["gap_cagr_after_tax_pp"])
        assert part["rule_minus_spy_zero_cost_zero_tax_pp"] == pytest.approx(total)
    assert block["by_loss"]["loss0"]["dead_stock_effect_pp"] == pytest.approx(0.0)
    assert block["by_loss"]["loss1"]["dead_stock_effect_pp"] < 0.0  # the dead stock is written off
    assert block["by_loss"]["loss1"]["trading_cost_effect_pp"] < 0.0


def test_holdings_audit_function_counts_dead_holdings_and_history_split() -> None:
    days = pd.bdate_range("2010-01-04", "2010-04-30")
    t = lambda d, sym, q: Trade(pd.Timestamp(d), sym, q, 10.0, "rebalance")  # noqa: E731
    trades = [t("2010-01-05", "Y", 5.0), t("2010-01-05", "C", 5.0), t("2010-02-26", "Y", -5.0)]
    ends = pd.Series(pd.to_datetime(["2010-03-15", "2010-03-20"]), index=["C", "Y"])
    month_ends = [pd.Timestamp(d) for d in ("2010-01-29", "2010-02-26", "2010-03-31")]
    kinds = {"Y": "linked_yahoo", "C": "placeholder"}
    audit = study.holdings_audit(trades, month_ends, ends, frozenset({"Y"}), kinds)
    rows = audit["months"]
    assert [r["n_held"] for r in rows] == [2, 1, 1]
    assert [r["n_dead_holdings"] for r in rows] == [1, 1, 0]  # C dies 2010-03-15; Y is later-spell
    assert rows[0]["n_yahoo_history"] == 1 and rows[0]["n_crsp_only"] == 1
    assert rows[1]["n_yahoo_history"] == 0 and rows[1]["n_crsp_only"] == 1
    assert audit["totals"]["held_stock_months"] == 4
    assert (
        study.holdings_audit(trades, month_ends, ends, frozenset(), None)["months"][0][
            "n_yahoo_history"
        ]
        is None
    )
    assert len(days) > 0


def test_run_study_reports_holdings_for_the_three_losses(dead: dict) -> None:
    assert set(dead["holdings_audit"]) == {
        study.cell_key(loss, 0.00183, "low") for loss in (1.0, 0.5, 0.0)
    }
    months = dead["holdings_audit"][study.cell_key(1.0, 0.00183, "low")]["months"]
    assert any(m["n_dead_holdings"] for m in months)  # D is held before its death
    assert all(m["n_yahoo_history"] is None for m in months)  # no release kinds were passed


def test_holdings_split_uses_release_kinds() -> None:
    data = death_inputs()
    kinds = pd.DataFrame(
        {
            "symbol": list(data["close"].columns),
            "permno": range(len(data["close"].columns)),
            "kind": ["yahoo"] * 6 + ["placeholder"] * (len(data["close"].columns) - 6),
        }
    )
    result = run(data, release_spans=kinds)
    totals = result["holdings_audit"][study.cell_key(1.0, 0.00183, "low")]["totals"]
    assert (
        totals["yahoo_history_stock_months"] + totals["crsp_only_stock_months"]
        == totals["held_stock_months"]
    )


def test_n_ended_series_counts_only_the_window() -> None:
    data = death_inputs()
    close = data["close"]
    close["W"] = 20.0
    close.loc[close.index > "2009-06-30", "W"] = np.nan
    data["eligible"] = pd.DataFrame(True, index=close.index, columns=close.columns)
    data["span_rows"] = pd.concat(
        [
            data["span_rows"],
            pd.DataFrame(
                {
                    "symbol": ["W"],
                    "last_date": [pd.Timestamp("2009-06-30")],
                    "has_successor": [False],
                    "total_loss": [False],
                }
            ),
        ],
        ignore_index=True,
    )
    result = run(data)
    assert result["n_ended_series_in_window"] == 1
    assert result["n_ended_series_in_read"] == 2


def test_later_spell_labels_replace_the_buyout_labels(dead: dict) -> None:
    year = dead["coverage"]["2011"]
    assert "ever_eligible_later_spell" in year and "held_at_later_spell" in year
    assert "ever_eligible_bought_out" not in year and "held_at_buyout" not in year


def test_skipped_buys_note_says_the_reasons_are_not_split(dead: dict) -> None:
    assert "does not split" in dead["skipped_buys_note"]
    assert all("skipped_buys" in c for c in dead["strategy"].values())


# trades side file


def stub_run_study_with_trades(monkeypatch: pytest.MonkeyPatch, driver) -> None:
    def fake(*_a, trade_sink=None, **_k):
        if trade_sink is not None:
            trade_sink["loss1_cost18.3bps_taxlow"] = [
                Trade(pd.Timestamp("2020-01-03"), "AAA", 3.5, 10.0, "rebalance"),
                Trade(pd.Timestamp("2020-02-03"), "AAA", -3.5, 11.0, "liquidation"),
            ]
            trade_sink["spy_cost18.3bps_taxlow"] = [
                Trade(pd.Timestamp("2020-01-02"), "SPY", 1.0, 300.0, "rebalance")
            ]
        return {"strategy": {}, "spy": {}}

    monkeypatch.setattr(driver.study, "run_study", fake)


def test_trades_side_file_is_written_before_the_main_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = load_driver()
    stub_run_study_with_trades(monkeypatch, driver)
    real = driver.write_json
    seen = []

    def watching(path, payload, **kw):
        seen.append((Path(path).name, (tmp_path / "holdout-A-trades.csv.gz").exists()))
        return real(path, payload, **kw)

    monkeypatch.setattr(driver, "write_json", watching)
    driver.main(holdout_argv(), loader=fake_inputs, results_dir=tmp_path)
    assert seen == [("holdout-A.json", True)]
    with gzip.open(tmp_path / "holdout-A-trades.csv.gz", "rt") as handle:
        rows = handle.read().splitlines()
    assert rows[0] == "cell,date,symbol,qty,price,reason"
    assert "loss1_cost18.3bps_taxlow,2020-01-03,AAA,3.5,10.0,rebalance" in rows
    assert "spy_cost18.3bps_taxlow,2020-01-02,SPY,1.0,300.0,rebalance" in rows
    assert list(tmp_path.glob("*.tmp")) == []


def test_existing_trades_file_refuses_the_holdout_before_the_claim(tmp_path: Path) -> None:
    driver = load_driver()
    (tmp_path / "holdout-A-trades.csv.gz").write_bytes(b"x")
    with pytest.raises(SystemExit) as refused:
        driver.main(holdout_argv(), loader=fake_inputs, results_dir=tmp_path)
    assert "holdout-A-trades.csv.gz" in str(refused.value)
    assert not (tmp_path / "holdout-A.claim").exists()


def test_every_cell_of_a_real_run_lands_in_the_trade_file(tmp_path: Path) -> None:
    sink: dict[str, list] = {}
    data = death_inputs()
    study.run_study(
        data["close"],
        data["spy_close"],
        data["eligible"],
        data["span_rows"],
        start=START,
        end=END,
        trade_sink=sink,
    )
    assert len(sink) == 24 and sum(k.startswith("spy_") for k in sink) == 6
    path = study.write_trades_csv(tmp_path / "t.csv.gz", sink)
    with gzip.open(path, "rt") as handle:
        cells = {line.split(",")[0] for line in handle.read().splitlines()[1:]}
    assert cells == set(sink)


# false-death audit


class StubCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self.conn.queries.append((sql, params))

    def fetchall(self):
        return self.conn.rows


class StubConn:
    def __init__(self, rows):
        self.rows = rows
        self.queries: list = []

    def cursor(self):
        return StubCursor(self)


def audit_spans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [
                "OK",
                "FALSE",
                "LATER1",
                "LATER2",
                "OUTSIDE",
                "ALIVE",
                "Y2A",
                "Y2B",
                "NOPERM",
            ],
            "permno": [1, 2, 3, 3, 4, 5, 6, 6, None],
            "spell": [1, 1, 1, 2, 1, 1, 1, 1, 1],
            "kind": [
                "yahoo",
                "linked_yahoo",
                "spell",
                "spell",
                "yahoo",
                "placeholder",
                "yahoo",
                "linked_yahoo",
                "yahoo",
            ],
            "first_date": pd.to_datetime(["2000-01-03"] * 9),
            "last_date": pd.to_datetime(
                [
                    "2012-05-01",
                    "2012-05-01",
                    "2012-05-01",
                    "2015-01-02",
                    "2022-05-01",
                    "2024-12-31",
                    "2012-06-01",
                    "2012-06-01",
                    "2012-05-01",
                ]
            ),
        }
    )


def test_false_death_audit_counts_only_confirmed_links_without_a_later_spell() -> None:
    conn = StubConn(
        [
            (1, pd.Timestamp("2012-05-05")),  # 4 days: inside the 7-day grace, fine
            (2, pd.Timestamp("2014-03-03")),  # linked_yahoo ended, CRSP runs on: suspect
            (3, pd.Timestamp("2015-01-02")),  # spell column with a later spell: informational
            (6, pd.Timestamp("2012-06-01")),
            (7, pd.Timestamp("2013-01-02")),  # unlinked yahoo candidate: informational
            (8, pd.Timestamp("2012-05-02")),  # a linked_yahoo that really died: nothing
        ]
    )
    spans = pd.concat(
        [
            audit_spans(),
            pd.DataFrame(
                {
                    "symbol": ["CAND", "DIED"],
                    "permno": [7, 8],
                    "spell": [1, 1],
                    "kind": ["yahoo", "linked_yahoo"],
                    "first_date": pd.to_datetime(["2000-01-03"] * 2),
                    "last_date": pd.to_datetime(["2012-05-01"] * 2),
                }
            ),
        ],
        ignore_index=True,
    )
    audit = study.audit_false_deaths(
        conn,
        spans,
        crsp_load=1,
        window_start="2005-01-03",
        window_end="2019-12-31",
        crsp_end="2024-12-31",
    )
    assert audit["n_checked"] == 8  # OUTSIDE (2022), ALIVE (file end) and NOPERM are not checked
    assert [s["symbol"] for s in audit["suspects"]] == ["FALSE"]
    assert audit["n_suspect"] == 1
    assert audit["suspects"][0]["permno"] == 2 and audit["suspects"][0]["kind"] == "linked_yahoo"
    assert audit["suspects"][0]["crsp_last_date"] == "2014-03-03"
    assert audit["n_informational"] == 2
    info = audit["informational"]
    assert info["counts_by_kind"] == {"spell": 1, "yahoo": 1}
    assert {r["symbol"] for r in info["first_rows"]} == {"LATER1", "CAND"}
    assert "never blocks" in info["label"]
    spell_row = next(r for r in info["first_rows"] if r["symbol"] == "LATER1")
    assert spell_row["has_later_spell"] is True
    for name in ("n_checked", "n_suspect", "suspects", "multi_symbol_permnos"):
        assert name in audit
    assert audit["grace_calendar_days"] == 7 and audit["crsp_end"] == "2024-12-31"
    assert audit["window"] == {"start": "2005-01-03", "end": "2019-12-31"}
    sql, params = conn.queries[0]
    assert "kaggle_crsp.daily_of" in sql and "max(date)" in sql and params[0] == 1
    assert "close" not in sql.lower().replace("kaggle_crsp", "")  # no prices or returns read


def test_multi_symbol_permnos_use_linked_yahoo_only() -> None:
    spans = pd.DataFrame(
        {
            "symbol": ["L1", "L2", "U1", "U2", "ONE"],
            "permno": [10, 10, 20, 20, 30],
            "spell": [1] * 5,
            "kind": ["linked_yahoo", "linked_yahoo", "yahoo", "yahoo", "linked_yahoo"],
            "first_date": pd.to_datetime(["2000-01-03"] * 5),
            "last_date": pd.to_datetime(["2024-12-31"] * 5),
        }
    )
    audit = study.audit_false_deaths(
        StubConn([]),
        spans,
        crsp_load=1,
        window_start="2005-01-03",
        window_end="2019-12-31",
        crsp_end="2024-12-31",
    )
    assert audit["multi_symbol_permnos"] == [{"permno": 10, "symbols": ["L1", "L2"]}]
    assert audit["n_unlinked_candidate_permnos_with_several_symbols"] == 1


def test_informational_rows_are_cut_at_twenty() -> None:
    n = 30
    spans = pd.DataFrame(
        {
            "symbol": [f"Y{i}" for i in range(n)],
            "permno": list(range(100, 100 + n)),
            "spell": [1] * n,
            "kind": ["yahoo"] * n,
            "first_date": pd.to_datetime(["2000-01-03"] * n),
            "last_date": pd.to_datetime(["2012-05-01"] * n),
        }
    )
    conn = StubConn([(100 + i, pd.Timestamp("2015-01-02")) for i in range(n)])
    audit = study.audit_false_deaths(
        conn,
        spans,
        crsp_load=1,
        window_start="2005-01-03",
        window_end="2019-12-31",
        crsp_end="2024-12-31",
    )
    assert audit["n_suspect"] == 0 and audit["n_informational"] == 30
    assert len(audit["informational"]["first_rows"]) == 20


def test_audit_with_nothing_to_check_asks_the_database_nothing() -> None:
    conn = StubConn([])
    empty = audit_spans().iloc[0:0]
    audit = study.audit_false_deaths(
        conn,
        empty,
        crsp_load=1,
        window_start="2005-01-03",
        window_end="2019-12-31",
        crsp_end="2024-12-31",
    )
    assert audit["n_checked"] == 0 and conn.queries == []


def test_audit_only_never_claims_and_never_loads_prices(tmp_path: Path) -> None:
    driver = load_driver()

    def boom(*_a, **_k):
        raise AssertionError("prices must not be loaded")

    block = {"n_checked": 3, "n_suspect": 0, "suspects": [], "multi_symbol_permnos": []}
    code = driver.main(
        [*holdout_argv(), "--audit-only"],
        loader=boom,
        audit_loader=lambda _a: block,
        results_dir=tmp_path,
    )
    assert code == 0
    payload = json.loads((tmp_path / "holdout-A-audit.json").read_text())
    assert payload["false_death_audit"] == block
    assert list(tmp_path.glob("*.claim")) == [] and not (tmp_path / "holdout-A.json").exists()
    # it may be repeated, and it still never touches the claim
    driver.main(
        [*holdout_argv(), "--audit-only"], audit_loader=lambda _a: block, results_dir=tmp_path
    )


def test_main_run_includes_the_audit_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    driver = load_driver()
    stub_study(monkeypatch, driver)

    def loader(args):
        return {**fake_inputs(args), "false_death_audit": {"n_checked": 7}}

    driver.main(
        ["--window", "dev", "--universe", "A", "--release-id", "1"],
        loader=loader,
        results_dir=tmp_path,
    )
    result = json.loads((tmp_path / "dev-A.json").read_text())["result"]
    assert result["false_death_audit"] == {"n_checked": 7}


# drift


def test_reader_carries_drift_into_its_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    import bars_reader

    def fake_load_bars(symbols, start, end, *, source, release_id):
        days = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=3), tz="UTC")
        frame = pd.DataFrame(1.0, index=days, columns=sorted(symbols))
        drifted = "B" in symbols
        return SimpleNamespace(
            close=frame,
            spans=pd.DataFrame({"symbol": [], "last_date": [], "has_successor": []}),
            provenance={
                "data_release": {"release_id": 1},
                "bars_sha256": "x",
                "drift": drifted,
                "drift_symbols": ["B"] if drifted else [],
            },
        )

    monkeypatch.setattr(bars_reader, "load_bars", fake_load_bars)
    clean: dict = {}
    bars_reader.read_close_batched(
        ["A"], "2020-01-02", "2020-01-06", release_id=1, provenance_out=clean
    )
    assert clean["drift"] is False and clean["drift_symbols"] == []
    found: dict = {}
    bars_reader.read_close_batched(
        ["A", "B", "C"], "2020-01-02", "2020-01-06", release_id=1, batch=1, provenance_out=found
    )
    assert found["drift"] is True and found["drift_symbols"] == ["B"]


def test_linked_drift_preflight_reads_metadata_only() -> None:
    import bars_reader

    calls = []

    class Cursor(StubCursor):
        def execute(self, sql, params):
            calls.append((sql, params))
            self.rows = [("L1",), ("L2",)] if "FROM market.stock_spans" in sql else [("L2",)]

        def fetchall(self):
            return self.rows

    class Conn:
        def cursor(self):
            return Cursor(self)

    assert bars_reader.read_linked_drift(Conn(), 1) == ["L2"]
    assert calls[1][1] == (1, ["L1", "L2"])
    assert "ohlcv" not in calls[1][0].lower() and "daily" not in calls[1][0].lower()


def test_holdout_refuses_on_drift_before_the_claim(tmp_path: Path) -> None:
    driver = load_driver()
    with pytest.raises(SystemExit) as refused:
        driver.main(
            holdout_argv(),
            loader=lambda _a: pytest.fail("data was read"),
            results_dir=tmp_path,
            drift_preflight=lambda _a: ["LNK1", "LNK2"],
        )
    assert "drift" in str(refused.value) and "nothing was claimed" in str(refused.value)
    assert list(tmp_path.glob("*")) == []


def test_holdout_stops_after_the_read_when_drift_shows_up_and_keeps_the_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = load_driver()
    stub_study(monkeypatch, driver)

    def drifting(args):
        data = fake_inputs(args)
        data["data_provenance"] = {"candidates": {"drift": True, "drift_symbols": ["XYZ"]}}
        return data

    with pytest.raises(SystemExit) as refused:
        driver.main(holdout_argv(), loader=drifting, results_dir=tmp_path)
    assert "XYZ" in str(refused.value)
    assert (tmp_path / "holdout-A.claim").exists() and not (tmp_path / "holdout-A.json").exists()


def test_dev_records_drift_but_does_not_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = load_driver()
    stub_study(monkeypatch, driver)

    def drifting(args):
        data = fake_inputs(args)
        data["data_provenance"] = {"candidates": {"drift": True, "drift_symbols": ["XYZ"]}}
        return data

    driver.main(
        ["--window", "dev", "--universe", "A", "--release-id", "1"],
        loader=drifting,
        results_dir=tmp_path,
        drift_preflight=lambda _a: pytest.fail("dev must not run the holdout preflight"),
    )
    payload = json.loads((tmp_path / "dev-A.json").read_text())
    assert payload["provenance"]["drift_symbols"] == ["XYZ"]


# pins


def test_pin_constants() -> None:
    assert study.PIN_RELEASE_ID == 1 and study.PIN_CRSP_LOAD == 1


@pytest.mark.parametrize(
    ("release", "crsp", "dirty", "word"),
    [(2, 1, False, "release-id"), (1, 2, False, "crsp-load"), (1, 1, True, "working tree")],
)
def test_holdout_pins_refuse_before_the_claim(
    tmp_path: Path, release: int, crsp: int, dirty: bool, word: str
) -> None:
    driver = load_driver()
    argv = ["--window", "holdout", "--universe", "A", "--release-id", str(release)]
    argv += ["--crsp-load", str(crsp)]
    with pytest.raises(SystemExit) as refused:
        driver.main(
            argv,
            loader=lambda _a: pytest.fail("data was read"),
            results_dir=tmp_path,
            git_dirty=lambda: dirty,
        )
    assert word in str(refused.value)
    assert list(tmp_path.glob("*")) == []


def test_clean_pinned_holdout_passes_the_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = load_driver()
    stub_study(monkeypatch, driver)
    assert (
        driver.main(
            holdout_argv(),
            loader=fake_inputs,
            results_dir=tmp_path,
            git_dirty=lambda: False,
            drift_preflight=lambda _a: [],
        )
        == 0
    )


def test_dev_ignores_the_pins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    driver = load_driver()
    stub_study(monkeypatch, driver)
    argv = ["--window", "dev", "--universe", "A", "--release-id", "2", "--crsp-load", "3"]
    assert driver.main(argv, loader=fake_inputs, results_dir=tmp_path, git_dirty=lambda: True) == 0


def test_git_tree_dirty_reads_tracked_changes_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return SimpleNamespace(stdout=" M some/file.py\n")

    monkeypatch.setattr(study.subprocess, "run", fake_run)
    assert study.git_tree_dirty(Path(".")) is True
    assert seen["cmd"] == ["git", "status", "--porcelain", "--untracked-files=no"]
    monkeypatch.setattr(study.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=""))
    assert study.git_tree_dirty(Path(".")) is False


def test_a_real_run_writes_valid_json_with_every_new_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No stubs: the whole result, including the new blocks, must serialise."""
    driver = load_driver()
    monkeypatch.setattr(study, "WINDOWS", {**study.WINDOWS, "dev": (START, END)})
    data = death_inputs()
    driver.main(
        ["--window", "dev", "--universe", "A", "--release-id", "1"],
        loader=lambda _a: {**data, "false_death_audit": {"n_checked": 0}},
        results_dir=tmp_path,
    )
    payload = json.loads((tmp_path / "dev-A.json").read_text())
    result = payload["result"]
    for name in (
        "strategy",
        "spy",
        "coverage",
        "sanity",
        "window_check",
        "death_price_paths",
        "gap_decomposition",
        "holdings_audit",
        "n_file_end_censored",
        "n_rebalance_steps",
        "n_columns",
        "n_ended_series_in_window",
        "n_ended_series_in_read",
        "false_death_audit",
    ):
        assert name in result
    assert payload["provenance"]["crsp_end"] == "2024-12-31"
    assert (tmp_path / "dev-A-trades.csv.gz").exists()
