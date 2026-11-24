"""Tests for the dead-stock first test driver: the grid, the windows and the holdout guard."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import dead_stock_first_test as study
import numpy as np
import pandas as pd
import pytest

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
