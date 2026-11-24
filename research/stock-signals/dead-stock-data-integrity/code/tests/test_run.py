"""Tests for the Phase 1 driver: batching, argument refusals, labels, symbol mapping, outputs."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import dead_stock_data_integrity as study
import numpy as np
import pandas as pd
import pytest
import run

pytestmark = pytest.mark.study

CHECK_KEYS = {"duplicates", "bad_price", "spikes", "gaps", "stale", "shares"}
DATES = pd.bdate_range("2020-01-01", periods=40)


def _stock(permno: int, spike: bool) -> pd.DataFrame:
    """One stock over 40 business days; the first stock carries one flag of every kind."""
    frame = pd.DataFrame(
        {
            "permno": permno,
            "date": DATES,
            "prc": 100.0 + np.arange(40),
            "ret": 0.01,
            "shrout": 1000.0,
        }
    )
    if not spike:
        return frame
    frame.loc[5, "ret"] = 0.8
    frame.loc[6, "ret"] = -0.4444
    frame.loc[20:29, "prc"] = 50.0
    frame.loc[22, "shrout"] = np.nan  # 2020-01-31, the last January row
    frame = frame.drop(index=range(12, 17))  # five days missing
    return pd.concat([frame, frame[frame["date"] == DATES[30]]], ignore_index=True)


def _intervals() -> dict[str, pd.DataFrame]:
    def one(start: int, end: int) -> pd.DataFrame:
        return pd.DataFrame({"permno": [10], "start": [DATES[start]], "end": [DATES[end]]})

    return {"dev-A": one(3, 8), "dev-B": one(10, 25)}


def test_integrity_phase_writes_json_with_every_check_name(tmp_path: Path) -> None:
    batches = [_stock(10, True), _stock(20, False)]
    result = study.compute_integrity(batches, pd.DatetimeIndex(DATES), _intervals())
    assert CHECK_KEYS <= set(result) and "holdings_overlap" in result
    path = tmp_path / "x.json"
    path.write_text(json.dumps(result))
    loaded = json.loads(path.read_text())
    for key in CHECK_KEYS:
        assert set(loaded[key]) >= {
            "rows_flagged",
            "stocks_flagged",
            "examples",
            "flags_inside_holdings",
        }
    assert loaded["scan"] == {"batches": 2, "permnos": 2, "rows": 76}


def test_every_flag_is_counted_and_overlap_follows_the_intervals() -> None:
    result = study.compute_integrity(
        [_stock(10, True), _stock(20, False)], pd.DatetimeIndex(DATES), _intervals()
    )
    counts = {k: result[k]["rows_flagged"] for k in CHECK_KEYS}
    assert counts == {
        "duplicates": 1,
        "bad_price": 0,
        "spikes": 1,
        "gaps": 1,
        "stale": 1,
        "shares": 1,
    }
    assert result["spikes"]["examples"][0]["magnitude"] == pytest.approx(0.8)
    assert result["gaps"]["examples"][0]["detail"] == "5 days missing"
    inside = {
        name: {k: result[k]["flags_inside_holdings"][name] for k in CHECK_KEYS}
        for name in ("dev-A", "dev-B")
    }
    assert inside["dev-A"] == {
        "duplicates": 0,
        "bad_price": 0,
        "spikes": 1,
        "gaps": 0,
        "stale": 0,
        "shares": 0,
    }
    assert inside["dev-B"] == {
        "duplicates": 0,
        "bad_price": 0,
        "spikes": 0,
        "gaps": 1,
        "stale": 1,
        "shares": 1,
    }
    overlap = result["holdings_overlap"]["dev-B"]["gaps"]
    assert overlap["flags_inside"] == 1 and overlap["stocks_inside"] == 1


def test_examples_are_capped_at_twenty_but_every_flag_is_counted() -> None:
    rows = pd.concat([_stock(p, False).assign(prc=0.0) for p in range(100, 130)])
    result = study.compute_integrity([rows], pd.DatetimeIndex(DATES), {})
    assert result["bad_price"]["rows_flagged"] == 30 * 40
    assert result["bad_price"]["stocks_flagged"] == 30
    assert len(result["bad_price"]["examples"]) == study.EXAMPLES_PER_CHECK


def test_batches_are_500_permnos_each() -> None:
    sizes = [len(b) for b in run.permno_batches(list(range(1203)), size=500)]
    assert sizes == [500, 500, 203]
    assert [len(b) for b in study.permno_batches(list(range(1203)))] == [500, 500, 203]


def test_refuses_eval_end_after_crsp_end(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        run.main(_argv(tmp_path, "--eval-end", "2025-01-02"))


def test_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    (tmp_path / study.OUTPUT_JSON).write_text("{}")
    with pytest.raises(SystemExit):
        run.main(_argv(tmp_path))


def _argv(out: Path, *extra: str) -> list[str]:
    return [
        "--phase",
        "integrity",
        "--release-id",
        "1",
        "--crsp-load",
        "1",
        "--mcgill-load",
        "2",
        "--out",
        str(out),
        *extra,
    ]


def test_label_roster_chains_returns_up_to_the_exit_date() -> None:
    dates = pd.bdate_range("2020-01-01", periods=30)
    returns = pd.DataFrame({"permno": 7, "date": dates, "ret": 0.0})
    returns.loc[28, "ret"] = -0.9  # after the exit date: must not count
    roster = pd.DataFrame(
        {
            "permno": pd.array([7, None], dtype="Int64"),
            "exit_date": [dates[26], dates[26]],
        }
    )
    out = study.label_roster(roster, returns)
    assert out.loc[0, "price_label"] == "calm_like_deal"
    assert out.loc[0, "ratio"] == pytest.approx(1.0)
    assert out.loc[1, "price_label"] == "too_short"


def test_ticker_symbols_are_rewritten_through_the_symbol_map() -> None:
    day = dt.date
    mapping = pd.DataFrame(
        [
            (7, "AAA", day(2019, 1, 1), day(2020, 12, 31), "x", 1),
            (8, "AAA", day(2021, 1, 1), day(2022, 12, 31), "x", 1),
        ],
        columns=["permno", "symbol", "valid_from", "valid_to", "source", "release_id"],
    )
    trades = pd.DataFrame(
        {
            "cell": "c",
            "date": pd.to_datetime(["2020-06-01", "2021-06-01", "2020-06-01", "2020-06-01"]),
            "symbol": ["AAA", "AAA", "ZZZ", "PERMNO-5"],
            "qty": 1.0,
        }
    )
    out, info = study.map_trade_symbols(trades, mapping)
    assert list(out["symbol"]) == ["PERMNO-7", "PERMNO-8", "ZZZ", "PERMNO-5"]
    assert info["ticker_rows"] == 3 and info["mapped_rows"] == 2
    assert info["unmapped_rows"] == 1 and info["unmapped_symbols"] == {"ZZZ": 1}
    assert info["fallback_rows"] == 0


def test_run_integrity_writes_roster_and_json_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roster = pd.DataFrame(
        {
            "symbol": ["PERMNO-10"],
            "permno": pd.array([10], dtype="Int64"),
            "window": ["dev"],
            "universe": ["A"],
            "exit_date": [DATES[26]],
            "last_close": [1.0],
            "name_source": ["mcgill"],
        }
    )
    returns = pd.DataFrame({"permno": 10, "date": DATES, "ret": 0.0})
    monkeypatch.setattr(study.roster_mod, "build_roster", lambda *a, **k: roster)
    monkeypatch.setattr(study, "read_roster_returns", lambda *a, **k: returns)
    monkeypatch.setattr(study, "read_scan_permnos", lambda *a, **k: [10, 20])
    monkeypatch.setattr(study, "require_complete_load", lambda *a, **k: None)
    monkeypatch.setattr(study, "read_calendar", lambda *a, **k: pd.DatetimeIndex(DATES))
    monkeypatch.setattr(
        study,
        "load_intervals",
        lambda *a, **k: (_intervals(), {n: {"positions": 1} for n in _intervals()}),
    )

    def batches(conn: object, ids: list[int]):
        assert ids == [10, 20]
        yield _stock(10, True)
        yield _stock(20, False)

    payload = study.run_integrity(
        None,
        release_id=1,
        crsp_load=1,
        mcgill_load=2,
        eval_start="2005-01-03",
        eval_end="2024-12-31",
        out_dir=tmp_path,
        repo_root=tmp_path,
        run_date="2026-11-24",
        batches_fn=batches,
    )
    on_disk = json.loads((tmp_path / study.OUTPUT_JSON).read_text())
    assert CHECK_KEYS <= set(on_disk) and "holdings_overlap" in on_disk
    assert on_disk == json.loads(json.dumps(payload))
    prov = on_disk["provenance"]
    assert prov["rows_scanned"] == 76 and prov["permnos_scanned"] == 2
    assert {"release_id", "crsp_load", "mcgill_load", "eval_end", "git_head", "run_date"} <= set(
        prov
    )
    assert on_disk["holdings_overlap"]["dev-A"]["positions"] == 1
    csv = pd.read_csv(tmp_path / study.OUTPUT_ROSTER)
    assert {"ratio", "daily_std", "price_label"} <= set(csv.columns)
    assert list(tmp_path.glob("*.tmp")) == []


def test_sql_is_parameterised_and_reads_by_load_id() -> None:
    for sql in (study.PERMNOS_SQL, study.CALENDAR_SQL, study.BATCH_SQL, study.ROSTER_RET_SQL):
        assert "d.load_id = %s::bigint" in sql
        assert "daily_of" not in sql


def test_spell_suffix_symbols_fold_into_their_permno() -> None:
    trades = pd.DataFrame(
        {
            "cell": "c",
            "date": pd.to_datetime(["2020-06-01", "2020-06-02"]),
            "symbol": ["PERMNO-38420~1", "PERMNO-38420"],
            "qty": [1.0, -1.0],
        }
    )
    empty = pd.DataFrame(columns=["permno", "symbol", "valid_from", "valid_to"])
    out, info = study.map_trade_symbols(trades, empty)
    assert list(out["symbol"]) == ["PERMNO-38420", "PERMNO-38420"]
    assert info["spell_suffix_rows"] == 1 and info["ticker_rows"] == 0
    assert len(study.ic.holding_intervals(out, "c")) == 1
