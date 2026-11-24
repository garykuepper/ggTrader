"""Tests for the Phase 1 driver: batching, argument refusals, labels, symbol mapping, outputs."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import dead_stock_data_integrity as study
import numpy as np
import pandas as pd
import pytest
import roster as roster_mod
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


class FakeCrsp:
    """A fake connection holding CRSP-like rows; it filters rows only if the SQL asks for it."""

    def __init__(self, rows: pd.DataFrame) -> None:
        self.rows = rows
        self.executed: list[tuple[str, tuple]] = []
        self._out: list[tuple] = []

    def cursor(self) -> FakeCrsp:
        return self

    def __enter__(self) -> FakeCrsp:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def execute(self, sql: str, params: tuple) -> None:
        self.executed.append((sql, params))
        frame = self.rows
        if "d.ret FROM" in sql:
            wanted = set(params[1])
            frame = frame[frame["permno"].isin(wanted)]
            self._out = list(frame[["permno", "date", "ret"]].itertuples(index=False, name=None))
        elif "SELECT DISTINCT d.permno" in sql:
            # PERMNOS_SQL: the codes choose stocks; one qualifying row is enough.
            keep = frame[
                frame["shrcd"].isin(params[1])
                & frame["exchcd"].isin(params[2])
                & (frame["date"] >= pd.Timestamp(params[3]))
            ]
            self._out = [(int(p),) for p in sorted(keep["permno"].unique())]
        else:
            # BATCH_SQL: only the permno list, dates; a code filter would only apply if present.
            assert "shrcd" not in sql and "exchcd" not in sql
            frame = frame[frame["permno"].isin(set(params[1]))]
            frame = frame[frame["date"] >= pd.Timestamp(params[2])]
            cols = ["permno", "date", "prc", "ret", "shrout"]
            self._out = list(frame[cols].itertuples(index=False, name=None))

    def fetchmany(self, n: int) -> list[tuple]:
        rows, self._out = self._out[:n], self._out[n:]
        return rows

    def fetchall(self) -> list[tuple]:
        rows, self._out = self._out, []
        return rows


def _two_exchange_stock() -> pd.DataFrame:
    """One stock whose days 12 to 16 sit on exchange 2 (NYSE American); no day is missing."""
    frame = _stock(30, False)
    frame["shrcd"] = 11
    frame["exchcd"] = 3
    frame.loc[12:16, "exchcd"] = 2
    return frame


def test_exchange_move_creates_no_gap_flag() -> None:
    conn = FakeCrsp(_two_exchange_stock())
    batch = study.read_batch(conn, 1, [30], "2024-12-31")
    assert len(batch) == 40
    result = study.compute_integrity([batch], pd.DatetimeIndex(DATES), {})
    assert result["gaps"]["rows_flagged"] == 0
    # The old row filter would have dropped five days and made a five-day "gap".
    old = _two_exchange_stock().query("exchcd in (1, 3)")
    flags = study.ic.check_gaps(old, pd.DatetimeIndex(DATES))
    assert list(flags["detail"]) == ["5 days missing"]


def test_codes_pick_permnos_but_one_qualifying_row_is_enough() -> None:
    frames = [
        _two_exchange_stock(),  # moves between exchanges 3 and 2: qualifies
        _stock(31, False).assign(shrcd=11, exchcd=2),  # never on exchange 1 or 3
        _stock(32, False).assign(shrcd=73, exchcd=1),  # not an ordinary share
    ]
    conn = FakeCrsp(pd.concat(frames, ignore_index=True))
    assert study.read_scan_permnos(conn, 1, 1, "2024-12-31") == [30]
    sql, params = conn.executed[0]
    assert sql is study.PERMNOS_SQL
    assert params[1] == study.SHARE_CODES and params[2] == study.EXCHANGE_CODES
    assert params[3] == study.SCAN_START and params[5] == 1


def test_roster_price_labels_do_not_depend_on_share_or_exchange_codes() -> None:
    assert "shrcd" not in study.ROSTER_RET_SQL and "exchcd" not in study.ROSTER_RET_SQL
    stock = _two_exchange_stock().assign(permno=7)
    conn = FakeCrsp(stock)
    returns = study.read_roster_returns(conn, 1, [7], "2024-12-31")
    assert len(returns) == 40
    roster = pd.DataFrame({"permno": pd.array([7], dtype="Int64"), "exit_date": [DATES[39]]})
    plain = study.label_roster(roster, stock[["permno", "date", "ret"]])
    assert plain.loc[0, "price_label"] == study.label_roster(roster, returns).loc[0, "price_label"]


def _patch_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub every database read except the roster's own file reading."""
    monkeypatch.setattr(study, "require_complete_load", lambda *a, **k: None)
    monkeypatch.setattr(
        study,
        "read_roster_returns",
        lambda *a, **k: pd.DataFrame(
            {
                "permno": pd.Series(dtype="int64"),
                "date": pd.Series(dtype="datetime64[ns]"),
                "ret": pd.Series(dtype="float64"),
            }
        ),
    )
    monkeypatch.setattr(study, "read_scan_permnos", lambda *a, **k: [10])
    monkeypatch.setattr(study, "read_calendar", lambda *a, **k: pd.DatetimeIndex(DATES))
    monkeypatch.setattr(
        study,
        "load_intervals",
        lambda *a, **k: (_intervals(), {n: {"positions": 1} for n in _intervals()}),
    )


def _run(tmp_path: Path, **extra: object) -> dict:
    return study.run_integrity(
        None,
        release_id=1,
        crsp_load=1,
        mcgill_load=2,
        eval_start="2005-01-03",
        eval_end="2024-12-31",
        out_dir=tmp_path,
        repo_root=tmp_path,
        run_date="2026-11-24",
        batches_fn=lambda conn, ids: iter([_stock(10, False)]),
        **extra,
    )


def test_out_folder_does_not_change_where_the_roster_inputs_are_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_scan(monkeypatch)

    def names_none(conn: object, stacked: pd.DataFrame, *a: object) -> pd.DataFrame:
        return stacked.assign(name_source="none")

    monkeypatch.setattr(roster_mod, "attach_names", names_none)
    real_build = roster_mod.build_roster
    seen: list[Path] = []

    def spy(conn: object, study_root: Path, *a: object) -> pd.DataFrame:
        seen.append(Path(study_root))
        return real_build(conn, study_root, *a)

    monkeypatch.setattr(roster_mod, "build_roster", spy)
    out = tmp_path / "unrelated" / "elsewhere"
    _run(out)
    assert seen == [study.STUDY_DIR]
    assert study.STUDY_DIR.name == "dead-stock-data-integrity"
    # The 141 real exits of the first test were read although --out is an unrelated folder.
    assert len(pd.read_csv(out / study.OUTPUT_ROSTER)) == 141


def test_git_provenance_comes_from_git_and_never_hides_a_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    answers = {"rev-parse": "abc123", "status": " M file.py"}
    monkeypatch.setattr(study, "_git", lambda root, *args: answers[args[0]])
    assert study.git_provenance(tmp_path) == {"git_head": "abc123", "git_dirty": True}
    answers["status"] = ""
    assert study.git_provenance(tmp_path) == {"git_head": "abc123", "git_dirty": False}
    monkeypatch.setattr(study, "_git", lambda root, *args: None)
    assert study.git_provenance(tmp_path) == {"git_head": "unknown", "git_dirty": "unknown"}


def test_git_failure_is_none_not_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*a: object, **k: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(study.subprocess, "run", broken)
    assert study._git(tmp_path, "status") is None


def test_payload_records_head_dirty_flag_and_real_code_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_scan(monkeypatch)
    monkeypatch.setattr(study.roster_mod, "build_roster", lambda *a, **k: _tiny_roster())
    monkeypatch.setattr(
        study, "_git", lambda root, *args: {"rev-parse": "deadbeef"}.get(args[0], " M x")
    )
    prov = _run(tmp_path)["provenance"]
    assert prov["git_head"] == "deadbeef" and prov["git_dirty"] is True
    code_dir = Path(study.__file__).resolve().parent
    assert set(prov["code_sha256"]) == {
        "run.py",
        "dead_stock_data_integrity.py",
        "integrity_checks.py",
        "price_path.py",
        "roster.py",
    }
    for name, digest in prov["code_sha256"].items():
        assert digest == hashlib.sha256((code_dir / name).read_bytes()).hexdigest()


def _tiny_roster() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["PERMNO-10"],
            "permno": pd.array([10], dtype="Int64"),
            "exit_date": [DATES[26]],
            "name_source": ["mcgill"],
        }
    )


def test_failed_replacement_keeps_the_previous_file_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / study.OUTPUT_JSON
    dest.write_text('{"old": true}\n')

    def fail(src: object, dst: object) -> None:
        raise OSError("disk full while swapping the file in")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        study.write_json(dest, {"new": True})
    assert dest.read_text() == '{"old": true}\n'
    assert list(tmp_path.glob("*.tmp")) == []
