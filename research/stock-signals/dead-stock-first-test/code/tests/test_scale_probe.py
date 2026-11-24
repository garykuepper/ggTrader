"""Tests for the scale probe's arithmetic and argument handling (no database)."""

from __future__ import annotations

import json

import pandas as pd
import pytest
import scale_probe

pytestmark = pytest.mark.study

STATUS = "Name:\tpython\nVmPeak:\t  9000 kB\nVmHWM:\t  2048000 kB\nVmRSS:\t  1000 kB\n"


def test_peak_memory_reader_returns_mb(monkeypatch):
    monkeypatch.setattr(scale_probe, "read_status_text", lambda: STATUS)
    assert scale_probe.peak_memory_mb() == pytest.approx(2000.0)


def test_peak_memory_unavailable_is_an_error_not_zero(monkeypatch):
    monkeypatch.setattr(scale_probe, "read_status_text", lambda: "")
    with pytest.raises(scale_probe.PeakMemoryUnavailable):
        scale_probe.peak_memory_mb()
    assert scale_probe.safe_peak_mb()[0] is None
    monkeypatch.setattr(scale_probe, "read_status_text", lambda: "Name:\tpython\n")
    with pytest.raises(scale_probe.PeakMemoryUnavailable):
        scale_probe.peak_memory_mb()


def test_missing_peak_is_never_ok():
    record = scale_probe.build_record(
        n_columns=3,
        n_dates=5,
        seconds=1.0,
        peak_mb=None,
        release_id=1,
        provenance={},
        measurement_error="no VmHWM",
    )
    assert record["ok"] is False
    assert record["peak_memory_mb"] is None
    assert record["measurement_error"] == "no VmHWM"


def test_arg_parsing_n_all_and_modes():
    args = scale_probe.build_parser().parse_args(
        ["--n", "all", "--mode", "one_call", "--universe", "B", "--out", "x.json"]
    )
    assert args.n == "all" and args.mode == "one_call" and args.universe == "B"
    assert args.release_id == 1 and args.crsp_load == 1
    assert (args.start, args.end) == ("2004-01-01", "2019-12-31")
    args = scale_probe.build_parser().parse_args(["--n", "1500", "--mode", "batched"])
    assert args.n == 1500 and args.mode == "batched" and args.universe == "A"
    with pytest.raises(SystemExit):
        scale_probe.build_parser().parse_args(["--n", "0", "--mode", "batched"])
    with pytest.raises(SystemExit):
        scale_probe.build_parser().parse_args(["--n", "5", "--mode", "weird"])
    with pytest.raises(SystemExit):
        scale_probe.build_parser().parse_args(["--n", "many", "--mode", "batched"])


def test_candidate_columns_are_those_ever_eligible():
    fields = pd.DataFrame(
        {
            "permno": [1, 2, 3],
            "month_end": pd.to_datetime(["2019-12-31"] * 3),
            "shrcd": [11, 11, 73],  # permno 3 is not common stock
            "exchcd": [1, 1, 1],
            "abs_prc": [20.0, 2.0, 20.0],  # permno 2 is under the $5 floor
            "cap": [1e9, 1e9, 1e9],
        }
    )
    spans = pd.DataFrame(
        {
            "symbol": ["ZZZ", "AAA", "BBB", "CCC", "DDD"],
            "permno": [1, 1, 2, 3, 4],
            "kind": ["placeholder", "yahoo", "placeholder", "placeholder", "placeholder"],
            "first_date": pd.to_datetime(["1990-01-01"] * 5),
            "last_date": pd.to_datetime(["2030-01-01"] * 5),
            "possible_double": [False] * 5,
        }
    )
    got = scale_probe.candidate_columns(fields, spans, "A", "2020-01-01", "2020-01-31")
    # AAA is a yahoo twin of ZZZ (same company): the placeholder wins, so only ZZZ stays.
    assert got == ["ZZZ"]


def test_select_first_n_sorted():
    cols = ["B", "A", "D", "C"]
    assert scale_probe.first_n(cols, 2) == ["A", "B"]
    assert scale_probe.first_n(cols, "all") == ["A", "B", "C", "D"]
    assert scale_probe.first_n(cols, 99) == ["A", "B", "C", "D"]


def test_record_marks_ok_by_memory_bar():
    fit = scale_probe.build_record(
        n_columns=3, n_dates=5, seconds=1.0, peak_mb=2500.0, release_id=1, provenance={}
    )
    over = scale_probe.build_record(
        n_columns=3, n_dates=5, seconds=1.0, peak_mb=2500.1, release_id=1, provenance={}
    )
    assert fit["ok"] is True and over["ok"] is False
    assert "date-bounded batches" in fit["decision_rule"]


def _fake_close():
    return pd.DataFrame(
        {"A": [1.0, 2.0], "B": [3.0, 4.0]}, index=pd.bdate_range("2020-01-02", periods=2)
    )


def test_setup_failure_replaces_old_finished_file(monkeypatch, tmp_path):
    out = tmp_path / "probe.json"
    out.write_text(json.dumps({"status": "finished", "ok": True}))
    seen = {}

    def boom(args):
        seen["during_setup"] = json.loads(out.read_text())
        raise RuntimeError("database gone")

    monkeypatch.setattr(scale_probe, "pick_columns", boom)
    with pytest.raises(RuntimeError, match="database gone"):
        scale_probe.main(["--n", "5", "--mode", "batched", "--out", str(out)])
    assert seen["during_setup"]["status"] == "started"
    final = json.loads(out.read_text())
    assert final["status"] == "failed"
    assert final["error_type"] == "RuntimeError"
    assert final["error_message"] == "database gone"
    assert final["crsp_load"] == 1 and final["release_id"] == 1


def test_main_writes_finished_record_with_crsp_load(monkeypatch, tmp_path):
    out = tmp_path / "probe.json"
    monkeypatch.setattr(scale_probe, "pick_columns", lambda args: ["A", "B"])
    monkeypatch.setattr(scale_probe, "read_status_text", lambda: STATUS)

    def fake_read(columns, start, end, *, release_id, batch, provenance_out):
        provenance_out["data_release"] = {"release_id": release_id}
        return _fake_close(), pd.DataFrame()

    monkeypatch.setattr(scale_probe.bars_reader, "read_close_batched", fake_read)
    assert (
        scale_probe.main(
            ["--n", "all", "--mode", "one_call", "--crsp-load", "4", "--out", str(out)]
        )
        == 0
    )
    record = json.loads(out.read_text())
    assert record["status"] == "finished" and record["ok"] is True
    assert record["n_columns"] == 2 and record["n_dates"] == 2
    assert record["crsp_load"] == 4 and record["release_id"] == 1
    assert record["peak_memory_mb"] == pytest.approx(2000.0)
    assert record["setup_peak_memory_mb"] == pytest.approx(2000.0)
    assert record["provenance"]["data_release"] == {"release_id": 1}


def test_main_without_peak_data_is_not_ok(monkeypatch, tmp_path):
    out = tmp_path / "probe.json"
    monkeypatch.setattr(scale_probe, "pick_columns", lambda args: ["A", "B"])
    monkeypatch.setattr(scale_probe, "read_status_text", lambda: "")
    monkeypatch.setattr(
        scale_probe.bars_reader,
        "read_close_batched",
        lambda *a, **k: (_fake_close(), pd.DataFrame()),
    )
    scale_probe.main(["--n", "2", "--mode", "batched", "--out", str(out)])
    record = json.loads(out.read_text())
    assert record["status"] == "finished"
    assert record["ok"] is False and record["peak_memory_mb"] is None
    assert "VmHWM" in record["measurement_error"]
