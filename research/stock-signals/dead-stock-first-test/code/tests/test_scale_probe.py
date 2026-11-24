"""Tests for the scale probe's arithmetic and argument handling (no database)."""

from __future__ import annotations

import pandas as pd
import pytest
import scale_probe

pytestmark = pytest.mark.study

STATUS = "Name:\tpython\nVmPeak:\t  9000 kB\nVmHWM:\t  2048000 kB\nVmRSS:\t  1000 kB\n"


def test_peak_memory_reader_returns_mb(monkeypatch):
    monkeypatch.setattr(scale_probe, "read_status_text", lambda: STATUS)
    assert scale_probe.peak_memory_mb() == pytest.approx(2000.0)


def test_peak_memory_reader_without_proc_returns_zero(monkeypatch):
    monkeypatch.setattr(scale_probe, "read_status_text", lambda: "")
    assert scale_probe.peak_memory_mb() == 0.0


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
