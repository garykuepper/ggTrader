"""End-to-end tests for the dead-stock-momentum-deaths driver on synthetic inputs."""

from __future__ import annotations

import json
from pathlib import Path

import dm_replay
import numpy as np
import pandas as pd
import pytest

from ggTrader.utils.study import load_driver, study_dir

pytestmark = pytest.mark.study

run = load_driver(study_dir(__file__) / "code" / "run.py")

WINDOW = "holdout"
DAYS = pd.bdate_range("2018-06-01", "2024-12-31", tz="UTC")
SYMBOLS = [f"S{i:02d}" for i in range(16)]


def inputs_for(_args) -> dict:
    rng = np.random.default_rng(5)
    prices = 40.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, (len(DAYS), len(SYMBOLS))), axis=0))
    close = pd.DataFrame(prices, index=DAYS, columns=SYMBOLS)
    spy = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, len(DAYS)))), index=DAYS)
    # S03 dies mid-window; S04 is a successor; S05 ends on CRSP_END (censored).
    close.loc["2022-03-01":, "S03"] = np.nan
    close.loc["2023-05-01":, "S04"] = np.nan
    spans = pd.DataFrame(
        {
            "symbol": ["S03", "S04", "S05"],
            "last_date": pd.to_datetime(["2022-02-28", "2023-04-28", "2024-12-31"]),
            "has_successor": [False, True, False],
            "total_loss": [False, False, False],
        }
    )
    eligible = pd.DataFrame(True, index=DAYS, columns=SYMBOLS)
    return {
        "close": close,
        "spy_close": spy,
        "eligible": eligible,
        "span_rows": spans,
        "release_spans": spans,
        "data_provenance": {
            "candidates": {
                "data_release": {"release_id": 1, "release_sha256": "abc123"},
                "bars_sha256": ["hash_batch_0"],
                "n_batches": 1,
                "drift": False,
                "drift_symbols": [],
            },
            "spy": {
                "data_release": {"release_id": 1, "release_sha256": "abc123"},
                "bars_sha256": ["hash_spy"],
                "n_batches": 1,
                "drift": False,
                "drift_symbols": [],
            },
            "n_candidate_columns": len(SYMBOLS),
        },
    }


def write_saved_first_test(tmp_path: Path, held_at_death: int = 1) -> Path:
    path = tmp_path / "first_test.json"
    path.write_text(
        json.dumps({"result": {"sanity": {"held_at_death": held_at_death}}}),
        encoding="utf-8",
    )
    return path


def write_random_baseline(tmp_path: Path, median: float = 8.0) -> Path:
    path = tmp_path / "random_baseline.json"
    path.write_text(
        json.dumps({"summary": {"R": {"loss1": {"n_dead_exits": {"p50": median}}}}}),
        encoding="utf-8",
    )
    return path


def args_for(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--window",
        WINDOW,
        "--release-id",
        "1",
        "--crsp-load",
        "1",
        "--out",
        str(tmp_path / "out.json"),
        *extra,
    ]


def test_full_run_writes_measures_provenance_and_hashes(tmp_path: Path) -> None:
    saved = write_saved_first_test(tmp_path, held_at_death=1)
    rb = write_random_baseline(tmp_path, median=8.0)
    code = run.main(
        args_for(tmp_path),
        loader=inputs_for,
        saved_paths={WINDOW: saved},
        random_baseline_paths={WINDOW: rb},
        git_dirty=lambda: False,
    )
    assert code == 0
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert out["study"] == "dead-stock-momentum-deaths"
    assert out["window"] == WINDOW
    assert out["release_id"] == 1
    assert out["crsp_load"] == 1
    assert out["check_first"]["ok"] is True
    assert out["provenance"]["release_id"] == 1
    assert out["provenance"]["crsp_load"] == 1
    assert out["provenance"]["window"] == WINDOW
    assert out["provenance"]["data_release"] == {"release_id": 1, "release_sha256": "abc123"}
    assert out["provenance"]["bars_sha256"] == ["hash_batch_0"]
    assert out["data_release"] == {"release_id": 1, "release_sha256": "abc123"}
    assert out["bars_sha256"] == ["hash_batch_0"]

    measures = out["measures"]
    for key in (
        "A1_totals",
        "A2_by_band",
        "A3_rate_ratio",
        "A4_expected_vs_actual",
        "A5_by_year",
        "B_dying_profile",
        "C_one_day_jumps",
        "D_first_test_check",
    ):
        assert key in measures
    # Only S03 is a real death (S04 is successor, S05 is censored file-end), and with 16 stocks
    # all 16 are in top 50, so top_50_deaths == 1.
    assert measures["A1_totals"]["deaths"] == 1
    assert measures["A4_expected_vs_actual"]["actual_top_50_deaths"] == 1
    assert measures["A4_expected_vs_actual"]["random_baseline_fresh_draw_median"] == 8.0
    assert measures["D_first_test_check"]["first_test_held_at_death"] == 1
    assert measures["D_first_test_check"]["difference"] == 0
    assert measures["D_first_test_check"]["exceeds_10pct_flag"] is False


def test_plan_mismatch_exits_3_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = write_saved_first_test(tmp_path)
    rb = write_random_baseline(tmp_path)
    monkeypatch.setattr(run.dm_replay, "plan_trades", lambda *a, **k: [])
    code = run.main(
        args_for(tmp_path),
        loader=inputs_for,
        saved_paths={WINDOW: saved},
        random_baseline_paths={WINDOW: rb},
        git_dirty=lambda: False,
    )
    assert code == run.EXIT_CHECK_FAILED
    assert not (tmp_path / "out.json").exists()
    # Sanity: dm_replay itself is untouched after monkeypatch.
    assert callable(dm_replay.plan_trades)


def test_refusals_wrong_pins_dirty_tree_existing_output_and_drift(tmp_path: Path) -> None:
    saved = write_saved_first_test(tmp_path)
    rb = write_random_baseline(tmp_path)

    # Wrong release id.
    assert (
        run.main(
            [
                "--window",
                WINDOW,
                "--release-id",
                "2",
                "--crsp-load",
                "1",
                "--out",
                str(tmp_path / "bad_rel.json"),
            ],
            loader=inputs_for,
            saved_paths={WINDOW: saved},
            random_baseline_paths={WINDOW: rb},
            git_dirty=lambda: False,
        )
        == run.EXIT_REFUSED
    )
    # Wrong CRSP load.
    assert (
        run.main(
            [
                "--window",
                WINDOW,
                "--release-id",
                "1",
                "--crsp-load",
                "2",
                "--out",
                str(tmp_path / "bad_load.json"),
            ],
            loader=inputs_for,
            saved_paths={WINDOW: saved},
            random_baseline_paths={WINDOW: rb},
            git_dirty=lambda: False,
        )
        == run.EXIT_REFUSED
    )
    # Dirty tracked tree.
    assert (
        run.main(
            args_for(tmp_path),
            loader=inputs_for,
            saved_paths={WINDOW: saved},
            random_baseline_paths={WINDOW: rb},
            git_dirty=lambda: True,
        )
        == run.EXIT_REFUSED
    )

    # Drifted inputs.
    def drifted(args):
        data = inputs_for(args)
        data["data_provenance"]["candidates"]["drift"] = True
        data["data_provenance"]["candidates"]["drift_symbols"] = ["S01"]
        return data

    assert (
        run.main(
            args_for(tmp_path),
            loader=drifted,
            saved_paths={WINDOW: saved},
            random_baseline_paths={WINDOW: rb},
            git_dirty=lambda: False,
        )
        == run.EXIT_REFUSED
    )
    # Existing output file.
    (tmp_path / "out.json").write_text("{}", encoding="utf-8")
    assert (
        run.main(
            args_for(tmp_path),
            loader=inputs_for,
            saved_paths={WINDOW: saved},
            random_baseline_paths={WINDOW: rb},
            git_dirty=lambda: False,
        )
        == run.EXIT_REFUSED
    )


def test_registered_windows_and_help() -> None:
    assert run.momentum_inputs.WINDOWS == {
        "early": ("2000-01-03", "2004-12-31"),
        "dev": ("2005-01-03", "2019-12-31"),
        "holdout": ("2020-01-02", "2024-12-31"),
    }
    parser = run.build_parser()
    assert "--window" in parser.format_help()
    assert "--release-id" in parser.format_help()
    assert "--crsp-load" in parser.format_help()
