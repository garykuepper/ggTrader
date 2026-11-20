"""``FoldRecord`` holds the 28 fold values and gives back the legacy dict unchanged."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.wfo import run
from ggTrader.lab.wfo.fold_record import FOLD_KEYS, FoldRecord

EXPECTED_KEYS = (
    "fold_num",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "winner_combo",
    "winner_params",
    "train_score",
    "oos_score",
    "ndh_passed",
    "dsr_passed",
    "gates_passed",
    "ndh_density",
    "dsr_value",
    "wfe",
    "is_sharpe",
    "oos_sharpe",
    "halted",
    "halted_entering",
    "used_anchor",
    "anchor_combo",
    "anchor_params",
    "anchor_reason",
    "deployed_combo",
    "halt_cleared",
    "winner_trades_in_window",
    "winner_trades_whole_simulation",
    "od12_ndh_flip",
)


def _record() -> FoldRecord:
    return FoldRecord(
        fold_num=1,
        train_start=pd.Timestamp("2021-01-01"),
        train_end=pd.Timestamp("2022-01-01"),
        test_start=pd.Timestamp("2022-01-02"),
        test_end=pd.Timestamp("2022-02-01"),
        winner_combo="a",
        winner_params={"delay": 3},
        train_score=np.float64(1.25),
        oos_score=0.5,
        ndh_passed=True,
        dsr_passed=False,
        gates_passed=False,
        ndh_density=0.9,
        dsr_value=0.1,
        wfe=None,
        is_sharpe=1.0,
        oos_sharpe=float("nan"),
        halted=False,
        halted_entering=False,
        used_anchor=True,
        anchor_combo="b",
        anchor_params={"delay": 50},
        anchor_reason=None,
        deployed_combo="b",
        halt_cleared=False,
        winner_trades_in_window=2,
        winner_trades_whole_simulation=7,
        od12_ndh_flip=False,
    )


def test_fold_keys_are_the_28_in_todays_order() -> None:
    assert FOLD_KEYS == EXPECTED_KEYS
    assert len(FOLD_KEYS) == 28


def test_fields_equal_fold_keys() -> None:
    assert tuple(f.name for f in dataclasses.fields(FoldRecord)) == FOLD_KEYS


def test_as_dict_key_order() -> None:
    assert list(_record().as_dict()) == list(EXPECTED_KEYS)


def test_as_dict_returns_the_same_objects() -> None:
    rec = _record()
    d = rec.as_dict()
    assert d["winner_params"] is rec.winner_params
    assert d["anchor_params"] is rec.anchor_params
    assert d["train_start"] is rec.train_start
    assert d["test_end"] is rec.test_end
    assert d["train_score"] is rec.train_score
    assert isinstance(d["train_score"], np.float64)
    assert d["wfe"] is None


def test_as_dict_returns_a_new_dict_each_call() -> None:
    rec = _record()
    first, second = rec.as_dict(), rec.as_dict()
    assert first is not second
    first["fold_num"] = 99
    assert rec.as_dict()["fold_num"] == 1


def test_record_is_frozen() -> None:
    rec = _record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.fold_num = 2  # type: ignore[misc]


def test_run_module_exposes_fold_record() -> None:
    assert run.FoldRecord is FoldRecord
