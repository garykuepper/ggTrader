"""N-of-M vote helper: thresholds, independent exit, double vote, stacking, validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.strategies.contract import group_columns
from ggTrader.lab.strategies.vote import VoterArrays, threshold, vote

T, S = 12, 3
INDEX = pd.bdate_range("2024-01-01", periods=T, tz="UTC")
SYMS = ["A", "B", "C"]


def _voter(name: str, entry: np.ndarray, exit_: np.ndarray, keys=()) -> VoterArrays:
    return VoterArrays(name=name, keys=keys, arrays={(): (entry, exit_)})


def _rand(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((T, S)) < 0.4


def _cols(combos, levels=("min_agree", "min_agree_exit")) -> pd.MultiIndex:
    return group_columns(combos, list(levels), SYMS)


def test_threshold_none_exit_means_entry_threshold() -> None:
    assert threshold({"min_agree": 3, "min_agree_exit": None}, "min_agree", "min_agree_exit") == (
        3,
        3,
    )
    assert threshold({"min_agree": 3}, "min_agree", "min_agree_exit") == (3, 3)
    assert threshold({"min_agree": 3, "min_agree_exit": 1}, "min_agree", "min_agree_exit") == (3, 1)


def test_entries_need_min_agree_votes_and_exits_min_agree_exit() -> None:
    v = [_voter(f"v{i}", _rand(i), _rand(10 + i)) for i in range(4)]
    combos = [{"min_agree": m, "min_agree_exit": x} for m in (1, 2, 3) for x in (1, 2)]
    ent, ext = vote(v, combos, INDEX, _cols(combos), independent_exit=None)
    for i, c in enumerate(combos):
        sl = slice(i * S, (i + 1) * S)
        want_e = sum(a.arrays[()][0].astype(int) for a in v) >= c["min_agree"]
        want_x = sum(a.arrays[()][1].astype(int) for a in v) >= c["min_agree_exit"]
        assert (ent.to_numpy()[:, sl] == want_e).all()
        assert (ext.to_numpy()[:, sl] == want_x).all()


def test_independent_exit_fires_alone_and_also_counts_as_a_vote() -> None:
    rsi_exit = np.zeros((T, S), bool)
    rsi_exit[3, 0] = True
    other_exit = np.zeros((T, S), bool)
    other_exit[5, 0] = True  # one vote only
    voters = [
        _voter("rsi", np.zeros((T, S), bool), rsi_exit),
        _voter("bb", np.zeros((T, S), bool), other_exit),
    ]
    combos = [{"min_agree": 2, "min_agree_exit": 2}]
    _, ext = vote(voters, combos, INDEX, _cols(combos))
    assert ext.iloc[3, 0] and not ext.iloc[5, 0]  # rsi alone exits, bb alone does not
    _, ext_no = vote(voters, combos, INDEX, _cols(combos), independent_exit=None)
    assert not ext_no.iloc[3, 0]  # without the independent path one vote is below 2


def test_independent_exit_is_ignored_when_that_voter_is_not_active() -> None:
    v = [_voter("bb", _rand(1), _rand(2))]
    combos = [{"min_agree": 1, "min_agree_exit": 1}]
    _, ext = vote(v, combos, INDEX, _cols(combos), independent_exit="rsi")
    assert (ext.to_numpy() == v[0].arrays[()][1]).all()


def test_the_same_array_listed_twice_votes_twice() -> None:
    entry = np.zeros((T, S), bool)
    entry[4, 1] = True
    v = [_voter("bb", entry, entry), _voter("vbb", entry, entry)]  # the OD3a double vote
    combos = [{"min_agree": 2, "min_agree_exit": 2}]
    ent, ext = vote(v, combos, INDEX, _cols(combos), independent_exit=None)
    assert ent.iloc[4, 1] and ext.iloc[4, 1]


def test_param_keys_select_different_arrays_per_combination() -> None:
    a, b = _rand(1), _rand(2)
    voter = VoterArrays("bb", ("p",), {(1,): (a, a), (2,): (b, b)})
    combos = [
        {"p": 1, "min_agree": 1, "min_agree_exit": 1},
        {"p": 2, "min_agree": 1, "min_agree_exit": 1},
    ]
    ent, _ = vote([voter], combos, INDEX, _cols(combos, ("p",)), independent_exit=None)
    assert (ent.to_numpy()[:, :S] == a).all() and (ent.to_numpy()[:, S:] == b).all()


def test_vote_sum_cache_does_not_leak_between_combinations() -> None:
    voter = VoterArrays("bb", ("p",), {(1,): (_rand(1), _rand(2)), (2,): (_rand(3), _rand(4))})
    combos = [
        {"p": 1, "min_agree": 1, "min_agree_exit": 1},
        {"p": 2, "min_agree": 1, "min_agree_exit": 1},
        {"p": 1, "min_agree": 1, "min_agree_exit": 1},
    ]
    ent, _ = vote([voter], combos, INDEX, _cols(combos, ("p",)), independent_exit=None)
    values = ent.to_numpy()
    assert (values[:, :S] == values[:, 2 * S :]).all() and not (
        values[:, :S] == values[:, S : 2 * S]
    ).all()


def test_output_shape_dtype_and_columns() -> None:
    combos = [{"min_agree": 1, "min_agree_exit": 1}, {"min_agree": 2, "min_agree_exit": 1}]
    cols = _cols(combos)
    ent, ext = vote([_voter("bb", _rand(1), _rand(2))], combos, INDEX, cols, independent_exit=None)
    assert ent.shape == (T, 2 * S) == ext.shape
    assert ent.dtypes.eq(bool).all() and ext.dtypes.eq(bool).all()
    assert ent.columns.equals(cols) and ent.index.equals(INDEX)


def test_validation() -> None:
    combos = [{"min_agree": 1, "min_agree_exit": 1}]
    with pytest.raises(ValueError, match="at least one voter"):
        vote([], combos, INDEX, _cols(combos))
    with pytest.raises(ValueError, match="expected"):
        vote([_voter("bb", _rand(1), _rand(2))], combos * 2, INDEX, _cols(combos))
