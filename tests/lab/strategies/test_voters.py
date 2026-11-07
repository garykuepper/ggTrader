"""Voter arrays against the legacy pandas voters (class (ii): equal after burn-in)."""

from __future__ import annotations

import numpy as np
import pytest

from ggTrader.lab.strategies import voters as vt
from tests.lab.strategies.helpers import diff_after_burn_in, make_bars, settled
from tests.oracles.legacy_voters import (
    bb_signals,
    ema_signals,
    macd_signals,
    rsi_signals,
    volume_bb_signals,
)


@pytest.fixture(scope="module", params=[False, True], ids=["clean", "holes"])
def bars(request):
    return make_bars(seed=5, n_bars=900, n_syms=9, holes=request.param)


def _check(new, old_pair, close, label: str) -> None:
    assert diff_after_burn_in(new[0], old_pair[0].to_numpy(), close) == 0, f"{label} entries"
    assert diff_after_burn_in(new[1], old_pair[1].to_numpy(), close) == 0, f"{label} exits"


def test_bb(bars) -> None:
    pairs = [(10, 1.5), (20, 2.0), (20, 2.5), (30, 3.0)]
    got = vt.bb_pairs(bars.close, pairs)
    assert set(got) == set(pairs)
    for p, s in pairs:
        _check(got[(p, s)], bb_signals(bars.close, p, s), bars.close, f"bb{p},{s}")


def test_rsi(bars) -> None:
    triples = [(14, 30, 50), (14, 25, 50), (7, 20, 55), (21, 30, 60)]
    got = vt.rsi_triples(bars.close, triples)
    for n, o, x in triples:
        _check(got[(n, o, x)], rsi_signals(bars.close, n, o, x), bars.close, f"rsi{n},{o},{x}")


def test_ema(bars) -> None:
    pairs = [(5, 20), (20, 50), (50, 200), (10, 50)]
    got = vt.ema_pairs(bars.close, pairs)
    for f, s in pairs:
        _check(got[(f, s)], ema_signals(bars.close, f, s), bars.close, f"ema{f},{s}")


def test_macd(bars) -> None:
    quads = [(12, 26, 9, 20), (12, 26, 9, 10), (8, 21, 9, 10)]
    got = vt.macd_quads(bars.close, quads)
    for f, s, g, w in quads:
        _check(got[(f, s, g, w)], macd_signals(bars.close, f, s, g, w), bars.close, "macd")


def test_vbb(bars) -> None:
    quads = [(20, 2.0, 20, 2.0), (20, 2.5, 20, 1.5), (15, 2.0, 20, 2.5)]
    got = vt.vbb_quads(bars.close, bars.volume, quads)
    for p, s, vp, vm in quads:
        _check(
            got[(p, s, vp, vm)],
            volume_bb_signals(bars.close, bars.volume, p, s, vp, vm),
            bars.close,
            "vbb",
        )


def test_vbb_entries_are_a_subset_of_bb_entries_and_exits_are_identical() -> None:
    """The OD3a double vote: a volume spike can never add an entry the plain voter lacks."""
    b = make_bars(seed=9, n_bars=800, n_syms=8)
    bb = vt.bb_pairs(b.close, [(20, 2.0)])[(20, 2.0)]
    vbb = vt.vbb_quads(b.close, b.volume, [(20, 2.0, 20, 1.5)])[(20, 2.0, 20, 1.5)]
    assert not (vbb[0] & ~bb[0]).any()
    assert vbb[0].sum() < bb[0].sum()
    assert np.array_equal(vbb[1], bb[1])


def test_voter_arrays_computes_each_parameter_tuple_once() -> None:
    b = make_bars(seed=1, n_bars=300, n_syms=3)
    combos = [{"bb_period": 20, "bb_std": 2.0}] * 3 + [{"bb_period": 20, "bb_std": 2.5}]
    arrays = vt.voter_arrays("bb", b, combos)
    assert set(arrays.arrays) == {(20, 2.0), (20, 2.5)}
    assert arrays.keys == ("bb_period", "bb_std")


def test_signals_are_boolean_arrays_of_panel_shape() -> None:
    b = make_bars(seed=2, n_bars=300, n_syms=4)
    for entry, exit_ in vt.bb_pairs(b.close, [(20, 2.0)]).values():
        assert entry.dtype == bool and exit_.dtype == bool and entry.shape == b.close.shape


def test_burn_in_helper_marks_each_symbols_own_start() -> None:
    b = make_bars(seed=3, n_bars=500, n_syms=4, holes=True)
    mask = settled(b.close)
    assert not mask[:200].any(axis=0).any()
    assert mask[352:, 2].all() and not mask[:350, 2].any()  # late listing at bar 150


def test_macd_entries_are_masked_for_the_first_slow_signal_window_bars() -> None:
    """The legacy warm-up mask is a global bar count (slow + signal + window), kept as is."""
    b = make_bars(seed=17, n_bars=140, n_syms=80)
    entry = vt.macd_quads(b.close, [(12, 26, 9, 10)])[(12, 26, 9, 10)][0]
    assert not entry[: 26 + 9 + 10].any()
    assert entry[26 + 9 + 10 :].any()  # not vacuous: entries do occur right after the mask
