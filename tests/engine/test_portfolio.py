"""L1/L2: run_signals / run_weights reproduce the legacy lab/simulate.py curves bit for bit.

The legacy ``simulate_signals`` / ``simulate_weights`` were deleted in P5c. Their outputs on the
seeded fixtures of this file are frozen in ``golden/legacy_simulate.npz`` (written by
``research/programme/vectorbt-engine/p5c-evidence/capture_legacy_simulate.py`` while the legacy
module still existed), and the comparisons below read that file.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ggTrader.engine import config
from ggTrader.engine import portfolio as pfm

from .panels import make_panel

GOLDEN = np.load(Path(__file__).parent / "golden" / "legacy_simulate.npz")

BASE = {"START_CASH": 10000.0, "FEES": 0.0, "SLIPPAGE": 0.0005, "FREQ": "1D"}
STOCK = {**BASE, "SIGNAL_POSITION_SIZE": 0.03}


def _signals(panel: dict[str, pd.DataFrame], seed: int, rate: float = 0.04) -> tuple:
    rng = np.random.default_rng(seed)
    shape = panel["close"].shape
    entries = pd.DataFrame(
        rng.random(shape) < rate, index=panel["close"].index, columns=panel["close"].columns
    )
    exits = pd.DataFrame(rng.random(shape) < rate, index=entries.index, columns=entries.columns)
    return entries, exits


def _two_strategy_inputs(holes: bool = True):
    panel = make_panel(seed=3, n_bars=500, n_syms=6, holes=holes)
    names = {"alpha": ["S00", "S01", "S02"], "beta": ["S03", "S04", "S05"]}
    entries, exits, closes, groups = [], [], [], []
    for i, (name, syms) in enumerate(names.items()):
        e, x = _signals(panel, seed=10 + i)
        cols = pd.MultiIndex.from_product([[name], syms], names=["strategy", "symbol"])
        entries.append(e[syms].set_axis(cols, axis=1))
        exits.append(x[syms].set_axis(cols, axis=1))
        closes.append(panel["close"][syms].set_axis(cols, axis=1))
        groups += [name] * len(syms)
    return (
        panel,
        None,
        pd.concat(entries, axis=1),
        pd.concat(exits, axis=1),
        pd.concat(closes, axis=1),
        groups,
    )


EXTRAS = {
    "plain": {},
    "trail": {"ts_stop": 0.05},
    "take_profit": {"tp_stop": 0.08},
    "trail_and_tp": {"ts_stop": 0.05, "tp_stop": 0.08},
    "fees": {"FEES": 0.001},
}


@pytest.mark.parametrize("key", list(EXTRAS))
def test_run_signals_equity_is_bit_identical_to_legacy(engine: str, key: str) -> None:
    _, _, entries, exits, close, groups = _two_strategy_inputs()
    cfg = {**STOCK, **EXTRAS[key]}
    result = pfm.run_signals(entries, exits, close, cfg, groups=groups)
    assert result.value.columns.tolist() == ["alpha", "beta"]
    assert np.array_equal(result.value.to_numpy(), GOLDEN[f"signals_{key}_value"])
    assert result.engine == engine
    assert [result.n_trades["alpha"], result.n_trades["beta"]] == list(
        GOLDEN[f"signals_{key}_n_trades"]
    )


def test_run_signals_returns_match_the_legacy_definition(engine: str) -> None:
    _, _, entries, exits, close, groups = _two_strategy_inputs()
    result = pfm.run_signals(entries, exits, close, STOCK, groups=groups)
    assert np.array_equal(result.returns.to_numpy(), GOLDEN["signals_returns"])
    assert (result.returns.iloc[0] == 0.0).all()


def test_run_signals_with_conviction_sizes_matches_legacy(engine: str) -> None:
    panel = make_panel(seed=4, n_bars=400, n_syms=4)
    e, x = _signals(panel, seed=21)
    sizes = pd.DataFrame(
        np.random.default_rng(1).uniform(0.01, 0.05, e.shape), index=e.index, columns=e.columns
    )
    cols = pd.MultiIndex.from_product([["s"], e.columns], names=["strategy", "symbol"])
    result = pfm.run_signals(
        e.set_axis(cols, axis=1),
        x.set_axis(cols, axis=1),
        panel["close"].set_axis(cols, axis=1),
        STOCK,
        groups=["s"] * 4,
        size=sizes.set_axis(cols, axis=1),
    )
    assert np.array_equal(result.value.to_numpy(), GOLDEN["conviction_value"])


def test_run_signals_atr_stop_matches_legacy_once_both_atrs_are_valid(engine: str) -> None:
    """Legacy ATR starts one bar later (NaN first true range), so an entry on bar 13 gets
    a stop from vectorbt and none from legacy. After the burn-in the curves agree."""
    panel = make_panel(seed=6, n_bars=500, n_syms=4)
    e, x = _signals(panel, seed=31)
    e.iloc[:30] = False
    cfg = {**STOCK, "atr_mult": 2.0, "atr_period": 14}
    cols = pd.MultiIndex.from_product([["s"], e.columns], names=["strategy", "symbol"])
    result = pfm.run_signals(
        e.set_axis(cols, axis=1),
        x.set_axis(cols, axis=1),
        panel["close"].set_axis(cols, axis=1),
        cfg,
        groups=["s"] * 4,
        high=panel["high"].set_axis(cols, axis=1),
        low=panel["low"].set_axis(cols, axis=1),
    )
    np.testing.assert_allclose(result.value.to_numpy(), GOLDEN["atr_value"], rtol=1e-9)


def test_atr_stop_differs_from_legacy_only_at_documented_structural_cells() -> None:
    """Review MINOR-3: with interior gaps and late listings the ATR stop is NOT identical to
    the legacy ``compute_atr_stop`` everywhere. The differences are single cells at (a) bar
    ``atr_period - 1`` (legacy's first true range is NaN, so its ATR starts one bar later),
    (b) the bar where the ATR window first excludes an interior gap (vectorbt's true range
    skips a NaN previous close, legacy propagates it), (c) a late listing's first valid ATR.
    Pinned on a seeded panel so a vectorbt change that moves these cells is attributed."""
    panel = make_panel(seed=6, n_bars=300, n_syms=5, holes=True)
    high, low, close = panel["high"], panel["low"], panel["close"]
    new = pfm.atr_stop_fraction(high, low, close, 14, 2.0).to_numpy()
    with np.errstate(invalid="ignore"):
        old = GOLDEN["atr_stop_ffilled"]
        gap = np.abs(new - old)
    both = np.isfinite(new) & np.isfinite(old)
    differs = (np.isfinite(new) != np.isfinite(old)) | (both & (gap > 1e-9))
    cells = {(int(bar), close.columns[col]) for bar, col in zip(*np.nonzero(differs), strict=True)}
    assert cells == {(13, "S00"), (13, "S01"), (13, "S03"), (13, "S04"), (117, "S01"), (163, "S02")}
    assert gap[both & ~differs].max() < 1e-9  # everywhere else the stops agree


def test_atr_stop_requires_high_and_low() -> None:
    _, _, entries, exits, close, groups = _two_strategy_inputs(holes=False)
    with pytest.raises(ValueError, match="high and low"):
        pfm.run_signals(entries, exits, close, {**STOCK, "atr_mult": 2.0}, groups=groups)


def test_run_weights_equity_is_bit_identical_to_legacy(engine: str) -> None:
    panel = make_panel(seed=8, n_bars=400, n_syms=6)
    rng = np.random.default_rng(2)
    targets = {}
    for name, syms in {"w1": ["S00", "S01", "S02"], "w2": ["S03", "S04", "S05"]}.items():
        w = pd.DataFrame(np.nan, index=panel["close"].index, columns=syms)
        for d in range(20, 400, 21):
            w.iloc[d] = rng.dirichlet(np.ones(3)) * 0.9
        targets[name] = w
    cols = [
        pd.MultiIndex.from_product([[n], t.columns], names=["strategy", "symbol"])
        for n, t in targets.items()
    ]
    weights = pd.concat(
        [t.set_axis(c, axis=1) for t, c in zip(targets.values(), cols, strict=True)], axis=1
    )
    close = pd.concat(
        [
            panel["close"][t.columns].set_axis(c, axis=1)
            for t, c in zip(targets.values(), cols, strict=True)
        ],
        axis=1,
    )
    result = pfm.run_weights(weights, close, BASE, groups=["w1"] * 3 + ["w2"] * 3)
    assert np.array_equal(result.value.to_numpy(), GOLDEN["weights_value"])
    assert [result.n_trades["w1"], result.n_trades["w2"]] == list(GOLDEN["weights_n_trades"])


def test_run_weights_empty_group_stays_flat_like_legacy() -> None:
    panel = make_panel(seed=8, n_bars=120, n_syms=3)
    w = pd.DataFrame(np.nan, index=panel["close"].index, columns=panel["close"].columns)
    w.iloc[10] = 0.3
    result = pfm.run_weights(
        w, panel["close"], BASE, groups=["live"] * 3, all_groups=["live", "empty"]
    )
    assert result.value.columns.tolist() == ["live", "empty"]
    assert np.array_equal(result.value.to_numpy(), GOLDEN["weights_empty_value"])
    assert (result.value["empty"] == BASE["START_CASH"]).all()
    assert result.n_trades["empty"] == 0


def test_chunking_does_not_change_a_single_number() -> None:
    panel, _, entries, exits, close, groups = _two_strategy_inputs()
    cfg = {**STOCK, "ts_stop": 0.05}
    whole = pfm.run_signals(entries, exits, close, cfg, groups=groups)
    tiny = pfm.estimate_peak_bytes(len(close), 3, 1)  # exactly one group per chunk
    chunked = pfm.run_signals(entries, exits, close, cfg, groups=groups, max_bytes=tiny)
    assert whole.n_chunks == 1 and chunked.n_chunks == 2
    assert np.array_equal(whole.value.to_numpy(), chunked.value.to_numpy())
    pd.testing.assert_frame_equal(whole.trades, chunked.trades)
    assert whole.n_trades.to_dict() == chunked.n_trades.to_dict()


def _many_groups(n_groups: int = 6, per_group: int = 2, n_bars: int = 4000):
    panel = make_panel(seed=21, n_bars=n_bars, n_syms=n_groups * per_group)
    entries, exits = _signals(panel, seed=22, rate=0.01)
    groups = [f"g{i // per_group}" for i in range(n_groups * per_group)]
    return panel["close"], entries, exits, groups


def test_each_chunk_is_released_before_the_next_one_runs(monkeypatch) -> None:
    """P4 finding: a finished chunk must not stay alive (portfolio, value frame) while the
    next one is simulated, or the memory budget bounds nothing."""
    import gc
    import weakref

    import vectorbt as vbt

    close, entries, exits, groups = _many_groups()
    budget = pfm.estimate_peak_bytes(len(close), 2, 1)  # exactly one group per chunk
    alive: list[weakref.ref] = []
    seen_alive: list[int] = []
    original = vbt.Portfolio.from_signals

    def spy(*args, **kwargs):
        gc.collect()
        seen_alive.append(sum(ref() is not None for ref in alive))
        pf = original(*args, **kwargs)
        alive.append(weakref.ref(pf))
        return pf

    monkeypatch.setattr(vbt.Portfolio, "from_signals", staticmethod(spy))
    out = pfm.run_signals(entries, exits, close, STOCK, groups=groups, max_bytes=budget)
    assert out.n_chunks == 6
    assert seen_alive == [0] * 6, "a finished chunk's Portfolio was still alive"


def test_six_chunks_equal_one_call_bit_for_bit() -> None:
    close, entries, exits, groups = _many_groups(n_bars=500)
    whole = pfm.run_signals(entries, exits, close, STOCK, groups=groups)
    chunked = pfm.run_signals(
        entries, exits, close, STOCK, groups=groups, max_bytes=pfm.estimate_peak_bytes(500, 2, 1)
    )
    assert (whole.n_chunks, chunked.n_chunks) == (1, 6)
    assert np.array_equal(whole.value.to_numpy(), chunked.value.to_numpy())
    pd.testing.assert_frame_equal(whole.trades, chunked.trades)


def test_a_group_larger_than_the_budget_still_runs_with_a_warning() -> None:
    panel, _, entries, exits, close, groups = _two_strategy_inputs()
    with pytest.warns(ResourceWarning, match="cannot be split"):
        out = pfm.run_signals(entries, exits, close, STOCK, groups=groups, max_bytes=1)
    assert out.n_chunks == 2


@pytest.mark.parametrize("seed", range(5))
def test_planned_chunks_partition_the_groups_and_never_split_one(seed: int) -> None:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 40, size=int(rng.integers(3, 30)))
    codes = np.repeat(np.arange(len(sizes)), sizes)
    budget = pfm.estimate_peak_bytes(500, int(rng.integers(20, 120)), 2)
    oversized = pfm.estimate_peak_bytes(500, int(sizes.max()), 2) > budget
    with pytest.warns(ResourceWarning) if oversized else contextlib.nullcontext():
        chunks = pfm._plan_chunks(codes, 500, 2, budget)
    flat = [c for chunk in chunks for c in chunk]
    assert sorted(flat) == list(range(len(sizes)))  # every group exactly once, none split
    for chunk in chunks:
        cols = int(sum(sizes[c] for c in chunk))
        assert len(chunk) == 1 or pfm.estimate_peak_bytes(500, cols, 2) <= budget


@pytest.mark.parametrize("key", ["vol_target", "vol_lookback", "vol_cap"])
def test_legacy_vol_targeting_keys_raise_instead_of_being_ignored(key: str) -> None:
    panel, _, entries, exits, close, groups = _two_strategy_inputs()
    with pytest.raises(ValueError, match=key):
        pfm.run_signals(entries, exits, close, {**STOCK, key: 0.1}, groups=groups)
    with pytest.raises(ValueError, match=key):
        pfm.run_weights(entries * 0.0, close, {**BASE, key: 0.1}, groups=groups)
    # a key present but None is the legacy "off" and stays allowed
    pfm.run_signals(entries, exits, close, {**STOCK, key: None}, groups=groups)


@pytest.mark.parametrize("key", ["ts_stop", "atr_mult", "tp_stop", "stop_entry_price"])
def test_run_weights_rejects_stop_keys_it_cannot_apply(key: str) -> None:
    panel, _, entries, _, close, groups = _two_strategy_inputs()
    with pytest.raises(ValueError, match=key):
        pfm.run_weights(entries * 0.0, close, {**BASE, key: 0.05}, groups=groups)


def test_peak_estimate_formula() -> None:
    assert pfm.estimate_peak_bytes(1700, 24000, 2) == 1700 * 24000 * 8 * (2 + pfm.WORKING_ARRAYS)


def test_mask_blocks_entries_outside_the_universe(engine: str) -> None:
    _, _, entries, exits, close, groups = _two_strategy_inputs(holes=False)
    mask = pd.DataFrame(False, index=entries.index, columns=entries.columns)
    mask.iloc[200:, :4] = True
    out = pfm.run_signals(entries, exits, close, STOCK, groups=groups, mask=mask)
    assert len(out.trades) > 0
    allowed = mask.to_numpy()
    assert all(allowed[row.entry_idx, row.col] for row in out.trades.itertuples())


def test_time_stop_exits_exactly_n_bars_after_entry(engine: str) -> None:
    panel = make_panel(seed=9, n_bars=200, n_syms=1)
    entries = pd.DataFrame(False, index=panel["close"].index, columns=panel["close"].columns)
    entries.iloc[[20, 80, 140]] = True
    no_exits = entries.copy() * False
    out = pfm.run_signals(entries, no_exits, panel["close"], STOCK, td_stop=5)
    assert len(out.trades) == 3
    assert (out.trades["exit_idx"] - out.trades["entry_idx"] == 5).all()


def test_pit_mask_is_applied_before_the_time_stop() -> None:
    """A masked-out entry must not schedule a time exit (spec 5.3 ordering)."""
    panel = make_panel(seed=9, n_bars=100, n_syms=1)
    entries = pd.DataFrame(False, index=panel["close"].index, columns=panel["close"].columns)
    entries.iloc[10] = True
    entries.iloc[12] = True
    mask = pd.DataFrame(True, index=entries.index, columns=entries.columns)
    mask.iloc[10] = False  # the bar-10 entry is not eligible
    out = pfm.run_signals(entries, entries * False, panel["close"], STOCK, mask=mask, td_stop=3)
    assert out.trades["entry_idx"].tolist() == [12]
    assert out.trades["exit_idx"].tolist() == [15]


def test_mask_then_time_stop_equals_prefiltered_entries_and_differs_from_the_other_order() -> None:
    """Second, independent test of the ordering: random data, whole-result equality.

    Correct order: mask the entries, then derive the time exit. The wrong order (time exit
    derived from every raw entry, mask applied to entries only) leaves exits from unmasked
    entries that cut later, legitimate holds short.
    """
    from ggTrader.engine import signals as sig

    panel = make_panel(seed=44, n_bars=400, n_syms=6)
    entries, exits = _signals(panel, seed=45, rate=0.05)
    exits = exits & False  # only the time stop exits, so any stray time exit shows
    mask = pd.DataFrame(
        np.random.default_rng(46).random(entries.shape) < 0.35,
        index=entries.index,
        columns=entries.columns,
    )
    close = panel["close"]
    right = pfm.run_signals(entries, exits, close, STOCK, mask=mask, td_stop=7)
    prefiltered = pfm.run_signals(entries & mask, exits, close, STOCK, td_stop=7)
    assert np.array_equal(right.value.to_numpy(), prefiltered.value.to_numpy())
    wrong = pfm.run_signals(
        entries & mask, exits | sig.time_stop(entries, 7), close, STOCK
    )  # time exits scheduled by entries the universe mask rejected
    assert not np.array_equal(right.value.to_numpy(), wrong.value.to_numpy())
    entry_bars = right.trades[["col", "entry_idx"]].to_numpy()
    assert all(mask.to_numpy()[bar, col] for col, bar in entry_bars)


def test_value_has_no_nan_after_the_first_bar(engine: str) -> None:
    panel, _, entries, exits, close, groups = _two_strategy_inputs(holes=True)
    out = pfm.run_signals(entries, exits, close, STOCK, groups=groups)
    assert out.value.notna().all().all()
    # a bar-0 fill pays slippage, so the first equity is at most the starting cash
    assert (out.value.iloc[0] <= STOCK["START_CASH"]).all()
    assert (out.value.iloc[0] > 0.99 * STOCK["START_CASH"]).all()


def test_simulation_really_runs_on_the_configured_engine() -> None:
    import vectorbt_rust.portfolio as rp

    _, _, entries, exits, close, groups = _two_strategy_inputs(holes=False)
    for name, expected in (("rust", True), ("numba", False)):
        config.set_engine(name)
        calls = {"n": 0}
        original = rp.simulate_from_signals_rs

        def counted(*a, _o=original, **k):
            calls["n"] += 1
            return _o(*a, **k)

        rp.simulate_from_signals_rs = counted
        try:
            pfm.run_signals(entries, exits, close, STOCK, groups=groups)
        finally:
            rp.simulate_from_signals_rs = original
        assert (calls["n"] > 0) is expected


def test_rejects_misaligned_inputs() -> None:
    _, _, entries, exits, close, groups = _two_strategy_inputs(holes=False)
    with pytest.raises(ValueError, match="same index"):
        pfm.run_signals(entries.iloc[:-1], exits, close, STOCK, groups=groups)
    with pytest.raises(ValueError, match="START_CASH"):
        pfm.run_signals(entries, exits, close, {"FEES": 0.0}, groups=groups)
    with pytest.raises(ValueError, match="labels"):
        pfm.run_signals(entries, exits, close, STOCK, groups=["a"])


def test_trade_records_carry_group_symbol_and_global_columns() -> None:
    _, _, entries, exits, close, groups = _two_strategy_inputs(holes=False)
    out = pfm.run_signals(entries, exits, close, STOCK, groups=groups, max_bytes=1)  # 2 chunks
    t = out.trades
    assert list(t.columns) == list(pfm.TRADE_COLUMNS)
    assert t["id"].is_unique
    assert set(t["group"]) == {"alpha", "beta"}
    cols = close.columns
    assert all(cols[r.col] == (r.group, r.symbol) for r in t.itertuples())


@pytest.mark.parametrize("field", ["entries", "exits", "size", "high", "low", "weights"])
def test_rejects_same_shape_with_reordered_stock_columns(field: str) -> None:
    close = pd.DataFrame({"A": [100.0, 100.0], "B": [100.0, 100.0]})
    entries = pd.DataFrame({"A": [False, False], "B": [True, False]})
    exits = entries & False
    reversed_columns = ["B", "A"]
    config = {**STOCK, "atr_mult": 2.0, "atr_period": 1}
    if field == "weights":
        with pytest.raises(ValueError, match="columns"):
            pfm.run_weights(close[reversed_columns] * 0.0, close, STOCK)
        return
    args = {"high": close, "low": close, "size": close * 0.0 + 0.1}
    if field == "entries":
        entries = entries[reversed_columns]
    elif field == "exits":
        exits = exits[reversed_columns]
    else:
        args[field] = args[field][reversed_columns]
    with pytest.raises(ValueError, match="columns"):
        pfm.run_signals(entries, exits, close, config, **args)
