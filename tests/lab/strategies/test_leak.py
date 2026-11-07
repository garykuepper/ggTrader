"""Lookahead checks (truncation and perturbation) on every ported strategy, plus a leaky one."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ggTrader.lab.data import rebalance_dates
from ggTrader.lab.strategies import REGISTRY
from ggTrader.lab.strategies.contract import ParamGrid, SignalOut, group_columns, resolve_combos
from ggTrader.lab.strategies.cross_asset_sleeve import UNIVERSE
from ggTrader.lab.strategies.ensemble import Ensemble
from ggTrader.lab.strategies.leak import (
    perturb_after,
    signal_leak_check,
    truncate_bars,
    weight_leak_check,
)
from ggTrader.lab.strategies.retail_attention import RetailAttention
from tests.lab.strategies.helpers import make_bars

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def bars():
    return make_bars(seed=61, n_bars=520, n_syms=6, holes=True)


def _dates(bars, n: int = 6) -> list[pd.Timestamp]:
    rng = np.random.default_rng(1)
    pos = sorted(rng.choice(np.arange(60, len(bars.close) - 2), size=n, replace=False))
    return [bars.close.index[p] for p in pos]


def small_grid(grid: ParamGrid, max_swept: int = 2) -> ParamGrid:
    """A few combinations of ``grid``: two values for the first swept keys, the rest pinned."""
    swept = grid.swept()[:max_swept]
    values = {k: (v[:2] if k in swept else v[:1]) for k, v in grid.values.items()}
    reduced = ParamGrid(values, grid.rules)
    return reduced if len(reduced) else ParamGrid.single(grid.combos()[0])


def _interest_loader(syms, start, end):
    months = pd.date_range("2019-01-01", "2021-06-01", freq="MS")
    rows = [(s, m, 10.0 + (int(s[1:]) + m.month * 5) % 19) for s in syms for m in months]
    df = pd.DataFrame(rows, columns=["symbol", "date", "search_interest"])
    return df[(df["date"] >= start) & (df["date"] <= end)]


#: Strategies that need more than synthetic price bars: ``name -> (instance, bars)``. A new
#: strategy needs an entry only when it reads data beyond bars, or specific symbols.
SPECIAL = {
    "cross_asset_sleeve": lambda: (
        REGISTRY["cross_asset_sleeve"](),
        make_bars(seed=63, n_bars=520, n_syms=5, symbols=sorted(UNIVERSE)),
    ),
    "retail_attention": lambda: (
        RetailAttention(interest_loader=_interest_loader, min_history_bars=1),
        make_bars(seed=62, n_bars=520, n_syms=10, holes=True),
    ),
}

#: Grids beyond the generic reduced one that carry the deployed or sensitive configuration.
EXTRA_SIGNAL_GRIDS = {
    "ensemble": [
        Ensemble.live_grid(),
        ParamGrid.from_sweep({**Ensemble.grid.values, "min_agree": [2, 3]}),
    ],
}


def _signal_cases():
    for name, cls in REGISTRY.items():
        if cls.kind != "signals":
            continue
        yield pytest.param(name, small_grid(cls.grid), id=f"{name}-small")
        for i, grid in enumerate(EXTRA_SIGNAL_GRIDS.get(name, [])):
            yield pytest.param(name, grid, id=f"{name}-extra{i}")


def _weight_cases():
    for name, cls in REGISTRY.items():
        if cls.kind == "weights":
            yield pytest.param(name, small_grid(cls.grid), id=name)


@pytest.mark.parametrize(("name", "grid"), list(_signal_cases()))
def test_signal_strategies_have_no_lookahead(bars, name, grid) -> None:
    """Every registered signal strategy, on its reduced grid (a new module is covered at once)."""
    assert signal_leak_check(REGISTRY[name](), bars, grid, _dates(bars)) == []


@pytest.mark.parametrize(("name", "grid"), list(_weight_cases()))
def test_weight_strategies_have_no_lookahead(name, grid) -> None:
    """Every registered weight strategy: decision rows do not move when later bars change."""
    strategy, wbars = (
        SPECIAL[name]()
        if name in SPECIAL
        else (REGISTRY[name](), make_bars(seed=62, n_bars=520, n_syms=10, holes=True))
    )
    rebal = list(rebalance_dates(wbars.close.index, wbars.close.index[300], wbars.close.index[-1]))[
        :4
    ]
    assert weight_leak_check(strategy, wbars, grid, rebal) == []


def test_every_registered_strategy_is_covered_by_a_leak_case() -> None:
    covered = {c.values[0] for c in _signal_cases()} | {c.values[0] for c in _weight_cases()}
    assert covered == set(REGISTRY)


class _PeekingStrategy:
    """Deliberately leaky: enters when tomorrow's close is higher (a forward-looking signal)."""

    name = "peek"
    kind = "signals"
    grid = ParamGrid.from_sweep({"k": [1]})
    defaults = {"k": 1}

    def build(self, bars, grid=None) -> SignalOut:
        combos = resolve_combos(grid or self.grid, self.defaults)
        entries = (bars.close.shift(-1) > bars.close).fillna(False)
        cols = group_columns(combos, [], bars.symbols)
        e = entries.set_axis(cols, axis=1)
        return SignalOut(entries=e, exits=~e, combos=tuple(combos))


def test_the_check_catches_a_forward_looking_strategy(bars) -> None:
    problems = signal_leak_check(_PeekingStrategy(), bars, _PeekingStrategy.grid, _dates(bars, 4))
    assert any("(truncate)" in p for p in problems) and any("(perturb)" in p for p in problems)


def test_truncate_and_perturb_helpers(bars) -> None:
    t = bars.close.index[300]
    short = truncate_bars(bars, t)
    assert short.close.index[-1] == t and short.volume.shape[0] == 301
    moved = perturb_after(bars, t)
    assert moved.close.index.equals(bars.close.index)
    assert moved.close.loc[:t].equals(bars.close.loc[:t])
    assert not moved.close.iloc[310].equals(bars.close.iloc[310])


def test_no_vectorbt_label_functions_in_strategy_sources() -> None:
    """vectorbt.labels look forward by design (spec 7.2); no strategy code may reference them."""
    import ast
    from pathlib import Path

    banned = {"labels", "FMEAN", "FSTD", "FMIN", "FMAX", "TRENDLB", "BOLB", "MEANLB", "LEXLB"}
    root = Path(__file__).resolve().parents[3] / "src" / "ggTrader" / "lab" / "strategies"
    files = sorted(root.glob("*.py"))
    assert len(files) > 8
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            names = set()
            if isinstance(node, ast.Name | ast.Attribute):
                names.add(node.id if isinstance(node, ast.Name) else node.attr)
            elif isinstance(node, ast.ImportFrom):
                names |= {a.name for a in node.names} | set((node.module or "").split("."))
            elif isinstance(node, ast.Import):
                names |= {part for a in node.names for part in a.name.split(".")}
            assert not (names & banned), f"{path.name}:{node.lineno} uses {names & banned}"
