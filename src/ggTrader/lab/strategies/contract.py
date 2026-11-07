"""Strategy contract of the vectorbt-engine rebuild (spec 5.3): grid, signal output, helpers.

A strategy declares a :class:`ParamGrid` (values plus validity rules as data) and implements
``build(bars, grid)``. It composes engine presets and returns a :class:`SignalOut` with one
column per ``(grid levels..., symbol)``. It never calls ``Portfolio`` and never applies the
point-in-time (PIT) mask: the caller tiles the mask onto the grid columns (:func:`tile_columns`)
and the engine ANDs it into the entries before the time stop (``engine.run_signals``).

The legacy ``select()/to_targets()`` classes this contract replaced were deleted in P5c.
"""

from __future__ import annotations

import itertools
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol

import numpy as np
import pandas as pd

from ggTrader.data.bars import Bars
from ggTrader.engine import portfolio as engine_portfolio

SYMBOL_LEVEL = "symbol"
#: Column level used when a grid has no swept key (a single combination, e.g. the live config).
COMBO_LEVEL = "combo"


@dataclass(frozen=True)
class Lt:
    """Validity rule as data: ``params[small] < params[large]`` (legacy ``fast < slow``).

    The legacy ``_is_valid_combo`` pairs every key containing ``fast`` with the same-ranked key
    containing ``slow`` (alphabetically); the pairs used by ported strategies are written out
    here so the rule is visible and testable instead of being inferred from key names.
    """

    small: str
    large: str

    def ok(self, combo: Mapping[str, Any]) -> bool:
        """True when the combination satisfies the rule (a rule with a missing key is skipped)."""
        if self.small not in combo or self.large not in combo:
            return True
        return bool(combo[self.small] < combo[self.large])


@dataclass(frozen=True)
class ParamGrid:
    """Parameter values plus validity rules; ``combos()`` equals the legacy ``build_grid``.

    Attributes:
        values: every parameter name with its candidate values (single-valued names are
            pinned parameters, exactly as the legacy ``sweep_params()`` lists them).
        rules: validity rules applied to every combination.
    """

    values: Mapping[str, tuple[Any, ...]]
    rules: tuple[Lt, ...] = ()

    @classmethod
    def from_sweep(cls, values: Mapping[str, Sequence[Any]], rules: Sequence[Lt] = ()) -> ParamGrid:
        """Grid from a legacy-style ``{name: [values]}`` mapping."""
        return cls({k: tuple(v) for k, v in values.items()}, tuple(rules))

    @classmethod
    def single(cls, params: Mapping[str, Any]) -> ParamGrid:
        """One-combination grid (the live defaults, OD14)."""
        return cls({k: (v,) for k, v in params.items()})

    def override(self, overrides: Mapping[str, Sequence[Any]]) -> ParamGrid:
        """Grid with some names replaced or added (``--sweep-param``)."""
        merged = {**self.values, **{k: tuple(v) for k, v in overrides.items()}}
        return replace(self, values=merged)

    def combos(self) -> list[dict[str, Any]]:
        """Valid combinations in the legacy order (cartesian product over sorted names)."""
        keys = sorted(self.values)
        raw = (
            dict(zip(keys, vals, strict=True))
            for vals in itertools.product(*(self.values[k] for k in keys))
        )
        return [c for c in raw if all(rule.ok(c) for rule in self.rules)]

    def __len__(self) -> int:
        return len(self.combos())

    def used_values(self, name: str) -> list[Any]:
        """Distinct values of ``name`` that occur in a valid combination (first-seen order)."""
        seen: list[Any] = []
        for combo in self.combos():
            if combo[name] not in seen:
                seen.append(combo[name])
        return seen

    def swept(self) -> list[str]:
        """Names that take more than one value across the valid combinations (sorted)."""
        combos = self.combos()
        return [k for k in sorted(self.values) if len({c[k] for c in combos}) > 1]


def resolve_combos(grid: ParamGrid, defaults: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Valid combinations with the strategy's constructor defaults filled in for absent names."""
    return [{**defaults, **combo} for combo in grid.combos()]


def group_columns(
    combos: Sequence[Mapping[str, Any]], levels: Sequence[str], symbols: Sequence[str]
) -> pd.MultiIndex:
    """Column index ``(levels..., symbol)``, one block of ``symbols`` per combination.

    With no swept level the single combination is labelled by a constant ``combo`` level, so
    every output has the same shape and the engine can group on all levels but the last.
    """
    names = list(levels) or [COMBO_LEVEL]
    keys: list[tuple[Hashable, ...]] = []
    for i, combo in enumerate(combos):
        keys.append(tuple(combo[k] for k in levels) if levels else (i,))
    tuples = [(*key, sym) for key in keys for sym in symbols]
    return pd.MultiIndex.from_tuples(tuples, names=[*names, SYMBOL_LEVEL])


def tile_columns(frame: pd.DataFrame, columns: pd.MultiIndex) -> pd.DataFrame:
    """Repeat a ``dates x symbols`` frame onto grid columns (close, volume, the PIT mask).

    Raises:
        KeyError: a grid symbol is missing from ``frame``.
    """
    symbols = columns.get_level_values(-1)
    missing = sorted(set(symbols) - set(frame.columns))
    if missing:
        raise KeyError(f"frame has no column for {missing[:5]}")
    return frame.reindex(columns=symbols).set_axis(columns, axis=1)


@dataclass(frozen=True, eq=False)
class SignalOut:
    """What ``SignalStrategy.build`` returns (spec 5.3).

    Attributes:
        entries, exits: bool frames, columns ``(grid levels..., symbol)``.
        combos: the parameter dict of each group, in first-appearance column order.
        size: per-entry fraction of the group's equity; ``None`` means the caller's
            ``SIGNAL_POSITION_SIZE`` (lab default 0.03). A frame holds per-bar sizes, NaN
            where the base size applies (conviction sizing).
        sl_stop, sl_trail, tp_stop: stop metadata, applied by the caller through the engine.
        td_stop: forced exit this many bars after each (masked) entry; ggTrader glue, the
            engine applies it after the PIT mask.
        meta: free-form diagnostics (voters used, notes) for run JSON.
    """

    entries: pd.DataFrame
    exits: pd.DataFrame
    combos: tuple[dict[str, Any], ...]
    size: float | pd.DataFrame | None = None
    sl_stop: float | None = None
    sl_trail: bool = False
    tp_stop: float | None = None
    td_stop: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def symbols(self) -> list[str]:
        """Distinct symbols in column order."""
        return list(dict.fromkeys(self.entries.columns.get_level_values(-1)))

    def group_labels(self) -> list[tuple[Hashable, ...]]:
        """Group label (the grid-level tuple) of every column."""
        cols = self.entries.columns
        return [tuple(k[:-1]) for k in cols.to_list()]

    def group(self, i: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Entries and exits of group ``i`` as plain ``dates x symbols`` frames."""
        n = len(self.symbols)
        sl = slice(i * n, (i + 1) * n)
        syms = self.symbols
        return (
            self.entries.iloc[:, sl].set_axis(syms, axis=1),
            self.exits.iloc[:, sl].set_axis(syms, axis=1),
        )


def next_bar_positions(index: pd.DatetimeIndex, rebal: Sequence[pd.Timestamp]) -> list[int]:
    """Bar position each rebalance date takes effect on: the first bar strictly after it.

    A date with no later bar is dropped (nothing to trade after it), as the legacy
    ``to_targets``.
    """
    pos = index.searchsorted(pd.DatetimeIndex(rebal), side="right")
    return [int(p) if p < len(index) else -1 for p in pos]


def stack_weight_plans(
    index: pd.DatetimeIndex,
    rebal: Sequence[pd.Timestamp],
    plans: Sequence[Mapping[pd.Timestamp, Mapping[str, float]]],
    combos: Sequence[Mapping[str, Any]],
    levels: Sequence[str],
) -> pd.DataFrame:
    """Target-weight frame for a grid: ``plans[g][asof]`` maps selected symbols to weights.

    On the first bar after each rebalance date every column of the grid is set to 0.0 (exit
    whatever is not re-selected) and the selected symbols get their weight; all other bars are
    NaN (hold). Columns are the symbols selected anywhere in the grid, sorted, one block per
    combination: a symbol never selected by a combination just stays at 0.0 there.
    """
    symbols = sorted({s for plan in plans for sel in plan.values() for s in sel})
    col_pos = {s: j for j, s in enumerate(symbols)}
    n_sym = len(symbols)
    values = np.full((len(index), len(plans) * n_sym), np.nan)
    positions = next_bar_positions(index, rebal)
    for g, plan in enumerate(plans):
        base = g * n_sym
        for asof, pos in zip(rebal, positions, strict=True):
            if pos < 0:
                continue
            values[pos, base : base + n_sym] = 0.0
            for sym, weight in plan.get(asof, {}).items():
                values[pos, base + col_pos[sym]] = float(weight)
    return pd.DataFrame(values, index=index, columns=group_columns(combos, levels, symbols))


class SignalStrategy(Protocol):
    """A signal strategy: a grid and a ``build`` that composes engine presets only."""

    name: ClassVar[str]
    kind: ClassVar[str]
    grid: ClassVar[ParamGrid]
    live: ClassVar[dict[str, Any] | None]

    def build(self, bars: Bars, grid: ParamGrid) -> SignalOut:
        """Entries and exits for every combination of ``grid`` over ``bars``."""
        ...


class WeightStrategy(Protocol):
    """A weight strategy: target weights on the bar after each rebalance date."""

    name: ClassVar[str]
    kind: ClassVar[str]
    grid: ClassVar[ParamGrid]

    def build(
        self,
        bars: Bars,
        grid: ParamGrid,
        rebal: pd.DatetimeIndex,
        eligible: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Target weights (NaN = hold), columns ``(grid levels..., symbol)``."""
        ...


def simulate_signal_out(
    out: SignalOut,
    close: pd.DataFrame,
    base_config: Mapping[str, Any],
    *,
    mask: pd.DataFrame | None = None,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    max_bytes: int = engine_portfolio.DEFAULT_MAX_BYTES,
) -> engine_portfolio.SimResult:
    """Run a :class:`SignalOut` through ``engine.run_signals`` (mask tiled, one group per combo).

    A thin convenience for tests and drivers; the P5 WFO will call the engine directly.
    """
    cols = out.entries.columns
    config = dict(base_config)
    if out.sl_stop is not None:
        config["ts_stop"] = float(out.sl_stop)
    if out.tp_stop is not None:
        config["tp_stop"] = float(out.tp_stop)
    size = out.size
    if isinstance(size, pd.DataFrame):
        size = size.fillna(float(config.get("SIGNAL_POSITION_SIZE", 0.02)))
    return engine_portfolio.run_signals(
        out.entries,
        out.exits,
        tile_columns(close, cols),
        config,
        groups=out.group_labels(),
        mask=None if mask is None else tile_columns(mask, cols),
        td_stop=out.td_stop,
        size=size,
        high=None if high is None else tile_columns(high, cols),
        low=None if low is None else tile_columns(low, cols),
        max_bytes=max_bytes,
    )
