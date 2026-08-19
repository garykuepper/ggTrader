"""Market-neutral pairs / statistical-arbitrage mean reversion.

Long one stock, short a correlated same-sector partner, betting the price
spread reverts after diverging. The first market-neutral (simultaneous
long+short) construction in this lab -- every prior strategy is long-only
directional. v1 uses a naive 1:1 log-price spread (no beta-hedging) and
does NOT model short-borrow/margin-interest costs; see
docs/research/RESEARCH_SNAPSHOT.md and the eventual dated research report
for those disclosures.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List

import pandas as pd

from ggTrader.lab.strategies.indicators import extract_close
from ggTrader.lab.strategies.registry import load_sector_map
from ggTrader.lab.strategy import LabConfig, Plan


def enumerate_sector_pairs(symbols: List[str], sector_map: Dict[str, str]) -> List[tuple[str, str]]:
    """All same-sector 2-combinations from symbols, grouped by GICS sector.

    Symbols with no sector entry ("Unknown") are skipped -- no economic
    rationale for pairing unclassified stocks.
    """
    groups: Dict[str, List[str]] = {}
    for s in symbols:
        sec = sector_map.get(s, "Unknown")
        if sec == "Unknown":
            continue
        groups.setdefault(sec, []).append(s)
    pairs: List[tuple[str, str]] = []
    for members in groups.values():
        pairs.extend(combinations(sorted(members), 2))
    return pairs


def pairwise_correlations(
    close: pd.DataFrame, pairs: List[tuple[str, str]], lookback: int
) -> Dict[tuple[str, str], float]:
    """Trailing-window Pearson correlation of daily log returns, per candidate pair."""
    import numpy as np

    log_ret = np.log(close).diff()
    window = log_ret.iloc[-lookback:]
    out: Dict[tuple[str, str], float] = {}
    for a, b in pairs:
        if a not in window.columns or b not in window.columns:
            continue
        sub = window[[a, b]].dropna()
        if len(sub) < lookback * 0.8:
            continue
        c = sub[a].corr(sub[b])
        if pd.notna(c):
            out[(a, b)] = float(c)
    return out


def spread_zscore(close_a: pd.Series, close_b: pd.Series, lookback: int) -> pd.Series:
    """Naive 1:1 log-price spread, z-scored over a trailing window.

    spread_t = log(close_a_t) - log(close_b_t)
    z_t = (spread_t - rolling_mean) / rolling_std
    """
    import numpy as np

    spread = np.log(close_a) - np.log(close_b)
    mu = spread.rolling(lookback, min_periods=lookback).mean()
    sigma = spread.rolling(lookback, min_periods=lookback).std()
    return (spread - mu) / sigma


def pair_signal(z: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    """Directional position state per date: +1.0 (long A/short B), -1.0
    (short A/long B), 0.0 (flat). z's sign determines direction (spread too
    high => A relatively expensive => short A); magnitude determines
    entry/exit. State persists (forward-filled) between the entry and exit
    thresholds -- mirrors the vectorized-threshold-crossing style of
    bb_signals (indicators.py), not leveraged_rotation.rotate_positions'
    per-series hysteresis loop (that helper's symmetric-threshold semantics
    don't map onto a directional, magnitude-gated signal like this one)."""
    long_a = z <= -entry_z
    short_a = z >= entry_z
    flat = z.abs() <= exit_z

    raw = pd.Series(float("nan"), index=z.index)
    raw[long_a] = 1.0
    raw[short_a] = -1.0
    raw[flat] = 0.0
    return raw.ffill().fillna(0.0)


#: Memoizes same-sector candidate-pair enumeration + correlation filtering +
#: latest z-score, keyed by (asof, corr_lookback, corr_min, eligible symbols).
#: None of that work depends on entry_z/exit_z/max_pairs/min_hold_months
#: (the only swept params), but the WFO harness constructs a fresh strategy
#: instance per grid combo and calls select() on the SAME data window for
#: every one of them -- without this cache, the full sector-pair correlation
#: scan (potentially ~1000 pairs) gets redundantly recomputed once per
#: combo, dozens of times per fold. Mirrors leveraged_rotation.py's
#: _breadth_cache and leveraged_trend.py's _underlying_cache -- also a
#: process-local dict, so under joblib's per-combo worker parallelism each
#: worker still pays for its own first miss per (asof, corr_lookback,
#: corr_min, eligible); only redundant recompute *within* a worker process
#: is eliminated.
_pair_candidate_cache: dict[tuple, list[tuple[tuple[str, str], float]]] = {}


def _cached_qualifying_pairs(
    asof: pd.Timestamp,
    data: pd.DataFrame,
    eligible: tuple[str, ...],
    corr_lookback: int,
    corr_min: float,
) -> list[tuple[tuple[str, str], float]]:
    key = (asof, corr_lookback, corr_min, eligible)
    if key not in _pair_candidate_cache:
        sector_map = load_sector_map()
        candidate_pairs = enumerate_sector_pairs(list(eligible), sector_map)
        if not candidate_pairs:
            _pair_candidate_cache[key] = []
            return _pair_candidate_cache[key]

        pair_symbols = sorted({s for pair in candidate_pairs for s in pair})
        close = extract_close(data, pair_symbols)
        corrs = pairwise_correlations(close, candidate_pairs, corr_lookback)
        # corr_min is a fixed structural gate (not swept -- see the
        # implementation plan's rationale): a pair whose correlation has
        # broken down is excluded here even if previously held, which
        # doubles as a correlation-breakdown exit trigger.
        qualifying = [p for p in candidate_pairs if corrs.get(p, -1.0) >= corr_min]

        out: list[tuple[tuple[str, str], float]] = []
        for a, b in qualifying:
            if a not in close.columns or b not in close.columns:
                continue
            z_series = spread_zscore(close[a], close[b], corr_lookback)
            if z_series.empty or pd.isna(z_series.iloc[-1]):
                continue
            out.append(((a, b), float(z_series.iloc[-1])))
        _pair_candidate_cache[key] = out
    return _pair_candidate_cache[key]


class PairsStatArb:
    """Market-neutral: long/short a same-sector pair on log-spread z-score
    divergence. target_kind="weights" -- the vol-targeting overlay (signals-
    only, see leveraged_trend.py) does not apply here; this is a pure
    weight/targetpercent strategy, verified to support short (negative)
    weights via simulate_weights (tests/lab/test_simulate.py)."""

    name = "pairs_stat_arb"
    target_kind = "weights"

    def __init__(
        self,
        cfg: LabConfig,
        corr_lookback: int = 126,
        corr_min: float = 0.7,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        max_pairs: int = 10,
        min_hold_months: int = 1,
    ) -> None:
        self.cfg = cfg
        self.corr_lookback = corr_lookback
        self.corr_min = corr_min
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_pairs = max_pairs
        self.min_hold_months = min_hold_months
        # Sequential per-instance state: safe because wfo.py constructs a
        # fresh instance per grid combo and calls select() in date order
        # only within that one combo's own fold -- see wfo.py:338.
        self._held_since: Dict[tuple[str, str], int] = {}
        self._rebalance_idx = 0

    @classmethod
    def sweep_params(cls) -> dict[str, list]:
        return {
            "entry_z": [1.5, 2.0, 2.5],
            "exit_z": [0.25, 0.5, 0.75],
            "max_pairs": [5, 10, 20],
            "min_hold_months": [1, 2],
        }

    def select(self, asof: pd.Timestamp, data: pd.DataFrame, eligible: List[str]) -> Plan:
        data = data.loc[:asof]  # defense in depth: invariant to post-asof rows
        current_idx = self._rebalance_idx
        self._rebalance_idx += 1

        qualifying_with_z = _cached_qualifying_pairs(
            asof, data, tuple(sorted(eligible)), self.corr_lookback, self.corr_min
        )
        if not qualifying_with_z:
            self._held_since = {}
            return []

        scored: list[tuple[float, tuple[str, str], float]] = []
        for (a, b), z in qualifying_with_z:
            held_since = self._held_since.get((a, b))
            if held_since is not None:
                hold_elapsed = (current_idx - held_since) >= self.min_hold_months
                # Force-include while still divergent OR the hold floor
                # hasn't elapsed yet; only drop once BOTH converged and the
                # minimum hold period has passed.
                if abs(z) > self.exit_z or not hold_elapsed:
                    scored.append((abs(z), (a, b), z))
            elif abs(z) >= self.entry_z:
                scored.append((abs(z), (a, b), z))

        scored.sort(key=lambda t: t[0], reverse=True)

        selected: list[tuple[tuple[str, str], float]] = []
        used_symbols: set[str] = set()
        for _, pair, z in scored:
            if len(selected) >= self.max_pairs:
                break
            a, b = pair
            if a in used_symbols or b in used_symbols:
                continue
            selected.append((pair, z))
            used_symbols.add(a)
            used_symbols.add(b)

        self._held_since = {pair: self._held_since.get(pair, current_idx) for pair, _z in selected}

        weight = 1.0 / self.max_pairs
        plan: Plan = []
        for (a, b), z in selected:
            if z >= 0:
                # Spread high -> A relatively expensive vs B -> short A, long B.
                plan.append({"symbol_long": b, "symbol_short": a, "weight": weight, "z": z})
            else:
                plan.append({"symbol_long": a, "symbol_short": b, "weight": weight, "z": z})
        return plan

    def to_targets(self, plans: Dict[pd.Timestamp, Plan], data: pd.DataFrame) -> pd.DataFrame:
        import numpy as np

        symbols = sorted(
            {
                leg
                for plan in plans.values()
                for sel in plan
                for leg in (sel["symbol_long"], sel["symbol_short"])
            }
        )
        if not symbols or len(data.index) == 0:
            return pd.DataFrame(index=data.index, columns=symbols)

        targets = pd.DataFrame(np.nan, index=data.index, columns=symbols)
        for asof in sorted(plans):
            forward = data.index[data.index > asof]
            if len(forward) == 0:
                continue
            bar = forward[0]
            targets.loc[bar, symbols] = 0.0  # reset: close any leg not re-selected
            for sel in plans[asof]:
                # Later entries in the same rebalance's plan list overwrite
                # earlier ones for a shared symbol -- defined last-write-wins
                # behavior (select()'s own dedup prevents this in practice).
                targets.loc[bar, sel["symbol_long"]] = float(sel["weight"])
                targets.loc[bar, sel["symbol_short"]] = -float(sel["weight"])
        return targets
