"""Feature catalog base types (spec §5.6, widened in Phase 3.5).

A ``Feature`` is a named, time-indexed value derived from raw data. Phase 3
keyed everything on a single instrument; Phase 3.5 widens to handle pair
features (basis = f(spot, future)) and universe features (cross-sectional
rank, beta-to-leader). All features now operate on a *sequence* of instruments
and return a ``pandas.DataFrame`` indexed by tz-aware UTC timestamps with one
column per instrument-tuple.

``FeatureStore`` is the abstract registry/resolver. The concrete Phase 3.5
implementations (``SyntheticFeatureStore``) and the future TimescaleDB-backed
implementation (Phase 2 proper) both satisfy this Protocol.

Strategies still want ergonomic per-bar access. ``get_at`` returns a single
row (a Series indexed by feature-column) for one timestamp without forcing
each caller to slice DataFrames.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal, Protocol, Sequence, Union, runtime_checkable

import pandas as pd

from ggTrader.core.instrument import Instrument
from ggTrader.core.universe import Universe

Arity = Literal["single", "pair", "universe"]
InstrumentArg = Union[Instrument, Sequence[Instrument], Universe]


def _normalize_instruments(arg: InstrumentArg, ts: datetime) -> list[Instrument]:
    """Resolve any of (Instrument, Sequence[Instrument], Universe) to a list."""
    if isinstance(arg, Instrument):
        return [arg]
    if isinstance(arg, Universe):
        return list(arg.members(ts))
    return list(arg)


class Feature(ABC):
    """Base for all features. Subclasses set ``name`` and ``arity`` and
    implement ``compute``."""

    name: str
    arity: Arity = "single"
    dependencies: list[str] = []

    @abstractmethod
    def compute(
        self,
        instruments: Sequence[Instrument],
        start: datetime,
        end: datetime,
        store: "FeatureStore",
    ) -> pd.DataFrame:
        """Return a DataFrame indexed by tz-aware UTC timestamps.

        Column convention:
          - ``arity="single"``: one column per instrument, labeled by
            ``instrument.symbol``.
          - ``arity="pair"``: one column for the pair, labeled by a tuple
            of symbols joined with ``|`` (e.g. ``"BTC-USD|BTC-USD-260626"``).
            Callers pass ``[spot, future]`` in that order.
          - ``arity="universe"``: one column per universe member, labeled
            by ``instrument.symbol``. Membership is point-in-time.
        """


@runtime_checkable
class FeatureStore(Protocol):
    """Registry + resolver for features.

    ``get`` returns a DataFrame for any arity; ``get_at`` returns a single
    row keyed by feature column. Strategies typically use ``get_at``; backtest
    sweeps typically use ``get``.
    """

    def get(
        self,
        feature_name: str,
        instruments: InstrumentArg,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame: ...

    def get_at(
        self,
        feature_name: str,
        instruments: InstrumentArg,
        ts: datetime,
    ) -> pd.Series: ...


def pair_column_label(instruments: Sequence[Instrument]) -> str:
    """Canonical column label for a pair feature: ``"BTC-USD|BTC-USD-260626"``."""
    return "|".join(i.symbol for i in instruments)
