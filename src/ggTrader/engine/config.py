"""Engine settings: which vectorbt engine runs and how returns are annualised.

Every setting below is documented by reference to the design spec
(docs/superpowers/specs/2026-11-06-vectorbt-engine-design.md):

* ``GGT_VBT_ENGINE`` in {``rust``, ``numba``}, default ``rust``, never ``auto`` (spec 5.5).
  ``auto`` silently falls back to numba when the vectorbt-rust wheel is missing or its
  major.minor differs, costing 20-60 s of JIT per fresh process with no error.
* ``year_freq`` ``"252 days"`` (spec OD2, 5.1): vectorbt's default is ``"365 days"``, which
  inflates Sharpe by 20% and annualised volatility by 20% on daily equity bars.
* ``freq`` ``"1D"`` (spec 5.1): the bar frequency every simulation and returns accessor
  uses, so no result depends on an index that could not infer it.
* ``contract`` ``CONTRACT_VERSION`` (spec 5.8): an integer bumped when an adapter definition
  changes (annualisation factor, ddof, Sortino, CAGR convention, DSR form), stamped into
  every result so old and new numbers are never compared as one series.
* ``MIN_LOSING_DAYS`` (default 20): a window needs at least this many losing days before its
  Sortino counts (a degeneracy guard like the flat-curve guard, not a new definition; P5b,
  review MAJOR-1). With a handful of losing days the downside deviation is estimated from
  almost nothing and the ratio explodes (fold 17 of the 17-fold core run: Sortino 28.6 from one
  losing day), so a near-flat combination would win a composite. Pass ``min_losing_days=`` to
  ``window_stats`` / ``run_wfo`` to override; the value used is stamped into every WFO result.
* The vectorbt/vectorbt-rust major.minor mismatch ``UserWarning`` becomes an error (spec 5.1).

This module is the only place engine settings are written. It imports vectorbt (as does
every module in this package) and nothing from ggTrader.
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import vectorbt as vbt

ENGINES: tuple[str, ...] = ("rust", "numba")
ENV_ENGINE = "GGT_VBT_ENGINE"
DEFAULT_ENGINE = "rust"
YEAR_FREQ = "252 days"
BAR_FREQ = "1D"
TRADING_DAYS = 252
#: Fewest losing days (returns < 0) a scoring window needs for its Sortino to count; below it
#: the Sortino is NaN and the composite floors it like any other missing statistic. 20 is the
#: smallest threshold that removes exactly the four 17-fold-core winners whose Sortino was an
#: artefact of 1 to 14 losing days (reviewer's measurement, P5 review).
MIN_LOSING_DAYS = 20

#: Bump when an adapter definition changes (spec 5.8, Appendix D Q6).
CONTRACT_VERSION = 1
#: E1 = first baseline on vectorbt's definitions (owner decision 2026-11-06); a later definition
#: change starts E2.
BASELINE_EPOCH = "E1"

SETTINGS_DOC = (
    "GGT_VBT_ENGINE year_freq freq contract: see the module docstring; each setting cites the spec."
)

# The exact text vectorbt._engine.is_rust_available warns with on a minor mismatch.
_MISMATCH_MESSAGE = re.compile(r"vectorbt .*version mismatch")


@dataclass(frozen=True)
class EngineSettings:
    """Explicit engine configuration (spec 5.1)."""

    engine: str = DEFAULT_ENGINE
    year_freq: str = YEAR_FREQ
    freq: str = BAR_FREQ

    def __post_init__(self) -> None:
        if self.engine not in ENGINES:
            raise ValueError(
                f"{ENV_ENGINE} must be one of {ENGINES}, got {self.engine!r} "
                "('auto' is rejected: it silently falls back to numba)"
            )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> EngineSettings:
        """Read ``GGT_VBT_ENGINE`` (blank means the default); raise on anything else."""
        env = os.environ if environ is None else environ
        raw = (env.get(ENV_ENGINE) or "").strip().lower()
        return cls(engine=raw or DEFAULT_ENGINE)


def configure(settings: EngineSettings | None = None) -> EngineSettings:
    """Apply the settings to vectorbt, process-wide; runs once on ``import ggTrader.engine``.

    Args:
        settings: explicit settings; ``None`` reads the environment.

    Returns:
        The settings that were applied.
    """
    applied = settings if settings is not None else EngineSettings.from_env()
    vbt.settings["engine"] = applied.engine
    vbt.settings["returns"]["year_freq"] = applied.year_freq
    vbt.settings["array_wrapper"]["freq"] = applied.freq
    warnings.filterwarnings("error", message=_MISMATCH_MESSAGE.pattern, category=UserWarning)
    return applied


def active_engine() -> str:
    """The engine vectorbt is currently configured to use ("rust" or "numba")."""
    return str(vbt.settings["engine"])


def set_engine(engine: str) -> None:
    """Switch the process-wide engine (tests and the numba escape hatch); rejects ``auto``."""
    vbt.settings["engine"] = EngineSettings(engine=engine).engine
