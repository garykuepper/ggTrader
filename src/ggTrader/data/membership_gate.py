"""The live gate on the S&P 500 list: pass, warn or refuse, from how stale the list is.

Spec docs/superpowers/specs/2026-11-07-membership-database-design.md, section 3.4 with decisions
D1 and D8, and Appendix A item 1. Two clocks decide, both counted in calendar days at ``asof``:

* **Verification age**, ``asof - verified_at``: how long ago our copy was last proved equal to
  upstream (``universe.checks`` row ``unchanged``, or the load itself). More than
  :data:`MAX_VERIFY_AGE_DAYS` (45 until the weekly job exists; the environment variable
  ``GGT_MEMBERSHIP_MAX_VERIFY_AGE_DAYS`` overrides it) refuses.
* **Lag**, ``asof - coverage_end``: how far the list's last change date trails the day. Equal to
  or beyond :data:`LAG_WARN_DAYS` (90) warns; equal to or beyond :data:`LAG_REFUSE_DAYS` (180)
  refuses. Equality counts because D8 says "warn at 90 days, refuse at 180".  A verification
  can never cure a lag, because the check only sees what upstream already knows.

The gate applies to a real run only: a wall-clock ``now`` reading the *latest* ``db`` vintage
(:func:`gate_applies`). It is skipped for fixtures, for a pinned ``db:<id>`` and when a caller
injects ``now`` (replays and tests), so evidence runs are never blocked. Nothing here reads the
database or the clock; :func:`decide_gate` is a pure function of a vintage and a day.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Literal

from ggTrader.data.membership import (
    MembershipError,
    MembershipSource,
    MembershipVintage,
    age_since_verified,
)
from ggTrader.data.universe_algebra import as_date

#: Longest tolerated age of the last upstream check, in days (spec D1: 45 until the weekly job).
MAX_VERIFY_AGE_DAYS = 45
#: Environment variable that overrides :data:`MAX_VERIFY_AGE_DAYS` (a positive whole number).
VERIFY_AGE_ENV = "GGT_MEMBERSHIP_MAX_VERIFY_AGE_DAYS"
#: The list's last change date this many days before the run warns (D8).
LAG_WARN_DAYS = 90
#: ... and this many days before the run refuses (D8).
LAG_REFUSE_DAYS = 180

#: ``SignalGateError`` codes (spec 3.4 names the first two; the lag code is the D8 counterpart).
CODE_UNAVAILABLE = "membership_unavailable"
CODE_STALE = "membership_stale"
CODE_LAG = "membership_lag"
CODE_CONFIG = "membership_gate_config"

Level = Literal["pass", "warn", "refuse"]

CHECK_FIX = "run `ggt db membership check-upstream`"
LOAD_FIX = (
    "load the newer list: `ggt db load-sp500` (a dry run shows the diff; `check-upstream` says "
    "whether upstream has one). A check alone cannot cure a lag"
)


class GateConfigError(MembershipError, ValueError):
    """The age-limit override is not a positive whole number; the gate refuses to guess."""


def max_verify_age_days(environ: Mapping[str, str] | None = None) -> int:
    """The verification-age limit in force: the environment override, else the constant."""
    text = (os.environ if environ is None else environ).get(VERIFY_AGE_ENV)
    if text is None or text == "":
        return MAX_VERIFY_AGE_DAYS
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value < 1:
        raise GateConfigError(
            f"{VERIFY_AGE_ENV}={text!r} is not a positive whole number of days; "
            "unset it to use the default of "
            f"{MAX_VERIFY_AGE_DAYS}"
        )
    return value


@dataclass(frozen=True)
class GateDecision:
    """What the gate decided at one day, with the numbers behind it."""

    level: Level
    code: str | None
    asof: date
    age_since_verified_days: int
    lag_days: int
    max_verify_age_days: int
    lag_warn_days: int
    lag_refuse_days: int
    reasons: tuple[str, ...]

    @property
    def message(self) -> str:
        """One line a human can act on: every reason that fired, each with its fix."""
        return "; ".join(self.reasons)

    def as_dict(self) -> dict[str, Any]:
        """Plain-data form for the signals dict and the run record."""
        out = asdict(self)
        out["asof"] = self.asof.isoformat()
        out["reasons"] = list(self.reasons)
        out["message"] = self.message
        out["applied"] = True
        return out


def decide_gate(
    vintage: MembershipVintage,
    asof: date,
    *,
    max_verify_age: int | None = None,
    lag_warn: int = LAG_WARN_DAYS,
    lag_refuse: int = LAG_REFUSE_DAYS,
) -> GateDecision:
    """Pass, warn or refuse ``vintage`` at calendar day ``asof``; no clock, no database.

    ``max_verify_age=None`` reads :func:`max_verify_age_days`. A vintage verified after ``asof``
    has a negative age and never trips the age check. The code is the first refusal found
    (stale before lag); ``reasons`` lists all that fired.
    """
    day = as_date(asof)
    limit = max_verify_age_days() if max_verify_age is None else max_verify_age
    verified_age = age_since_verified(vintage, day)
    lag = (day - vintage.coverage_end).days
    refuse: list[tuple[str, str]] = []
    warn: list[str] = []
    if verified_age > limit:
        refuse.append(
            (
                CODE_STALE,
                f"the S&P 500 list was last verified against upstream {verified_age} days before "
                f"{day} (limit {limit}); {CHECK_FIX}",
            )
        )
    if lag >= lag_refuse:
        refuse.append(
            (
                CODE_LAG,
                f"the list's last change date is {vintage.coverage_end}, {lag} days before "
                f"{day} (refuse from {lag_refuse}); {LOAD_FIX}",
            )
        )
    elif lag >= lag_warn:
        warn.append(
            f"the list's last change date is {vintage.coverage_end}, {lag} days before "
            f"{day} (warn from {lag_warn}, refuse from {lag_refuse}); {LOAD_FIX}"
        )
    level: Level = "refuse" if refuse else "warn" if warn else "pass"
    return GateDecision(
        level=level,
        code=refuse[0][0] if refuse else None,
        asof=day,
        age_since_verified_days=verified_age,
        lag_days=lag,
        max_verify_age_days=limit,
        lag_warn_days=lag_warn,
        lag_refuse_days=lag_refuse,
        reasons=tuple(text for _, text in refuse) + tuple(warn),
    )


def gate_applies(source: MembershipSource, *, wall_clock: bool) -> bool:
    """True only for a wall-clock run on the latest ``db`` vintage (Appendix A item 1).

    A source without a ``pinned`` attribute counts as unpinned, so the safe answer is to gate.
    """
    return wall_clock and source.vintage().kind == "db" and not getattr(source, "pinned", False)
