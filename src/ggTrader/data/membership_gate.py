"""The live gate on the S&P 500 list: pass, warn or refuse, from an independent match.

Spec docs/superpowers/specs/2026-11-07-membership-database-design.md, section 3.4 as revised by
Appendix C3 (the upstream feed is dead, so the list is attested against the SPDR SPY holdings
file, ``ggt db membership check-holdings``). Two clocks decide, both counted in calendar days at
``asof``:

* **Freshness**, ``asof - last matched check``: how long ago our list was last proved equal to
  the SPY holdings (``universe.checks`` row ``matched``). More than :data:`MAX_VERIFY_AGE_DAYS`
  (45 until the weekly job exists, 10 after; the environment variable
  ``GGT_MEMBERSHIP_MAX_VERIFY_AGE_DAYS`` overrides it, 1 to 90 days only) refuses. A list that
  was never matched refuses outright. A ``check-upstream`` result does not extend this clock.
* **Lag**, ``asof - known_through``: how far the date through which the list is attested trails
  the day. ``known_through`` is the holdings as-of date of the newest match, never earlier than
  the vintage's own last change date. Equal to or beyond :data:`LAG_WARN_DAYS` (90) warns; equal
  to or beyond :data:`LAG_REFUSE_DAYS` (180) refuses (inclusive, as built in M4). A quiet spell
  without index changes raises no alarm while the matches keep coming; a dead pipeline does.
  A never-matched list is measured from its last change date.

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
    age_since_matched,
)
from ggTrader.data.universe_algebra import as_date

#: Longest tolerated age of the last holdings match, in days (D1: 45 until the weekly job).
MAX_VERIFY_AGE_DAYS = 45
#: Largest value the override may take: a bigger one would quietly switch the freshness clock off.
MAX_VERIFY_AGE_CEILING_DAYS = 90
#: Environment variable that overrides :data:`MAX_VERIFY_AGE_DAYS` (a whole number of days from 1
#: to :data:`MAX_VERIFY_AGE_CEILING_DAYS`).
VERIFY_AGE_ENV = "GGT_MEMBERSHIP_MAX_VERIFY_AGE_DAYS"
#: The date the list is attested through is this many days before the run: warn (C3).
LAG_WARN_DAYS = 90
#: ... and refuse (C3).
LAG_REFUSE_DAYS = 180
#: A match may be dated at most this many calendar days after the run's day (time zones); a
#: match from further in the future is a bad row or a bad clock, never a fresh attestation.
FUTURE_TOLERANCE_DAYS = 1

#: ``SignalGateError`` codes (spec 3.4 names the first two; the lag code is the D8 counterpart).
CODE_UNAVAILABLE = "membership_unavailable"
CODE_STALE = "membership_stale"
CODE_LAG = "membership_lag"
CODE_FUTURE = "membership_future_match"
CODE_CONFIG = "membership_gate_config"

Level = Literal["pass", "warn", "refuse"]

CHECK_FIX = "run `ggt db membership check-holdings`"
LAG_FIX = (
    "run `ggt db membership check-holdings` (a match advances the date the list is attested "
    "through); if it reports a mismatch, load the missing changes with `ggt db load-sp500`"
)


class GateConfigError(MembershipError, ValueError):
    """The age-limit override is not a whole number of days in range; the gate refuses to guess."""


def max_verify_age_days(environ: Mapping[str, str] | None = None) -> int:
    """The verification-age limit in force: the environment override, else the constant."""
    text = (os.environ if environ is None else environ).get(VERIFY_AGE_ENV)
    if text is None or text == "":
        return MAX_VERIFY_AGE_DAYS
    try:
        value = int(text)
    except ValueError:
        value = 0
    if not 1 <= value <= MAX_VERIFY_AGE_CEILING_DAYS:
        raise GateConfigError(
            f"{VERIFY_AGE_ENV}={text!r} is not a whole number of days from 1 to "
            f"{MAX_VERIFY_AGE_CEILING_DAYS}; unset it to use the default of "
            f"{MAX_VERIFY_AGE_DAYS}"
        )
    return value


@dataclass(frozen=True)
class GateDecision:
    """What the gate decided at one day, with the numbers behind it.

    ``age_since_match_days`` is ``None`` when the list was never matched; ``known_through`` is
    ``None`` then too, and ``lag_days`` is measured from the last change date instead.
    """

    level: Level
    code: str | None
    asof: date
    age_since_match_days: int | None
    known_through: date | None
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
        out["known_through"] = (
            None if self.known_through is None else self.known_through.isoformat()
        )
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

    ``max_verify_age=None`` reads :func:`max_verify_age_days`. A match whose instant, or whose
    attested date, lies more than :data:`FUTURE_TOLERANCE_DAYS` day after ``asof`` refuses
    (``membership_future_match``): the gate runs on the wall clock, so such a row is a defect, not
    a replay. The code is the first refusal found (future, stale, lag); ``reasons`` lists all that
    fired.
    """
    day = as_date(asof)
    limit = max_verify_age_days() if max_verify_age is None else max_verify_age
    match_age = age_since_matched(vintage, day)
    attested = vintage.known_through
    lag = (day - (attested if attested is not None else vintage.coverage_end)).days
    refuse: list[tuple[str, str]] = []
    warn: list[str] = []
    ahead = max(
        -match_age if match_age is not None else 0,
        (attested - day).days if attested is not None else 0,
    )
    if ahead > FUTURE_TOLERANCE_DAYS:
        refuse.append(
            (
                CODE_FUTURE,
                f"the newest SPDR SPY holdings match of the S&P 500 list is dated {ahead} days "
                f"after {day} (allowed: {FUTURE_TOLERANCE_DAYS}); a match from the future "
                "cannot be trusted; inspect universe.checks and the system clock",
            )
        )
    if match_age is None:
        refuse.append(
            (
                CODE_STALE,
                "the S&P 500 list has never been matched against the SPDR SPY holdings "
                f"(last change date {vintage.coverage_end}); {CHECK_FIX}",
            )
        )
    elif match_age > limit:
        refuse.append(
            (
                CODE_STALE,
                f"the S&P 500 list was last matched against the SPDR SPY holdings {match_age} "
                f"days before {day} (limit {limit}); {CHECK_FIX}",
            )
        )
    since = (
        f"the list is attested only through {attested}"
        if attested is not None
        else f"the list's last change date is {vintage.coverage_end} and it was never matched"
    )
    if lag >= lag_refuse:
        refuse.append(
            (CODE_LAG, f"{since}, {lag} days before {day} (refuse from {lag_refuse}); {LAG_FIX}")
        )
    elif lag >= lag_warn:
        warn.append(
            f"{since}, {lag} days before {day} (warn from {lag_warn}, refuse from "
            f"{lag_refuse}); {LAG_FIX}"
        )
    level: Level = "refuse" if refuse else "warn" if warn else "pass"
    return GateDecision(
        level=level,
        code=refuse[0][0] if refuse else None,
        asof=day,
        age_since_match_days=match_age,
        known_through=attested,
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
