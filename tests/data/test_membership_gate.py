"""Fast tests of ``ggTrader.data.membership_gate``: the live gate's thresholds, as a table.

Pure: no database, no clock, no network. The boundary rows pin where each threshold bites.
Verification age is *stale above* the limit (spec 3.4); the lag warns and refuses *from* its
limits (decision D8: "warn at 90 days, refuse at 180").
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from ggTrader.data.membership import DbMembershipSource, MembershipVintage
from ggTrader.data.membership_gate import (
    CODE_LAG,
    CODE_STALE,
    LAG_REFUSE_DAYS,
    LAG_WARN_DAYS,
    MAX_VERIFY_AGE_DAYS,
    VERIFY_AGE_ENV,
    GateConfigError,
    decide_gate,
    gate_applies,
    max_verify_age_days,
)
from tests.data.membership_fakes import FakeMembershipStore, synthetic_history

ASOF = date(2026, 11, 8)


def _vintage(*, verified_days_ago: int, lag_days: int, asof: date = ASOF) -> MembershipVintage:
    verified = datetime.combine(asof - timedelta(days=verified_days_ago), datetime.min.time())
    verified = verified.replace(tzinfo=timezone.utc)
    return MembershipVintage(
        index_name="sp500",
        kind="db",
        load_id=1,
        fixture=None,
        content_sha256="0" * 64,
        raw_sha256="1" * 64,
        coverage_start=date(1996, 1, 2),
        coverage_end=asof - timedelta(days=lag_days),
        retrieved_at=verified,
        verified_at=verified,
    )


def test_the_named_constants_are_the_decided_values():
    assert MAX_VERIFY_AGE_DAYS == 45  # D1, until the weekly job exists
    assert (LAG_WARN_DAYS, LAG_REFUSE_DAYS) == (90, 180)  # D8
    assert VERIFY_AGE_ENV == "GGT_MEMBERSHIP_MAX_VERIFY_AGE_DAYS"


@pytest.mark.parametrize(
    ("verified_days_ago", "lag_days", "level", "code"),
    [
        (0, 0, "pass", None),  # fresh on both clocks
        (10, 30, "pass", None),
        (44, 0, "pass", None),
        (45, 0, "pass", None),  # exactly at the limit is not stale
        (46, 0, "refuse", CODE_STALE),
        (0, 89, "pass", None),
        (0, 90, "warn", None),  # warn from 90
        (0, 91, "warn", None),
        (0, 179, "warn", None),
        (0, 180, "refuse", CODE_LAG),  # refuse from 180
        (0, 181, "refuse", CODE_LAG),
        (46, 90, "refuse", CODE_STALE),  # a refusal beats a warning; stale is reported first
        (46, 180, "refuse", CODE_STALE),  # both refuse: one code, both reasons
        (45, 90, "warn", None),
        (-5, 0, "pass", None),  # verified after the run's day (a replay) is never stale
    ],
)
def test_the_gate_table(verified_days_ago, lag_days, level, code):
    decision = decide_gate(
        _vintage(verified_days_ago=verified_days_ago, lag_days=lag_days), ASOF, max_verify_age=45
    )
    assert (decision.level, decision.code) == (level, code)
    assert decision.age_since_verified_days == verified_days_ago
    assert decision.lag_days == lag_days
    assert bool(decision.reasons) == (level != "pass")


def test_both_clocks_over_the_line_report_both_reasons():
    decision = decide_gate(_vintage(verified_days_ago=60, lag_days=200), ASOF, max_verify_age=45)
    assert decision.level == "refuse" and decision.code == CODE_STALE
    assert len(decision.reasons) == 2
    assert "check-upstream" in decision.message and "load-sp500" in decision.message


def test_every_refusal_names_the_command_that_fixes_it():
    stale = decide_gate(_vintage(verified_days_ago=60, lag_days=0), ASOF, max_verify_age=45)
    assert "ggt db membership check-upstream" in stale.message
    lag = decide_gate(_vintage(verified_days_ago=0, lag_days=181), ASOF, max_verify_age=45)
    assert "ggt db load-sp500" in lag.message


def test_the_decision_is_plain_data_for_the_run_record():
    decision = decide_gate(_vintage(verified_days_ago=1, lag_days=100), ASOF, max_verify_age=45)
    record = decision.as_dict()
    assert record["level"] == "warn" and record["applied"] is True
    assert record["asof"] == "2026-11-08" and record["lag_days"] == 100
    assert record["max_verify_age_days"] == 45 and record["reasons"] == [record["message"]]


def test_the_real_vintage_one_would_have_been_refused_today_and_on_later_dates():
    """Vintage 1: verified 2026-06-09, last change 2026-06-02 (the numbers in the spec)."""
    verified = datetime(2026, 6, 9, tzinfo=timezone.utc)
    v1 = MembershipVintage(
        "sp500", "db", 1, None, "0" * 64, "1" * 64, date(1996, 1, 2), date(2026, 6, 2),
        verified, verified,
    )  # fmt: skip
    today = decide_gate(v1, date(2026, 11, 8), max_verify_age=45)
    assert today.level == "refuse" and today.code == CODE_STALE
    assert (today.age_since_verified_days, today.lag_days) == (152, 159)
    assert len(today.reasons) == 2  # stale (refuse) plus the lag warning
    nov29 = decide_gate(v1, date(2026, 11, 29), max_verify_age=45)
    assert nov29.lag_days == 180 and nov29.level == "refuse"
    assert decide_gate(v1, date(2026, 12, 31), max_verify_age=45).level == "refuse"
    # Even with a fresh verification the lag alone refuses from 2026-11-29 and not a day before.
    fresh = MembershipVintage(
        "sp500", "db", 1, None, "0" * 64, "1" * 64, date(1996, 1, 2), date(2026, 6, 2),
        datetime(2026, 11, 28, tzinfo=timezone.utc), datetime(2026, 11, 28, tzinfo=timezone.utc),
    )  # fmt: skip
    assert decide_gate(fresh, date(2026, 11, 28), max_verify_age=45).level == "warn"
    assert decide_gate(fresh, date(2026, 11, 29), max_verify_age=45).level == "refuse"


def test_asof_must_be_a_plain_date_not_a_timestamp():
    with pytest.raises(TypeError, match="datetime.date"):
        decide_gate(_vintage(verified_days_ago=0, lag_days=0), datetime(2026, 11, 8))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- the age limit


def test_the_default_limit_is_the_constant():
    assert max_verify_age_days({}) == 45
    assert max_verify_age_days({VERIFY_AGE_ENV: ""}) == 45


def test_the_environment_overrides_the_limit(monkeypatch):
    assert max_verify_age_days({VERIFY_AGE_ENV: "10"}) == 10
    monkeypatch.setenv(VERIFY_AGE_ENV, "60")
    stale_at_50 = _vintage(verified_days_ago=50, lag_days=0)
    assert decide_gate(stale_at_50, ASOF).level == "pass"  # reads the environment by default
    monkeypatch.setenv(VERIFY_AGE_ENV, "10")
    assert decide_gate(stale_at_50, ASOF).level == "refuse"


@pytest.mark.parametrize("bad", ["0", "-3", "abc", "4.5", " "])
def test_a_bad_override_is_an_error_not_a_silent_default(bad):
    with pytest.raises(GateConfigError, match=VERIFY_AGE_ENV):
        max_verify_age_days({VERIFY_AGE_ENV: bad})


# --------------------------------------------------------------------------- where it applies


class _Source:
    """The two things ``gate_applies`` reads: ``vintage().kind`` and an optional ``pinned``."""

    def __init__(self, kind: str, pinned: bool | None) -> None:
        self._kind = kind
        if pinned is not None:
            self.pinned = pinned

    def vintage(self) -> MembershipVintage:
        return replace(_vintage(verified_days_ago=0, lag_days=0), kind=self._kind)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kind", "pinned", "wall_clock", "applies"),
    [
        ("db", False, True, True),  # a real run on the latest vintage: gated
        ("db", None, True, True),  # no flag: gate (the safe answer)
        ("db", True, True, False),  # db:<id> pinned: a replay
        ("db", False, False, False),  # now injected: a replay or a test
        ("fixture", None, True, False),  # a frozen file is never stale
    ],
)
def test_the_gate_applies_only_to_a_wall_clock_run_on_the_latest_db_vintage(
    kind, pinned, wall_clock, applies
):
    assert gate_applies(_Source(kind, pinned), wall_clock=wall_clock) is applies


def test_the_db_source_records_whether_it_was_pinned():
    store = FakeMembershipStore()
    store.add(synthetic_history())
    assert DbMembershipSource("latest", store=store).pinned is False
    assert DbMembershipSource(1, store=store).pinned is True
