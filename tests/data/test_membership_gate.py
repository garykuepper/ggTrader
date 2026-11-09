"""Fast tests of ``ggTrader.data.membership_gate``: the live gate's thresholds, as a table.

Pure: no database, no clock, no network. Spec Appendix C3: freshness is the age of the last match
against the SPDR SPY holdings (stale *above* the limit), the lag is ``asof - known_through`` (warn
and refuse *from* 90 and 180 days), and a list never matched is refused outright.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from ggTrader.data.membership import DbMembershipSource, MembershipVintage
from ggTrader.data.membership_gate import (
    CODE_FUTURE,
    CODE_LAG,
    CODE_MISMATCH,
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
from tests.data.test_membership_holdings import (
    AS_OF,
    NOW,
    FakeStore,
    base_tickers,
    history,
    run,
)
from tests.data.test_membership_upstream import vintage_data
from tests.data.xlsx_builder import holdings_xlsx

ASOF = date(2026, 11, 8)


def _at(day: date) -> datetime:
    return datetime.combine(day, datetime.min.time()).replace(tzinfo=timezone.utc)


def _vintage(
    *,
    match_days_ago: int | None,
    lag_days: int,
    change_days_ago: int | None = None,
    asof: date = ASOF,
) -> MembershipVintage:
    """A latest ``db`` vintage matched ``match_days_ago`` days ago (None: never matched).

    ``lag_days`` is ``asof - known_through`` (or ``asof - coverage_end`` when never matched).
    ``change_days_ago`` is how long ago the list last changed; by default it equals the lag when
    never matched and ``lag + 30`` otherwise (a quiet spell behind the match).
    """
    attested = asof - timedelta(days=lag_days)
    if match_days_ago is None:
        coverage_end = attested
    else:
        coverage_end = asof - timedelta(days=change_days_ago or lag_days + 30)
    matched = None if match_days_ago is None else _at(asof - timedelta(days=match_days_ago))
    retrieved = _at(date(2026, 6, 9))
    return MembershipVintage(
        index_name="sp500",
        kind="db",
        load_id=1,
        fixture=None,
        content_sha256="0" * 64,
        raw_sha256="1" * 64,
        coverage_start=date(1996, 1, 2),
        coverage_end=coverage_end,
        retrieved_at=retrieved,
        verified_at=matched or retrieved,
        matched_at=matched,
        known_through=None if matched is None else attested,
    )


def test_the_named_constants_are_the_decided_values():
    assert MAX_VERIFY_AGE_DAYS == 45  # C3: 45 now, 10 once the weekly job runs
    assert (LAG_WARN_DAYS, LAG_REFUSE_DAYS) == (90, 180)  # C3
    assert VERIFY_AGE_ENV == "GGT_MEMBERSHIP_MAX_VERIFY_AGE_DAYS"


@pytest.mark.parametrize(
    ("match_days_ago", "lag_days", "level", "code"),
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
        (-1, 0, "pass", None),  # one day ahead is tolerated (time zones)
        (-2, 0, "refuse", CODE_FUTURE),  # two days ahead is not
        (-30, 0, "refuse", CODE_FUTURE),
        (None, 0, "refuse", CODE_STALE),  # never matched: refused, whatever the lag
        (None, 30, "refuse", CODE_STALE),
    ],
)
def test_the_gate_table(match_days_ago, lag_days, level, code):
    decision = decide_gate(
        _vintage(match_days_ago=match_days_ago, lag_days=lag_days), ASOF, max_verify_age=45
    )
    assert (decision.level, decision.code) == (level, code)
    assert decision.age_since_match_days == match_days_ago
    assert decision.lag_days == lag_days
    assert bool(decision.reasons) == (level != "pass")


@pytest.mark.parametrize(
    ("limit", "age", "level"), [(10, 9, "pass"), (10, 10, "pass"), (10, 11, "refuse")]
)
def test_the_tightened_limit_of_the_weekly_job_bites_at_ten(limit, age, level):
    decision = decide_gate(_vintage(match_days_ago=age, lag_days=0), ASOF, max_verify_age=limit)
    assert decision.level == level


def test_a_quiet_spell_does_not_alarm():
    """No index change for 120 days but a match yesterday: the lag runs from the match."""
    quiet = _vintage(match_days_ago=1, lag_days=1, change_days_ago=120)
    assert (ASOF - quiet.coverage_end).days == 120
    decision = decide_gate(quiet, ASOF, max_verify_age=45)
    assert decision.level == "pass" and decision.code is None and decision.reasons == ()
    assert (decision.age_since_match_days, decision.lag_days) == (1, 1)
    assert decision.known_through == ASOF - timedelta(days=1)


def test_a_dead_pipeline_does_alarm_even_after_a_long_quiet_spell():
    """The matches stop: the same list warns from 90 days and refuses from 180."""
    quiet = _vintage(match_days_ago=0, lag_days=0, change_days_ago=120)
    assert decide_gate(quiet, ASOF + timedelta(days=89), max_verify_age=10_000).level == "pass"
    assert decide_gate(quiet, ASOF + timedelta(days=90), max_verify_age=10_000).level == "warn"
    assert decide_gate(quiet, ASOF + timedelta(days=180), max_verify_age=10_000).code == CODE_LAG


def test_a_match_dated_in_the_future_refuses_with_a_clear_message():
    future_check = _vintage(match_days_ago=-3, lag_days=0)
    decision = decide_gate(future_check, ASOF, max_verify_age=45)
    assert decision.level == "refuse" and decision.code == CODE_FUTURE
    assert "3 days after 2026-11-08" in decision.message and "universe.checks" in decision.message


def test_an_attested_date_in_the_future_refuses_even_when_the_check_instant_is_not():
    vintage = replace(_vintage(match_days_ago=0, lag_days=0), known_through=date(2026, 11, 20))
    decision = decide_gate(vintage, ASOF, max_verify_age=45)
    assert decision.level == "refuse" and decision.code == CODE_FUTURE
    assert "12 days after" in decision.message
    at_edge = replace(vintage, known_through=ASOF + timedelta(days=1))
    assert decide_gate(at_edge, ASOF, max_verify_age=45).level == "pass"


def test_a_never_matched_list_names_the_command_that_fixes_it():
    decision = decide_gate(_vintage(match_days_ago=None, lag_days=159), ASOF, max_verify_age=45)
    assert decision.level == "refuse" and decision.code == CODE_STALE
    assert decision.age_since_match_days is None and decision.known_through is None
    assert "never been matched" in decision.message
    assert "ggt db membership check-holdings" in decision.message
    assert decision.lag_days == 159  # measured from the last change date, and it warns as well
    assert len(decision.reasons) == 2


def test_an_upstream_check_does_not_extend_either_clock():
    """`verified_at`, which an upstream check used to move, is not what the gate reads."""
    vintage = replace(
        _vintage(match_days_ago=None, lag_days=159), verified_at=_at(ASOF), retrieved_at=_at(ASOF)
    )
    decision = decide_gate(vintage, ASOF, max_verify_age=45)
    assert decision.level == "refuse" and decision.age_since_match_days is None


def test_both_clocks_over_the_line_report_both_reasons():
    decision = decide_gate(_vintage(match_days_ago=60, lag_days=200), ASOF, max_verify_age=45)
    assert decision.level == "refuse" and decision.code == CODE_STALE
    assert len(decision.reasons) == 2
    assert decision.message.count("ggt db membership check-holdings") == 2
    assert "ggt db load-sp500" in decision.message


def test_every_refusal_names_the_command_that_fixes_it():
    stale = decide_gate(_vintage(match_days_ago=60, lag_days=0), ASOF, max_verify_age=45)
    assert "ggt db membership check-holdings" in stale.message
    lag = decide_gate(_vintage(match_days_ago=0, lag_days=181), ASOF, max_verify_age=45)
    assert "ggt db membership check-holdings" in lag.message and "load-sp500" in lag.message


def test_the_decision_is_plain_data_for_the_run_record():
    decision = decide_gate(_vintage(match_days_ago=1, lag_days=100), ASOF, max_verify_age=45)
    record = decision.as_dict()
    assert record["level"] == "warn" and record["applied"] is True
    assert record["asof"] == "2026-11-08" and record["lag_days"] == 100
    assert record["known_through"] == "2026-07-31" and record["age_since_match_days"] == 1
    assert record["max_verify_age_days"] == 45 and record["reasons"] == [record["message"]]
    never = decide_gate(_vintage(match_days_ago=None, lag_days=1), ASOF, max_verify_age=45)
    assert never.as_dict()["known_through"] is None


def test_the_real_vintage_one_is_refused_today_and_a_match_would_open_the_gate():
    """Vintage 1: retrieved 2026-06-09, last change 2026-06-02, never matched (today)."""
    retrieved = _at(date(2026, 6, 9))
    v1 = MembershipVintage(
        "sp500", "db", 1, None, "0" * 64, "1" * 64, date(1996, 1, 2), date(2026, 6, 2),
        retrieved, retrieved,
    )  # fmt: skip
    today = decide_gate(v1, date(2026, 11, 8), max_verify_age=45)
    assert today.level == "refuse" and today.code == CODE_STALE
    assert (today.age_since_match_days, today.lag_days) == (None, 159)
    assert len(today.reasons) == 2  # never matched (refuse) plus the lag warning
    assert decide_gate(v1, date(2026, 11, 29), max_verify_age=45).code == CODE_STALE
    # If the SPY file of 2026-11-05 had matched on 2026-11-08, the gate would be open.
    matched = replace(v1, matched_at=_at(date(2026, 11, 8)), known_through=date(2026, 11, 5))
    opened = decide_gate(matched, date(2026, 11, 8), max_verify_age=45)
    assert opened.level == "pass" and (opened.age_since_match_days, opened.lag_days) == (0, 3)
    # ... and it closes again on its own: a match 46 days old refuses, the lag warns from 90.
    assert decide_gate(matched, date(2026, 12, 24), max_verify_age=45).code == CODE_STALE
    assert decide_gate(matched, date(2027, 2, 3), max_verify_age=10_000).level == "warn"


def test_asof_must_be_a_plain_date_not_a_timestamp():
    with pytest.raises(TypeError, match="datetime.date"):
        decide_gate(_vintage(match_days_ago=0, lag_days=0), datetime(2026, 11, 8))  # type: ignore[arg-type]


# --------------------------------------------------------------------------- the age limit


def test_the_default_limit_is_the_constant():
    assert max_verify_age_days({}) == 45
    assert max_verify_age_days({VERIFY_AGE_ENV: ""}) == 45


def test_the_environment_overrides_the_limit(monkeypatch):
    assert max_verify_age_days({VERIFY_AGE_ENV: "10"}) == 10
    monkeypatch.setenv(VERIFY_AGE_ENV, "60")
    stale_at_50 = _vintage(match_days_ago=50, lag_days=0)
    assert decide_gate(stale_at_50, ASOF).level == "pass"  # reads the environment by default
    monkeypatch.setenv(VERIFY_AGE_ENV, "10")
    assert decide_gate(stale_at_50, ASOF).level == "refuse"


def test_the_override_accepts_exactly_one_to_ninety_days():
    assert max_verify_age_days({VERIFY_AGE_ENV: "1"}) == 1
    assert max_verify_age_days({VERIFY_AGE_ENV: "90"}) == 90


@pytest.mark.parametrize("bad", ["0", "-3", "abc", "4.5", " ", "91", "99999"])
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
        return replace(_vintage(match_days_ago=0, lag_days=0), kind=self._kind)  # type: ignore[arg-type]


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


# --------------------------------------------------------------------------- from the store


def test_the_db_source_derives_known_through_from_the_newest_match():
    """known_through is the matched as-of date, never earlier than the vintage's coverage end."""
    history = synthetic_history()
    coverage_end = max(history)
    store = FakeMembershipStore()
    store.add(history, matched_at=_at(coverage_end + timedelta(days=10)),
              matched_as_of=coverage_end + timedelta(days=9))  # fmt: skip
    vintage = DbMembershipSource("latest", store=store).vintage()
    assert vintage.known_through == coverage_end + timedelta(days=9)
    assert vintage.matched_at == _at(coverage_end + timedelta(days=10))
    early = FakeMembershipStore()
    early.add(history, matched_at=_at(coverage_end + timedelta(days=1)),
              matched_as_of=coverage_end - timedelta(days=5))  # fmt: skip
    assert DbMembershipSource("latest", store=early).vintage().known_through == coverage_end


def test_a_db_vintage_without_a_match_has_no_known_through():
    store = FakeMembershipStore()
    store.add(synthetic_history())
    vintage = DbMembershipSource("latest", store=store).vintage()
    assert vintage.matched_at is None and vintage.known_through is None


# --------------------------------------------------------------------------- end to end


def test_a_recorded_match_opens_the_gate_and_a_mismatch_records_none():
    """From the runner's result to the gate: only the matched row carries a load id and a date."""
    ours = base_tickers()
    latest = vintage_data(history(ours))
    recorder = FakeStore(latest)
    matched = run(recorder, holdings_xlsx(ours)).result
    mismatch = run(recorder, holdings_xlsx(ours[:-1] + ["NEWCO"])).result
    assert (matched.outcome, matched.matched_load_id) == ("matched", 1)
    assert (mismatch.outcome, mismatch.matched_load_id) == ("mismatch", None)

    as_dict = {d: tuple(t) for d, t in latest.history().items()}
    retrieved = datetime(2026, 6, 9, tzinfo=timezone.utc)
    never = FakeMembershipStore()
    never.add(as_dict, retrieved_at=retrieved)
    assert decide_gate(DbMembershipSource("latest", store=never).vintage(), ASOF).level == "refuse"
    after = FakeMembershipStore()
    after.add(as_dict, retrieved_at=retrieved, matched_at=NOW, matched_as_of=matched.holdings_as_of)
    vintage = DbMembershipSource("latest", store=after).vintage()
    assert vintage.known_through == AS_OF
    assert decide_gate(vintage, ASOF).level == "pass"


def test_the_vintage_dataclass_defaults_keep_old_callers_working():
    v = MembershipVintage(
        "sp500", "fixture", None, "x", "0" * 64, "1" * 64, date(2020, 1, 1), date(2020, 2, 1),
        NOW, NOW,
    )  # fmt: skip
    assert v.matched_at is None and v.known_through is None


# --------------------------------------------------------------------------- a known mismatch


def _checked(outcome: str | None, as_of_days_ago: int = 0, check_id: int | None = 12):
    base = _vintage(match_days_ago=0, lag_days=0)
    if outcome is None:
        return base
    return replace(
        base,
        latest_check=outcome,
        latest_check_as_of=ASOF - timedelta(days=as_of_days_ago),
        latest_check_id=check_id,
    )


@pytest.mark.parametrize(
    ("outcome", "level", "code"),
    [
        (None, "pass", None),  # no holdings check recorded beyond the match itself
        ("matched", "pass", None),  # a match is the newest: open
        ("mismatch", "refuse", CODE_MISMATCH),  # a match, then a newer mismatch: closed
    ],
)
def test_the_newest_holdings_check_decides_a_match_then_a_mismatch_closes_the_gate(
    outcome, level, code
):
    decision = decide_gate(_checked(outcome), ASOF, max_verify_age=45)
    assert (decision.level, decision.code) == (level, code)


def test_a_mismatch_refusal_names_the_check_and_the_human_steps():
    decision = decide_gate(_checked("mismatch", as_of_days_ago=3), ASOF, max_verify_age=45)
    text = decision.message
    assert "check 12" in text and "2026-11-05" in text
    for needle in ("membership weekly", "membership proposal", "load-sp500", "--commit"):
        assert needle in text
    assert decision.age_since_match_days == 0  # the match itself is fresh: only the mismatch closes


def test_a_mismatch_with_no_match_at_all_reports_both_reasons_and_mismatch_leads():
    never = replace(_vintage(match_days_ago=None, lag_days=159), latest_check="mismatch")
    decision = decide_gate(never, ASOF, max_verify_age=45)
    assert decision.code == CODE_MISMATCH and len(decision.reasons) >= 2


def test_the_status_command_prints_the_newest_comparison(monkeypatch, capsys):
    from ggTrader.cli import cmd_membership
    from ggTrader.data.membership import DbMembershipSource
    from tests.data.membership_fakes import FakeMembershipStore, synthetic_history

    store = FakeMembershipStore()
    store.add(
        synthetic_history(3),
        matched_at=_at(ASOF),
        matched_as_of=ASOF,
        latest_check="mismatch",
        latest_check_as_of=ASOF,
        latest_check_id=9,
    )
    monkeypatch.setattr(
        cmd_membership, "DbMembershipSource", lambda which: DbMembershipSource(which, store=store)
    )
    monkeypatch.setattr(cmd_membership, "recent_checks", lambda *a, **k: [])
    args = type("A", (), {"asof": ASOF})()
    assert cmd_membership.run_status(args) == 0
    out = capsys.readouterr().out
    assert "newest comparison  mismatch as of 2026-11-08 (check 9)" in out
    assert "REFUSE" in out and "membership weekly" in out
