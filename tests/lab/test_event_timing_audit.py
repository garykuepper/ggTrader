"""Unit tests for Event-Timing Audit Tool.

Covers:
- Clock: regular close, after-close to next close, 30m buffer, early close,
  DST, date-only no double buffer.
- Quarantine codes: pre-1990, tx > filing, future-dated, unparseable, timestamp conflict.
- Sampler determinism: SHA-256(20261021|key), censusing small strata, shortfall reporting,
  smoke cap.
- A6 Ratio arithmetic: inclusion of quarantined/censored in denominators, missingness bounds.
- Status ordering: §5 branches and eligible window resolution (tie-breaking to most recent).

Uses fixtures only: no network, no database.
"""

from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from ggTrader.lab.event_timing_audit import (
    AnnualA6Coverage,
    AuditClock,
    QuarantineReason,
    check_earnings_quarantine,
    check_transaction_quarantine,
    compute_deterministic_sample_hash,
    evaluate_eligible_window,
    sample_stratum,
)

# ============================================================================
# 1. Clock Tests
# ============================================================================


class TestAuditClock:
    @pytest.fixture
    def clock(self) -> AuditClock:
        return AuditClock("NYSE")

    def test_more_than_30m_before_close_trades_same_day(self, clock: AuditClock):
        # 2023-05-04 (Thursday): normal close at 16:00 EDT
        # Filing available at 15:15:00 EDT -> cutoff is 15:45:00 EDT <= 16:00:00 EDT
        avail = pd.Timestamp("2023-05-04 15:15:00", tz="America/New_York")
        t_date, t_close, prec = clock.tradeable_close_for_timestamp(avail)
        assert t_date == datetime.date(2023, 5, 4)
        assert t_close.hour == 16 and t_close.minute == 0
        assert prec == "timestamp"

    def test_within_30m_of_close_trades_next_session_close(self, clock: AuditClock):
        # Filing available at 15:45:00 EDT -> cutoff is 16:15:00 EDT > 16:00:00 EDT
        # Must trade at next session's close: 2023-05-05 16:00:00 EDT
        avail = pd.Timestamp("2023-05-04 15:45:00", tz="America/New_York")
        t_date, t_close, prec = clock.tradeable_close_for_timestamp(avail)
        assert t_date == datetime.date(2023, 5, 5)
        assert t_close.hour == 16 and t_close.minute == 0
        assert prec == "timestamp"

    def test_after_close_trades_next_session_close(self, clock: AuditClock):
        # Filing available at 16:30:00 EDT -> cutoff is 17:00:00 EDT
        # Must trade at next session's close: 2023-05-05 16:00:00 EDT
        avail = pd.Timestamp("2023-05-04 16:30:00", tz="America/New_York")
        t_date, t_close, prec = clock.tradeable_close_for_timestamp(avail)
        assert t_date == datetime.date(2023, 5, 5)
        assert t_close.hour == 16 and t_close.minute == 0
        assert prec == "timestamp"

    def test_friday_after_close_trades_monday_close(self, clock: AuditClock):
        # 2023-05-05 (Friday) after close -> next session is Monday 2023-05-08
        avail = pd.Timestamp("2023-05-05 17:00:00", tz="America/New_York")
        t_date, t_close, prec = clock.tradeable_close_for_timestamp(avail)
        assert t_date == datetime.date(2023, 5, 8)
        assert t_close.hour == 16 and t_close.minute == 0

    def test_early_close_black_friday(self, clock: AuditClock):
        # 2023-11-24 (Black Friday): early close at 13:00 EST
        # Case A: Available at 12:15 EST -> cutoff 12:45 EST <= 13:00 EST -> trades same day
        avail_a = pd.Timestamp("2023-11-24 12:15:00", tz="America/New_York")
        t_date_a, t_close_a, _ = clock.tradeable_close_for_timestamp(avail_a)
        assert t_date_a == datetime.date(2023, 11, 24)
        assert t_close_a.hour == 13 and t_close_a.minute == 0

        # Case B: Available at 12:45 EST -> cutoff 13:15 EST > 13:00 EST -> trades Monday 11-27
        avail_b = pd.Timestamp("2023-11-24 12:45:00", tz="America/New_York")
        t_date_b, t_close_b, _ = clock.tradeable_close_for_timestamp(avail_b)
        assert t_date_b == datetime.date(2023, 11, 27)
        assert t_close_b.hour == 16 and t_close_b.minute == 0

    def test_date_only_no_double_buffer(self, clock: AuditClock):
        # Date-only fallback per §8 A2/A3: first session AFTER date. Buffer is NOT applied.
        event_d = datetime.date(2023, 5, 4)  # Thursday
        t_date, t_close, prec = clock.tradeable_close_for_date_only(event_d)
        assert t_date == datetime.date(2023, 5, 5)
        assert prec == "date_only"

        # Friday date-only -> Monday close
        event_fri = datetime.date(2023, 5, 5)
        t_date_f, t_close_f, prec_f = clock.tradeable_close_for_date_only(event_fri)
        assert t_date_f == datetime.date(2023, 5, 8)
        assert prec_f == "date_only"

    def test_dst_transition(self, clock: AuditClock):
        # Summer (EDT, UTC-4) vs Winter (EST, UTC-5)
        # Summer: 2023-06-15
        sched_summer = clock.get_schedule("2023-06-15", "2023-06-15")
        close_summer = sched_summer.iloc[0]["market_close"]
        assert close_summer.tzname() == "EDT"
        assert close_summer.hour == 16

        # Winter: 2023-01-12
        sched_winter = clock.get_schedule("2023-01-12", "2023-01-12")
        close_winter = sched_winter.iloc[0]["market_close"]
        assert close_winter.tzname() == "EST"
        assert close_winter.hour == 16

    def test_tradeable_open_diagnostic(self, clock: AuditClock):
        # Pre-market availability at 08:30:00 EDT -> cutoff 09:00:00 EDT <= 09:30:00 EDT open
        avail = pd.Timestamp("2023-05-04 08:30:00", tz="America/New_York")
        t_date, t_open, prec = clock.tradeable_open_for_timestamp(avail)
        assert t_date == datetime.date(2023, 5, 4)
        assert t_open.hour == 9 and t_open.minute == 30
        assert prec == "timestamp"

        # Availability at 09:15:00 EDT -> cutoff 09:45:00 EDT > 09:30:00 EDT -> next session open
        avail2 = pd.Timestamp("2023-05-04 09:15:00", tz="America/New_York")
        t_date2, t_open2, _ = clock.tradeable_open_for_timestamp(avail2)
        assert t_date2 == datetime.date(2023, 5, 5)
        assert t_open2.hour == 9 and t_open2.minute == 30


# ============================================================================
# 2. Quarantine Reason Codes Tests
# ============================================================================


class TestQuarantineRules:
    run_date = datetime.date(2026, 10, 21)

    def test_tx_date_pre_1990(self):
        res = check_transaction_quarantine(
            transaction_date="1989-12-31",
            filing_date="2020-01-01",
            query_run_date=self.run_date,
        )
        assert res == QuarantineReason.TX_DATE_PRE_1990

    def test_tx_date_post_filing(self):
        res = check_transaction_quarantine(
            transaction_date="2023-06-15",
            filing_date="2023-06-10",
            query_run_date=self.run_date,
        )
        assert res == QuarantineReason.TX_DATE_POST_FILING

    def test_future_dated(self):
        res = check_transaction_quarantine(
            transaction_date="2030-01-01",
            filing_date="2030-01-02",
            query_run_date=self.run_date,
        )
        assert res == QuarantineReason.FUTURE_DATED

    def test_unparseable_or_missing_date(self):
        res = check_transaction_quarantine(
            transaction_date=None,
            filing_date="2023-01-01",
            query_run_date=self.run_date,
        )
        assert res == QuarantineReason.UNPARSEABLE_OR_MISSING_DATE

        res2 = check_transaction_quarantine(
            transaction_date="invalid-date",
            filing_date="2023-01-01",
            query_run_date=self.run_date,
        )
        assert res2 == QuarantineReason.UNPARSEABLE_OR_MISSING_DATE

    def test_timestamp_conflict(self):
        res = check_transaction_quarantine(
            transaction_date="2023-01-01",
            filing_date="2023-01-02",
            query_run_date=self.run_date,
            timestamp_conflict=True,
        )
        assert res == QuarantineReason.UNRESOLVED_TIMESTAMP_CONFLICT

    def test_clean_transaction(self):
        res = check_transaction_quarantine(
            transaction_date="2023-05-01",
            filing_date="2023-05-03",
            query_run_date=self.run_date,
        )
        assert res is None

    def test_earnings_quarantine(self):
        assert check_earnings_quarantine("1988-05-01", self.run_date) is None
        assert (
            check_earnings_quarantine("2030-01-01", self.run_date) == QuarantineReason.FUTURE_DATED
        )
        assert (
            check_earnings_quarantine(None, self.run_date)
            == QuarantineReason.UNPARSEABLE_OR_MISSING_DATE
        )
        assert check_earnings_quarantine("2023-05-04", self.run_date) is None


# ============================================================================
# 3. Sampler Determinism Tests
# ============================================================================


class TestDeterministicSampler:
    def test_sha256_hash_computation(self):
        key = "AAPL|2023-05-04"
        expected = hashlib.sha256(f"20261021|{key}".encode("utf-8")).hexdigest()
        assert compute_deterministic_sample_hash(key) == expected

    def test_sample_reproducibility(self):
        items = [f"item_{i}" for i in range(100)]
        res1, sf1 = sample_stratum(items, key_fn=lambda x: x, quota=10)
        res2, sf2 = sample_stratum(items, key_fn=lambda x: x, quota=10)
        assert res1 == res2
        assert sf1 == sf2 == 0
        assert len(res1) == 10

        # Verify ordering is ascending by SHA-256
        hashes = [compute_deterministic_sample_hash(x) for x in res1]
        assert hashes == sorted(hashes)

    def test_census_when_stratum_le_quota(self):
        items = ["A", "B", "C"]
        selected, shortfall = sample_stratum(items, key_fn=lambda x: x, quota=5)
        assert len(selected) == 3
        assert set(selected) == {"A", "B", "C"}
        assert shortfall == 2

    def test_smoke_mode_quota_cap(self):
        items = [f"item_{i}" for i in range(50)]
        selected, shortfall = sample_stratum(items, key_fn=lambda x: x, quota=25, smoke=True)
        assert len(selected) == 2
        assert shortfall == 0


# ============================================================================
# 4. A6 Ratio Arithmetic Tests
# ============================================================================


class TestA6RatioArithmetic:
    def test_quarantined_stays_in_denominator(self):
        # Suppose 100 in-scope events, 10 quarantined, 85 priced clean
        cov = AnnualA6Coverage(
            year=2023,
            denominator_units=2000,
            resolved_identity_units=1980,  # 99%
            retrieved_units=1950,  # 98.4%
            in_scope_events=100,
            quarantined_events=10,
            priced_events=85,  # 85% priced
            missing_units=10,  # 0.5%
            missing_events=1,  # 1.0%
        )
        assert cov.identity_ratio == 1980 / 2000 >= 0.95
        assert cov.retrieval_ratio == 1950 / 1980 >= 0.95
        # Price ratio is priced / in_scope_events (which includes quarantined)
        assert cov.price_ratio == 85 / 100
        # Fails A6 because price ratio < 0.95
        assert not cov.passes_a6()

    def test_all_bounds_pass(self):
        cov = AnnualA6Coverage(
            year=2023,
            denominator_units=2000,
            resolved_identity_units=1950,  # 97.5%
            retrieved_units=1900,  # 97.4%
            in_scope_events=100,
            quarantined_events=2,
            priced_events=96,  # 96%
            missing_units=20,  # 1.0%
            missing_events=1,  # 1.0%
        )
        assert cov.passes_a6()

    def test_missingness_ratio_failure(self):
        cov = AnnualA6Coverage(
            year=2023,
            denominator_units=2000,
            resolved_identity_units=1950,
            retrieved_units=1900,
            in_scope_events=100,
            quarantined_events=0,
            priced_events=98,
            missing_units=50,  # 50 / 2000 = 2.5% > 2% max
            missing_events=0,
        )
        assert cov.unit_missingness_ratio > 0.02
        assert not cov.passes_a6()


# ============================================================================
# 5. Status Ordering & Eligible Window Tests (§5, §8 A7)
# ============================================================================


class TestStatusOrderingAndEligibleWindow:
    def test_eligible_window_contiguous_5_years(self):
        # 2015..2020 pass, 2021 fails, 2022..2025 pass (4 years)
        annual = {
            2015: True,
            2016: True,
            2017: True,
            2018: True,
            2019: True,
            2020: True,
            2021: False,
            2022: True,
            2023: True,
            2024: True,
            2025: True,
        }
        win, length = evaluate_eligible_window(annual, min_years=5)
        assert win == (2015, 2020)
        assert length == 6

    def test_eligible_window_tie_breaks_to_most_recent(self):
        # Two 5-year runs: 2010-2014 and 2018-2022
        annual = {
            2010: True,
            2011: True,
            2012: True,
            2013: True,
            2014: True,
            2015: False,
            2016: False,
            2017: False,
            2018: True,
            2019: True,
            2020: True,
            2021: True,
            2022: True,
            2023: False,
        }
        win, length = evaluate_eligible_window(annual, min_years=5)
        assert win == (2018, 2022)
        assert length == 5

    def test_eligible_window_none_when_under_5_years(self):
        annual = {2018: True, 2019: True, 2020: True, 2021: True, 2022: False}
        win, length = evaluate_eligible_window(annual, min_years=5)
        assert win is None
        assert length == 0


def test_eligible_window_does_not_bridge_a_missing_year():
    assert evaluate_eligible_window(dict.fromkeys([2010, 2011, 2013, 2014, 2015], True)) == (
        None,
        0,
    )


def test_eligible_window_rejects_partial_2026():
    assert evaluate_eligible_window(dict.fromkeys(range(2022, 2027), True)) == (None, 0)


def test_missing_filing_date_is_quarantined():
    assert check_transaction_quarantine("2023-01-01", None, datetime.date(2026, 10, 21)) == (
        QuarantineReason.UNPARSEABLE_OR_MISSING_DATE
    )


def test_membership_in_force_counts_unchanged_months(tmp_path):
    from ggTrader.lab.event_timing_audit import build_universe_denominators

    membership = tmp_path / "members.csv"
    membership.write_text('date,tickers\n2000-01-01,"A,B"\n2025-03-15,"B,C"\n')
    result = build_universe_denominators(membership, tmp_path / "absent")
    assert result.insider_member_months_detail["2025-01"] == 2
    assert result.insider_member_months_detail["2025-03"] == 3
    assert result.pead_member_quarters_detail["2025Q1"] == 3
    assert result.insider_member_months_detail["2025-04"] == 2


def test_bad_checkpoint_fails_closed(tmp_path):
    from ggTrader.lab.event_timing_audit import SecEdgarClient

    (tmp_path / "checkpoint.json").write_text("broken")
    with pytest.raises(ValueError, match="checkpoint"):
        SecEdgarClient(cache_dir=tmp_path)


def test_no_detector_evidence_does_not_mean_clean(tmp_path):
    from ggTrader.lab.event_timing_audit import load_existing_artifact_flags

    assert load_existing_artifact_flags(tmp_path / "absent") is None


@pytest.fixture
def raw_cache(tmp_path):
    import json

    from ggTrader.lab.event_timing_sources import CheckpointState, SecEdgarClient, asdict, digest

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "checkpoint.json").write_text(json.dumps(asdict(CheckpointState())))
    (cache / "MANIFEST.json").write_text('{"files": []}')
    client = SecEdgarClient(cache)

    def add(raw, url="https://www.sec.gov/Archives/evidence.json"):
        data = raw.encode() if isinstance(raw, str) else raw
        name = digest(data) + ".raw"
        (cache / name).write_bytes(data)
        ref = {
            "url": url,
            "sha256": digest(data),
            "cache_file": name,
            "status_code": 200,
            "timestamp": "2026-10-21T00:00:00+00:00",
        }
        client.manifest["files"].append(ref)
        provider = "yf" if "finance.yahoo.com" in url else "sec"
        counter = f"{provider}_requests_total"
        setattr(client.checkpoint, counter, getattr(client.checkpoint, counter) + 1)
        client.checkpoint.request_log.append(
            {
                "url": url,
                "provider": provider,
                "attempt": 1,
                "outcome": "success",
                "sha256": ref["sha256"],
            }
        )
        client.save_checkpoint()
        (cache / "MANIFEST.json").write_text(json.dumps(client.manifest))
        return ref

    return cache, client, add


def reviewed_fixture(record, kind, add, fields):
    """Attach field-specific review receipts to explicit synthetic source facts."""
    import json

    from ggTrader.lab.event_timing_audit import semantic_linkage

    def facts(value):
        if isinstance(value, dict):
            return {
                k: facts(v) for k, v in value.items() if k not in ("field_reviews", "source_refs")
            }
        return [facts(v) for v in value] if isinstance(value, list) else value

    text = json.dumps(facts(record), sort_keys=True, default=str)
    ref = add(
        text,
        f"https://www.sec.gov/Archives/synthetic-{hashlib.sha256(text.encode()).hexdigest()}.json",
    )
    record["source_refs"] = [ref]
    if kind == "routes":
        for doc in record.get("documentation", {}).values():
            doc["citation_url"] = ref["url"]
    scope = semantic_linkage(kind, record)
    record["field_reviews"] = []
    for name in fields:
        value = record
        for part in name.split("."):
            value = value[part]
        record["field_reviews"].append(
            {
                "field": name,
                "value": value,
                "linkage": scope,
                "source_ref": ref,
                "byte_start": 0,
                "byte_end": len(text.encode()),
                "extract": text,
                "reviewer": "synthetic-fixture",
                "reviewed_at": "2026-10-21T00:00:00Z",
                "interpretation": "Synthetic document explicitly states this scoped field.",
            }
        )
    return record


def reviewed_route(record, add):
    """Supply explicit product/representation review context for synthetic route tests."""
    record.update(product="Synthetic product", representation="all permitted uses")
    fields = [
        f"documentation.{d}.{f}"
        for d in record.get("documentation", {})
        for f in ("assessment", "statement")
    ]
    fields.extend(
        k
        for k in (
            "history_start",
            "history_end",
            "requirements",
            "includes_delisted_acquired",
            "all_permitted_representations_ruled_out",
        )
        if k in record
    )
    return reviewed_fixture(record, "routes", add, fields)


def test_within_minute_uses_latest_same_field_and_acceptance_is_separate():
    from ggTrader.lab.event_timing_audit import event_clock

    event = event_clock(
        {
            "earnings_date": "2023-05-04",
            "announcement_readings": ["2023-05-04T15:29:30-04:00", "2023-05-04T15:30:30-04:00"],
            "acceptance": "2023-05-04T16:10:00-04:00",
        },
        "pead",
        AuditClock(),
        datetime.date(2026, 10, 21),
    )
    assert event["quarantine"] is None
    assert str(event["announcement_time"]) == "2023-05-04 15:30:30-04:00"
    assert event["availability"] == "2023-05-04T16:10:00-04:00"
    assert event["tradeable_date"] == "2023-05-05"


def test_conflicting_acceptance_quarantines_and_never_assigns_execution():
    from ggTrader.lab.event_timing_audit import event_clock

    event = event_clock(
        {
            "filing_date": "2023-05-04",
            "acceptance": "2023-05-04T15:00:00-04:00",
            "header_acceptance": "2023-05-04T15:02:00-04:00",
        },
        "insider_cluster",
        AuditClock(),
        datetime.date(2026, 10, 21),
    )
    assert event["quarantine"] == QuarantineReason.UNRESOLVED_TIMESTAMP_CONFLICT
    assert event["tradeable_date"] is None


def test_cached_bytes_require_matching_manifest(raw_cache):
    cache, client, add = raw_cache
    ref = add("immutable")
    assert client.fetch_sec_url(ref["url"]) == "immutable"
    (cache / ref["cache_file"]).write_text("partial")
    with pytest.raises(ValueError, match="checksum"):
        client.fetch_sec_url(ref["url"])
    assert (cache / ref["cache_file"]).read_text() == "partial"


def test_corrupt_manifest_and_missing_accounting_fail_closed(raw_cache):
    cache, _, add = raw_cache
    add("evidence")
    (cache / "checkpoint.json").unlink()
    from ggTrader.lab.event_timing_sources import SecEdgarClient

    with pytest.raises(ValueError, match="checkpoint"):
        SecEdgarClient(cache)
    (cache / "checkpoint.json").write_text('{"sec_requests_total":0,"yf_requests_total":0}')
    (cache / "MANIFEST.json").write_text("broken")
    with pytest.raises(ValueError, match="checkpoint"):
        SecEdgarClient(cache)


def test_retry_budget_is_persisted_before_each_attempt(raw_cache, monkeypatch):
    import json
    import urllib.error

    from ggTrader.lab.event_timing_sources import SecEdgarClient

    cache, _, _ = raw_cache
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "fixture@example.invalid")
    client = SecEdgarClient(cache, offline=False)
    monkeypatch.setattr("ggTrader.lab.event_timing_sources.time.sleep", lambda _: None)
    calls = []

    def fail(*args, **kwargs):
        calls.append(json.loads((cache / "checkpoint.json").read_text())["sec_requests_total"])
        raise urllib.error.URLError("fixture failure")

    monkeypatch.setattr(client.opener, "open", fail)
    assert client.fetch_sec_url("https://www.sec.gov/Archives/missing.xml") is None
    assert calls == [1, 2, 3, 4]
    resumed = SecEdgarClient(cache)
    assert resumed.checkpoint.sec_requests_total == 4
    assert len(resumed.checkpoint.request_log) == 4


def test_first_403_stops_sec_and_yahoo(raw_cache, monkeypatch):
    import urllib.error

    from ggTrader.lab.event_timing_sources import SecEdgarClient

    cache, _, _ = raw_cache
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "fixture@example.invalid")
    client = SecEdgarClient(cache, offline=False)
    monkeypatch.setattr("ggTrader.lab.event_timing_sources.time.sleep", lambda _: None)

    def forbidden(*args, **kwargs):
        raise urllib.error.HTTPError("https://www.sec.gov/a", 403, "fixture", {}, None)

    monkeypatch.setattr(client.opener, "open", forbidden)
    assert client.fetch_sec_url("https://www.sec.gov/a") is None
    assert client.fetch("https://finance.yahoo.com/calendar/earnings?symbol=A", "yf") is None
    assert client.checkpoint.sec_requests_total == 1
    assert client.checkpoint.yf_requests_total == 0
    assert SecEdgarClient(cache).checkpoint.stopped_due_to_rate_limit


def test_runtime_budget_accumulates(raw_cache, monkeypatch):
    import json

    from ggTrader.lab.event_timing_sources import MAX_RUNTIME, SecEdgarClient

    cache, _, _ = raw_cache
    state = json.loads((cache / "checkpoint.json").read_text())
    state["elapsed_seconds"] = MAX_RUNTIME
    (cache / "checkpoint.json").write_text(json.dumps(state))
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "fixture@example.invalid")
    client = SecEdgarClient(cache, offline=False)
    assert not client.within_deadline()
    assert client.fetch_sec_url("https://www.sec.gov/a") is None
    assert client.checkpoint.sec_requests_total == 0


def test_empty_inventory_rules_and_identity_gap():
    from ggTrader.lab.event_timing_audit import resolve_unit, unit_inventory

    unit = {
        "symbol": "A",
        "start": "2023-01-01",
        "end": "2023-01-31",
        "membership_intervals": [["2023-01-01", "2023-01-31"]],
    }
    mapping = {
        "symbol": "A",
        "valid_from": "2020-01-01",
        "valid_to": "2024-01-01",
        "cik": "1",
        "security_id": "A-common",
        "price_symbol": "A-USD",
    }
    assert resolve_unit(unit, [mapping])
    evidence = {
        "inventories": [
            {
                "cik": "1",
                "start": "2015-01-01",
                "end": "2026-10-21",
                "complete": True,
                "request_succeeded": True,
                "filings": [],
                "source_refs": [],
            }
        ],
        "pead_fetches": [
            {"symbol": "A", "request_succeeded": True, "records": [], "source_refs": []}
        ],
    }
    assert unit_inventory(unit, [mapping], "insider_cluster", evidence)["retrieved"]
    assert not unit_inventory(unit, [mapping], "pead", evidence)["retrieved"]
    mapping["valid_to"] = "2023-01-15"
    assert resolve_unit(unit, [mapping]) is None


def test_unknown_detector_cannot_pass_unique_bar():
    from ggTrader.lab.event_timing_audit import check_tape_integrity_batch

    class CountDb:
        def execute_select(self, sql, params):
            assert "count(*)" in sql
            return ["count"], [(1,)]

    event = {
        "event_id": "A:1",
        "symbol": "A",
        "tradeable_date": "2023-05-04",
        "identity_resolved": True,
        "price_symbol": "A-USD",
    }
    result = check_tape_integrity_batch(CountDb(), AuditClock(), [event], None)
    assert result["A:1"]["bar_count"] == 1
    assert result["A:1"]["predicate"] == "UNVERIFIED"


@pytest.fixture
def ownership_xml():
    return """<ownershipDocument><issuer><issuerCik>1</issuerCik>
    <issuerTradingSymbol>A</issuerTradingSymbol></issuer>
    <reportingOwner><reportingOwnerId><rptOwnerCik>10</rptOwnerCik></reportingOwnerId></reportingOwner>
    <reportingOwner><reportingOwnerId><rptOwnerCik>11</rptOwnerCik></reportingOwnerId></reportingOwner>
    <aff10b5One><value>0</value></aff10b5One><nonDerivativeTable>
    <nonDerivativeTransaction><securityTitle><value>Common A</value></securityTitle>
    <transactionDate><value>2023-05-01</value></transactionDate>
    <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>100</value></transactionShares></transactionAmounts></nonDerivativeTransaction>
    <nonDerivativeTransaction><securityTitle><value>Common B</value></securityTitle>
    <transactionDate><value>2023-05-01</value></transactionDate>
    <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>100</value></transactionShares></transactionAmounts></nonDerivativeTransaction>
    </nonDerivativeTable></ownershipDocument>"""


def test_all_owners_plan_encodings_and_legitimate_lots(ownership_xml):
    from ggTrader.lab.event_timing_reconstruction import reconcile_lots
    from ggTrader.lab.event_timing_sources import parse_form4

    filing = {"accession": "0000000001-23-000001", "filing_date": "2023-05-03", "form": "4/A"}
    parsed = parse_form4(ownership_xml, filing)
    assert [o["cik"] for o in parsed["owners"]] == ["10", "11"]
    assert parsed["plan_state"] == "PRESENT-FALSE"
    assert len(parsed["lots"]) == 2
    assert len(reconcile_lots(parsed["lots"], [])["collisions"]) == 2
    assert reconcile_lots(parsed["lots"], [])["predicate"] == "PASS"
    assert (
        parse_form4(ownership_xml.replace("<value>0</value>", "<value>1</value>"), filing)[
            "plan_state"
        ]
        == "PRESENT-TRUE"
    )
    assert (
        parse_form4(ownership_xml.replace("<aff10b5One><value>0</value></aff10b5One>", ""), filing)[
            "plan_state"
        ]
        == "ABSENT"
    )


@pytest.fixture
def filing_submission(ownership_xml):
    """Canonical synthetic Form 4 envelope, with explicit accession and issuer."""
    filing = {
        "accession": "0000000001-23-000001",
        "form": "4",
        "filing_date": "2023-05-04",
        "acceptance": "2023-05-04T19:00:00Z",
        "primary_document": "xslF345X05/form4.xml",
    }
    header = (
        "<SEC-HEADER>\nACCESSION NUMBER: 0000000001-23-000001\n"
        "<ACCEPTANCE-DATETIME>20230504150000\n<FILED-AS-OF-DATE>20230504\n"
        "ISSUER:\n\tCOMPANY DATA:\n\t\tCENTRAL INDEX KEY: 0000000001\n"
        "</SEC-HEADER>\n"
    )
    document = (
        "<DOCUMENT>\n<TYPE>4\n<SEQUENCE>1\n<FILENAME>form4.xml\n"
        f"<TEXT>\n<XML>\n{ownership_xml}\n</XML>\n</TEXT>\n</DOCUMENT>\n"
    )
    return filing, header, document


@pytest.fixture
def filing_transport(raw_cache, monkeypatch):
    """Replace only HTTP I/O; retain the real cache, budget and stop machinery."""
    import io

    _, client, add = raw_cache
    client.offline = False
    client.user_agent = "fixture fixture@example.invalid"
    monkeypatch.setattr("ggTrader.lab.event_timing_sources.time.sleep", lambda _: None)
    responses = {}
    calls = []

    def open_fixture(request, **kwargs):
        url = request.full_url
        calls.append(url)
        assert url in responses, f"unexpected transport request: {url}"
        body = responses[url]
        if isinstance(body, Exception):
            raise body
        response = io.BytesIO(body.encode())
        response.status = 200
        response.headers = {}
        return response

    monkeypatch.setattr(client.opener, "open", open_fixture)
    return client, add, responses, calls


@pytest.mark.parametrize("form", ["4", "4/A"])
def test_collect_filing_full_submission_only_once(
    filing_submission, filing_transport, monkeypatch, form
):
    from ggTrader.lab import event_timing_sources as sources

    filing, header, document = filing_submission
    client, _, responses, calls = filing_transport
    filing["form"] = form
    document = document.replace("<TYPE>4\n", f"<TYPE>{form}\n")
    # Neither a same-filename exhibit nor a same-type different file is the primary.
    unrelated = document.replace(f"<TYPE>{form}\n", "<TYPE>EX-99\n")
    unrelated += document.replace("<FILENAME>form4.xml", "<FILENAME>other.xml")
    unrelated = unrelated.replace("<issuerCik>1</issuerCik>", "<issuerCik>2</issuerCik>")
    document_url, full_url = sources.filing_urls("1", filing)
    responses[full_url] = header + unrelated + document
    parsed = []
    original = sources.parse_form4

    def parse_once(xml, item):
        parsed.append(xml)
        return original(xml, item)

    monkeypatch.setattr(sources, "parse_form4", parse_once)
    result = sources.collect_filing(client, "1", filing)
    assert result["raw_complete"]
    assert result["issuer_cik"] == "1"
    assert len(result["lots"]) == 2
    assert all(lot["form"] == form for lot in result["lots"])
    assert result["header_filing_date"] == "2023-05-04"
    assert result["header_acceptance"] == "2023-05-04T15:00:00"
    assert len(parsed) == 1
    assert calls == [full_url]
    assert client.checkpoint.sec_requests_total == 1
    assert result["source_refs"] == [client.reference(full_url)]
    assert client.reference(document_url) is None
    assert sources.collect_filing(client, "1", filing) == result
    assert calls == [full_url]


@pytest.mark.parametrize("content", ["header_only", "other_document", "no_xml"])
def test_collect_filing_full_submission_fallback(
    filing_submission, filing_transport, ownership_xml, content
):
    from ggTrader.lab.event_timing_sources import collect_filing, filing_urls

    filing, header, document = filing_submission
    client, _, responses, calls = filing_transport
    document_url, full_url = filing_urls("1", filing)
    if content == "other_document":
        header += document.replace("<FILENAME>form4.xml", "<FILENAME>other.xml")
    elif content == "no_xml":
        header += document.replace("<XML>", "").replace("</XML>", "")
    responses.update({full_url: header, document_url: ownership_xml})
    result = collect_filing(client, "1", filing)
    assert result["raw_complete"]
    assert result["header_filing_date"] == "2023-05-04"
    assert len(result["lots"]) == 2
    assert calls == [full_url, document_url]
    assert client.checkpoint.sec_requests_total == 2
    assert {r["url"] for r in result["source_refs"]} == {full_url, document_url}


def test_collect_filing_cached_primary_preserves_results_and_refs(
    filing_submission, filing_transport, ownership_xml
):
    from ggTrader.lab.event_timing_sources import collect_filing, filing_urls, parse_form4

    filing, header, document = filing_submission
    client, add, responses, calls = filing_transport
    document_url, full_url = filing_urls("1", filing)
    primary = add(ownership_xml, document_url)
    # The existing cached primary remains authoritative for raw reconstruction.
    responses[full_url] = header + document.replace("<issuerCik>1", "<issuerCik>2")
    expected = dict(
        filing,
        cik="1",
        raw_complete=True,
        source_refs=[primary],
        **parse_form4(ownership_xml, filing),
    )
    result = collect_filing(client, "1", filing)
    expected.update(
        header_acceptance="2023-05-04T15:00:00",
        header_filing_date="2023-05-04",
        source_refs=[primary, client.reference(full_url)],
    )
    assert result == expected
    assert calls == [full_url]


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate",
        "wrong_issuer",
        "wrong_accession",
        "missing_accession",
        "wrong_header_issuer",
        "unclosed_document",
        "nested_document",
        "duplicate_filename",
        "duplicate_type",
        "unclosed_xml",
        "duplicate_xml",
        "invalid_xml",
        "entity",
        "doctype",
    ],
)
def test_collect_filing_embedded_invalid_fails_closed(filing_submission, filing_transport, defect):
    from ggTrader.lab.event_timing_sources import collect_filing, filing_urls

    filing, header, document = filing_submission
    client, _, responses, calls = filing_transport
    _, full_url = filing_urls("1", filing)
    if defect == "duplicate":
        document += document
    elif defect == "wrong_issuer":
        document = document.replace("<issuerCik>1</issuerCik>", "<issuerCik>2</issuerCik>")
    elif defect == "wrong_accession":
        header = header.replace(filing["accession"], "0000000001-23-999999")
    elif defect == "missing_accession":
        header = header.replace(filing["accession"], "")
    elif defect == "wrong_header_issuer":
        header = header.replace("CENTRAL INDEX KEY: 0000000001", "CENTRAL INDEX KEY: 0000000002")
    elif defect == "unclosed_document":
        document = document.replace("</DOCUMENT>", "")
    elif defect == "nested_document":
        document = "<DOCUMENT>\n" + document + "</DOCUMENT>\n"
    elif defect == "duplicate_filename":
        document = document.replace("<FILENAME>", "<FILENAME>other.xml\n<FILENAME>")
    elif defect == "duplicate_type":
        document = document.replace("<TYPE>", "<TYPE>EX-99\n<TYPE>")
    elif defect == "unclosed_xml":
        document = document.replace("</XML>", "")
    elif defect == "duplicate_xml":
        document = document.replace("</XML>", "</XML>\n<XML><ownershipDocument/></XML>")
    elif defect == "invalid_xml":
        document = document.replace("</ownershipDocument>", "</broken>")
    elif defect == "entity":
        document = document.replace("<ownershipDocument>", '<!ENTITY foo "bar"><ownershipDocument>')
    elif defect == "doctype":
        document = document.replace(
            "<ownershipDocument>", "<!DOCTYPE ownershipDocument><ownershipDocument>"
        )
    responses[full_url] = header + document
    result = collect_filing(client, "1", filing)
    assert not result["raw_complete"]
    assert result.get("reason") or result.get("schema_issues")
    assert result["source_refs"] == [client.reference(full_url)]
    assert client.referenced_bytes(result["source_refs"][0]) == responses[full_url].encode()
    assert calls == [full_url]


def test_collect_filing_rejects_xml_after_text_close(
    filing_submission, filing_transport, ownership_xml
):
    from ggTrader.lab.event_timing_sources import collect_filing, filing_urls

    filing, header, document = filing_submission
    client, _, responses, calls = filing_transport
    document_url, full_url = filing_urls("1", filing)
    damaged = document.replace("</TEXT>\n", "")
    damaged = damaged.replace("<XML>\n", "</TEXT>\n<XML>\n", 1)
    damaged = damaged.replace("</ownershipDocument>", "</broken>")
    responses.update({full_url: header + damaged, document_url: ownership_xml})

    result = collect_filing(client, "1", filing)

    assert not result["raw_complete"]
    assert result.get("reason")
    assert result["source_refs"] == [client.reference(full_url)]
    assert client.referenced_bytes(result["source_refs"][0]) == responses[full_url].encode()
    assert calls == [full_url]
    assert client.checkpoint.sec_requests_total == 1


@pytest.mark.parametrize("failed_source", ["header", "primary"])
@pytest.mark.parametrize("status", [403, 429])
def test_collect_filing_transport_stop_has_no_followup(
    filing_submission, filing_transport, failed_source, status
):
    import urllib.error

    from ggTrader.lab.event_timing_sources import collect_filing, filing_urls

    filing, header, _ = filing_submission
    client, _, responses, calls = filing_transport
    primary, full_url = filing_urls("1", filing)
    failed_url = full_url if failed_source == "header" else primary
    responses[full_url] = header
    responses[failed_url] = urllib.error.HTTPError(failed_url, status, "fixture stop", {}, None)
    result = collect_filing(client, "1", filing)
    assert not result["raw_complete"]
    assert calls == ([full_url] if failed_source == "header" else [full_url, primary])
    assert client.checkpoint.sec_requests_total == len(calls)
    assert client.checkpoint.stopped_due_to_rate_limit
    assert client.reference(failed_url)["status_code"] == status
    assert failed_url in {ref["url"] for ref in result["source_refs"]}
    assert result.get("reason")


@pytest.mark.parametrize("failed_source", ["header", "primary"])
def test_collect_filing_transport_failures_are_counted(
    filing_submission, filing_transport, ownership_xml, failed_source
):
    import urllib.error

    from ggTrader.lab.event_timing_sources import collect_filing, filing_urls

    filing, header, _ = filing_submission
    client, _, responses, calls = filing_transport
    primary, full_url = filing_urls("1", filing)
    failed_url = full_url if failed_source == "header" else primary
    responses.update({full_url: header, primary: ownership_xml})
    responses[failed_url] = urllib.error.URLError("fixture failure")
    result = collect_filing(client, "1", filing)
    assert calls == (
        [full_url] * 4 + [primary] if failed_source == "header" else [full_url] + [primary] * 4
    )
    assert client.checkpoint.sec_requests_total == 5
    assert result["raw_complete"] == (failed_source == "header")
    if failed_source == "header":
        assert result.get("header_filing_date") is None


@pytest.mark.parametrize("budget", [0, 1])
def test_collect_filing_fallback_cannot_exceed_budget(filing_submission, filing_transport, budget):
    from ggTrader.lab.event_timing_sources import collect_filing, filing_urls

    filing, header, _ = filing_submission
    client, _, responses, calls = filing_transport
    _, full_url = filing_urls("1", filing)
    client.sec_budget_limit = budget
    responses[full_url] = header
    result = collect_filing(client, "1", filing)
    assert not result["raw_complete"]
    assert calls == [full_url] * budget
    assert client.checkpoint.sec_requests_total == budget


@pytest.mark.parametrize(
    "defect", [None, "missing_header", "wrong_accession", "wrong_header_issuer"]
)
def test_collect_filing_full_only_raw_revalidation_and_date_corroboration(
    amendment_runner, filing_submission, defect
):
    from ggTrader.lab.event_timing_audit import filing_date_corroboration, validate_evidence_record
    from ggTrader.lab.event_timing_sources import ReferencedSources, collect_filing

    runner = amendment_runner
    filing = runner.evidence["filings"][0]
    primary, header_ref = filing["source_refs"]
    xml = runner.client.referenced_bytes(primary).decode()
    header = runner.client.referenced_bytes(header_ref).decode()
    if defect == "missing_header":
        header = ""
    elif defect == "wrong_accession":
        header = header.replace(filing["accession"], "0000000001-23-999999")
    elif defect == "wrong_header_issuer":
        header = header.replace("CENTRAL INDEX KEY: 0000000001", "CENTRAL INDEX KEY: 0000000002")
    document = filing_submission[2]
    document = document[: document.index("<XML>")] + f"<XML>\n{xml}\n</XML>\n</TEXT>\n</DOCUMENT>\n"
    ref = runner.fixture_add(header + "\n" + document, header_ref["url"])
    # A primary exists globally but is not declared: replay must use only the full response.
    inventories = runner.evidence["inventories"]
    item = next(f for f in inventories[0]["filings"] if f["accession"] == filing["accession"])
    record = collect_filing(ReferencedSources(runner.client, [ref]), "1", item)
    assert record["source_refs"] == [ref]
    rebuilt = validate_evidence_record("filings", record, runner.client, inventories)
    corroboration = filing_date_corroboration(record, runner.client, inventories)
    if defect is None:
        assert record["raw_complete"] and rebuilt["raw_complete"]
        assert len(rebuilt["lots"]) == 2
        assert corroboration["predicate"] == "PASS"
        event = amendment_clock(runner, record)
        assert event["tradeable_date"] == "2023-05-16"
        assert event["availability"] is None
        assert event["dissemination_predicate"] == "UNVERIFIED"
    else:
        assert not record["raw_complete"]
        assert corroboration["predicate"] != "PASS"


@pytest.fixture
def cluster_lots():
    return [
        {
            "lot_id": f"f{i}:0",
            "accession": f"f{i}",
            "form": "4",
            "symbol": "A",
            "transaction_date": f"2023-05-{day:02}",
            "filing_date": "2023-05-15",
            "transaction_code": "P",
            "owners": [str(i)],
            "plan_state": "PRESENT-FALSE",
            "availability": f"2023-05-15T{10 + i:02}:00:00-04:00",
            "quarantine": None,
        }
        for i, day in enumerate([1, 9, 14])
    ]


def test_reconstruction_uses_14_days_plan_exclusion_and_public_time(cluster_lots):
    from ggTrader.lab.event_timing_reconstruction import replay_clusters

    clock = AuditClock()
    events, issues = replay_clusters(cluster_lots, [], clock.normalize_to_ny("2023-05-31 16:00"))
    assert len(events) == 1 and not issues
    assert events[0]["owners"] == ["0", "1", "2"]
    assert replay_clusters(cluster_lots, [], clock.normalize_to_ny("2023-05-15 11:00"))[0] == []
    cluster_lots[0]["plan_state"] = "PRESENT-TRUE"
    assert replay_clusters(cluster_lots, [], clock.normalize_to_ny("2023-05-31 16:00"))[0] == []


def test_prefix_months_use_exchange_closes_include_april_and_unknown_inputs(cluster_lots):
    from ggTrader.lab.event_timing_reconstruction import evaluate_prefix_invariance

    result = evaluate_prefix_invariance(cluster_lots, [], AuditClock(), {"2023-05": True})
    assert len(result["decisions"]) == 136
    assert result["decisions"][-1]["month"] == "2026-04"
    assert result["decisions"][0]["decision"].startswith("2015-01-30T16:00")
    assert next(r for r in result["decisions"] if r["month"] == "2023-05")["predicate"] == "PASS"
    assert result["annual"][2023] == "UNVERIFIED"


def test_unlinked_amendment_is_unknown_and_later_link_cannot_rewrite_past(cluster_lots):
    from ggTrader.lab.event_timing_reconstruction import replay_clusters

    clock = AuditClock()
    before = replay_clusters(cluster_lots, [], clock.normalize_to_ny("2023-05-31 16:00"))[0]
    amendment = dict(
        cluster_lots[0],
        lot_id="amend:0",
        accession="amend",
        form="4/A",
        availability="2023-06-01T10:00:00-04:00",
    )
    assert (
        replay_clusters(cluster_lots + [amendment], [], clock.normalize_to_ny("2023-05-31 16:00"))[
            0
        ]
        == before
    )
    assert any(
        "linkage unresolved" in i
        for i in replay_clusters(
            cluster_lots + [amendment], [], clock.normalize_to_ny("2023-06-30 16:00")
        )[1]
    )
    links = [{"accession": "amend", "replaces_lot_ids": ["f0:0"], "source_refs": ["fixture"]}]
    assert (
        replay_clusters(
            cluster_lots + [amendment], links, clock.normalize_to_ny("2023-05-31 16:00")
        )[0]
        == before
    )


def test_zero_evidence_runner_never_measures_pass_and_keeps_smoke_output_partial(
    tmp_path, monkeypatch
):
    from ggTrader.lab.event_timing_audit import EventTimingAuditRunner

    membership = tmp_path / "members.csv"
    membership.write_text("date,tickers\n2000-01-01,A\n")
    monkeypatch.setattr(
        "urllib.request.OpenerDirector.open", lambda *a, **k: pytest.fail("network called")
    )
    runner = EventTimingAuditRunner(
        smoke=True,
        output_file=tmp_path / "smoke.json",
        cache_dir=tmp_path / "cache",
        membership_file=membership,
    )
    report = runner.run()
    for signal in ("pead", "insider_cluster"):
        assert report[signal]["status"] == "UNRESOLVED"
        assert {r["predicate"] for r in report[signal]["requirement_matrix"]} == {"UNVERIFIED"}
    assert runner.output_file.with_suffix(".md").exists()
    assert report["audit_meta"]["sec_requests_used"] == 0
    assert not report["audit_meta"]["complete"]
    assert all(len(s["selected_keys"]) <= 2 for s in report["samples"].values())
    with pytest.raises(ValueError, match="/tmp"):
        EventTimingAuditRunner(smoke=True, output_file=Path("docs/research/final.json"))


def test_absent_plan_and_joint_owners_are_inventoried_without_new_filters(cluster_lots):
    from ggTrader.lab.event_timing_reconstruction import replay_clusters

    cluster_lots[0]["plan_state"] = "ABSENT"
    cluster_lots[0]["owners"] = ["0", "4"]
    events, issues = replay_clusters(cluster_lots, [], AuditClock().normalize_to_ny("2023-05-31"))
    assert events and not issues
    assert "f0:0" in events[0]["absent_plan_lots"]
    assert "4" in events[0]["owners"]


def test_date_only_cluster_does_not_impute_observed_availability_or_double_buffer(cluster_lots):
    from ggTrader.lab.event_timing_reconstruction import replay_clusters

    for lot in cluster_lots:
        lot["availability"] = None
        lot["decision_eligibility_time"] = "2023-05-16T16:00:00-04:00"
        lot["tradeable_close"] = "2023-05-16T16:00:00-04:00"
    events, issues = replay_clusters(
        cluster_lots, [], AuditClock().normalize_to_ny("2023-05-16 16:00")
    )
    assert len(events) == 1 and not issues
    assert events[0]["availability"] is None
    assert events[0]["precision"] == "date_only"
    assert events[0]["tradeable_close"] == "2023-05-16T16:00:00-04:00"


def test_quarantined_inputs_cannot_form_clusters(cluster_lots):
    from ggTrader.lab.event_timing_reconstruction import replay_clusters

    cluster_lots[0]["quarantine"] = QuarantineReason.TX_DATE_POST_FILING
    assert replay_clusters(cluster_lots, [], AuditClock().normalize_to_ny("2023-05-31"))[0] == []


def test_unknown_population_has_no_fabricated_event_denominator():
    from ggTrader.lab.event_timing_audit import annual_accounting

    unit = {
        "year": 2023,
        "resolved": False,
        "retrieved": False,
        "attempted": False,
        "stored": False,
        "missing_reason": "identity_unverified",
    }
    counts, predicate = annual_accounting(2023, [unit], [], False, False)
    assert predicate == "UNVERIFIED"
    assert counts["in_scope_events"] is None
    assert counts["price_ratio"] is None
    assert counts["observed_event_lower_bound"] == 0


def test_censoring_depends_only_on_window_end_and_not_missing_bars():
    from ggTrader.lab.event_timing_audit import holding_diagnostics

    early = holding_diagnostics(
        {"tradeable_date": "2023-05-04"}, "insider_cluster", AuditClock(), set()
    )
    assert not any(r["administrative_right_censored"] for r in early)
    assert all(r["missing_observed_bars"] > 0 for r in early)
    late = holding_diagnostics(
        {"tradeable_date": "2026-04-30"}, "insider_cluster", AuditClock(), set()
    )
    assert all(r["administrative_right_censored"] for r in late)


def test_db_rejects_writes_before_connecting_and_omits_dsn(monkeypatch):
    import psycopg2

    from ggTrader.lab.event_timing_audit import ReadOnlyDb

    secret = "must-never-appear"
    monkeypatch.setenv(
        "POSTGRES_CONNECTION_STRING", f"host=localhost password={secret} dbname=fixture"
    )
    monkeypatch.setattr(
        "ggTrader.lab.event_timing_audit.psycopg2.connect",
        lambda *a, **k: (_ for _ in ()).throw(psycopg2.OperationalError(secret)),
    )
    db = ReadOnlyDb()
    for sql in (
        "DELETE FROM ohlcv",
        "SELECT 1; DELETE FROM ohlcv",
        "SELECT * INTO temp FROM ohlcv",
    ):
        with pytest.raises(ValueError, match="SELECT"):
            db.execute_select(sql)
    with pytest.raises(RuntimeError, match="unavailable") as error:
        db.execute_select("SELECT count(*) FROM ohlcv")
    assert secret not in str(error.value)
    assert secret not in str(db.query_log)
    assert not hasattr(db, "masked_conn_str")


def test_yahoo_collection_uses_installed_calendar_schema_and_keeps_no_pit_claim(raw_cache):
    from ggTrader.lab.event_timing_sources import collect_yahoo

    _, client, add = raw_cache
    html = """<table><thead><tr><th>Symbol</th><th>Company</th><th>Earnings Date</th>
    <th>EPS Estimate</th><th>Reported EPS</th><th>Surprise (%)</th></tr></thead>
    <tbody><tr><td>A</td><td>Fixture</td><td>May 04, 2023 at 4 PM EDT</td>
    <td>1</td><td>2</td><td>100</td></tr></tbody></table>"""
    add(html, "https://finance.yahoo.com/calendar/earnings?symbol=A&offset=0&size=100")
    result = collect_yahoo(client, "A")
    assert result["complete"] and result["request_succeeded"]
    assert result["records"][0]["announcement_readings"] == ["2023-05-04T16:00:00-04:00"]
    assert "consensus" not in result["records"][0]


def test_sec_pagination_keeps_amendments_and_partial_inventory_is_not_complete(raw_cache):
    import json

    from ggTrader.lab.event_timing_sources import enumerate_sec

    _, client, add = raw_cache
    block = {
        "accessionNumber": ["0000000001-23-000001"],
        "form": ["4/A"],
        "filingDate": ["2023-05-04"],
        "primaryDocument": ["form4.xml"],
        "acceptanceDateTime": ["2023-05-04T15:00:00-04:00"],
    }
    add(
        json.dumps(
            {
                "cik": "1",
                "filings": {
                    "recent": block,
                    "files": [
                        {
                            "name": "CIK1-old.json",
                            "filingFrom": "2000-01-01",
                            "filingTo": "2022-12-31",
                        }
                    ],
                },
            }
        ),
        "https://data.sec.gov/submissions/CIK0000000001.json",
    )
    result = enumerate_sec(client, "1")
    assert result["filings"][0]["form"] == "4/A"
    assert result["request_succeeded"] and not result["complete"]
    assert result["start"] == "2023-05-04"


def test_status_requires_full_matrix_and_vendor_fields_cannot_repair_coverage(raw_cache):
    from ggTrader.lab.event_timing_audit import resolve_signal_status

    _, client, add = raw_cache
    ref = add("Fixture documentation")
    matrix = [
        {"year": year, "requirement": requirement, "predicate": "PASS"}
        for year in range(2020, 2025)
        for requirement in ("coverage", "timestamp", "tape", "actual_eps", "consensus")
    ]
    assert resolve_signal_status("pead", matrix, [], client) == "FREE—VERIFIED"
    for row in matrix:
        if row["requirement"] in ("consensus", "coverage"):
            row["predicate"] = "UNVERIFIED"
    route = {
        "route": "R1",
        "source_refs": [ref],
        "requirements": ["consensus"],
        "history_start": "2010-01-01",
        "history_end": "2025-12-31",
        "includes_delisted_acquired": True,
        "documentation": {
            item: {
                "assessment": "supports",
                "citation_url": ref["url"],
                "statement": "fixture",
                "retrieved_date": "2026-10-21",
            }
            for item in ("fields", "history", "point_in_time", "access")
        },
    }
    reviewed_route(route, add)
    assert resolve_signal_status("pead", matrix, [route], client) == "UNRESOLVED"
    for row in matrix:
        if row["requirement"] == "coverage":
            row["predicate"] = "PASS"
    assert resolve_signal_status("pead", matrix, [route], client) == "UNRESOLVED"
    excluded = []
    for name in ("R2", "R3", "R4", "R5", "R6", "R7"):
        exclusion = dict(route, route=name, all_permitted_representations_ruled_out=True)
        exclusion["documentation"] = {
            item: dict(value, assessment="contradicts" if item == "access" else "supports")
            for item, value in route["documentation"].items()
        }
        reviewed_route(exclusion, add)
        excluded.append(exclusion)
    assert (
        resolve_signal_status("pead", matrix, [route, *excluded], client) == "PAID—DOCUMENTED PATH"
    )
    assert resolve_signal_status("pead", matrix, [route], client, smoke=True) == "UNRESOLVED"


def test_r8_exact_timestamp_exclusion_does_not_exclude_date_only_representation(raw_cache):
    from ggTrader.lab.event_timing_audit import assess_source_routes

    _, client, add = raw_cache
    ref = add("date-only fields")
    record = {
        "route": "R8",
        "source_refs": [ref],
        "documentation": {
            item: {
                "assessment": "contradicts" if item == "fields" else "supports",
                "citation_url": ref["url"],
                "retrieved_date": "2026-10-21",
                "statement": "fixture",
            }
            for item in ("fields", "history", "point_in_time", "access")
        },
    }
    reviewed_route(record, add)
    result = assess_source_routes("insider_cluster", [], [record], client)
    assert next(r for r in result["routes"] if r["route"] == "R8")["predicate"] == "UNVERIFIED"


@pytest.fixture
def reconstructed_runner(raw_cache, tmp_path, ownership_xml):
    import json

    from ggTrader.lab.event_timing_audit import EventTimingAuditRunner
    from ggTrader.lab.event_timing_sources import enumerate_sec, filing_urls, parse_form4

    cache, client, add = raw_cache
    base = add("Synthetic historical facts")
    filings = []
    inventory = []
    for i in range(3):
        accession = f"0000000001-23-{i + 1:06}"
        item = {
            "accession": accession,
            "form": "4",
            "filing_date": "2023-05-15",
            "acceptance": f"2023-05-15T{10 + i}:00:00-04:00",
            "primary_document": "form4.xml",
        }
        inventory.append(item)
        xml = ownership_xml.replace(
            "<reportingOwner><reportingOwnerId><rptOwnerCik>11</rptOwnerCik></reportingOwnerId></reportingOwner>",
            "",
        )
        xml = xml.replace("<rptOwnerCik>10</rptOwnerCik>", f"<rptOwnerCik>{10 + i}</rptOwnerCik>")
        document_url, header_url = filing_urls("1", item)
        ref = add(xml, document_url)
        header = add(
            f"ACCESSION NUMBER: {accession}\n<ACCEPTANCE-DATETIME>20230515{10 + i:02}0000"
            "\n<FILED-AS-OF-DATE>20230515",
            header_url,
        )
        filings.append(
            {
                **item,
                **parse_form4(xml, item),
                "cik": "1",
                "raw_complete": True,
                "header_acceptance": item["acceptance"],
                "source_refs": [ref, header],
            }
        )
    evidence = {
        "schema": "event-timing-evidence-v1",
        "identities": [
            {
                "symbol": "A",
                "cik": "1",
                "security_id": "A-common",
                "price_symbol": "A-USD",
                "valid_from": "2000-01-01",
                "valid_to": "2026-10-21",
                "source_refs": [base],
            }
        ],
        "inventories": [
            {
                "cik": "1",
                "complete": True,
                "request_succeeded": True,
                "start": "2000-01-01",
                "end": "2026-10-21",
                "filings": inventory,
                "source_refs": [base],
            }
        ],
        "filings": filings,
    }
    reviewed_fixture(
        evidence["identities"][0],
        "identities",
        add,
        ["symbol", "cik", "security_id", "price_symbol", "valid_from", "valid_to"],
    )
    all_filings = [
        {
            "accession": "0000000001-00-000001",
            "form": "10-K",
            "filing_date": "2000-01-01",
            "primary_document": "report.htm",
            "acceptance": None,
        },
        *inventory,
    ]
    block = {
        column: [row.get(field) for row in all_filings]
        for column, field in (
            ("accessionNumber", "accession"),
            ("form", "form"),
            ("filingDate", "filing_date"),
            ("primaryDocument", "primary_document"),
            ("acceptanceDateTime", "acceptance"),
        )
    }
    add(
        json.dumps({"cik": "1", "filings": {"recent": block, "files": []}}),
        "https://data.sec.gov/submissions/CIK0000000001.json",
    )
    evidence["inventories"] = [enumerate_sec(client, "1")]
    source = tmp_path / "evidence.json"
    source.write_text(json.dumps(evidence))
    members = tmp_path / "members.csv"
    members.write_text("date,tickers\n2000-01-01,A\n")
    detector = tmp_path / "flags.json"
    base = add(
        json.dumps(
            {
                "schema": "event-timing-detector-flags-v1",
                "flags": [],
                "screened": [{"symbol": "A-USD", "start": "2021-01-01", "end": "2026-04-30"}],
            }
        ),
        "https://www.sec.gov/Archives/synthetic-detector.json",
    )
    detector.write_text(
        json.dumps(
            {
                "schema": "event-timing-detector-flags-v1",
                "flags": [],
                "screened": [
                    {
                        "symbol": "A-USD",
                        "start": "2021-01-01",
                        "end": "2026-04-30",
                        "source_ref": base,
                    }
                ],
            }
        )
    )

    class MetadataDb:
        query_log = []

        def execute_select(self, sql, params=None):
            self.query_log.append({"sql": sql, "params": list(params or ())})
            assert "price_per_share" not in sql
            assert " close" not in sql.lower()
            if "count(*)" in sql:
                return ["count"], [(1,)]
            if "timestamp::date" in sql:
                schedule = AuditClock().get_schedule(str(params[1]), str(params[2]))
                return ["date"], [(d.date(),) for d in schedule.index]
            if "FROM form4_transactions" in sql:
                return [
                    "symbol",
                    "issuer_cik",
                    "insider_cik",
                    "accession_number",
                    "transaction_date",
                    "transaction_code",
                    "shares",
                    "acquired_disposed_code",
                    "is_10b5_1_plan",
                    "filing_date",
                ], [
                    (
                        "A",
                        1,
                        10,
                        inventory[0]["accession"],
                        datetime.date(2023, 5, 1),
                        "P",
                        100,
                        "A",
                        False,
                        datetime.date(2023, 5, 15),
                    )
                ]
            assert "FROM earnings_surprise" in sql
            return ["symbol", "earnings_date", "eps_estimate", "eps_reported", "surprise_pct"], []

    runner = EventTimingAuditRunner(
        cache_dir=cache,
        evidence_file=source,
        membership_file=members,
        db=MetadataDb(),
        detector_file=detector,
        output_file=tmp_path / "replay.json",
    )
    runner.client.manifest = client.manifest
    runner.fixture_add = add
    return runner


def test_fixture_runner_measures_clusters_not_sampled_filings_and_raw_collision_recovery(
    reconstructed_runner,
):
    report = reconstructed_runner.run()
    insider = report["insider_cluster"]
    assert len([e for e in insider["events"] if e["in_scope"]]) == 1
    assert insider["annual_coverage"][2023]["in_scope_events"] == 1
    assert insider["annual_coverage"][2023]["denominator_units"] == 12
    assert insider["annual_coverage"][2023]["priced_events"] == 1
    rows = {r["requirement"]: r for r in insider["requirement_matrix"] if r["year"] == 2023}
    assert rows["coverage"]["predicate"] == "PASS"
    assert rows["collisions"]["predicate"] == "PASS"
    assert rows["amendments"]["predicate"] == "PASS"
    assert rows["prefix"]["predicate"] == "PASS"
    assert rows["timestamp"]["predicate"] == "PASS"
    assert rows["collisions"]["details"]["stored_diagnostics"][0]["collisions"]
    # The synthetic complete census proves empty years too; A6 imposes no activity minimum.
    assert insider["status"] == "FREE—VERIFIED"
    assert len(report["samples"]["insider:2023"]["selected_keys"]) == 3
    assert report["samples"]["insider:2023"]["shortfall"] == 22
    assert report["phase2_build_requirements"]


def test_runner_resume_reuses_checked_evidence_and_rejects_changed_membership(reconstructed_runner):
    from ggTrader.lab.event_timing_audit import EventTimingAuditRunner

    first = reconstructed_runner.run()
    evidence_file = reconstructed_runner.output_file.parent / "evidence.json"
    second_runner = EventTimingAuditRunner(
        resume=True,
        detector_file=reconstructed_runner.output_file.parent / "flags.json",
        cache_dir=reconstructed_runner.cache_dir,
        evidence_file=evidence_file,
        membership_file=reconstructed_runner.membership_file,
        output_file=reconstructed_runner.output_file.parent / "resume.json",
    )
    second = second_runner.run()
    assert second["samples"] == first["samples"]
    assert second["audit_meta"]["sec_requests_used"] == first["audit_meta"]["sec_requests_used"]
    assert second["audit_meta"]["query_run_date"] == first["audit_meta"]["query_run_date"]
    reconstructed_runner.membership_file.write_text('date,tickers\n2000-01-01,"A,B"\n')
    with pytest.raises(ValueError, match="membership"):
        EventTimingAuditRunner(
            resume=True,
            cache_dir=reconstructed_runner.cache_dir,
            evidence_file=evidence_file,
            membership_file=reconstructed_runner.membership_file,
        )


def test_pead_lineage_requires_snapshot_before_release_not_before_acceptance(reconstructed_runner):
    runner = reconstructed_runner
    ref = runner.evidence["identities"][0]["source_refs"][0]
    provenance = {
        "fiscal_period": "2023Q1",
        "issuer_cik": "1",
        "basis": "GAAP",
        "currency": "USD",
        "split_basis": "as_reported",
        "source_refs": [ref],
    }
    event = {
        "symbol": "A",
        "earnings_date": "2023-05-04",
        "fiscal_period": "2023Q1",
        "issuer_cik": "1",
        "announcement_readings": ["2023-05-04T15:00:00-04:00"],
        "acceptance": "2023-05-04T16:00:00-04:00",
        "release_matched_by_issuer_period": True,
        "release_issuer_cik": "1",
        "release_fiscal_period": "2023Q1",
        "eps_reported": 2,
        "eps_estimate": 1,
        "surprise_pct": 100,
        "source_refs": [ref],
        "actual_eps": {**provenance, "value": 1},
        "consensus": {**provenance, "value": 1, "snapshot_time": "2023-05-04T14:00:00-04:00"},
    }
    eps_fields = ["issuer_cik", "fiscal_period", "value", "basis", "currency", "split_basis"]
    reviewed_fixture(event["actual_eps"], "eps", runner.fixture_add, eps_fields)
    reviewed_fixture(event["consensus"], "eps", runner.fixture_add, [*eps_fields, "snapshot_time"])
    reviewed_fixture(
        event,
        "pead_events",
        runner.fixture_add,
        [
            "symbol",
            "issuer_cik",
            "fiscal_period",
            "earnings_date",
            "announcement_readings",
            "acceptance",
            "release_matched_by_issuer_period",
            "release_issuer_cik",
            "release_fiscal_period",
        ],
    )
    runner.evidence["pead_events"] = [event]
    events, _ = runner.prepare_pead()
    assert events[0]["consensus_predicate"] == "PASS"
    assert events[0]["actual_eps_predicate"] == "PASS"
    assert events[0]["stored_reported_eps_mismatch"]
    event["actual_eps"]["currency"] = "EUR"
    assert runner.prepare_pead()[0][0]["consensus_predicate"] == "UNVERIFIED"
    event["actual_eps"]["currency"] = "USD"
    event["actual_eps"]["issuer_cik"] = "2"
    assert runner.prepare_pead()[0][0]["actual_eps_predicate"] == "UNVERIFIED"
    event["actual_eps"]["issuer_cik"] = "1"
    event["consensus"]["snapshot_time"] = "invalid"
    assert runner.prepare_pead()[0][0]["consensus_predicate"] == "UNVERIFIED"
    event["consensus"]["snapshot_time"] = "2023-05-04T15:30:00-04:00"
    reviewed_fixture(event["consensus"], "eps", runner.fixture_add, [*eps_fields, "snapshot_time"])
    assert runner.prepare_pead()[0][0]["consensus_predicate"] == "FAIL"
    event["consensus"]["snapshot_time"] = "2027-01-01T12:00:00-05:00"
    future = runner.prepare_pead()[0][0]
    assert future["quarantine"] == QuarantineReason.FUTURE_DATED
    assert future["tradeable_date"] is None
    event["consensus"]["snapshot_time"] = "2023-05-04T15:30:00-04:00"
    event["earnings_date"] = None
    quarantined = runner.prepare_pead()[0][0]
    assert quarantined["quarantine"] == QuarantineReason.UNPARSEABLE_OR_MISSING_DATE
    assert quarantined["tradeable_date"] is None


def test_forward_session_query_failure_checkpoints_and_continues(reconstructed_runner, monkeypatch):
    runner = reconstructed_runner
    original = runner.db.execute_select

    def fail_dates(sql, params=None):
        if sql.startswith("SELECT timestamp::date"):
            raise RuntimeError("fixture unavailable")
        return original(sql, params)

    monkeypatch.setattr(runner.db, "execute_select", fail_dates)
    report = runner.run()
    assert report["pending"]
    assert runner.progress_file.exists()
    assert (
        report["insider_cluster"]["events"][0]["holding_diagnostics"][0]["missing_observed_bars"]
        is None
    )


def test_reconciliation_sampler_deduplicates_canonical_keys():
    selected, shortfall = sample_stratum(["A", "A", "B"], str, 3)
    assert set(selected) == {"A", "B"}
    assert len(selected) == 2 and shortfall == 1


def test_raw_cache_rejects_symlink(raw_cache, tmp_path):
    cache, client, add = raw_cache
    ref = add("fixture")
    outside = tmp_path / "outside.txt"
    outside.write_text("fixture")
    (cache / ref["cache_file"]).unlink()
    (cache / ref["cache_file"]).symlink_to(outside)
    with pytest.raises(ValueError, match="checksum"):
        client.reference(ref["url"])


def test_interrupted_raw_write_preserves_uncommitted_bytes(tmp_path, monkeypatch):
    from ggTrader.lab.event_timing_sources import atomic_bytes

    def fail(*args):
        raise OSError("fixture interruption")

    monkeypatch.setattr("ggTrader.lab.event_timing_sources.os.replace", fail)
    with pytest.raises(OSError):
        atomic_bytes(tmp_path / "evidence.raw", b"raw-response", preserve_on_failure=True)
    assert any(p.read_bytes() == b"raw-response" for p in tmp_path.iterdir())


def test_cached_http_failure_is_not_successful_evidence(raw_cache):
    from ggTrader.lab.event_timing_audit import evidence_supported

    _, client, _ = raw_cache
    client.store_response("https://www.sec.gov/Archives/failure", "sec", b"forbidden", 403)
    ref = client.reference("https://www.sec.gov/Archives/failure")
    assert not evidence_supported({"source_refs": [ref]}, client)
    assert client.fetch_sec_url(ref["url"]) is None


def test_sec_rate_spacing_and_successful_raw_response_cache(raw_cache, monkeypatch):
    from ggTrader.lab.event_timing_sources import SecEdgarClient

    cache, _, _ = raw_cache
    monkeypatch.setenv("SEC_CONTACT_EMAIL", "fixture@example.invalid")
    now = [100.0]
    monkeypatch.setattr("ggTrader.lab.event_timing_sources.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "ggTrader.lab.event_timing_sources.time.sleep",
        lambda delay: now.__setitem__(0, now[0] + delay),
    )
    client = SecEdgarClient(cache, offline=False)
    times = []

    class Response:
        status = 200
        headers = {}

        def __enter__(self):
            times.append(now[0])
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return b"{}"

    monkeypatch.setattr(client.opener, "open", lambda *a, **k: Response())
    for i in range(4):
        assert client.fetch_sec_url(f"https://www.sec.gov/Archives/{i}.json") == "{}"
    assert all(b - a >= 0.199999 for a, b in zip(times[:-1], times[1:]))
    assert client.fetch_sec_url("https://www.sec.gov/Archives/0.json") == "{}"
    assert client.checkpoint.sec_requests_total == 4


def test_corrected_execution_prices_do_not_require_pit_versions(raw_cache):
    from ggTrader.lab.event_timing_audit import assess_source_routes

    _, client, add = raw_cache
    ref = add("fixture public product documentation")
    record = {
        "route": "R7",
        "source_refs": [ref],
        "history_start": "2000-01-01",
        "history_end": "2025-12-31",
        "requirements": ["tape"],
        "includes_delisted_acquired": True,
        "documentation": {
            item: {
                "assessment": "supports",
                "citation_url": ref["url"],
                "statement": "fixture",
                "retrieved_date": "2026-10-21",
            }
            for item in ("fields", "history", "access")
        },
    }
    reviewed_route(record, add)
    result = assess_source_routes("insider_cluster", [], [record], client)
    route = next(r for r in result["routes"] if r["route"] == "R7")
    assert route["predicate"] == "PASS"
    assert min(route["years"]) == 2015


def test_lag_and_real_monthly_load_boundary_counts_exclude_quarantine(cluster_lots):
    from ggTrader.lab.event_timing_reconstruction import filing_lags, load_boundaries

    clock = AuditClock()
    cluster_lots[0]["quarantine"] = QuarantineReason.TX_DATE_POST_FILING
    assert filing_lags(cluster_lots, clock)["by_filing_year"][2023]["count"] == 2
    cluster_lots[1]["transaction_date"] = "2022-01-01"
    result = load_boundaries(cluster_lots, clock)
    assert len(result) == 408
    assert any(
        "f1:0" in r["omitted_disclosure_lot_ids"]
        for r in result
        if r["decision"].startswith("2023-05")
    )


def test_identity_discovery_never_certifies_historical_ticker_map(raw_cache):
    import json

    from ggTrader.lab.event_timing_sources import collect_identity_candidates

    _, client, add = raw_cache
    add(
        json.dumps({"0": {"ticker": "A", "cik_str": 1, "title": "Current name"}}),
        "https://www.sec.gov/files/company_tickers.json",
    )
    result = collect_identity_candidates(client, ["A", "REMOVED"])
    assert result["candidates"] == [{"symbol": "A", "cik": "1", "current_name": "Current name"}]
    assert result["unresolved_symbols"] == ["REMOVED"]
    assert not result["historical_identity_verified"]


def test_release_collection_keeps_publisher_metadata_unconfirmed(raw_cache):
    import json

    from ggTrader.lab.event_timing_sources import collect_release_candidates, filing_urls

    _, client, add = raw_cache
    filing = {
        "accession": "0000000001-23-000001",
        "form": "8-K",
        "primary_document": "report.htm",
        "filing_date": "2023-05-04",
        "acceptance": "2023-05-04T16:30:00-04:00",
    }
    document, header = filing_urls("1", filing)
    base = document.rsplit("/", 1)[0]
    add(json.dumps({"directory": {"item": [{"name": "ex99.htm"}]}}), base + "/index.json")
    add('<a href="ex99.htm">Exhibit 99</a>', document)
    add("ACCESSION NUMBER: 0000000001-23-000001\n<ACCEPTANCE-DATETIME>20230504163000", header)
    add(
        '<meta property="article:published_time" content="2023-05-04T15:00:00-04:00">'
        '<meta name="fiscal-period-end" content="2023-03-31">',
        base + "/ex99.htm",
    )
    result = collect_release_candidates(client, "1", filing)
    assert result["complete"]
    assert result["header_acceptance"] == "2023-05-04T16:30:00"
    assert result["releases"][0]["fiscal_period"] == "2023-03-31"
    assert not result["releases"][0].get("original_release_time_confirmed")


def test_prefix_checker_detects_a_retrospective_rewrite(cluster_lots, monkeypatch):
    import ggTrader.lab.event_timing_reconstruction as reconstruction

    later = dict(
        cluster_lots[0],
        lot_id="future:0",
        accession="future",
        availability="2023-06-01T10:00:00-04:00",
    )
    original = reconstruction.replay_clusters

    def broken_replay(lots, *args):
        outputs, issues = original(lots, *args)
        if outputs and any(r["lot_id"] == "future:0" for r in lots):
            outputs[0] = dict(outputs[0], owners=["retrospectively-changed"])
        return outputs, issues

    monkeypatch.setattr(reconstruction, "replay_clusters", broken_replay)
    result = reconstruction.evaluate_prefix_invariance(
        cluster_lots + [later], [], AuditClock(), {"2023-05": True}
    )
    assert next(r for r in result["decisions"] if r["month"] == "2023-05")["predicate"] == "FAIL"


def test_dst_gap_is_quarantined_instead_of_imputed():
    from ggTrader.lab.event_timing_audit import event_clock

    event = event_clock(
        {"filing_date": "2023-03-12", "acceptance": "2023-03-12 02:30:00"},
        "insider_cluster",
        AuditClock(),
        datetime.date(2026, 10, 21),
    )
    assert event["quarantine"] == QuarantineReason.UNPARSEABLE_OR_MISSING_DATE
    assert event["availability"] is None and event["tradeable_date"] is None


@pytest.mark.parametrize("empty_lots", [False, True])
def test_later_unlinked_amendment_does_not_unverify_an_earlier_year(
    reconstructed_runner, empty_lots
):
    runner = reconstructed_runner
    original = runner.evidence["filings"][0]
    amendment = dict(
        original,
        accession="0000000001-24-000099",
        form="4/A",
        filing_date="2024-05-15",
        acceptance="2024-05-15T10:00:00-04:00",
        header_acceptance="2024-05-15T10:00:00-04:00",
    )
    amendment["lots"] = [
        dict(
            original["lots"][0],
            lot_id="0000000001-24-000099:0",
            accession=amendment["accession"],
            form="4/A",
            filing_date="2024-05-15",
            transaction_date="2024-05-01",
        )
    ]
    if empty_lots:
        amendment["lots"] = []
    runner.evidence["filings"].append(amendment)
    runner.evidence["inventories"][0]["filings"].append(
        {
            key: amendment[key]
            for key in ("accession", "form", "filing_date", "acceptance", "primary_document")
        }
    )
    report = runner.run()
    rows = report["insider_cluster"]["requirement_matrix"]
    assert (
        next(r for r in rows if r["year"] == 2023 and r["requirement"] == "prefix")["predicate"]
        == "PASS"
    )
    assert (
        next(r for r in rows if r["year"] == 2023 and r["requirement"] == "amendments")["predicate"]
        == "PASS"
    )
    assert (
        next(r for r in rows if r["year"] == 2024 and r["requirement"] == "amendments")["predicate"]
        == "UNVERIFIED"
    )


def test_reconstruction_does_not_mix_reused_ticker_issuers(cluster_lots):
    from ggTrader.lab.event_timing_reconstruction import replay_clusters

    for lot, cik in zip(cluster_lots, ("1", "2", "1")):
        lot["issuer_cik"] = cik
    events, _ = replay_clusters(cluster_lots, [], AuditClock().normalize_to_ny("2023-05-31"))
    assert events == []


def test_unfrozen_event_year_boundary_remains_explicit(reconstructed_runner):
    from ggTrader.lab.event_timing_audit import build_universe_denominators, event_clock

    runner = reconstructed_runner
    runner.db = None
    event = event_clock(
        {"event_id": "A|2023Q4", "symbol": "A", "earnings_date": "2023-12-29"},
        "pead",
        runner.clock,
        runner.query_run_date,
    )
    units = build_universe_denominators(runner.membership_file).units["pead"]
    result = runner.attach_population("pead", [event], units)[0]
    assert result["in_scope"]
    assert result["scope_assignment"] != "known"
    assert result["affected_years"] == [2023, 2024]


def test_zero_lot_amendment_cancels_only_from_its_own_public_time(cluster_lots):
    from ggTrader.lab.event_timing_reconstruction import cluster_snapshot

    clock = AuditClock()
    action = {
        "accession": "empty-amendment",
        "replaces_lot_ids": ["f0:0"],
        "source_refs": ["fixture"],
        "availability": "2023-06-01T10:00:00-04:00",
        "symbol": "A",
    }
    assert (
        len(cluster_snapshot(cluster_lots, clock.normalize_to_ny("2023-05-31 16:00"), [action])[0])
        == 1
    )
    assert (
        cluster_snapshot(cluster_lots, clock.normalize_to_ny("2023-06-01 16:00"), [action])[0] == []
    )


def test_malformed_empty_ownership_document_is_not_a_verified_empty_inventory():
    from ggTrader.lab.event_timing_sources import parse_form4

    parsed = parse_form4(
        "<ownershipDocument/>", {"accession": "f", "filing_date": "2023-05-01", "form": "4"}
    )
    assert parsed.get("schema_issues")


def test_legacy_orphan_cache_cannot_reset_lost_request_accounting(tmp_path):
    from ggTrader.lab.event_timing_sources import SecEdgarClient

    (tmp_path / "sec_legacy_response.txt").write_text("unvalidated legacy response")
    with pytest.raises(ValueError, match="checkpoint"):
        SecEdgarClient(tmp_path)


def test_malformed_cached_response_stays_unverified_and_preserved(raw_cache):
    cache, client, add = raw_cache
    ref = add(b"\xff", "https://www.sec.gov/Archives/bad-encoding")
    assert client.fetch_sec_url(ref["url"]) is None
    assert (cache / ref["cache_file"]).read_bytes() == b"\xff"


def test_known_missingness_cause_is_not_counted_as_unexplained(reconstructed_runner, monkeypatch):
    runner = reconstructed_runner
    original = runner.db.execute_select
    ref = runner.evidence["identities"][0]["source_refs"][0]
    runner.evidence["coverage_causes"] = [
        {
            "symbol": "A",
            "start": "2023-05-01",
            "end": "2023-12-31",
            "reason": "halt",
            "source_refs": [ref],
        }
    ]
    reviewed_fixture(
        runner.evidence["coverage_causes"][0],
        "coverage_causes",
        runner.fixture_add,
        ["symbol", "start", "end", "reason"],
    )

    def no_bars(sql, params=None):
        if "count(*)" in sql:
            return ["count"], [(0,)]
        if sql.startswith("SELECT timestamp::date"):
            return ["date"], []
        return original(sql, params)

    monkeypatch.setattr(runner.db, "execute_select", no_bars)
    report = runner.run()
    annual = report["insider_cluster"]["annual_coverage"][2023]
    assert annual["missing_events"] == 0
    assert annual["price_ratio"] == 0
    diagnostic = report["insider_cluster"]["events"][0]["holding_diagnostics"][0]
    assert diagnostic["missing_by_cause"]["halt"] > 0


def test_round2_unrelated_bytes_cannot_authorize_identity(reconstructed_runner):
    import json

    from ggTrader.lab.event_timing_audit import EventTimingAuditRunner

    runner = reconstructed_runner
    source = runner.output_file.parent / "evidence.json"
    data = json.loads(source.read_text())
    data["identities"][0].pop("field_reviews", None)
    source.write_text(json.dumps(data))
    checked = EventTimingAuditRunner(
        evidence_file=source,
        cache_dir=runner.cache_dir,
        membership_file=runner.membership_file,
    )
    assert checked.evidence["identities"] == []
    assert any(r["kind"] == "identities" for r in checked.input_errors)


def test_round2_receipt_without_dissemination_uses_supported_fallback():
    from ggTrader.lab.event_timing_audit import event_clock

    event = event_clock(
        {
            "filing_date": "2023-05-04",
            "header_filing_date": "2023-05-04",
            "acceptance": "2023-05-04T15:00:00-04:00",
            "header_acceptance": "2023-05-04T15:00:00-04:00",
        },
        "insider_cluster",
        AuditClock(),
        datetime.date(2026, 10, 21),
    )
    assert event["acceptance_time"] is not None
    assert event["availability"] is None
    assert event["precision"] == "date_only"
    assert event["tradeable_date"] == "2023-05-05"
    assert event["dissemination_predicate"] == "UNVERIFIED"


def test_round2_missing_prior_year_xml_blocks_later_population(reconstructed_runner):
    runner = reconstructed_runner
    runner.evidence["inventories"][0]["filings"].append(
        {
            "accession": "0000000001-22-000099",
            "form": "4",
            "filing_date": "2022-12-30",
            "primary_document": "missing.xml",
        }
    )
    report = runner.run()["insider_cluster"]
    assert report["annual_coverage"][2023]["in_scope_events"] is None
    assert (
        next(
            r
            for r in report["requirement_matrix"]
            if r["year"] == 2023 and r["requirement"] == "prefix"
        )["predicate"]
        == "UNVERIFIED"
    )


def test_round2_reduced_checkpoint_with_valid_manifest_is_rejected(raw_cache):
    from ggTrader.lab.event_timing_sources import SecEdgarClient

    cache, _, add = raw_cache
    add("preserved")
    (cache / "checkpoint.json").write_text('{"sec_requests_total":0,"yf_requests_total":0}')
    with pytest.raises(ValueError, match="checkpoint"):
        SecEdgarClient(cache)


@pytest.mark.parametrize(
    "target",
    [
        "https://www.sec.gov/redirect",
        "http://www.sec.gov/redirect",
        "https://www.sec.gov:444/redirect",
    ],
)
def test_round2_redirects_are_not_automatic(target):
    import urllib.request

    from ggTrader.lab.event_timing_sources import SafeRedirect

    request = urllib.request.Request("https://www.sec.gov/start")
    assert SafeRedirect().redirect_request(request, None, 302, "redirect", {}, target) is None


def test_round2_resume_binds_detector(reconstructed_runner):
    import json

    from ggTrader.lab.event_timing_audit import EventTimingAuditRunner

    runner = reconstructed_runner
    runner.run()
    detector = runner.output_file.parent / "flags.json"
    payload = json.loads(detector.read_text())
    payload["flags"].append({"symbol": "A-USD", "date": "2023-05-16"})
    detector.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="detector"):
        EventTimingAuditRunner(
            resume=True,
            evidence_file=runner.output_file.parent / "evidence.json",
            membership_file=runner.membership_file,
            cache_dir=runner.cache_dir,
            detector_file=detector,
        )


def test_round2_resume_keeps_stored_snapshot(reconstructed_runner, monkeypatch):
    import json

    from ggTrader.lab.event_timing_audit import EventTimingAuditRunner

    runner = reconstructed_runner
    first = runner.run()
    resumed = EventTimingAuditRunner(
        resume=True,
        evidence_file=runner.output_file.parent / "evidence.json",
        membership_file=runner.membership_file,
        cache_dir=runner.cache_dir,
        detector_file=runner.output_file.parent / "flags.json",
        db=runner.db,
    )
    monkeypatch.setattr(runner.db, "execute_select", lambda *a, **k: pytest.fail("snapshot reread"))
    resumed.read_stored()
    assert json.loads(json.dumps(resumed.stored, default=str)) == json.loads(
        json.dumps(first["stored_records"], default=str)
    )


def test_round2_proven_no_cluster_prefix_passes(cluster_lots):
    from ggTrader.lab.event_timing_reconstruction import evaluate_prefix_invariance

    result = evaluate_prefix_invariance(cluster_lots[:2], [], AuditClock(), {"2023-05": True})
    row = next(r for r in result["decisions"] if r["month"] == "2023-05")
    assert row["predicate"] == "PASS"
    assert row["prefix_outputs"] == []


def test_round2_coowner_missing_lot_is_counted(ownership_xml):
    from ggTrader.lab.event_timing_reconstruction import reconcile_lots
    from ggTrader.lab.event_timing_sources import parse_form4

    filing = {"accession": "f", "form": "4", "filing_date": "2023-05-03"}
    lots = parse_form4(ownership_xml, filing)["lots"]
    stored = [dict(r, insider_cik="10") for r in lots]
    result = reconcile_lots(lots, stored)
    assert result["missing_stored_lots"] == ["f:0", "f:1"]
    assert result["missing_stored_owner_lots"] == [
        {"lot_id": "f:0", "owner": "11"},
        {"lot_id": "f:1", "owner": "11"},
    ]


def test_round2_malformed_owner_survives_preparation(reconstructed_runner):
    runner = reconstructed_runner
    runner.evidence["filings"][0]["lots"][0]["owners"] = ["abc"]
    runner.evidence["filings"][0]["lots"][0].pop("raw_lot_sha256")
    lots, checks = runner.prepare_insider()
    result = checks["filings"][0]["lot_reconciliation"]
    assert result["predicate"] == "UNVERIFIED"
    assert result["unassignable_rows"]
    assert lots[0]["reconstruction_issue"]


def test_round2_expired_cpu_preparation_checkpoints_remaining(reconstructed_runner, monkeypatch):
    runner = reconstructed_runner
    monkeypatch.setattr(runner.client, "within_deadline", lambda: False)
    lots, _ = runner.prepare_insider()
    assert lots == []
    assert runner.pending
    assert runner.progress_file.exists()


def test_round2_selected_pead_exact_key_reconciles_missing_and_changed(reconstructed_runner):
    runner = reconstructed_runner
    runner.stored["pead"] = [
        {"symbol": "A", "earnings_date": "2023-05-04", "surprise_pct": 100},
        {"symbol": "A", "earnings_date": "2023-08-03", "surprise_pct": 50},
    ]
    from ggTrader.lab.event_timing_audit import build_universe_denominators

    runner.evidence["pead_fetches"] = [
        {
            "symbol": "A",
            "request_succeeded": True,
            "complete": True,
            "source_refs": runner.evidence["identities"][0]["source_refs"],
            "records": [{"earnings_date": "2023-05-04", "surprise_pct": 101}],
        }
    ]
    runner.collect(build_universe_denominators(runner.membership_file))
    outcomes = runner.samples["pead:2023"]["reconciliation"]
    assert {r["key"]: r["outcome"] for r in outcomes} == {
        "A|2023-05-04": "changed",
        "A|2023-08-03": "unmatched",
    }
    assert all(r["source_refs"] and not r["proves_pit"] for r in outcomes)


def test_round2_boundary_uncertainty_blocks_both_annual_populations(
    reconstructed_runner, monkeypatch
):
    from ggTrader.lab.event_timing_audit import build_universe_denominators, event_clock

    runner = reconstructed_runner
    runner.db = None
    dates = [
        "2023-02-01",
        "2023-05-04",
        "2023-08-03",
        "2023-12-29",
        "2024-02-01",
        "2024-05-02",
        "2024-08-01",
        "2024-11-01",
    ]
    events = []
    for date in dates:
        event = event_clock(
            {"symbol": "A", "earnings_date": date, "issuer_cik": "1"},
            "pead",
            runner.clock,
            runner.query_run_date,
        )
        event.update(event_id=date, fiscal_identity_verified=True)
        events.append(event)
    # Isolate annual boundary accounting from the independently tested lineage contract.
    monkeypatch.setattr(runner, "prepare_pead", lambda: (events, []))
    runner.evidence["pead_fetches"] = [
        {
            "symbol": "A",
            "request_succeeded": True,
            "complete": True,
            "source_refs": [],
            "records": [{"earnings_date": d} for d in dates],
        }
    ]
    report = runner.evaluate_signal("pead", build_universe_denominators(runner.membership_file))
    assert all(u["retrieved"] for u in report["units"] if u["year"] in (2023, 2024))
    for year in (2023, 2024):
        assert report["annual_coverage"][year]["in_scope_events"] is None


def test_round2_completed_bounded_audit_can_publish_unresolved(reconstructed_runner, monkeypatch):
    import ggTrader.lab.event_timing_audit as audit

    runner = reconstructed_runner
    monkeypatch.setattr(audit, "ROOT", runner.output_file.parent)
    runner.output_file = audit.ROOT / "docs/research/_event_timing_audit.json"
    runner.offline = False  # Transport stays offline; no endpoint is invoked.
    report = runner.run()
    assert report["audit_meta"]["complete"]
    assert report["pead"]["status"] == "UNRESOLVED"
    assert runner.output_file.exists()
    assert (audit.ROOT / "docs/research/2026-10-21-event-timing-audit-report.md").exists()


def test_round2_decisive_exclusion_does_not_require_positive_dimensions(raw_cache):
    from ggTrader.lab.event_timing_audit import assess_source_routes

    _, client, add = raw_cache
    text = "No individual licensing for any permitted representation."
    ref = add(text)
    record = {
        "route": "R8",
        "source_refs": [ref],
        "all_permitted_representations_ruled_out": True,
        "documentation": {
            "access": {
                "assessment": "contradicts",
                "citation_url": ref["url"],
                "retrieved_date": "2026-10-21",
                "statement": text,
            }
        },
    }
    reviewed_route(record, add)
    result = assess_source_routes("insider_cluster", [], [record], client)
    assert next(r for r in result["routes"] if r["route"] == "R8")["predicate"] == "FAIL"


@pytest.mark.parametrize(
    "defect", ["unrelated_extract", "wrong_period", "wrong_value", "bad_offset"]
)
def test_round2_review_receipt_cannot_promote_unrelated_facts(raw_cache, defect):
    from ggTrader.lab.event_timing_audit import semantic_fields_supported

    _, client, add = raw_cache
    record = {
        "symbol": "A",
        "cik": "1",
        "security_id": "A-common",
        "price_symbol": "A-USD",
        "valid_from": "2000-01-01",
        "valid_to": "2026-04-30",
    }
    fields = list(record)
    reviewed_fixture(record, "identities", add, fields)
    assert semantic_fields_supported("identities", record, client)
    if defect == "unrelated_extract":
        text = "Checksum-correct unrelated current ticker bytes"
        ref = add(text, "https://www.sec.gov/Archives/unrelated.txt")
        record["source_refs"] = [ref]
        for review in record["field_reviews"]:
            review.update(source_ref=ref, extract=text, byte_start=0, byte_end=len(text))
    elif defect == "wrong_period":
        record["field_reviews"][0]["linkage"] = {"symbol": "A", "valid_from": "2023-01-01"}
    elif defect == "wrong_value":
        record["price_symbol"] = "DIFFERENT-USD"
    else:
        record["field_reviews"][0]["byte_start"] = 1
    assert not semantic_fields_supported("identities", record, client)


@pytest.mark.parametrize(
    "defect", ["omitted_accession", "forged_interval", "wrong_issuer", "missing_page"]
)
def test_round2_inventory_is_rederived_not_asserted(reconstructed_runner, defect):
    import copy
    import json

    from ggTrader.lab.event_timing_audit import validate_evidence_record

    runner = reconstructed_runner
    inventory = copy.deepcopy(runner.evidence["inventories"][0])
    assert validate_evidence_record("inventories", inventory, runner.client, [])
    if defect == "omitted_accession":
        inventory["filings"].pop()
    elif defect == "forged_interval":
        inventory["start"] = "1990-01-01"
    elif defect == "wrong_issuer":
        inventory["cik"] = "2"
    else:
        ref = inventory["source_refs"][0]
        payload = json.loads(runner.client.referenced_bytes(ref))
        payload["filings"]["files"] = [
            {"name": "CIK1-missing.json", "filingFrom": "2000-01-01", "filingTo": "2022-12-31"}
        ]
        changed = runner.fixture_add(json.dumps(payload), ref["url"])
        inventory["source_refs"] = [changed]
    assert validate_evidence_record("inventories", inventory, runner.client, []) is None


def test_round2_malformed_raw_owner_is_reported_by_full_orchestration(reconstructed_runner):
    import json

    from ggTrader.lab.event_timing_audit import EventTimingAuditRunner

    runner = reconstructed_runner
    source = runner.output_file.parent / "evidence.json"
    payload = json.loads(source.read_text())
    filing = payload["filings"][0]
    ref = filing["source_refs"][0]
    xml = (
        runner.client.referenced_bytes(ref)
        .decode()
        .replace("<rptOwnerCik>10</rptOwnerCik>", "<rptOwnerCik>abc</rptOwnerCik>")
    )
    filing["source_refs"][0] = runner.fixture_add(xml, ref["url"])
    source.write_text(json.dumps(payload))
    checked = EventTimingAuditRunner(
        evidence_file=source,
        cache_dir=runner.cache_dir,
        membership_file=runner.membership_file,
        output_file=runner.output_file.parent / "malformed.json",
    )
    report = checked.run()
    raw = next(
        r
        for r in report["insider_cluster"]["raw_reconstruction"]["filings"]
        if r["accession"] == filing["accession"]
    )
    assert not raw["raw_complete"]
    assert raw["lot_reconciliation"]["unassignable_rows"]
    assert report["insider_cluster"]["annual_coverage"][2023]["in_scope_events"] is None
    assert checked.output_file.exists()


def test_round2_expired_runner_emits_partial_without_cpu_or_db_work(
    reconstructed_runner, monkeypatch
):
    runner = reconstructed_runner
    monkeypatch.setattr(runner.client, "within_deadline", lambda: False)
    monkeypatch.setattr(runner.db, "execute_select", lambda *a, **k: pytest.fail("expired DB work"))
    report = runner.run()
    assert not report["audit_meta"]["complete"]
    assert any("runtime budget exhausted" in reason for reason in report["pending"])
    assert all(
        row["predicate"] == "UNVERIFIED"
        for signal in ("pead", "insider_cluster")
        for row in report[signal]["requirement_matrix"]
    )


def test_round2_cpu_replay_resumes_completed_disclosures(cluster_lots):
    from ggTrader.lab.event_timing_reconstruction import replay_clusters

    clock = AuditClock()
    progress = {}
    calls = []

    def checkpoint():
        calls.append(1)

    # Stop at the next disclosure after one completed public-time work unit.
    replay_clusters(
        cluster_lots,
        [],
        clock.normalize_to_ny("2023-05-31"),
        lambda: not calls,
        progress,
        checkpoint,
    )
    assert progress["A"]["through"] == cluster_lots[0]["availability"]
    assert not progress["A"]["complete"]
    resumed = replay_clusters(
        cluster_lots, [], clock.normalize_to_ny("2023-05-31"), lambda: True, progress, checkpoint
    )
    assert resumed == replay_clusters(cluster_lots, [], clock.normalize_to_ny("2023-05-31"))
    assert progress["A"]["complete"]


def test_round2_manifest_requires_matching_reservation(raw_cache):
    import json

    from ggTrader.lab.event_timing_sources import SecEdgarClient

    cache, client, add = raw_cache
    add("synthetic observed response")
    state = json.loads((cache / "checkpoint.json").read_text())
    state["request_log"] = []
    state["sec_requests_total"] = 0
    (cache / "checkpoint.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="MANIFEST"):
        SecEdgarClient(cache)
    assert client.manifest["files"]


def test_round2_receipt_agreement_alone_cannot_pass_mandatory_timestamp():
    from ggTrader.lab.event_timing_audit import event_clock

    event = event_clock(
        {
            "filing_date": "2023-05-04",
            "acceptance": "2023-05-04T15:00:00-04:00",
            "header_acceptance": "2023-05-04T15:00:00-04:00",
        },
        "insider_cluster",
        AuditClock(),
        datetime.date(2026, 10, 21),
    )
    assert event["timestamp_predicate"] == "UNVERIFIED"
    assert event["availability"] is None
    assert event["tradeable_date"] == "2023-05-05"


def test_round2_prepared_pead_resumes_without_recomputing_clocks(reconstructed_runner, monkeypatch):
    import ggTrader.lab.event_timing_audit as audit

    runner = reconstructed_runner
    runner.stored["pead"] = [{"symbol": "A", "earnings_date": "2023-05-04"}]
    expected = runner.prepare_pead()
    assert runner.cpu_progress["pead_preparation"]["next"] == 1
    monkeypatch.setattr(
        audit, "event_clock", lambda *a, **k: pytest.fail("completed clock repeated")
    )
    assert runner.prepare_pead() == expected


@pytest.mark.parametrize("declared", [3, -1, "2", 2.0, True, None, 2])
def test_residual_historical_page_count_matches_all_parsed_rows(raw_cache, declared):
    import json

    from ggTrader.lab.event_timing_audit import validate_evidence_record
    from ggTrader.lab.event_timing_sources import enumerate_sec

    _, client, add = raw_cache
    historical = {
        "accessionNumber": ["0000000001-10-000001", "0000000001-22-000001"],
        "form": ["10-K", "4/A"],
        "filingDate": ["2010-01-01", "2022-12-31"],
        "primaryDocument": ["report.htm", "form4.xml"],
    }
    descriptor = {
        "name": "CIK1-old.json",
        "filingFrom": "2010-01-01",
        "filingTo": "2022-12-31",
        "filingCount": declared,
    }
    add(
        json.dumps(
            {
                "cik": "1",
                "filings": {
                    "recent": {key: [] for key in historical},
                    "files": [descriptor],
                },
            }
        ),
        "https://data.sec.gov/submissions/CIK0000000001.json",
    )
    raw = json.dumps(historical)
    ref = add(raw, "https://data.sec.gov/submissions/CIK1-old.json")
    inventory = enumerate_sec(client, "1")
    valid_count = type(declared) is int and declared == 2
    assert inventory["complete"] is valid_count
    assert inventory["request_succeeded"]
    assert ref in inventory["source_refs"]
    assert client.referenced_bytes(ref).decode() == raw
    if valid_count:
        # filing_rows preserves every form; filingCount is not a Form 4-only count.
        assert [row["form"] for row in inventory["filings"]] == ["10-K", "4/A"]
        assert validate_evidence_record("inventories", inventory, client, [])
    else:
        assert "filingCount" in inventory["reason"]
        assert validate_evidence_record("inventories", inventory, client, []) is None


@pytest.mark.parametrize("header_case", ["missing", "wrong_accession", "matching", "checked_date"])
def test_residual_bound_timestamp_needs_independent_matching_receipts(raw_cache, header_case):
    from ggTrader.lab.event_timing_audit import event_clock
    from ggTrader.lab.event_timing_sources import parse_filing_header

    _, client, add = raw_cache
    accession = "0000000001-23-000001"
    record = {
        "cik": "1",
        "accession": accession,
        "form": "4",
        "filing_date": "2023-05-04",
        "acceptance": "2023-05-04T15:00:00-04:00",
        "dissemination_bound_seconds": 60,
    }
    reviewed_fixture(record, "dissemination", add, ["dissemination_bound_seconds"])
    if header_case != "missing":
        header_accession = "0000000001-23-000002" if header_case == "wrong_accession" else accession
        header = f"ACCESSION NUMBER: {header_accession}\n<FILED-AS-OF-DATE>20230504"
        if header_case != "checked_date":
            header += "\n<ACCEPTANCE-DATETIME>20230504150000"
        else:
            record.pop("dissemination_bound_seconds")
        record.update(parse_filing_header(header, accession))
    event = event_clock(
        record, "insider_cluster", AuditClock(), datetime.date(2026, 10, 21), client
    )
    if header_case in ("matching", "checked_date"):
        assert event["timestamp_predicate"] == "PASS"
        assert event["precision"] == ("timestamp" if header_case == "matching" else "date_only")
        assert event["tradeable_date"] == (
            "2023-05-04" if header_case == "matching" else "2023-05-05"
        )
    else:
        assert (
            event["dissemination_predicate"] == "PASS"
        )  # Synthetic valid bound, not real evidence.
        assert event["timestamp_predicate"] == "UNVERIFIED"


@pytest.mark.parametrize("changed", ["revision", "membership_helper"])
def test_residual_resume_rejects_changed_revision_or_membership_code(
    reconstructed_runner, monkeypatch, changed
):
    from types import SimpleNamespace

    import ggTrader.lab.event_timing_audit as audit

    runner = reconstructed_runner
    runner.checkpoint("frozen_registration", force=True)
    preserved = runner.progress_file.read_bytes()
    if changed == "revision":
        # Even an unrelated commit with unchanged module hashes requires a new run.
        monkeypatch.setattr(
            audit.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="0" * 40)
        )
        message = "revision"
    else:
        original = audit.compute_file_sha256
        helper = Path(audit.load_sp500_history.__code__.co_filename).resolve()

        def changed_hash(path):
            return "f" * 64 if Path(path).resolve() == helper else original(path)

        # Simulate changed on-disk helper bytes without touching the shared source file.
        monkeypatch.setattr(audit, "compute_file_sha256", changed_hash)
        message = "implementation"
    with pytest.raises(ValueError, match=message):
        audit.EventTimingAuditRunner(
            resume=True,
            evidence_file=runner.output_file.parent / "evidence.json",
            cache_dir=runner.cache_dir,
            membership_file=runner.membership_file,
            detector_file=runner.output_file.parent / "flags.json",
        )
    assert runner.progress_file.read_bytes() == preserved


def test_residual_completed_event_observation_survives_interruption_on_next_event(
    reconstructed_runner,
):
    import json

    from ggTrader.lab.event_timing_audit import (
        EventTimingAuditRunner,
        build_universe_denominators,
        event_clock,
    )

    runner = reconstructed_runner
    first_date, second_date = datetime.date(2023, 5, 5), datetime.date(2023, 5, 8)

    class ObservationDb:
        def __init__(self, interrupt):
            self.interrupt = interrupt
            self.query_log = []
            self.count_dates = []

        def execute_select(self, sql, params=None):
            self.query_log.append({"sql": sql, "params": list(params or ())})
            if "count(*)" in sql:
                date = params[1]
                self.count_dates.append(date)
                if self.interrupt and date == second_date:
                    raise KeyboardInterrupt("fixture interruption during second event")
                if not self.interrupt and date == first_date:
                    pytest.fail("completed first tape observation was replaced on resume")
                return ["count"], [(1,)]
            assert sql.startswith("SELECT timestamp::date")
            return ["date"], []

    events = [
        event_clock(
            {"event_id": key, "symbol": "A", "earnings_date": date},
            "pead",
            runner.clock,
            runner.query_run_date,
        )
        for key, date in [("first", "2023-05-04"), ("second", "2023-05-05")]
    ]
    units = build_universe_denominators(runner.membership_file).units["pead"]
    runner.db = ObservationDb(interrupt=True)
    runner.checkpoint("before_population", force=True)  # Exercise the normal 30-second throttle.
    with pytest.raises(KeyboardInterrupt, match="second event"):
        runner.attach_population("pead", events, units)
    progress = json.loads(runner.progress_file.read_text())
    completed = progress["cpu_progress"].get("population:pead", {})
    assert set(completed) == {"first"}
    assert completed["first"]["tape"]["bar_count"] == 1
    receipts = progress["observation_queries"]
    assert any("count(*)" in row["sql"] and row["params"][1] == str(first_date) for row in receipts)
    assert not any(
        "count(*)" in row["sql"] and row["params"][1] == str(second_date) for row in receipts
    )
    resumed_db = ObservationDb(interrupt=False)
    resumed = EventTimingAuditRunner(
        resume=True,
        evidence_file=runner.output_file.parent / "evidence.json",
        cache_dir=runner.cache_dir,
        membership_file=runner.membership_file,
        detector_file=runner.output_file.parent / "flags.json",
        db=resumed_db,
    )
    output = resumed.attach_population("pead", events, units)
    assert resumed_db.count_dates == [second_date]
    assert output[0]["tape"] == completed["first"]["tape"]
    assert resumed.observation_queries[: len(receipts)] == receipts
    saved = json.loads(resumed.progress_file.read_text())
    assert set(saved["cpu_progress"]["population:pead"]) == {"first", "second"}


def test_amendment_original_fixture_regression(reconstructed_runner):
    """Pin the full original assessments before changing the implementation."""
    import json

    report = reconstructed_runner.run()
    for name in ("insider_cluster", "pead"):
        value = json.dumps(report[name], sort_keys=True, default=str).encode()
        expected = {
            "insider_cluster": "b7514e570e98469a7dc12efb3e4b0d7abaebee5cf3f71d1b402899fec3abf042",
            "pead": "a1a43816918377d254cefc2ee28947b487b73aa14b1b3546c854c44956a0469c",
        }
        assert hashlib.sha256(value).hexdigest() == expected[name]


@pytest.fixture
def amendment_runner(reconstructed_runner):
    """Same census, with explicit raw header issuer identity for the new representation."""
    from ggTrader.lab.event_timing_audit import validate_evidence_record

    runner = reconstructed_runner
    for filing in runner.evidence["filings"]:
        ref = filing["source_refs"][1]
        header = runner.client.referenced_bytes(ref).decode()
        ref = runner.fixture_add(
            header + "\nISSUER:\n\tCOMPANY DATA:\n\t\tCENTRAL INDEX KEY: 0000000001\n", ref["url"]
        )
        filing["source_refs"][1] = ref
        rebuilt = validate_evidence_record(
            "filings", filing, runner.client, runner.evidence["inventories"]
        )
        filing.update(rebuilt)
    return runner


def amendment_clock(runner, filing):
    from ggTrader.lab.event_timing_audit import event_clock

    return event_clock(
        filing,
        "insider_cluster",
        runner.clock,
        runner.query_run_date,
        runner.client,
        representation="insider_filing_date_only",
        inventories=runner.evidence["inventories"],
    )


def amendment_receipt_conflict(runner, filing):
    """Make the API/header conflict real in the synthetic raw sources too."""
    import json

    filing["acceptance"] = "2023-05-15T14:00:00-04:00"
    inventory = runner.evidence["inventories"][0]
    ref = inventory["source_refs"][0]
    api = json.loads(runner.client.referenced_bytes(ref))
    index = api["filings"]["recent"]["accessionNumber"].index(filing["accession"])
    api["filings"]["recent"]["acceptanceDateTime"][index] = filing["acceptance"]
    inventory["source_refs"][0] = runner.fixture_add(json.dumps(api), ref["url"])
    for row in inventory["filings"]:
        if row["accession"] == filing["accession"]:
            row["acceptance"] = filing["acceptance"]


@pytest.mark.parametrize("conflict", [False, True])
def test_amendment_receipts_retained_without_public_time_or_source_overwrite(
    amendment_runner, conflict
):
    import copy

    from ggTrader.lab.event_timing_audit import event_clock

    runner = amendment_runner
    filing = copy.deepcopy(runner.evidence["filings"][0])
    if conflict:
        amendment_receipt_conflict(runner, filing)
    else:
        refs = filing["source_refs"]
        filing["dissemination_bound_seconds"] = 0
        reviewed_fixture(
            filing, "dissemination", runner.fixture_add, ["dissemination_bound_seconds"]
        )
        filing["source_refs"] = [*refs, *filing["source_refs"]]
    before = copy.deepcopy(filing)
    original = event_clock(
        filing, "insider_cluster", runner.clock, runner.query_run_date, runner.client
    )
    event = amendment_clock(runner, filing)
    assert event["tradeable_date"] == "2023-05-16"
    assert event["tradeable_close"] == "2023-05-16T16:00:00-04:00"
    assert event["timestamp_predicate"] == "PASS"
    assert event["precision"] == "date_only"
    assert event["availability"] is None
    assert event["dissemination_predicate"] == "UNVERIFIED"
    assert event["original_clock"] == original
    assert event["acceptance_reconciliation"] == ("FAIL" if conflict else "PASS")
    assert original["quarantine"] == (
        QuarantineReason.UNRESOLVED_TIMESTAMP_CONFLICT if conflict else None
    )
    if not conflict:
        assert original["precision"] == "timestamp"
        assert original["tradeable_date"] == "2023-05-15"
        assert original["availability"] == "2023-05-15T10:00:00-04:00"
    assert filing == before
    assert event["acceptance"] == filing["acceptance"]
    assert event["header_acceptance"] == filing["header_acceptance"]


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_issuer",
        "wrong_accession",
        "missing",
        "corrupt",
        "different_date",
        "missing_date",
        "asserted_date",
        "asserted_api_date",
    ],
)
def test_amendment_requires_actual_same_issuer_accession_dates(amendment_runner, defect):
    import copy

    runner = amendment_runner
    filing = copy.deepcopy(runner.evidence["filings"][0])
    ref = filing["source_refs"][1]
    header = runner.client.referenced_bytes(ref).decode()
    if defect == "wrong_issuer":
        header = header.replace("CENTRAL INDEX KEY: 0000000001", "CENTRAL INDEX KEY: 0000000002")
    elif defect == "wrong_accession":
        header = header.replace(filing["accession"], "0000000001-23-999999")
    elif defect == "missing":
        filing["source_refs"].remove(ref)
    elif defect == "corrupt":
        (runner.cache_dir / ref["cache_file"]).write_text("corrupt fixture")
    elif defect == "different_date":
        header = header.replace("<FILED-AS-OF-DATE>20230515", "<FILED-AS-OF-DATE>20230516")
    elif defect == "missing_date":
        header = header.replace("<FILED-AS-OF-DATE>20230515", "")
    elif defect == "asserted_date":
        filing["header_filing_date"] = "2023-05-16"
    else:
        filing["filing_date"] = "2023-05-16"
        filing["header_filing_date"] = "2023-05-16"
    if defect in ("wrong_issuer", "wrong_accession", "different_date", "missing_date"):
        filing["source_refs"][1] = runner.fixture_add(header, ref["url"])
    event = amendment_clock(runner, filing)
    assert event["timestamp_predicate"] != "PASS"
    assert event["tradeable_date"] is None
    assert event["availability"] is None
    assert event["filing_date_corroboration"]["predicate"] != "PASS"


@pytest.mark.parametrize(
    "filed,expected",
    [
        ("2023-05-15", "2023-05-16T16:00:00-04:00"),
        ("2023-05-27", "2023-05-30T16:00:00-04:00"),
        ("2023-07-03", "2023-07-05T16:00:00-04:00"),
        ("2023-11-22", "2023-11-24T13:00:00-05:00"),
    ],
)
def test_amendment_strict_next_session_holiday_and_early_close(amendment_runner, filed, expected):
    import json

    runner = amendment_runner
    filing = runner.evidence["filings"][0]
    inventory = runner.evidence["inventories"][0]
    ref = inventory["source_refs"][0]
    api = json.loads(runner.client.referenced_bytes(ref))
    api["filings"]["recent"]["filingDate"][1] = filed
    inventory["source_refs"][0] = runner.fixture_add(json.dumps(api), ref["url"])
    filing["filing_date"] = filing["header_filing_date"] = filed
    ref = filing["source_refs"][1]
    header = (
        runner.client.referenced_bytes(ref)
        .decode()
        .replace("<FILED-AS-OF-DATE>20230515", "<FILED-AS-OF-DATE>" + filed.replace("-", ""))
    )
    filing["source_refs"][1] = runner.fixture_add(header, ref["url"])
    event = amendment_clock(runner, filing)
    assert event["tradeable_close"] == expected
    assert event["availability"] is None
    assert event["precision"] == "date_only"


@pytest.mark.parametrize(
    "transaction,reason",
    [
        ("1989-12-31", "TX_DATE_PRE_1990"),
        ("2023-05-16", "TX_DATE_POST_FILING"),
        ("2030-01-01", "FUTURE_DATED"),
        (None, "UNPARSEABLE_OR_MISSING_DATE"),
    ],
)
def test_amendment_retains_other_transaction_quarantines(amendment_runner, transaction, reason):
    runner = amendment_runner
    amendment_receipt_conflict(runner, runner.evidence["filings"][0])
    for lot in runner.evidence["filings"][0]["lots"]:
        lot["transaction_date"] = transaction
    lots, checks = runner.prepare_insider(representation="insider_filing_date_only")
    affected = [r for r in lots if r["accession"] == runner.evidence["filings"][0]["accession"]]
    assert affected and all(r["quarantine"] == reason for r in affected)
    assert all(r["availability"] is None for r in affected)
    assert len(checks["filings"]) == 3


def test_amendment_uniform_population_and_separate_matrices(amendment_runner):
    runner = amendment_runner
    amendment_receipt_conflict(runner, runner.evidence["filings"][0])
    report = runner.run()
    original, amended = report["insider_cluster"], report["insider_filing_date_only"]
    assert (
        len(original["raw_reconstruction"]["filings"])
        == len(amended["raw_reconstruction"]["filings"])
        == 3
    )
    assert len(amended["raw_reconstruction"]["lots"]) == 6
    assert {r["clock"]["precision"] for r in amended["raw_reconstruction"]["filings"]} == {
        "date_only"
    }
    assert len(original["quarantined_lots"]) == 2
    assert not amended["quarantined_lots"]
    assert original["annual_coverage"][2023]["in_scope_events"] is None
    assert amended["annual_coverage"][2023]["in_scope_events"] == 1
    assert amended["annual_coverage"][2023]["denominator_units"] == 12
    assert all(
        r["representation_assessed"] == "insider_filing_date_only"
        for r in amended["requirement_matrix"]
    )
    assert all(
        r["representation_assessed"] == "raw-source reconstruction"
        for r in original["requirement_matrix"]
    )
    assert len(amended["prefix"]["decisions"]) == 136
    assert all(e["availability"] is None for e in amended["events"])
    assert "insider_filing_date_only" in runner.output_file.with_suffix(".md").read_text()
    assert (
        report["audit_meta"]["approved_amendment"]["sha256"]
        == hashlib.sha256(
            Path(
                "docs/research/briefs/2026-10-22-event-timing-source-conflict-proposal.md"
            ).read_bytes()
        ).hexdigest()
    )


def test_amendment_missing_pair_keeps_population_unknown(amendment_runner):
    runner = amendment_runner
    runner.evidence["filings"][0]["source_refs"].pop(1)
    report = runner.run()["insider_filing_date_only"]
    assert len(report["raw_reconstruction"]["filings"]) == 3
    assert len(report["raw_reconstruction"]["lots"]) == 6
    assert report["annual_coverage"][2023]["in_scope_events"] is None
    assert report["annual_coverage"][2023]["observed_event_lower_bound"] == 0
    assert all(
        r["predicate"] == "UNVERIFIED" for r in report["requirement_matrix"] if r["year"] == 2023
    )
    # The synthetic census still proves earlier, entirely empty years.
    assert report["eligible_window"][1] <= 2022


def test_amendment_prefix_includes_only_at_declared_close(cluster_lots):
    from ggTrader.lab.event_timing_reconstruction import evaluate_prefix_invariance

    for lot in cluster_lots:
        lot.update(
            availability=None,
            decision_eligibility_time="2023-06-01T16:00:00-04:00",
            tradeable_close="2023-06-01T16:00:00-04:00",
        )
    result = evaluate_prefix_invariance(
        cluster_lots,
        [],
        AuditClock(),
        {str(p): True for p in pd.period_range("2015-01", "2026-04", freq="M")},
    )
    decisions = {r["month"]: r for r in result["decisions"]}
    assert len(decisions) == 136
    assert decisions["2023-05"]["full_outputs"] == []
    assert decisions["2023-06"]["full_outputs"][0]["availability"] is None
    assert decisions["2023-06"]["full_outputs"][0]["formed_at"] == "2023-06-01T16:00:00-04:00"
    assert all(r["predicate"] == "PASS" for r in result["decisions"])


@pytest.mark.parametrize("changed", ["hash", "legacy_missing"])
def test_amendment_resume_rejects_changed_policy(
    reconstructed_runner, monkeypatch, tmp_path, changed
):
    import json

    import ggTrader.lab.event_timing_audit as audit

    runner = reconstructed_runner
    runner.checkpoint("registered", force=True)
    progress = json.loads(runner.progress_file.read_text())
    assert progress["approved_amendment"]["sha256"]
    if changed == "legacy_missing":
        progress.pop("approved_amendment")
        runner.progress_file.write_text(json.dumps(progress))
    else:
        policy = tmp_path / "policy.md"
        policy.write_text("Changed amendment")
        monkeypatch.setattr(audit, "APPROVED_AMENDMENT", policy)
    preserved = runner.progress_file.read_bytes()
    with pytest.raises(ValueError, match="amendment"):
        audit.EventTimingAuditRunner(
            resume=True,
            evidence_file=runner.output_file.parent / "evidence.json",
            cache_dir=runner.cache_dir,
            membership_file=runner.membership_file,
            detector_file=runner.output_file.parent / "flags.json",
        )
    assert runner.progress_file.read_bytes() == preserved


@pytest.mark.parametrize("reading", ["2030-01-01T10:00:00Z", "invalid timestamp"])
def test_amendment_receipt_conflict_does_not_erase_other_date_quarantine(amendment_runner, reading):
    runner = amendment_runner
    filing = dict(runner.evidence["filings"][0], acceptance=reading)
    event = amendment_clock(runner, filing)
    assert event["quarantine"] in ("FUTURE_DATED", "UNPARSEABLE_OR_MISSING_DATE")
    assert event["tradeable_date"] is None
    assert event["availability"] is None


def test_amendment_zero_lot_missing_clock_is_not_complete_population(amendment_runner):
    runner = amendment_runner
    filing = runner.evidence["filings"][0]
    filing["lots"] = []
    filing["source_refs"].pop(1)
    report = runner.run()["insider_filing_date_only"]
    assert report["annual_coverage"][2023]["in_scope_events"] is None
    assert report["annual_coverage"][2023]["population_complete"] is False
    assert report["prefix"]["annual"][2023] == "UNVERIFIED"


@pytest.mark.parametrize(
    "eligible,may_events",
    [
        ("2023-05-31T16:00:00-04:00", 1),
        ("2023-06-01T16:00:00-04:00", 0),
    ],
)
def test_amendment_prefix_exact_month_end_close(cluster_lots, eligible, may_events):
    from ggTrader.lab.event_timing_reconstruction import evaluate_prefix_invariance

    for lot in cluster_lots:
        lot.update(availability=None, decision_eligibility_time=eligible, tradeable_close=eligible)
    result = evaluate_prefix_invariance(cluster_lots, [], AuditClock(), {"2023-05": True})
    may = next(r for r in result["decisions"] if r["month"] == "2023-05")
    assert len(may["full_outputs"]) == len(may["prefix_outputs"]) == may_events
    assert may["predicate"] == "PASS"
    assert all(e["availability"] is None for e in may["full_outputs"])
