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

import pandas as pd
import pytest

from ggTrader.lab.event_timing_audit import (
    STATUS_FREE_VERIFIED,
    STATUS_NO_VERIFIED_PATH,
    STATUS_PAID_DOCUMENTED,
    STATUS_UNRESOLVED,
    AnnualA6Coverage,
    AuditClock,
    QuarantineReason,
    check_earnings_quarantine,
    check_transaction_quarantine,
    compute_deterministic_sample_hash,
    evaluate_eligible_window,
    resolve_signal_status,
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
        assert (
            check_earnings_quarantine("1988-05-01", self.run_date)
            == QuarantineReason.TX_DATE_PRE_1990
        )
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

    def test_status_resolution_branches(self):
        # Branch 1: has eligible window
        assert resolve_signal_status(has_eligible_window=True) == STATUS_FREE_VERIFIED

        # Branch 2: no eligible window, but paid path documented
        assert (
            resolve_signal_status(has_eligible_window=False, paid_path_documented=True)
            == STATUS_PAID_DOCUMENTED
        )

        # Branch 3: all commercial ruled out
        assert (
            resolve_signal_status(
                has_eligible_window=False,
                paid_path_documented=False,
                all_commercial_ruled_out=True,
            )
            == STATUS_NO_VERIFIED_PATH
        )

        # Branch 4: default fallback
        assert (
            resolve_signal_status(
                has_eligible_window=False,
                paid_path_documented=False,
                all_commercial_ruled_out=False,
            )
            == STATUS_UNRESOLVED
        )
