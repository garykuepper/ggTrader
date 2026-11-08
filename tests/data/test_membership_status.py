"""Fast tests of ``ggt db membership status`` and ``diff`` with the database replaced."""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

import pytest

from ggTrader.cli import cmd_membership
from ggTrader.data.membership import DbMembershipSource, MembershipUnavailable
from ggTrader.data.universe_db import diff_vintages
from tests.data.membership_fakes import FailingStore, FakeMembershipStore, synthetic_history
from tests.data.test_membership_upstream import vintage_data

BASE = datetime(2026, 6, 9, tzinfo=timezone.utc)


def _status_args(asof: date | None) -> argparse.Namespace:
    return argparse.Namespace(membership_command="status", asof=asof)


def _patch_source(monkeypatch, store, checks=()):
    monkeypatch.setattr(
        cmd_membership, "DbMembershipSource", lambda which: DbMembershipSource(which, store=store)
    )
    monkeypatch.setattr(cmd_membership, "recent_checks", lambda *a, **k: list(checks))


def test_status_reports_the_vintage_and_the_gate_at_several_days(monkeypatch, capsys):
    history = synthetic_history()
    store = FakeMembershipStore()
    store.add(history, retrieved_at=BASE)
    _patch_source(monkeypatch, store)
    coverage_end = max(history)
    fresh_day = BASE.date() + timedelta(days=5)
    cmd_membership.run_status(_status_args(fresh_day))
    out = capsys.readouterr().out
    assert "latest load 1" in out and f"{coverage_end}" in out
    assert "verification age   5 days" in out

    stale = BASE.date() + timedelta(days=46)
    assert cmd_membership.run_status(_status_args(stale)) == 0
    out = capsys.readouterr().out
    assert "live gate would    REFUSE" in out and "check-upstream" in out


def test_status_says_so_when_the_database_cannot_be_read(monkeypatch, capsys):
    monkeypatch.setattr(
        cmd_membership,
        "DbMembershipSource",
        lambda which: DbMembershipSource(which, store=FailingStore()),
    )
    assert cmd_membership.run_status(_status_args(None)) == 1
    captured = capsys.readouterr()
    assert "REFUSE (membership_unavailable)" in captured.out
    assert "cannot read membership" in captured.err


def test_status_shows_the_recorded_checks(monkeypatch, capsys):
    store = FakeMembershipStore()
    store.add(synthetic_history(), retrieved_at=BASE)
    row = {
        "check_id": 4,
        "checked_at": BASE + timedelta(days=1),
        "outcome": "refused",
        "detail": {"result": "differs"},
    }
    _patch_source(monkeypatch, store, [row])
    cmd_membership.run_status(_status_args(BASE.date()))
    assert "check   4" in capsys.readouterr().out


def test_diff_of_a_vintage_with_itself_is_empty():
    data = vintage_data(synthetic_history())
    report = diff_vintages(data, data)
    assert report["same_content"] is True
    assert report["dates_only_in_from"] == report["dates_only_in_to"] == []
    assert report["intervals_only_in_from"] == report["intervals_only_in_to"] == 0
    assert report["would_need_force"] == []


def test_diff_classifies_a_tail_change_and_a_historical_rewrite():
    history = synthetic_history(n_dates=10)
    older = vintage_data(history, load_id=1)
    newer_history = dict(history)
    last = max(history)
    newer_history[last + timedelta(days=30)] = tuple(
        sorted(list(history[last])[3:] + ["N1", "N2", "N3"])
    )
    tail = diff_vintages(older, vintage_data(newer_history, load_id=2))
    assert (
        tail["dates_only_in_to"] == [last + timedelta(days=30)] and len(tail["tail_changes"]) == 6
    )
    assert tail["historical_rewrites"] == [] and not tail["same_content"]

    days = sorted(history)
    rewritten = dict(history)
    rewritten[days[2]] = tuple(sorted(list(history[days[2]])[1:] + ["ZZZZ"]))
    rewrite = diff_vintages(older, vintage_data(rewritten, load_id=3))
    assert rewrite["historical_rewrites"] and list(rewrite["dates_changed"]) == [
        days[2].isoformat()
    ]


@pytest.mark.parametrize("missing", ["from", "to"])
def test_diff_reports_a_missing_load_id(monkeypatch, capsys, missing):
    class Cur:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    class Conn:
        def cursor(self):
            return Cur()

        def close(self):
            return None

    def fake_read(cur, load_id, index_name="sp500"):
        if load_id == (1 if missing == "from" else 2):
            raise LookupError(f"no load {load_id} for index 'sp500'")
        return vintage_data(synthetic_history(), load_id=load_id)

    monkeypatch.setattr(cmd_membership, "connect", lambda read_only=False: Conn())
    monkeypatch.setattr(cmd_membership, "read_vintage", fake_read)
    args = argparse.Namespace(membership_command="diff", from_id=1, to_id=2, json=None)
    assert cmd_membership.run_diff(args) == 1
    assert "no load" in capsys.readouterr().err


def test_unavailable_error_type_is_the_one_the_live_path_converts():
    with pytest.raises(MembershipUnavailable):
        DbMembershipSource("latest", store=FailingStore())
